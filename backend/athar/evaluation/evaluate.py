"""Evaluate Athar on the held-out synthetic evaluation set.

    python -m athar.evaluation.evaluate [--extractor lexicon|llm|auto] [--compare]

Two different questions are reported separately, on purpose:
  A. Constructed-scenario agreement: did Athar find the documented patterns we deliberately built?
  B. Hidden-label agreement: how often does a flag coincide with the synthetic 'endometriosis' label?
Neither demonstrates real-world clinical accuracy (synthetic data, no external validation).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from sklearn.metrics import roc_auc_score

from ..config import OUTPUT_DIR
from ..pipeline import Athar, extract_all, load_records, load_truth
from ..schemas import Category

CATS = [Category.review.value, Category.insufficient.value, Category.no_pattern.value]
CAVEAT = ("These results measure performance on synthetic data generated for development. They show that the pipeline "
          "behaves as designed on constructed scenarios; they do NOT demonstrate clinical accuracy. Real-world performance "
          "requires real, labelled, de-identified records and external validation.")


def binary(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    prec = tp / (tp + fp) if tp + fp else None
    f1 = 2 * prec * sens / (prec + sens) if prec and sens else 0.0
    r = lambda x: round(x, 3) if x is not None else None
    return {"n": len(y_true), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "sensitivity": r(sens), "specificity": r(spec), "precision": r(prec),
            "f1": r(f1), "false_positive_rate": r(1 - spec) if spec is not None else None, "false_negative_rate": r(1 - sens) if sens is not None else None}


def extraction_quality(records, exs, truth) -> dict:
    tp = fp = fn = st_ok = st_n = ln_ok = ln_n = 0
    for r in records:
        res = exs[r.patient_id]
        skip = {i.encounter_id for i in res.issues if i.kind == "duplicate_encounter"}
        pred = {(f.encounter_id, f.concept): f for f in res.findings if f.category == "symptom"}
        gold = {(e["encounter_id"], s["concept"]): s for e in truth[r.patient_id]["encounters"] if e["encounter_id"] not in skip for s in e["symptoms"]}
        for k, g in gold.items():
            if k in pred:
                tp += 1
                st_n += 1; st_ok += pred[k].status.value == g["status"]
                if g["status"] == "present" and g["concept"] != "dysmenorrhoea":
                    ln_n += 1; ln_ok += pred[k].menstrual_link.value == g["menstrual_link"]
            else:
                fn += 1
        fp += sum(1 for k in pred if k not in gold)
    return {"symptom_precision": round(tp / max(1, tp + fp), 3), "symptom_recall": round(tp / max(1, tp + fn), 3),
            "status_accuracy_when_found": round(st_ok / max(1, st_n), 3), "menstrual_link_accuracy_when_found": round(ln_ok / max(1, ln_n), 3),
            "note": "Present/absent/uncertain status and menstrual link scored against the generator's ground truth, per encounter."}


def _roc(y: list[int], p: list[float]) -> list[list[float]]:
    from sklearn.metrics import roc_curve
    if len(set(y)) < 2:
        return []
    fpr, tpr, _ = roc_curve(y, p)
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(fpr, tpr)]


def _threshold_curve(extractor: str, ath: Athar) -> dict | None:
    """Out-of-fold sensitivity and specificity on the development set at every threshold, as used to pick the operating point."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from ..model.model import candidates, to_matrix
    from ..temporal.features import compute_features
    if not ath.model:
        return None
    try:
        recs, truth = load_records("dev"), load_truth("dev")
        exs = extract_all(recs, extractor, "dev")
    except BaseException:
        return None
    X = to_matrix([compute_features(r, exs[r.patient_id]).values for r in recs])
    y = np.array([int(truth[r.patient_id]["endometriosis"]) for r in recs])
    p = cross_val_predict(candidates()[ath.model.meta["chosen_model"]], X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    pts = []
    for t in np.round(np.arange(0.02, 1.0, 0.02), 2):
        sens = float(((p >= t) & (y == 1)).sum() / max(1, (y == 1).sum()))
        spec = float(((p < t) & (y == 0)).sum() / max(1, (y == 0).sum()))
        pts.append({"t": float(t), "sensitivity": round(sens, 3), "specificity": round(spec, 3)})
    return {"points": pts, "chosen": ath.model.meta["threshold"], "rule": ath.model.meta["threshold_rule"], "n_dev": int(len(y))}


def _documentation_groups(rows: list[dict]) -> dict:
    flag = lambda rs: [int(x["cat"] == Category.review.value) for x in rs]
    def grp(rs):
        return {"n": len(rs), "label_positive": sum(x["label"] for x in rs), **{k: binary([x["label"] for x in rs], flag(rs))[k] for k in ("sensitivity", "specificity")}}
    with_pain = [x for x in rows if x["relevant"] > 0]
    return {
        "by_record_length": {
            "1 to 3 documented visits": grp([x for x in rows if x["content_visits"] <= 3]),
            "4 to 7 documented visits": grp([x for x in rows if 4 <= x["content_visits"] <= 7]),
            "8 or more documented visits": grp([x for x in rows if x["content_visits"] >= 8]),
        },
        "by_note_completeness": {
            "Rarely recorded (under a third)": grp([x for x in with_pain if x["doc_completeness"] < 1 / 3]),
            "Sometimes recorded": grp([x for x in with_pain if 1 / 3 <= x["doc_completeness"] < 2 / 3]),
            "Usually recorded (two thirds or more)": grp([x for x in with_pain if x["doc_completeness"] >= 2 / 3]),
        },
        "note": "Note completeness = share of pain visits whose note records the link to periods, the severity or the response to treatment. "
                "Only patients with at least one relevant pain visit are grouped by completeness.",
    }


def evaluate(extractor: str = "auto") -> dict:
    recs, truth = load_records("eval"), load_truth("eval")
    exs = extract_all(recs, extractor, "eval")
    ath = Athar()
    from ..agents import sceptic as sc
    sceptic = sc.load_results("eval")
    rows = []
    for r in recs:
        a = ath.assess(r, exs[r.patient_id])
        t = truth[r.patient_id]
        res = sceptic.get(r.patient_id)
        if res and (res.digest != sc.digest(r, exs[r.patient_id]) or res.original_category != a.category.value):
            res = None
        rows.append({"id": r.patient_id, "cat": a.category.value, "cat_raw": a.category.value,
                     "cat_sceptic": sc.apply_to_assessment(a, res).category.value, "sceptic_ran": bool(res and res.ran),
                     "sceptic_changed": bool(res and res.changed), "sceptic_verdict": res.verdict if res and res.ran else None,
                     "doc_completeness": a.features.get("documentation_completeness", 0.0), "content_visits": a.features.get("n_content_encounters", 0.0),
                     "relevant": a.relevant_encounters, "score": a.model_score, "label": int(t["endometriosis"]),
                     "expected": t["expected_category"], "scenario": t["scenario_name"], "subtype": t["subtype"],
                     "quality_issues": len([i for i in a.data_quality if i.kind != "unverified_quote"])})
    flagged_raw = [x for x in rows if x["cat_raw"] == Category.review.value]
    coverage = sum(x["sceptic_ran"] for x in flagged_raw) / max(1, len(flagged_raw))
    use_sceptic = coverage == 1.0 and len(flagged_raw) > 0
    if use_sceptic:
        for x in rows:
            x["cat"] = x["cat_sceptic"]
    flag = lambda rs: [int(x["cat"] == Category.review.value) for x in rs]
    flag_by = lambda rs, k: [int(x[k] == Category.review.value) for x in rs]
    conf3 = {e: {p: sum(1 for x in rows if x["expected"] == e and x["cat"] == p) for p in CATS} for e in CATS}
    acc3 = sum(conf3[c][c] for c in CATS) / len(rows)
    scored = [x for x in rows if x["score"] is not None]
    report = {
        "caveat": CAVEAT, "extractor": next(iter(exs.values())).extractor, "model": ath.model.meta if ath.model else None,
        "n_patients": len(rows), "category_counts": dict(Counter(x["cat"] for x in rows)),
        "A_constructed_scenarios": {
            "description": "Athar's category vs the category the scenario was built to show (documented pattern, not diagnosis).",
            "confusion_matrix_expected_rows_predicted_cols": conf3, "three_class_accuracy": round(acc3, 3),
            "review_vs_not": binary([int(x["expected"] == Category.review.value) for x in rows], flag(rows)),
            "insufficient_detection": binary([int(x["expected"] == Category.insufficient.value) for x in rows],
                                             [int(x["cat"] == Category.insufficient.value) for x in rows]),
        },
        "B_hidden_label": {
            "description": "Flag (PATTERN WARRANTS CLINICAL REVIEW) vs the synthetic hidden endometriosis label. INSUFFICIENT counts as not flagged.",
            "all_patients": binary([x["label"] for x in rows], flag(rows)),
            "model_score_auc": round(roc_auc_score([x["label"] for x in scored], [x["score"] for x in scored]), 3) if scored else None,
        },
        "pipeline": "LLM reader + model + rules + sceptic agent" if use_sceptic else "LLM reader + model + rules (sceptic not run on every flag)",
        "sceptic": {
            "coverage": round(coverage, 3), "flags_reviewed": sum(x["sceptic_ran"] for x in flagged_raw), "flags_total": len(flagged_raw),
            "downgraded": sum(x["sceptic_changed"] for x in rows),
            "downgraded_label_negative": sum(1 for x in rows if x["sceptic_changed"] and not x["label"]),
            "downgraded_label_positive": sum(1 for x in rows if x["sceptic_changed"] and x["label"]),
            "without": {"vs_label": binary([x["label"] for x in rows], flag_by(rows, "cat_raw")),
                        "vs_constructed": binary([int(x["expected"] == Category.review.value) for x in rows], flag_by(rows, "cat_raw"))},
            "with": {"vs_label": binary([x["label"] for x in rows], flag_by(rows, "cat_sceptic")),
                     "vs_constructed": binary([int(x["expected"] == Category.review.value) for x in rows], flag_by(rows, "cat_sceptic"))},
            "note": "Both rows use the same 200 patients and the same extraction; only the sceptic step differs.",
            "by_verdict": {v: {"n": sum(1 for x in rows if x["sceptic_verdict"] == v),
                               "label_positive": sum(1 for x in rows if x["sceptic_verdict"] == v and x["label"])}
                           for v in ("upheld", "weakened", "challenged")},
        },
        "documentation": _documentation_groups(rows),
        "roc_eval": _roc([x["label"] for x in scored], [x["score"] for x in scored]),
        "threshold_curve": _threshold_curve(extractor, ath),
        "subgroups": {},
        "extraction_quality": extraction_quality(recs, exs, truth),
    }
    groups = {
        "incomplete_records": lambda x: x["scenario"] == "insufficient_records" or x["quality_issues"] > 0,
        "alternative_condition_scenarios": lambda x: x["scenario"] in ("gi_non_menstrual", "overlapping_conditions", "pelvic_pain_no_menstrual_link"),
        "varied_terminology": lambda x: x["scenario"] == "varied_terminology",
        "inconsistent_documentation": lambda x: x["scenario"] == "inconsistent_documentation",
        "silent_positives_and_routine": lambda x: x["scenario"] == "routine_unrelated",
    }
    for g, fn in groups.items():
        rs = [x for x in rows if fn(x)]
        report["subgroups"][g] = {
            "n": len(rs), "categories": dict(Counter(x["cat"] for x in rs)),
            "vs_constructed": binary([int(x["expected"] == Category.review.value) for x in rs], flag(rs)),
            "vs_hidden_label": binary([x["label"] for x in rs], flag(rs)),
        }
    report["errors_sample"] = {
        "false_negatives_vs_label": [x for x in rows if x["label"] and x["cat"] != Category.review.value][:15],
        "false_positives_vs_label": [x for x in rows if not x["label"] and x["cat"] == Category.review.value][:15],
    }
    return report


def to_markdown(rep: dict) -> str:
    A, B = rep["A_constructed_scenarios"], rep["B_hidden_label"]
    m = rep.get("model") or {}
    L = ["# Athar evaluation report (synthetic held-out set)", "", f"> {rep['caveat']}", "",
         f"Extractor: `{rep['extractor']}` · Model: `{m.get('chosen_model')}` (threshold {m.get('threshold')}, {m.get('threshold_rule', '')}) · Patients: {rep['n_patients']}", "",
         "## Model comparison (5-fold CV on development set)", "", "| Model | CV ROC-AUC | CV avg precision |", "|---|---|---|"]
    for k, v in (m.get("candidates") or {}).items():
        L.append(f"| {k} | {v['cv_auc']} | {v['cv_average_precision']} |")
    L += ["", "## A. Constructed-scenario agreement", "", "Rows = category the scenario was built to show; columns = Athar's output.", "",
          "| expected \\ predicted | REVIEW | INSUFFICIENT | NO PATTERN |", "|---|---|---|---|"]
    short = {CATS[0]: "REVIEW", CATS[1]: "INSUFFICIENT", CATS[2]: "NO PATTERN"}
    for e in CATS:
        L.append(f"| {short[e]} | " + " | ".join(str(A["confusion_matrix_expected_rows_predicted_cols"][e][p]) for p in CATS) + " |")
    L += ["", f"Three-class accuracy: **{A['three_class_accuracy']}**", ""]

    def tbl(title, d):
        return [f"**{title}** — sensitivity {d['sensitivity']}, specificity {d['specificity']}, precision {d['precision']}, F1 {d['f1']}, "
                f"FPR {d['false_positive_rate']}, FNR {d['false_negative_rate']} (TP {d['tp']}, FP {d['fp']}, TN {d['tn']}, FN {d['fn']})", ""]
    L += tbl("Review vs not (constructed)", A["review_vs_not"])
    L += tbl("Insufficient-information detection", A["insufficient_detection"])
    L += ["## B. Agreement with the hidden synthetic label", ""] + tbl("Flag vs hidden label", B["all_patients"])
    L += [f"Model-score ROC-AUC on the evaluation set: **{B['model_score_auc']}** (published symptom-based models: ~0.65–0.85).", "",
          "## Subgroups", "", "| Subgroup | n | sens (constructed) | spec (constructed) | sens (label) | spec (label) |", "|---|---|---|---|---|---|"]
    for g, d in rep["subgroups"].items():
        c, h = d["vs_constructed"], d["vs_hidden_label"]
        L.append(f"| {g} | {d['n']} | {c['sensitivity']} | {c['specificity']} | {h['sensitivity']} | {h['specificity']} |")
    e = rep["extraction_quality"]
    L += ["", "## Extraction quality (vs generator ground truth)", "",
          f"Symptom precision {e['symptom_precision']}, recall {e['symptom_recall']}, status accuracy {e['status_accuracy_when_found']}, "
          f"menstrual-link accuracy {e['menstrual_link_accuracy_when_found']}."]
    if "extractor_comparison" in rep:
        L += ["", "## LLM vs keyword extractor", "", "| Extractor | symptom recall | precision | flag sensitivity (label) | specificity (label) |", "|---|---|---|---|---|"]
        for k, v in rep["extractor_comparison"].items():
            L.append(f"| {k} | {v['symptom_recall']} | {v['symptom_precision']} | {v['sensitivity']} | {v['specificity']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default="auto", choices=["auto", "lexicon", "llm"])
    ap.add_argument("--compare", action="store_true", help="also evaluate the other extractor (needs both)")
    a = ap.parse_args()
    rep = evaluate(a.extractor)
    if a.compare:
        comp = {}
        for name in ("lexicon", "llm"):
            try:
                r = evaluate(name)
                comp[r["extractor"]] = {**r["extraction_quality"], **{k: r["B_hidden_label"]["all_patients"][k] for k in ("sensitivity", "specificity")}}
            except BaseException as ex:  # e.g. no API key for the LLM extractor
                comp[name] = {"error": str(ex), "symptom_recall": None, "symptom_precision": None, "sensitivity": None, "specificity": None}
        rep["extractor_comparison"] = comp
    (OUTPUT_DIR / "evaluation_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (OUTPUT_DIR / "evaluation_report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
