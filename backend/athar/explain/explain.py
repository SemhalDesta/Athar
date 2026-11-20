"""Explainability: build a fully traceable Assessment and a plain-language summary."""
from __future__ import annotations

import datetime as dt
import json

from .. import llm
from ..engine.decide import Decision
from ..schemas import Assessment, Category, Contributor, EvidenceRef, ExtractionResult, Status
from ..temporal.features import FEATURE_LABELS, FeatureSet
from ..vocab import ALTERNATIVE_DIAGNOSES, INVESTIGATIONS, MEDICATIONS, PAIN_CONCEPTS, SYMPTOMS


def _d(x):
    return x.encounter_date or dt.date.max


def build_assessment(rec_id: str, ex: ExtractionResult, fs: FeatureSet, decision: Decision,
                     score: float | None, shap: dict[str, float] | None, rec=None) -> Assessment:
    skip = {i.encounter_id for i in ex.issues if i.kind == "duplicate_encounter"}
    F = [f for f in ex.findings if f.encounter_id not in skip]
    contributors: list[Contributor] = []
    if shap:
        for feat, sv in sorted(shap.items(), key=lambda kv: -abs(kv[1])):
            val = fs.values.get(feat, 0.0)
            if abs(sv) < 0.02:
                continue
            label = FEATURE_LABELS.get(feat, feat)
            if val == 0 and feat.startswith("n_"):
                label += " — none documented (not the same as absent)"
            contributors.append(Contributor(feature=feat, label=label, value=val, shap=round(sv, 3),
                                            direction="raises" if sv > 0 else "lowers", evidence=fs.evidence.get(feat, [])[:8]))
            if len(contributors) >= 8:
                break
    timeline = sorted([EvidenceRef(encounter_id=f.encounter_id, encounter_date=f.encounter_date, quote=f.evidence_quote,
                                   concept=SYMPTOMS.get(f.concept, {}).get("label", f.concept), status=f.status.value,
                                   menstrual_link=f.menstrual_link.value)
                       for f in F if f.category == "symptom"], key=_d)
    tx = []
    for f in sorted([f for f in F if f.category in ("medication", "investigation", "referral", "treatment_response")], key=_d):
        if f.category == "medication":
            label = MEDICATIONS.get(f.concept, {}).get("label", f.concept)
        elif f.category == "investigation":
            label = f"{INVESTIGATIONS.get(f.concept, {}).get('label', f.concept)} ({(f.result or 'unknown').replace('_', ' ')})"
        elif f.category == "referral":
            label = f"Referral: {f.concept}"
        else:
            label = "Treatment not working" if f.concept == "treatment_failure" else "Improved on treatment"
        tx.append(EvidenceRef(encounter_id=f.encounter_id, encounter_date=f.encounter_date, quote=f.evidence_quote, concept=label, status=f.category))
    alts = []
    seen = set()
    for f in F:
        key = f.concept if f.category == "diagnosis" else ((f.result or "").split(":")[1] if (f.result or "").startswith("abnormal:") else None)
        if key in ALTERNATIVE_DIAGNOSES and key not in seen:
            seen.add(key)
            info = ALTERNATIVE_DIAGNOSES[key]
            alts.append(EvidenceRef(encounter_id=f.encounter_id, encounter_date=f.encounter_date, quote=f.evidence_quote,
                                    concept=info["label"] + (f" — {info['caution']}" if info["caution"] else ""), status="documented"))
    a = Assessment(
        patient_id=rec_id, category=decision.category, model_score=round(score, 3) if score is not None else None,
        priority=decision.priority, reasons=decision.reasons, guardrails_applied=decision.guardrails, contributors=contributors,
        features=fs.values, relevant_encounters=int(fs.values["n_relevant_encounters"]), timeline=timeline,
        treatments_and_investigations=tx, alternative_explanations=alts, missing_or_uncertain=fs.missing,
        data_quality=ex.issues, extractor=ex.extractor,
    )
    if rec is not None:
        from .temporal_summary import summarise
        a.temporal_patterns, a.visits_by_year = summarise(rec, ex)
    a.summary = template_summary(a)
    return a


def template_summary(a: Assessment) -> str:
    v = a.features
    if a.category == Category.insufficient:
        return f"Not enough documented history to assess a pattern. {a.reasons[0] if a.reasons else ''}"
    parts = []
    syms = sorted({t.concept for t in a.timeline if t.status == Status.present.value})
    if syms:
        parts.append("Documented symptoms: " + ", ".join(s.split(" (")[0].lower() for s in syms) + ".")
    if v.get("n_relevant_encounters"):
        parts.append(f"Relevant pain appears in {int(v['n_relevant_encounters'])} separate encounters over {v['pain_span_months']:.0f} months.")
    if v.get("n_cyclical_nonmenstrual"):
        parts.append("Non-period pain is explicitly linked to the menstrual cycle in the notes.")
    if v.get("persist_after_treatment"):
        parts.append("Pain continues to be documented after pain treatment was started.")
    if v.get("persist_after_normal_investigation"):
        parts.append("Pain persists after a normal investigation (a normal result does not rule out endometriosis, NICE NG73).")
    if a.alternative_explanations:
        parts.append("Alternative explanations documented: " + "; ".join(x.concept.split(" — ")[0] for x in a.alternative_explanations) + ".")
    lead = {
        Category.review: "The documented history shows a pattern that may warrant clinical assessment for possible endometriosis.",
        Category.no_pattern: "The available records do not currently show a notable pattern warranting an endometriosis-related flag.",
    }[a.category]
    return lead + " " + " ".join(parts)


SUMMARY_SYSTEM = """You write a short, neutral summary (max 90 words) for a clinician about a screening-support output.
Use ONLY the facts in the JSON provided. Do not diagnose, do not give probabilities, do not add facts, do not
recommend treatment. Mention the category, the main documented features with dates, important missing
information, and any documented alternative explanations. Return JSON: {"summary": "..."}"""


def llm_summary(a: Assessment) -> Assessment:
    if not llm.available():
        return a
    facts = {
        "category": a.category.value, "reasons": a.reasons, "guardrails": a.guardrails_applied,
        "top_contributors": [{"feature": c.label, "direction": c.direction, "value": c.value} for c in a.contributors[:5]],
        "timeline": [{"date": str(t.encounter_date), "symptom": t.concept, "status": t.status} for t in a.timeline[:25]],
        "treatments_investigations": [{"date": str(t.encounter_date), "item": t.concept} for t in a.treatments_and_investigations[:15]],
        "alternative_explanations": [x.concept for x in a.alternative_explanations],
        "missing_or_uncertain": a.missing_or_uncertain,
    }
    try:
        out = llm.chat_json(SUMMARY_SYSTEM, json.dumps(facts, default=str), temperature=0.2, max_tokens=400)
        if out.get("summary"):
            a.summary, a.summary_source = out["summary"], "llm"
    except Exception:
        pass
    return a
