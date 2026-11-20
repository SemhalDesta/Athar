"""Sceptic agent: an LLM reviewer that argues against each flag.

It may raise three kinds of challenge, and every one must quote the record word for word:
  misread      a finding the quote does not support (negation, hedging, history, someone else)
  alternative  another condition documented in the record that could explain the pain
  gap          something needed to judge the pattern that was never recorded (informational)

Code, not the agent, decides what a challenge is worth:
  * quotes that cannot be found verbatim in the cited visit are discarded;
  * verified misreads correct the extracted findings, then features, model and rules run again;
  * a verified major alternative explanation downgrades a flag to INSUFFICIENT INFORMATION;
  * the outcome can only move down. The sceptic can never create a flag.

CLI (needs OPENROUTER_API_KEY):
    python -m athar.agents.sceptic --dataset eval            # all flagged patients, resumable
    python -m athar.agents.sceptic --dataset eval --limit 5  # quick test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .. import llm
from ..config import OUTPUT_DIR, rules
from ..schemas import Assessment, Category, ExtractionResult, PatientRecord, Status

RANK = {Category.no_pattern: 0, Category.insufficient: 1, Category.review: 2}
CORRECTIONS = ("remove", "set_absent", "set_uncertain", "flip_treatment_response")


class Challenge(BaseModel):
    type: Literal["misread", "alternative", "gap"]
    severity: Literal["major", "minor"] = "minor"
    claim: str
    encounter_id: Optional[str] = None
    quote: str = ""
    finding_ids: list[str] = Field(default_factory=list)
    correction: Optional[str] = None
    verified: bool = False
    rejected_reason: Optional[str] = None


class ScepticResult(BaseModel):
    patient_id: str
    ran: bool
    model: Optional[str] = None
    digest: str = ""
    verdict: Literal["upheld", "weakened", "challenged", "not_run"] = "not_run"
    summary: str = ""
    challenges: list[Challenge] = Field(default_factory=list)
    original_category: str
    final_category: str
    changed: bool = False
    reason: str = ""
    corrections_applied: list[str] = Field(default_factory=list)
    score_before: Optional[float] = None
    score_after: Optional[float] = None


SYSTEM = """You are the sceptic on a clinical decision-support team. Another system has flagged a patient's records as showing a
pattern that may warrant assessment for possible endometriosis. Your only job is to look for reasons the flag could be WRONG.
You never diagnose, and you can never add a flag or a symptom.

You receive every visit note (verbatim) and the findings another AI extracted, each with an id (F1, F2, ...) and the quote it relied on.

Raise a challenge only when the record supports it. Three kinds:
1. "misread": an extracted finding is not supported by its own note. Examples: a denied symptom recorded as present
   ("denies dyspareunia"), a hedge recorded as certain, a symptom from the past or from a relative, a treatment response read
   the wrong way round ("not much better with the pill" recorded as improvement). Give the finding ids and one correction:
   "remove", "set_absent", "set_uncertain" or "flip_treatment_response".
2. "alternative": another condition DOCUMENTED in the notes that could explain the relevant pain. Severity "major" only if the
   record shows it was confirmed (positive test, imaging or specialist diagnosis) AND the pain settled after it was treated.
   A working label alone (IBS, primary dysmenorrhoea, UTI without a positive culture, PID without a positive test) is "minor":
   endometriosis is often first labelled as one of these, and NICE NG73 says a normal ultrasound does not exclude it.
3. "gap": something needed to judge the pattern was never recorded (e.g. link to the menstrual cycle, effect of treatment).
   Gaps are minor and need no quote.

Rules:
- Every misread or alternative MUST include the encounter_id and a quote copied EXACTLY, character for character, from that note.
- Do not speculate about conditions that are not written in the notes.
- If the flag holds up, say so: return verdict "upheld" and an empty challenges list. Most flags should hold up.
- Keep claims to one plain sentence a doctor can check.

Return JSON only:
{"verdict": "upheld" | "weakened" | "challenged",
 "summary": "one or two plain sentences",
 "challenges": [{"type": "misread|alternative|gap", "severity": "major|minor", "claim": "...",
                 "encounter_id": "...", "quote": "...", "finding_ids": ["F3"], "correction": "remove|set_absent|set_uncertain|flip_treatment_response"}]}"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip(" .;,")


def _encounter_text(rec: PatientRecord, eid: str) -> str:
    for e in rec.encounters:
        if e.encounter_id == eid:
            parts = [e.note_text, " ".join(e.coded_diagnoses)]
            parts += [m.name for m in e.medications] + [i.result_text for i in e.investigations] + [f"{r.to} {r.reason}" for r in e.referrals]
            return " \n ".join(p for p in parts if p)
    return ""


def digest(rec: PatientRecord, ex: ExtractionResult) -> str:
    return hashlib.sha256((rec.model_dump_json() + ex.model_dump_json()).encode()).hexdigest()[:16]


def _payload(rec: PatientRecord, ex: ExtractionResult, a: Assessment) -> tuple[str, dict[str, int]]:
    ids: dict[str, int] = {}
    findings = []
    for i, f in enumerate(ex.findings):
        fid = f"F{i + 1}"
        ids[fid] = i
        findings.append({"id": fid, "encounter_id": f.encounter_id, "date": str(f.encounter_date), "category": f.category,
                         "concept": f.concept, "status": f.status.value, "menstrual_link": f.menstrual_link.value, "quote": f.evidence_quote})
    notes = [{"encounter_id": e.encounter_id, "date": str(e.date), "setting": e.setting, "note": e.note_text,
              "coded_diagnoses": e.coded_diagnoses, "prescriptions": [m.name for m in e.medications],
              "results": [f"{i.type}: {i.result_text}" for i in e.investigations], "referrals": [f"{r.to}: {r.reason}" for r in e.referrals]}
             for e in rec.encounters]
    body = {"flag": a.category.value, "reasons": a.reasons, "patient_birth_year": rec.birth_year, "visits": notes, "extracted_findings": findings}
    return json.dumps(body, ensure_ascii=False), ids


def _apply_corrections(ex: ExtractionResult, challenges: list[Challenge], ids: dict[str, int]) -> tuple[ExtractionResult, list[str]]:
    findings = [f.model_copy() for f in ex.findings]
    drop, notes = set(), []
    for c in challenges:
        if c.type != "misread" or not c.verified:
            continue
        for fid in c.finding_ids:
            i = ids[fid]
            f = findings[i]
            if c.correction == "remove":
                drop.add(i); notes.append(f"Removed {f.concept} ({f.encounter_id}): {c.claim}")
            elif c.correction == "set_absent":
                f.status = Status.absent; notes.append(f"{f.concept} ({f.encounter_id}) set to denied: {c.claim}")
            elif c.correction == "set_uncertain":
                f.status = Status.uncertain; notes.append(f"{f.concept} ({f.encounter_id}) set to uncertain: {c.claim}")
            elif c.correction == "flip_treatment_response" and f.category == "treatment_response":
                f.concept = "treatment_failure" if f.concept == "treatment_success" else "treatment_success"
                notes.append(f"Treatment response at {f.encounter_id} reversed: {c.claim}")
    out = ex.model_copy()
    out.findings = [f for i, f in enumerate(findings) if i not in drop]
    return out, notes


def _verify(raw: dict, rec: PatientRecord, ex: ExtractionResult, ids: dict[str, int]) -> list[Challenge]:
    out = []
    for item in (raw.get("challenges") or [])[:8]:
        try:
            c = Challenge(**{k: item.get(k) for k in ("type", "severity", "claim", "encounter_id", "quote", "finding_ids", "correction") if item.get(k) is not None})
        except Exception:
            continue
        if c.type == "gap":
            c.verified, c.severity = True, "minor"
            out.append(c); continue
        text = _encounter_text(rec, c.encounter_id or "")
        if not text:
            c.rejected_reason = "cited visit does not exist"
        elif not c.quote or _norm(c.quote) not in _norm(text):
            c.rejected_reason = "quote not found in the cited note"
        elif c.type == "misread":
            fids = [f for f in c.finding_ids if f in ids]
            if not fids:
                c.rejected_reason = "no valid finding id"
            elif c.correction not in CORRECTIONS:
                c.rejected_reason = "unknown correction"
            elif any(ex.findings[ids[f]].encounter_id != c.encounter_id for f in fids):
                c.rejected_reason = "finding is from a different visit"
            else:
                c.finding_ids, c.verified = fids, True
        else:
            c.verified = True
        out.append(c)
    return out


def review(rec: PatientRecord, ex: ExtractionResult, a: Assessment, athar, model: str | None = None) -> ScepticResult:
    """Run the sceptic on one assessed patient. `athar` is a pipeline.Athar used to re-assess after corrections."""
    base = dict(patient_id=rec.patient_id, original_category=a.category.value, final_category=a.category.value,
                digest=digest(rec, ex), score_before=a.model_score, score_after=a.model_score)
    if a.category != Category.review:
        return ScepticResult(ran=False, reason="Only flagged records are challenged.", **base)
    body, ids = _payload(rec, ex, a)
    raw = llm.chat_json(SYSTEM, body, temperature=0.0, model=model, max_tokens=1500)
    challenges = _verify(raw if isinstance(raw, dict) else {}, rec, ex, ids)
    res = ScepticResult(ran=True, model=model or llm.llm_model(), verdict=(raw or {}).get("verdict", "upheld") if isinstance(raw, dict) else "upheld",
                        summary=str((raw or {}).get("summary", ""))[:600] if isinstance(raw, dict) else "", challenges=challenges, **base)
    if res.verdict not in ("upheld", "weakened", "challenged"):
        res.verdict = "weakened" if any(c.verified and c.type != "gap" for c in challenges) else "upheld"

    final, reasons = a.category, []
    corrected, notes = _apply_corrections(ex, challenges, ids)
    if notes:
        after = athar.assess(rec, corrected)
        res.corrections_applied, res.score_after = notes, after.model_score
        if RANK[after.category] < RANK[final]:
            final = after.category
            reasons.append(f"After correcting {len(notes)} misread finding(s), the pattern no longer meets the flag criteria.")
    if final == Category.review and any(c.verified and c.type == "alternative" and c.severity == "major" for c in challenges):
        final = Category.insufficient
        reasons.append("A confirmed alternative explanation is documented; more information is needed to tell the two apart.")
    res.final_category, res.changed = final.value, final != a.category
    res.reason = " ".join(reasons) if reasons else "The flag stands. Challenges are shown for the doctor to weigh."
    return res


# ---------------------------------------------------------------------------- storage

def dataset_extractor(dataset: str) -> str:
    """The extractor the app uses: cached LLM extractions when they exist, otherwise the keyword baseline."""
    from ..config import CACHE_DIR
    return "llm" if any(CACHE_DIR.glob(f"extraction_{dataset}_llm*.json")) else "lexicon"


def results_path(dataset: str):
    return OUTPUT_DIR / f"sceptic_{dataset}.json"


def load_results(dataset: str) -> dict[str, ScepticResult]:
    p = results_path(dataset)
    if not p.exists():
        return {}
    return {k: ScepticResult(**v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}


def apply_to_assessment(a: Assessment, res: Optional[ScepticResult]) -> Assessment:
    """Return the assessment with the sceptic's (downgrade-only) effect applied."""
    if not res or not res.changed:
        return a
    out = a.model_copy(deep=True)
    out.category = Category(res.final_category)
    out.reasons = out.reasons + [f"Sceptic agent: {res.reason}"]
    out.guardrails_applied = out.guardrails_applied + [f"Flag downgraded by the sceptic agent: {res.reason}"]
    return out


def run(dataset: str, limit: int | None = None, ids: list[str] | None = None) -> dict[str, ScepticResult]:
    from ..pipeline import Athar, extract_all, load_records
    if not llm.available():
        raise SystemExit("OPENROUTER_API_KEY is not set: the sceptic needs the LLM.")
    recs = load_records(dataset)
    exs = extract_all(recs, dataset_extractor(dataset), dataset)
    ath = Athar()
    done = load_results(dataset)
    todo = []
    for r in recs:
        if ids and r.patient_id not in ids:
            continue
        a = ath.assess(r, exs[r.patient_id])
        prev = done.get(r.patient_id)
        if prev and prev.digest == digest(r, exs[r.patient_id]) and prev.original_category == a.category.value:
            continue
        todo.append((r, a))
    flagged = [t for t in todo if t[1].category == Category.review]
    if limit:
        flagged = flagged[:limit]
    for r, a in todo:  # records that are not flagged are stored as "not run", no LLM call
        if a.category != Category.review:
            done[r.patient_id] = review(r, exs[r.patient_id], a, ath)
    print(f"Sceptic: {len(flagged)} flagged record(s) to review in {dataset} ({len(done)} already stored).", flush=True)

    def one(t):
        r, a = t
        try:
            return review(r, exs[r.patient_id], a, ath), None
        except Exception as e:
            return None, (r.patient_id, e)

    workers = rules()["llm"]["max_concurrency"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (res, err) in enumerate(pool.map(one, flagged), 1):
            if res:
                done[res.patient_id] = res
            else:
                print(f"  failed {err[0]}: {err[1]} (run again to retry)", flush=True)
            if i % 10 == 0 or i == len(flagged):
                results_path(dataset).write_text(json.dumps({k: json.loads(v.model_dump_json()) for k, v in done.items()}, indent=1), encoding="utf-8")
                print(f"  {i}/{len(flagged)} reviewed", flush=True)
    results_path(dataset).write_text(json.dumps({k: json.loads(v.model_dump_json()) for k, v in done.items()}, indent=1), encoding="utf-8")
    ran = [v for v in done.values() if v.ran]
    print(f"Done. Reviewed {len(ran)} flags: {sum(v.verdict == 'upheld' for v in ran)} upheld, "
          f"{sum(v.changed for v in ran)} downgraded, "
          f"{sum(1 for v in ran for c in v.challenges if c.verified)} verified and {sum(1 for v in ran for c in v.challenges if not c.verified)} rejected challenges.")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval", choices=["eval", "dev"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    a = ap.parse_args()
    run(a.dataset, a.limit, a.ids)
    if a.dataset == "eval":
        print("Now run: py -m athar.evaluation.evaluate --compare   (adds the with/without-sceptic comparison)")


if __name__ == "__main__":
    main()
