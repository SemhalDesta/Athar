"""LLM clinical information extractor (OpenRouter).

The LLM reads free-text notes and returns findings in a fixed JSON schema. Safety measures:
  * Every finding must carry a verbatim evidence quote; quotes not found in the source note are
    marked unverified and excluded from features (hallucination guard).
  * "Not documented" is never emitted as a finding: missing information stays missing.
  * Output is validated against the Pydantic schema; invalid items are dropped and reported.
  * Structured EHR fields are mapped deterministically (no LLM).
"""
from __future__ import annotations

import json
import re

from .. import llm
from ..schemas import DataQualityIssue, ExtractionResult, Finding, MenstrualLink, PatientRecord, Status
from ..vocab import ALTERNATIVE_DIAGNOSES, SYMPTOMS
from .common import cross_encounter_contradictions, merge_encounter_findings, quality_checks, structured_findings

SYMPTOM_GUIDE = "\n".join(f"- {k}: {v['label']}" for k, v in SYMPTOMS.items())
ALT_GUIDE = "\n".join(f"- {k}: {v['label']}" for k, v in ALTERNATIVE_DIAGNOSES.items())

SYSTEM = f"""You are a clinical information extraction component in a decision-support prototype.
You read fictional clinical notes and extract ONLY what is explicitly written. You never diagnose.

Symptom concepts (use these codes only):
{SYMPTOM_GUIDE}

Alternative diagnoses the clinician states as a working/confirmed diagnosis (codes):
{ALT_GUIDE}

For every symptom mention return:
- concept: one code above
- status: "present" (documented as present), "absent" (explicitly denied/negative, e.g. "denies", "no", "nil"),
  or "uncertain" ("?", "possible", "query", "unsure", vague history). If a symptom is not mentioned, return nothing.
- menstrual_link: "explicit_yes" only if the note explicitly relates it to periods/menses/cycle
  (e.g. "worse with periods", "cyclical", "premenstrual", "catamenial"); "explicit_no" only if the note explicitly
  says it is not related; otherwise "not_documented". Painful periods are always "explicit_yes".
- severity: "mild" | "moderate" | "severe" | null (only if stated, e.g. "severe", "8/10")
- functional_impact: true only if the note states impact (missed work/school, bed-bound, ED attendance for pain)
- evidence_quote: the exact contiguous text from the note (copy verbatim, including typos), max ~25 words

Also return, per encounter:
- treatment_response: "treatment_failure" if the note says treatment is not working / pain persists despite it,
  "treatment_success" if it says symptoms improved/settled, otherwise omit. Include an evidence_quote.
- diagnoses: alternative diagnoses the clinician states (not questioned ones like "?IBS"), with evidence_quote.

Handle abbreviations (c/o, HMB, LIF, TVUS, NAD), typos and synonyms ("crippling cramps each month" = dysmenorrhoea,
"stabbing pain passing stool" = dyschezia, "pain with relations" = dyspareunia).

Return JSON exactly:
{{"encounters": [{{"encounter_id": "...", "symptoms": [{{"concept": "...", "status": "...", "menstrual_link": "...",
"severity": null, "functional_impact": false, "evidence_quote": "..."}}], "treatment_response": [{{"concept": "treatment_failure",
"evidence_quote": "..."}}], "diagnoses": [{{"concept": "ibs", "evidence_quote": "..."}}]}}]}}"""

BATCH = 8


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower()).strip(" .")


def _verified(quote: str, note: str) -> bool:
    q, n = _norm(quote), _norm(note)
    if not q:
        return False
    if q in n:
        return True
    # tolerate minor punctuation differences: require 90% of quote tokens in order-insensitive match
    qt = q.split()
    return len(qt) >= 3 and sum(t in n for t in qt) / len(qt) >= 0.9


class LLMExtractor:
    name = "llm"

    def __init__(self, model: str | None = None):
        self.model = model

    def _call(self, batch: list[dict]) -> dict:
        return llm.chat_json(SYSTEM, json.dumps({"encounters": batch}, ensure_ascii=False), temperature=0.0, model=self.model, max_tokens=6000)

    def extract(self, rec: PatientRecord) -> ExtractionResult:
        issues, skip = quality_checks(rec)
        findings = structured_findings(rec, skip)
        enc = {e.encounter_id: e for e in rec.encounters if e.encounter_id not in skip and e.note_text.strip()}
        items = [{"encounter_id": k, "note_text": e.note_text} for k, e in enc.items()]
        for i in range(0, len(items), BATCH):
            out = self._call(items[i : i + BATCH])
            for er in out.get("encounters", []):
                eid = er.get("encounter_id")
                if eid not in enc:
                    continue
                e = enc[eid]
                for s in er.get("symptoms", []) or []:
                    try:
                        if s.get("concept") not in SYMPTOMS:
                            continue
                        f = Finding(
                            encounter_id=eid, encounter_date=e.date, category="symptom", concept=s["concept"],
                            status=Status(s.get("status", "present")),
                            menstrual_link=MenstrualLink(s.get("menstrual_link") or "not_documented"),
                            severity=s.get("severity") if s.get("severity") in ("mild", "moderate", "severe") else None,
                            functional_impact=bool(s.get("functional_impact")), evidence_quote=s.get("evidence_quote", ""),
                            extractor="llm",
                        )
                    except (ValueError, KeyError) as ex:
                        issues.append(DataQualityIssue(encounter_id=eid, kind="unverified_quote", detail=f"Invalid LLM item dropped: {ex}"))
                        continue
                    if f.concept == "dysmenorrhoea":
                        f.menstrual_link = MenstrualLink.explicit_yes
                    f.quote_verified = _verified(f.evidence_quote, e.note_text)
                    findings.append(f)
                for t in er.get("treatment_response", []) or []:
                    if t.get("concept") in ("treatment_failure", "treatment_success"):
                        findings.append(Finding(encounter_id=eid, encounter_date=e.date, category="treatment_response", concept=t["concept"],
                                                evidence_quote=t.get("evidence_quote", ""), extractor="llm",
                                                quote_verified=_verified(t.get("evidence_quote", ""), e.note_text)))
                for d in er.get("diagnoses", []) or []:
                    if d.get("concept") in ALTERNATIVE_DIAGNOSES:
                        findings.append(Finding(encounter_id=eid, encounter_date=e.date, category="diagnosis", concept=d["concept"],
                                                evidence_quote=d.get("evidence_quote", ""), extractor="llm",
                                                quote_verified=_verified(d.get("evidence_quote", ""), e.note_text)))
        unverified = [f for f in findings if not f.quote_verified]
        for f in unverified:
            issues.append(DataQualityIssue(encounter_id=f.encounter_id, kind="unverified_quote",
                                           detail=f"LLM finding '{f.concept}' could not be matched to the note text and was not used."))
        findings = [f for f in findings if f.quote_verified]
        findings, merge_issues = merge_encounter_findings(findings)
        issues += merge_issues + cross_encounter_contradictions(findings)
        return ExtractionResult(patient_id=rec.patient_id, extractor=f"llm:{self.model or llm.llm_model()}", findings=findings, issues=issues)


def get_extractor(name: str = "auto"):
    from .lexicon import LexiconExtractor

    if name == "lexicon" or (name == "auto" and not llm.available()):
        return LexiconExtractor()
    return LLMExtractor()
