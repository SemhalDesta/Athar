"""Shared extraction logic: structured fields, text cues and data-quality checks.

Structured EHR fields (prescriptions, investigations, referrals, coded diagnoses) are
mapped deterministically; only free-text notes need NLP / the LLM.
"""
from __future__ import annotations

import datetime as dt
import re

from ..schemas import DataQualityIssue, Finding, MenstrualLink, PatientRecord, Status
from ..vocab import (
    ALTERNATIVE_DIAGNOSES, FAMILY_HISTORY_PATTERNS, MEDICATIONS, NORMAL_RESULT_TERMS, STRONG_IMAGING_TERMS,
)


def rx(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


STRONG_IMG_RX = rx(STRONG_IMAGING_TERMS)
NORMAL_RX = rx(NORMAL_RESULT_TERMS)
ALT_RX = {k: rx(v["patterns"]) for k, v in ALTERNATIVE_DIAGNOSES.items()}
FH_RX = rx(FAMILY_HISTORY_PATTERNS)
NUM_RX = re.compile(r"(\d+(?:\.\d+)?)")


def classify_medication(name: str) -> str | None:
    n = name.lower()
    for cls, v in MEDICATIONS.items():
        if any(x.lower() in n for x in v["names"]):
            return cls
    return None


def classify_result(inv_type: str, text: str) -> str:
    """normal | endometriosis_suggestive | abnormal:<detail> | elevated | not_elevated | unknown"""
    t = text or ""
    if inv_type == "ca125":
        m = NUM_RX.search(t)
        if m:
            return "elevated" if float(m.group(1)) >= 35 else "not_elevated"  # NICE: >=35 IU/ml "may be consistent"
        return "unknown"
    if STRONG_IMG_RX.search(t):
        return "endometriosis_suggestive"
    for key, r in ALT_RX.items():
        if r.search(t):
            return f"abnormal:{key}"
    if re.search(r"positive|E\. ?coli|>10\^?5|growth of", t, re.I) and not re.search(r"no growth", t, re.I):
        return "abnormal:positive"
    if NORMAL_RX.search(t):
        return "normal"
    return "unknown"


def structured_findings(rec: PatientRecord, skip: set[str]) -> list[Finding]:
    out: list[Finding] = []
    for e in rec.encounters:
        if e.encounter_id in skip:
            continue
        for m in e.medications:
            cls = m.drug_class or classify_medication(m.name)
            out.append(Finding(encounter_id=e.encounter_id, encounter_date=m.start_date or e.date, category="medication",
                               concept=cls or "other_medication", evidence_quote=m.name, source="structured", extractor="structured"))
        for i in e.investigations:
            out.append(Finding(encounter_id=e.encounter_id, encounter_date=i.date or e.date, category="investigation", concept=i.type,
                               result=classify_result(i.type, i.result_text), evidence_quote=i.result_text, source="structured", extractor="structured"))
        for rf in e.referrals:
            out.append(Finding(encounter_id=e.encounter_id, encounter_date=rf.date or e.date, category="referral", concept=rf.to,
                               evidence_quote=f"Referral to {rf.to}: {rf.reason}", source="structured", extractor="structured"))
        for d in e.coded_diagnoses:
            for key, r in ALT_RX.items():
                if r.search(d):
                    out.append(Finding(encounter_id=e.encounter_id, encounter_date=e.date, category="diagnosis", concept=key,
                                       evidence_quote=d, source="structured", extractor="structured"))
    if rec.family_history_text:
        fh = FH_RX.search(rec.family_history_text)
        out.append(Finding(encounter_id="record", category="family_history", concept="endometriosis",
                           status=Status.present if fh else Status.absent, evidence_quote=rec.family_history_text,
                           source="structured", extractor="structured"))
    return out


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip().lower()


def quality_checks(rec: PatientRecord, today: dt.date | None = None) -> tuple[list[DataQualityIssue], set[str]]:
    """Return issues and the set of encounter ids to exclude from counting (exact duplicates)."""
    today = today or dt.date.today()
    issues: list[DataQualityIssue] = []
    skip: set[str] = set()
    seen: dict[tuple, str] = {}
    prev_text = ""
    for e in rec.encounters:
        if e.date is None:
            issues.append(DataQualityIssue(encounter_id=e.encounter_id, kind="missing_date", detail="Encounter has no date; excluded from timing features."))
        elif e.date > today:
            issues.append(DataQualityIssue(encounter_id=e.encounter_id, kind="future_date", detail=f"Date {e.date} is in the future; excluded from timing features."))
        elif rec.birth_year and e.date.year < rec.birth_year:
            issues.append(DataQualityIssue(encounter_id=e.encounter_id, kind="date_before_birth", detail=f"Date {e.date} is before birth year {rec.birth_year}."))
        txt = _norm(e.note_text)
        if not txt:
            issues.append(DataQualityIssue(encounter_id=e.encounter_id, kind="empty_note", detail="No note text."))
        key = (txt, e.date)
        if txt and key in seen:
            skip.add(e.encounter_id)
            issues.append(DataQualityIssue(encounter_id=e.encounter_id, kind="duplicate_encounter",
                                           detail=f"Identical to {seen[key]} (same date and text); counted once."))
        elif txt and prev_text and len(prev_text) > 40 and txt.startswith(prev_text):
            issues.append(DataQualityIssue(encounter_id=e.encounter_id, kind="duplicate_encounter",
                                           detail="Note copies the previous entry (copy-forward); findings are not counted as a new occurrence."))
            skip.add(e.encounter_id)
        else:
            seen[key] = e.encounter_id
        prev_text = txt
    return issues, skip


def valid_date(d: dt.date | None, today: dt.date | None = None) -> bool:
    return d is not None and d <= (today or dt.date.today())


def merge_encounter_findings(findings: list[Finding]) -> tuple[list[Finding], list[DataQualityIssue]]:
    """Collapse repeated mentions of the same symptom within one encounter (no double counting).
    present > uncertain > absent; an explicit link wins over not documented; a present+absent
    clash inside one note is reported as a contradiction."""
    rank = {Status.present: 3, Status.uncertain: 2, Status.absent: 1}
    merged: dict[tuple, Finding] = {}
    issues: list[DataQualityIssue] = []
    out: list[Finding] = []
    for f in findings:
        if f.category not in ("symptom", "treatment_response", "diagnosis"):
            out.append(f)
            continue
        k = (f.encounter_id, f.category, f.concept)
        if k not in merged:
            merged[k] = f
            continue
        g = merged[k]
        if {g.status, f.status} == {Status.present, Status.absent}:
            issues.append(DataQualityIssue(encounter_id=f.encounter_id, kind="contradiction",
                                           detail=f"'{f.concept}' is both documented and denied in the same note."))
        best = f if rank[f.status] > rank[g.status] else g
        link = g.menstrual_link
        if MenstrualLink.explicit_yes in (g.menstrual_link, f.menstrual_link):
            link = MenstrualLink.explicit_yes
        elif MenstrualLink.explicit_no in (g.menstrual_link, f.menstrual_link):
            link = MenstrualLink.explicit_no
        merged[k] = best.model_copy(update={
            "menstrual_link": link,
            "severity": best.severity or g.severity or f.severity,
            "functional_impact": g.functional_impact or f.functional_impact,
        })
    return out + list(merged.values()), issues


def cross_encounter_contradictions(findings: list[Finding]) -> list[DataQualityIssue]:
    issues = []
    by_concept: dict[str, list[Finding]] = {}
    for f in findings:
        if f.category == "symptom":
            by_concept.setdefault(f.concept, []).append(f)
    for c, fs in by_concept.items():
        st = {f.status for f in fs}
        if Status.present in st and Status.absent in st:
            issues.append(DataQualityIssue(kind="contradiction", detail=f"'{c}' is documented in some encounters and denied in others; both are shown to the clinician."))
        links = {f.menstrual_link for f in fs if f.status == Status.present}
        if MenstrualLink.explicit_yes in links and MenstrualLink.explicit_no in links:
            issues.append(DataQualityIssue(kind="contradiction", detail=f"'{c}' is described as period-related in some notes and not period-related in others."))
    return issues
