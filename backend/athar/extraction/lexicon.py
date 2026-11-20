"""Keyword + rule baseline extractor (offline, deterministic).

Used for tests, offline demos and as the comparison baseline for the LLM extractor.
NegEx-style: negation/uncertainty cues are searched in the clause *before* the term,
after removing pseudo-negations such as "no improvement".
"""
from __future__ import annotations

import re

from ..schemas import ExtractionResult, Finding, MenstrualLink, PatientRecord, Status
from ..vocab import (
    ALTERNATIVE_DIAGNOSES, IMPACT_CUES, MENSTRUAL_LINK_NO, MENSTRUAL_LINK_YES, MILD_CUES, MODERATE_CUES,
    NEGATION_CUES, PSEUDO_NEGATION, SEVERE_CUES, SYMPTOMS, TREATMENT_FAILURE, TREATMENT_SUCCESS, UNCERTAINTY_CUES,
)
from .common import (
    ALT_RX, cross_encounter_contradictions, merge_encounter_findings, quality_checks, rx, structured_findings,
)

SYM_RX = {k: rx(v["patterns"]) for k, v in SYMPTOMS.items()}
NEG_RX, UNC_RX, PSEUDO_RX = rx(NEGATION_CUES), rx(UNCERTAINTY_CUES), rx(PSEUDO_NEGATION)
LINK_YES_RX, LINK_NO_RX = rx(MENSTRUAL_LINK_YES), rx(MENSTRUAL_LINK_NO)
SEV_RX, MOD_RX, MILD_RX, IMPACT_RX = rx(SEVERE_CUES), rx(MODERATE_CUES), rx(MILD_CUES), rx(IMPACT_CUES)
FAIL_RX, SUCC_RX = rx(TREATMENT_FAILURE), rx(TREATMENT_SUCCESS)
SENT_SPLIT = re.compile(r"(?<=[.;!])\s+|\n+")
CLAUSE_BREAK = re.compile(r"\b(?:but|however|although)\b|,", re.I)
WINDOW_WORDS = 6


def _status(sentence: str, start: int) -> Status:
    before = sentence[:start]
    # restrict to the current clause and a short word window
    parts = CLAUSE_BREAK.split(before)
    clause = parts[-1] if parts else before
    clause = " ".join(clause.split()[-WINDOW_WORDS:]) if clause.split() else clause
    if before.rstrip().endswith("?") or sentence[start - 1 : start] == "?":
        return Status.uncertain
    clause_clean = PSEUDO_RX.sub(" ", clause)
    if UNC_RX.search(clause_clean) or re.search(r"\b(?:not sure|unsure)\b", clause, re.I):
        return Status.uncertain
    if NEG_RX.search(clause_clean):
        return Status.absent
    return Status.present


def _severity(s: str):
    if SEV_RX.search(s):
        return "severe"
    if MOD_RX.search(s):
        return "moderate"
    if MILD_RX.search(s):
        return "mild"
    return None


def extract_note(encounter_id: str, encounter_date, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for sent in SENT_SPLIT.split(text or ""):
        s = sent.strip()
        if not s:
            continue
        for concept, r in SYM_RX.items():
            for m in r.finditer(s):
                status = _status(s, m.start())
                link = MenstrualLink.not_documented
                after = s[m.end():]
                if concept == "dysmenorrhoea":
                    link = MenstrualLink.explicit_yes  # menstrual by definition
                elif LINK_NO_RX.search(s):
                    link = MenstrualLink.explicit_no
                elif LINK_YES_RX.search(after) or LINK_YES_RX.search(s[: m.start()][-25:]):
                    link = MenstrualLink.explicit_yes
                findings.append(Finding(
                    encounter_id=encounter_id, encounter_date=encounter_date, category="symptom", concept=concept,
                    status=status, menstrual_link=link, severity=_severity(s) if status == Status.present else None,
                    functional_impact=bool(IMPACT_RX.search(s)) and status == Status.present, evidence_quote=s, extractor="lexicon",
                ))
                break  # one hit per concept per sentence
        if FAIL_RX.search(s):
            findings.append(Finding(encounter_id=encounter_id, encounter_date=encounter_date, category="treatment_response",
                                    concept="treatment_failure", evidence_quote=s, extractor="lexicon"))
        elif SUCC_RX.search(s):
            findings.append(Finding(encounter_id=encounter_id, encounter_date=encounter_date, category="treatment_response",
                                    concept="treatment_success", evidence_quote=s, extractor="lexicon"))
        for key, r in ALT_RX.items():
            m = r.search(s)
            if m and not re.search(r"\?|differential|consider|rule out|exclude", s[: m.start()], re.I):
                if NEG_RX.search(PSEUDO_RX.sub(" ", s[: m.start()][-30:])):
                    continue
                findings.append(Finding(encounter_id=encounter_id, encounter_date=encounter_date, category="diagnosis",
                                        concept=key, evidence_quote=s, extractor="lexicon"))
    return findings


class LexiconExtractor:
    name = "lexicon"

    def extract(self, rec: PatientRecord) -> ExtractionResult:
        issues, skip = quality_checks(rec)
        findings = structured_findings(rec, skip)
        for e in rec.encounters:
            if e.encounter_id in skip:
                continue
            findings.extend(extract_note(e.encounter_id, e.date, e.note_text))
        findings, merge_issues = merge_encounter_findings(findings)
        issues += merge_issues + cross_encounter_contradictions(findings)
        return ExtractionResult(patient_id=rec.patient_id, extractor=self.name, findings=findings, issues=issues)
