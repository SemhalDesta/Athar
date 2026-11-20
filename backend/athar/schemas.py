"""Data contracts for Athar.

Input records (PatientRecord) are designed so that real EHR exports can replace the
synthetic data: free-text notes plus the structured lists most systems already hold
(prescriptions, investigations, referrals, coded diagnoses).

Ground truth for synthetic patients is stored in a separate file and never passed
to the pipeline.
"""
from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- input


class Medication(BaseModel):
    name: str
    drug_class: Optional[str] = None  # e.g. nsaid, combined_hormonal, opioid
    start_date: Optional[dt.date] = None
    indication: Optional[str] = None


class Investigation(BaseModel):
    type: str  # e.g. pelvic_ultrasound, mri_pelvis, ca125, urine_culture
    date: Optional[dt.date] = None
    result_text: str = ""


class Referral(BaseModel):
    to: str  # e.g. gynaecology, gastroenterology
    date: Optional[dt.date] = None
    reason: str = ""


class Encounter(BaseModel):
    encounter_id: str
    date: Optional[dt.date] = None
    setting: str = "primary_care"  # primary_care, emergency, gynaecology_clinic, ...
    clinician: Optional[str] = None
    note_text: str = ""
    coded_diagnoses: list[str] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    investigations: list[Investigation] = Field(default_factory=list)
    referrals: list[Referral] = Field(default_factory=list)


class PatientRecord(BaseModel):
    patient_id: str
    birth_year: Optional[int] = None
    sex: str = "F"
    family_history_text: Optional[str] = None
    encounters: list[Encounter] = Field(default_factory=list)


# ---------------------------------------------------------------------- extraction


class Status(str, Enum):
    present = "present"          # explicitly documented as present
    absent = "absent"            # explicitly documented as absent ("denies ...")
    uncertain = "uncertain"      # documented with uncertainty ("?", "possible")
    # "not documented" is represented by the absence of a finding, never by a value.


class MenstrualLink(str, Enum):
    explicit_yes = "explicit_yes"   # note explicitly ties it to periods / cycle
    explicit_no = "explicit_no"     # note explicitly says it is NOT related
    not_documented = "not_documented"


class Finding(BaseModel):
    """One clinical finding extracted from one encounter."""

    encounter_id: str
    encounter_date: Optional[dt.date] = None
    category: Literal[
        "symptom", "medication", "investigation", "referral", "diagnosis", "family_history",
        "treatment_response",
    ]
    concept: str  # canonical code, e.g. dysmenorrhoea, nsaid, pelvic_ultrasound
    status: Status = Status.present
    menstrual_link: MenstrualLink = MenstrualLink.not_documented
    severity: Optional[Literal["mild", "moderate", "severe"]] = None
    functional_impact: bool = False  # e.g. missed work/school, bed-bound
    result: Optional[str] = None  # for investigations: normal / abnormal_<detail> / value
    evidence_quote: str = ""  # verbatim text from the source record
    source: Literal["note", "structured"] = "note"
    quote_verified: bool = True  # quote found verbatim in source text
    extractor: str = "lexicon"


class DataQualityIssue(BaseModel):
    encounter_id: Optional[str] = None
    kind: Literal[
        "missing_date", "future_date", "date_before_birth", "duplicate_encounter",
        "contradiction", "empty_note", "unverified_quote",
    ]
    detail: str


class ExtractionResult(BaseModel):
    patient_id: str
    extractor: str
    findings: list[Finding] = Field(default_factory=list)
    issues: list[DataQualityIssue] = Field(default_factory=list)


# ----------------------------------------------------------------------- outputs


class Category(str, Enum):
    review = "PATTERN WARRANTS CLINICAL REVIEW"
    insufficient = "INSUFFICIENT INFORMATION"
    no_pattern = "NO NOTABLE PATTERN IDENTIFIED"


class EvidenceRef(BaseModel):
    encounter_id: str
    encounter_date: Optional[dt.date] = None
    quote: str = ""
    concept: str = ""
    status: str = ""
    menstrual_link: Optional[str] = None


class Contributor(BaseModel):
    feature: str
    label: str
    value: float
    shap: float  # contribution to model score (log-odds space)
    direction: Literal["raises", "lowers"]
    evidence: list[EvidenceRef] = Field(default_factory=list)


class PatternStatement(BaseModel):
    kind: Literal["recurrence", "trend", "severity", "co_occurrence", "cyclical", "persistence"]
    text: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class YearCount(BaseModel):
    year: int
    pain_visits: int
    other_visits: int


class Assessment(BaseModel):
    patient_id: str
    category: Category
    model_score: Optional[float] = None  # illustrative model score, NOT a probability
    priority: int = 3  # 1 = review first
    reasons: list[str] = Field(default_factory=list)  # plain-language decision trail
    guardrails_applied: list[str] = Field(default_factory=list)
    contributors: list[Contributor] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)
    relevant_encounters: int = 0
    timeline: list[EvidenceRef] = Field(default_factory=list)
    treatments_and_investigations: list[EvidenceRef] = Field(default_factory=list)
    alternative_explanations: list[EvidenceRef] = Field(default_factory=list)
    missing_or_uncertain: list[str] = Field(default_factory=list)
    temporal_patterns: list[PatternStatement] = Field(default_factory=list)
    visits_by_year: list[YearCount] = Field(default_factory=list)
    data_quality: list[DataQualityIssue] = Field(default_factory=list)
    summary: str = ""
    summary_source: Literal["llm", "template"] = "template"
    extractor: str = "lexicon"
    disclaimer: str = (
        "Screening-support output. Not a diagnosis and not a clinically validated "
        "probability. The model was developed on synthetic data."
    )
