"""Temporal pattern analysis: turn extracted findings into longitudinal features.

Each feature keeps references to the findings that produced it, so every model
contribution can be traced back to encounters and quotes.
Rules: one count per concept per encounter; encounters flagged as duplicates/copy-forward are
excluded; encounters with missing or future dates are excluded from timing features only;
"not documented" never counts as absent.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from ..config import rules
from ..schemas import EvidenceRef, ExtractionResult, Finding, MenstrualLink, PatientRecord, Status
from ..vocab import DOMAINS, PAIN_CONCEPTS, PAIN_TREATMENT_CLASSES, SYMPTOMS

STRONG = {k for k, v in SYMPTOMS.items() if v["tier"] == "strong"}
SUPPORTING = {k for k, v in SYMPTOMS.items() if v["tier"] == "supporting"}
WEAK = {k for k, v in SYMPTOMS.items() if v["tier"] == "weak"}
TRIVIAL_NOTE = ("dna follow up", "telephone call, pt unavailable", "phone consult - see previous", "seen. discussed. review if needed.")

FEATURE_LABELS = {
    "n_relevant_encounters": "Separate visits with relevant pain",
    "pain_span_months": "Months between first and last relevant pain",
    "n_strong_domains": "Different strong symptom domains documented",
    "n_domains": "Different symptom domains documented",
    "n_dysmenorrhoea": "Visits documenting painful periods",
    "n_pelvic_pain": "Visits documenting pelvic pain",
    "n_dyspareunia": "Visits documenting pain with sex",
    "n_dyschezia": "Visits documenting painful bowel movements",
    "n_urinary": "Visits documenting urinary pain/blood",
    "n_subfertility": "Visits documenting difficulty conceiving",
    "n_hmb": "Visits documenting heavy bleeding",
    "n_weak_symptoms": "Visits with only nonspecific symptoms (fatigue, bloating, bowel change, back pain)",
    "n_cyclical_nonmenstrual": "Non-period pain explicitly linked to the cycle",
    "n_explicit_noncyclical": "Pain explicitly documented as not period-related",
    "severe_or_impact": "Visits with severe pain or functional impact",
    "persist_after_treatment": "Pain documented after pain treatment had been started",
    "documented_treatment_failure": "Notes stating treatment is not working",
    "documented_treatment_success": "Notes stating symptoms improved on treatment",
    "persist_after_normal_investigation": "Pain documented after a normal test",
    "suggestive_imaging": "Imaging report suggesting endometrioma / deep disease",
    "n_alt_explanations": "Alternative diagnoses documented",
    "alt_explains_all": "Alternative diagnosis plausibly explains the symptoms",
    "escalation": "Relevant visits concentrated in the last 12 months",
    "gyn_referral": "Gynaecology referral documented",
    "family_history": "Family history of endometriosis (1 yes, 0 no, -1 not recorded)",
    "n_uncertain": "Symptom mentions documented with uncertainty",
    "documentation_completeness": "Share of pain visits documenting cycle link, severity or treatment response",
    "n_content_encounters": "Visits with meaningful documentation",
    "record_span_months": "Length of the record (months)",
    "age": "Age at last visit",
}
MODEL_FEATURES = [k for k in FEATURE_LABELS if k not in ("n_content_encounters",)]


@dataclass
class FeatureSet:
    values: dict[str, float]
    evidence: dict[str, list[EvidenceRef]] = field(default_factory=dict)
    relevant_encounter_ids: list[str] = field(default_factory=list)
    content_encounter_ids: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _ref(f: Finding) -> EvidenceRef:
    return EvidenceRef(encounter_id=f.encounter_id, encounter_date=f.encounter_date, quote=f.evidence_quote, concept=f.concept, status=f.status.value)


def _months(a: dt.date, b: dt.date) -> float:
    return round(abs((b - a).days) / 30.44, 1)


def compute_features(rec: PatientRecord, ex: ExtractionResult, today: dt.date | None = None) -> FeatureSet:
    cfg = rules()["temporal"]
    today = today or dt.date.today()
    skip = {i.encounter_id for i in ex.issues if i.kind == "duplicate_encounter"}
    bad_date = {i.encounter_id for i in ex.issues if i.kind in ("missing_date", "future_date", "date_before_birth")}
    enc_date = {e.encounter_id: e.date for e in rec.encounters}
    F = [f for f in ex.findings if f.encounter_id not in skip]
    sym = [f for f in F if f.category == "symptom"]
    present = [f for f in sym if f.status == Status.present]
    ev: dict[str, list[EvidenceRef]] = {}
    v: dict[str, float] = {}

    def add(name: str, fs: list[Finding]):
        ev.setdefault(name, []).extend(_ref(f) for f in fs)

    # content encounters: any extracted finding, or a substantive note
    content = []
    for e in rec.encounters:
        if e.encounter_id in skip:
            continue
        txt = (e.note_text or "").strip().lower()
        has_find = any(f.encounter_id == e.encounter_id for f in F)
        if has_find or (len(txt) > 15 and not txt.startswith(TRIVIAL_NOTE)):
            content.append(e.encounter_id)
    v["n_content_encounters"] = len(content)

    pain = [f for f in present if f.concept in PAIN_CONCEPTS]
    relevant_ids = sorted({f.encounter_id for f in pain}, key=lambda i: enc_date.get(i) or dt.date.max)
    v["n_relevant_encounters"] = len(relevant_ids)
    add("n_relevant_encounters", pain)

    dated_pain = [f for f in pain if f.encounter_id not in bad_date and f.encounter_date]
    if dated_pain:
        ds = sorted(f.encounter_date for f in dated_pain)
        v["pain_span_months"] = _months(ds[0], ds[-1])
        add("pain_span_months", [min(dated_pain, key=lambda f: f.encounter_date), max(dated_pain, key=lambda f: f.encounter_date)])
    else:
        v["pain_span_months"] = 0.0

    concepts_present = {f.concept for f in present}
    v["n_strong_domains"] = len({DOMAINS[c] for c in concepts_present if c in STRONG and c in DOMAINS})
    v["n_domains"] = len({DOMAINS[c] for c in concepts_present if c in DOMAINS})
    add("n_strong_domains", [f for f in present if f.concept in STRONG])

    def count(concept, name):
        fs = [f for f in present if f.concept == concept]
        v[name] = len({f.encounter_id for f in fs})
        add(name, fs)

    for c, n in [("dysmenorrhoea", "n_dysmenorrhoea"), ("pelvic_pain", "n_pelvic_pain"), ("dyspareunia", "n_dyspareunia"),
                 ("dyschezia", "n_dyschezia"), ("urinary", "n_urinary"), ("subfertility", "n_subfertility"), ("heavy_menstrual_bleeding", "n_hmb")]:
        count(c, n)
    weak_only = [f for f in present if f.concept in WEAK - {"heavy_menstrual_bleeding"}]
    v["n_weak_symptoms"] = len({f.encounter_id for f in weak_only})
    add("n_weak_symptoms", weak_only)

    cyc = [f for f in pain if f.concept != "dysmenorrhoea" and f.menstrual_link == MenstrualLink.explicit_yes]
    v["n_cyclical_nonmenstrual"] = len({f.encounter_id for f in cyc}); add("n_cyclical_nonmenstrual", cyc)
    noncyc = [f for f in pain if f.menstrual_link == MenstrualLink.explicit_no]
    v["n_explicit_noncyclical"] = len({f.encounter_id for f in noncyc}); add("n_explicit_noncyclical", noncyc)
    sev = [f for f in pain if f.severity == "severe" or f.functional_impact]
    v["severe_or_impact"] = len({f.encounter_id for f in sev}); add("severe_or_impact", sev)

    # treatment persistence
    treat = [f for f in F if f.category == "medication" and f.concept in PAIN_TREATMENT_CLASSES and f.encounter_date and f.encounter_id not in bad_date]
    persist = []
    if treat:
        first_treat = min(f.encounter_date for f in treat)
        persist = [f for f in dated_pain if (f.encounter_date - first_treat).days >= cfg["persistence_after_treatment_days"]]
        if persist:
            add("persist_after_treatment", [min(treat, key=lambda f: f.encounter_date)] + persist)
    v["persist_after_treatment"] = float(bool(persist))
    fail = [f for f in F if f.category == "treatment_response" and f.concept == "treatment_failure"]
    succ = [f for f in F if f.category == "treatment_response" and f.concept == "treatment_success"]
    v["documented_treatment_failure"] = len({f.encounter_id for f in fail}); add("documented_treatment_failure", fail)
    v["documented_treatment_success"] = len({f.encounter_id for f in succ}); add("documented_treatment_success", succ)

    inv = [f for f in F if f.category == "investigation"]
    normal = [f for f in inv if f.result in ("normal", "not_elevated") and f.encounter_date and f.encounter_id not in bad_date]
    after_normal = []
    if normal:
        first_norm = min(f.encounter_date for f in normal)
        after_normal = [f for f in dated_pain if (f.encounter_date - first_norm).days >= cfg["persistence_after_normal_investigation_days"]]
        if after_normal:
            add("persist_after_normal_investigation", [min(normal, key=lambda f: f.encounter_date)] + after_normal)
    v["persist_after_normal_investigation"] = float(bool(after_normal))
    sugg = [f for f in inv if f.result == "endometriosis_suggestive"]
    v["suggestive_imaging"] = float(bool(sugg)); add("suggestive_imaging", sugg)

    # alternative explanations
    from ..vocab import ALTERNATIVE_DIAGNOSES
    alt = [f for f in F if f.category == "diagnosis"]
    alt += [f for f in inv if (f.result or "").startswith("abnormal:") and f.result.split(":")[1] in ALTERNATIVE_DIAGNOSES]
    alt_keys = {f.concept if f.category == "diagnosis" else f.result.split(":")[1] for f in alt}
    v["n_alt_explanations"] = len(alt_keys); add("n_alt_explanations", alt)
    explained = set()
    for k in alt_keys:
        explained |= ALTERNATIVE_DIAGNOSES[k]["explains"]
    pain_concepts = {f.concept for f in pain}
    v["alt_explains_all"] = float(bool(pain_concepts) and pain_concepts <= explained and not cyc)

    # escalation: share of relevant visits in last N months (relative to last dated visit)
    if dated_pain:
        last = max(f.encounter_date for f in dated_pain)
        recent = {f.encounter_id for f in dated_pain if _months(f.encounter_date, last) <= cfg["escalation_recent_window_months"]}
        v["escalation"] = round(len(recent) / max(1, len({f.encounter_id for f in dated_pain})), 2)
    else:
        v["escalation"] = 0.0

    gyn = [f for f in F if f.category == "referral" and "gyn" in f.concept.lower()]
    v["gyn_referral"] = float(bool(gyn)); add("gyn_referral", gyn)
    fh = [f for f in F if f.category == "family_history"]
    v["family_history"] = 1.0 if any(f.status == Status.present for f in fh) else (0.0 if fh else -1.0)
    add("family_history", fh)
    unc = [f for f in sym if f.status == Status.uncertain]
    v["n_uncertain"] = len(unc); add("n_uncertain", unc)

    # documentation completeness over pain visits
    if relevant_ids:
        documented = 0
        for eid in relevant_ids:
            fs = [f for f in F if f.encounter_id == eid]
            if any((f.category == "symptom" and (f.menstrual_link != MenstrualLink.not_documented and f.concept != "dysmenorrhoea" or f.severity))
                   or f.category in ("treatment_response", "medication", "investigation") for f in fs):
                documented += 1
        v["documentation_completeness"] = round(documented / len(relevant_ids), 2)
    else:
        v["documentation_completeness"] = 0.0

    dated = [d for e in rec.encounters if e.encounter_id not in skip and e.encounter_id not in bad_date and (d := e.date)]
    v["record_span_months"] = _months(min(dated), max(dated)) if dated else 0.0
    last_date = max(dated) if dated else today
    v["age"] = float(last_date.year - rec.birth_year) if rec.birth_year else -1.0

    # what is missing / uncertain (for the explanation, not the model)
    missing = []
    if pain and not any(f.menstrual_link != MenstrualLink.not_documented for f in pain if f.concept != "dysmenorrhoea") and any(f.concept != "dysmenorrhoea" for f in pain):
        missing.append("Relationship between non-period pain and the menstrual cycle is never documented.")
    if pain and "dyspareunia" not in {f.concept for f in sym}:
        missing.append("Pain during or after sex is neither documented nor explicitly denied.")
    if pain and "dyschezia" not in {f.concept for f in sym}:
        missing.append("Painful bowel movements are neither documented nor explicitly denied.")
    if pain and "urinary" not in {f.concept for f in sym}:
        missing.append("Urinary symptoms are neither documented nor explicitly denied.")
    if v["family_history"] == -1:
        missing.append("Family history of endometriosis is not recorded.")
    if pain and not any(f.severity for f in pain):
        missing.append("Pain severity is never recorded.")
    if treat and not (fail or succ):
        missing.append("Response to treatment is not documented.")
    if unc:
        missing.append(f"{len(unc)} symptom mention(s) are documented with uncertainty.")
    if bad_date:
        missing.append(f"{len(bad_date)} encounter(s) have missing or invalid dates.")
    return FeatureSet(values=v, evidence=ev, relevant_encounter_ids=relevant_ids, content_encounter_ids=content, missing=missing)
