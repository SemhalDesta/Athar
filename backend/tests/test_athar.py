"""Automated tests for the Athar pipeline (offline: keyword extractor + rules, no LLM needed)."""
from __future__ import annotations

import datetime as dt

import pytest

from athar.engine.decide import decide
from athar.extraction.lexicon import LexiconExtractor
from athar.extraction.llm_extractor import _verified
from athar.schemas import Category, Encounter, Investigation, Medication, MenstrualLink, PatientRecord, Status
from athar.temporal.features import compute_features

EX = LexiconExtractor()
HIGH, LOW, THR = 0.95, 0.05, 0.5


def rec(*notes, birth_year=1995, inv=None, meds=None, dates=None, fh=None) -> PatientRecord:
    encs = []
    for i, n in enumerate(notes):
        d = dates[i] if dates else dt.date(2023, 1, 1) + dt.timedelta(days=60 * i)
        e = Encounter(encounter_id=f"E{i+1}", date=d, note_text=n)
        if inv and i in inv:
            e.investigations = [inv[i]]
        if meds and i in meds:
            e.medications = [meds[i]]
        encs.append(e)
    return PatientRecord(patient_id="T1", birth_year=birth_year, encounters=encs, family_history_text=fh)


def run(r: PatientRecord, score=HIGH):
    ex = EX.extract(r)
    fs = compute_features(r, ex)
    return ex, fs, decide(fs, score, THR)


def sym(ex, concept, eid=None):
    return [f for f in ex.findings if f.category == "symptom" and f.concept == concept and (eid is None or f.encounter_id == eid)]


# 1. repeated symptoms -------------------------------------------------------------
def test_repeated_symptoms_counted_per_encounter():
    r = rec("Severe period pain, missing school.", "Dysmenorrhoea again, 8/10.", "c/o painful periods, ibuprofen not helping.")
    ex, fs, d = run(r)
    assert fs.values["n_dysmenorrhoea"] == 3
    assert fs.values["n_relevant_encounters"] == 3
    assert d.category == Category.review


# 2. menstrual association ----------------------------------------------------------
@pytest.mark.parametrize("note,expected", [
    ("Pelvic pain - worse with her periods.", MenstrualLink.explicit_yes),
    ("Pelvic pain, not related to her periods.", MenstrualLink.explicit_no),
    ("Pelvic pain for 3 days.", MenstrualLink.not_documented),
    ("Pain on opening bowels, cyclical.", MenstrualLink.explicit_yes),
])
def test_menstrual_link(note, expected):
    ex = EX.extract(rec(note))
    f = [x for x in ex.findings if x.category == "symptom"][0]
    assert f.menstrual_link == expected


def test_cyclical_link_feature_ignores_dysmenorrhoea_itself():
    r = rec("Period pain.", "Period pain again.", "Dyschezia around menses.")
    _, fs, _ = run(r)
    assert fs.values["n_cyclical_nonmenstrual"] == 1


# 3. isolated symptom ---------------------------------------------------------------
def test_isolated_symptom_never_flags_even_with_high_score():
    r = rec("Sore throat for 4 days.", "Contraception review.", "Pelvic pain for 2 days.", "Asthma review, well controlled.")
    _, fs, d = run(r, score=HIGH)
    assert d.category == Category.no_pattern
    assert any("single visit" in g for g in d.guardrails)


def test_single_encounter_record_is_insufficient():
    _, _, d = run(rec("Severe pelvic pain with periods."), score=HIGH)
    assert d.category == Category.insufficient


# 4. missing information ------------------------------------------------------------
def test_not_documented_is_not_absent():
    r = rec("Period pain, severe.", "Period pain again.")
    ex, fs, _ = run(r)
    assert sym(ex, "dyspareunia") == []  # nothing invented
    assert any("Pain during or after sex is neither documented nor explicitly denied" in m for m in fs.missing)
    assert fs.values["family_history"] == -1


def test_explicit_negative_is_absent():
    ex = EX.extract(rec("Pelvic pain. Denies dyspareunia. No dysuria or dyschezia."))
    assert sym(ex, "dyspareunia")[0].status == Status.absent
    assert sym(ex, "urinary")[0].status == Status.absent
    assert sym(ex, "dyschezia")[0].status == Status.absent


def test_pseudo_negation_does_not_negate():
    ex = EX.extract(rec("No improvement in her pelvic pain on the pill."))
    assert sym(ex, "pelvic_pain")[0].status == Status.present
    assert any(f.concept == "treatment_failure" for f in ex.findings)


# 5. contradictory records ----------------------------------------------------------
def test_contradiction_across_encounters_is_reported_not_resolved():
    r = rec("Pelvic pain. Denies dyspareunia.", "Pelvic pain. Deep dyspareunia now.")
    ex, _, _ = run(r)
    assert {f.status for f in sym(ex, "dyspareunia")} == {Status.absent, Status.present}
    assert any(i.kind == "contradiction" for i in ex.issues)


def test_contradiction_within_encounter_counts_once():
    ex = EX.extract(rec("Painful sex. Later in consult: no dyspareunia."))
    fs = sym(ex, "dyspareunia")
    assert len(fs) == 1 and fs[0].status == Status.present
    assert any(i.kind == "contradiction" for i in ex.issues)


# 6. alternative explanations -------------------------------------------------------
def test_alternative_explanation_blocks_flag_when_it_explains_everything():
    pos = Investigation(type="urine_culture", result_text="MSU: E. coli >10^5")
    r = rec("Dysuria. Likely UTI.", "Dysuria again. UTI.", "Pain passing urine, UTI.", inv={0: pos, 1: pos, 2: pos})
    _, fs, d = run(r, score=HIGH)
    assert fs.values["alt_explains_all"] == 1
    assert d.category == Category.no_pattern
    assert any("alternative diagnosis" in g for g in d.guardrails)


def test_ibs_label_does_not_hide_dyschezia():
    r = rec("Bloating. Likely IBS.", "Pain on opening bowels during her periods.", "Dyschezia with menses again.")
    _, fs, d = run(r, score=HIGH)
    assert fs.values["alt_explains_all"] == 0
    assert d.category == Category.review


# 7. normal imaging -----------------------------------------------------------------
def test_normal_imaging_does_not_lower_and_adds_persistence():
    base = ("Pelvic pain with periods.", "Pelvic pain with periods, severe.", "Ongoing pelvic pain worse with periods.")
    us = Investigation(type="pelvic_ultrasound", result_text="TVUS: normal uterus and ovaries.")
    _, fs_a, d_a = run(rec(*base), score=HIGH)
    _, fs_b, d_b = run(rec(*base, inv={1: us}), score=HIGH)
    assert d_a.category == d_b.category == Category.review
    assert fs_b.values["persist_after_normal_investigation"] == 1


def test_suggestive_imaging_rule_override():
    us = Investigation(type="pelvic_ultrasound", result_text="Right ovarian endometrioma 2.8cm.")
    _, _, d = run(rec("Pelvic pain.", "Back pain review.", inv={0: us}), score=LOW)
    assert d.category == Category.review and d.priority == 1


# 8. duplicate encounters -----------------------------------------------------------
def test_exact_duplicate_counted_once():
    day = dt.date(2024, 1, 1)
    r = rec("Severe period pain.", "Severe period pain.", "Period pain again.", dates=[day, day, dt.date(2024, 3, 1)])
    ex, fs, _ = run(r)
    assert fs.values["n_dysmenorrhoea"] == 2
    assert any(i.kind == "duplicate_encounter" for i in ex.issues)


def test_copy_forward_not_counted_as_new_occurrence():
    first = "PC: severe pelvic pain with her periods, off work 2 days. Plan: naproxen."
    r = rec(first, first + "\nF/U: as previous entry.", "Pelvic pain again.")
    _, fs, _ = run(r)
    assert fs.values["n_relevant_encounters"] == 2


def test_symptom_repeated_in_one_note_counted_once():
    ex = EX.extract(rec("Pelvic pain. Pelvic pain radiating to back. Lower abdominal pain."))
    assert len(sym(ex, "pelvic_pain")) == 1


# 9. incorrect dates ----------------------------------------------------------------
def test_future_and_missing_dates_flagged_and_excluded_from_timing():
    r = rec("Pelvic pain.", "Pelvic pain.", "Pelvic pain.", dates=[dt.date(2023, 1, 1), dt.date(2099, 1, 1), None])
    ex, fs, _ = run(r)
    kinds = {i.kind for i in ex.issues}
    assert {"future_date", "missing_date"} <= kinds
    assert fs.values["pain_span_months"] == 0.0  # only one valid date
    assert fs.values["n_relevant_encounters"] == 3  # still counted as encounters


def test_date_before_birth():
    r = rec("Pelvic pain.", "Pelvic pain.", birth_year=2000, dates=[dt.date(1990, 1, 1), dt.date(2024, 1, 1)])
    ex, _, _ = run(r)
    assert any(i.kind == "date_before_birth" for i in ex.issues)


# 10. uncertain documentation -------------------------------------------------------
@pytest.mark.parametrize("note", ["?dyspareunia.", "Possibly some dyspareunia - unclear.", "Unsure whether dyspareunia."])
def test_uncertain_status(note):
    ex = EX.extract(rec(note))
    assert sym(ex, "dyspareunia")[0].status == Status.uncertain


def test_uncertain_not_counted_as_present():
    r = rec("Period pain.", "?dyschezia.", "Period pain.")
    _, fs, _ = run(r)
    assert fs.values["n_dyschezia"] == 0 and fs.values["n_uncertain"] == 1


# guardrails & safety ---------------------------------------------------------------
def test_weak_symptoms_alone_never_flag():
    _, _, d = run(rec("Fatigue.", "Bloating and fatigue.", "Heavy periods."), score=HIGH)
    assert d.category != Category.review


def test_medication_and_visits_alone_are_not_evidence():
    meds = {i: Medication(name="Microgynon") for i in range(4)}
    _, fs, d = run(rec("Contraception review.", "Pill check.", "Pill check.", "Pill check.", meds=meds), score=HIGH)
    assert fs.values["n_relevant_encounters"] == 0 and d.category != Category.review


def test_llm_quote_verification_rejects_hallucinations():
    note = "c/o painful perods for 6 months, worse with her cycle."
    assert _verified("painful perods for 6 months", note)
    assert not _verified("deep dyspareunia", note)


def test_family_history():
    _, fs, _ = run(rec("Pelvic pain.", "Pelvic pain.", fh="Mother has endometriosis."))
    assert fs.values["family_history"] == 1
