"""Render structured skeletons into realistic, varied clinical notes (offline, no LLM).

Several clinician styles, abbreviations, explicit negatives, uncertainty, typos and
copy-forward text are produced so the extractor has to cope with real-world variation.
`llm_writer.py` can optionally rewrite these notes with an LLM for more natural prose.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

from ..schemas import Encounter, Investigation, Medication, PatientRecord, Referral
from ..vocab import SYMPTOMS
from .skeleton import REFERENCE_DATE, EncounterSkeleton, PatientSkeleton, SymptomMention

CLINICIANS = ["Dr A. Rahman", "Dr S. Patel", "Dr L. Okafor", "Dr M. Haddad", "Dr J. Smith", "Dr F. Al Mansoori", "Dr K. Chen", "NP R. Thomas"]

LINK_YES = [" - worse with her periods", ", always around menses", ", cyclical, peaks just before her period", " which flares during menstruation",
            ", typically the week before her period", " (menstrual)", ", comes on with each period", ", worse at the time of the month"]
LINK_NO = [", not related to her periods", ", no cyclical pattern", ", present throughout the month", ", no relation to her cycle"]
DURATIONS = ["for 3 days", "x 2/7", "for the past week", "on and off for months", "for over a year", "since menarche", "for 6 months", "x3/12"]
SEV_TXT = {"severe": ["severe ", "8/10 ", "debilitating "], "moderate": ["moderate ", "6/10 "], "mild": ["mild ", "3/10 "]}
IMPACT = ["; has missed work twice this month", "; missing school because of it", "; off work 2 days each cycle", "; was in bed for 2 days"]
EXAMS = ["Abdo soft, mild tenderness suprapubically, no guarding.", "Abdo soft, tender LIF, no masses.", "Abdomen soft non-tender. PV declined.",
         "Obs normal, abdomen soft, tender in RIF, no rebound.", "Abdo SNT. Speculum: cervix healthy.", "Examination unremarkable."]


def _typo(r: random.Random, text: str, rate=0.03) -> str:
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 5 and r.random() < rate:
            j = r.randrange(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _phrase(r: random.Random, s: SymptomMention) -> str:
    v = SYMPTOMS[s.concept]
    if s.rare_phrase and v["gen_rare"]:
        return r.choice(v["gen_rare"])
    return r.choice(v["gen"])


def symptom_sentence(r: random.Random, s: SymptomMention, lead: bool) -> str:
    ph = _phrase(r, s)
    if s.status == "absent":
        return r.choice([f"Denies {ph}.", f"No {ph}.", f"Nil {ph}.", f"Specifically no {ph}."])
    if s.status == "uncertain":
        return r.choice([f"?{ph}.", f"Possibly some {ph} - unclear.", f"Unsure whether {ph}.", f"Query {ph}, vague history."])
    sev = r.choice(SEV_TXT[s.severity]) if s.severity else ""
    link = r.choice(LINK_YES) if s.menstrual_link == "explicit_yes" else (r.choice(LINK_NO) if s.menstrual_link == "explicit_no" else "")
    imp = r.choice(IMPACT) if s.impact else ""
    if lead:
        body = r.choice([f"c/o {sev}{ph} {r.choice(DURATIONS)}", f"Presents with {sev}{ph} {r.choice(DURATIONS)}", f"{sev}{ph} {r.choice(DURATIONS)}".capitalize(),
                         f"Reports {sev}{ph}"])
    else:
        body = r.choice([f"Also reports {sev}{ph}", f"Associated {sev}{ph}", f"Additionally {sev}{ph}", f"Also {sev}{ph}"])
    return f"{body}{link}{imp}."


def _med_sentence(r, name, cls):
    return r.choice([f"Start {name}.", f"Commenced {name}.", f"Trial of {name}.", f"Rx {name}."])


def _response_sentence(r, resp):
    if resp == "failure":
        return r.choice(["No improvement on current treatment.", "Pain persisting despite the pill.", "Ibuprofen not helping.", "Mefenamic acid ineffective, ongoing despite NSAIDs.",
                         "Not much better with the pill."])
    return r.choice(["Much better on the pill.", "Pain well controlled now.", "Symptoms settled with treatment.", "Good response to NSAIDs."])


ALT_LABEL = {"ibs": "IBS", "uti": "UTI", "pid": "PID", "fibroids": "fibroids", "adenomyosis": "adenomyosis", "primary_dysmenorrhoea": "primary dysmenorrhoea",
             "simple_cyst": "simple ovarian cyst", "msk_back": "mechanical back pain", "interstitial_cystitis": "bladder pain syndrome"}


def render_note(r: random.Random, e: EncounterSkeleton, age: int, clinician: str) -> str:
    if e.unrelated_text:
        return r.choice([f"PC: {e.unrelated_text}. Advice given. Review PRN.", f"{age}F seen with {e.unrelated_text}. Managed conservatively.",
                         f"S: {e.unrelated_text}\nO: obs stable\nA/P: supportive care, safety-net advice."])
    if e.sparse and not e.symptoms:
        return r.choice(["Seen. Discussed. Review if needed.", "Phone consult - see previous.", "DNA follow up.", "Telephone call, pt unavailable."])
    sents: list[str] = []
    present = [s for s in e.symptoms if s.status != "absent"]
    absent = [s for s in e.symptoms if s.status == "absent"]
    for i, s in enumerate(present):
        sents.append(symptom_sentence(r, s, i == 0))
    for s in absent:
        sents.append(symptom_sentence(r, s, False))
    if e.sparse:
        return _typo(r, " ".join(sents[:1]) + " Plan: review.", 0.02)
    if e.response:
        sents.append(_response_sentence(r, e.response))
    exam = r.choice(EXAMS)
    imp = ""
    if e.diagnoses:
        imp = "Likely " + " / ".join(ALT_LABEL.get(d, d) for d in e.diagnoses) + "."
    elif any(s.concept in ("dysmenorrhoea", "pelvic_pain") for s in present) and r.random() < 0.15:
        imp = r.choice(["?endometriosis - consider.", "Differential includes endometriosis.", "Impression: cause of pain unclear."])
    plan: list[str] = []
    for name, cls in e.meds:
        plan.append(_med_sentence(r, name, cls))
    for inv in e.investigations:
        plan.append(r.choice([f"Result: {inv.result_text}", f"Ix: {inv.result_text}", inv.result_text]))
    for to, why in e.referrals:
        plan.append(r.choice([f"Refer to {to} ({why}).", f"Referral to {to} clinic for {why}."]))
    if not plan:
        plan.append(r.choice(["Analgesia advice. Symptom diary.", "Safety-netted. Review 4-6 weeks.", "Continue current plan.", "Review if worse."]))
    style = r.choice(["terse", "narrative", "soap", "ed" if e.setting == "emergency" else "terse"])
    hx = " ".join(sents)
    if style == "terse":
        txt = f"PC/HPC: {hx}\nO/E: {exam}\n" + (f"Imp: {imp}\n" if imp else "") + f"Plan: {' '.join(plan)}"
    elif style == "narrative":
        txt = f"{age} year old woman. {hx} {exam} {imp} {' '.join(plan)}".replace("  ", " ")
    elif style == "soap":
        txt = f"S: {hx}\nO: {exam}\nA: {imp or 'as above'}\nP: {' '.join(plan)}"
    else:
        txt = f"ED attendance. Hx: {hx}\nExam: {exam}\nDx: {imp or 'abdominal pain NYD'}\nD/C: {' '.join(plan)} GP follow-up."
    return _typo(r, txt)


def documented_category(ps: PatientSkeleton) -> str:
    """Constructed-scenario label from what the skeleton documents (NOT the hidden diagnosis).

    A guideline-informed reading of the record: recurrent strong-tier pain plus a cyclical link,
    multiple domains, or persistence despite treatment/normal tests. Used only to report
    'did Athar find the patterns we built', never as clinical truth.
    """
    content = [e for e in ps.encounters if not e.duplicate_of and not (e.sparse and not e.symptoms)]
    relevant = [e for e in content if any(s.status == "present" and s.concept in ("dysmenorrhoea", "pelvic_pain", "dyspareunia", "dyschezia", "urinary") for s in e.symptoms)]
    if len(content) < 2 or (len(relevant) == 1 and len(content) < 3) or (relevant and all(e.sparse for e in relevant)):
        return "INSUFFICIENT INFORMATION"
    if len(relevant) < 2:
        return "NO NOTABLE PATTERN IDENTIFIED"
    strong = {"dysmenorrhoea", "pelvic_pain", "dyspareunia", "dyschezia"}
    domains = {s.concept for e in relevant for s in e.symptoms if s.status == "present" and s.concept in strong | {"urinary"}}
    cyclical = sum(1 for e in relevant for s in e.symptoms if s.status == "present" and s.menstrual_link == "explicit_yes")
    failure = sum(1 for e in relevant if e.response == "failure")
    alt_explains = any(d in ("uti", "pid", "fibroids") for e in content for d in e.diagnoses) and len(domains & strong) <= 1 and cyclical == 0
    endometrioma = any("endometrioma" in i.result_text for e in content for i in e.investigations)
    if alt_explains:
        return "NO NOTABLE PATTERN IDENTIFIED"
    if endometrioma or (len(relevant) >= 3 and (cyclical >= 2 or len(domains & strong) >= 2 or failure >= 1)) or (len(relevant) >= 2 and len(domains & strong) >= 3):
        return "PATTERN WARRANTS CLINICAL REVIEW"
    return "NO NOTABLE PATTERN IDENTIFIED"


ALT_LINK_YES = [" - she notes it is always worse just before and during bleeding", ", catamenial pattern", ", premenstrual exacerbation"]
ALT_DURATIONS = ["for ~2 years", "recurrent since age 15", "worsening over 18 months"]


def render_patient(ps: PatientSkeleton, seed: int, alt_style: bool = False) -> tuple[PatientRecord, dict]:
    """alt_style=True adds extra phrasings so the evaluation set is not worded like the dev set."""
    r = random.Random(seed)
    global LINK_YES, DURATIONS
    base_link, base_dur = LINK_YES, DURATIONS
    if alt_style:
        LINK_YES, DURATIONS = base_link + ALT_LINK_YES, base_dur + ALT_DURATIONS
    age_end = REFERENCE_DATE.year - ps.birth_year
    encounters: list[Encounter] = []
    truth_enc = []
    prev_note = ""
    for e in ps.encounters:
        d = e.date
        age = age_end - max(0, (REFERENCE_DATE - (d or REFERENCE_DATE)).days // 365)
        clin = r.choice(CLINICIANS)
        if e.duplicate_of and encounters:
            note = encounters[-1].note_text
        elif e.copy_forward_of and prev_note:
            note = prev_note + r.choice(["\n[Seen again] As above, no change.", "\nF/U: as previous entry.", "\nReviewed - same as before."])
        else:
            note = render_note(r, e, age, clin)
        prev_note = note
        out_date = d
        if e.date_error == "future":
            out_date = REFERENCE_DATE + timedelta(days=r.randint(200, 900))
        elif e.date_error == "missing":
            out_date = None
        meds = [Medication(name=n, start_date=d) for n, _c in e.meds]
        invs = [Investigation(type=i.type, date=d, result_text=i.result_text) for i in e.investigations]
        refs = [Referral(to=t, date=d, reason=why) for t, why in e.referrals]
        encounters.append(Encounter(encounter_id=e.encounter_id, date=out_date, setting=e.setting, clinician=clin, note_text=note,
                                    coded_diagnoses=[ALT_LABEL.get(x, x) for x in e.diagnoses], medications=meds, investigations=invs, referrals=refs))
        truth_enc.append({
            "encounter_id": e.encounter_id, "true_date": d.isoformat() if d else None, "date_error": e.date_error,
            "duplicate_of_previous": bool(e.duplicate_of), "copy_forward": bool(e.copy_forward_of),
            "symptoms": [vars(s) for s in e.symptoms], "response": e.response,
        })
    LINK_YES, DURATIONS = base_link, base_dur
    rec = PatientRecord(patient_id=ps.patient_id, birth_year=ps.birth_year, family_history_text=ps.family_history, encounters=encounters)
    truth = {
        "patient_id": ps.patient_id, "endometriosis": ps.label, "scenario": ps.scenario, "scenario_name": __import__("athar.generator.skeleton", fromlist=["SCENARIOS"]).SCENARIOS[ps.scenario],
        "subtype": ps.subtype, "expected_category": documented_category(ps), "encounters": truth_enc,
    }
    return rec, truth
