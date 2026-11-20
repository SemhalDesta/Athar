"""Label-first synthetic patient generator (structured skeleton).

Order of generation (see Clinical Evidence Document, "Synthetic data generation"):
  1. hidden label (endometriosis yes/no), ~30% positive
  2. scenario conditioned on the label (10 brief scenarios + routine care)
  3. a structured encounter timeline drawn with evidence-shaped probabilities
  4. note text is rendered later (notes.py) or rewritten by an LLM (llm_writer.py)

The skeleton is the ground truth: it records which findings each note really contains.
Nothing here is a validated clinical model; probabilities are chosen to mirror the
direction and rough size of published associations while keeping heavy overlap.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, timedelta

REFERENCE_DATE = date(2026, 10, 1)

SCENARIOS = {
    1: "cyclical_pelvic_pain",
    2: "dysmenorrhoea_only",
    3: "gi_non_menstrual",
    4: "pelvic_pain_no_menstrual_link",
    5: "overlapping_conditions",
    6: "repeated_inconclusive",
    7: "insufficient_records",
    8: "varied_terminology",
    9: "isolated_symptom",
    10: "inconsistent_documentation",
    11: "routine_unrelated",
}
POS_WEIGHTS = {1: .22, 2: .07, 3: .08, 4: .08, 5: .05, 6: .12, 7: .06, 8: .10, 9: .04, 10: .10, 11: .08}
NEG_WEIGHTS = {1: .07, 2: .18, 3: .12, 4: .08, 5: .15, 6: .08, 7: .07, 8: .03, 9: .07, 10: .04, 11: .11}

UNRELATED_REASONS = [
    ("urti", "sore throat and cough for 4 days"), ("contraception", "contraception review"),
    ("skin", "itchy rash on forearms"), ("mood", "low mood and poor sleep"), ("vaccine", "travel vaccinations"),
    ("headache", "tension-type headaches"), ("ankle", "twisted ankle playing football"), ("smear", "cervical screening"),
    ("asthma", "asthma review"), ("acne", "acne"), ("eye", "red itchy eye"), ("ear", "ear pain"),
]


@dataclass
class SymptomMention:
    concept: str
    status: str = "present"           # present / absent / uncertain
    menstrual_link: str = "not_documented"  # explicit_yes / explicit_no / not_documented
    severity: str | None = None
    impact: bool = False
    rare_phrase: bool = False         # use wording outside the keyword lexicon


@dataclass
class EventInv:
    type: str
    result_text: str
    normal: bool


@dataclass
class EncounterSkeleton:
    encounter_id: str
    date: date | None
    setting: str
    reason: str  # pain / unrelated:<x> / followup / dx:<alt>
    symptoms: list[SymptomMention] = field(default_factory=list)
    meds: list[tuple[str, str]] = field(default_factory=list)   # (name, class)
    investigations: list[EventInv] = field(default_factory=list)
    referrals: list[tuple[str, str]] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    response: str | None = None  # failure / success
    unrelated_text: str | None = None
    copy_forward_of: str | None = None
    duplicate_of: str | None = None
    date_error: str | None = None  # future / missing
    sparse: bool = False


@dataclass
class PatientSkeleton:
    patient_id: str
    label: bool
    scenario: int
    subtype: str
    birth_year: int
    family_history: str | None
    encounters: list[EncounterSkeleton]
    expected_category: str = ""
    notes: list[str] = field(default_factory=list)


class Gen:
    def __init__(self, seed: int, id_prefix: str):
        self.r = random.Random(seed)
        self.id_prefix = id_prefix
        self.n = 0

    # ------------------------------------------------------------------ helpers
    def p(self, prob: float) -> bool:
        return self.r.random() < prob

    def pick(self, seq):
        return self.r.choice(seq)

    def wpick(self, weights: dict):
        keys = list(weights)
        return self.r.choices(keys, weights=[weights[k] for k in keys])[0]

    def dates(self, start: date, end: date, n: int, escalate: bool) -> list[date]:
        span = max((end - start).days, 30)
        out = []
        for _ in range(n):
            u = self.r.random()
            if escalate:
                u = u ** 0.55  # cluster toward the end (Danish registry: rising use before diagnosis)
            out.append(start + timedelta(days=int(u * span)))
        return sorted(out)

    # ----------------------------------------------------------------- patient
    def patient(self, label: bool | None = None, scenario: int | None = None) -> PatientSkeleton:
        self.n += 1
        pid = f"{self.id_prefix}-{self.n:04d}"
        if label is None:
            label = self.p(0.30)
        if scenario is None:
            scenario = self.wpick(POS_WEIGHTS if label else NEG_WEIGHTS)
        age_end = self.r.randint(16, 45)
        birth_year = REFERENCE_DATE.year - age_end
        fh = None
        if self.p(0.12 if label else 0.04):
            fh = self.pick(["Mother has endometriosis.", "Sister diagnosed with endometriosis.", "FH endometriosis (aunt)."])
        elif self.p(0.15):
            fh = self.pick(["No significant family history.", "FH: type 2 diabetes (father).", "Mother had fibroids."])
        builder = getattr(self, f"_s{scenario}")
        encs, subtype = builder(label, age_end)
        pain_encs = [e for e in encs if e.reason == "pain" and e.symptoms]
        if label and pain_encs and self.p(0.22):
            # misdiagnosis / co-diagnosis before endometriosis is recognised (Seaman 2008: IBS OR 3.5, PID OR 5.9)
            self.pick(pain_encs).diagnoses.append(self.pick(["ibs", "pid", "primary_dysmenorrhoea", "uti"]))
        if label and pain_encs and self.p(0.3):
            # under-documented course: busy clinicians record less detail (Mazza 2025: 1/3 have no symptom recorded)
            for e in pain_encs:
                e.symptoms = [x for i, x in enumerate(e.symptoms) if i == 0 or not self.p(0.6)]
                for x in e.symptoms:
                    if self.p(0.6):
                        x.menstrual_link = "not_documented" if x.concept != "dysmenorrhoea" else x.menstrual_link
                if self.p(0.6):
                    e.response = None
        if not label and pain_encs and self.p(0.25):
            e = self.pick(pain_encs)  # strong-domain symptoms also occur in women without endometriosis
            e.symptoms.append(SymptomMention(self.pick(["dyspareunia", "dyschezia", "urinary"]), "present",
                                             "explicit_yes" if self.p(0.3) else "not_documented"))
        if not label and self.p(0.35):
            for e in encs:  # many non-endometriosis pain presentations never receive a firm alternative label
                e.diagnoses = []
        ps = PatientSkeleton(pid, label, scenario, subtype, birth_year, fh, encs)
        self._inject_quality_issues(ps)
        self._number(ps)
        return ps

    def _number(self, ps: PatientSkeleton):
        for i, e in enumerate(ps.encounters, 1):
            if not e.encounter_id:
                e.encounter_id = f"{ps.patient_id}-E{i:02d}"

    def _span(self, years_lo=1.0, years_hi=8.0):
        years = self.r.uniform(years_lo, years_hi)
        end = REFERENCE_DATE - timedelta(days=self.r.randint(0, 90))
        start = end - timedelta(days=int(years * 365))
        return start, end

    def _unrelated(self, d: date) -> EncounterSkeleton:
        code, text = self.pick(UNRELATED_REASONS)
        return EncounterSkeleton("", d, "primary_care", f"unrelated:{code}", unrelated_text=text)

    def _pad_unrelated(self, encs, start, end, k):
        for d in self.dates(start, end, k, False):
            encs.append(self._unrelated(d))
        encs.sort(key=lambda e: e.date or REFERENCE_DATE)

    # ------------------------------------------------------- endometriosis-like
    def _pain_course(
        self, start, end, n_pain, *, link_p, domains: dict, rare_p=0.05, escalate=True,
        treat=True, us_result="normal_mostly", response="failure", link_no_p=0.0,
        concept_main="mixed", severe_p=0.4, impact_p=0.3, sexually_active=True,
    ) -> list[EncounterSkeleton]:
        """A sequence of pain encounters following the usual treatment ladder."""
        encs: list[EncounterSkeleton] = []
        ds = self.dates(start, end, n_pain, escalate)
        ladder_step = 0
        for i, d in enumerate(ds):
            e = EncounterSkeleton("", d, self.pick(["primary_care"] * 6 + ["emergency"]), "pain")
            if concept_main == "dysmenorrhoea":
                main = "dysmenorrhoea"
            elif concept_main == "pelvic_pain":
                main = "pelvic_pain"
            else:
                main = self.pick(["dysmenorrhoea", "pelvic_pain", "dysmenorrhoea"])
            link = "not_documented"
            if self.p(link_p):
                link = "explicit_yes"
            elif self.p(link_no_p):
                link = "explicit_no"
            if main == "dysmenorrhoea" and link == "explicit_no":
                link = "not_documented"
            sev = self.pick(["severe"] * 2 + ["moderate"]) if self.p(severe_p) else (self.pick(["moderate", "mild", None]))
            e.symptoms.append(SymptomMention(main, "present", link, sev, self.p(impact_p), self.p(rare_p)))
            for concept, prob in domains.items():
                if concept == "dyspareunia" and not sexually_active:
                    continue
                if self.p(prob):
                    clink = "explicit_yes" if (concept in ("dyschezia", "urinary") and self.p(link_p * 0.8)) else "not_documented"
                    e.symptoms.append(SymptomMention(concept, "present", clink, None, False, self.p(rare_p)))
                elif concept in ("dyspareunia", "dyschezia") and self.p(0.15):
                    e.symptoms.append(SymptomMention(concept, "absent"))  # explicit negative
            if treat:
                if ladder_step == 0 and i < n_pain - 1:
                    e.meds.append((self.pick(["mefenamic acid", "naproxen", "ibuprofen"]), "nsaid")); ladder_step = 1
                elif ladder_step == 1 and i < n_pain - 1:
                    e.meds.append((self.pick(["Microgynon", "Rigevidon", "combined pill"]), "combined_hormonal")); ladder_step = 2
                    if i > 0:
                        e.response = response
                elif ladder_step == 2:
                    e.response = response
                    if self.p(0.5):
                        inv = self._ultrasound(us_result)
                        e.investigations.append(inv); ladder_step = 3
                    elif self.p(0.25):
                        e.meds.append((self.pick(["desogestrel", "Mirena", "norethisterone"]), "progestogen"))
                elif ladder_step == 3:
                    e.response = response
                    if self.p(0.4):
                        e.referrals.append(("gynaecology", "ongoing symptoms despite hormonal treatment")); ladder_step = 4
                    elif self.p(0.2):
                        e.investigations.append(EventInv("ca125", f"CA125 {self.r.randint(8, 30)} U/ml", True))
                    elif self.p(0.2):
                        e.meds.append(("co-codamol", "opioid"))
            encs.append(e)
        return encs

    def _ultrasound(self, mode: str) -> EventInv:
        if mode == "normal_mostly":
            roll = self.r.random()
            if roll < 0.72:
                return EventInv("pelvic_ultrasound", self.pick(["TVUS: normal uterus and ovaries.", "Pelvic USS NAD.", "Transvaginal ultrasound unremarkable, no adnexal mass."]), True)
            if roll < 0.88:
                return EventInv("pelvic_ultrasound", self.pick(["TVUS: 3.4cm left ovarian cyst with ground-glass echoes, appearances suggest endometrioma.", "Right ovarian endometrioma 2.8cm."]), False)
            return EventInv("pelvic_ultrasound", "TVUS: 2.5cm simple left ovarian cyst.", False)
        if mode == "fibroid":
            return EventInv("pelvic_ultrasound", "TVUS: bulky uterus with 3cm intramural fibroid.", False)
        if mode == "adenomyosis":
            return EventInv("pelvic_ultrasound", "TVUS: globular uterus, features suggestive of adenomyosis.", False)
        if mode == "simple_cyst":
            return EventInv("pelvic_ultrasound", "TVUS: 3cm simple right ovarian cyst, likely functional.", False)
        return EventInv("pelvic_ultrasound", "Pelvic ultrasound normal.", True)

    # ------------------------------------------------------------- scenarios
    def _s1(self, label, age):  # recurring pelvic pain with documented menstrual association
        start, end = self._span(1.5, 8)
        sa = age >= 18
        if label:
            encs = self._pain_course(start, end, self.r.randint(3, 10), link_p=0.5,
                                     domains={"dyspareunia": .25, "dyschezia": .2, "urinary": .1, "heavy_menstrual_bleeding": .25, "fatigue": .25, "subfertility": .1 if age > 25 else 0},
                                     sexually_active=sa)
            sub = "typical_cyclical"
        else:  # mimic: adenomyosis-like / chronic pain without endometriosis
            encs = self._pain_course(start, end, self.r.randint(3, 9), link_p=0.45,
                                     domains={"heavy_menstrual_bleeding": .45, "dyspareunia": .18, "dyschezia": .08, "urinary": .1, "fatigue": .25, "back_pain": .2},
                                     us_result=self.pick(["adenomyosis", "normal_mostly", "fibroid"]), sexually_active=sa)
            sub = "mimic_cyclical"
        self._pad_unrelated(encs, start, end, self.r.randint(0, 4))
        return encs, sub

    def _s2(self, label, age):  # repeated painful periods, nothing else
        start, end = self._span(1, 6)
        resp = "failure" if (label and self.p(0.55)) else ("success" if self.p(0.6) else "failure")
        encs = self._pain_course(start, end, self.r.randint(2, 5), link_p=0.0, domains={}, concept_main="dysmenorrhoea",
                                 response=resp, severe_p=0.5 if label else 0.3, impact_p=0.35 if label else 0.2, escalate=label)
        if not label and self.p(0.4):
            encs[-1].diagnoses.append("primary_dysmenorrhoea")
        self._pad_unrelated(encs, start, end, self.r.randint(0, 4))
        return encs, "dysmenorrhoea_persistent" if resp == "failure" else "primary_dysmenorrhoea_like"

    def _s3(self, label, age):  # recurring GI symptoms not linked to periods
        start, end = self._span(1, 7)
        encs = []
        for d in self.dates(start, end, self.r.randint(3, 7), label):
            e = EncounterSkeleton("", d, "primary_care", "pain")
            e.symptoms.append(SymptomMention("bowel_change", "present", "explicit_no" if self.p(0.5) else "not_documented"))
            if self.p(0.6):
                e.symptoms.append(SymptomMention("bloating"))
            if self.p(0.5):
                e.symptoms.append(SymptomMention("pelvic_pain", "present", "explicit_no" if self.p(0.4) else "not_documented", self.pick(["mild", "moderate", None])))
            if label and self.p(0.55):  # bowel-dominant endometriosis mislabelled as IBS
                e.symptoms.append(SymptomMention("dyschezia", "present", "explicit_yes" if self.p(0.6) else "not_documented"))
                if self.p(0.4):
                    e.symptoms.append(SymptomMention("dysmenorrhoea", "present", "not_documented", self.pick(["moderate", "severe"])))
            encs.append(e)
        encs[min(1, len(encs) - 1)].diagnoses.append("ibs")
        encs[min(1, len(encs) - 1)].meds.append(("mebeverine", "antispasmodic"))
        if self.p(0.4):
            encs[-1].investigations.append(EventInv("colonoscopy", "Colonoscopy: normal to terminal ileum.", True))
        self._pad_unrelated(encs, start, end, self.r.randint(0, 3))
        return encs, "bowel_dominant_mislabelled_ibs" if label else "ibs"

    def _s4(self, label, age):  # pelvic pain, no documented menstrual association
        start, end = self._span(1, 7)
        if label:
            encs = self._pain_course(start, end, self.r.randint(3, 8), link_p=0.0, domains={"dyspareunia": .25, "dyschezia": .15, "fatigue": .2},
                                     concept_main="pelvic_pain", sexually_active=age >= 18)
            sub = "noncyclical_documentation"
        else:
            encs = self._pain_course(start, end, self.r.randint(2, 5), link_p=0.0, link_no_p=0.4, domains={"back_pain": .4, "dyspareunia": .1},
                                     concept_main="pelvic_pain", us_result=self.pick(["simple_cyst", "normal_mostly"]), response=self.pick(["success", "failure"]), escalate=False)
            if self.p(0.5):
                encs[-1].diagnoses.append(self.pick(["msk_back", "simple_cyst", "pid"]))
            sub = "other_pelvic_pain"
        self._pad_unrelated(encs, start, end, self.r.randint(0, 3))
        return encs, sub

    def _s5(self, label, age):  # overlapping conditions
        start, end = self._span(1, 7)
        alt = self.pick(["fibroids", "adenomyosis", "uti", "pid", "interstitial_cystitis"])
        encs: list[EncounterSkeleton] = []
        n = self.r.randint(3, 7)
        for d in self.dates(start, end, n, label):
            e = EncounterSkeleton("", d, "primary_care", "pain")
            if alt in ("fibroids", "adenomyosis"):
                e.symptoms.append(SymptomMention("heavy_menstrual_bleeding"))
                if self.p(0.6):
                    e.symptoms.append(SymptomMention("dysmenorrhoea", "present", "not_documented", self.pick(["moderate", "severe"])))
            elif alt in ("uti", "interstitial_cystitis"):
                e.symptoms.append(SymptomMention("urinary", "present", "explicit_yes" if (label and self.p(0.4)) else "not_documented"))
                if self.p(0.4):
                    e.symptoms.append(SymptomMention("pelvic_pain", "present", "not_documented", "mild"))
            else:  # pid
                e.symptoms.append(SymptomMention("pelvic_pain", "present", "not_documented", "moderate"))
                if self.p(0.5) and age >= 18:
                    e.symptoms.append(SymptomMention("dyspareunia"))
            if label and self.p(0.5):
                e.symptoms.append(SymptomMention(self.pick(["dysmenorrhoea", "dyschezia", "dyspareunia" if age >= 18 else "dysmenorrhoea"]), "present",
                                                 "explicit_yes" if self.p(0.5) else "not_documented", self.pick(["moderate", "severe"])))
            encs.append(e)
        k = min(1, len(encs) - 1)
        if alt == "fibroids":
            encs[k].investigations.append(self._ultrasound("fibroid")); encs[k].diagnoses.append("fibroids"); encs[k].meds.append(("tranexamic acid", "tranexamic"))
        elif alt == "adenomyosis":
            encs[k].investigations.append(self._ultrasound("adenomyosis")); encs[k].diagnoses.append("adenomyosis")
        elif alt == "uti":
            for e in encs:
                pos = self.p(0.6 if not label else 0.35)
                e.investigations.append(EventInv("urine_culture", "MSU: E. coli >10^5" if pos else "MSU: no growth", not pos))
                if pos:
                    e.meds.append(("nitrofurantoin", "antibiotic")); e.diagnoses.append("uti")
        elif alt == "interstitial_cystitis":
            encs[-1].diagnoses.append("interstitial_cystitis")
        else:
            encs[k].investigations.append(EventInv("sti_screen", "Chlamydia NAAT positive" if not label else "STI screen negative", label))
            encs[k].diagnoses.append("pid"); encs[k].meds.append(("doxycycline", "antibiotic"))
        self._pad_unrelated(encs, start, end, self.r.randint(0, 3))
        return encs, f"overlap_{alt}"

    def _s6(self, label, age):  # repeated visits, inconclusive investigations
        start, end = self._span(2, 9)
        if label:
            encs = self._pain_course(start, end, self.r.randint(6, 12), link_p=0.45,
                                     domains={"dyspareunia": .25, "dyschezia": .2, "urinary": .1, "fatigue": .35, "bloating": .25}, rare_p=0.08,
                                     sexually_active=age >= 18)
        else:
            encs = self._pain_course(start, end, self.r.randint(5, 10), link_p=0.3,
                                     domains={"fatigue": .4, "bloating": .35, "back_pain": .3, "bowel_change": .25, "dyspareunia": .1, "urinary": .1},
                                     response=self.pick(["failure", "success"]), sexually_active=age >= 18)
        for e in encs:
            if self.p(0.25):
                e.investigations.append(self.pick([
                    EventInv("fbc", "FBC and ferritin normal.", True), EventInv("urine_culture", "Urine dip negative.", True),
                    EventInv("ca125", f"CA125 {self.r.randint(10, 33)} U/ml", True), EventInv("pelvic_ultrasound", "Pelvic USS NAD.", True)]))
        self._pad_unrelated(encs, start, end, self.r.randint(1, 4))
        return encs, "persistent_inconclusive" if label else "frequent_attender"

    def _s7(self, label, age):  # insufficient / incomplete records
        start, end = self._span(0.2, 2)
        encs = []
        n = self.pick([1, 1, 2])
        for d in self.dates(start, end, n, False):
            e = EncounterSkeleton("", d, "primary_care", "pain", sparse=True)
            if self.p(0.7):
                e.symptoms.append(SymptomMention(self.pick(["pelvic_pain", "dysmenorrhoea"]) if (label or self.p(0.5)) else "fatigue"))
            encs.append(e)
        return encs, "sparse_record"

    def _s8(self, label, age):  # different terminology
        start, end = self._span(1.5, 8)
        if label:
            encs = self._pain_course(start, end, self.r.randint(4, 9), link_p=0.65, rare_p=0.55,
                                     domains={"dyspareunia": .3, "dyschezia": .3, "urinary": .1}, sexually_active=age >= 18)
        else:
            encs = self._pain_course(start, end, self.r.randint(2, 5), link_p=0.4, rare_p=0.55, domains={"bloating": .3}, response="success", escalate=False)
        self._pad_unrelated(encs, start, end, self.r.randint(0, 3))
        return encs, "varied_terms"

    def _s9(self, label, age):  # one isolated symptom / encounter amid other care
        start, end = self._span(2, 8)
        encs = [self._unrelated(d) for d in self.dates(start, end, self.r.randint(3, 8), False)]
        idx = self.r.randrange(len(encs))
        e = EncounterSkeleton("", encs[idx].date, "primary_care", "pain")
        e.symptoms.append(SymptomMention(self.pick(["pelvic_pain", "dysmenorrhoea", "heavy_menstrual_bleeding", "fatigue"]), "present",
                                         "explicit_yes" if self.p(0.3) else "not_documented", self.pick(["mild", "moderate", None])))
        if self.p(0.4):
            e.meds.append(("ibuprofen", "nsaid"))
        encs[idx] = e
        return encs, "isolated"

    def _s10(self, label, age):  # inconsistent documentation across encounters
        encs, _ = self._s1(label, age)
        pain = [e for e in encs if e.reason == "pain"]
        if len(pain) >= 2:
            a, b = self.r.sample(pain, 2)
            if a.symptoms:
                a.symptoms[0].menstrual_link = "explicit_no" if a.symptoms[0].concept != "dysmenorrhoea" else "not_documented"
            # contradiction: one note denies dyspareunia, a later one documents it
            if age >= 18:
                first, second = sorted([a, b], key=lambda e: e.date)
                first.symptoms.append(SymptomMention("dyspareunia", "absent"))
                second.symptoms.append(SymptomMention("dyspareunia", "present"))
        for e in pain:
            for s in e.symptoms:
                if s.status == "present" and self.p(0.2):
                    s.status = "uncertain"
        return encs, "inconsistent"

    def _s11(self, label, age):  # routine, unrelated care (silent positives live here)
        start, end = self._span(1, 8)
        encs = [self._unrelated(d) for d in self.dates(start, end, self.r.randint(3, 9), False)]
        if label and self.p(0.3):
            d = encs[-1].date
            e = EncounterSkeleton("", d, "primary_care", "pain")
            e.symptoms.append(SymptomMention("fatigue"))
            encs[-1] = e
        return encs, "silent" if label else "routine"

    # ----------------------------------------------------- data quality noise
    def _inject_quality_issues(self, ps: PatientSkeleton):
        encs = ps.encounters
        if len(encs) >= 3 and self.p(0.10):  # copy-forward note
            i = self.r.randrange(1, len(encs))
            src = encs[i - 1]
            if src.reason == "pain":
                clone = EncounterSkeleton("", (src.date or REFERENCE_DATE) + timedelta(days=self.r.randint(7, 40)), src.setting, "pain",
                                          symptoms=[SymptomMention(**vars(s)) for s in src.symptoms], copy_forward_of="prev")
                encs.insert(i, clone)
        if len(encs) >= 2 and self.p(0.06):  # duplicated record (same encounter exported twice)
            i = self.r.randrange(len(encs))
            dup = EncounterSkeleton(**{**vars(encs[i]), "encounter_id": "", "duplicate_of": "prev"})
            encs.insert(i + 1, dup)
        if self.p(0.04):
            e = self.pick(encs)
            e.date_error = self.pick(["future", "missing"])
        encs.sort(key=lambda e: e.date or REFERENCE_DATE)
