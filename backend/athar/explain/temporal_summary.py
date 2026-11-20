"""Plain-language statements about how the documented history changes over time.

Every statement is computed from extracted findings (no LLM) and carries the encounters
it is based on, so the clinician can open each source note.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from itertools import combinations

from ..schemas import EvidenceRef, ExtractionResult, Finding, MenstrualLink, PatientRecord, PatternStatement, Status, YearCount
from ..vocab import PAIN_CONCEPTS, PAIN_TREATMENT_CLASSES, SYMPTOMS

SEV_RANK = {"mild": 1, "moderate": 2, "severe": 3}
SHORT = {
    "dysmenorrhoea": "painful periods", "pelvic_pain": "pelvic pain", "dyspareunia": "pain with sex",
    "dyschezia": "painful bowel movements", "urinary": "urinary pain", "subfertility": "difficulty conceiving",
    "heavy_menstrual_bleeding": "heavy periods",
}
TRACKED = PAIN_CONCEPTS | {"subfertility"}


def _fmt(d: dt.date) -> str:
    return d.strftime("%b %Y")


def _ref(f: Finding) -> EvidenceRef:
    return EvidenceRef(encounter_id=f.encounter_id, encounter_date=f.encounter_date, quote=f.evidence_quote,
                       concept=SYMPTOMS.get(f.concept, {}).get("label", f.concept), status=f.status.value)


def _months(a: dt.date, b: dt.date) -> int:
    return round(abs((b - a).days) / 30.44)


def summarise(rec: PatientRecord, ex: ExtractionResult, today: dt.date | None = None) -> tuple[list[PatternStatement], list[YearCount]]:
    today = today or dt.date.today()
    skip = {i.encounter_id for i in ex.issues if i.kind == "duplicate_encounter"}
    bad = {i.encounter_id for i in ex.issues if i.kind in ("missing_date", "future_date", "date_before_birth")}
    F = [f for f in ex.findings if f.encounter_id not in skip and f.encounter_id not in bad and f.encounter_date]
    pain = [f for f in F if f.category == "symptom" and f.status == Status.present and f.concept in TRACKED]
    out: list[PatternStatement] = []

    # visits per year (pain-related vs other), valid dates only
    pain_ids = {f.encounter_id for f in pain}
    years: dict[int, Counter] = defaultdict(Counter)
    for e in rec.encounters:
        if e.encounter_id in skip or e.encounter_id in bad or not e.date:
            continue
        years[e.date.year]["pain" if e.encounter_id in pain_ids else "other"] += 1
    by_year = []
    if years:  # include empty years in between so gaps are visible
        for y in range(min(years), max(years) + 1):
            c = years.get(y, Counter())
            by_year.append(YearCount(year=y, pain_visits=c["pain"], other_visits=c["other"]))

    # 1. recurrence of each symptom across visits
    by_concept: dict[str, list[Finding]] = defaultdict(list)
    for f in pain:
        by_concept[f.concept].append(f)
    for c, fs in sorted(by_concept.items(), key=lambda kv: -len({f.encounter_id for f in kv[1]})):
        visits = {f.encounter_id: f for f in sorted(fs, key=lambda f: f.encounter_date)}
        if len(visits) >= 2:
            first, last = min(f.encounter_date for f in fs), max(f.encounter_date for f in fs)
            out.append(PatternStatement(kind="recurrence", text=f"{SHORT.get(c, c).capitalize()} recorded at {len(visits)} separate visits over "
                                        f"{_months(first, last)} months ({_fmt(first)} to {_fmt(last)}).", evidence=[_ref(f) for f in visits.values()]))

    # 2. trend in pain-related visits per year
    py = [(y.year, y.pain_visits) for y in by_year]
    py = [(y, n) for y, n in py if y >= (min((y for y, n in py if n), default=9999))]  # start at first pain year
    if len(py) >= 2 and sum(n for _, n in py) >= 3:
        seq = " → ".join(f"{n} ({y})" for y, n in py)
        first_half = sum(n for _, n in py[: len(py) // 2]) / max(1, len(py) // 2)
        second_half = sum(n for _, n in py[len(py) // 2:]) / max(1, len(py) - len(py) // 2)
        if second_half > first_half * 1.3 and py[-1][1] >= 2:
            word = "increased"
        elif second_half < first_half * 0.7:
            word = "decreased"
        else:
            word = "stayed roughly steady"
        recent = [f for f in pain if f.encounter_date.year == py[-1][0]]
        out.append(PatternStatement(kind="trend", text=f"Pain-related visits {word}: {seq}.",
                                    evidence=[_ref(f) for f in {f.encounter_id: f for f in recent}.values()]))

    # 3. severity change over time
    sev = sorted([f for f in pain if f.severity], key=lambda f: f.encounter_date)
    if len(sev) >= 2:
        a, b = sev[0], sev[-1]
        if SEV_RANK[b.severity] > SEV_RANK[a.severity]:
            out.append(PatternStatement(kind="severity", text=f"Recorded pain severity rose from {a.severity} ({_fmt(a.encounter_date)}) to {b.severity} ({_fmt(b.encounter_date)}).", evidence=[_ref(a), _ref(b)]))
        elif SEV_RANK[b.severity] < SEV_RANK[a.severity]:
            out.append(PatternStatement(kind="severity", text=f"Recorded pain severity fell from {a.severity} ({_fmt(a.encounter_date)}) to {b.severity} ({_fmt(b.encounter_date)}).", evidence=[_ref(a), _ref(b)]))
        elif b.severity == "severe":
            n = len({f.encounter_id for f in sev if f.severity == "severe"})
            out.append(PatternStatement(kind="severity", text=f"Pain recorded as severe at {n} visits, from {_fmt(a.encounter_date)} to {_fmt(b.encounter_date)}.",
                                        evidence=[_ref(f) for f in sev if f.severity == "severe"]))

    # 4. symptoms documented together in the same visit
    per_enc: dict[str, dict[str, Finding]] = defaultdict(dict)
    for f in pain:
        per_enc[f.encounter_id][f.concept] = f
    pairs: dict[tuple, list[Finding]] = defaultdict(list)
    for fs in per_enc.values():
        for a_, b_ in combinations(sorted(fs), 2):
            pairs[(a_, b_)].extend([fs[a_]])
    for (a_, b_), fs in sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:2]:
        if len(fs) >= 2:
            out.append(PatternStatement(kind="co_occurrence", text=f"{SHORT.get(a_, a_).capitalize()} and {SHORT.get(b_, b_)} recorded together at {len(fs)} visits.",
                                        evidence=[_ref(f) for f in fs]))

    # 5. explicit cyclical link for non-period pain
    cyc = [f for f in pain if f.concept != "dysmenorrhoea" and f.menstrual_link == MenstrualLink.explicit_yes]
    if cyc:
        kinds = sorted({SHORT.get(f.concept, f.concept) for f in cyc})
        out.append(PatternStatement(kind="cyclical", text=f"{', '.join(kinds).capitalize()} explicitly linked to periods at {len({f.encounter_id for f in cyc})} visit(s).",
                                    evidence=[_ref(f) for f in cyc]))

    # 6. persistence after first pain treatment
    tx = sorted([f for f in F if f.category == "medication" and f.concept in PAIN_TREATMENT_CLASSES], key=lambda f: f.encounter_date)
    if tx and pain:
        later = [f for f in pain if (f.encounter_date - tx[0].encounter_date).days >= 56]
        if later:
            last = max(f.encounter_date for f in later)
            out.append(PatternStatement(kind="persistence", text=f"Pain still recorded {_months(tx[0].encounter_date, last)} months after the first pain treatment "
                                        f"({tx[0].evidence_quote}, {_fmt(tx[0].encounter_date)}), at {len({f.encounter_id for f in later})} later visit(s).",
                                        evidence=[EvidenceRef(encounter_id=tx[0].encounter_id, encounter_date=tx[0].encounter_date, quote=tx[0].evidence_quote, concept="Treatment started", status="medication")]
                                        + [_ref(f) for f in {f.encounter_id: f for f in later}.values()]))
    return out, by_year
