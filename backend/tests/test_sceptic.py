"""Sceptic reviewer: quotes are verified by code, and the outcome can only move down."""
from athar.agents import sceptic as sc
from athar.pipeline import Athar, extract_all, load_records
from athar.schemas import Category

ATH = Athar()
RECS = {r.patient_id: r for r in load_records("eval")}
EXS = extract_all(list(RECS.values()), "lexicon", "eval")


def _flagged():
    for pid, r in RECS.items():
        a = ATH.assess(r, EXS[pid])
        if a.category == Category.review:
            return r, EXS[pid], a
    raise AssertionError("no flagged patient")


def _fake(monkeypatch, payload):
    monkeypatch.setattr(sc.llm, "chat_json", lambda *a, **k: payload)


def test_unverifiable_quotes_are_rejected(monkeypatch):
    r, ex, a = _flagged()
    eid = r.encounters[0].encounter_id
    _fake(monkeypatch, {"verdict": "challenged", "summary": "x", "challenges": [
        {"type": "alternative", "severity": "major", "claim": "Confirmed fibroids", "encounter_id": eid, "quote": "this sentence is not in any note"},
        {"type": "gap", "claim": "Cycle link never recorded"}]})
    res = sc.review(r, ex, a, ATH)
    assert [c.verified for c in res.challenges] == [False, True]
    assert res.final_category == Category.review.value and not res.changed  # an invented quote cannot downgrade


def test_verified_major_alternative_downgrades(monkeypatch):
    r, ex, a = _flagged()
    e = next(e for e in r.encounters if len(e.note_text) > 20)
    quote = e.note_text[:20]
    _fake(monkeypatch, {"verdict": "challenged", "summary": "x", "challenges": [
        {"type": "alternative", "severity": "major", "claim": "Confirmed alternative", "encounter_id": e.encounter_id, "quote": quote}]})
    res = sc.review(r, ex, a, ATH)
    assert res.challenges[0].verified and res.changed and res.final_category == Category.insufficient.value


def test_misread_correction_reruns_pipeline_and_never_raises(monkeypatch):
    r, ex, a = _flagged()
    # remove every symptom finding: the re-assessed record must not be flagged
    fids = [f"F{i + 1}" for i, f in enumerate(ex.findings) if f.category == "symptom"]
    by_enc = {}
    for i, f in enumerate(ex.findings):
        if f.category == "symptom":
            by_enc.setdefault(f.encounter_id, []).append(f"F{i + 1}")
    ch = []
    for eid, ids in by_enc.items():
        f = ex.findings[int(ids[0][1:]) - 1]
        ch.append({"type": "misread", "severity": "major", "claim": "not supported", "encounter_id": eid, "quote": f.evidence_quote,
                   "finding_ids": ids, "correction": "remove"})
    _fake(monkeypatch, {"verdict": "challenged", "summary": "x", "challenges": ch})
    res = sc.review(r, ex, a, ATH)
    assert res.corrections_applied and res.changed
    assert sc.RANK[Category(res.final_category)] < sc.RANK[Category.review]


def test_unflagged_records_are_not_sent_to_the_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called")
    monkeypatch.setattr(sc.llm, "chat_json", boom)
    for pid, r in RECS.items():
        a = ATH.assess(r, EXS[pid])
        if a.category != Category.review:
            res = sc.review(r, EXS[pid], a, ATH)
            assert not res.ran and res.final_category == a.category.value
            break
