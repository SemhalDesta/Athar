"""Optional: rewrite template notes with an LLM so they read like real clinician notes.

The structured skeleton stays the ground truth, so the LLM is told to preserve every fact,
negation, uncertainty and menstrual relationship exactly and to add nothing clinical.
Duplicated / copy-forward encounters are left as-is so data-quality tests still work.

    python -m athar.generator.llm_writer dev
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from .. import llm
from ..config import DATA_DIR, rules

SYSTEM = """You rewrite synthetic, fictional clinical notes so they read like genuine notes written by busy
clinicians (GPs, ED doctors, gynaecologists) in UK/UAE practice. Rules:
- Preserve EVERY clinical fact exactly: each symptom, whether it is present, explicitly denied, or uncertain,
  any stated link (or explicit non-link) to periods, severity, functional impact, medications, results, referrals
  and impressions. Do not add, remove or strengthen any symptom, diagnosis, result or time relationship.
- Vary style and wording naturally (abbreviations, terse fragments, full sentences). Keep each note roughly
  the same length. Keep the same encounter_id.
- All people are fictional. Do not add names, dates or identifiers.
Return JSON: {"notes": [{"encounter_id": "...", "note_text": "..."}]}"""


def _rewrite_patient(p: dict) -> dict:
    if any(e.get("note_source") == "llm_rewrite" for e in p["encounters"]):
        return p  # already rewritten in an earlier run
    todo = []
    seen = set()
    for e in p["encounters"]:
        if e["note_text"] in seen or "As above" in e["note_text"] or "as previous" in e["note_text"]:
            continue  # keep duplicates / copy-forward identical
        seen.add(e["note_text"])
        todo.append({"encounter_id": e["encounter_id"], "note_text": e["note_text"]})
    if not todo:
        return p
    try:
        out = llm.chat_json(SYSTEM, json.dumps({"notes": todo}), temperature=rules()["llm"]["temperature_generation"])
        new = {n["encounter_id"]: n["note_text"] for n in out.get("notes", []) if n.get("note_text")}
    except Exception as ex:  # keep the template note if the call fails
        print(f"  {p['patient_id']}: rewrite failed ({ex}); keeping template notes")
        return p
    for e in p["encounters"]:
        if e["encounter_id"] in new:
            e["note_text"] = new[e["encounter_id"]]
            e.setdefault("note_source", "llm_rewrite")
    return p


def rewrite_dataset(name: str):
    if not llm.available():
        raise SystemExit("OPENROUTER_API_KEY not set - cannot rewrite notes.")
    path = DATA_DIR / f"{name}_records.json"
    recs = json.loads(path.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=rules()["llm"]["max_concurrency"]) as ex:
        recs = list(ex.map(_rewrite_patient, recs))
    path.write_text(json.dumps(recs, indent=1), encoding="utf-8")
    done = sum(any(e.get("note_source") == "llm_rewrite" for e in p["encounters"]) for p in recs)
    print(f"{name}: {done}/{len(recs)} patients have LLM-rewritten notes ({llm.llm_model()}). Re-run to finish any that failed.")


if __name__ == "__main__":
    rewrite_dataset(sys.argv[1] if len(sys.argv) > 1 else "dev")
