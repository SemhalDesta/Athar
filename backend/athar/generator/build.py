"""Build the synthetic development and evaluation datasets.

    python -m athar.generator.build                 # offline template notes
    python -m athar.generator.build --llm-rewrite   # then rewrite notes with the LLM (needs OPENROUTER_API_KEY)

Outputs (data/):
    dev_records.json / dev_truth.json     400 patients, used to build rules and train the model
    eval_records.json / eval_truth.json   200 patients, different seed + wording, never used in development
Records and ground truth are stored separately; the pipeline only ever reads *_records.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from ..config import DATA_DIR
from .notes import render_patient
from .skeleton import SCENARIOS, Gen

MIN_PER_SCENARIO = 6


def build(name: str, n: int, seed: int, prefix: str, alt_style: bool):
    g = Gen(seed, prefix)
    skeletons = []
    # guarantee a minimum count of every brief scenario, then sample label-first
    for sc in range(1, 11):
        for k in range(MIN_PER_SCENARIO):
            skeletons.append(g.patient(label=(k % 2 == 0), scenario=sc))
    while len(skeletons) < n:
        skeletons.append(g.patient())
    records, truths = [], []
    for i, ps in enumerate(skeletons):
        rec, truth = render_patient(ps, seed * 100003 + i, alt_style=alt_style)
        records.append(json.loads(rec.model_dump_json()))
        truths.append(truth)
    (DATA_DIR / f"{name}_records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (DATA_DIR / f"{name}_truth.json").write_text(json.dumps(truths, indent=1), encoding="utf-8")
    pos = sum(t["endometriosis"] for t in truths)
    print(f"{name}: {len(records)} patients, {pos} positive ({pos/len(records):.0%}), "
          f"{sum(len(r['encounters']) for r in records)} encounters")
    print("  scenarios:", dict(sorted(Counter(SCENARIOS[t['scenario']] for t in truths).items())))
    print("  expected:", dict(Counter(t["expected_category"] for t in truths)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=400)
    ap.add_argument("--eval", type=int, default=200)
    ap.add_argument("--llm-rewrite", action="store_true")
    a = ap.parse_args()
    build("dev", a.dev, 11, "DEV", alt_style=False)
    build("eval", a.eval, 29, "EVL", alt_style=True)
    if a.llm_rewrite:
        from .llm_writer import rewrite_dataset
        rewrite_dataset("dev")
        rewrite_dataset("eval")


if __name__ == "__main__":
    main()
