"""End-to-end pipeline: extraction -> temporal features -> model + rules -> explanation.

CLI:
    python -m athar.pipeline train   [--extractor lexicon|llm]   # train on dev set
    python -m athar.pipeline assess  [--dataset eval] [--extractor ...] [--llm-summary]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from .config import CACHE_DIR, DATA_DIR, OUTPUT_DIR, rules
from .engine.decide import decide
from .explain.explain import build_assessment, llm_summary
from .extraction.llm_extractor import get_extractor
from .model.model import AtharModel, train
from .schemas import Assessment, ExtractionResult, PatientRecord
from .temporal.features import compute_features


def load_records(dataset: str) -> list[PatientRecord]:
    return [PatientRecord(**r) for r in json.loads((DATA_DIR / f"{dataset}_records.json").read_text(encoding="utf-8"))]


def load_truth(dataset: str) -> dict[str, dict]:
    p = DATA_DIR / f"{dataset}_truth.json"
    return {t["patient_id"]: t for t in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else {}


def extract_all(records: list[PatientRecord], extractor_name: str, dataset: str, refresh: bool = False) -> dict[str, ExtractionResult]:
    ex = get_extractor(extractor_name)
    import hashlib
    # cache is keyed by the exact note content, so rewritten notes are never matched to stale results
    digest = hashlib.sha256("".join(r.model_dump_json() for r in records).encode()).hexdigest()[:12]
    cache = CACHE_DIR / f"extraction_{dataset}_{ex.name}_{digest}.json"
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if set(data) >= {r.patient_id for r in records}:
            return {k: ExtractionResult(**v) for k, v in data.items()}
    workers = rules()["llm"]["max_concurrency"] if ex.name == "llm" else 1

    def safe(rec):
        try:
            return rec.patient_id, ex.extract(rec), None
        except Exception as e:  # e.g. network drop; completed LLM calls stay cached
            return rec.patient_id, None, e

    out: dict[str, ExtractionResult] = {}
    pending = list(records)
    for attempt in range(3):
        failed, last_err = [], None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (pid, res, err) in enumerate(pool.map(safe, pending), 1):
                if err is None:
                    out[pid] = res
                else:
                    failed.append(pid); last_err = err
                if ex.name == "llm" and i % 25 == 0:
                    print(f"  {dataset}: {len(out)}/{len(records)} patients read", flush=True)
        if not failed:
            break
        print(f"  {len(failed)} patient(s) failed (last error: {last_err}); retrying in 30s...", flush=True)
        import time; time.sleep(30)
        pending = [r for r in records if r.patient_id in set(failed)]
    if len(out) < len(records):
        raise SystemExit(f"Could not read {len(records) - len(out)} patient(s) - check your internet connection and run the same "
                         f"command again. Completed work is cached, so it will resume quickly.")
    cache.write_text(json.dumps({k: json.loads(v.model_dump_json()) for k, v in out.items()}), encoding="utf-8")
    return out


class Athar:
    def __init__(self):
        self.model = None
        if AtharModel.exists():
            try:
                self.model = AtharModel()
            except Exception as e:  # e.g. library version mismatch: retrain with `python -m athar.pipeline train`
                print(f"[athar] could not load model ({e}); run: python -m athar.pipeline train --extractor lexicon")
        self.threshold = self.model.meta["threshold"] if self.model else 0.5

    def assess(self, rec: PatientRecord, ex: ExtractionResult, with_llm_summary: bool = False) -> Assessment:
        fs = compute_features(rec, ex)
        score = shap = None
        if self.model:
            score = self.model.score(fs.values)
            shap = self.model.shap_values(fs.values)
        d = decide(fs, score, self.threshold)
        a = build_assessment(rec.patient_id, ex, fs, d, score, shap, rec)
        return llm_summary(a) if with_llm_summary else a


def cmd_train(extractor: str):
    recs = load_records("dev")
    truth = load_truth("dev")
    exs = extract_all(recs, extractor, "dev")
    rows = [compute_features(r, exs[r.patient_id]).values for r in recs]
    y = [int(truth[r.patient_id]["endometriosis"]) for r in recs]
    meta = train(rows, y, extractor=next(iter(exs.values())).extractor)
    print(json.dumps(meta["candidates"], indent=1))
    print(f"chosen: {meta['chosen_model']}  threshold: {meta['threshold']}")


def cmd_assess(dataset: str, extractor: str, with_summary: bool):
    recs = load_records(dataset)
    exs = extract_all(recs, extractor, dataset)
    ath = Athar()
    out = [json.loads(ath.assess(r, exs[r.patient_id], with_summary).model_dump_json()) for r in recs]
    (OUTPUT_DIR / f"assessments_{dataset}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    from collections import Counter
    print(dataset, dict(Counter(a["category"] for a in out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "assess"])
    ap.add_argument("--dataset", default="eval")
    ap.add_argument("--extractor", default="auto", choices=["auto", "lexicon", "llm"])
    ap.add_argument("--llm-summary", action="store_true")
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a.extractor)
    else:
        cmd_assess(a.dataset, a.extractor, a.llm_summary)


if __name__ == "__main__":
    main()
