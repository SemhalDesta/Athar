# Athar (أثر): pattern detection for possible endometriosis

Athar reads a patient's historical records and flags patients whose documented history **may warrant clinical assessment** for possible endometriosis. It is screening support. It does not diagnose, and its scores are not probabilities. All data in this repository is synthetic and fictional.

## How it works

```
notes + structured EHR fields
   │
   ├─ 1. Extraction (AI)        LLM reads free-text notes → JSON findings with verbatim quotes
   │                            (keyword extractor = offline fallback and comparison baseline)
   ├─ 2. Temporal analysis      recurrence, cyclical links, symptom domains, persistence after
   │                            treatment / normal tests, escalation, duplicates & bad dates handled
   ├─ 3. ML model (AI)          logistic regression vs random forest vs XGBoost, chosen by CV
   ├─ 4. Rules engine           INSUFFICIENT gate, imaging override, guardrails that can block a flag
   ├─ 5. Explainability (AI)    SHAP contributions + evidence linked to encounters and quotes,
   │                            optional LLM summary grounded only in extracted findings
   └─ API (FastAPI) → React frontend
```

Outputs: `PATTERN WARRANTS CLINICAL REVIEW`, `INSUFFICIENT INFORMATION`, `NO NOTABLE PATTERN IDENTIFIED`.

The clinical basis, the feature tiers and the synthetic-data parameters are documented in the Clinical Evidence Document (NICE NG73, ESHRE 2022, Ballard 2008 and others). Every engine threshold is a configurable design choice in `backend/config/rules.yaml`.

## Project layout

```
backend/
  athar/generator/    synthetic data (label-first skeletons → notes; optional LLM rewrite)
  athar/extraction/   llm_extractor.py (OpenRouter), lexicon.py (baseline), common.py (structured + data quality)
  athar/temporal/     features.py
  athar/model/        model.py (training, scoring, SHAP)
  athar/engine/       decide.py (categories + guardrails)
  athar/explain/      explain.py (evidence-linked assessment, summaries)
  athar/evaluation/   evaluate.py (metrics, confusion matrix, subgroups, extractor comparison)
  athar/api/          main.py (FastAPI)
  config/rules.yaml   engine configuration
  tests/              pytest suite (30 tests, offline)
frontend/             React + Vite + TypeScript
data/                 *_records.json (input format), *_truth.json (synthetic ground truth, never used by the pipeline)
```

## Setup (Windows, PowerShell)

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then put your OpenRouter key in .env
```

macOS/Linux: `source .venv/bin/activate` and `cp .env.example .env`.

## Run with the AI steps (your OpenRouter key)

```powershell
# from backend/, with the venv active
python -m athar.generator.build --llm-rewrite        # 1. regenerate data, LLM rewrites notes (~600 calls)
python -m athar.pipeline train --extractor llm       # 2. LLM extraction of dev set + model training
python -m athar.evaluation.evaluate --extractor llm --compare   # 3. held-out evaluation, LLM vs keyword
uvicorn athar.api.main:app --port 8000               # 4. API (uses cached LLM extractions automatically)
```

All LLM responses are cached in `data/cache/llm/`, so re-runs are free. To choose a model, set `ATHAR_LLM_MODEL` in `.env` to any OpenRouter model ID.

The repository already ships with offline data, a trained model and an evaluation report, so the API and frontend work before you run any LLM step.

## Sceptic agent

After the model and rules, a second AI agent reviews every flagged record and argues against it: possible misread notes,
other documented explanations, and gaps in the record. Every challenge must quote the note; code discards any quote it
cannot find, and the sceptic can only downgrade a flag, never create one.

```powershell
cd backend
py -m athar.agents.sceptic --dataset eval --limit 3   # quick test on 3 flagged patients
py -m athar.agents.sceptic --dataset eval             # all flagged patients (resumable, cached)
py -m athar.evaluation.evaluate --compare             # adds the with/without-sceptic comparison
```

Restart the API afterwards so patient review shows the sceptic results. Doctor decisions are stored in `data/outputs/decisions.json`.

## Run the frontend

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173 (home page); patient review at /review
```

## Tests

```powershell
cd backend
python -m pytest -q
```

The tests cover repeated symptoms, menstrual associations, isolated symptoms, missing information, contradictory records, alternative explanations, normal imaging, duplicate encounters, incorrect dates and uncertain documentation, plus the guardrails and the LLM quote check.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | model, LLM availability, extractor per dataset |
| GET | `/api/patients?dataset=eval` | prioritised worklist |
| GET | `/api/patients/{id}?include_truth=false` | record + full assessment |
| POST | `/api/patients/{id}/summary` | LLM summary grounded in the findings |
| POST | `/api/assess` | assess any `PatientRecord` (`extractor`: auto / lexicon / llm) |
| GET | `/api/evaluation`, `/api/model`, `/api/rules` | report, model metadata, configuration |

Interactive documentation is at `http://localhost:8000/docs`.

## Using real data later

Export records into the `PatientRecord` format in `athar/schemas.py`: notes, plus prescriptions, investigations, referrals and coded diagnoses. Then re-run `train` with real labels. Synthetic results show that the pipeline behaves as designed; they do **not** show clinical accuracy.
