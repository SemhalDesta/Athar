# Athar evaluation report (synthetic held-out set)

> These results measure performance on synthetic data generated for development. They show that the pipeline behaves as designed on constructed scenarios; they do NOT demonstrate clinical accuracy. Real-world performance requires real, labelled, de-identified records and external validation.

Extractor: `llm:anthropic/claude-sonnet-5.5-20260928` · Model: `logistic_regression` (threshold 0.399, highest score threshold with out-of-fold sensitivity >= 80% on the development set) · Patients: 200

## Model comparison (5-fold CV on development set)

| Model | CV ROC-AUC | CV avg precision |
|---|---|---|
| logistic_regression | 0.857 | 0.814 |
| random_forest | 0.83 | 0.775 |
| xgboost | 0.851 | 0.787 |

## A. Constructed-scenario agreement

Rows = category the scenario was built to show; columns = Athar's output.

| expected \ predicted | REVIEW | INSUFFICIENT | NO PATTERN |
|---|---|---|---|
| REVIEW | 66 | 0 | 38 |
| INSUFFICIENT | 2 | 9 | 3 |
| NO PATTERN | 7 | 0 | 75 |

Three-class accuracy: **0.75**

**Review vs not (constructed)** — sensitivity 0.635, specificity 0.906, precision 0.88, F1 0.737, FPR 0.094, FNR 0.365 (TP 66, FP 9, TN 87, FN 38)

**Insufficient-information detection** — sensitivity 0.643, specificity 1.0, precision 1.0, F1 0.783, FPR 0.0, FNR 0.357 (TP 9, FP 0, TN 186, FN 5)

## B. Agreement with the hidden synthetic label

**Flag vs hidden label** — sensitivity 0.712, specificity 0.819, precision 0.693, F1 0.703, FPR 0.181, FNR 0.288 (TP 52, FP 23, TN 104, FN 21)

Model-score ROC-AUC on the evaluation set: **0.867** (published symptom-based models: ~0.65–0.85).

## Subgroups

| Subgroup | n | sens (constructed) | spec (constructed) | sens (label) | spec (label) |
|---|---|---|---|---|---|
| incomplete_records | 63 | 0.659 | 0.864 | 0.839 | 0.875 |
| alternative_condition_scenarios | 74 | 0.757 | 0.892 | 0.818 | 0.731 |
| varied_terminology | 10 | 0.556 | 1.0 | 1.0 | 1.0 |
| inconsistent_documentation | 17 | 0.471 | None | 1.0 | 1.0 |
| silent_positives_and_routine | 16 | None | 1.0 | 0.0 | 1.0 |

## Extraction quality (vs generator ground truth)

Symptom precision 0.994, recall 0.996, status accuracy 0.998, menstrual-link accuracy 0.96.

## LLM vs keyword extractor

| Extractor | symptom recall | precision | flag sensitivity (label) | specificity (label) |
|---|---|---|---|---|
| lexicon | 0.936 | 0.998 | 0.685 | 0.795 |
| llm:anthropic/claude-sonnet-5.5-20260928 | 0.996 | 0.994 | 0.712 | 0.819 |
