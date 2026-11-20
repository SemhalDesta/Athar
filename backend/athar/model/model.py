"""ML model: training (model comparison), scoring and SHAP explanations.

The model learns how the longitudinal features relate to the (synthetic) hidden label.
Its output is an illustrative model score, NOT a clinically validated probability.
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MODEL_DIR
from ..temporal.features import MODEL_FEATURES

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def candidates() -> dict:
    c = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1),
    }
    if XGBClassifier is not None:
        c["xgboost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                                     eval_metric="logloss", random_state=0, n_jobs=2)
    return c


def to_matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[float(r.get(f, 0.0)) for f in MODEL_FEATURES] for r in rows])


def train(rows: list[dict], y: list[int], extractor: str, target_sensitivity: float = 0.80) -> dict:
    X, y = to_matrix(rows), np.array(y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results, oof = {}, {}
    for name, est in candidates().items():
        p = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
        oof[name] = p
        results[name] = {"cv_auc": round(roc_auc_score(y, p), 3), "cv_average_precision": round(average_precision_score(y, p), 3)}
    best = max(results, key=lambda k: results[k]["cv_auc"])
    p = oof[best]
    # operating point: highest threshold whose out-of-fold sensitivity >= target (screening favours sensitivity)
    thr = 0.5
    for t in np.unique(np.round(p, 3))[::-1]:
        if ((p >= t) & (y == 1)).sum() / max(1, (y == 1).sum()) >= target_sensitivity:
            thr = float(t)
            break
    est = candidates()[best].fit(X, y)
    joblib.dump(est, MODEL_DIR / "model.joblib")
    np.save(MODEL_DIR / "background.npy", X[np.random.RandomState(0).choice(len(X), min(100, len(X)), replace=False)])
    meta = {
        "chosen_model": best, "selection": "highest 5-fold cross-validated ROC-AUC on the synthetic development set",
        "candidates": results, "threshold": round(thr, 3),
        "threshold_rule": f"highest score threshold with out-of-fold sensitivity >= {target_sensitivity:.0%} on the development set",
        "features": MODEL_FEATURES, "extractor": extractor, "n_train": int(len(y)), "n_positive": int(y.sum()),
        "warning": "Trained on synthetic data. Scores are illustrative and not clinically validated probabilities.",
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class AtharModel:
    def __init__(self):
        self.est = joblib.load(MODEL_DIR / "model.joblib")
        self.meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
        self.background = np.load(MODEL_DIR / "background.npy")
        self._explainer = None

    @staticmethod
    def exists() -> bool:
        return (MODEL_DIR / "model.joblib").exists()

    def score(self, feats: dict) -> float:
        return float(self.est.predict_proba(to_matrix([feats]))[0, 1])

    def _get_explainer(self):
        if self._explainer is None:
            import shap
            name = self.meta["chosen_model"]
            if name == "logistic_regression":
                scaler, lr = self.est.steps[0][1], self.est.steps[1][1]
                self._explainer = ("linear", shap.LinearExplainer(lr, scaler.transform(self.background)), scaler)
            else:
                self._explainer = ("tree", shap.TreeExplainer(self.est), None)
        return self._explainer

    def shap_values(self, feats: dict) -> dict[str, float]:
        kind, expl, scaler = self._get_explainer()
        x = to_matrix([feats])
        if scaler is not None:
            x = scaler.transform(x)
        sv = expl.shap_values(x)
        if isinstance(sv, list):  # older shap: per-class list
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, features, classes)
            sv = sv[:, :, 1]
        return {f: float(val) for f, val in zip(MODEL_FEATURES, sv[0])}
