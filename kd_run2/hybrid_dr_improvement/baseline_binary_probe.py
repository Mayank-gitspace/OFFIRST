"""Reproduce a development-only any-DR probe from saved KD Run 2 IDRiD features.

This is a frozen-feature baseline, not end-to-end fine-tuning. It uses the
existing official IDRiD 413/103 train/test split and does not touch images.
The test set was scored in the prior IDRiD grading report, so results are not
independent. No threshold is selected on test data (fixed p >= 0.5).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "kd_run2" / "idrid_grading_eval" / "embeddings"
OUT = Path(__file__).resolve().parent / "baseline_binary_probe.json"
GRID = [0.01, 0.1, 1.0]
THRESHOLD = 0.5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    start = time.perf_counter()
    ids_path = SOURCE / "ids.csv"
    embedding_path = SOURCE / "kd768.npy"
    meta = pd.read_csv(ids_path)
    embeddings = np.load(embedding_path)
    if len(meta) != 516 or embeddings.shape != (516, 768):
        raise ValueError(f"Unexpected IDRiD feature contract: {embeddings.shape}, {len(meta)}")
    if set(meta["split"]) != {"train", "test"}:
        raise ValueError("Expected the saved official IDRiD train/test split.")

    train = np.flatnonzero(meta["split"].to_numpy() == "train")
    test = np.flatnonzero(meta["split"].to_numpy() == "test")
    grade = meta["dr"].to_numpy(dtype=np.int64)
    y = (grade > 0).astype(np.int64)  # requested endpoint: grade 0 vs grades 1–4
    x = embeddings.astype(np.float64, copy=False)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(class_weight="balanced", max_iter=3000)),
    ])
    search = GridSearchCV(
        pipeline,
        {"logreg__C": GRID},
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        refit=True,
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(x[train], y[train])
    probability = search.best_estimator_.predict_proba(x[test])[:, 1]
    predicted = probability >= THRESHOLD
    truth = y[test]
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()

    per_grade = {}
    for g in range(5):
        selected = grade[test] == g
        per_grade[str(g)] = {
            "n": int(selected.sum()),
            "detected_at_fixed_threshold": int(predicted[selected].sum()),
            "sensitivity_within_grade": (
                float(predicted[selected].mean()) if g > 0 and selected.any() else None
            ),
        }

    result = {
        "experiment": "KD epoch-40 frozen 768-D features + balanced logistic probe; any DR",
        "classification": "grade 0 vs grades 1-4",
        "source": "saved kd_run2/idrid_grading_eval/embeddings; no image files read",
        "split": {
            "train_images": int(len(train)),
            "test_images": int(len(test)),
            "train_positive": int(y[train].sum()),
            "train_negative": int((y[train] == 0).sum()),
            "test_positive": int(truth.sum()),
            "test_negative": int((truth == 0).sum()),
            "patient_ids_available": False,
        },
        "model": {
            "encoder_feature_dim": 768,
            "l2_normalized": True,
            "scaler": "StandardScaler fit within CV folds and refit on all training rows",
            "classifier": "LogisticRegression(class_weight='balanced')",
            "c_grid": GRID,
            "cv": "5-fold stratified on train only, shuffle=True, random_state=0",
            "selection_metric": "ROC-AUC",
            "selected_c": float(search.best_params_["logreg__C"]),
            "selected_cv_auc": float(search.best_score_),
            "decision_threshold": THRESHOLD,
            "threshold_selected_on_test": False,
        },
        "test_metrics": {
            "accuracy": float(accuracy_score(truth, predicted)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "f1_positive_dr": float(f1_score(truth, predicted, zero_division=0)),
            "auroc": float(roc_auc_score(truth, probability)),
            "auprc": float(average_precision_score(truth, probability)),
            "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
            "grade_detail": per_grade,
        },
        "provenance": {
            "features_sha256": sha256(embedding_path),
            "ids_sha256": sha256(ids_path),
            "prior_scoring": (
                "These same 103 IDRiD test images were already scored in the frozen-encoder "
                "grading report; this probe is development evidence, not independent validation."
            ),
            "no_deepdrid_used": True,
        },
        "probe_fit_seconds": time.perf_counter() - start,
        "interpretation": "Frozen-feature probe only; not experiment A, B, or C end-to-end fine-tuning.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
