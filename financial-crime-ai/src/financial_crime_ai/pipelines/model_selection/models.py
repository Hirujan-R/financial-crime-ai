"""Model zoo and rigorous model selection.

Fits Logistic Regression, XGBoost and an MLP (neural network) over a small
hyperparameter grid.  All choices - hyperparameters, decision threshold and
which model to deploy - are made *only* on the validation split.  The
selected model is then locked and evaluated exactly once on the test split
to give an honest estimate of future performance.

Because the classes are heavily imbalanced (~0.13% fraud) we:
* balance class weights (LogReg/MLP) and scale_pos_weight (XGBoost),
* select the decision threshold on validation to maximise recall while
  keeping precision above a floor (typical AML triage),
* score model choice by validation average precision.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MLP_MAX_ROWS = 150_000


def fit_candidates(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    candidates: list[dict],
    feature_columns: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit every candidate and return (results_table, fitted_candidates)."""
    X_train, y_train = _split_label(train_features, feature_columns)
    X_val, y_val = _split_label(validation_features, feature_columns)

    scaler = StandardScaler().fit(X_train)
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    rows: list[dict] = []
    fitted: dict[str, object] = {}

    for cand in candidates:
        name = cand["name"]
        kind = cand["type"]
        for params in cand["grid"]:
            if kind == "logreg":
                model = LogisticRegression(
                    C=float(params["C"]),
                    class_weight="balanced",
                    max_iter=2000,
                    n_jobs=-1,
                    random_state=seed,
                )
                X_tr, X_v = scaler.transform(X_train), scaler.transform(X_val)
            elif kind == "xgboost":
                model = XGBClassifier(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]),
                    learning_rate=float(params["learning_rate"]),
                    scale_pos_weight=scale_pos_weight,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    n_jobs=-1,
                    random_state=seed,
                    eval_metric="logloss",
                )
                X_tr, X_v = X_train, X_val
            elif kind == "mlp":
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(
                        int(x) for x in params["hidden_layer_sizes"]
                    ),
                    alpha=float(params.get("alpha", 1e-3)),
                    max_iter=int(params.get("max_iter", 60)),
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=seed,
                )
                # NNs are expensive: fit on a capped stratified sample
                X_tr = _stratified_sample(X_train, y_train, MLP_MAX_ROWS, seed)
                y_tr = y_train.loc[X_tr.index]
                X_tr = scaler.transform(X_tr)
                X_v = scaler.transform(X_val)
            else:  # pragma: no cover
                raise ValueError(f"unknown model type {kind}")

            model.fit(X_tr, y_tr if kind == "mlp" else y_train)
            val_prob = model.predict_proba(X_v)[:, 1]
            threshold, f2, prec, rec = _select_threshold(y_val, val_prob)

            key = f"{name}|{json.dumps(params, sort_keys=True)}"
            fitted[key] = {
                "name": name,
                "kind": kind,
                "params": params,
                "model": model,
                "scaler": scaler,
                "threshold": threshold,
            }
            rows.append(
                {
                    "model": key,
                    "type": kind,
                    "params": json.dumps(params),
                    "val_auc": roc_auc_score(y_val, val_prob),
                    "val_ap": average_precision_score(y_val, val_prob),
                    "val_threshold": threshold,
                    "val_f2": f2,
                    "val_precision": prec,
                    "val_recall": rec,
                }
            )

    results = pd.DataFrame(rows).sort_values("val_ap", ascending=False)
    return results.reset_index(drop=True), fitted


def select_best(
    model_results: pd.DataFrame,
    fitted_candidates: dict[str, object],
    test_features: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Lock the best-by-validation model, then evaluate it once on test."""
    best_row = model_results.iloc[0]
    key = best_row["model"]
    cand = fitted_candidates[key]
    model = cand["model"]
    scaler = cand["scaler"]
    threshold = float(cand["threshold"])
    kind = cand["kind"]

    X_test, y_test = _split_label(test_features, feature_columns)
    X_test_scaled = scaler.transform(X_test) if kind != "xgboost" else X_test
    test_prob = model.predict_proba(X_test_scaled)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    best_model_info = {
        "model": model,
        "scaler": scaler,
        "kind": kind,
        "name": best_row["type"],
        "params": json.loads(best_row["params"]),
        "threshold": threshold,
        "feature_columns": list(feature_columns),
        "val_auc": float(best_row["val_auc"]),
        "val_ap": float(best_row["val_ap"]),
    }

    test_metrics = pd.DataFrame(
        [
            {
                "metric": "auc",
                "value": roc_auc_score(y_test, test_prob),
            },
            {
                "metric": "average_precision",
                "value": average_precision_score(y_test, test_prob),
            },
            {
                "metric": "precision_at_threshold",
                "value": precision_score(y_test, test_pred, zero_division=0),
            },
            {
                "metric": "recall_at_threshold",
                "value": recall_score(y_test, test_pred, zero_division=0),
            },
            {
                "metric": "f2_at_threshold",
                "value": fbeta_score(y_test, test_pred, beta=2, zero_division=0),
            },
            {"metric": "n_test", "value": int(len(y_test))},
            {"metric": "n_test_fraud", "value": int(y_test.sum())},
            {"metric": "n_flagged", "value": int(test_pred.sum())},
        ]
    )

    predictions = test_features[
        ["TX_ID", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"]
    ].copy()
    predictions["risk_score"] = test_prob
    predictions["predicted_fraud"] = test_pred
    predictions["IS_FRAUD"] = (
        test_features["IS_FRAUD"].astype(int).reset_index(drop=True).values
    )

    return best_model_info, test_metrics, predictions


def _split_label(
    frame: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    X = frame[list(feature_columns)]
    y = frame["IS_FRAUD"].astype(int)
    return X, y


def _stratified_sample(
    X: pd.DataFrame, y: pd.Series, max_rows: int, seed: int
) -> pd.DataFrame:
    if len(X) <= max_rows:
        return X
    pos = X[y == 1]
    neg = X[y == 0]
    n_pos = min(len(pos), int(max_rows * 0.5))
    n_neg = min(len(neg), max_rows - n_pos)
    idx = pd.concat(
        [pos.sample(n_pos, random_state=seed), neg.sample(n_neg, random_state=seed)]
    ).index
    return X.loc[idx]


def _select_threshold(
    y_true: pd.Series, prob: np.ndarray, min_precision: float = 0.5
) -> tuple[float, float, float, float]:
    """Threshold maximising F2 subject to a precision floor (validation)."""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    f2 = (5 * prec * rec) / np.maximum(4 * prec + rec, 1e-9)
    eligible = f2.copy()
    eligible[prec < min_precision] = -1.0
    idx = int(np.argmax(eligible))
    if eligible[idx] < 0:
        idx = int(np.argmax(f2))
    threshold = thr[idx] if idx < len(thr) else 0.5
    return float(threshold), float(f2[idx]), float(prec[idx]), float(rec[idx])
