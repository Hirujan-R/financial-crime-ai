"""Kedro nodes for the AMLSim generalisation experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from financial_crime_ai.pipelines.aml_data.features import (
    MODEL_FEATURES,
    build_train_account_features,
    build_transaction_features,
)

from .generator import generate_amlsim_data


def generate_amlsim(
    n_accounts: int,
    n_timestamps: int,
    seed: int,
    normal_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = generate_amlsim_data(
        n_accounts=n_accounts,
        n_timestamps=n_timestamps,
        seed=seed,
        normal_rate=normal_rate,
    )
    return (
        data["accounts"],
        data["train_transactions"],
        data["test_transactions"],
        data["summary"],
    )


def evaluate_generalisation(
    aml_train_transactions: pd.DataFrame,
    aml_test_transactions: pd.DataFrame,
    aml_accounts: pd.DataFrame,
    ibm_best_model_info: dict,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train on known typologies; score on held-out ones.

    A fresh model is trained on the AMLSim *training domain* (normal +
    structuring + layering) and evaluated on the held-out domain (cycle,
    fan_out, fan_in).  The IBM-locked model is also run on the same held-out
    transactions as a cross-source robustness check.
    """
    features = MODEL_FEATURES
    acct_feat = build_train_account_features(aml_train_transactions, aml_accounts)
    train_feat = build_transaction_features(aml_train_transactions, acct_feat)
    test_feat = build_transaction_features(aml_test_transactions, acct_feat)

    X_train, y_train = (
        train_feat,
        aml_train_transactions["IS_FRAUD"].astype(int).reset_index(drop=True).values,
    )
    X_test, y_test = (
        test_feat,
        aml_test_transactions["IS_FRAUD"].astype(int).reset_index(drop=True).values,
    )

    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X_train[features], y_train)

    # select threshold on the training domain, then lock it
    from financial_crime_ai.pipelines.model_selection.models import _select_threshold

    threshold, *_ = _select_threshold(
        pd.Series(y_train), model.predict_proba(X_train[features])[:, 1]
    )
    aml_test_prob = model.predict_proba(X_test[features])[:, 1]

    # per-typology generalisation report
    alert = aml_test_transactions["ALERT_TYPE"].reset_index(drop=True)
    rows = []
    for typ in ["none", "cycle", "fan_out", "fan_in"]:
        mask = (alert == typ).to_numpy()
        if mask.sum() == 0:
            continue
        prob = aml_test_prob[mask]
        truth = y_test[mask]
        auc = (
            roc_auc_score(truth, prob)
            if truth.sum() > 0 and truth.sum() < len(truth)
            else np.nan
        )
        detected = int((prob >= threshold).sum())
        recall = float((prob[truth == 1] >= threshold).sum() / max(1, int(truth.sum())))
        rows.append(
            {
                "alert_type": typ,
                "n": int(mask.sum()),
                "n_fraud": int(truth.sum()),
                "flagged": detected,
                "auc": auc,
                "recall_at_threshold": recall,
                "threshold": threshold,
            }
        )
    overall = {
        "alert_type": "overall",
        "n": len(y_test),
        "n_fraud": int(y_test.sum()),
        "flagged": int((aml_test_prob >= threshold).sum()),
        "auc": roc_auc_score(y_test, aml_test_prob),
        "recall_at_threshold": float(
            (aml_test_prob[y_test == 1] >= threshold).sum() / max(1, int(y_test.sum()))
        ),
        "threshold": threshold,
    }
    rows.append(overall)

    # cross-source: IBM-locked model on AMLSim held-out data
    ibm_model = ibm_best_model_info["model"]
    kind = ibm_best_model_info["kind"]
    X_ibm = X_test[ibm_best_model_info["feature_columns"]].fillna(0.0)
    if kind != "xgboost":
        X_ibm = ibm_best_model_info["scaler"].transform(X_ibm)
    ibm_prob = ibm_model.predict_proba(X_ibm)[:, 1]
    ibm_thr = ibm_best_model_info["threshold"]
    ibm_row = {
        "alert_type": "ibm_model_on_heldout",
        "n": len(y_test),
        "n_fraud": int(y_test.sum()),
        "flagged": int((ibm_prob >= ibm_thr).sum()),
        "auc": roc_auc_score(y_test, ibm_prob),
        "recall_at_threshold": float(
            (ibm_prob[y_test == 1] >= ibm_thr).sum() / max(1, int(y_test.sum()))
        ),
        "threshold": ibm_thr,
    }
    rows.append(ibm_row)

    metrics = pd.DataFrame(rows)
    info = {
        "model": model,
        "threshold": threshold,
        "feature_columns": features,
        "test_prob": aml_test_prob,
    }
    return metrics, info
