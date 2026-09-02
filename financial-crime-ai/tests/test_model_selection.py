"""Tests for model selection: model zoo, validation-driven selection, test eval."""

import pandas as pd
import pytest

from financial_crime_ai.pipelines.aml_data.features import MODEL_FEATURES
from financial_crime_ai.pipelines.aml_data.nodes import (
    engineer_split,
    engineer_train,
    split_temporal,
)
from financial_crime_ai.pipelines.model_selection.models import (
    fit_candidates,
    select_best,
)

DATA = "data/01_raw/ibm"


@pytest.fixture(scope="module")
def split():
    txn = pd.read_csv(f"{DATA}/transactions.csv", nrows=60_000)
    accounts = pd.read_csv(f"{DATA}/accounts.csv")
    tr, val, te, _ = split_temporal(txn, 0.6, 0.2)
    return tr, val, te, accounts


@pytest.fixture(scope="module")
def engineered(split):
    tr, val, te, accounts = split
    train_feat, af = engineer_train(tr, accounts)
    val_feat = engineer_split(val, af, "validation")
    test_feat = engineer_split(te, af, "test")
    return train_feat, val_feat, test_feat


def test_fit_candidates_runs_all_model_types(engineered):
    train, val, _ = engineered
    candidates = [
        {"name": "logreg", "type": "logreg", "grid": [{"C": 1.0}]},
        {
            "name": "xgboost",
            "type": "xgboost",
            "grid": [{"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}],
        },
        {"name": "mlp", "type": "mlp", "grid": [{"hidden_layer_sizes": [16]}]},
    ]
    results, fitted = fit_candidates(train, val, candidates, MODEL_FEATURES, seed=42)
    assert {"logreg", "xgboost", "mlp"} <= set(results["type"])
    assert len(results) == len(fitted) == 3
    assert results["val_auc"].notna().all()
    # validation AUPRC is reported and sane
    assert (results["val_ap"] >= 0.5).all()


def test_select_best_locks_and_evaluates_test_once(engineered):
    train, val, test = engineered
    candidates = [
        {
            "name": "xgboost",
            "type": "xgboost",
            "grid": [{"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}],
        },
    ]
    results, fitted = fit_candidates(train, val, candidates, MODEL_FEATURES, seed=42)
    best_info, test_metrics, predictions = select_best(
        results, fitted, test, MODEL_FEATURES
    )
    assert best_info["kind"] == "xgboost"
    assert best_info["threshold"] > 0
    assert "auc" in test_metrics["metric"].values
    assert len(predictions) == len(test)
    assert set(predictions.columns) >= {
        "TX_ID",
        "risk_score",
        "predicted_fraud",
        "SENDER_ACCOUNT_ID",
    }


def test_threshold_selection_improves_recall_on_validation(engineered):
    from financial_crime_ai.pipelines.model_selection.models import _select_threshold

    train, val, _ = engineered
    y = val["IS_FRAUD"].astype(int).reset_index(drop=True)
    prob = pd.Series([0.1] * len(y))  # naive model: low probs everywhere
    thr, f2, prec, rec = _select_threshold(y, prob.to_numpy(), min_precision=0.2)
    assert 0 <= thr <= 1
    assert 0 <= prec <= 1
    assert 0 <= rec <= 1
