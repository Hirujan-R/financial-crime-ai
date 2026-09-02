"""Tests for the AMLSim scenario generator and generalisation experiment."""

import pytest

from financial_crime_ai.pipelines.aml_data.features import MODEL_FEATURES
from financial_crime_ai.pipelines.aml_data.nodes import engineer_train
from financial_crime_ai.pipelines.aml_sim.generator import generate_amlsim_data
from financial_crime_ai.pipelines.aml_sim.nodes import evaluate_generalisation


@pytest.fixture(scope="module")
def scenarios():
    return generate_amlsim_data(
        n_accounts=800, n_timestamps=40, seed=7, normal_rate=0.4
    )


def test_domains_are_separated(scenarios):
    train = scenarios["train_transactions"]
    test = scenarios["test_transactions"]
    # normal background traffic ('none') legitimately appears in both domains
    train_types = set(train["ALERT_TYPE"]) - {"none"}
    test_types = set(test["ALERT_TYPE"]) - {"none"}
    assert {"structuring", "layering"} <= train_types
    assert {"cycle", "fan_out", "fan_in"} <= test_types
    assert train_types.isdisjoint(test_types)


def test_generalisation_metrics_per_typology(scenarios):
    # tiny training to keep the test fast
    train_txn = scenarios["train_transactions"]
    test_txn = scenarios["test_transactions"]
    accounts = scenarios["accounts"]

    train_feat, af = engineer_train(train_txn.head(10_000), accounts)
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=20, max_depth=3, learning_rate=0.1, n_jobs=-1, random_state=1
    )
    y = train_txn["IS_FRAUD"].astype(int).head(10_000).to_numpy()
    model.fit(train_feat.head(10_000)[MODEL_FEATURES], y)

    info = {
        "model": model,
        "kind": "xgboost",
        "scaler": None,
        "threshold": 0.5,
        "feature_columns": MODEL_FEATURES,
    }
    metrics, _ = evaluate_generalisation(
        train_txn.head(10_000), test_txn, accounts, info, seed=1
    )
    types = set(metrics["alert_type"])
    assert {"cycle", "fan_out", "fan_in", "overall"} <= types
    overall = metrics[metrics["alert_type"] == "overall"].iloc[0]
    assert overall["auc"] > 0.5
