"""Tests for IBM AML data loading, feature engineering and temporal split."""

import pandas as pd
import pytest

from financial_crime_ai.pipelines.aml_data.features import (
    GRAPH_FEATURES,
    MODEL_FEATURES,
    build_train_account_features,
    build_transaction_features,
)
from financial_crime_ai.pipelines.aml_data.nodes import (
    engineer_split,
    engineer_train,
    split_temporal,
)

DATA = "data/01_raw/ibm"


@pytest.fixture(scope="module")
def raw():
    txn = pd.read_csv(f"{DATA}/transactions.csv", nrows=30_000)
    accounts = pd.read_csv(f"{DATA}/accounts.csv")
    return txn, accounts


@pytest.fixture(scope="module")
def features(raw):
    txn, accounts = raw
    af = build_train_account_features(txn, accounts)
    f = build_transaction_features(txn, af)
    return f, af


def test_temporal_split_is_chronological(raw):
    txn, _ = raw
    train, val, test, bounds = split_temporal(txn, 0.7, 0.15)
    assert len(train) + len(val) + len(test) == len(txn)
    assert train["TIMESTAMP"].max() <= val["TIMESTAMP"].min()
    assert val["TIMESTAMP"].max() <= test["TIMESTAMP"].min()
    assert bounds["n_train"] == len(train)
    assert bounds["n_test"] == len(test)


def test_train_account_features_shape(features, raw):
    _, accounts = raw
    f, af = features
    assert set(GRAPH_FEATURES) <= set(af.columns)
    # only accounts present in the transaction sample get structure
    assert len(af) <= len(accounts)


def test_transaction_features_no_nan(features):
    f, _ = features
    missing = [c for c in MODEL_FEATURES if c not in f.columns]
    assert not missing
    assert f[MODEL_FEATURES].isna().sum().sum() == 0


def test_engineer_frames_carry_labels_and_ids(raw):
    txn, accounts = raw
    tr, val, te, _ = split_temporal(txn, 0.6, 0.2)
    train_feat, af = engineer_train(tr, accounts)
    val_feat = engineer_split(val, af, "validation")
    for frame in (train_feat, val_feat):
        assert "IS_FRAUD" in frame.columns
        assert "TX_ID" in frame.columns
        assert "SENDER_ACCOUNT_ID" in frame.columns
