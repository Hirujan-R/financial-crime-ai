"""Kedro nodes for the IBM AML data pipeline."""

from __future__ import annotations

import pandas as pd

from .features import (
    MODEL_FEATURES,
    build_train_account_features,
    build_transaction_features,
)


def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["IS_FRAUD"] = df["IS_FRAUD"].astype(bool)
    return df


def load_accounts(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_alerts(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_temporal(
    transactions: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Chronological 70/15/15 split on TIMESTAMP (no random shuffling)."""
    ordered = transactions.sort_values("TIMESTAMP").reset_index(drop=True)
    n = len(ordered)
    cut1 = int(n * train_fraction)
    cut2 = int(n * (train_fraction + val_fraction))
    train = ordered.iloc[:cut1]
    val = ordered.iloc[cut1:cut2]
    test = ordered.iloc[cut2:]
    boundaries = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "train_max_timestamp": int(train["TIMESTAMP"].max()),
        "val_max_timestamp": int(val["TIMESTAMP"].max()),
        "test_max_timestamp": int(test["TIMESTAMP"].max()),
        "train_fraud": int(train["IS_FRAUD"].sum()),
        "val_fraud": int(val["IS_FRAUD"].sum()),
        "test_fraud": int(test["IS_FRAUD"].sum()),
    }
    return train, val, test, boundaries


def engineer_train(
    train_transactions: pd.DataFrame,
    accounts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build train-scope account features, then the train feature matrix."""
    account_features = build_train_account_features(train_transactions, accounts)
    features = build_transaction_features(train_transactions, account_features)
    meta_cols = ["IS_FRAUD", "TX_ID", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"]
    for c in meta_cols:
        features[c] = train_transactions[c].reset_index(drop=True).values
    return features, account_features


def engineer_split(
    split_transactions: pd.DataFrame,
    account_features: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    features = build_transaction_features(split_transactions, account_features)
    meta_cols = ["IS_FRAUD", "TX_ID", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID"]
    for c in meta_cols:
        features[c] = split_transactions[c].reset_index(drop=True).values
    features["split"] = label
    return features


def feature_columns() -> list[str]:
    return MODEL_FEATURES
