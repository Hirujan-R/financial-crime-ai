"""Investigation queue and evidence store for the IBM AML data.

The locked model scores the *test-period* transactions (the future the model
was never trained on).  Suspicious predictions become an investigation queue
de-duplicated at the account level, and a structured evidence store is
snapshotted per account for the AI investigator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from financial_crime_ai.pipelines.aml_data.features import GRAPH_FEATURES


def build_account_edges(transactions: pd.DataFrame) -> pd.DataFrame:
    t = transactions[["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT"]]
    edges = (
        t.rename(columns={"SENDER_ACCOUNT_ID": "src", "RECEIVER_ACCOUNT_ID": "dst"})
        .groupby(["src", "dst"], as_index=False)
        .agg(count=("TX_AMOUNT", "size"), value=("TX_AMOUNT", "sum"))
    )
    return edges.sort_values("value", ascending=False).reset_index(drop=True)


def build_investigation_queue(
    test_predictions: pd.DataFrame,
    test_transactions: pd.DataFrame,
    alerts: pd.DataFrame,
    best_model_info: dict,
) -> pd.DataFrame:
    pred = test_predictions[
        ["TX_ID", "risk_score", "predicted_fraud", "IS_FRAUD"]
    ].copy()
    meta = test_transactions[
        ["TX_ID", "SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT", "TIMESTAMP"]
    ].copy()
    pred = pred.merge(meta, on="TX_ID", how="left")

    alert_map = alerts[alerts["ALERT_ID"] >= 0][
        ["TX_ID", "ALERT_TYPE"]
    ].drop_duplicates("TX_ID")
    pred = pred.merge(alert_map, on="TX_ID", how="left")
    pred["ALERT_TYPE"] = pred["ALERT_TYPE"].fillna("none")

    threshold = float(best_model_info["threshold"])
    pred["flagged"] = (pred["risk_score"] >= threshold).astype(int)

    flagged = pred[pred["flagged"] == 1]
    idx = flagged.groupby("SENDER_ACCOUNT_ID")["risk_score"].idxmax()
    queue = (
        flagged.loc[idx]
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
    )

    acct_risk = (
        pred.groupby("SENDER_ACCOUNT_ID")["risk_score"].max().rename("account_max_risk")
    )
    queue["account_max_risk"] = queue["SENDER_ACCOUNT_ID"].map(acct_risk)
    queue["priority"] = pd.cut(
        queue["risk_score"],
        bins=[-0.01, 0.6, 0.8, 1.01],
        labels=["medium", "high", "critical"],
    )
    queue["status"] = "open"
    queue = queue.rename(
        columns={
            "TX_ID": "txn_id",
            "SENDER_ACCOUNT_ID": "account_id",
            "RECEIVER_ACCOUNT_ID": "receiver_account_id",
            "TX_AMOUNT": "amount",
            "TIMESTAMP": "timestamp",
            "IS_FRAUD": "is_fraud",
            "ALERT_TYPE": "alert_type",
        }
    )
    cols = [
        "txn_id",
        "account_id",
        "receiver_account_id",
        "risk_score",
        "predicted_fraud",
        "is_fraud",
        "alert_type",
        "amount",
        "timestamp",
        "account_max_risk",
        "priority",
        "status",
    ]
    return queue[cols]


def build_evidence_store(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
    account_features: pd.DataFrame,
    alerts: pd.DataFrame,
) -> pd.DataFrame:
    acct_feat = account_features.set_index("ACCOUNT_ID")
    graph = acct_feat[GRAPH_FEATURES]

    # involved-account activity (both directions)
    sub = pd.concat(
        [
            transactions[["TX_ID", "SENDER_ACCOUNT_ID", "TX_AMOUNT"]].rename(
                columns={"SENDER_ACCOUNT_ID": "ACCOUNT_ID"}
            ),
            transactions[["TX_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT"]].rename(
                columns={"RECEIVER_ACCOUNT_ID": "ACCOUNT_ID"}
            ),
        ]
    )
    n_tx = sub.groupby("ACCOUNT_ID")["TX_ID"].nunique()
    total = sub.groupby("ACCOUNT_ID")["TX_AMOUNT"].sum()
    avg = sub.groupby("ACCOUNT_ID")["TX_AMOUNT"].mean()
    mx = sub.groupby("ACCOUNT_ID")["TX_AMOUNT"].max()

    sent = transactions.groupby("SENDER_ACCOUNT_ID")["TX_AMOUNT"].sum()
    received = transactions.groupby("RECEIVER_ACCOUNT_ID")["TX_AMOUNT"].sum()
    n_sent = transactions.groupby("SENDER_ACCOUNT_ID")["TX_ID"].nunique()
    n_recv = transactions.groupby("RECEIVER_ACCOUNT_ID")["TX_ID"].nunique()

    alert_accounts = set(alerts[alerts["ALERT_ID"] >= 0]["SENDER_ACCOUNT_ID"])
    alert_accounts |= set(alerts[alerts["ALERT_ID"] >= 0]["RECEIVER_ACCOUNT_ID"])

    acct = accounts.set_index("ACCOUNT_ID")
    acc = pd.DataFrame(
        {
            "account_id": accounts["ACCOUNT_ID"],
            "n_transactions": accounts["ACCOUNT_ID"].map(n_tx).fillna(0).astype(int),
            "total_value_moved": accounts["ACCOUNT_ID"].map(total).fillna(0.0),
            "avg_amount": accounts["ACCOUNT_ID"].map(avg).fillna(0.0),
            "max_amount": accounts["ACCOUNT_ID"].map(mx).fillna(0.0),
            "total_sent": accounts["ACCOUNT_ID"].map(sent).fillna(0.0),
            "total_received": accounts["ACCOUNT_ID"].map(received).fillna(0.0),
            "n_outgoing": accounts["ACCOUNT_ID"].map(n_sent).fillna(0).astype(int),
            "n_incoming": accounts["ACCOUNT_ID"].map(n_recv).fillna(0).astype(int),
            "init_balance_log": accounts["ACCOUNT_ID"]
            .map(acct["INIT_BALANCE"])
            .apply(lambda x: float(np.log1p(x or 0))),
            "behavior_id": accounts["ACCOUNT_ID"].map(acct["TX_BEHAVIOR_ID"]),
            "is_fraud_account": accounts["ACCOUNT_ID"]
            .map(acct["IS_FRAUD"])
            .astype(bool),
            "in_known_alert": accounts["ACCOUNT_ID"].isin(alert_accounts).astype(int),
        }
    )
    for c in GRAPH_FEATURES:
        acc[f"graph_{c}"] = accounts["ACCOUNT_ID"].map(graph[c]).fillna(0.0)
    return acc


def build_similar_case_index(
    account_features: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> dict[str, object]:
    feat = account_features.set_index("ACCOUNT_ID")
    risk = (
        test_predictions.groupby("SENDER_ACCOUNT_ID")["risk_score"]
        .max()
        .reindex(feat.index)
        .fillna(0.0)
    )
    X = feat[GRAPH_FEATURES].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return {
        "account_ids": list(X.index),
        "X_scaled": X_scaled,
        "scaler": scaler,
        "risk": risk.to_dict(),
    }
