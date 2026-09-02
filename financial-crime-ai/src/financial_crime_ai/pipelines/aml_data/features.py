"""Feature engineering for the IBM AML data (and AMLSim scenarios).

Works on the standard IBM AMLSim schema:

``transactions``: SENDER_ACCOUNT_ID, RECEIVER_ACCOUNT_ID, TX_AMOUNT, TIMESTAMP
``accounts``:     ACCOUNT_ID, INIT_BALANCE, TX_BEHAVIOR_ID, ...

Leakage discipline
------------------
Graph features are always computed from a *train-scope* set of transactions
(the historical period the model is allowed to see) and then broadcast onto
any later transaction.  ``account_features`` is therefore built once from the
training split and reused for validation/test/scenario scoring.  Velocity
features are cumulative up to the transaction's own time bucket, so they are
causal.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

# Transaction + account behavioural features
WINDOW_FEATURES = [
    "sender_activity_count",
    "sender_activity_total",
    "receiver_activity_count",
    "receiver_activity_total",
]

# Graph features computed per account
GRAPH_FEATURES = [
    "degree_out",
    "degree_in",
    "sum_out",
    "sum_in",
    "net_flow",
    "mean_out",
    "mean_in",
    "pagerank",
    "hub_score",
    "kcore_number",
    "community_size",
]

ACCOUNT_FEATURES = [
    "init_balance_log",
    "behavior_1",
    "behavior_2",
    "behavior_3",
    "behavior_4",
    "behavior_5",
]

MODEL_FEATURES = (
    ["tx_amount", "tx_amount_log", "amount_vs_sender_median", "timestamp"]
    + WINDOW_FEATURES
    + [f"sender_{f}" for f in GRAPH_FEATURES]
    + [f"receiver_{f}" for f in GRAPH_FEATURES]
    + [f"sender_{f}" for f in ACCOUNT_FEATURES]
)


def build_train_account_features(
    transactions: pd.DataFrame,
    accounts: pd.DataFrame,
) -> pd.DataFrame:
    """Account-level features from the train-scope transaction graph only."""
    t = transactions[["SENDER_ACCOUNT_ID", "RECEIVER_ACCOUNT_ID", "TX_AMOUNT"]]
    t = t[t["SENDER_ACCOUNT_ID"] != t["RECEIVER_ACCOUNT_ID"]]
    edges = (
        t.rename(columns={"SENDER_ACCOUNT_ID": "src", "RECEIVER_ACCOUNT_ID": "dst"})
        .groupby(["src", "dst"], as_index=False)
        .agg(count=("TX_AMOUNT", "size"), value=("TX_AMOUNT", "sum"))
    )
    G = nx.DiGraph()
    G.add_weighted_edges_from((r.src, r.dst, r.value) for r in edges.itertuples())

    pagerank = nx.pagerank(G, weight="value")
    try:
        hub, _ = nx.hits(G, max_iter=100, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        hub = dict.fromkeys(G.nodes(), 0.0)
    core = nx.core_number(G.to_undirected())
    communities = {
        node: i
        for i, comm in enumerate(
            nx.community.label_propagation_communities(G.to_undirected())
        )
        for node in comm
    }
    comm_size = _community_sizes(communities)

    out_g = t.groupby("SENDER_ACCOUNT_ID")
    in_g = t.groupby("RECEIVER_ACCOUNT_ID")

    feat = pd.DataFrame(index=G.nodes())
    feat["degree_out"] = out_g["RECEIVER_ACCOUNT_ID"].nunique()
    feat["degree_in"] = in_g["SENDER_ACCOUNT_ID"].nunique()
    feat["sum_out"] = out_g["TX_AMOUNT"].sum()
    feat["sum_in"] = in_g["TX_AMOUNT"].sum()
    feat["net_flow"] = feat["sum_in"] - feat["sum_out"]
    feat["mean_out"] = out_g["TX_AMOUNT"].mean()
    feat["mean_in"] = in_g["TX_AMOUNT"].mean()
    feat["pagerank"] = feat.index.map(pagerank)
    feat["hub_score"] = feat.index.map(hub)
    feat["kcore_number"] = feat.index.map(core).fillna(0)
    feat["community_size"] = feat.index.map(comm_size).fillna(1)

    # account attributes
    acct = accounts.set_index("ACCOUNT_ID")
    feat["init_balance_log"] = np.log1p(acct["INIT_BALANCE"])
    for i in range(1, 6):
        feat[f"behavior_{i}"] = (acct["TX_BEHAVIOR_ID"] == i).astype(float)
    feat = feat.fillna(0.0)

    for col in GRAPH_FEATURES:
        feat[col] = feat[col].astype(float)
    return feat.reset_index().rename(columns={"index": "ACCOUNT_ID"})


def build_transaction_features(
    transactions: pd.DataFrame,
    account_features: pd.DataFrame,
) -> pd.DataFrame:
    """Feature matrix for a set of transactions, using precomputed graph features."""
    t = transactions.copy()
    t["tx_amount"] = t["TX_AMOUNT"].astype(float)
    t["tx_amount_log"] = np.log1p(t["tx_amount"])
    t["timestamp"] = t["TIMESTAMP"].astype(float)

    acct = account_features.set_index("ACCOUNT_ID")

    # sender/receiver graph features (broadcast from the train graph)
    for col in GRAPH_FEATURES + ACCOUNT_FEATURES:
        t[f"sender_{col}"] = t["SENDER_ACCOUNT_ID"].map(acct[col])
        t[f"receiver_{col}"] = t["RECEIVER_ACCOUNT_ID"].map(acct[col])
    # accounts unseen in the train graph default to zero structure
    t = t.fillna(0.0)

    # cumulative activity windows per account (causal: up to the tx time bucket)
    t = t.merge(
        _account_accumulation(transactions).rename(
            columns={
                "cum_count": "sender_activity_count",
                "cum_total": "sender_activity_total",
            }
        ),
        left_on=["SENDER_ACCOUNT_ID", "TIMESTAMP"],
        right_on=["account_id", "TIMESTAMP"],
        how="left",
    )
    t = t.merge(
        _account_accumulation(transactions, receiver=True).rename(
            columns={
                "cum_count": "receiver_activity_count",
                "cum_total": "receiver_activity_total",
            }
        ),
        left_on=["RECEIVER_ACCOUNT_ID", "TIMESTAMP"],
        right_on=["account_id", "TIMESTAMP"],
        how="left",
    )

    # amount relative to the sender's median historical amount
    median = (
        transactions.groupby("SENDER_ACCOUNT_ID")["TX_AMOUNT"].median().rename("_med")
    )
    t["amount_vs_sender_median"] = t["tx_amount"] / t["SENDER_ACCOUNT_ID"].map(
        median
    ).clip(lower=1e-6)

    keep = MODEL_FEATURES
    return t[keep].astype(float).fillna(0.0)


def _account_accumulation(
    transactions: pd.DataFrame, receiver: bool = False
) -> pd.DataFrame:
    """Per (account, timestamp) cumulative count and value of activity."""
    col = "RECEIVER_ACCOUNT_ID" if receiver else "SENDER_ACCOUNT_ID"
    sub = transactions[["TX_ID", col, "TIMESTAMP", "TX_AMOUNT"]].rename(
        columns={col: "account_id"}
    )
    g = (
        sub.groupby(["account_id", "TIMESTAMP"], as_index=False)
        .agg(count=("TX_ID", "size"), total=("TX_AMOUNT", "sum"))
        .sort_values(["account_id", "TIMESTAMP"])
    )
    g["cum_count"] = g.groupby("account_id")["count"].cumsum()
    g["cum_total"] = g.groupby("account_id")["total"].cumsum()
    return g[["account_id", "TIMESTAMP", "cum_count", "cum_total"]]


def _community_sizes(communities: dict) -> dict:
    from collections import Counter

    size = Counter(communities.values())
    return {node: size[c] for node, c in communities.items()}
