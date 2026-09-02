"""Shared feature definitions used across pipelines.

Centralises the feature names used for anomaly detection, risk scoring and
the AI investigator so that every stage references the same columns. Also
exposes human-readable descriptions used to explain a case to an analyst.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared reference data (also used by the data generator)
# ---------------------------------------------------------------------------
COUNTRY_RISK = {
    # low risk (score 1)
    "US": 1,
    "GB": 1,
    "DE": 1,
    "FR": 1,
    "CA": 1,
    "JP": 1,
    "AU": 1,
    "SG": 1,
    "CH": 1,
    "NL": 1,
    "SE": 1,
    "NO": 1,
    "DK": 1,
    "FI": 1,
    "IE": 1,
    "NZ": 1,
    "KR": 1,
    "BE": 1,
    "AT": 1,
    "LU": 1,
    # medium risk (score 2)
    "BR": 2,
    "IN": 2,
    "MX": 2,
    "PL": 2,
    "CZ": 2,
    "PT": 2,
    "ES": 2,
    "IT": 2,
    "GR": 2,
    "TR": 2,
    "TH": 2,
    "MY": 2,
    "ID": 2,
    "PH": 2,
    "AE": 2,
    "SA": 2,
    "IL": 2,
    "ZA": 2,
    "AR": 2,
    "CL": 2,
    "HU": 2,
    "RO": 2,
    "UA": 2,
    "EG": 2,
    # high risk (score 3)
    "RU": 3,
    "CN": 3,
    "HK": 3,
    "PA": 3,
    "KY": 3,
    "BZ": 3,
    "VU": 3,
    "MM": 3,
    "IR": 3,
    "KP": 3,
    "SY": 3,
    "CU": 3,
    "VE": 3,
    "NG": 3,
    "BO": 3,
    "LB": 3,
    "PK": 3,
    "AF": 3,
    "SO": 3,
    "YE": 3,
    "BY": 3,
    "ZW": 3,
}

LOW_RISK = [c for c, r in COUNTRY_RISK.items() if r == 1]
MED_RISK = [c for c, r in COUNTRY_RISK.items() if r == 2]
HIGH_RISK = [c for c, r in COUNTRY_RISK.items() if r == 3]

CHANNELS = {"CASH": 0.9, "WIRE": 0.6, "ACH": 0.2, "CARD": 0.15, "CRYPTO": 0.85}
CHANNEL_PRIORS = ["ACH", "ACH", "CARD", "CARD", "CARD", "WIRE", "CASH", "CRYPTO"]

REPORTING_THRESHOLDS = (5_000.0, 10_000.0, 30_000.0, 100_000.0)

# ---------------------------------------------------------------------------
# Transaction-level features
# ---------------------------------------------------------------------------
TXN_FEATURES = [
    "amount_log",
    "amount_zscore",
    "amount_vs_median",
    "is_round",
    "round_proximity",
    "threshold_proximity",
    "hour_of_day",
    "is_weekend",
    "channel_risk",
    "country_risk",
    "is_foreign",
    "counterparty_high_risk",
    "counterparty_sanctioned",
    "account_age_days",
    "txn_count_24h",
    "txn_amount_24h",
    "txn_count_7d",
    "txn_amount_7d",
    "time_since_prev_h",
    "prev_txn_amount",
]

# ---------------------------------------------------------------------------
# Graph features (account-level, broadcast onto every transaction)
# ---------------------------------------------------------------------------
GRAPH_FEATURES = [
    "degree_in",
    "degree_out",
    "n_unique_counterparties",
    "sum_in",
    "sum_out",
    "net_flow",
    "mean_in",
    "mean_out",
    "max_txn",
    "fan_in",
    "fan_out",
    "pagerank",
    "hub_score",
    "kcore_number",
    "community_size",
    "min_hops_to_sanctioned",
    "amount_to_sanctioned",
    "frac_flow_high_risk",
    "structuring_score",
    "layering_score",
    "fresh_account",
    "embedding_0",
    "embedding_1",
    "embedding_2",
    "embedding_3",
    "embedding_4",
    "embedding_5",
    "embedding_6",
    "embedding_7",
]

# ---------------------------------------------------------------------------
# Full model feature set (transaction + graph)
# ---------------------------------------------------------------------------
MODEL_FEATURES = TXN_FEATURES + GRAPH_FEATURES

# ---------------------------------------------------------------------------
# Features that the anomaly baselines consume
# ---------------------------------------------------------------------------
ANOMALY_FEATURES = [
    "amount_log",
    "amount_zscore",
    "amount_vs_median",
    "is_round",
    "threshold_proximity",
    "channel_risk",
    "country_risk",
    "txn_count_24h",
    "txn_amount_24h",
    "txn_count_7d",
    "txn_amount_7d",
    "time_since_prev_h",
    "degree_in",
    "degree_out",
    "min_hops_to_sanctioned",
    "frac_flow_high_risk",
    "structuring_score",
    "layering_score",
]

# ---------------------------------------------------------------------------
# Risk model inputs = all anomaly outputs + graph features
# ---------------------------------------------------------------------------
RISK_FEATURES = (
    [
        "isolation_score",
        "xgboost_prob",
    ]
    + GRAPH_FEATURES
    + [
        "amount_log",
        "threshold_proximity",
        "country_risk",
        "txn_count_7d",
        "txn_amount_7d",
        "time_since_prev_h",
    ]
)

# ---------------------------------------------------------------------------
# Human-readable descriptions used by the explainer / investigator
# ---------------------------------------------------------------------------
FEATURE_LABELS: dict[str, str] = {
    "amount_log": "Log transaction amount",
    "amount_zscore": "Amount deviation from the account's own history",
    "amount_vs_median": "Amount relative to the account's median transaction",
    "is_round": "Transaction amount is round (multiple of 100)",
    "round_proximity": "How close the amount is to a round figure",
    "threshold_proximity": "Proximity to common reporting thresholds (e.g. $10k)",
    "hour_of_day": "Hour of day the transaction occurred",
    "is_weekend": "Transaction occurred on a weekend",
    "channel_risk": "Risk weight of the payment channel",
    "country_risk": "Risk score of the counterparty country",
    "is_foreign": "Transaction crosses into a foreign jurisdiction",
    "counterparty_high_risk": "Counterparty is in a high-risk jurisdiction",
    "counterparty_sanctioned": "Counterparty is on a sanctions/watchlist",
    "account_age_days": "Age of the account in days",
    "txn_count_24h": "Number of transactions on the account in the last 24h",
    "txn_amount_24h": "Total amount moved on the account in the last 24h",
    "txn_count_7d": "Number of transactions on the account in the last 7 days",
    "txn_amount_7d": "Total amount moved on the account in the last 7 days",
    "time_since_prev_h": "Hours since the account's previous transaction",
    "prev_txn_amount": "Previous transaction amount on the account",
    "degree_in": "Number of distinct senders to this account",
    "degree_out": "Number of distinct receivers from this account",
    "n_unique_counterparties": "Distinct counterparties transacting with the account",
    "sum_in": "Total value received by the account",
    "sum_out": "Total value sent by the account",
    "net_flow": "Net value flow (in minus out)",
    "mean_in": "Average incoming transaction value",
    "mean_out": "Average outgoing transaction value",
    "max_txn": "Largest single transaction on the account",
    "fan_in": "Senders per unit of outbound activity",
    "fan_out": "Receivers per unit of inbound activity",
    "pagerank": "PageRank of the account in the transaction graph",
    "hub_score": "HITS hub score of the account",
    "kcore_number": "K-core number of the account in the transaction graph",
    "community_size": "Size of the account's community",
    "min_hops_to_sanctioned": "Shortest path to a sanctioned entity",
    "amount_to_sanctioned": "Total value sent to sanctioned entities",
    "frac_flow_high_risk": "Share of funds flowing to high-risk jurisdictions",
    "structuring_score": "Evidence of structuring (deposits just under thresholds)",
    "layering_score": "Evidence of layering (rapid pass-through transfers)",
    "fresh_account": "Account opened recently (fresh/aged)",
    "embedding_0": "Structural embedding dimension 0",
    "embedding_1": "Structural embedding dimension 1",
    "embedding_2": "Structural embedding dimension 2",
    "embedding_3": "Structural embedding dimension 3",
    "embedding_4": "Structural embedding dimension 4",
    "embedding_5": "Structural embedding dimension 5",
    "embedding_6": "Structural embedding dimension 6",
    "embedding_7": "Structural embedding dimension 7",
    "isolation_score": "Isolation Forest anomaly score",
    "xgboost_prob": "Gradient-boosted anomaly probability",
    # ------------------------------------------------------------------
    # IBM AML / AMLSim feature labels (aml_data.features.MODEL_FEATURES)
    # ------------------------------------------------------------------
    "tx_amount": "Transaction amount",
    "tx_amount_log": "Log transaction amount",
    "amount_vs_sender_median": "Amount relative to the sender's median transfer",
    "timestamp": "Time bucket",
    "sender_activity_count": "Sender's cumulative transfer count (to this time bucket)",
    "sender_activity_total": "Sender's cumulative value moved (to this time bucket)",
    "receiver_activity_count": "Receiver's cumulative transfer count (to this time bucket)",
    "receiver_activity_total": "Receiver's cumulative value received (to this time bucket)",
    "sender_degree_out": "Distinct recipients the sender pays",
    "sender_degree_in": "Distinct payers into the sender",
    "sender_sum_out": "Total value the sender paid out",
    "sender_sum_in": "Total value the sender received",
    "sender_net_flow": "Sender's net flow (in minus out)",
    "sender_mean_out": "Sender's average outgoing transfer",
    "sender_mean_in": "Sender's average incoming transfer",
    "sender_pagerank": "Sender's PageRank in the payment graph",
    "sender_hub_score": "Sender's hub score in the payment graph",
    "sender_kcore_number": "Sender's k-core membership",
    "sender_community_size": "Sender's community size",
    "receiver_degree_out": "Distinct recipients the receiver pays",
    "receiver_degree_in": "Distinct payers into the receiver",
    "receiver_sum_out": "Total value the receiver paid out",
    "receiver_sum_in": "Total value the receiver received",
    "receiver_net_flow": "Receiver's net flow (in minus out)",
    "receiver_mean_out": "Receiver's average outgoing transfer",
    "receiver_mean_in": "Receiver's average incoming transfer",
    "receiver_pagerank": "Receiver's PageRank in the payment graph",
    "receiver_hub_score": "Receiver's hub score in the payment graph",
    "receiver_kcore_number": "Receiver's k-core membership",
    "receiver_community_size": "Receiver's community size",
    "init_balance_log": "Log opening balance of the account",
    "behavior_1": "Account behaviour type 1",
    "behavior_2": "Account behaviour type 2",
    "behavior_3": "Account behaviour type 3",
    "behavior_4": "Account behaviour type 4",
    "behavior_5": "Account behaviour type 5",
    "sender_init_balance_log": "Log opening balance of the sender",
    "sender_behavior_1": "Sender behaviour type 1",
    "sender_behavior_2": "Sender behaviour type 2",
    "sender_behavior_3": "Sender behaviour type 3",
    "sender_behavior_4": "Sender behaviour type 4",
    "sender_behavior_5": "Sender behaviour type 5",
}

# ---------------------------------------------------------------------------
# AML typology knowledge base used by the RAG component
# ---------------------------------------------------------------------------
TYPOLOGIES: list[dict[str, str]] = [
    {
        "id": "TYP-001",
        "name": "Structuring (Smurfing)",
        "summary": "Funds broken into smaller deposits to stay below reporting thresholds.",
        "indicators": "Repeated deposits just under $10k, many distinct senders, "
        "consolidation into a single account before onward transfer.",
        "recommendation": "File a Suspicious Activity Report (SAR), freeze the "
        "consolidation account, and map the deposit network.",
        "models": "detected_by: structuring_score, threshold_proximity",
    },
    {
        "id": "TYP-002",
        "name": "Layering (Mule chains)",
        "summary": "Funds moved rapidly through a chain of accounts to obscure origin.",
        "indicators": "Quick successive transfers across several fresh accounts with "
        "near-identical amounts, then an exit to a high-risk or offshore destination.",
        "recommendation": "Trace the full chain, freeze all mule accounts, and notify "
        "the receiving bank.",
        "models": "detected_by: layering_score, min_hops_to_sanctioned",
    },
    {
        "id": "TYP-003",
        "name": "Rapid cash-out / structuring out",
        "summary": "Large credit followed almost immediately by cash-out or offshore wire.",
        "indicators": "Fresh account receives a large lump sum then wires funds out "
        "within hours to a high-risk jurisdiction.",
        "recommendation": "Verify source of funds, consider freezing on suspicion, "
        "elevate account risk tier.",
        "models": "detected_by: amount_zscore, time_since_prev_h, account_age_days",
    },
    {
        "id": "TYP-004",
        "name": "Gambling and high-velocity churn",
        "summary": "Very high transaction velocity to gambling or similar merchants.",
        "indicators": "Large number of small-to-medium transactions over short windows.",
        "recommendation": "Review against gambling policy, monitor for funding from "
        "suspicious sources.",
        "models": "detected_by: txn_count_24h, txn_count_7d",
    },
    {
        "id": "TYP-005",
        "name": "Round-amount bulk transfers",
        "summary": "Frequent transfers at round amounts that may mask underlying flows.",
        "indicators": "Repeated amounts like 4,950 / 9,900 / 99,000 across many accounts.",
        "recommendation": "Correlate accounts sharing the same round amounts.",
        "models": "detected_by: is_round, round_proximity",
    },
    {
        "id": "TYP-006",
        "name": "Sanctions and high-risk jurisdiction flows",
        "summary": "Funds flowing to or from sanctioned entities / high-risk countries.",
        "indicators": "Counterparty on sanctions or watchlist, short path from a "
        "sanctioned node in the transaction graph.",
        "recommendation": "Escalate immediately, block the transaction, file report.",
        "models": "detected_by: min_hops_to_sanctioned, counterparty_sanctioned",
    },
    {
        "id": "TYP-007",
        "name": "Circular transfers (cycle)",
        "summary": "Funds move in a closed loop A->B->C->A to give the illusion of "
        "legitimate economic activity while returning value to the origin.",
        "indicators": "Transactions forming a short directed cycle with near-identical "
        "amounts and a rapid turnaround.",
        "recommendation": "Trace the loop members, verify the economic purpose of the "
        "circular flows, and consider freezing the ring.",
        "models": "detected_by: sender_degree_out, sender_net_flow, receiver_degree_in",
    },
    {
        "id": "TYP-008",
        "name": "Fan-out dispersal",
        "summary": "A single source disperses funds to many recipients, often to "
        "circumvent per-recipient limits or to seed mule accounts.",
        "indicators": "One sender with a high out-degree and many near-simultaneous "
        "outgoing transfers.",
        "recommendation": "Map all recipients, freeze the source account and place "
        "recipients under monitoring.",
        "models": "detected_by: sender_degree_out, sender_activity_count, sender_fan_out",
    },
    {
        "id": "TYP-009",
        "name": "Fan-in concentration",
        "summary": "Many senders concentrate value into a single account before onward "
        "movement - a classic collection point.",
        "indicators": "One receiver with a high in-degree drawing from many distinct "
        "sources in a short window.",
        "recommendation": "Trace the funding sources, freeze the collection account, "
        "and file a SAR.",
        "models": "detected_by: receiver_degree_in, receiver_activity_total, receiver_net_flow",
    },
]
