"""Model-agnostic feature attribution.

Explains *why* a transaction was flagged by attributing the risk score to
features, using whatever model won the selection step:

* XGBoost  -> SHAP-style per-feature contributions from the booster
* LogReg   -> exact linear attribution  coef_i * scaled(x_i)
* MLP      -> local perturbation sensitivity around the transaction

Each attribution is translated into plain language for the analyst.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_crime_ai.features import FEATURE_LABELS

_CLAUSES: dict[str, str] = {
    "tx_amount": "the transaction's absolute value",
    "tx_amount_log": "the log-scaled transaction value",
    "amount_vs_sender_median": "the transaction's size relative to the sender's median transfer",
    "timestamp": "the time bucket of the transaction",
    "sender_activity_count": "the sender's cumulative transfer count up to this point",
    "sender_activity_total": "the sender's cumulative value moved up to this point",
    "receiver_activity_count": "the receiver's cumulative transfer count up to this point",
    "receiver_activity_total": "the receiver's cumulative value received up to this point",
    "sender_degree_out": "the number of distinct recipients the sender pays",
    "sender_degree_in": "the number of distinct payers feeding the sender",
    "sender_sum_out": "the sender's total outward value",
    "sender_sum_in": "the sender's total inward value",
    "sender_net_flow": "the sender's net flow (in minus out)",
    "sender_mean_out": "the sender's average outgoing transfer",
    "sender_mean_in": "the sender's average incoming transfer",
    "sender_pagerank": "the sender's centrality in the payment graph",
    "sender_hub_score": "how hub-like the sender is in the network",
    "sender_kcore_number": "how deeply the sender sits in a dense subgraph",
    "sender_community_size": "the size of the sender's community",
    "receiver_degree_out": "the number of distinct recipients the receiver pays",
    "receiver_degree_in": "the number of distinct payers concentrating into the receiver",
    "receiver_sum_out": "the receiver's total outward value",
    "receiver_sum_in": "the receiver's total inward value",
    "receiver_net_flow": "the receiver's net flow (in minus out)",
    "receiver_mean_out": "the receiver's average outgoing transfer",
    "receiver_mean_in": "the receiver's average incoming transfer",
    "receiver_pagerank": "the receiver's centrality in the payment graph",
    "receiver_hub_score": "how hub-like the receiver is in the network",
    "receiver_kcore_number": "how deeply the receiver sits in a dense subgraph",
    "receiver_community_size": "the size of the receiver's community",
    "init_balance_log": "the account's opening balance",
    "behavior_1": "the account's behaviour-profile encoding",
    "behavior_2": "the account's behaviour-profile encoding",
    "behavior_3": "the account's behaviour-profile encoding",
    "behavior_4": "the account's behaviour-profile encoding",
    "behavior_5": "the account's behaviour-profile encoding",
    "sender_init_balance_log": "the sender's opening balance",
    "sender_behavior_1": "the sender's behaviour-profile encoding",
    "sender_behavior_2": "the sender's behaviour-profile encoding",
    "sender_behavior_3": "the sender's behaviour-profile encoding",
    "sender_behavior_4": "the sender's behaviour-profile encoding",
    "sender_behavior_5": "the sender's behaviour-profile encoding",
}


def compute_contributions(
    best_model_info: dict,
    feature_frame: pd.DataFrame,
    txn_id: str,
    top_n: int = 8,
) -> list[dict]:
    """Return the top-N attributed features for a single transaction."""
    frame = feature_frame.copy()
    frame["TX_ID"] = frame["TX_ID"].astype(str)
    row = frame[frame["TX_ID"] == str(txn_id)]
    if row.empty:
        return []
    feat_cols = list(best_model_info["feature_columns"])
    X = row[feat_cols].fillna(0.0)

    model = best_model_info["model"]
    kind = best_model_info["kind"]

    if kind == "xgboost":
        contribs = _tree_contributions(model, X)
    elif kind == "logreg":
        contribs = _linear_contributions(model, best_model_info["scaler"], X)
    else:
        contribs = _perturbation_contributions(
            model, best_model_info["scaler"], X, feat_cols
        )

    values = X.iloc[0].to_dict()
    out = []
    for feat, c in contribs.items():
        out.append(
            {
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "value": float(values.get(feat, 0.0)),
                "contribution": float(c),
            }
        )
    out.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return out[:top_n]


def _tree_contributions(model, X: pd.DataFrame) -> dict[str, float]:
    import xgboost as xgb

    booster = model.get_booster()
    contribs = booster.predict(
        xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True
    )[0]
    return {f: float(contribs[i]) for i, f in enumerate(X.columns)}


def _transform(X: pd.DataFrame, scaler) -> np.ndarray:
    return scaler.transform(X) if scaler is not None else X.to_numpy()


def _linear_contributions(model, scaler, X: pd.DataFrame) -> dict[str, float]:
    X_s = _transform(X, scaler)[0]
    coef = model.coef_[0]
    return {f: float(c * x) for f, c, x in zip(X.columns, coef, X_s)}


def _perturbation_contributions(
    model, scaler, X: pd.DataFrame, feat_cols: list[str]
) -> dict[str, float]:
    base = X.iloc[0].to_numpy().reshape(1, -1)
    p_base = model.predict_proba(_transform(X, scaler))[0, 1]
    contribs: dict[str, float] = {}
    median = X.median().to_numpy()
    for i, f in enumerate(feat_cols):
        perturbed = base.copy()
        perturbed[0, i] = median[i]
        p_pert = model.predict_proba(
            _transform(pd.DataFrame(perturbed, columns=X.columns), scaler)
        )[0, 1]
        contribs[f] = float(p_base - p_pert)
    return contribs


def why_flagged_messages(contributions: list[dict]) -> list[str]:
    """Human-readable reasons from the top contributions."""
    messages = []
    for c in contributions:
        clause = _CLAUSES.get(c["feature"])
        if clause is None:
            continue
        direction = "increases" if c["contribution"] > 0 else "decreases"
        messages.append(
            f"{c['label']} ({c['value']:.2f}) — {clause}, which "
            f"{direction} the risk score."
        )
    return messages
