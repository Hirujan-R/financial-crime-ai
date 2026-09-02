"""Retrieval-Augmented Generation (RAG) knowledge layer.

The investigator conditions its explanation on two retrieval sources:

1. **AML typology knowledge base** - scored by keyword overlap with the
   case's anomaly-driving features.
2. **Similar historical cases** - k-nearest accounts in the graph feature
   space of the similar-case index built during the pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from financial_crime_ai.features import FEATURE_LABELS, TYPOLOGIES

KEYWORDS: dict[str, list[str]] = {
    "structuring": [
        "structuring",
        "threshold",
        "deposit",
        "smurf",
        "10k",
        "consolidation",
        "threshold_proximity",
        "structuring_score",
        "cash",
    ],
    "layering": [
        "layering",
        "mule",
        "chain",
        "pass-through",
        "fresh",
        "multi-hop",
        "layering_score",
        "min_hops_to_sanctioned",
        "network",
    ],
    "cashout": [
        "cashout",
        "lump",
        "fresh",
        "wire",
        "offshore",
        "rapid",
        "amount_zscore",
        "time_since_prev_h",
        "account_age_days",
    ],
    "gambling": [
        "gambling",
        "velocity",
        "churn",
        "merchant",
        "high-velocity",
        "txn_count_24h",
        "txn_count_7d",
    ],
    "rounding": [
        "round",
        "rounding",
        "4,950",
        "9,900",
        "99,000",
        "is_round",
        "round_proximity",
    ],
    "sanctions": [
        "sanction",
        "watchlist",
        "high-risk",
        "jurisdiction",
        "pep",
        "offshore",
        "min_hops_to_sanctioned",
        "amount_to_sanctioned",
        "country_risk",
    ],
    "cycle": [
        "cycle",
        "circular",
        "loop",
        "ring",
        "sender_net_flow",
        "receiver_degree_in",
        "net_flow",
    ],
    "fan_out": [
        "fan-out",
        "dispersal",
        "out-degree",
        "sender_degree_out",
        "sender_activity_count",
        "recipients",
        "source",
    ],
    "fan_in": [
        "fan-in",
        "concentration",
        "collection",
        "in-degree",
        "receiver_degree_in",
        "receiver_activity_total",
        "sink",
    ],
}


TYPOLOGY_KEY: dict[str, str] = {
    "001": "structuring",
    "002": "layering",
    "003": "cashout",
    "004": "gambling",
    "005": "rounding",
    "006": "sanctions",
    "007": "cycle",
    "008": "fan_out",
    "009": "fan_in",
}


def _score_typology(typology: dict[str, str], evidence_tokens: set[str]) -> float:
    """Score a typology against the case's anomaly-driving feature names."""
    group = TYPOLOGY_KEY.get(typology["id"].split("-")[-1], "")
    score = 0.0
    # strongest signal: the case's own driving features
    for feat in evidence_tokens:
        if feat.startswith("feature:"):
            name = feat[8:]
            if name in typology.get("models", ""):
                score += 3.0
            label = FEATURE_LABELS.get(name, "").lower()
            if label and any(kw in label for kw in KEYWORDS.get(group, [])):
                score += 2.0
    # secondary signal: keyword overlap with the typology text
    text = " ".join(
        [
            typology["name"],
            typology["summary"],
            typology["indicators"],
            typology["models"],
        ]
    ).lower()
    for kw in KEYWORDS.get(group, []):
        if kw in text or kw in evidence_tokens:
            score += 0.5
    return score


def retrieve_typologies(
    contributor_features: list[str],
    top_k: int = 3,
) -> list[dict[str, str]]:
    """Score and return the most relevant AML typologies for a case."""
    tokens = {f"feature:{f}" for f in contributor_features}
    tokens |= {f.lower() for f in contributor_features}
    scored = [(t, _score_typology(t, tokens)) for t in TYPOLOGIES]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, s in scored[:top_k]]


def retrieve_similar_cases(
    account_id: str | int,
    index: dict[str, Any],
    evidence: dict[str, Any],
    k: int = 5,
) -> list[dict]:
    """Nearest neighbours of an account in the standardised graph space."""
    ids = [int(a) for a in index["account_ids"]]
    X = index["X_scaled"]
    account_id = int(account_id)
    if account_id not in ids:
        return []
    pos = ids.index(account_id)
    query = X[pos]
    dists = np.linalg.norm(X - query, axis=1)
    order = np.argsort(dists)
    result = []
    for p in order:
        if p == pos:
            continue
        other = ids[p]
        result.append(
            {
                "account_id": other,
                "similarity": round(float(1.0 / (1.0 + dists[p])), 4),
                "risk_score": round(float(index["risk"].get(other, 0.0)), 4),
                "evidence": _evidence_snippet(evidence, other),
            }
        )
        if len(result) >= k:
            break
    return result


def _evidence_snippet(evidence: dict[str, Any], account_id: str) -> dict:
    row = evidence.get(account_id)
    if not row:
        return {}
    return {
        "customer": row.get("customer_name", ""),
        "segment": row.get("segment", ""),
        "country": row.get("customer_country", ""),
        "n_transactions": row.get("n_transactions", 0),
        "total_flow": round(
            float(row.get("total_in", 0.0) or 0.0)
            + float(row.get("total_out", 0.0) or 0.0),
            0,
        ),
        "fresh": bool(row.get("is_fresh", False)),
        "pattern": row.get("injected_pattern", "none"),
    }
