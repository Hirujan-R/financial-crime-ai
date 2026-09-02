"""Deterministic investigation engine.

A rule-based investigator that executes the same tools as the LLM agent to
produce a complete structured case report.  It is the no-API-key fallback
and also guarantees the dashboard always has populated evidence.
"""

from __future__ import annotations

import json
from typing import Any

from financial_crime_ai.features import FEATURE_LABELS

from .knowledge import retrieve_typologies
from .report import CaseReport, FeatureContribution


def investigate(registry, ctx, txn_id: str) -> CaseReport:
    """Build a structured case report for a flagged transaction."""
    risk = _tool_json(registry, "get_risk_score", {"txn_id": int(txn_id)})
    account_id = ctx._txn_by_id.get(str(txn_id), {}).get("SENDER_ACCOUNT_ID")

    profile = (
        _tool_json(registry, "get_account_profile", {"account_id": account_id})
        if account_id is not None
        else {}
    )
    contribs = _tool_json(registry, "get_top_contributors", {"txn_id": int(txn_id)})
    network = (
        _tool_json(registry, "get_account_network", {"account_id": account_id})
        if account_id is not None
        else {}
    )
    similar = (
        _tool_json(registry, "get_similar_cases", {"account_id": account_id})
        if account_id is not None
        else {}
    )
    prior_alerts = (
        _tool_json(registry, "check_prior_alerts", {"account_id": account_id})
        if account_id is not None
        else {}
    )

    top = contribs.get("contributors", [])
    why_flagged = contribs.get("why_flagged", [])
    matched = retrieve_typologies([c["feature"] for c in top], top_k=3)

    report = CaseReport(
        txn_id=str(txn_id),
        account_id=str(account_id) if account_id is not None else "",
        account_label=f"Account {account_id}" if account_id is not None else "",
        risk_score=float(risk.get("risk_score", 0.0)),
        model_confidence=_confidence(float(risk.get("risk_score", 0.0))),
        why_flagged=why_flagged,
        top_contributors=[
            FeatureContribution(
                feature=c["feature"],
                label=FEATURE_LABELS.get(c["feature"], c["feature"]),
                value=c["value"],
                contribution=c["contribution"],
            )
            for c in top
        ],
        matched_typologies=[t["name"] for t in matched],
        connected_accounts=network.get("counterparties", []),
        historical_behaviour=_history(profile, registry, account_id),
        similar_cases=similar.get("similar_cases", []),
        supporting_evidence=_evidence_list(profile, prior_alerts),
        sanctions_matches=_prior_alert_messages(prior_alerts),
        recommended_action=_recommend(
            float(risk.get("risk_score", 0.0)),
            prior_alerts.get("prior_alert_events", 0),
        ),
        action_confidence=_confidence(float(risk.get("risk_score", 0.0))),
        generated_by="deterministic-engine",
    )
    report.summary = _summary(report)
    return report


def _tool_json(registry, name: str, args: dict[str, Any]) -> dict:
    result = registry.execute(name, args)
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {}


def _confidence(score: float) -> float:
    return round(float(min(0.99, 0.5 + abs(score - 0.5))), 3)


def _history(profile: dict, registry, account_id) -> dict:
    hist = (
        _tool_json(
            registry, "get_transaction_history", {"account_id": account_id, "limit": 8}
        )
        if account_id is not None
        else {}
    )
    return {
        "account_id": account_id,
        "init_balance_log": profile.get("init_balance_log"),
        "behavior_id": profile.get("behavior_id"),
        "n_transactions": profile.get("n_transactions", 0),
        "total_sent": round(float(profile.get("total_sent") or 0.0), 2),
        "total_received": round(float(profile.get("total_received") or 0.0), 2),
        "avg_amount": round(float(profile.get("avg_amount") or 0.0), 2),
        "max_amount": round(float(profile.get("max_amount") or 0.0), 2),
        "n_outgoing": profile.get("n_outgoing", 0),
        "n_incoming": profile.get("n_incoming", 0),
        "in_known_alert": bool(profile.get("in_known_alert", False)),
        "recent": hist.get("transactions", [])[:5],
    }


def _evidence_list(profile: dict, prior_alerts: dict) -> list[str]:
    out = []
    if (
        float(profile.get("total_received") or 0.0)
        > float(profile.get("total_sent") or 0.0) * 3
    ):
        out.append(
            "The account receives far more than it sends (collection behaviour)."
        )
    if (
        float(profile.get("total_sent") or 0.0)
        > float(profile.get("total_received") or 0.0) * 3
    ):
        out.append(
            "The account disperses far more than it receives (dispersal behaviour)."
        )
    if profile.get("n_transactions", 0) > 300:
        out.append(f"Very high activity: {profile.get('n_transactions')} transactions.")
    if profile.get("in_known_alert"):
        out.append(
            "The account is already linked to previously identified suspicious alerts."
        )
    for typ, n in (prior_alerts.get("typologies") or {}).items():
        out.append(f"Prior alert activity of type '{typ}' ({n} events).")
    return out


def _prior_alert_messages(prior_alerts: dict) -> list[str]:
    if not prior_alerts:
        return []
    msgs = []
    for typ, n in (prior_alerts.get("typologies") or {}).items():
        msgs.append(f"Involved in {n} prior {typ} alert(s).")
    return msgs


def _recommend(score: float, prior_alert_events: int) -> str:
    if prior_alert_events > 0 and score >= 0.7:
        return (
            "Freeze the account immediately and escalate to a senior investigator: "
            "the account has prior alert involvement and a high model risk score."
        )
    if score >= 0.8:
        return (
            "Place the account under temporary freeze pending review by a senior "
            "investigator; trace counterparties and request source-of-funds evidence."
        )
    if score >= 0.6:
        return (
            "Apply enhanced monitoring and trace this transaction's counterparties "
            "before any large onward movement."
        )
    return "Review the transaction and close the case if no further risk emerges."


def _summary(report: CaseReport) -> str:
    top = report.top_contributors[0] if report.top_contributors else None
    pattern = (
        report.matched_typologies[0]
        if report.matched_typologies
        else "suspicious activity"
    )
    driver = f" The strongest driver is {top.label.lower()}." if top else ""
    return (
        f"Transaction {report.txn_id} on {report.account_label} carries a risk "
        f"score of {report.risk_score:.2f}, consistent with {pattern.lower()}.{driver} "
        f"Recommended action: {report.recommended_action}"
    )
