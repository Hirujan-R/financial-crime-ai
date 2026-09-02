"""Structured tools the AI investigator can call.

Implements a small function-calling registry.  Each tool exposes a name,
description, JSON-schema parameters and an executable function so that the
same tools are used by both the deterministic fallback engine and the LLM
agent loop.  The tools operate on the IBM AML transaction graph.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from financial_crime_ai.features import FEATURE_LABELS

from .explainer import compute_contributions, why_flagged_messages
from .knowledge import retrieve_similar_cases


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    required: list[str] = field(default_factory=list)


class ToolRegistry:
    """Holds and executes the investigator's tools."""

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self._tools: dict[str, Tool] = {}
        self._register()

    def _register(self) -> None:
        self.add(
            Tool(
                name="get_account_profile",
                description=(
                    "Return the account profile for an account_id: balances, "
                    "activity volume, graph-based indicators and prior alert "
                    "involvement."
                ),
                parameters={
                    "type": "object",
                    "properties": {"account_id": {"type": "integer"}},
                },
                required=["account_id"],
                func=self._get_account_profile,
            )
        )
        self.add(
            Tool(
                name="get_transaction_history",
                description=(
                    "Return recent transactions for an account_id (as sender or "
                    "receiver) with counterparty, amount, timestamp and whether "
                    "the transaction was flagged."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                },
                required=["account_id"],
                func=self._get_transaction_history,
            )
        )
        self.add(
            Tool(
                name="get_account_network",
                description=(
                    "Return counterparties connected to an account_id with the "
                    "value and number of transfers between them."
                ),
                parameters={
                    "type": "object",
                    "properties": {"account_id": {"type": "integer"}},
                },
                required=["account_id"],
                func=self._get_account_network,
            )
        )
        self.add(
            Tool(
                name="get_top_contributors",
                description=(
                    "Return the top model features contributing to the risk "
                    "score for a txn_id (why it was flagged)."
                ),
                parameters={
                    "type": "object",
                    "properties": {"txn_id": {"type": "integer"}},
                },
                required=["txn_id"],
                func=self._get_top_contributors,
            )
        )
        self.add(
            Tool(
                name="get_similar_cases",
                description=(
                    "Return similar accounts for an account_id based on graph "
                    "and behavioural features."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "integer"},
                        "k": {"type": "integer"},
                    },
                },
                required=["account_id"],
                func=self._get_similar_cases,
            )
        )
        self.add(
            Tool(
                name="check_prior_alerts",
                description=(
                    "Check whether an account_id was involved in previously "
                    "identified suspicious alerts (and which typologies)."
                ),
                parameters={
                    "type": "object",
                    "properties": {"account_id": {"type": "integer"}},
                },
                required=["account_id"],
                func=self._check_prior_alerts,
            )
        )
        self.add(
            Tool(
                name="get_risk_score",
                description=(
                    "Return the model risk score, decision threshold and "
                    "prediction for a txn_id."
                ),
                parameters={
                    "type": "object",
                    "properties": {"txn_id": {"type": "integer"}},
                },
                required=["txn_id"],
                func=self._get_risk_score,
            )
        )

    # --------------------------------------------------------------- plumbing
    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tools_json(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = tool.func(**arguments)
            if isinstance(result, str):
                return result
            return _json(result)
        except Exception as exc:  # noqa: BLE001
            return f"Error calling {name}: {exc}"

    # ------------------------------------------------------------- handlers
    def _get_account_profile(self, account_id: int) -> dict:
        ctx = self.ctx
        row = ctx._evidence_by_account.get(int(account_id))
        if row is None:
            return {"error": f"unknown account {account_id}"}
        return {k: v for k, v in row.items()}

    def _get_transaction_history(self, account_id: int, limit: int = 20) -> dict:
        ctx = self.ctx
        t = ctx.transactions
        hist = (
            t[
                (t["SENDER_ACCOUNT_ID"] == account_id)
                | (t["RECEIVER_ACCOUNT_ID"] == account_id)
            ]
            .sort_values("TIMESTAMP", ascending=False)
            .head(limit)
        )
        recs = hist[
            [
                "TX_ID",
                "SENDER_ACCOUNT_ID",
                "RECEIVER_ACCOUNT_ID",
                "TX_AMOUNT",
                "TIMESTAMP",
                "IS_FRAUD",
            ]
        ].to_dict("records")
        for r in recs:
            r["TX_AMOUNT"] = round(float(r["TX_AMOUNT"]), 2)
            r["IS_FRAUD"] = bool(r["IS_FRAUD"])
            r["direction"] = "OUT" if r["SENDER_ACCOUNT_ID"] == account_id else "IN"
        return {"account_id": account_id, "n": len(recs), "transactions": recs}

    def _get_account_network(self, account_id: int) -> dict:
        ctx = self.ctx
        out = (
            ctx.account_edges[ctx.account_edges["src"] == account_id]
            .sort_values("value", ascending=False)
            .head(15)
        )
        recs = [
            {
                "counterparty_id": int(r.dst),
                "value": round(float(r.value), 2),
                "count": int(r.count),
            }
            for r in out.itertuples()
        ]
        return {"account_id": account_id, "counterparties": recs}

    def _get_top_contributors(self, txn_id: int) -> dict:
        ctx = self.ctx
        contribs = compute_contributions(
            ctx.best_model_info, ctx.test_features, str(txn_id), top_n=8
        )
        return {
            "txn_id": txn_id,
            "contributors": [
                {
                    "feature": c["feature"],
                    "label": FEATURE_LABELS.get(c["feature"], c["feature"]),
                    "value": round(c["value"], 4),
                    "contribution": round(c["contribution"], 4),
                }
                for c in contribs
            ],
            "why_flagged": why_flagged_messages(contribs),
        }

    def _get_similar_cases(self, account_id: int, k: int = 5) -> dict:
        ctx = self.ctx
        similar = retrieve_similar_cases(
            str(account_id),
            ctx.similar_case_index,
            ctx._evidence_by_account,
            k=k,
        )
        return {"account_id": account_id, "similar_cases": similar}

    def _check_prior_alerts(self, account_id: int) -> dict:
        ctx = self.ctx
        a = ctx.alerts[ctx.alerts["ALERT_ID"] >= 0]
        involved = a[
            (a["SENDER_ACCOUNT_ID"] == account_id)
            | (a["RECEIVER_ACCOUNT_ID"] == account_id)
        ]
        types = involved["ALERT_TYPE"].value_counts().to_dict()
        return {
            "account_id": account_id,
            "prior_alert_events": int(len(involved)),
            "typologies": {str(k): int(v) for k, v in types.items()},
        }

    def _get_risk_score(self, txn_id: int) -> dict:
        ctx = self.ctx
        row = ctx._txn_by_id.get(str(txn_id))
        if row is None:
            return {"error": f"unknown txn {txn_id}"}
        return {
            "txn_id": txn_id,
            "risk_score": round(float(row["risk_score"]), 4),
            "threshold": round(float(ctx.best_model_info.get("threshold", 0.5)), 4),
            "predicted_fraud": bool(row.get("predicted_fraud", False)),
        }


def _json(obj: Any) -> str:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    return json.dumps(obj, default=default, indent=2)
