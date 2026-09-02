"""Narrative case summarisation.

Generates the analyst-facing narrative.  When an LLM provider is available
it produces free-form prose grounded in the retrieved evidence (RAG-style);
otherwise a deterministic template summary is produced from the structured
case data.
"""

from __future__ import annotations

from typing import Any

from .llm import LLMProvider


def summarise(
    case: Any,
    provider: LLMProvider | None = None,
) -> str:
    if provider is not None and provider.available:
        return _llm_summary(case, provider)
    return case.summary or _template_summary(case)


def _llm_summary(case: Any, provider: LLMProvider) -> str:
    evidence = {
        "txn_id": case.txn_id,
        "account": case.account_label,
        "risk_score": case.risk_score,
        "why_flagged": case.why_flagged,
        "typologies": case.matched_typologies,
        "top_contributors": [
            {"feature": c.label, "contribution": round(c.contribution, 3)}
            for c in case.top_contributors[:5]
        ],
        "similar_cases": case.similar_cases[:3],
        "recommended_action": case.recommended_action,
    }
    import json

    messages = [
        {
            "role": "system",
            "content": (
                "You summarise financial-crime investigations for AML analysts. "
                "Write a concise 3-5 sentence narrative that explains why the "
                "transaction is suspicious, the behavioural pattern, and the "
                "recommended action. Use the evidence only."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(evidence, indent=2),
        },
    ]
    try:
        msg = provider.chat(messages, temperature=0.3)
        content = (msg.get("content") or "").strip()
        return content if content else case.summary
    except Exception:  # noqa: BLE001
        return case.summary


def _template_summary(case: Any) -> str:
    return case.summary
