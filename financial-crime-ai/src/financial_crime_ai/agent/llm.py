"""Lightweight OpenAI-compatible LLM client with structured tool calling.

Uses only the standard library (``urllib``) so no extra dependency is
required.  When ``OPENAI_API_KEY`` is not configured, the caller should
fall back to the deterministic investigator.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

SYSTEM_PROMPT = """You are the AI Investigator for a financial-crime investigation platform \
used by AML analysts. Given a flagged transaction you must investigate the case by calling the \
provided tools to gather evidence (account profile, transaction history, network, sanctions check, \
top model contributors, similar cases, risk score).

You must call tools before answering. When you have enough evidence, produce your final answer as \
a single JSON object with exactly these keys:
  summary        : 2-3 sentence narrative of the case,
  why_flagged    : list of plain-language reasons drawn from the top contributors,
  typologies     : list of matched AML typology names,
  recommended_action : one concrete next action for the analyst,
  action_confidence  : LOW/MEDIUM/HIGH,
  notes          : any extra observations from the network/sanctions evidence.

Do not mention tool names in the JSON. Keep values factual and derived from the tool outputs."""


class LLMProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL", "")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]


def run_agent_with_tools(
    registry,
    query: str,
    provider: LLMProvider,
    max_rounds: int = 6,
) -> str:
    """Run the tool-calling loop and return the final model text."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    tools = registry.tools_json()
    for _ in range(max_rounds):
        msg = provider.chat(messages, tools=tools)
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )
            for call in tool_calls:
                fn = call["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = registry.execute(fn["name"], args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )
            continue
        content = msg.get("content")
        if content:
            return content
        return "No answer produced."
    return "Agent exceeded maximum tool-calling rounds."


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from model text."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return {}
