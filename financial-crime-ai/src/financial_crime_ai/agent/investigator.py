"""AI Investigator facade.

Combines the deterministic evidence engine with (optionally) an LLM
agent that uses structured tool calling, and produces a single ``CaseReport``
for the analyst dashboard.
"""

from __future__ import annotations

from .context import InvestigatorContext
from .fallback import investigate as deterministic_investigate
from .llm import LLMProvider, extract_json, run_agent_with_tools
from .report import CaseReport
from .summariser import summarise
from .tools import ToolRegistry

_QUERY = (
    "Investigate the flagged transaction {txn_id}. Gather evidence using the "
    "tools and produce the final JSON case report."
)


class CaseInvestigator:
    def __init__(
        self,
        ctx: InvestigatorContext,
        provider: LLMProvider | None = None,
    ) -> None:
        self.ctx = ctx
        self.registry = ToolRegistry(ctx)
        self.provider = provider or LLMProvider()

    @property
    def mode(self) -> str:
        return "llm-agent" if self.provider.available else "deterministic-engine"

    def investigate(self, txn_id: str) -> CaseReport:
        base = deterministic_investigate(self.registry, self.ctx, txn_id)
        if not self.provider.available:
            return base

        try:
            text = run_agent_with_tools(
                self.registry, _QUERY.format(txn_id=txn_id), self.provider
            )
            llm = extract_json(text)
        except Exception:  # noqa: BLE001
            llm = {}

        if llm:
            base.generated_by = "llm-agent"
            base.summary = str(llm.get("summary", base.summary))
            if llm.get("why_flagged"):
                base.why_flagged = [str(x) for x in llm["why_flagged"]]
            if llm.get("typologies"):
                base.matched_typologies = [str(x) for x in llm["typologies"]]
            if llm.get("recommended_action"):
                base.recommended_action = str(llm["recommended_action"])
            if llm.get("action_confidence"):
                base.action_confidence = str(llm["action_confidence"])
            if llm.get("notes"):
                base.supporting_evidence.append(str(llm["notes"]))
        else:
            base.summary = summarise(base, self.provider)
        return base
