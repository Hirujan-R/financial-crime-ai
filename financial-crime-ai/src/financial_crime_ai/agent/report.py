"""Case report data model returned by the AI investigator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class FeatureContribution:
    feature: str
    label: str
    value: float
    contribution: float


@dataclass
class CaseReport:
    txn_id: str = ""
    account_id: str = ""
    account_label: str = ""
    risk_score: float = 0.0
    model_confidence: float = 0.0
    summary: str = ""
    why_flagged: list[str] = field(default_factory=list)
    top_contributors: list[FeatureContribution] = field(default_factory=list)
    matched_typologies: list[str] = field(default_factory=list)
    connected_accounts: list[dict] = field(default_factory=list)
    historical_behaviour: dict = field(default_factory=dict)
    similar_cases: list[dict] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    sanctions_matches: list[str] = field(default_factory=list)
    recommended_action: str = ""
    action_confidence: str = ""
    generated_by: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["top_contributors"] = [asdict(c) for c in self.top_contributors]
        return d
