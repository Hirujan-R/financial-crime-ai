"""Tests for the AI investigator agent components."""

from pathlib import Path

import pandas as pd
import pytest

from financial_crime_ai.agent.explainer import (
    compute_contributions,
    why_flagged_messages,
)
from financial_crime_ai.agent.knowledge import (
    retrieve_typologies,
)
from financial_crime_ai.features import FEATURE_LABELS

ROOT = Path(__file__).resolve().parents[1]


def test_typology_retrieval_matches_novel_typologies():
    cases = [
        (["sender_degree_out", "sender_activity_count"], "Fan-out dispersal"),
        (["receiver_degree_in", "receiver_activity_total"], "Fan-in concentration"),
        (["sender_net_flow", "receiver_degree_in"], "Circular transfers (cycle)"),
    ]
    for feats, expected in cases:
        result = retrieve_typologies(feats, top_k=3)
        assert any(t["name"] == expected for t in result), (
            feats,
            [t["name"] for t in result],
        )


def test_all_model_features_have_labels():
    from financial_crime_ai.pipelines.aml_data.features import MODEL_FEATURES

    for f in MODEL_FEATURES:
        assert f in FEATURE_LABELS, f"missing label for {f}"


@pytest.mark.integration
def test_investigator_on_pipeline_outputs():
    """Run the full deterministic investigation against persisted outputs."""
    risk_scores = ROOT / "data" / "03_primary" / "test_predictions.parquet"
    if not risk_scores.exists():
        pytest.skip("pipeline outputs not generated - run `kedro run` first")

    from financial_crime_ai.agent.context import InvestigatorContext
    from financial_crime_ai.agent.investigator import CaseInvestigator

    ctx = InvestigatorContext.from_project_dir(ROOT)
    queue = pd.read_parquet(
        ROOT / "data" / "06_reporting" / "investigation_queue.parquet"
    )
    assert len(queue) > 0

    report = CaseInvestigator(ctx).investigate(str(queue.iloc[0]["txn_id"]))
    assert report.risk_score > 0
    assert report.summary
    assert report.recommended_action
    assert report.top_contributors
    assert report.why_flagged
    assert report.generated_by == "deterministic-engine"
    assert len(report.similar_cases) > 0


def test_explainer_handles_all_model_kinds():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier

    from financial_crime_ai.pipelines.aml_data.features import MODEL_FEATURES

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
    y = (rng.random(200) > 0.7).astype(int)
    X["TX_ID"] = [f"{i}" for i in range(200)]
    frame = X.copy()

    for kind, make in [
        ("logreg", lambda: LogisticRegression(max_iter=500)),
        ("xgboost", lambda: XGBClassifier(n_estimators=10, max_depth=2)),
        ("mlp", lambda: MLPClassifier(hidden_layer_sizes=(8,), max_iter=50)),
    ]:
        model = make()
        model.fit(X[MODEL_FEATURES], y)
        info = {
            "model": model,
            "kind": kind,
            "scaler": None,
            "threshold": 0.5,
            "feature_columns": MODEL_FEATURES,
        }
        contribs = compute_contributions(info, frame, "0", top_n=5)
        assert 0 < len(contribs) <= 5
        assert why_flagged_messages(contribs)
