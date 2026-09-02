"""Investigation pipeline (IBM AML data)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import (
    build_account_edges,
    build_evidence_store,
    build_investigation_queue,
    build_similar_case_index,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    return Pipeline(
        [
            node(
                func=build_account_edges,
                inputs="ibm_transactions_raw",
                outputs="ibm_account_edges",
                name="build_ibm_account_edges",
            ),
            node(
                func=build_investigation_queue,
                inputs=[
                    "test_predictions",
                    "test_transactions",
                    "ibm_alerts",
                    "best_model_info",
                ],
                outputs="investigation_queue",
                name="build_investigation_queue",
            ),
            node(
                func=build_evidence_store,
                inputs=[
                    "ibm_transactions_raw",
                    "ibm_accounts",
                    "ibm_account_features",
                    "ibm_alerts",
                ],
                outputs="evidence_store",
                name="build_evidence_store",
            ),
            node(
                func=build_similar_case_index,
                inputs=["ibm_account_features", "test_predictions"],
                outputs="similar_case_index",
                name="build_similar_case_index",
            ),
        ]
    )
