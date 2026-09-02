"""IBM AML data pipeline: load, engineer features, chronological split."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import (
    engineer_split,
    engineer_train,
    load_accounts,
    load_alerts,
    load_transactions,
    split_temporal,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    return Pipeline(
        [
            node(
                func=load_transactions,
                inputs="params:ibm_transactions_path",
                outputs="ibm_transactions_raw",
                name="load_ibm_transactions",
            ),
            node(
                func=load_accounts,
                inputs="params:ibm_accounts_path",
                outputs="ibm_accounts",
                name="load_ibm_accounts",
            ),
            node(
                func=load_alerts,
                inputs="params:ibm_alerts_path",
                outputs="ibm_alerts",
                name="load_ibm_alerts",
            ),
            node(
                func=split_temporal,
                inputs=[
                    "ibm_transactions_raw",
                    "params:train_fraction",
                    "params:val_fraction",
                ],
                outputs=[
                    "train_transactions",
                    "validation_transactions",
                    "test_transactions",
                    "split_boundaries",
                ],
                name="temporal_split",
            ),
            node(
                func=engineer_train,
                inputs=["train_transactions", "ibm_accounts"],
                outputs=["train_features", "ibm_account_features"],
                name="engineer_train_features",
            ),
            node(
                func=engineer_split,
                inputs=[
                    "validation_transactions",
                    "ibm_account_features",
                    "params:val_label",
                ],
                outputs="validation_features",
                name="engineer_validation_features",
            ),
            node(
                func=engineer_split,
                inputs=[
                    "test_transactions",
                    "ibm_account_features",
                    "params:test_label",
                ],
                outputs="test_features",
                name="engineer_test_features",
            ),
        ]
    )
