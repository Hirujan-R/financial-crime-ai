"""Model selection pipeline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import run_model_selection


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    return Pipeline(
        [
            node(
                func=run_model_selection,
                inputs=[
                    "train_features",
                    "validation_features",
                    "test_features",
                    "params:model_candidates",
                    "params:feature_columns",
                    "params:seed",
                ],
                outputs=[
                    "model_results",
                    "fitted_candidates",
                    "best_model_info",
                    "test_metrics",
                    "test_predictions",
                ],
                name="fit_and_select_models",
            ),
        ]
    )
