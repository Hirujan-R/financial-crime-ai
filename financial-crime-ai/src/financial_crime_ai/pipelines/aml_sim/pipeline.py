"""AMLSim generalisation experiment pipeline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from .nodes import evaluate_generalisation, generate_amlsim


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    return Pipeline(
        [
            node(
                func=generate_amlsim,
                inputs=[
                    "params:amlsim_n_accounts",
                    "params:amlsim_n_timestamps",
                    "params:amlsim_seed",
                    "params:amlsim_normal_rate",
                ],
                outputs=[
                    "aml_accounts",
                    "aml_train_transactions",
                    "aml_test_transactions",
                    "aml_scenario_summary",
                ],
                name="generate_amlsim_scenarios",
            ),
            node(
                func=evaluate_generalisation,
                inputs=[
                    "aml_train_transactions",
                    "aml_test_transactions",
                    "aml_accounts",
                    "best_model_info",
                    "params:amlsim_seed",
                ],
                outputs=["generalisation_metrics", "generalisation_info"],
                name="evaluate_generalisation",
            ),
        ]
    )
