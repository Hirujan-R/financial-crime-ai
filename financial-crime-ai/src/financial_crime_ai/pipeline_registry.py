"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines(raise_errors=True)

    # Run every stage end-to-end in dependency order.
    pipelines["__default__"] = sum(pipelines.values())

    pipelines["aml_data"] = pipelines["aml_data"]
    pipelines["model_selection"] = pipelines["model_selection"]
    pipelines["aml_sim"] = pipelines["aml_sim"]
    pipelines["investigation"] = pipelines["investigation"]
    pipelines["agent"] = pipelines["agent"]

    return pipelines
