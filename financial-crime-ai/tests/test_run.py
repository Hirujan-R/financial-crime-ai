"""Tests for the Kedro project wiring."""

from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession

from financial_crime_ai.pipeline_registry import register_pipelines


def test_default_pipeline_is_registered():
    configure_project("financial_crime_ai")
    pipelines = register_pipelines()
    assert "__default__" in pipelines
    for name in [
        "aml_data",
        "model_selection",
        "aml_sim",
        "investigation",
        "agent",
    ]:
        assert name in pipelines
    assert len(pipelines["__default__"].nodes) >= 13


def test_kedro_session_loads_catalog():
    configure_project("financial_crime_ai")
    with KedroSession.create(project_path=Path.cwd()) as session:
        context = session.load_context()
        catalog = context.catalog
        for dataset in [
            "train_features",
            "validation_features",
            "test_features",
            "test_predictions",
            "best_model_info",
            "investigation_queue",
        ]:
            assert catalog.exists(dataset)
