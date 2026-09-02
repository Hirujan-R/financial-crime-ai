"""AI investigator pipeline.

Materialises the RAG knowledge base used by the investigator.  The
interactive investigation itself runs inside the dashboard / agent service.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from financial_crime_ai.features import TYPOLOGIES


def build_knowledge_base() -> dict:
    return {"typologies": TYPOLOGIES}


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    return Pipeline(
        [
            node(
                func=build_knowledge_base,
                inputs=None,
                outputs="knowledge_base",
                name="build_knowledge_base",
            )
        ]
    )
