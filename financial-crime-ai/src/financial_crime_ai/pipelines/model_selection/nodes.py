"""Kedro nodes for the model selection pipeline."""

from __future__ import annotations

import pandas as pd

from .models import fit_candidates, select_best


def run_model_selection(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
    candidates: list[dict],
    feature_columns: list[str],
    seed: int,
) -> tuple[
    pd.DataFrame, dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame
]:
    model_results, fitted = fit_candidates(
        train_features, validation_features, candidates, feature_columns, seed
    )
    best_model_info, test_metrics, test_predictions = select_best(
        model_results, fitted, test_features, feature_columns
    )
    return model_results, fitted, best_model_info, test_metrics, test_predictions
