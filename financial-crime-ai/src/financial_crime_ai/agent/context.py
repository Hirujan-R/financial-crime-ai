"""Investigator context: loads the pipeline outputs the agent needs."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from financial_crime_ai.pipelines.aml_data.features import MODEL_FEATURES


@dataclass
class InvestigatorContext:
    """In-memory access layer over the IBM AML pipeline outputs."""

    transactions: pd.DataFrame = field(default_factory=pd.DataFrame)
    accounts: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: pd.DataFrame = field(default_factory=pd.DataFrame)
    account_edges: pd.DataFrame = field(default_factory=pd.DataFrame)
    account_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    train_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    split_boundaries: dict = field(default_factory=dict)
    investigation_queue: pd.DataFrame = field(default_factory=pd.DataFrame)
    evidence_store: pd.DataFrame = field(default_factory=pd.DataFrame)
    similar_case_index: dict = field(default_factory=dict)
    generalisation_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    best_model_info: dict = field(default_factory=dict)

    _evidence_by_account: dict = field(default_factory=dict, repr=False)
    _txn_by_id: dict = field(default_factory=dict, repr=False)
    _alert_accounts: set = field(default_factory=set, repr=False)

    @property
    def feature_columns(self) -> list[str]:
        return self.best_model_info.get("feature_columns", MODEL_FEATURES)

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_project_dir(cls, project_dir: str | Path) -> InvestigatorContext:
        root = Path(project_dir) / "data"
        raw = root / "01_raw"
        inter = root / "02_intermediate"
        prim = root / "03_primary"
        models = root / "04_models"
        reporting = root / "06_reporting"

        def read_csv(path: Path) -> pd.DataFrame:
            return pd.read_csv(path) if path.exists() else pd.DataFrame()

        def read_pq(path: Path) -> pd.DataFrame:
            return pd.read_parquet(path) if path.exists() else pd.DataFrame()

        def read_pkl(path: Path) -> object:
            with open(path, "rb") as fh:
                return pickle.load(fh)

        ctx = cls(
            transactions=read_csv(raw / "ibm" / "transactions.csv"),
            accounts=read_csv(raw / "ibm" / "accounts.csv"),
            alerts=read_csv(raw / "ibm" / "alerts.csv"),
            account_edges=read_csv(inter / "ibm_account_edges.csv"),
            account_features=read_pq(prim / "ibm_account_features.parquet"),
            train_features=read_pq(prim / "train_features.parquet"),
            validation_features=read_pq(prim / "validation_features.parquet"),
            test_features=read_pq(prim / "test_features.parquet"),
            test_predictions=read_pq(prim / "test_predictions.parquet"),
            model_results=read_pq(prim / "model_results.parquet"),
            test_metrics=read_csv(reporting / "test_metrics.csv"),
            split_boundaries=read_pkl(reporting / "split_boundaries.pkl"),
            investigation_queue=read_pq(reporting / "investigation_queue.parquet"),
            evidence_store=read_pq(reporting / "evidence_store.parquet"),
            similar_case_index=read_pkl(models / "similar_case_index.pkl"),
            generalisation_metrics=read_csv(reporting / "generalisation_metrics.csv"),
            best_model_info=read_pkl(models / "best_model_info.pkl"),
        )
        ctx._index()
        return ctx

    def _index(self) -> None:
        self._evidence_by_account = {
            row["account_id"]: row for row in self.evidence_store.to_dict("records")
        }
        if not self.test_predictions.empty:
            self._txn_by_id = {
                str(row["TX_ID"]): row
                for row in self.test_predictions.to_dict("records")
            }
        if not self.alerts.empty:
            a = self.alerts[self.alerts["ALERT_ID"] >= 0]
            self._alert_accounts = set(a["SENDER_ACCOUNT_ID"]) | set(
                a["RECEIVER_ACCOUNT_ID"]
            )
