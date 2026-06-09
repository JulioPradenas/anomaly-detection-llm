"""Drift detection using Evidently 0.7 — monitors feature distribution shift over time."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


@dataclass
class DriftReport:
    drift_score: float
    drift_detected: bool
    features_drifted: list[str]
    feature_scores: dict[str, float]
    n_features_total: int
    recommendation: str


class DriftDetector:
    """Monitors whether incoming log features have drifted from the training distribution.

    Uses Evidently DataDriftPreset (KS test per feature). A feature is considered
    drifted when the KS p-value falls below `threshold` (default 0.05).
    The overall drift_score is the share of drifted features.
    """

    def __init__(self, threshold: float = 0.05, drift_share: float = 0.5) -> None:
        self.threshold = threshold
        self.drift_share = drift_share
        self._reference: pd.DataFrame | None = None
        self._last_report: DriftReport | None = None

    def fit_reference(self, reference_df: pd.DataFrame) -> "DriftDetector":
        """Store reference distribution (training data)."""
        self._reference = reference_df.copy()
        return self

    def detect(self, current_df: pd.DataFrame) -> DriftReport:
        """Compute drift between current data and the reference distribution."""
        if self._reference is None:
            raise RuntimeError("Call fit_reference() before detect().")

        shared_cols = [c for c in self._reference.columns if c in current_df.columns]
        ref = self._reference[shared_cols]
        cur = current_df[shared_cols]

        report = Report([DataDriftPreset(drift_share=self.drift_share)])
        snapshot = report.run(current_data=cur, reference_data=ref)
        drift_report = _parse_snapshot(snapshot, self.threshold)
        self._last_report = drift_report
        return drift_report

    def get_drift_score(self) -> float:
        """Return the drift score from the last detect() call. 0 = no drift, 1 = full drift."""
        if self._last_report is None:
            return 0.0
        return self._last_report.drift_score

    def generate_html_report(self, path: str | Path) -> None:
        """Run detect internally using the last snapshot and save HTML to path."""
        if self._reference is None:
            raise RuntimeError("Call fit_reference() before generate_html_report().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raise RuntimeError(
            "Call detect() first, then use generate_html_report_from_data() with current_df."
        )

    def generate_html_report_from_data(
        self, current_df: pd.DataFrame, path: str | Path
    ) -> DriftReport:
        """Run drift detection and save the HTML report. Returns the DriftReport."""
        if self._reference is None:
            raise RuntimeError("Call fit_reference() before generate_html_report_from_data().")

        shared_cols = [c for c in self._reference.columns if c in current_df.columns]
        ref = self._reference[shared_cols]
        cur = current_df[shared_cols]

        report = Report([DataDriftPreset(drift_share=self.drift_share)])
        snapshot = report.run(current_data=cur, reference_data=ref)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(path))

        drift_report = _parse_snapshot(snapshot, self.threshold)
        self._last_report = drift_report
        return drift_report


def _parse_snapshot(snapshot: Any, threshold: float) -> DriftReport:
    """Extract drift metrics from an Evidently 0.7 Snapshot."""
    data = json.loads(snapshot.json())
    metrics = data.get("metrics", [])

    drift_score = 0.0
    feature_scores: dict[str, float] = {}

    for m in metrics:
        name = m.get("metric_name", "")
        value = m.get("value")

        if "DriftedColumnsCount" in name and isinstance(value, dict):
            drift_score = float(value.get("share", 0.0))

        if "ValueDrift(column=" in name and isinstance(value, (int, float)):
            col = m["config"].get("column", "unknown")
            feature_scores[col] = float(value)

    features_drifted = [col for col, pval in feature_scores.items() if pval < threshold]

    n_total = len(feature_scores) if feature_scores else 1
    if not feature_scores:
        drift_score = 0.0

    drift_detected = drift_score > 0.0 and len(features_drifted) > 0

    if not drift_detected:
        recommendation = "No se detectó drift — el modelo puede continuar en producción."
    elif drift_score < 0.3:
        recommendation = "Drift leve — monitorear durante los próximos días."
    elif drift_score < 0.7:
        recommendation = "Drift moderado — considerar re-entrenamiento con datos recientes."
    else:
        recommendation = "Drift severo — re-entrenamiento recomendado con /retrain."

    return DriftReport(
        drift_score=round(drift_score, 4),
        drift_detected=drift_detected,
        features_drifted=features_drifted,
        feature_scores={k: round(v, 6) for k, v in feature_scores.items()},
        n_features_total=n_total,
        recommendation=recommendation,
    )
