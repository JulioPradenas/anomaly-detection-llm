"""Synthetic event generator for real-time streaming demo.

Generates BGL-like log events based on reference distribution statistics.
Mixes normal events (default 95%) with synthetic anomalies (5%).
"""

import random
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

_BGL_NODES = [f"R{r:02d}-M{m}-N{n}" for r in range(1, 24) for m in range(2) for n in range(4)]
_COMPONENTS = ["KERNEL", "APP", "RAS", "MEMORY", "NETWORK", "IO", "FILESYSTEM"]
_LEVELS_NORMAL = ["INFO", "INFO", "INFO", "INFO", "WARNING", "DEBUG"]
_LEVELS_ANOMALY = ["FATAL", "FATAL", "ERROR", "ERROR", "WARNING"]
_ANOMALY_MESSAGES = [
    "TLB parity error detected on processor core",
    "ciod: failed to read message prefix on CioStream socket",
    "machine check: NMI received — hardware error suspected",
    "SIGTERM sent to job — possible OOM condition",
    "kernel: EXT3-fs error (device sdb): ext3_find_entry",
    "uncorrectable memory error on DIMM slot 2",
    "network interface eth0: carrier lost",
    "excessive kernel soft lockup detected — 120s",
]
_NORMAL_MESSAGES = [
    "instruction cache parity error corrected",
    "memory scrubbing complete — no errors found",
    "job {job_id} submitted to scheduler",
    "node heartbeat received",
    "temperature sensor reading within normal range",
    "network packet checksum validated",
    "filesystem sync completed",
]


@dataclass
class LogEvent:
    event_id: str
    timestamp: datetime
    node: str
    level: str
    component: str
    content: str
    is_anomaly: bool
    anomaly_score: float
    severity: str = "LOW"

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "node": self.node,
            "level": self.level,
            "component": self.component,
            "content": self.content,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 4),
            "severity": self.severity,
        }


class EventGenerator:
    """Generates synthetic BGL-like log events for streaming demo.

    Args:
        reference_df: Training features DataFrame — used to fit score distributions.
        anomaly_rate: Fraction of events that are anomalies (default 0.05).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame | None = None,
        anomaly_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.anomaly_rate = anomaly_rate
        self._rng = np.random.default_rng(seed)
        self._random = random.Random(seed)
        self._base_time = datetime.now(tz=UTC)
        self._tick = 0

        # Fit score distribution from reference if provided
        if reference_df is not None and len(reference_df) > 10:
            score_col = "anomaly_score" if "anomaly_score" in reference_df.columns else None
            if score_col and "is_anomaly" in reference_df.columns:
                is_anom: pd.Series = reference_df["is_anomaly"].astype(bool)
                scores: pd.Series = reference_df[score_col].astype(float)
                normal = scores[~is_anom]
                anom = scores[is_anom]
                self._normal_score_mean = float(normal.mean()) if len(normal) > 0 else 0.15
                self._normal_score_std = float(normal.std()) if len(normal) > 0 else 0.08
                self._anom_score_mean = float(anom.mean()) if len(anom) > 0 else 0.78
                self._anom_score_std = float(anom.std()) if len(anom) > 0 else 0.12
            else:
                self._set_default_score_params()
        else:
            self._set_default_score_params()

    def _set_default_score_params(self) -> None:
        self._normal_score_mean = 0.15
        self._normal_score_std = 0.08
        self._anom_score_mean = 0.82
        self._anom_score_std = 0.10

    def next_event(self) -> LogEvent:
        """Generate the next event (normal or anomaly based on anomaly_rate)."""
        self._tick += 1
        ts = self._base_time + timedelta(seconds=self._tick * 0.5)
        is_anomaly = self._rng.random() < self.anomaly_rate
        return self._make_event(ts, is_anomaly=is_anomaly)

    def inject_anomaly(self) -> LogEvent:
        """Force a high-severity anomaly event — for demo button."""
        self._tick += 1
        ts = datetime.now(tz=UTC)
        return self._make_event(ts, is_anomaly=True, forced=True)

    def stream(self) -> Generator[LogEvent, None, None]:
        """Infinite generator of events. Use next_event() for tick-based usage."""
        while True:
            yield self.next_event()

    def _make_event(self, ts: datetime, is_anomaly: bool, forced: bool = False) -> LogEvent:
        node = self._random.choice(_BGL_NODES)
        component = self._random.choice(_COMPONENTS)

        if is_anomaly:
            level = self._random.choice(_LEVELS_ANOMALY)
            content = self._random.choice(_ANOMALY_MESSAGES)
            score = float(
                np.clip(self._rng.normal(self._anom_score_mean, self._anom_score_std), 0.55, 1.0)
            )
            if forced:
                score = float(np.clip(self._rng.normal(0.92, 0.04), 0.85, 1.0))
            severity = _score_to_severity(score)
        else:
            level = self._random.choice(_LEVELS_NORMAL)
            template = self._random.choice(_NORMAL_MESSAGES)
            content = template.format(job_id=self._rng.integers(1000, 9999))
            score = float(
                np.clip(
                    self._rng.normal(self._normal_score_mean, self._normal_score_std), 0.0, 0.49
                )
            )
            severity = "LOW"

        return LogEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=ts,
            node=node,
            level=level,
            component=component,
            content=content,
            is_anomaly=is_anomaly,
            anomaly_score=score,
            severity=severity,
        )


def _score_to_severity(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.70:
        return "HIGH"
    if score >= 0.55:
        return "MEDIUM"
    return "LOW"


# Mock LLM summaries for demo mode (when Ollama not available)
MOCK_SUMMARIES = [
    {
        "resumen": "Error de paridad TLB detectado en procesador. El sistema ejecutó corrección automática.",
        "severidad": "HIGH",
        "causa_probable": "Fallo intermitente en caché L1/L2 del procesador.",
        "accion_recomendada": "Monitorear el nodo 24h; si persiste, programar reemplazo de CPU.",
    },
    {
        "resumen": "Fallo de comunicación en socket CioStream. El proceso ciod no pudo leer el prefijo de mensaje.",
        "severidad": "CRITICAL",
        "causa_probable": "Corrupción de memoria o fallo de red en la interconexión del supercomputador.",
        "accion_recomendada": "Reiniciar el nodo afectado y revisar logs de red. Escalar a equipo de hardware.",
    },
    {
        "resumen": "Machine check NMI recibido. Posible error de hardware no recuperable.",
        "severidad": "CRITICAL",
        "causa_probable": "Error en DIMM de memoria o fallo en bus del sistema.",
        "accion_recomendada": "Evacuar jobs del nodo inmediatamente. Ejecutar diagnóstico de hardware.",
    },
    {
        "resumen": "Error de sistema de archivos detectado en dispositivo sdb. Posible corrupción de disco.",
        "severidad": "HIGH",
        "causa_probable": "Error de E/S en disco o inconsistencia en journal ext3.",
        "accion_recomendada": "Ejecutar fsck en modo mantenimiento. Revisar SMART del disco.",
    },
]
