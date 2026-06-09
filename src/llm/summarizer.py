"""LLM-based summarizer for detected anomalies using Ollama + LangChain."""

import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

import structlog

logger = structlog.get_logger(__name__)

_FALLBACK_RESPONSE = {
    "resumen": "LLM not available — summary skipped.",
    "severidad": "UNKNOWN",
    "causa_probable": "LLM not available",
    "accion_recomendada": "Review logs manually",
    "response_time_ms": 0.0,
}


class LogSummarizer:
    """Summarizes IT log anomalies using a local LLM via Ollama.

    Falls back gracefully if Ollama is not running — the detection pipeline
    continues uninterrupted.
    """

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._chain: Any | None = None
        self._available = False
        self._init_chain()

    def _init_chain(self) -> None:
        try:
            from langchain_ollama import ChatOllama

            from src.llm.prompts import anomaly_summary_prompt

            llm = ChatOllama(model=self.model, base_url=self.base_url, temperature=0.1)
            self._chain = anomaly_summary_prompt | llm
            self._available = True
            logger.info("ollama_connected", model=self.model)
        except Exception as exc:
            logger.warning("ollama_unavailable", error=str(exc))
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def summarize(self, anomaly_context: dict[str, Any]) -> dict[str, Any]:
        """Generate a natural-language summary for a detected anomaly.

        Args:
            anomaly_context: Dict with keys: node, timestamp, total_events,
                error_count, error_rate, warning_count, fatal_count,
                components, avg_severity, sample_messages, anomaly_score.

        Returns:
            Dict with keys: resumen, severidad, causa_probable,
                accion_recomendada, response_time_ms.
        """
        if not self._available or self._chain is None:
            return dict(_FALLBACK_RESPONSE)

        t0 = time.monotonic()
        try:
            response = self._chain.invoke(anomaly_context)
            elapsed_ms = (time.monotonic() - t0) * 1000

            content = response.content if hasattr(response, "content") else str(response)
            result = _parse_json_response(content)
            result["response_time_ms"] = round(elapsed_ms, 1)

            logger.info(
                "anomaly_summarized",
                node=anomaly_context.get("node"),
                severity=result.get("severidad"),
                response_time_ms=result["response_time_ms"],
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            logger.error("summarize_failed", error=str(exc))
            fallback = dict(_FALLBACK_RESPONSE)
            fallback["response_time_ms"] = round(elapsed_ms, 1)
            return fallback


def build_anomaly_context(
    df_window: "pd.DataFrame",  # type: ignore[name-defined]
    anomaly_score: float,
    node: str,
    timestamp: str,
    n_sample_messages: int = 5,
) -> dict[str, Any]:
    """Build the context dict required by LogSummarizer.summarize().

    Args:
        df_window: Slice of the BGL DataFrame covering ±5min around the anomaly.
        anomaly_score: Raw score from AnomalyDetector.score_samples().
        node: Node identifier where the anomaly was detected.
        timestamp: ISO timestamp string of the detection.
        n_sample_messages: Number of representative log lines to include.
    """
    total = len(df_window)
    error_count = int((df_window["severity_score"] >= 2).sum())
    warning_count = int((df_window["severity_score"] == 1).sum())
    fatal_count = int((df_window["severity_score"] >= 3).sum())
    error_rate = f"{error_count / total:.1%}" if total > 0 else "0.0%"
    avg_severity = round(float(df_window["severity_score"].mean()), 2) if total > 0 else 0.0

    components = ", ".join(df_window["component"].value_counts().head(5).index.astype(str).tolist())
    sample_msgs = df_window["content"].dropna().head(n_sample_messages).tolist()
    sample_messages = "\n".join(f"  - {m}" for m in sample_msgs)

    return {
        "node": node,
        "timestamp": timestamp,
        "total_events": total,
        "error_count": error_count,
        "error_rate": error_rate,
        "warning_count": warning_count,
        "fatal_count": fatal_count,
        "components": components,
        "avg_severity": avg_severity,
        "sample_messages": sample_messages,
        "anomaly_score": anomaly_score,
    }


def _parse_json_response(content: str) -> dict[str, Any]:
    """Extract JSON from LLM response.

    llama3.2 sometimes wraps JSON in markdown fences or adds preamble text.
    We try several strategies before giving up.
    """
    cleaned = content.strip()

    # 1. Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        cleaned = cleaned.strip()

    # 2. Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Extract first {...} block from the response (handles preamble/postamble text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    # 4. Nothing worked — return raw text so at least resumen is visible
    return {
        "resumen": cleaned[:300],
        "severidad": "UNKNOWN",
        "causa_probable": "Could not parse LLM response as JSON",
        "accion_recomendada": "Review logs manually",
    }
