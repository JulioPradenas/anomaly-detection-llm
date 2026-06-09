"""Tests for LogSummarizer — Ollama is mocked, never required in CI."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.llm.summarizer import LogSummarizer, _parse_json_response, build_anomaly_context


@pytest.fixture
def sample_window_df():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2005-06-04 00:24:32", "2005-06-04 00:24:33"]),
        "node": ["R23-M1-N8", "R23-M1-N8"],
        "component": ["KERNEL", "APP"],
        "content": ["memory error detected", "ciod: failed to read"],
        "severity_score": [2, 4],
    })


def test_fallback_when_ollama_unavailable():
    """LogSummarizer must return a safe fallback dict when Ollama is down."""
    with patch("src.llm.summarizer.LogSummarizer._init_chain") as mock_init:
        mock_init.side_effect = lambda: None
        summarizer = LogSummarizer()
        summarizer._available = False

    result = summarizer.summarize({"node": "R1", "timestamp": "2005-06-04"})
    assert result["severidad"] == "UNKNOWN"
    assert "LLM not available" in result["resumen"]
    assert "response_time_ms" in result


def test_summarize_calls_chain():
    """When Ollama is available, summarize() must call the LangChain chain."""
    fake_response = MagicMock()
    fake_response.content = '{"resumen": "test", "severidad": "HIGH", "causa_probable": "x", "accion_recomendada": "y"}'

    with patch("src.llm.summarizer.LogSummarizer._init_chain"):
        summarizer = LogSummarizer()
        summarizer._available = True
        summarizer._chain = MagicMock()
        summarizer._chain.invoke.return_value = fake_response

    context = {"node": "R1", "timestamp": "2005-06-04", "total_events": 10,
               "error_count": 5, "error_rate": "50.0%", "warning_count": 2,
               "fatal_count": 1, "components": "KERNEL", "avg_severity": 2.5,
               "sample_messages": "  - error msg", "anomaly_score": 0.8}

    result = summarizer.summarize(context)
    assert result["severidad"] == "HIGH"
    assert result["resumen"] == "test"
    assert result["response_time_ms"] >= 0


def test_parse_json_valid():
    raw = '{"resumen": "ok", "severidad": "LOW", "causa_probable": "x", "accion_recomendada": "y"}'
    result = _parse_json_response(raw)
    assert result["severidad"] == "LOW"


def test_parse_json_with_markdown_fence():
    raw = '```json\n{"resumen": "ok", "severidad": "MEDIUM", "causa_probable": "x", "accion_recomendada": "y"}\n```'
    result = _parse_json_response(raw)
    assert result["severidad"] == "MEDIUM"


def test_parse_json_invalid_returns_fallback():
    result = _parse_json_response("not json at all")
    assert result["severidad"] == "UNKNOWN"


def test_build_anomaly_context(sample_window_df):
    ctx = build_anomaly_context(
        df_window=sample_window_df,
        anomaly_score=0.75,
        node="R23-M1-N8",
        timestamp="2005-06-04 00:24:32",
    )
    assert ctx["total_events"] == 2
    assert ctx["error_count"] == 2  # severity >= 2: score=2 (ERROR) + score=4 (FATAL) = 2
    assert ctx["fatal_count"] == 1
    assert ctx["node"] == "R23-M1-N8"
    assert ctx["anomaly_score"] == 0.75
