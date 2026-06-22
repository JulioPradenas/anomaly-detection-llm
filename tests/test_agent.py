"""Tests for LogAgent — LLM conversational agent with tools and memory."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.llm.agent import LogAgent, _build_message, _extract_tools_used

# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def labels_file(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "node": ["R02-M1-N0", "R02-M1-N0", "R23-M1-N8"],
            "severidad": ["CRITICAL", "HIGH", "MEDIUM"],
            "llm_label": [1, 1, 0],
            "timestamp": pd.to_datetime(["2005-06-03", "2005-06-04", "2005-06-05"]),
        }
    )
    path = tmp_path / "llm_confirmed.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def agent_no_llm(tmp_path, labels_file):
    a = LogAgent(labels_path=labels_file, features_path=tmp_path / "nonexistent.parquet")
    a._available = False  # skip Ollama check
    return a


# ── is_available / fallback ───────────────────────────────────────────────────


def test_agent_unavailable_returns_fallback(agent_no_llm):
    result = agent_no_llm.chat("¿Cuántas anomalías hay?", session_id="s1")
    assert result["session_id"] == "s1"
    assert result["tools_used"] == []
    assert "LLM no disponible" in result["response"]


def test_agent_is_available_false_when_ollama_down(tmp_path):
    a = LogAgent(features_path=tmp_path / "x.parquet", labels_path=tmp_path / "y.parquet")
    with patch("src.llm.agent.get_chat_model") as mock_factory:
        mock_factory.return_value.invoke.side_effect = ConnectionError("ollama down")
        a._available = None
        assert a.is_available is False


# ── tool logic ────────────────────────────────────────────────────────────────


def test_tool_query_anomaly_history_returns_count(agent_no_llm):
    tools = {t.name: t for t in agent_no_llm._build_tools()}
    result = tools["query_anomaly_history"].invoke({"node": "R02", "days_back": 7})
    assert "R02" in result
    assert "2" in result or "CRITICAL" in result


def test_tool_query_anomaly_history_no_data(tmp_path):
    a = LogAgent(labels_path=tmp_path / "missing.parquet")
    a._available = False
    tools = {t.name: t for t in a._build_tools()}
    result = tools["query_anomaly_history"].invoke({"node": "R02"})
    assert "Sin historial" in result or "no disponible" in result.lower()


def test_tool_get_anomaly_details_no_id_column(agent_no_llm):
    tools = {t.name: t for t in agent_no_llm._build_tools()}
    result = tools["get_anomaly_details"].invoke({"anomaly_id": "abc-123"})
    assert "abc-123" in result


def test_tool_compare_incidents_no_id_column(agent_no_llm):
    tools = {t.name: t for t in agent_no_llm._build_tools()}
    result = tools["compare_incidents"].invoke({"anomaly_id_1": "a1", "anomaly_id_2": "a2"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_tool_create_mock_ticket_returns_inc(agent_no_llm):
    tools = {t.name: t for t in agent_no_llm._build_tools()}
    result = tools["create_mock_ticket"].invoke(
        {
            "severity": "CRITICAL",
            "summary": "TLB errors on kernel",
            "node": "R02-M1-N0",
        }
    )
    assert "INC" in result
    assert "CRITICAL" in result
    assert "R02-M1-N0" in result
    assert "OPEN" in result


# ── session management ────────────────────────────────────────────────────────


def test_clear_session_nonexistent_returns_false(agent_no_llm):
    assert agent_no_llm.clear_session("nonexistent-session") is False


def test_clear_session_existing_returns_true(agent_no_llm):
    agent_no_llm._checkpointer.storage["fake-session"] = {"data": {}}
    assert agent_no_llm.clear_session("fake-session") is True
    assert "fake-session" not in agent_no_llm._checkpointer.storage


# ── helpers ───────────────────────────────────────────────────────────────────


def test_build_message_no_context():
    assert _build_message("hello", None) == "hello"


def test_build_message_with_context():
    ctx = {"node": "R02", "anomaly_score": 0.95, "timestamp": "2005-06-03"}
    msg = _build_message("analiza esto", ctx)
    assert "R02" in msg
    assert "0.950" in msg
    assert "analiza esto" in msg


def test_extract_tools_used_empty():
    assert _extract_tools_used([]) == []


def test_extract_tools_used_from_tool_messages():
    class FakeToolMsg:
        name = "query_anomaly_history"
        __class__ = type("ToolMessage", (), {"__name__": "ToolMessage"})()

    class NotTool:
        name = "query_anomaly_history"

    msgs = [FakeToolMsg()]
    # ToolMessage check uses __class__.__name__
    msgs[0].__class__.__name__ = "ToolMessage"
    result = _extract_tools_used(msgs)
    assert "query_anomaly_history" in result
