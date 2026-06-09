"""Tests for EventGenerator — synthetic event streaming for demo."""

import numpy as np
import pandas as pd

from src.streaming.event_generator import EventGenerator, LogEvent, _score_to_severity


def test_next_event_returns_log_event():
    gen = EventGenerator(seed=1)
    ev = gen.next_event()
    assert isinstance(ev, LogEvent)
    assert ev.node != ""
    assert ev.level in ("INFO", "WARNING", "DEBUG", "ERROR", "FATAL")
    assert 0.0 <= ev.anomaly_score <= 1.0


def test_inject_anomaly_is_anomaly():
    gen = EventGenerator(seed=2)
    ev = gen.inject_anomaly()
    assert ev.is_anomaly is True
    assert ev.anomaly_score >= 0.85
    assert ev.severity == "CRITICAL"


def test_inject_anomaly_high_score():
    gen = EventGenerator(seed=3)
    scores = [gen.inject_anomaly().anomaly_score for _ in range(10)]
    assert all(s >= 0.85 for s in scores)


def test_normal_events_low_score():
    gen = EventGenerator(anomaly_rate=0.0, seed=4)
    scores = [gen.next_event().anomaly_score for _ in range(20)]
    assert all(s < 0.5 for s in scores)


def test_anomaly_rate_respected():
    gen = EventGenerator(anomaly_rate=1.0, seed=5)
    events = [gen.next_event() for _ in range(20)]
    assert all(e.is_anomaly for e in events)


def test_to_dict_has_required_keys():
    gen = EventGenerator(seed=6)
    d = gen.next_event().to_dict()
    for key in (
        "event_id",
        "timestamp",
        "node",
        "level",
        "component",
        "content",
        "is_anomaly",
        "anomaly_score",
        "severity",
    ):
        assert key in d


def test_stream_generator():
    gen = EventGenerator(seed=7)
    events = [next(gen.stream()) for _ in range(5)]
    assert len(events) == 5
    assert all(isinstance(e, LogEvent) for e in events)


def test_with_reference_df():
    rng = np.random.default_rng(42)
    ref = pd.DataFrame(
        {
            "anomaly_score": rng.uniform(0, 0.4, 100),
            "is_anomaly": [False] * 80 + [True] * 20,
        }
    )
    gen = EventGenerator(reference_df=ref, seed=8)
    assert gen._normal_score_mean < 0.5


def test_score_to_severity():
    assert _score_to_severity(0.90) == "CRITICAL"
    assert _score_to_severity(0.72) == "HIGH"
    assert _score_to_severity(0.60) == "MEDIUM"
    assert _score_to_severity(0.30) == "LOW"


def test_tick_increments():
    gen = EventGenerator(seed=9)
    assert gen._tick == 0
    gen.next_event()
    assert gen._tick == 1
    gen.inject_anomaly()
    assert gen._tick == 2
