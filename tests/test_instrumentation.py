from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from instrumentation.metrics import LatencyMetrics
from instrumentation.tracer import SpanTracer


@pytest.fixture()
def tracer(tmp_path, monkeypatch):
    db_path = tmp_path / "traces.db"
    monkeypatch.setattr("instrumentation.tracer.Path", lambda *args, **kwargs: db_path.parent)
    monkeypatch.setattr("instrumentation.metrics.Path", lambda *args, **kwargs: db_path.parent)
    instance = SpanTracer()
    instance._conn.close()
    instance._conn = sqlite3.connect(db_path, check_same_thread=False)
    instance._conn.row_factory = sqlite3.Row
    instance._initialize_schema()
    instance.metrics = LatencyMetrics()
    instance.metrics._db_path = db_path
    return instance


def test_span_write_and_read_roundtrip(tracer):
    span_id, _ = tracer.start_span("llm_signal", "trace-1")
    tracer.end_span(span_id, "ok", {"foo": "bar"})
    df = tracer.metrics.load_spans(operation="llm_signal", last_n=10)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["span_id"] == span_id
    assert row["trace_id"] == "trace-1"
    assert row["status"] == "ok"


def test_is_degraded_returns_true_when_p99_exceeds_threshold(tracer):
    for index in range(5):
        span_id, _ = tracer.start_span("llm_signal", f"trace-{index}")
        tracer.end_span(span_id, "ok", {})
        with sqlite3.connect(tracer.metrics._db_path) as conn:
            conn.execute("UPDATE spans SET latency_ms = ? WHERE span_id = ?", (3000 + index * 100, span_id))
            conn.commit()
    assert tracer.metrics.is_degraded("llm_signal", p99_threshold_ms=2000.0) is True


def test_rolling_p99_with_synthetic_latency_data(tracer):
    for index in range(30):
        span_id, _ = tracer.start_span("llm_signal", f"trace-{index}")
        tracer.end_span(span_id, "ok", {})
        with sqlite3.connect(tracer.metrics._db_path) as conn:
            conn.execute("UPDATE spans SET latency_ms = ? WHERE span_id = ?", (100 + index, span_id))
            conn.commit()
    rolling = tracer.metrics.rolling_p99("llm_signal", window=20)
    assert rolling >= 100.0
    assert rolling <= 129.0
