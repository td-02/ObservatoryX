from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
from uuid import uuid4

from instrumentation.span import SpanRecord


class SpanTracer:
    def __init__(self) -> None:
        db_path = Path(__file__).resolve().with_name("traces.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._spans: dict[str, tuple[str, float, str, Optional[str]]] = {}
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    operation TEXT NOT NULL,
                    start_ts REAL NOT NULL,
                    end_ts REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def start_span(
        self,
        operation: str,
        trace_id: str,
        parent_span_id: Optional[str] = None,
    ) -> tuple[str, float]:
        span_id = uuid4().hex
        start_ts = time.time()
        with self._lock:
            self._spans[span_id] = (operation, start_ts, trace_id, parent_span_id)
        return span_id, start_ts

    def end_span(
        self,
        span_id: str,
        status: str,
        metadata: dict[str, Any] | None = None,
    ) -> SpanRecord:
        metadata = metadata or {}
        end_ts = time.time()
        with self._lock:
            operation, start_ts, trace_id, parent_span_id = self._spans.pop(span_id)
            latency_ms = (end_ts - start_ts) * 1000.0
            record = SpanRecord(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                operation=operation,
                start_ts=start_ts,
                end_ts=end_ts,
                latency_ms=latency_ms,
                status=status,
                metadata=metadata,
            )
            self._conn.execute(
                """
                INSERT INTO spans (
                    span_id, trace_id, parent_span_id, operation,
                    start_ts, end_ts, latency_ms, status, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.span_id,
                    record.trace_id,
                    record.parent_span_id,
                    record.operation,
                    record.start_ts,
                    record.end_ts,
                    record.latency_ms,
                    record.status,
                    json.dumps(record.metadata),
                ),
            )
            self._conn.commit()
        return record

    @contextmanager
    def trace(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> Iterator[str]:
        trace_id = trace_id or uuid4().hex
        span_id, _ = self.start_span(operation, trace_id, parent_span_id)
        try:
            yield span_id
        except TimeoutError:
            self.end_span(span_id, "timeout", {})
            raise
        except Exception:
            self.end_span(span_id, "error", {})
            raise
        else:
            self.end_span(span_id, "ok", {})
