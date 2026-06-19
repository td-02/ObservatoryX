from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SpanRecord(BaseModel):
    span_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    trace_id: str
    parent_span_id: Optional[str] = None
    operation: str
    start_ts: float
    end_ts: float
    latency_ms: float
    status: Literal["ok", "error", "timeout"]
    metadata: dict[str, Any] = Field(default_factory=dict)
