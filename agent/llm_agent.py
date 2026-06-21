from __future__ import annotations

import json
import time
from typing import Any, Literal

try:
    import ollama
except ModuleNotFoundError:  # pragma: no cover
    ollama = None

from instrumentation.tracer import SpanTracer


class LLMAgent:
    def __init__(self, model: str, tracer: SpanTracer, timeout_ms: float = 3000.0) -> None:
        self.model = model
        self.tracer = tracer
        self.timeout_ms = timeout_ms

    def generate_signal(self, features_row: dict, trace_id: str) -> dict:
        if ollama is None:
            raise ModuleNotFoundError("ollama")
        started_at = time.perf_counter()
        span_id = ""
        timed_out = False

        rsi = features_row.get("rsi_14")
        macd_signal = features_row.get("macd_signal")
        realized_vol = features_row.get("realized_vol_20")
        returns = features_row.get("returns")
        last_returns = features_row.get("last_3_returns")
        if last_returns is None:
            last_returns = []

        system_prompt = (
            "Respond ONLY with valid JSON, no markdown, no explanation. "
            f"Current RSI: {rsi}. Current MACD: {macd_signal}. "
            f"Current realized_vol: {realized_vol}. Last 3 returns: {last_returns}."
        )
        user_prompt = (
            "Return a JSON object with keys signal, confidence, and reason. "
            f"Use the latest features: RSI={rsi}, MACD={macd_signal}, "
            f"realized_vol={realized_vol}, returns={returns}."
        )

        try:
            with self.tracer.trace("llm_signal", trace_id=trace_id) as traced_span_id:
                span_id = traced_span_id
                try:
                    response = ollama.chat(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        options={"timeout": self.timeout_ms / 1000.0},
                    )
                except Exception as exc:
                    if "timeout" in str(exc).lower():
                        timed_out = True
                        raise TimeoutError(str(exc)) from exc
                    raise

                content = response["message"]["content"]
                parsed = json.loads(content)
                signal = parsed["signal"]
                confidence = float(parsed["confidence"])
                reason = str(parsed["reason"])
        except TimeoutError:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "LLM request timed out",
                "latency_ms": latency_ms,
                "span_id": span_id,
                "timed_out": True,
            }
        except Exception:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "reason": "Invalid LLM response",
                "latency_ms": latency_ms,
                "span_id": span_id,
                "timed_out": timed_out,
            }

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "latency_ms": latency_ms,
            "span_id": span_id,
            "timed_out": timed_out,
        }
