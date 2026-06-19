from __future__ import annotations

from typing import Any

from agent.heuristic import HeuristicAgent
from agent.llm_agent import LLMAgent
from instrumentation.tracer import SpanTracer


class EscalationPolicy:
    def __init__(
        self,
        tracer: SpanTracer,
        p99_threshold_ms: float = 2000.0,
        min_confidence_delta: float = 0.1,
    ) -> None:
        self.tracer = tracer
        self.p99_threshold_ms = p99_threshold_ms
        self.min_confidence_delta = min_confidence_delta

    def should_escalate(self, features_row: dict) -> tuple[bool, str]:
        if self.tracer.metrics.is_degraded("llm_signal", self.p99_threshold_ms):
            return False, "llm_degraded"
        if features_row["realized_vol_20"] > 0.025:
            return False, "high_volatility"
        if abs(features_row["rsi_14"] - 50) < 5:
            return False, "rsi_neutral"
        return True, "escalate_to_llm"

    def decide(
        self,
        features_row: dict,
        trace_id: str,
        heuristic_agent: HeuristicAgent,
        llm_agent: LLMAgent,
    ) -> dict:
        heuristic_result = heuristic_agent.generate_signal(features_row)
        escalate, reason = self.should_escalate(features_row)

        span_id, _ = self.tracer.start_span("escalation_decision", trace_id)
        self.tracer.end_span(
            span_id,
            "ok",
            {
                "escalate": escalate,
                "reason": reason,
                "heuristic_signal": heuristic_result.get("signal"),
                "heuristic_confidence": heuristic_result.get("confidence"),
            },
        )

        if escalate:
            llm_result = llm_agent.generate_signal(features_row, trace_id)
            llm_result["escalated"] = True
            llm_result["escalation_reason"] = reason
            return llm_result

        heuristic_result["escalated"] = False
        heuristic_result["escalation_reason"] = reason
        return heuristic_result
