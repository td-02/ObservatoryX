from __future__ import annotations

from types import SimpleNamespace

from agent.heuristic import HeuristicAgent
from scheduler.escalation import EscalationPolicy


class FakeTracer:
    def __init__(self, degraded: bool = False):
        self.metrics = SimpleNamespace(is_degraded=lambda *args, **kwargs: degraded)
        self.spans = []

    def start_span(self, operation, trace_id, parent_span_id=None):
        span_id = f"span-{len(self.spans) + 1}"
        self.spans.append((operation, trace_id, span_id))
        return span_id, 0.0

    def end_span(self, span_id, status, metadata=None):
        return {"span_id": span_id, "status": status, "metadata": metadata or {}}


class FakeHeuristicAgent:
    def __init__(self, signal="HOLD", confidence=0.5):
        self.signal = signal
        self.confidence = confidence

    def generate_signal(self, features_row):
        return {
            "signal": self.signal,
            "confidence": self.confidence,
            "reason": "heuristic",
            "latency_ms": 1.0,
        }


class FakeLLMAgent:
    def __init__(self):
        self.calls = 0

    def generate_signal(self, features_row, trace_id):
        self.calls += 1
        return {
            "signal": "BUY",
            "confidence": 0.9,
            "reason": "llm",
            "latency_ms": 2.0,
            "span_id": "llm-span",
            "timed_out": False,
        }


def test_high_volatility_suppresses_llm_escalation():
    tracer = FakeTracer(degraded=False)
    policy = EscalationPolicy(tracer=tracer)
    assert policy.should_escalate({"realized_vol_20": 0.03, "rsi_14": 20}) == (False, "high_volatility")


def test_degraded_llm_suppresses_escalation():
    tracer = FakeTracer(degraded=True)
    policy = EscalationPolicy(tracer=tracer)
    assert policy.should_escalate({"realized_vol_20": 0.01, "rsi_14": 20}) == (False, "llm_degraded")


def test_neutral_rsi_suppresses_escalation():
    tracer = FakeTracer(degraded=False)
    policy = EscalationPolicy(tracer=tracer)
    assert policy.should_escalate({"realized_vol_20": 0.01, "rsi_14": 52}) == (False, "rsi_neutral")


def test_normal_conditions_trigger_escalation():
    tracer = FakeTracer(degraded=False)
    policy = EscalationPolicy(tracer=tracer)
    assert policy.should_escalate({"realized_vol_20": 0.01, "rsi_14": 20}) == (True, "escalate_to_llm")
