from __future__ import annotations

import time
from typing import Literal


class HeuristicAgent:
    def generate_signal(self, features_row: dict) -> dict:
        started_at = time.perf_counter()

        rsi = features_row.get("rsi_14")
        macd_signal = features_row.get("macd_signal")
        realized_vol_20 = features_row.get("realized_vol_20")

        if rsi is not None and macd_signal is not None and rsi < 30 and macd_signal > 0:
            signal = "BUY"
            confidence = 0.7
            reason = "RSI below 30 and MACD signal positive"
        elif rsi is not None and macd_signal is not None and rsi > 70 and macd_signal < 0:
            signal = "SELL"
            confidence = 0.7
            reason = "RSI above 70 and MACD signal negative"
        elif realized_vol_20 is not None and realized_vol_20 > 0.02:
            signal = "HOLD"
            confidence = 0.6
            reason = "Realized volatility above threshold"
        else:
            signal = "HOLD"
            confidence = 0.5
            reason = "Default hold"

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return {
            "signal": signal,
            "confidence": confidence,
            "reason": reason,
            "latency_ms": latency_ms,
        }
