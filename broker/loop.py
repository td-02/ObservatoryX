from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from agent.heuristic import HeuristicAgent
from data.features import compute_features
from scheduler.escalation import EscalationPolicy
from instrumentation.tracer import SpanTracer


class TradingLoop:
    def __init__(self) -> None:
        self.console = Console()
        self.db_path = Path(__file__).resolve().parents[1] / "instrumentation" / "traces.db"

    def _ensure_trade_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                escalated INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                pnl_est REAL NOT NULL
            )
            """
        )
        conn.commit()

    def _log_trade(
        self,
        conn: sqlite3.Connection,
        timestamp: str,
        symbol: str,
        signal: str,
        escalated: bool,
        latency_ms: float,
        pnl_est: float,
    ) -> None:
        conn.execute(
            """
            INSERT INTO trades (timestamp, symbol, signal, escalated, latency_ms, pnl_est)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, symbol, signal, int(escalated), latency_ms, pnl_est),
        )
        conn.commit()

    def _simulate_fill(self, row: dict, signal: dict) -> dict:
        close_price = float(row.get("close", 0.0))
        qty = 0
        if signal.get("signal") == "BUY":
            qty = 1
        elif signal.get("signal") == "SELL":
            qty = 1
        return {
            "action": "simulated",
            "fill_price": close_price,
            "qty": qty,
            "status": "filled",
        }

    def run(
        self,
        symbols: list[str],
        bars_df: dict[str, pd.DataFrame],
        tracer: SpanTracer,
        mode: str = "paper",
        model: str = "mistral",
        timeout_ms: float = 3000.0,
    ):
        heuristic_agent = HeuristicAgent()
        policy = EscalationPolicy(tracer=tracer)
        executor = None
        llm_agent = None
        if mode == "paper":
            from broker.executor import PaperExecutor
            from agent.llm_agent import LLMAgent

            executor = PaperExecutor()
            llm_agent = LLMAgent(model=model, tracer=tracer, timeout_ms=timeout_ms)

        with sqlite3.connect(self.db_path) as conn:
            self._ensure_trade_table(conn)

            combined_rows: list[dict[str, Any]] = []
            for symbol in symbols:
                frame = compute_features(bars_df[symbol]).copy()
                frame["symbol"] = symbol
                combined_rows.extend(frame.to_dict("records"))

            combined_rows.sort(key=lambda row: row.get("timestamp"))

            for row in combined_rows:
                symbol = str(row["symbol"])
                trace_id = f"{symbol}-{row['timestamp']}"
                if mode == "backtest":
                    decision = heuristic_agent.generate_signal(row)
                    decision["escalated"] = False
                    decision["escalation_reason"] = "backtest_simulation"
                else:
                    decision = policy.decide(row, trace_id, heuristic_agent, llm_agent)
                result = {"action": "hold"}
                if mode == "paper":
                    result = executor.submit_order(symbol, decision)
                elif mode == "backtest":
                    result = self._simulate_fill(row, decision)
                pnl_est = 0.0
                if decision.get("signal") == "BUY":
                    pnl_est = float(row.get("close", 0.0)) * float(result.get("qty", 0))
                elif decision.get("signal") == "SELL":
                    pnl_est = float(row.get("close", 0.0)) * float(result.get("qty", 0))

                self._log_trade(
                    conn,
                    str(row.get("timestamp")),
                    symbol,
                    str(decision.get("signal")),
                    bool(decision.get("escalated")),
                    float(decision.get("latency_ms", 0.0)),
                    pnl_est,
                )

                table = Table(title="Trading Loop")
                table.add_column("timestamp")
                table.add_column("symbol")
                table.add_column("signal")
                table.add_column("escalated")
                table.add_column("latency_ms")
                table.add_column("pnl_est")
                table.add_row(
                    str(row.get("timestamp")),
                    symbol,
                    str(decision.get("signal")),
                    str(bool(decision.get("escalated"))),
                    f"{float(decision.get('latency_ms', 0.0)):.2f}",
                    f"{pnl_est:.2f}",
                )
                self.console.print(table)
