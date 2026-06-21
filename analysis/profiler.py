from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go


class ResultsProfiler:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.output_dir = Path(__file__).resolve().parents[1] / "paper"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self, db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        with sqlite3.connect(db_path) as conn:
            spans = pd.read_sql_query("SELECT * FROM spans", conn)
            trades = pd.read_sql_query("SELECT * FROM trades", conn)
        return spans, trades

    def _load_frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.load_all(str(self.db_path))

    def latency_report(self) -> dict[str, Any]:
        spans, _ = self._load_frames()
        if spans.empty:
            report = {"overall_pipeline_p99": 0.0, "per_operation": {}}
            pd.DataFrame([report]).to_csv(self.output_dir / "latency_report.csv", index=False)
            return report

        spans = spans.copy()
        spans["latency_ms"] = spans["latency_ms"].astype(float)
        per_operation: dict[str, dict[str, float]] = {}
        for operation, group in spans.groupby("operation"):
            latency = group["latency_ms"]
            per_operation[str(operation)] = {
                "p50": float(latency.quantile(0.50)),
                "p95": float(latency.quantile(0.95)),
                "p99": float(latency.quantile(0.99)),
                "mean": float(latency.mean()),
                "std": float(latency.std(ddof=0)),
                "timeout_rate": float((group["status"] == "timeout").mean()),
                "error_rate": float((group["status"] == "error").mean()),
            }

        report = {
            "overall_pipeline_p99": float(spans["latency_ms"].quantile(0.99)),
            "per_operation": per_operation,
        }

        rows = []
        for operation, metrics in per_operation.items():
            row = {"operation": operation, **metrics}
            rows.append(row)
        rows.append({"operation": "__overall__", "p99": report["overall_pipeline_p99"]})
        pd.DataFrame(rows).to_csv(self.output_dir / "latency_report.csv", index=False)
        return report

    def trading_report(self) -> dict[str, Any]:
        _, trades = self._load_frames()
        if trades.empty:
            report = {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "escalation_rate": 0.0,
                "llm_call_count": 0,
                "heuristic_only_count": 0,
            }
            pd.DataFrame([report]).to_csv(self.output_dir / "trading_report.csv", index=False)
            return report

        trades = trades.copy()
        trades["latency_ms"] = trades["latency_ms"].astype(float)
        trades["pnl_est"] = trades["pnl_est"].astype(float)
        trades["escalated"] = trades["escalated"].astype(int)

        pnl = trades["pnl_est"]
        equity_curve = pnl.cumsum()
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak.replace(0, pd.NA)
        drawdown = drawdown.fillna(0.0)
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        returns = pnl / 10000.0
        mean_return = float(returns.mean())
        std_return = float(returns.std(ddof=0))
        sharpe_ratio = 0.0
        if std_return > 0:
            sharpe_ratio = (mean_return / std_return) * (252**0.5) * (390**0.5)

        total_trades = int(len(trades))
        llm_call_count = int(trades["escalated"].sum())
        heuristic_only_count = int(total_trades - llm_call_count)

        report = {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "escalation_rate": float(llm_call_count / total_trades) if total_trades else 0.0,
            "llm_call_count": llm_call_count,
            "heuristic_only_count": heuristic_only_count,
        }
        pd.DataFrame([report]).to_csv(self.output_dir / "trading_report.csv", index=False)
        return report

    def plot_latency_distribution(self, save_path: str = "paper/latency_dist.html"):
        spans, trades = self._load_frames()
        if spans.empty:
            fig = go.Figure()
            fig.write_html(save_path, include_plotlyjs=True, full_html=True)
            return fig

        spans = spans.copy()
        spans["latency_ms"] = spans["latency_ms"].astype(float)
        merged = spans.merge(
            trades[["timestamp", "symbol", "escalated"]],
            how="left",
            left_on=["trace_id"],
            right_on=["timestamp"],
        ) if not trades.empty and "trace_id" in spans.columns else spans.assign(escalated=0)

        escalated = merged[merged["escalated"].fillna(0).astype(int) == 1]
        non_escalated = merged[merged["escalated"].fillna(0).astype(int) == 0]
        p50 = float(spans["latency_ms"].quantile(0.50))
        p95 = float(spans["latency_ms"].quantile(0.95))
        p99 = float(spans["latency_ms"].quantile(0.99))

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=non_escalated["latency_ms"], name="Non-escalated", opacity=0.65))
        fig.add_trace(go.Histogram(x=escalated["latency_ms"], name="Escalated", opacity=0.65))
        for value, name in [(p50, "P50"), (p95, "P95"), (p99, "P99")]:
            fig.add_vline(x=value, line_dash="dash", annotation_text=f"{name}: {value:.2f} ms")
        fig.update_layout(barmode="overlay", title="LLM Latency Distribution", xaxis_title="latency_ms")
        fig.write_html(save_path, include_plotlyjs=True, full_html=True)
        return fig

    def plot_rolling_p99(self, save_path: str = "paper/rolling_p99.html"):
        spans, _ = self._load_frames()
        if spans.empty:
            fig = go.Figure()
            fig.write_html(save_path, include_plotlyjs=True, full_html=True)
            return fig

        spans = spans.copy()
        spans["end_ts"] = pd.to_datetime(spans["end_ts"], unit="s", errors="coerce")
        llm_spans = spans[spans["operation"] == "llm_signal"].sort_values("end_ts")
        if llm_spans.empty:
            fig = go.Figure()
            fig.write_html(save_path, include_plotlyjs=True, full_html=True)
            return fig

        rolling = llm_spans["latency_ms"].astype(float).rolling(20, min_periods=1).quantile(0.99)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=llm_spans["end_ts"], y=rolling, mode="lines", name="Rolling P99"))
        fig.update_layout(title="Rolling 20-bar P99 LLM Latency", xaxis_title="time", yaxis_title="p99 latency_ms")
        fig.write_html(save_path, include_plotlyjs=True, full_html=True)
        return fig
