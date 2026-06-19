from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class LatencyMetrics:
    def __init__(self) -> None:
        self._db_path = Path(__file__).resolve().with_name("traces.db")

    def load_spans(self, operation: Optional[str] = None, last_n: int = 100) -> pd.DataFrame:
        if not self._db_path.exists():
            return pd.DataFrame()

        query = "SELECT * FROM spans"
        params: list[Any] = []
        if operation is not None:
            query += " WHERE operation = ?"
            params.append(operation)
        query += " ORDER BY end_ts DESC LIMIT ?"
        params.append(last_n)

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty and "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(
                lambda value: json.loads(value) if isinstance(value, str) and value else {}
            )
        return df

    def compute_percentiles(self, df: pd.DataFrame) -> dict[str, float]:
        if df.empty or "latency_ms" not in df.columns:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "std": 0.0}

        latency = df["latency_ms"].astype(float)
        return {
            "p50": float(latency.quantile(0.50)),
            "p95": float(latency.quantile(0.95)),
            "p99": float(latency.quantile(0.99)),
            "mean": float(latency.mean()),
            "std": float(latency.std(ddof=0)),
        }

    def is_degraded(self, operation: str, p99_threshold_ms: float = 2000.0) -> bool:
        df = self.load_spans(operation=operation, last_n=100)
        return self.compute_percentiles(df)["p99"] > p99_threshold_ms

    def rolling_p99(self, operation: str, window: int = 20) -> float:
        df = self.load_spans(operation=operation, last_n=window)
        if df.empty or "latency_ms" not in df.columns:
            return 0.0
        return float(df["latency_ms"].astype(float).quantile(0.99))
