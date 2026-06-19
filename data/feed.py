from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame


UNIVERSE = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "JPM", "BAC", "SPY"]


class DataFeedError(Exception):
    def __init__(self, message: str, symbol: str) -> None:
        super().__init__(f"{message} (symbol={symbol})")
        self.message = message
        self.symbol = symbol


@dataclass
class AlpacaFeed:
    api_key: str
    secret_key: str
    base_url: str

    def __init__(self) -> None:
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = os.getenv("ALPACA_BASE_URL", "")
        self._client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            url_override=self.base_url or None,
        )

    def _with_retries(self, action, symbol: str):
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return action()
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise DataFeedError(str(exc), symbol) from exc
        raise DataFeedError(str(last_error) if last_error else "Unknown error", symbol)

    def get_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        def action() -> pd.DataFrame:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
                end=datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc),
            )
            bars = self._client.get_stock_bars(request)
            df = bars.df.reset_index()
            if df.empty:
                return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            df = df.rename(
                columns={
                    "timestamp": "timestamp",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )
            return df[["timestamp", "open", "high", "low", "close", "volume"]]

        return self._with_retries(action, symbol)

    def get_latest_bar(self, symbol: str) -> dict[str, Any]:
        def action() -> dict[str, Any]:
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            latest = self._client.get_stock_latest_trade(request)
            bar = latest[symbol]
            return {
                "timestamp": bar.timestamp,
                "open": bar.price,
                "high": bar.price,
                "low": bar.price,
                "close": bar.price,
                "volume": getattr(bar, "size", 0),
            }

        return self._with_retries(action, symbol)
