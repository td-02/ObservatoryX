from __future__ import annotations

import math
import os
from typing import Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


class PaperExecutor:
    def __init__(self) -> None:
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = os.getenv("ALPACA_BASE_URL", "")
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=True,
        )
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            url_override=self.base_url or None,
        )

    def _latest_price(self, symbol: str) -> float:
        latest = self.data_client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbol)
        )
        trade = latest[symbol]
        return float(trade.price)

    def submit_order(self, symbol: str, signal: dict, equity: float = 10000.0) -> dict:
        try:
            action = signal.get("signal", "HOLD")
            if action == "HOLD":
                return {"action": "hold"}

            if action == "BUY":
                latest_price = self._latest_price(symbol)
                qty = math.floor(equity * 0.05 / latest_price)
                if qty <= 0:
                    return {"action": "hold"}
                order = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                return {
                    "order_id": str(order.id),
                    "symbol": symbol,
                    "side": "buy",
                    "qty": qty,
                    "status": str(order.status),
                }

            if action == "SELL":
                positions = self.get_positions()
                position = positions.get(symbol)
                if not position:
                    return {"action": "hold"}
                qty = int(position.get("qty", 0))
                if qty <= 0:
                    return {"action": "hold"}
                order = self.trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                return {
                    "order_id": str(order.id),
                    "symbol": symbol,
                    "side": "sell",
                    "qty": qty,
                    "status": str(order.status),
                }

            return {"action": "hold"}
        except Exception as exc:
            return {"action": "error", "message": str(exc)}

    def get_portfolio_value(self) -> float:
        account = self.trading_client.get_account()
        return float(account.portfolio_value)

    def get_positions(self) -> dict[str, dict]:
        positions: dict[str, dict] = {}
        for position in self.trading_client.get_all_positions():
            positions[str(position.symbol)] = {
                "qty": str(position.qty),
                "side": str(getattr(position, "side", "")),
                "market_value": str(getattr(position, "market_value", "")),
            }
        return positions
