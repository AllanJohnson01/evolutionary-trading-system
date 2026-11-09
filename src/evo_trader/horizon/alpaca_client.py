"""
Thin wrapper around alpaca-py's paper-trading clients. No module in this
package imports alpaca-py directly except this one, so a future SDK swap
touches only this file.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

OrderSideName = Literal["buy", "sell"]


class AlpacaClientError(Exception):
    """Raised for any failure talking to Alpaca — never let a raw SDK
    exception escape this module."""


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar with Decimal prices to avoid float drift."""

    timestamp: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class OrderResult:
    """Acknowledgement of a submitted order."""

    order_id: str
    status: str


class AlpacaPaperClient:
    """Paper-trading-only client — base_url is hardcoded to the paper
    endpoint by passing paper=True to both underlying clients."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        self._data_client = StockHistoricalDataClient(api_key, api_secret)
        self._trading_client = TradingClient(api_key, api_secret, paper=True)

    def get_recent_bars(self, symbol: str, limit: int) -> list[Bar]:
        """Fetch the most recent minute bars for ``symbol``.

        Raises:
            AlpacaClientError: On any SDK failure or unexpected response shape.
        """
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=limit,
            )
            response = self._data_client.get_stock_bars(request)
            return [
                Bar(
                    timestamp=bar.timestamp.isoformat(),
                    open=Decimal(str(bar.open)),
                    high=Decimal(str(bar.high)),
                    low=Decimal(str(bar.low)),
                    close=Decimal(str(bar.close)),
                    volume=Decimal(str(bar.volume)),
                )
                for bar in response.data[symbol]
            ]
        except Exception as exc:
            raise AlpacaClientError(f"get_recent_bars failed: {exc}") from exc

    def submit_order(
        self, symbol: str, side: OrderSideName, notional: Decimal
    ) -> OrderResult:
        """Submit a day-TIF market order sized by dollar notional.

        Raises:
            ValueError: If ``side`` is not ``"buy"`` or ``"sell"``.
            AlpacaClientError: On any SDK failure.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        try:
            request = MarketOrderRequest(
                symbol=symbol,
                notional=float(notional),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._trading_client.submit_order(request)
            return OrderResult(order_id=str(order.id), status=order.status.value)
        except Exception as exc:
            raise AlpacaClientError(f"submit_order failed: {exc}") from exc
