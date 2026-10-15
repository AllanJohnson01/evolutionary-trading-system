"""
Alpaca paper-trading adapters for the tactical runner's ``BarsClient`` and
``OrderClient`` protocols.

Paper-only by construction: ``base_url`` must be the Alpaca paper host and the
SDK trading client is always created with ``paper=True``.

Known limitation: Alpaca tracks positions per account/symbol, not per strategy,
and the ``OrderClient`` protocol carries no strategy id, stop or take-profit on
``submit_order``. ``get_open_positions`` therefore cannot be reconstructed from
the broker and returns an empty list (so strategy exit rules do not fire).
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from urllib.parse import urlparse

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from evo_trader.strategies.base import Bar, Position

logger = logging.getLogger(__name__)

_PAPER_HOST = "paper-api.alpaca.markets"
_REGULAR_SESSION_MINUTES = 390


class TacticalAlpacaError(Exception):
    """Raised for any failure talking to Alpaca — raw SDK exceptions never escape."""


def _require_paper(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or parsed.hostname != _PAPER_HOST:
        raise ValueError(
            f"Refusing non-paper Alpaca base_url {base_url!r}; "
            f"only https://{_PAPER_HOST} is allowed"
        )


class AlpacaBarsClient:
    """Historical stock bars via alpaca-py, satisfying ``BarsClient``."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str,
        *,
        feed: DataFeed = DataFeed.IEX,
        data_client: StockHistoricalDataClient | None = None,
    ) -> None:
        _require_paper(base_url)
        self._feed = feed
        self._data_client = data_client or StockHistoricalDataClient(api_key, secret_key)

    def get_bars(self, symbol: str, timeframe_minutes: int, limit: int) -> list[Bar]:
        """Return up to ``limit`` bars, oldest first (newest last)."""
        # Alpaca returns the *oldest* bars from ``start`` when limited, so request a
        # generous window (covering nights/weekends) and trim to the newest ``limit``.
        sessions = math.ceil(limit * timeframe_minutes / _REGULAR_SESSION_MINUTES)
        start = datetime.now(timezone.utc) - timedelta(days=sessions * 2 + 4)
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
                start=start,
                feed=self._feed,
            )
            response = self._data_client.get_stock_bars(request)
            raw = response.data.get(symbol, [])
            return [
                Bar(
                    timestamp=b.timestamp.isoformat(),
                    open=Decimal(str(b.open)),
                    high=Decimal(str(b.high)),
                    low=Decimal(str(b.low)),
                    close=Decimal(str(b.close)),
                    volume=Decimal(str(b.volume)),
                )
                for b in raw[-limit:]
            ]
        except Exception as exc:
            raise TacticalAlpacaError(f"get_bars({symbol}) failed: {exc}") from exc


class AlpacaOrderClient:
    """Paper market orders via alpaca-py, satisfying ``OrderClient``."""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str,
        *,
        trading_client: TradingClient | None = None,
    ) -> None:
        _require_paper(base_url)
        self._trading_client = trading_client or TradingClient(
            api_key, secret_key, paper=True
        )
        self._warned_positions = False

    def submit_order(self, symbol: str, direction: str, notional: Decimal) -> str:
        """Submit a day-TIF market order sized by dollar notional; return the order id."""
        sides = {"long": OrderSide.BUY, "short": OrderSide.SELL}
        if direction not in sides:
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        try:
            order = self._trading_client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    notional=float(notional),
                    side=sides[direction],
                    time_in_force=TimeInForce.DAY,
                )
            )
            return str(order.id)
        except Exception as exc:
            raise TacticalAlpacaError(f"submit_order({symbol}) failed: {exc}") from exc

    def get_open_positions(self, strategy_id: str) -> list[Position]:
        """Always empty — see module docstring (no per-strategy attribution)."""
        if not self._warned_positions:
            self._warned_positions = True
            logger.warning(
                "Alpaca has no per-strategy position attribution; "
                "get_open_positions returns [] and strategy exits will not fire"
            )
        return []

    def close_position(self, strategy_id: str, symbol: str, reason: str) -> None:
        """Flatten the account-level position in ``symbol``."""
        try:
            self._trading_client.close_position(symbol)
        except Exception as exc:
            raise TacticalAlpacaError(f"close_position({symbol}) failed: {exc}") from exc
        logger.info("Closed %s for %s — %s", symbol, strategy_id, reason)
