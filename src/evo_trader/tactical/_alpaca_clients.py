"""
Alpaca paper-trading adapters for the tactical runner's ``BarsClient`` and
``OrderClient`` protocols.

Paper-only by construction: ``base_url`` must be the Alpaca paper host and the
SDK trading client is always created with ``paper=True``.

Position tracking: Alpaca tracks positions per account/symbol, not per
strategy. ``AlpacaOrderClient`` therefore maintains an in-memory per-strategy
position registry. ``submit_order`` records the entry; ``close_position``
computes realized P&L and appends a record to ``trade_log.jsonl``.
``get_open_positions`` returns the in-memory registry for a given strategy_id.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
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
    """Paper market orders via alpaca-py, satisfying ``OrderClient``.

    Maintains an in-memory per-strategy position registry so that
    ``get_open_positions`` returns live entries and strategy exit logic fires.
    When a position is closed ``close_position`` queries Alpaca for the current
    market price, computes realized P&L, and appends a record to
    ``trade_log_path`` (default ``state/trade_log.jsonl``).
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        base_url: str,
        *,
        trading_client: TradingClient | None = None,
        trade_log_path: Path | str | None = None,
    ) -> None:
        _require_paper(base_url)
        self._trading_client = trading_client or TradingClient(
            api_key, secret_key, paper=True
        )
        # strategy_id → list of open Position objects
        self._open_positions: dict[str, list[Position]] = {}
        # strategy_id → agent_id (populated when submit_order is called with context)
        self._agent_ids: dict[str, str] = {}
        # Path for the trade outcome log
        if trade_log_path is None:
            trade_log_path = Path(__file__).resolve().parents[4] / "state" / "trade_log.jsonl"
        self._trade_log_path = Path(trade_log_path)

    def set_agent_id(self, strategy_id: str, agent_id: str) -> None:
        """Record the stable agent_id for a strategy so it appears in trade log entries."""
        self._agent_ids[strategy_id] = agent_id

    def submit_order(
        self,
        symbol: str,
        direction: str,
        notional: Decimal,
        *,
        strategy_id: str | None = None,
        stop_price: Decimal | None = None,
        take_profit_price: Decimal | None = None,
        trailing_stop_distance: Decimal | None = None,
    ) -> str:
        """Submit a day-TIF market order sized by dollar notional; return the order id.

        When ``strategy_id`` is provided the resulting position is registered
        in the in-memory registry so subsequent ``get_open_positions`` calls
        return it and exit rules can fire.
        """
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
            order_id = str(order.id)
        except Exception as exc:
            raise TacticalAlpacaError(f"submit_order({symbol}) failed: {exc}") from exc

        # Register the position locally for exit tracking.
        if strategy_id:
            entry_price = self._get_current_price(symbol)
            if entry_price and entry_price > Decimal("0"):
                size = notional / entry_price
                # Derive stop/TP/trail from notional if not supplied (safe defaults).
                sp = stop_price or (
                    entry_price * Decimal("0.98") if direction == "long" else entry_price * Decimal("1.02")
                )
                tp = take_profit_price or (
                    entry_price * Decimal("1.03") if direction == "long" else entry_price * Decimal("0.97")
                )
                trail = trailing_stop_distance or Decimal("0.01")
                pos = Position(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    size=size,
                    stop_price=sp,
                    take_profit_price=tp,
                    trailing_stop_distance=trail,
                    entry_time=datetime.now(timezone.utc).isoformat(),
                )
                self._open_positions.setdefault(strategy_id, []).append(pos)
                logger.info(
                    "Registered local position: %s %s entry=%.4f size=%.4f order=%s",
                    strategy_id, symbol, entry_price, size, order_id,
                )

        return order_id

    def _get_current_price(self, symbol: str) -> Decimal | None:
        """Fetch the latest trade price for a symbol via Alpaca; return None on failure."""
        try:
            positions = self._trading_client.get_all_positions()
            for p in positions:
                if p.symbol == symbol:
                    return Decimal(str(p.current_price))
        except Exception:
            pass
        return None

    def get_open_positions(self, strategy_id: str) -> list[Position]:
        """Return locally-tracked open positions for the given strategy."""
        return list(self._open_positions.get(strategy_id, []))

    def close_position(self, strategy_id: str, symbol: str, reason: str) -> None:
        """Flatten the account-level position in ``symbol`` and log the trade outcome."""
        exit_price: Decimal | None = None
        exit_time = datetime.now(timezone.utc).isoformat()

        try:
            # Fetch exit price before closing so we capture the fill level.
            exit_price = self._get_current_price(symbol)
            self._trading_client.close_position(symbol)
        except Exception as exc:
            raise TacticalAlpacaError(f"close_position({symbol}) failed: {exc}") from exc

        logger.info("Closed %s for %s — %s", symbol, strategy_id, reason)

        # Find and remove the local position record for this strategy+symbol.
        positions = self._open_positions.get(strategy_id, [])
        matched: Position | None = None
        remaining: list[Position] = []
        for p in positions:
            if p.symbol == symbol and matched is None:
                matched = p
            else:
                remaining.append(p)
        if remaining:
            self._open_positions[strategy_id] = remaining
        else:
            self._open_positions.pop(strategy_id, None)

        if matched is None or exit_price is None or exit_price <= Decimal("0"):
            # Log a best-effort entry even without prices.
            self._append_trade_log({
                "timestamp": exit_time,
                "agent_id": self._agent_ids.get(strategy_id, ""),
                "strategy_id": strategy_id,
                "symbol": symbol,
                "side": matched.direction if matched else "unknown",
                "entry_price": str(matched.entry_price) if matched else None,
                "exit_price": str(exit_price) if exit_price else None,
                "entry_time": matched.entry_time if matched else None,
                "exit_time": exit_time,
                "size": str(matched.size) if matched else None,
                "realized_pnl": None,
                "exit_reason": reason,
                "note": "price_unavailable",
            })
            return

        size = matched.size
        if matched.direction == "long":
            realized_pnl = (exit_price - matched.entry_price) * size
        else:
            realized_pnl = (matched.entry_price - exit_price) * size

        entry: dict[str, Any] = {
            "timestamp": exit_time,
            "agent_id": self._agent_ids.get(strategy_id, ""),
            "strategy_id": strategy_id,
            "symbol": symbol,
            "side": matched.direction,
            "entry_price": str(matched.entry_price),
            "exit_price": str(exit_price),
            "entry_time": matched.entry_time,
            "exit_time": exit_time,
            "size": str(size),
            "realized_pnl": str(realized_pnl.quantize(Decimal("0.01"))),
            "exit_reason": reason,
        }
        self._append_trade_log(entry)
        logger.info(
            "Trade closed: %s %s entry=%.4f exit=%.4f pnl=%.2f reason=%s",
            strategy_id, symbol, matched.entry_price, exit_price, realized_pnl, reason,
        )

    def _append_trade_log(self, entry: dict[str, Any]) -> None:
        """Append a JSON record to the trade outcome log."""
        self._trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._trade_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
