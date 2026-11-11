"""
yfinance-based bars client for the tactical layer.

Implements the ``BarsClient`` Protocol from ``evo_trader.tactical.runner``
using Yahoo Finance data — no API key required.  Keeps alpaca-py out of
the data path so the free paper account's 401s no longer block the runner.
"""
from __future__ import annotations

import math
from datetime import timezone
from decimal import Decimal

import yfinance as yf

from evo_trader.strategies.base import Bar

# ---------------------------------------------------------------------------
# Interval mapping: tactical timeframe_minutes → yfinance interval string
# ---------------------------------------------------------------------------
_INTERVAL_MAP: dict[int, str] = {
    1: "1m",
    5: "5m",
    15: "15m",
    30: "30m",
    60: "60m",
}
_DEFAULT_INTERVAL = "1d"

# yfinance only supports up to 60 days of history for minute-level intervals.
_MINUTE_MAX_DAYS = 59


class YFinanceBarsError(Exception):
    """Raised when yfinance fails to return bars — wraps all SDK exceptions."""


class YFinanceBarsClient:
    """Fetches OHLCV bars from Yahoo Finance.

    No credentials needed; instantiate with ``YFinanceBarsClient()``.
    Satisfies the ``BarsClient`` Protocol expected by
    ``evo_trader.tactical.runner.StrategyRunner``.
    """

    def __init__(self) -> None:
        pass  # no credentials

    def get_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        limit: int,
    ) -> list[Bar]:
        """Fetch up to *limit* recent OHLCV bars, oldest first.

        Args:
            symbol: Ticker symbol (e.g. ``"SPY"``).
            timeframe_minutes: Bar interval in minutes.  Mapped to a yfinance
                interval string; any value not in the map defaults to ``"1d"``.
            limit: Maximum number of bars to return.

        Returns:
            List of :class:`~evo_trader.strategies.base.Bar` objects in
            chronological order (oldest first, newest last).  May be shorter
            than ``limit`` if Yahoo Finance has insufficient history.  Returns
            ``[]`` when no data is available.

        Raises:
            YFinanceBarsError: On any yfinance failure.
        """
        interval = _INTERVAL_MAP.get(timeframe_minutes, _DEFAULT_INTERVAL)
        is_intraday = timeframe_minutes < 60

        # Calculate how many calendar days we need to cover *limit* bars.
        # Each trading day has ~390 minutes; double it and add buffer days.
        days = max(7, math.ceil(limit * timeframe_minutes / 390) * 2 + 4)
        if is_intraday:
            days = min(days, _MINUTE_MAX_DAYS)

        period = f"{days}d"

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
        except Exception as exc:
            raise YFinanceBarsError(
                f"yfinance fetch failed for {symbol!r}: {exc}"
            ) from exc

        if hist is None or hist.empty:
            return []

        bars: list[Bar] = []
        try:
            for bar_ts, row in hist.iterrows():
                utc_iso = bar_ts.astimezone(timezone.utc).isoformat()
                bars.append(
                    Bar(
                        timestamp=utc_iso,
                        open=Decimal(str(row["Open"])),
                        high=Decimal(str(row["High"])),
                        low=Decimal(str(row["Low"])),
                        close=Decimal(str(row["Close"])),
                        volume=Decimal(str(row["Volume"])),
                    )
                )
        except Exception as exc:
            raise YFinanceBarsError(
                f"Failed to parse yfinance response for {symbol!r}: {exc}"
            ) from exc

        # Return the *limit* most-recent bars (oldest-first within that window).
        return bars[-limit:] if len(bars) > limit else bars
