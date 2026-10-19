"""
yfinance-based bars client for the horizon layer.

Drop-in replacement for ``AlpacaPaperClient.get_recent_bars()`` — satisfies
the ``BarsProvider`` Protocol in ``evo_trader.horizon.market_state`` without
any API credentials.
"""
from __future__ import annotations

from datetime import timezone
from decimal import Decimal

import yfinance as yf

from evo_trader.horizon.alpaca_client import Bar


class YFinanceBarsError(Exception):
    """Raised when yfinance fails to return bars — wraps all SDK exceptions."""


class YFinanceBarsClient:
    """Fetches minute OHLCV bars from Yahoo Finance for the Horizon layer.

    No credentials needed; instantiate with ``YFinanceBarsClient()``.
    Satisfies the ``BarsProvider`` Protocol expected by
    ``evo_trader.horizon.market_state.fetch_market_state``.
    """

    def __init__(self) -> None:
        pass  # no credentials

    def get_recent_bars(self, symbol: str, limit: int) -> list[Bar]:
        """Fetch the most recent *limit* minute bars for *symbol*.

        Args:
            symbol: Ticker symbol (e.g. ``"SPY"``).
            limit: Maximum number of bars to return (newest last).

        Returns:
            List of :class:`~evo_trader.horizon.alpaca_client.Bar` objects in
            chronological order (oldest first, newest last).  May be shorter
            than ``limit`` if Yahoo Finance has insufficient history.  Returns
            ``[]`` when no data is available.

        Raises:
            YFinanceBarsError: On any yfinance failure.
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="7d", interval="1m")
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

        return bars[-limit:] if len(bars) > limit else bars
