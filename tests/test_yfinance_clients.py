"""Tests for yfinance-based bars clients — no live network."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evo_trader.tactical._yfinance_client import YFinanceBarsClient as TacticalYFClient
from evo_trader.tactical._yfinance_client import YFinanceBarsError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal yfinance-shaped DataFrame with *n* rows."""
    ts = pd.date_range("2026-10-01 14:30:00", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [1.0] * n,
            "High": [2.0] * n,
            "Low": [0.5] * n,
            "Close": [1.5] * n,
            "Volume": [100.0] * n,
            "Dividends": [0.0] * n,
            "Stock Splits": [0.0] * n,
        },
        index=ts,
    )


# ---------------------------------------------------------------------------
# Tactical YFinanceBarsClient
# ---------------------------------------------------------------------------

class TestTacticalYFinanceBarsClient:

    @patch("evo_trader.tactical._yfinance_client.yf.Ticker")
    def test_returns_bars_oldest_first(self, mock_ticker_cls):
        df = _make_df(5)
        mock_ticker_cls.return_value.history.return_value = df

        client = TacticalYFClient()
        bars = client.get_bars("SPY", 15, limit=3)

        assert len(bars) == 3
        # Oldest first — first bar is index 2 of the 5-row DF (we slice [-3:])
        assert bars[0].close == Decimal("1.5")
        assert bars[-1].close == Decimal("1.5")

    @patch("evo_trader.tactical._yfinance_client.yf.Ticker")
    def test_bar_fields_are_decimal(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(1)

        client = TacticalYFClient()
        bars = client.get_bars("SPY", 5, limit=1)

        assert len(bars) == 1
        bar = bars[0]
        assert isinstance(bar.open, Decimal)
        assert isinstance(bar.high, Decimal)
        assert isinstance(bar.low, Decimal)
        assert isinstance(bar.close, Decimal)
        assert isinstance(bar.volume, Decimal)

    @patch("evo_trader.tactical._yfinance_client.yf.Ticker")
    def test_timestamp_is_utc_iso(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(1)

        client = TacticalYFClient()
        bars = client.get_bars("SPY", 1, limit=1)

        assert len(bars) == 1
        ts = bars[0].timestamp
        # Should be parseable as a UTC-offset ISO string
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    @patch("evo_trader.tactical._yfinance_client.yf.Ticker")
    def test_empty_dataframe_returns_empty_list(self, mock_ticker_cls):
        empty_df = pd.DataFrame()
        mock_ticker_cls.return_value.history.return_value = empty_df

        client = TacticalYFClient()
        bars = client.get_bars("SPY", 15, limit=10)

        assert bars == []

    @patch("evo_trader.tactical._yfinance_client.yf.Ticker")
    def test_wraps_exceptions(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.side_effect = RuntimeError("network error")

        client = TacticalYFClient()
        with pytest.raises(YFinanceBarsError):
            client.get_bars("SPY", 15, limit=10)

    @patch("evo_trader.tactical._yfinance_client.yf.Ticker")
    def test_interval_mapping_15m(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(3)

        client = TacticalYFClient()
        client.get_bars("SPY", 15, limit=3)

        call_kwargs = mock_ticker_cls.return_value.history.call_args
        assert call_kwargs.kwargs.get("interval") == "15m" or (
            len(call_kwargs.args) > 1 and call_kwargs.args[1] == "15m"
        )
        # Verify via keyword
        assert mock_ticker_cls.return_value.history.call_args[1]["interval"] == "15m"

    @patch("evo_trader.tactical._yfinance_client.yf.Ticker")
    def test_interval_mapping_1h(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(3)

        client = TacticalYFClient()
        client.get_bars("SPY", 60, limit=3)

        assert mock_ticker_cls.return_value.history.call_args[1]["interval"] == "60m"

    @patch("evo_trader.tactical._yfinance_client.yf.Ticker")
    def test_limit_respected_when_df_larger(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(20)

        client = TacticalYFClient()
        bars = client.get_bars("SPY", 5, limit=7)

        assert len(bars) == 7

    @patch("evo_trader.tactical._yfinance_client.yf.Ticker")
    def test_returns_all_bars_when_df_smaller_than_limit(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(3)

        client = TacticalYFClient()
        bars = client.get_bars("SPY", 5, limit=100)

        assert len(bars) == 3
