"""Tests for the horizon-layer yfinance bars client — no live network."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
import pytest

from evo_trader.horizon._yfinance_client import YFinanceBarsClient, YFinanceBarsError


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(n: int = 5) -> pd.DataFrame:
    ts = pd.date_range("2026-10-01 14:30:00", periods=n, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [10.0] * n,
            "High": [12.0] * n,
            "Low": [9.0] * n,
            "Close": [11.0] * n,
            "Volume": [500.0] * n,
            "Dividends": [0.0] * n,
            "Stock Splits": [0.0] * n,
        },
        index=ts,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHorizonYFinanceBarsClient:

    @patch("evo_trader.horizon._yfinance_client.yf.Ticker")
    def test_get_recent_bars_returns_bars(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(5)

        client = YFinanceBarsClient()
        bars = client.get_recent_bars("SPY", limit=3)

        assert len(bars) == 3
        bar = bars[0]
        assert isinstance(bar.open, Decimal)
        assert bar.close == Decimal("11.0")

    @patch("evo_trader.horizon._yfinance_client.yf.Ticker")
    def test_get_recent_bars_timestamp_utc(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(1)

        client = YFinanceBarsClient()
        bars = client.get_recent_bars("SPY", limit=1)

        assert len(bars) == 1
        parsed = datetime.fromisoformat(bars[0].timestamp)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset().total_seconds() == 0

    @patch("evo_trader.horizon._yfinance_client.yf.Ticker")
    def test_get_recent_bars_empty_df_returns_empty_list(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = pd.DataFrame()

        client = YFinanceBarsClient()
        bars = client.get_recent_bars("SPY", limit=10)

        assert bars == []

    @patch("evo_trader.horizon._yfinance_client.yf.Ticker")
    def test_get_recent_bars_wraps_error(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.side_effect = ConnectionError("timeout")

        client = YFinanceBarsClient()
        with pytest.raises(YFinanceBarsError):
            client.get_recent_bars("SPY", limit=10)

    @patch("evo_trader.horizon._yfinance_client.yf.Ticker")
    def test_uses_1m_interval_and_7d_period(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(3)

        client = YFinanceBarsClient()
        client.get_recent_bars("SPY", limit=3)

        call_kwargs = mock_ticker_cls.return_value.history.call_args[1]
        assert call_kwargs["interval"] == "1m"
        assert call_kwargs["period"] == "7d"

    @patch("evo_trader.horizon._yfinance_client.yf.Ticker")
    def test_limit_applied(self, mock_ticker_cls):
        mock_ticker_cls.return_value.history.return_value = _make_df(20)

        client = YFinanceBarsClient()
        bars = client.get_recent_bars("SPY", limit=5)

        assert len(bars) == 5
