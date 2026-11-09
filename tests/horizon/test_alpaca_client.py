"""Tests for AlpacaPaperClient — wraps alpaca-py, never leaks raw SDK errors."""
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from evo_trader.horizon.alpaca_client import (
    AlpacaClientError,
    AlpacaPaperClient,
    Bar,
    OrderResult,
)


class TestGetRecentBars:
    @patch("evo_trader.horizon.alpaca_client.StockHistoricalDataClient")
    def test_returns_bars_on_success(self, mock_data_client_cls):
        mock_bar = MagicMock(
            timestamp=MagicMock(isoformat=lambda: "2026-01-01T00:00:00"),
            open=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0,
        )
        mock_response = MagicMock()
        mock_response.data = {"AAPL": [mock_bar]}
        mock_instance = mock_data_client_cls.return_value
        mock_instance.get_stock_bars.return_value = mock_response

        client = AlpacaPaperClient(api_key="k", api_secret="s")
        bars = client.get_recent_bars("AAPL", limit=1)

        assert len(bars) == 1
        assert bars[0] == Bar(
            timestamp="2026-01-01T00:00:00",
            open=Decimal("100.0"), high=Decimal("101.0"),
            low=Decimal("99.0"), close=Decimal("100.5"),
            volume=Decimal("1000.0"),
        )

    @patch("evo_trader.horizon.alpaca_client.StockHistoricalDataClient")
    def test_wraps_sdk_exception(self, mock_data_client_cls):
        mock_instance = mock_data_client_cls.return_value
        mock_instance.get_stock_bars.side_effect = RuntimeError("timeout")

        client = AlpacaPaperClient(api_key="k", api_secret="s")
        with pytest.raises(AlpacaClientError, match="timeout"):
            client.get_recent_bars("AAPL", limit=1)


class TestSubmitOrder:
    @patch("evo_trader.horizon.alpaca_client.TradingClient")
    def test_returns_order_result_on_success(self, mock_trading_client_cls):
        mock_order = MagicMock(id="order-123", status=MagicMock(value="filled"))
        mock_instance = mock_trading_client_cls.return_value
        mock_instance.submit_order.return_value = mock_order

        client = AlpacaPaperClient(api_key="k", api_secret="s")
        result = client.submit_order("AAPL", side="buy", notional=Decimal("500"))

        assert result == OrderResult(order_id="order-123", status="filled")

    @patch("evo_trader.horizon.alpaca_client.TradingClient")
    def test_wraps_sdk_exception_on_submit(self, mock_trading_client_cls):
        mock_instance = mock_trading_client_cls.return_value
        mock_instance.submit_order.side_effect = RuntimeError("rejected")

        client = AlpacaPaperClient(api_key="k", api_secret="s")
        with pytest.raises(AlpacaClientError, match="rejected"):
            client.submit_order("AAPL", side="buy", notional=Decimal("500"))


class TestHardening:
    @patch("evo_trader.horizon.alpaca_client.StockHistoricalDataClient")
    def test_malformed_response_is_wrapped(self, mock_data_client_cls):
        mock_response = MagicMock()
        mock_response.data = {}  # symbol missing -> KeyError inside parsing
        mock_data_client_cls.return_value.get_stock_bars.return_value = mock_response

        client = AlpacaPaperClient(api_key="k", api_secret="s")
        with pytest.raises(AlpacaClientError, match="get_recent_bars failed"):
            client.get_recent_bars("AAPL", limit=1)

    @patch("evo_trader.horizon.alpaca_client.TradingClient")
    def test_rejects_unknown_side_without_submitting(self, mock_trading_client_cls):
        client = AlpacaPaperClient(api_key="k", api_secret="s")
        with pytest.raises(ValueError, match="side must be"):
            client.submit_order("AAPL", side="short", notional=Decimal("500"))  # type: ignore[arg-type]
        mock_trading_client_cls.return_value.submit_order.assert_not_called()
