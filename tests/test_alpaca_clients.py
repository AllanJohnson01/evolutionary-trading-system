"""
Tests for the tactical Alpaca adapters.

Fake SDK clients are injected — no network access, no alpaca-py calls.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from evo_trader.tactical._alpaca_clients import (
    AlpacaBarsClient,
    AlpacaOrderClient,
    TacticalAlpacaError,
)

PAPER_URL = "https://paper-api.alpaca.markets/v2"


class FakeDataClient:
    def __init__(self, bars=None, error=None):
        self._bars = bars or []
        self._error = error
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        if self._error:
            raise self._error
        return SimpleNamespace(data={request.symbol_or_symbols: self._bars})


class FakeTradingClient:
    def __init__(self, error=None):
        self._error = error
        self.orders = []
        self.closed = []

    def submit_order(self, request):
        if self._error:
            raise self._error
        self.orders.append(request)
        return SimpleNamespace(id="order-123")

    def close_position(self, symbol):
        if self._error:
            raise self._error
        self.closed.append(symbol)


def _sdk_bar(minute: int, close: str):
    return SimpleNamespace(
        timestamp=datetime(2026, 10, 15, 14, minute, tzinfo=timezone.utc),
        open=1.0,
        high=2.0,
        low=0.5,
        close=float(close),
        volume=100.0,
    )


# ---------------------------------------------------------------------------
# Paper-only guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://api.alpaca.markets",
        "https://api.alpaca.markets/v2",
        "https://paper-api.alpaca.markets.evil.com",
        "http://paper-api.alpaca.markets",
        "",
    ],
)
def test_non_paper_base_url_rejected(url):
    with pytest.raises(ValueError, match="paper"):
        AlpacaOrderClient("k", "s", url, trading_client=FakeTradingClient())
    with pytest.raises(ValueError, match="paper"):
        AlpacaBarsClient("k", "s", url, data_client=FakeDataClient())


def test_paper_base_url_with_and_without_v2_accepted():
    AlpacaOrderClient("k", "s", PAPER_URL, trading_client=FakeTradingClient())
    AlpacaOrderClient(
        "k", "s", "https://paper-api.alpaca.markets", trading_client=FakeTradingClient()
    )


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


def test_get_bars_returns_decimal_bars_oldest_first_trimmed_to_limit():
    sdk_bars = [_sdk_bar(m, "10.5") for m in range(5)]
    client = AlpacaBarsClient("k", "s", PAPER_URL, data_client=FakeDataClient(sdk_bars))

    bars = client.get_bars("SPY", 15, limit=3)

    assert len(bars) == 3
    # newest last: the trimmed window is the final three SDK bars
    assert [b.timestamp for b in bars] == [b.timestamp.isoformat() for b in sdk_bars[-3:]]
    assert bars[0].close == Decimal("10.5")
    assert isinstance(bars[0].volume, Decimal)


def test_get_bars_requests_requested_timeframe_and_symbol():
    fake = FakeDataClient([_sdk_bar(0, "1")])
    AlpacaBarsClient("k", "s", PAPER_URL, data_client=fake).get_bars("SPY", 30, limit=10)

    req = fake.requests[0]
    assert req.symbol_or_symbols == "SPY"
    assert req.timeframe.amount == 30
    assert req.start is not None  # never rely on the SDK's default start


def test_get_bars_wraps_sdk_errors():
    client = AlpacaBarsClient(
        "k", "s", PAPER_URL, data_client=FakeDataClient(error=RuntimeError("boom"))
    )
    with pytest.raises(TacticalAlpacaError, match="boom"):
        client.get_bars("SPY", 15, limit=5)


def test_get_bars_missing_symbol_returns_empty():
    fake = FakeDataClient()
    fake.get_stock_bars = lambda request: SimpleNamespace(data={})
    client = AlpacaBarsClient("k", "s", PAPER_URL, data_client=fake)
    assert client.get_bars("SPY", 15, limit=5) == []


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction,side", [("long", "buy"), ("short", "sell")])
def test_submit_order_maps_direction_and_returns_order_id(direction, side):
    trading = FakeTradingClient()
    client = AlpacaOrderClient("k", "s", PAPER_URL, trading_client=trading)

    order_id = client.submit_order("SPY", direction, Decimal("1234.56"))

    assert order_id == "order-123"
    req = trading.orders[0]
    assert req.symbol == "SPY"
    assert req.side.value == side
    assert req.notional == pytest.approx(1234.56)


@pytest.mark.parametrize("direction", ["flat", "buy", ""])
def test_submit_order_rejects_unknown_direction(direction):
    trading = FakeTradingClient()
    client = AlpacaOrderClient("k", "s", PAPER_URL, trading_client=trading)
    with pytest.raises(ValueError):
        client.submit_order("SPY", direction, Decimal("100"))
    assert trading.orders == []


@pytest.mark.parametrize("notional", [Decimal("0"), Decimal("-5")])
def test_submit_order_rejects_non_positive_notional(notional):
    trading = FakeTradingClient()
    client = AlpacaOrderClient("k", "s", PAPER_URL, trading_client=trading)
    with pytest.raises(ValueError):
        client.submit_order("SPY", "long", notional)
    assert trading.orders == []


def test_submit_order_wraps_sdk_errors():
    client = AlpacaOrderClient(
        "k", "s", PAPER_URL, trading_client=FakeTradingClient(error=RuntimeError("rejected"))
    )
    with pytest.raises(TacticalAlpacaError, match="rejected"):
        client.submit_order("SPY", "long", Decimal("100"))


def test_close_position_flattens_symbol():
    trading = FakeTradingClient()
    client = AlpacaOrderClient("k", "s", PAPER_URL, trading_client=trading)
    client.close_position("momentum-1", "SPY", "stop hit")
    assert trading.closed == ["SPY"]


def test_close_position_wraps_sdk_errors():
    client = AlpacaOrderClient(
        "k", "s", PAPER_URL, trading_client=FakeTradingClient(error=RuntimeError("nope"))
    )
    with pytest.raises(TacticalAlpacaError, match="nope"):
        client.close_position("momentum-1", "SPY", "stop hit")


def test_get_open_positions_returns_empty_when_no_positions_registered():
    client = AlpacaOrderClient("k", "s", PAPER_URL, trading_client=FakeTradingClient())
    assert client.get_open_positions("a") == []
    assert client.get_open_positions("b") == []


# ---------------------------------------------------------------------------
# In-memory position tracking and trade log
# ---------------------------------------------------------------------------


class FakeTradingClientWithPrice:
    """Trading client that returns a known current_price from get_all_positions."""

    def __init__(self, symbol: str, price: str):
        self.orders = []
        self.closed: list[str] = []
        self._symbol = symbol
        self._price = price

    def submit_order(self, request):
        self.orders.append(request)
        return SimpleNamespace(id="order-abc")

    def get_all_positions(self):
        return [
            SimpleNamespace(symbol=self._symbol, current_price=self._price)
        ]

    def close_position(self, symbol):
        self.closed.append(symbol)


def test_submit_order_with_strategy_id_registers_position(tmp_path):
    trading = FakeTradingClientWithPrice("SPY", "100.00")
    client = AlpacaOrderClient(
        "k", "s", PAPER_URL, trading_client=trading,
        trade_log_path=tmp_path / "trade_log.jsonl",
    )
    client.submit_order("SPY", "long", Decimal("1000"), strategy_id="intraday_momentum_abc")

    positions = client.get_open_positions("intraday_momentum_abc")
    assert len(positions) == 1
    assert positions[0].symbol == "SPY"
    assert positions[0].direction == "long"
    assert positions[0].entry_price == Decimal("100.00")


def test_close_position_writes_trade_log_with_realized_pnl(tmp_path):
    log_path = tmp_path / "trade_log.jsonl"
    trading = FakeTradingClientWithPrice("SPY", "105.00")
    client = AlpacaOrderClient(
        "k", "s", PAPER_URL, trading_client=trading,
        trade_log_path=log_path,
    )
    client.set_agent_id("intraday_momentum_abc", "agent-uuid-1234")
    # Register entry at $100
    trading._price = "100.00"
    client.submit_order("SPY", "long", Decimal("1000"), strategy_id="intraday_momentum_abc")
    # Exit at $105
    trading._price = "105.00"
    client.close_position("intraday_momentum_abc", "SPY", "take_profit")

    assert log_path.exists()
    import json
    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    e = entries[0]
    assert e["strategy_id"] == "intraday_momentum_abc"
    assert e["agent_id"] == "agent-uuid-1234"
    assert e["symbol"] == "SPY"
    assert e["side"] == "long"
    assert Decimal(e["realized_pnl"]) > Decimal("0")
    assert e["exit_reason"] == "take_profit"
    # Position should be cleared
    assert client.get_open_positions("intraday_momentum_abc") == []


def test_close_position_writes_log_even_without_matched_position(tmp_path):
    log_path = tmp_path / "trade_log.jsonl"
    trading = FakeTradingClientWithPrice("SPY", "100.00")
    client = AlpacaOrderClient(
        "k", "s", PAPER_URL, trading_client=trading,
        trade_log_path=log_path,
    )
    client.close_position("intraday_momentum_abc", "SPY", "eod_flatten")
    assert log_path.exists()
