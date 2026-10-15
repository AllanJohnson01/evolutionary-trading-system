"""Tests for market state adapter — Alpaca data → candidates + rollout state."""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from evo_trader.horizon.alpaca_client import Bar
from evo_trader.horizon.market_state import (
    Candidate,
    MarketState,
    build_candidates,
    fetch_market_state,
)


class TestBuildCandidates:
    def test_no_position_excludes_exit(self):
        candidates = build_candidates(has_open_position=False)
        assert Candidate.EXIT not in candidates
        assert Candidate.ENTER_LONG in candidates
        assert Candidate.ENTER_SHORT in candidates
        assert Candidate.HOLD in candidates

    def test_open_position_includes_exit_and_resize(self):
        candidates = build_candidates(has_open_position=True)
        assert Candidate.EXIT in candidates
        assert Candidate.RESIZE in candidates
        assert Candidate.HEDGE in candidates

    def test_open_position_excludes_duplicate_entry(self):
        candidates = build_candidates(has_open_position=True)
        assert Candidate.ENTER_LONG not in candidates
        assert Candidate.ENTER_SHORT not in candidates


class TestFetchMarketState:
    def test_builds_state_from_client_bars(self):
        mock_client = MagicMock()
        mock_client.get_recent_bars.return_value = [
            Bar(
                timestamp="2026-01-01T00:00:00",
                open=Decimal("100"), high=Decimal("101"),
                low=Decimal("99"), close=Decimal("100.5"),
                volume=Decimal("1000"),
            )
        ]

        state = fetch_market_state(
            mock_client, symbol="AAPL", lookback_bars=10,
            has_open_position=False, position_notional=Decimal("0"),
        )

        assert state == MarketState(
            symbol="AAPL",
            bars=mock_client.get_recent_bars.return_value,
            has_open_position=False,
            position_notional=Decimal("0"),
        )
        mock_client.get_recent_bars.assert_called_once_with("AAPL", limit=10)

    def test_propagates_client_error(self):
        from evo_trader.horizon.alpaca_client import AlpacaClientError

        mock_client = MagicMock()
        mock_client.get_recent_bars.side_effect = AlpacaClientError("down")

        with pytest.raises(AlpacaClientError):
            fetch_market_state(
                mock_client, symbol="AAPL", lookback_bars=10,
                has_open_position=False, position_notional=Decimal("0"),
            )
