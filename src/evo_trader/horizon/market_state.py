"""
Market state adapter — Alpaca bars/position → the causal-entropic-optionality
skill's candidate-action and rollout vocabulary. Deliberately carries no
predictive, fundamental, or technical-indicator features: only OHLCV, which
is what's needed to define reachable future price paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from evo_trader.horizon.alpaca_client import AlpacaPaperClient, Bar


class Candidate(str, Enum):
    """Actions the decision loop may consider, given position state."""

    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"
    RESIZE = "resize"
    HOLD = "hold"
    HEDGE = "hedge"


def build_candidates(has_open_position: bool) -> list[Candidate]:
    """Only offer candidates that are actually legal given current position."""
    if has_open_position:
        return [Candidate.EXIT, Candidate.RESIZE, Candidate.HOLD, Candidate.HEDGE]
    return [Candidate.ENTER_LONG, Candidate.ENTER_SHORT, Candidate.HOLD]


@dataclass(frozen=True)
class MarketState:
    """Snapshot of OHLCV bars and position for one symbol."""

    symbol: str
    bars: list[Bar]
    has_open_position: bool
    position_notional: Decimal


def fetch_market_state(
    client: AlpacaPaperClient,
    symbol: str,
    lookback_bars: int,
    has_open_position: bool,
    position_notional: Decimal,
) -> MarketState:
    """Build a MarketState for ``symbol`` from the client's recent bars.

    Raises:
        AlpacaClientError: Propagated from ``client`` on any fetch failure.
            Callers must not catch-and-continue here; the decision loop
            decides what to do on failure.
    """
    bars = client.get_recent_bars(symbol, limit=lookback_bars)
    return MarketState(
        symbol=symbol,
        bars=bars,
        has_open_position=has_open_position,
        position_notional=position_notional,
    )
