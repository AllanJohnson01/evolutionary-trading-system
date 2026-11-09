"""
Shared strategy interface for the evolutionary trading system.

A Strategy produces signals and manages its own position state.
It does NOT size positions — the runner calls portfolio/sizing.py for that.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar


@dataclass
class StrategyParams(ABC):
    """Abstract base for all strategy parameter dataclasses.

    Every concrete params dataclass must implement ``mutation_bounds`` so the
    evolution engine knows which fields are tunable and what range is legal.
    """

    @classmethod
    @abstractmethod
    def mutation_bounds(cls) -> dict[str, tuple[Decimal, Decimal]]:
        """Return a mapping of field_name → (min_val, max_val) for every
        numeric field the evolution engine may mutate.

        Returns:
            Dict mapping field names to (min, max) Decimal tuples.
        """
        ...


@dataclass
class Bar:
    """A single OHLCV price bar.

    Attributes:
        timestamp: RFC3339 bar open time.
        open: Opening price.
        high: Session high.
        low: Session low.
        close: Closing price.
        volume: Volume traded during the bar.
    """

    timestamp: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass
class Signal:
    """A directional trade signal emitted by a strategy.

    Strategies emit signals; the runner is responsible for translating them
    into sized orders via portfolio/sizing.py.

    Attributes:
        strategy_id: Unique identifier of the emitting strategy.
        symbol: Tradeable instrument (e.g. "BTC/USDT", "AAPL").
        direction: Trade direction — "long", "short", or "flat".
        strength: Normalised signal confidence in [0, 1].
        stop_distance: Fraction of price to use as hard stop distance.
        take_profit_distance: Fraction of price to use as take-profit distance.
        timestamp: RFC3339 time the signal was generated.
    """

    strategy_id: str
    symbol: str
    direction: str          # "long" | "short" | "flat"
    strength: Decimal       # [0, 1]
    stop_distance: Decimal
    take_profit_distance: Decimal
    timestamp: str


@dataclass
class Position:
    """An open position tracked by a strategy.

    Attributes:
        strategy_id: Strategy that owns this position.
        symbol: Tradeable instrument.
        direction: "long" or "short".
        entry_price: Price at which the position was entered.
        size: Position size in base currency units.
        stop_price: Absolute hard stop price level.
        take_profit_price: Absolute take-profit price level.
        trailing_stop_distance: Current trailing stop distance (fraction of price).
        entry_time: RFC3339 time the position was entered.
    """

    strategy_id: str
    symbol: str
    direction: str
    entry_price: Decimal
    size: Decimal
    stop_price: Decimal
    take_profit_price: Decimal
    trailing_stop_distance: Decimal
    entry_time: str


@dataclass
class ExitDecision:
    """Result of a strategy exit check.

    Attributes:
        should_exit: Whether the position should be closed now.
        reason: One of "take_profit", "trailing_stop", "hard_stop",
                "eod_flatten", or "no_exit".
    """

    should_exit: bool
    reason: str     # "take_profit" | "trailing_stop" | "hard_stop" | "eod_flatten" | "no_exit"


class Strategy(ABC):
    """Abstract base for all trading strategies.

    A strategy receives bars, produces signals, and evaluates exit conditions
    for open positions.  It never calls external services and never sizes
    positions — those responsibilities belong to the runner and sizing module
    respectively.
    """

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier for this strategy instance."""
        ...

    @property
    @abstractmethod
    def symbol(self) -> str:
        """Tradeable instrument this strategy is configured for."""
        ...

    @property
    @abstractmethod
    def cadence_minutes(self) -> int:
        """Bar cadence in minutes (15 or 30)."""
        ...

    @property
    @abstractmethod
    def is_day_trader(self) -> bool:
        """True if this strategy must flatten equity positions before market close."""
        ...

    @property
    @abstractmethod
    def params(self) -> StrategyParams:
        """Current parameter set for this strategy instance."""
        ...

    @abstractmethod
    def generate_signal(self, bars: list[Bar]) -> Signal | None:
        """Evaluate bars and return a Signal, or None if there is no actionable signal.

        Args:
            bars: Historical price bars in chronological order (oldest first).
                  Must contain at least enough bars for the strategy's lookback.

        Returns:
            A Signal if the strategy has a view, or None for no opinion.
        """
        ...

    @abstractmethod
    def check_exits(
        self,
        position: Position,
        current_bar: Bar,
        is_near_close: bool,
    ) -> ExitDecision:
        """Decide whether an open position should be closed.

        Args:
            position: The open position to evaluate.
            current_bar: The most recent completed bar.
            is_near_close: True when the current time is within 15 minutes of
                           the equity market close (16:00 ET).

        Returns:
            ExitDecision indicating whether to exit and the reason.
        """
        ...
