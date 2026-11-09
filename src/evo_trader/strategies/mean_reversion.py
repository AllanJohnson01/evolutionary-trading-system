"""
30-minute z-score swing mean reversion strategy.

Designed for multi-day holds — no EOD flatten.
Enters when price deviates significantly from its rolling mean and targets
reversion back toward the mean.
All numeric values are Decimal — no float leakage.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional

from evo_trader.strategies.base import (
    Bar,
    ExitDecision,
    Position,
    Signal,
    Strategy,
    StrategyParams,
)


@dataclass
class MeanReversionParams(StrategyParams):
    """Tunable parameters for the z-score mean reversion strategy.

    Attributes:
        lookback_bars: Rolling window used to compute mean and standard deviation.
        entry_z: Enter when |z-score| >= entry_z (price is this many sigma from mean).
        exit_z: Exit (take-profit) when |z-score| reverts to <= exit_z.
        atr_lookback: Number of bars used to compute ATR for hard stop sizing.
        atr_multiplier_stop: ATR multiplier for the hard stop distance.
        atr_multiplier_trail: ATR multiplier for the trailing stop distance.
        kelly_cap: Concentration cap override passed to the sizing module.
    """

    lookback_bars: int = 40
    entry_z: Decimal = Decimal("2.0")
    exit_z: Decimal = Decimal("0.5")
    atr_lookback: int = 14
    atr_multiplier_stop: Decimal = Decimal("3.0")
    atr_multiplier_trail: Decimal = Decimal("2.0")
    kelly_cap: Decimal = Decimal("0.15")

    @classmethod
    def mutation_bounds(cls) -> dict[str, tuple[Decimal, Decimal]]:
        """Bounds for every numeric field the evolution engine may mutate."""
        return {
            "entry_z": (Decimal("1.0"), Decimal("4.0")),
            "exit_z": (Decimal("0.1"), Decimal("1.5")),
            "atr_multiplier_stop": (Decimal("1.5"), Decimal("6.0")),
            "atr_multiplier_trail": (Decimal("1.0"), Decimal("4.0")),
            "kelly_cap": (Decimal("0.05"), Decimal("0.25")),
        }


def _compute_mean_std(closes: list[Decimal], period: int) -> tuple[Decimal, Decimal]:
    """Compute rolling arithmetic mean and population std-dev over last `period` closes.

    Args:
        closes: List of closing prices (oldest first).
        period: Lookback period.

    Returns:
        (mean, std_dev) tuple, or (Decimal("0"), Decimal("0")) if insufficient data.
    """
    if len(closes) < period:
        return Decimal("0"), Decimal("0")
    window = closes[-period:]
    mean = sum(window) / Decimal(period)
    variance = sum((c - mean) ** 2 for c in window) / Decimal(period)
    # Use Decimal sqrt via integer Newton's method approach via float bridge
    # (float used only internally; result cast back immediately)
    variance_float = float(variance)
    if variance_float <= 0:
        return mean, Decimal("0")
    import math
    std = Decimal(str(math.sqrt(variance_float)))
    return mean, std


def _compute_atr(bars: list[Bar], period: int) -> Decimal:
    """Compute simple ATR (average of high - low) over the last `period` bars."""
    if len(bars) < period:
        return Decimal("0")
    ranges = [b.high - b.low for b in bars[-period:]]
    return sum(ranges) / Decimal(period)


class MeanReversionStrategy(Strategy):
    """30-minute z-score swing mean reversion.

    Enters long when price is ``entry_z`` standard deviations below the rolling
    mean, and short when ``entry_z`` sigma above.  Targets reversion to
    ``exit_z`` sigma from mean as take-profit.  Multi-day holds are permitted
    (``is_day_trader = False``).

    Args:
        symbol: Tradeable instrument.
        params: Strategy parameter set.
        strategy_id: Optional explicit ID; auto-generated if omitted.
    """

    def __init__(
        self,
        symbol: str,
        params: MeanReversionParams | None = None,
        strategy_id: str | None = None,
    ) -> None:
        self._symbol = symbol
        self._params = params or MeanReversionParams()
        self._strategy_id = strategy_id or f"mean_reversion_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def cadence_minutes(self) -> int:
        return 30

    @property
    def is_day_trader(self) -> bool:
        return False

    @property
    def params(self) -> MeanReversionParams:
        return self._params

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, bars: list[Bar]) -> Signal | None:
        """Generate a mean reversion signal.

        Requires at least ``max(lookback_bars, atr_lookback)`` bars.
        Returns None if insufficient data or standard deviation is zero
        (flat market — no valid z-score).

        Args:
            bars: Historical price bars (oldest first).

        Returns:
            Signal when |z| >= entry_z, otherwise None.
        """
        p = self._params
        min_bars = max(p.lookback_bars, p.atr_lookback)
        if len(bars) < min_bars:
            return None

        closes = [b.close for b in bars]
        current_close = closes[-1]

        mean, std = _compute_mean_std(closes, p.lookback_bars)
        if std == Decimal("0"):
            return None

        z = (current_close - mean) / std

        atr = _compute_atr(bars, p.atr_lookback)
        if atr == Decimal("0"):
            return None

        # Stop distance: ATR-scaled hard stop
        stop_distance = atr * p.atr_multiplier_stop / current_close

        now = datetime.now(tz=timezone.utc).isoformat()

        if z <= -p.entry_z:
            # Long entry: price is entry_z sigma below mean
            # TP distance: distance from current z back to exit_z (long side)
            # At exit_z below mean the price would be: mean - exit_z * std
            tp_price = mean - p.exit_z * std
            tp_distance = (tp_price - current_close).copy_abs() / current_close
            strength = min(
                Decimal("1"),
                ((-z) - p.entry_z) / p.entry_z,
            )
            return Signal(
                strategy_id=self._strategy_id,
                symbol=self._symbol,
                direction="long",
                strength=strength,
                stop_distance=stop_distance,
                take_profit_distance=tp_distance,
                timestamp=now,
            )

        if z >= p.entry_z:
            # Short entry: price is entry_z sigma above mean
            tp_price = mean + p.exit_z * std
            tp_distance = (current_close - tp_price).copy_abs() / current_close
            strength = min(
                Decimal("1"),
                (z - p.entry_z) / p.entry_z,
            )
            return Signal(
                strategy_id=self._strategy_id,
                symbol=self._symbol,
                direction="short",
                strength=strength,
                stop_distance=stop_distance,
                take_profit_distance=tp_distance,
                timestamp=now,
            )

        return None

    # ------------------------------------------------------------------
    # Exit management
    # ------------------------------------------------------------------

    def check_exits(
        self,
        position: Position,
        current_bar: Bar,
        is_near_close: bool,
    ) -> ExitDecision:
        """Evaluate exit conditions for an open mean-reversion position.

        No EOD flatten (multi-day holds).  Checks hard stop, take-profit
        (z reverts to exit_z), and advances trailing stop in-place.

        Args:
            position: The open position.
            current_bar: Most recent completed bar.
            is_near_close: Ignored — this strategy holds overnight.

        Returns:
            ExitDecision.
        """
        if position.direction == "long":
            if current_bar.low <= position.stop_price:
                return ExitDecision(should_exit=True, reason="hard_stop")
            if current_bar.high >= position.take_profit_price:
                return ExitDecision(should_exit=True, reason="take_profit")
            new_stop = current_bar.close - position.trailing_stop_distance * current_bar.close
            if new_stop > position.stop_price:
                position.stop_price = new_stop
        else:
            if current_bar.high >= position.stop_price:
                return ExitDecision(should_exit=True, reason="hard_stop")
            if current_bar.low <= position.take_profit_price:
                return ExitDecision(should_exit=True, reason="take_profit")
            new_stop = current_bar.close + position.trailing_stop_distance * current_bar.close
            if new_stop < position.stop_price:
                position.stop_price = new_stop

        return ExitDecision(should_exit=False, reason="no_exit")
