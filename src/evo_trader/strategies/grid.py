"""
Grid market-making strategy for crypto (24/7).

Places layered buy/sell levels around a rolling mean.  Grid spacing adapts to
volatility via ATR.  No EOD flatten.
All numeric values are Decimal — no float leakage.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from evo_trader.strategies.base import (
    Bar,
    ExitDecision,
    Position,
    Signal,
    Strategy,
    StrategyParams,
)


@dataclass
class GridParams(StrategyParams):
    """Tunable parameters for the grid market-making strategy.

    Attributes:
        grid_spacing_pct: Base distance between grid levels as a fraction of price.
        grid_levels: Number of grid levels on each side of center.
        atr_lookback: Number of bars used to compute ATR.
        atr_scale_factor: Scales grid_spacing by ATR to adapt to current volatility.
        max_position_fraction: Maximum fraction of allocated capital per grid level.
        mean_lookback: Rolling window for computing the grid center price.
        kelly_cap: Concentration cap override passed to the sizing module.
    """

    grid_spacing_pct: Decimal = Decimal("0.005")
    grid_levels: int = 5
    atr_lookback: int = 20
    atr_scale_factor: Decimal = Decimal("0.5")
    max_position_fraction: Decimal = Decimal("0.10")
    mean_lookback: int = 50
    kelly_cap: Decimal = Decimal("0.10")

    @classmethod
    def mutation_bounds(cls) -> dict[str, tuple[Decimal, Decimal]]:
        """Bounds for every numeric field the evolution engine may mutate."""
        return {
            "grid_spacing_pct": (Decimal("0.001"), Decimal("0.02")),
            "atr_scale_factor": (Decimal("0.1"), Decimal("2.0")),
            "max_position_fraction": (Decimal("0.02"), Decimal("0.20")),
            "kelly_cap": (Decimal("0.03"), Decimal("0.20")),
        }


def _compute_atr(bars: list[Bar], period: int) -> Decimal:
    """Compute simple ATR (average of high - low) over the last `period` bars."""
    if len(bars) < period:
        return Decimal("0")
    ranges = [b.high - b.low for b in bars[-period:]]
    return sum(ranges) / Decimal(period)


def _compute_mean(closes: list[Decimal], period: int) -> Decimal:
    """Compute rolling arithmetic mean of the last `period` closes."""
    if len(closes) < period:
        return Decimal("0")
    window = closes[-period:]
    return sum(window) / Decimal(period)


class GridStrategy(Strategy):
    """Volatility-adaptive grid market-maker for crypto.

    Computes a rolling mean as the grid center and places ``grid_levels`` buy
    levels below and ``grid_levels`` sell levels above it.  Effective grid
    spacing widens when ATR is elevated, preventing fill on low-significance
    ticks in choppy conditions.

    Args:
        symbol: Tradeable instrument (e.g. "BTC/USDT").
        params: Strategy parameter set.
        strategy_id: Optional explicit ID; auto-generated if omitted.
    """

    def __init__(
        self,
        symbol: str,
        params: GridParams | None = None,
        strategy_id: str | None = None,
    ) -> None:
        self._symbol = symbol
        self._params = params or GridParams()
        self._strategy_id = strategy_id or f"grid_{uuid.uuid4().hex[:8]}"

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
        return 15

    @property
    def is_day_trader(self) -> bool:
        return False

    @property
    def params(self) -> GridParams:
        return self._params

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, bars: list[Bar]) -> Signal | None:
        """Generate a grid entry signal.

        Returns a "long" signal when price is at or below the nearest buy level
        and a "short" (sell) signal when price is at or above the nearest sell
        level.  Returns None if there is insufficient data.

        Stop distance: 2× grid spacing below the triggered buy level (long) or
        above the triggered sell level (short).
        Take-profit distance: reversion to grid center.

        Args:
            bars: Historical price bars (oldest first).

        Returns:
            Signal or None.
        """
        p = self._params
        min_bars = max(p.mean_lookback, p.atr_lookback)
        if len(bars) < min_bars:
            return None

        closes = [b.close for b in bars]
        current_price = closes[-1]

        center = _compute_mean(closes, p.mean_lookback)
        if center == Decimal("0"):
            return None

        atr = _compute_atr(bars, p.atr_lookback)
        if atr == Decimal("0"):
            return None

        effective_spacing = p.grid_spacing_pct * (
            Decimal("1") + p.atr_scale_factor * (atr / current_price)
        )

        # Build buy and sell levels
        buy_levels = [
            center * (Decimal("1") - Decimal(i) * effective_spacing)
            for i in range(1, p.grid_levels + 1)
        ]
        sell_levels = [
            center * (Decimal("1") + Decimal(i) * effective_spacing)
            for i in range(1, p.grid_levels + 1)
        ]

        now = datetime.now(tz=timezone.utc).isoformat()

        # Long signal: price at or below nearest buy level
        nearest_buy = buy_levels[0]
        if current_price <= nearest_buy:
            # Stop: 2× spacing below the triggered level
            stop_distance = Decimal("2") * effective_spacing
            # TP: reversion to center
            tp_distance = (nearest_buy - center).copy_abs() / current_price
            strength = min(
                Decimal("1"),
                (nearest_buy - current_price) / (nearest_buy * effective_spacing),
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

        # Short signal: price at or above nearest sell level
        nearest_sell = sell_levels[0]
        if current_price >= nearest_sell:
            stop_distance = Decimal("2") * effective_spacing
            tp_distance = (center - nearest_sell).copy_abs() / current_price
            strength = min(
                Decimal("1"),
                (current_price - nearest_sell) / (nearest_sell * effective_spacing),
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
        """Evaluate exit conditions for an open grid position.

        No EOD flatten.  Checks hard stop, take-profit (mean reversion to
        center), and advances trailing stop in-place.

        Args:
            position: The open position.
            current_bar: Most recent completed bar.
            is_near_close: Ignored for crypto strategies.

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
