"""
15-minute volatility / opening-range breakout strategy for crypto (24/7).

Crypto symbols run continuously — there is no EOD flatten.
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
class BreakoutParams(StrategyParams):
    """Tunable parameters for the opening-range breakout strategy.

    Attributes:
        range_bars: Number of recent bars that define the reference range.
        breakout_multiplier: Close must exceed range_high × multiplier (long) or
            fall below range_low / multiplier (short) to trigger a signal.
        atr_lookback: Number of bars used to compute ATR.
        atr_multiplier_stop: ATR multiplier for the hard stop distance.
        atr_multiplier_tp: ATR multiplier for the take-profit distance.
        atr_multiplier_trail: ATR multiplier for the trailing stop distance.
        min_range_atr_ratio: Range size must be at least this fraction of ATR;
            smaller ranges are considered choppy and no signal is emitted.
        kelly_cap: Concentration cap override passed to the sizing module.
    """

    range_bars: int = 8
    breakout_multiplier: Decimal = Decimal("1.005")
    atr_lookback: int = 14
    atr_multiplier_stop: Decimal = Decimal("1.5")
    atr_multiplier_tp: Decimal = Decimal("2.5")
    atr_multiplier_trail: Decimal = Decimal("1.0")
    min_range_atr_ratio: Decimal = Decimal("0.5")
    kelly_cap: Decimal = Decimal("0.15")

    @classmethod
    def mutation_bounds(cls) -> dict[str, tuple[Decimal, Decimal]]:
        """Bounds for every numeric field the evolution engine may mutate."""
        return {
            "breakout_multiplier": (Decimal("1.001"), Decimal("1.02")),
            "atr_multiplier_stop": (Decimal("0.5"), Decimal("3.0")),
            "atr_multiplier_tp": (Decimal("1.0"), Decimal("5.0")),
            "atr_multiplier_trail": (Decimal("0.5"), Decimal("2.5")),
            "min_range_atr_ratio": (Decimal("0.1"), Decimal("1.5")),
            "kelly_cap": (Decimal("0.05"), Decimal("0.25")),
        }


def _compute_atr(bars: list[Bar], period: int) -> Decimal:
    """Compute simple ATR (average of high - low) over the last `period` bars.

    Args:
        bars: Price bars (oldest first).
        period: Lookback period.

    Returns:
        Average true range, or Decimal("0") if insufficient bars.
    """
    if len(bars) < period:
        return Decimal("0")
    ranges = [b.high - b.low for b in bars[-period:]]
    return sum(ranges) / Decimal(period)


class BreakoutStrategy(Strategy):
    """15-minute opening-range breakout for crypto markets.

    Identifies a reference range from the most recent ``range_bars`` bars and
    signals a breakout when price closes decisively above (long) or below
    (short) that range.  Validates range quality against ATR to avoid choppy
    market entries.

    Args:
        symbol: Tradeable instrument (e.g. "BTC/USDT").
        params: Strategy parameter set.
        strategy_id: Optional explicit ID; auto-generated if omitted.
    """

    def __init__(
        self,
        symbol: str,
        params: BreakoutParams | None = None,
        strategy_id: str | None = None,
    ) -> None:
        self._symbol = symbol
        self._params = params or BreakoutParams()
        self._strategy_id = strategy_id or f"breakout_{uuid.uuid4().hex[:8]}"

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
    def params(self) -> BreakoutParams:
        return self._params

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, bars: list[Bar]) -> Signal | None:
        """Generate a breakout signal from the bar series.

        Requires at least ``max(range_bars, atr_lookback) + 1`` bars.
        Returns None if insufficient data or range is too narrow.

        Args:
            bars: Historical price bars (oldest first).

        Returns:
            Signal on confirmed breakout, or None.
        """
        p = self._params
        # Need range_bars + 1 bars for the range window (excluding current) + current bar,
        # plus enough for ATR lookback.
        min_bars = max(p.range_bars + 1, p.atr_lookback) + 1
        if len(bars) < min_bars:
            return None

        atr = _compute_atr(bars, p.atr_lookback)
        if atr == Decimal("0"):
            return None

        # Exclude the current bar so the breakout close can exceed the range high.
        range_window = bars[-(p.range_bars + 1):-1]
        range_high = max(b.high for b in range_window)
        range_low = min(b.low for b in range_window)
        range_size = range_high - range_low

        # Reject choppy ranges
        if range_size < p.min_range_atr_ratio * atr:
            return None

        current_close = bars[-1].close
        stop_distance = atr * p.atr_multiplier_stop / current_close
        tp_distance = atr * p.atr_multiplier_tp / current_close

        now = datetime.now(tz=timezone.utc).isoformat()

        if current_close > range_high * p.breakout_multiplier:
            # Strength: proportional to how far above the range_high we are
            excess = (current_close - range_high) / range_high
            strength = min(Decimal("1"), excess / (p.breakout_multiplier - Decimal("1")))
            return Signal(
                strategy_id=self._strategy_id,
                symbol=self._symbol,
                direction="long",
                strength=strength,
                stop_distance=stop_distance,
                take_profit_distance=tp_distance,
                timestamp=now,
            )

        if current_close < range_low / p.breakout_multiplier:
            excess = (range_low - current_close) / range_low
            strength = min(Decimal("1"), excess / (p.breakout_multiplier - Decimal("1")))
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
        """Evaluate exit conditions for an open breakout position.

        No EOD flatten (crypto trades 24/7).  Checks hard stop, take-profit,
        and advances trailing stop in-place when price moves favorably.

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
            # Advance trailing stop upward
            new_stop = current_bar.close - position.trailing_stop_distance * current_bar.close
            if new_stop > position.stop_price:
                position.stop_price = new_stop
        else:
            if current_bar.high >= position.stop_price:
                return ExitDecision(should_exit=True, reason="hard_stop")
            if current_bar.low <= position.take_profit_price:
                return ExitDecision(should_exit=True, reason="take_profit")
            # Advance trailing stop downward
            new_stop = current_bar.close + position.trailing_stop_distance * current_bar.close
            if new_stop < position.stop_price:
                position.stop_price = new_stop

        return ExitDecision(should_exit=False, reason="no_exit")
