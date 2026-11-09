"""
15-minute intraday momentum day trader for equities and crypto.

Equity positions are flattened before market close (is_day_trader = True).
All numeric values are Decimal — no float leakage.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import ClassVar

from evo_trader.strategies.base import (
    Bar,
    ExitDecision,
    Position,
    Signal,
    Strategy,
    StrategyParams,
)


@dataclass
class IntradayMomentumParams(StrategyParams):
    """Tunable parameters for the intraday momentum strategy.

    All numeric thresholds and multipliers live here — none are hardcoded in
    strategy logic.

    Attributes:
        lookback_bars: RSI / momentum / ATR lookback window in bars.
        rsi_oversold: RSI level below which a long signal is considered (RSI crosses above).
        rsi_overbought: RSI level above which a short signal is considered.
        momentum_threshold: Minimum rate-of-change magnitude to confirm a signal.
        atr_multiplier_stop: ATR multiplier applied to derive the hard stop distance.
        atr_multiplier_tp: ATR multiplier applied to derive the take-profit distance.
        atr_multiplier_trail: ATR multiplier applied to derive the trailing stop distance.
        min_volume_ratio: Minimum current volume / average volume required.
        kelly_cap: Concentration cap override for this strategy passed to sizing.
    """

    lookback_bars: int = 20
    rsi_oversold: Decimal = Decimal("30")
    rsi_overbought: Decimal = Decimal("70")
    momentum_threshold: Decimal = Decimal("0.005")
    atr_multiplier_stop: Decimal = Decimal("2.0")
    atr_multiplier_tp: Decimal = Decimal("3.0")
    atr_multiplier_trail: Decimal = Decimal("1.5")
    min_volume_ratio: Decimal = Decimal("1.2")
    kelly_cap: Decimal = Decimal("0.15")

    @classmethod
    def mutation_bounds(cls) -> dict[str, tuple[Decimal, Decimal]]:
        """Bounds for every numeric field the evolution engine may mutate."""
        return {
            "rsi_oversold": (Decimal("15"), Decimal("40")),
            "rsi_overbought": (Decimal("60"), Decimal("85")),
            "momentum_threshold": (Decimal("0.001"), Decimal("0.02")),
            "atr_multiplier_stop": (Decimal("1.0"), Decimal("4.0")),
            "atr_multiplier_tp": (Decimal("1.5"), Decimal("6.0")),
            "atr_multiplier_trail": (Decimal("0.5"), Decimal("3.0")),
            "min_volume_ratio": (Decimal("1.0"), Decimal("3.0")),
            "kelly_cap": (Decimal("0.05"), Decimal("0.25")),
        }


def _compute_rsi(closes: list[Decimal], period: int) -> Decimal:
    """Compute RSI using Wilder smoothing over `period` bars.

    Args:
        closes: List of closing prices (oldest first).
        period: Lookback period in bars.

    Returns:
        RSI value in [0, 100], or Decimal("50") if insufficient data.
    """
    if len(closes) < period + 1:
        return Decimal("50")

    gains: list[Decimal] = []
    losses: list[Decimal] = []
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta >= Decimal("0"):
            gains.append(delta)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(-delta)

    avg_gain = sum(gains) / Decimal(period)
    avg_loss = sum(losses) / Decimal(period)

    # Wilder smoothing over remaining bars
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = delta if delta >= Decimal("0") else Decimal("0")
        loss = -delta if delta < Decimal("0") else Decimal("0")
        avg_gain = (avg_gain * Decimal(period - 1) + gain) / Decimal(period)
        avg_loss = (avg_loss * Decimal(period - 1) + loss) / Decimal(period)

    if avg_loss == Decimal("0"):
        return Decimal("100")
    rs = avg_gain / avg_loss
    return Decimal("100") - (Decimal("100") / (Decimal("1") + rs))


def _compute_atr(bars: list[Bar], period: int) -> Decimal:
    """Compute simple ATR (average of high - low) over the last `period` bars.

    Args:
        bars: Price bars (oldest first).
        period: Lookback period.

    Returns:
        Average ATR, or Decimal("0") if insufficient bars.
    """
    if len(bars) < period:
        return Decimal("0")
    ranges = [b.high - b.low for b in bars[-period:]]
    return sum(ranges) / Decimal(period)


class IntradayMomentumStrategy(Strategy):
    """15-minute RSI + momentum day trader.

    Signals long when RSI is oversold and price has positive momentum with
    above-average volume.  Signals short on the inverse.  Flattens equity
    positions (symbols without '/') before market close.

    Args:
        symbol: Tradeable instrument.
        params: Strategy parameter set.
        strategy_id: Optional explicit ID; auto-generated if omitted.
    """

    def __init__(
        self,
        symbol: str,
        params: IntradayMomentumParams | None = None,
        strategy_id: str | None = None,
    ) -> None:
        self._symbol = symbol
        self._params = params or IntradayMomentumParams()
        self._strategy_id = strategy_id or f"intraday_momentum_{uuid.uuid4().hex[:8]}"

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
        return True

    @property
    def params(self) -> IntradayMomentumParams:
        return self._params

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, bars: list[Bar]) -> Signal | None:
        """Generate a momentum signal from the bar series.

        Requires at least ``lookback_bars + 1`` bars.  Returns None if
        insufficient data is available.

        Args:
            bars: Historical price bars (oldest first).

        Returns:
            Signal with direction, strength, stop and TP distances, or None.
        """
        p = self._params
        min_bars = p.lookback_bars + 1
        if len(bars) < min_bars:
            return None

        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        rsi = _compute_rsi(closes, p.lookback_bars)
        atr = _compute_atr(bars, p.lookback_bars)

        if atr == Decimal("0"):
            return None

        # Rate of change: (close[-1] - close[-lookback_bars]) / close[-lookback_bars]
        roc = (closes[-1] - closes[-p.lookback_bars]) / closes[-p.lookback_bars]

        # Volume confirmation
        avg_vol = sum(volumes[-p.lookback_bars:]) / Decimal(p.lookback_bars)
        volume_ratio = (
            volumes[-1] / avg_vol
            if avg_vol > Decimal("0")
            else Decimal("0")
        )

        stop_distance = atr * p.atr_multiplier_stop / closes[-1]
        tp_distance = atr * p.atr_multiplier_tp / closes[-1]

        now = datetime.now(tz=timezone.utc).isoformat()

        if (
            rsi < p.rsi_oversold
            and roc > p.momentum_threshold
            and volume_ratio > p.min_volume_ratio
        ):
            strength = min(
                Decimal("1"),
                (p.rsi_oversold - rsi) / p.rsi_oversold * Decimal("2"),
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

        if (
            rsi > p.rsi_overbought
            and roc < -p.momentum_threshold
            and volume_ratio > p.min_volume_ratio
        ):
            strength = min(
                Decimal("1"),
                (rsi - p.rsi_overbought) / (Decimal("100") - p.rsi_overbought) * Decimal("2"),
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
        """Evaluate exit conditions for an open position.

        Checks (in priority order):
        1. EOD flatten for equity symbols when ``is_near_close`` is True.
        2. Hard stop breach.
        3. Take-profit breach.
        4. Trailing stop advance (updates in-place; does not exit).

        Args:
            position: The open position.
            current_bar: Most recent completed bar.
            is_near_close: True when within 15 minutes of 16:00 ET.

        Returns:
            ExitDecision.
        """
        p = self._params

        # 1. EOD flatten — equity symbols (no '/' in symbol) must exit before close
        is_equity = "/" not in position.symbol
        if is_near_close and is_equity and self.is_day_trader:
            return ExitDecision(should_exit=True, reason="eod_flatten")

        if position.direction == "long":
            # Hard stop
            if current_bar.low <= position.stop_price:
                return ExitDecision(should_exit=True, reason="hard_stop")
            # Take profit
            if current_bar.high >= position.take_profit_price:
                return ExitDecision(should_exit=True, reason="take_profit")
            # Trailing stop advance — update stop_price in-place if price moved up
            trail_abs = current_bar.close * p.atr_multiplier_trail
            # trail_abs here is an absolute ATR-scaled distance; we use trailing_stop_distance
            new_stop = current_bar.close - position.trailing_stop_distance * current_bar.close
            if new_stop > position.stop_price:
                # Mutate in-place — caller is responsible for persisting
                position.stop_price = new_stop
        else:
            # Hard stop (short: price rises through stop)
            if current_bar.high >= position.stop_price:
                return ExitDecision(should_exit=True, reason="hard_stop")
            # Take profit (short: price falls through TP)
            if current_bar.low <= position.take_profit_price:
                return ExitDecision(should_exit=True, reason="take_profit")
            # Trailing stop advance — tighten stop downward as price falls
            new_stop = current_bar.close + position.trailing_stop_distance * current_bar.close
            if new_stop < position.stop_price:
                position.stop_price = new_stop

        return ExitDecision(should_exit=False, reason="no_exit")
