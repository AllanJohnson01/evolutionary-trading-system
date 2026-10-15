"""
Tests for all four strategy implementations.

No live network calls.  All bar data is constructed inline.
Values are Decimal throughout.
"""
from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import List

import pytest

from evo_trader.strategies.base import Bar, ExitDecision, Position, Signal
from evo_trader.strategies.breakout import BreakoutParams, BreakoutStrategy
from evo_trader.strategies.grid import GridParams, GridStrategy
from evo_trader.strategies.intraday_momentum import (
    IntradayMomentumParams,
    IntradayMomentumStrategy,
)
from evo_trader.strategies.mean_reversion import (
    MeanReversionParams,
    MeanReversionStrategy,
)


# ---------------------------------------------------------------------------
# Shared bar factory
# ---------------------------------------------------------------------------


def make_bars(
    n: int,
    base_price: Decimal,
    trend: Decimal = Decimal("0"),
    vol: Decimal = Decimal("0.002"),
) -> List[Bar]:
    """Generate ``n`` Bar objects with sequential ISO timestamps.

    Args:
        n: Number of bars to produce.
        base_price: Reference price for the first bar.
        trend: Total fractional drift from first to last bar.
                  close[i] = base_price * (1 + trend * i / n)
        vol: Fractional half-spread; high = close*(1+vol), low = close*(1-vol).

    Returns:
        List of Bar objects in chronological order (oldest first).
    """
    bars: List[Bar] = []
    prev_close = base_price
    for i in range(n):
        close = base_price * (Decimal("1") + trend * Decimal(i) / Decimal(n))
        high = close * (Decimal("1") + vol)
        low = close * (Decimal("1") - vol)
        open_ = prev_close
        volume = Decimal("10000") + Decimal("100") * Decimal(i)
        bars.append(
            Bar(
                timestamp=f"2024-01-01T{(i // 60):02d}:{(i % 60):02d}:00Z",
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        prev_close = close
    return bars


def _make_position(
    strategy_id: str,
    symbol: str,
    direction: str = "long",
    entry_price: Decimal = Decimal("100"),
    stop_price: Decimal = Decimal("95"),
    take_profit_price: Decimal = Decimal("110"),
    trailing_stop_distance: Decimal = Decimal("0.02"),
) -> Position:
    return Position(
        strategy_id=strategy_id,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        size=Decimal("10"),
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        trailing_stop_distance=trailing_stop_distance,
        entry_time="2024-01-01T00:00:00Z",
    )


def _make_bar(
    close: Decimal,
    high: Decimal | None = None,
    low: Decimal | None = None,
    open_: Decimal | None = None,
    volume: Decimal = Decimal("10000"),
) -> Bar:
    if high is None:
        high = close * Decimal("1.002")
    if low is None:
        low = close * Decimal("0.998")
    if open_ is None:
        open_ = close
    return Bar(
        timestamp="2024-01-01T10:00:00Z",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


# ---------------------------------------------------------------------------
# TestIntradayMomentumStrategy
# ---------------------------------------------------------------------------


class TestIntradayMomentumStrategy:
    def setup_method(self) -> None:
        self.params = IntradayMomentumParams(lookback_bars=10)
        self.strategy = IntradayMomentumStrategy(
            symbol="AAPL",
            params=self.params,
            strategy_id="test_momentum",
        )

    # --- generate_signal: insufficient bars ---

    def test_returns_none_with_insufficient_bars(self) -> None:
        # Needs lookback_bars + 1 = 11 bars minimum
        bars = make_bars(10, Decimal("100"))
        signal = self.strategy.generate_signal(bars)
        assert signal is None

    # --- generate_signal: neutral ---

    def test_no_signal_on_neutral_price(self) -> None:
        # Flat prices — RSI will be ~50, momentum near zero
        bars = make_bars(20, Decimal("100"), trend=Decimal("0"))
        signal = self.strategy.generate_signal(bars)
        # Either None or direction is not a strong signal
        assert signal is None or signal.direction not in ("long", "short")

    # --- generate_signal: long on oversold RSI ---

    def test_long_signal_on_oversold_rsi_with_momentum(self) -> None:
        """50 declining bars push RSI well below 55; the last bar spikes above bars[-10]."""
        # Lots of declining bars get RSI low (avg_loss >> avg_gain)
        bars: list[Bar] = []
        price = Decimal("100")
        for i in range(50):
            price = price * Decimal("0.99")
            bars.append(Bar(
                timestamp=f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00Z",
                open=price, high=price * Decimal("1.001"),
                low=price * Decimal("0.999"), close=price, volume=Decimal("10000"),
            ))
        # 9 more declines (bars[-10] to bars[-2])
        for i in range(9):
            price = price * Decimal("0.995")
            bars.append(Bar(
                timestamp=f"2024-01-01T{(50 + i) // 60:02d}:{(50 + i) % 60:02d}:00Z",
                open=price, high=price * Decimal("1.001"),
                low=price * Decimal("0.999"), close=price, volume=Decimal("10000"),
            ))
        # Spike close above bars[-10].close with high volume
        bar_10_close = bars[-1].close
        spike_close = bar_10_close * Decimal("1.08")
        bars.append(Bar(
            timestamp="2024-01-01T01:00:00Z",
            open=bars[-1].close, high=spike_close * Decimal("1.001"),
            low=bars[-1].close, close=spike_close, volume=Decimal("50000"),
        ))
        params = IntradayMomentumParams(
            lookback_bars=10,
            rsi_oversold=Decimal("55"),
            momentum_threshold=Decimal("0.001"),
            min_volume_ratio=Decimal("1.5"),
        )
        strategy = IntradayMomentumStrategy(symbol="AAPL", params=params, strategy_id="test_long")
        signal = strategy.generate_signal(bars)
        assert signal is not None
        assert signal.direction == "long"

    # --- generate_signal: short on overbought RSI ---

    def test_short_signal_on_overbought_rsi(self) -> None:
        """50 rising bars push RSI above 60; 9 flat bars; last bar drops 2% with volume."""
        bars: list[Bar] = []
        price = Decimal("100")
        for i in range(50):
            price = price * Decimal("1.01")
            bars.append(Bar(
                timestamp=f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00Z",
                open=price, high=price * Decimal("1.001"),
                low=price * Decimal("0.999"), close=price, volume=Decimal("10000"),
            ))
        # 9 flat bars so bars[-10] stays at current price level
        for i in range(9):
            bars.append(Bar(
                timestamp=f"2024-01-01T{(50 + i) // 60:02d}:{(50 + i) % 60:02d}:00Z",
                open=price, high=price * Decimal("1.001"),
                low=price * Decimal("0.999"), close=price, volume=Decimal("10000"),
            ))
        bar_10_close = bars[-1].close
        # 2% collapse: small enough to keep RSI above 60, but still negative ROC
        collapse_close = bar_10_close * Decimal("0.98")
        bars.append(Bar(
            timestamp="2024-01-01T01:00:00Z",
            open=price, high=price,
            low=collapse_close * Decimal("0.999"), close=collapse_close, volume=Decimal("50000"),
        ))
        params = IntradayMomentumParams(
            lookback_bars=10,
            rsi_overbought=Decimal("60"),
            momentum_threshold=Decimal("0.001"),
            min_volume_ratio=Decimal("1.5"),
        )
        strategy = IntradayMomentumStrategy(symbol="AAPL", params=params, strategy_id="test_short")
        signal = strategy.generate_signal(bars)
        assert signal is not None
        assert signal.direction == "short"

    # --- check_exits: EOD flatten ---

    def test_eod_flatten_for_equity_position(self) -> None:
        position = _make_position("test_momentum", "AAPL")
        bar = _make_bar(Decimal("105"))
        decision = self.strategy.check_exits(position, bar, is_near_close=True)
        assert decision.should_exit is True
        assert decision.reason == "eod_flatten"

    def test_no_eod_flatten_for_crypto(self) -> None:
        position = _make_position("test_momentum", "BTC/USDT")
        bar = _make_bar(Decimal("105"))
        decision = self.strategy.check_exits(position, bar, is_near_close=True)
        # BTC/USDT contains '/' — not equity; no EOD flatten
        assert decision.should_exit is False

    # --- check_exits: hard stop ---

    def test_hard_stop_triggers_for_long(self) -> None:
        position = _make_position(
            "test_momentum", "AAPL", direction="long", stop_price=Decimal("100")
        )
        bar = _make_bar(Decimal("101"), low=Decimal("99"))
        decision = self.strategy.check_exits(position, bar, is_near_close=False)
        assert decision.should_exit is True
        assert decision.reason == "hard_stop"

    def test_take_profit_triggers_for_long(self) -> None:
        position = _make_position(
            "test_momentum",
            "AAPL",
            direction="long",
            stop_price=Decimal("95"),
            take_profit_price=Decimal("110"),
        )
        bar = _make_bar(Decimal("109"), high=Decimal("111"))
        decision = self.strategy.check_exits(position, bar, is_near_close=False)
        assert decision.should_exit is True
        assert decision.reason == "take_profit"

    def test_hard_stop_triggers_for_short(self) -> None:
        position = _make_position(
            "test_momentum",
            "AAPL",
            direction="short",
            entry_price=Decimal("105"),
            stop_price=Decimal("110"),
            take_profit_price=Decimal("95"),
        )
        bar = _make_bar(Decimal("109"), high=Decimal("111"))
        decision = self.strategy.check_exits(position, bar, is_near_close=False)
        assert decision.should_exit is True
        assert decision.reason == "hard_stop"

    def test_take_profit_triggers_for_short(self) -> None:
        position = _make_position(
            "test_momentum",
            "AAPL",
            direction="short",
            entry_price=Decimal("105"),
            stop_price=Decimal("110"),
            take_profit_price=Decimal("95"),
        )
        bar = _make_bar(Decimal("96"), low=Decimal("94"))
        decision = self.strategy.check_exits(position, bar, is_near_close=False)
        assert decision.should_exit is True
        assert decision.reason == "take_profit"

    # --- check_exits: trailing stop advance ---

    def test_trailing_stop_advances_for_long(self) -> None:
        initial_stop = Decimal("90")
        position = _make_position(
            "test_momentum",
            "AAPL",
            direction="long",
            entry_price=Decimal("100"),
            stop_price=initial_stop,
            take_profit_price=Decimal("130"),
            trailing_stop_distance=Decimal("0.05"),
        )
        # Price rises well above entry so new_stop = 120 * (1 - 0.05) = 114 > 90
        bar = _make_bar(Decimal("120"), high=Decimal("121"), low=Decimal("119"))
        decision = self.strategy.check_exits(position, bar, is_near_close=False)
        assert decision.should_exit is False
        assert position.stop_price > initial_stop

    # --- mutation_bounds completeness ---

    def test_mutation_bounds_all_numeric_fields_covered(self) -> None:
        bounds = IntradayMomentumParams.mutation_bounds()
        params = IntradayMomentumParams()
        numeric_fields = {
            f.name
            for f in dataclasses.fields(params)
            if isinstance(getattr(params, f.name), Decimal)
        }
        assert set(bounds.keys()) == numeric_fields

    # --- strategy properties ---

    def test_is_day_trader(self) -> None:
        assert self.strategy.is_day_trader is True

    def test_cadence_minutes(self) -> None:
        assert self.strategy.cadence_minutes == 15

    # --- signal stop distance is non-zero ---

    def test_signal_stop_distance_is_nonzero(self) -> None:
        # Same reliable bar construction as test_long_signal_on_oversold_rsi_with_momentum
        bars: list[Bar] = []
        price = Decimal("100")
        for i in range(50):
            price = price * Decimal("0.99")
            bars.append(Bar(
                timestamp=f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00Z",
                open=price, high=price * Decimal("1.001"),
                low=price * Decimal("0.999"), close=price, volume=Decimal("10000"),
            ))
        for i in range(9):
            price = price * Decimal("0.995")
            bars.append(Bar(
                timestamp=f"2024-01-01T{(50 + i) // 60:02d}:{(50 + i) % 60:02d}:00Z",
                open=price, high=price * Decimal("1.001"),
                low=price * Decimal("0.999"), close=price, volume=Decimal("10000"),
            ))
        spike_close = bars[-1].close * Decimal("1.08")
        bars.append(Bar(
            timestamp="2024-01-01T01:00:00Z",
            open=bars[-1].close, high=spike_close * Decimal("1.001"),
            low=bars[-1].close, close=spike_close, volume=Decimal("50000"),
        ))
        params = IntradayMomentumParams(
            lookback_bars=10,
            rsi_oversold=Decimal("55"),
            momentum_threshold=Decimal("0.001"),
            min_volume_ratio=Decimal("1.5"),
        )
        strategy = IntradayMomentumStrategy(symbol="AAPL", params=params, strategy_id="test_stop_dist")
        signal = strategy.generate_signal(bars)
        assert signal is not None
        assert signal.stop_distance > Decimal("0")


# ---------------------------------------------------------------------------
# TestBreakoutStrategy
# ---------------------------------------------------------------------------


class TestBreakoutStrategy:
    def setup_method(self) -> None:
        self.params = BreakoutParams(
            range_bars=4,
            atr_lookback=4,
            breakout_multiplier=Decimal("1.005"),
            min_range_atr_ratio=Decimal("0.1"),
        )
        self.strategy = BreakoutStrategy(
            symbol="BTC/USDT",
            params=self.params,
            strategy_id="test_breakout",
        )

    def _sufficient_bars(
        self,
        range_high: Decimal,
        range_low: Decimal,
        last_close: Decimal,
        n_history: int = 10,
    ) -> List[Bar]:
        """Build a bar list where the range window has explicit high/low and last close
        is as specified."""
        # Historical context bars (baseline)
        context = make_bars(n_history, Decimal("100"), vol=Decimal("0.01"))
        # Construct the final range_bars (4) with controlled highs and lows
        spread = (range_high - range_low) / Decimal("2")
        center = (range_high + range_low) / Decimal("2")
        range_window: List[Bar] = []
        for _ in range(self.params.range_bars - 1):
            range_window.append(
                Bar(
                    timestamp="2024-01-02T00:00:00Z",
                    open=center,
                    high=range_high,
                    low=range_low,
                    close=center,
                    volume=Decimal("10000"),
                )
            )
        # Last bar with the triggering close
        range_window.append(
            Bar(
                timestamp="2024-01-02T01:00:00Z",
                open=center,
                high=max(last_close * Decimal("1.001"), range_high),
                low=range_low,
                close=last_close,
                volume=Decimal("10000"),
            )
        )
        return context + range_window

    # --- insufficient bars ---

    def test_returns_none_with_insufficient_bars(self) -> None:
        bars = make_bars(4, Decimal("100"))
        signal = self.strategy.generate_signal(bars)
        assert signal is None

    # --- long breakout ---

    def test_long_breakout_signal(self) -> None:
        range_high = Decimal("105")
        range_low = Decimal("100")
        # last_close must exceed range_high * 1.005
        last_close = range_high * Decimal("1.010")
        bars = self._sufficient_bars(range_high, range_low, last_close)
        signal = self.strategy.generate_signal(bars)
        assert signal is not None
        assert signal.direction == "long"

    # --- short breakout ---

    def test_short_breakout_signal(self) -> None:
        # Use context bars well above the range so they don't lower range_low.
        # With the fixed strategy (range = bars[-(range_bars+1):-1]), bars[-5:-1]
        # includes one context bar. Keep context high (price=110) so its low
        # (110*0.99=108.9) doesn't contaminate range_low (=100).
        range_high = Decimal("105")
        range_low = Decimal("100")
        context = make_bars(10, Decimal("110"), vol=Decimal("0.01"))
        center = (range_high + range_low) / Decimal("2")
        range_window = []
        for _ in range(self.params.range_bars - 1):
            range_window.append(
                Bar(
                    timestamp="2024-01-02T00:00:00Z",
                    open=center,
                    high=range_high,
                    low=range_low,
                    close=center,
                    volume=Decimal("10000"),
                )
            )
        # last_close must be below effective range_low / breakout_multiplier.
        # After the context bar's low (~108.9) and range bars' low (100),
        # range_low = 100; need last_close < 100 / 1.005 ≈ 99.5.
        last_close = Decimal("98")
        range_window.append(
            Bar(
                timestamp="2024-01-02T01:00:00Z",
                open=center,
                high=range_high,
                low=last_close * Decimal("0.999"),
                close=last_close,
                volume=Decimal("10000"),
            )
        )
        bars = context + range_window
        signal = self.strategy.generate_signal(bars)
        assert signal is not None
        assert signal.direction == "short"

    # --- no signal within range ---

    def test_no_signal_within_range(self) -> None:
        range_high = Decimal("105")
        range_low = Decimal("100")
        # last_close inside the range — no breakout
        last_close = Decimal("102")
        bars = self._sufficient_bars(range_high, range_low, last_close)
        signal = self.strategy.generate_signal(bars)
        assert signal is None

    # --- exit checks ---

    def test_hard_stop_long(self) -> None:
        position = _make_position(
            "test_breakout", "BTC/USDT", direction="long", stop_price=Decimal("100")
        )
        bar = _make_bar(Decimal("101"), low=Decimal("99"))
        decision = self.strategy.check_exits(position, bar, is_near_close=False)
        assert decision.should_exit is True
        assert decision.reason == "hard_stop"

    def test_take_profit_long(self) -> None:
        position = _make_position(
            "test_breakout",
            "BTC/USDT",
            direction="long",
            stop_price=Decimal("95"),
            take_profit_price=Decimal("110"),
        )
        bar = _make_bar(Decimal("109"), high=Decimal("111"))
        decision = self.strategy.check_exits(position, bar, is_near_close=False)
        assert decision.should_exit is True
        assert decision.reason == "take_profit"

    # --- no EOD flatten for non-day-trader ---

    def test_no_eod_flatten(self) -> None:
        assert self.strategy.is_day_trader is False
        position = _make_position("test_breakout", "BTC/USDT")
        bar = _make_bar(Decimal("105"))
        decision = self.strategy.check_exits(position, bar, is_near_close=True)
        # is_near_close is True but strategy is not a day trader — no EOD flatten
        assert decision.should_exit is False

    # --- mutation bounds ---

    def test_mutation_bounds_complete(self) -> None:
        bounds = BreakoutParams.mutation_bounds()
        params = BreakoutParams()
        numeric_fields = {
            f.name
            for f in dataclasses.fields(params)
            if isinstance(getattr(params, f.name), Decimal)
        }
        assert set(bounds.keys()) == numeric_fields


# ---------------------------------------------------------------------------
# TestGridStrategy
# ---------------------------------------------------------------------------


class TestGridStrategy:
    def setup_method(self) -> None:
        self.params = GridParams(
            grid_levels=3,
            mean_lookback=10,
            atr_lookback=10,
            grid_spacing_pct=Decimal("0.005"),
            atr_scale_factor=Decimal("0.5"),
        )
        self.strategy = GridStrategy(
            symbol="ETH/USDT",
            params=self.params,
            strategy_id="test_grid",
        )

    # --- insufficient bars ---

    def test_returns_none_with_insufficient_bars(self) -> None:
        # min_bars = max(mean_lookback=10, atr_lookback=10) = 10
        bars = make_bars(9, Decimal("1000"))
        signal = self.strategy.generate_signal(bars)
        assert signal is None

    # --- long signal at buy level ---

    def test_long_signal_at_buy_level(self) -> None:
        """Price drops below the nearest buy level → long signal."""
        # Build 10 bars near 1000 to establish a mean of ~1000.
        # Then the last bar's close should be below mean*(1 - 1*effective_spacing).
        # With grid_spacing_pct=0.005 and small ATR, buy_level[0] ≈ 1000*0.995 = 995.
        # We set last close to 994 (below the buy level).
        base = Decimal("1000")
        bars = make_bars(10, base, trend=Decimal("0"), vol=Decimal("0.001"))
        # Overwrite last bar close to be below the expected buy level
        below_buy = base * Decimal("0.990")
        last = bars[-1]
        bars[-1] = Bar(
            timestamp=last.timestamp,
            open=last.open,
            high=last.high,
            low=below_buy * Decimal("0.999"),
            close=below_buy,
            volume=last.volume,
        )
        signal = self.strategy.generate_signal(bars)
        assert signal is not None
        assert signal.direction == "long"

    # --- short signal at sell level ---

    def test_short_signal_at_sell_level(self) -> None:
        """Price rises above the nearest sell level → short signal."""
        base = Decimal("1000")
        bars = make_bars(10, base, trend=Decimal("0"), vol=Decimal("0.001"))
        above_sell = base * Decimal("1.010")
        last = bars[-1]
        bars[-1] = Bar(
            timestamp=last.timestamp,
            open=last.open,
            high=above_sell * Decimal("1.001"),
            low=last.low,
            close=above_sell,
            volume=last.volume,
        )
        signal = self.strategy.generate_signal(bars)
        assert signal is not None
        assert signal.direction == "short"

    # --- neutral price produces no signal ---

    def test_neutral_price_no_signal(self) -> None:
        """Price at the computed mean → no signal."""
        bars = make_bars(10, Decimal("1000"), trend=Decimal("0"), vol=Decimal("0.001"))
        # The last bar close is exactly the mean (~1000) — between buy and sell levels
        signal = self.strategy.generate_signal(bars)
        assert signal is None

    # --- mutation bounds ---

    def test_mutation_bounds_complete(self) -> None:
        bounds = GridParams.mutation_bounds()
        params = GridParams()
        numeric_fields = {
            f.name
            for f in dataclasses.fields(params)
            if isinstance(getattr(params, f.name), Decimal)
        }
        assert set(bounds.keys()) == numeric_fields

    # --- cadence ---

    def test_cadence_minutes(self) -> None:
        assert self.strategy.cadence_minutes == 15


# ---------------------------------------------------------------------------
# TestMeanReversionStrategy
# ---------------------------------------------------------------------------


class TestMeanReversionStrategy:
    def setup_method(self) -> None:
        self.params = MeanReversionParams(
            lookback_bars=20,
            entry_z=Decimal("2.0"),
            exit_z=Decimal("0.5"),
            atr_lookback=14,
        )
        self.strategy = MeanReversionStrategy(
            symbol="BTC/USDT",
            params=self.params,
            strategy_id="test_mean_rev",
        )

    # --- insufficient bars ---

    def test_returns_none_with_insufficient_bars(self) -> None:
        # min_bars = max(lookback_bars=20, atr_lookback=14) = 20
        bars = make_bars(19, Decimal("100"))
        signal = self.strategy.generate_signal(bars)
        assert signal is None

    # --- long on negative z-score ---

    def test_long_entry_on_negative_z(self) -> None:
        """Price drops 3σ below rolling mean → long signal."""
        # Build 20 bars near a mean of 100 with small std (~0.1).
        # Then force the last bar well below -2σ.
        base = Decimal("100")
        bars = make_bars(20, base, trend=Decimal("0"), vol=Decimal("0.001"))
        # Estimate std of the flat series: very small.
        # Drop last bar by 5% to ensure |z| >= 2.0 even against a tight std.
        drop_close = base * Decimal("0.94")
        last = bars[-1]
        bars[-1] = Bar(
            timestamp=last.timestamp,
            open=last.open,
            high=last.high,
            low=drop_close * Decimal("0.999"),
            close=drop_close,
            volume=last.volume,
        )
        signal = self.strategy.generate_signal(bars)
        assert signal is not None
        assert signal.direction == "long"

    # --- short on positive z-score ---

    def test_short_entry_on_positive_z(self) -> None:
        """Price rises 3σ above rolling mean → short signal."""
        base = Decimal("100")
        bars = make_bars(20, base, trend=Decimal("0"), vol=Decimal("0.001"))
        spike_close = base * Decimal("1.06")
        last = bars[-1]
        bars[-1] = Bar(
            timestamp=last.timestamp,
            open=last.open,
            high=spike_close * Decimal("1.001"),
            low=last.low,
            close=spike_close,
            volume=last.volume,
        )
        signal = self.strategy.generate_signal(bars)
        assert signal is not None
        assert signal.direction == "short"

    # --- no signal within z band ---

    def test_no_signal_within_z_band(self) -> None:
        """Price at rolling mean → z ≈ 0 → no signal."""
        bars = make_bars(20, Decimal("100"), trend=Decimal("0"), vol=Decimal("0.001"))
        signal = self.strategy.generate_signal(bars)
        assert signal is None

    # --- no EOD flatten ---

    def test_no_eod_flatten(self) -> None:
        assert self.strategy.is_day_trader is False
        position = _make_position("test_mean_rev", "BTC/USDT")
        bar = _make_bar(Decimal("105"))
        decision = self.strategy.check_exits(position, bar, is_near_close=True)
        assert decision.should_exit is False

    # --- cadence ---

    def test_cadence_minutes(self) -> None:
        assert self.strategy.cadence_minutes == 30

    # --- mutation bounds ---

    def test_mutation_bounds_complete(self) -> None:
        bounds = MeanReversionParams.mutation_bounds()
        params = MeanReversionParams()
        numeric_fields = {
            f.name
            for f in dataclasses.fields(params)
            if isinstance(getattr(params, f.name), Decimal)
        }
        assert set(bounds.keys()) == numeric_fields

    # --- take profit on z reversion ---

    def test_tp_triggers_on_z_reversion(self) -> None:
        """Long position; bar.high >= take_profit_price → should_exit=True, reason=take_profit."""
        # entry_price=95, take_profit_price=100 (price reverted to mean)
        position = _make_position(
            "test_mean_rev",
            "BTC/USDT",
            direction="long",
            entry_price=Decimal("95"),
            stop_price=Decimal("90"),
            take_profit_price=Decimal("100"),
        )
        bar = _make_bar(Decimal("100"), high=Decimal("101"), low=Decimal("99"))
        decision = self.strategy.check_exits(position, bar, is_near_close=False)
        assert decision.should_exit is True
        assert decision.reason == "take_profit"
