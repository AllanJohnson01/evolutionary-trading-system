"""
Tests for half-Kelly position sizing, EWMA volatility, and blackout mechanics.

Covers:
- kelly_fraction: positive edge, negative edge, zero payoff, breakeven
- position_fraction: cold-start path, normal path, concentration cap, risk_dial
- ewma_volatility: known sequence, empty / single-bar edge cases
- stop_distance: 3 x ewma_vol relationship
- PortfolioManager.approve_position: blackout rejection and pass-through
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from evo_trader.agents.base import AgentType, TradingAgent
from evo_trader.config.parameters import SizingConfig
from evo_trader.portfolio.manager import (
    BlackoutWindow,
    MarketPosture,
    PortfolioManager,
    PositionRequest,
)
from evo_trader.portfolio.sizing import kelly_fraction, position_fraction
from evo_trader.portfolio.stops import ewma_volatility, stop_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pm_with_agent() -> tuple[PortfolioManager, TradingAgent]:
    """Return a fresh PortfolioManager with one active agent allocated $12 000."""
    pm = PortfolioManager()
    agent = TradingAgent(agent_type=AgentType.GRID)
    pm.register_agent(agent, Decimal("12000"))
    return pm, agent


# ---------------------------------------------------------------------------
# Kelly fraction tests
# ---------------------------------------------------------------------------


class TestKellyFraction:
    def test_positive_edge(self):
        """Classic 60/40 win rate with 1:1 payoff → f* = 0.2."""
        # f = (1 * 0.6 - 0.4) / 1 = 0.2
        result = kelly_fraction(Decimal("0.6"), Decimal("1"))
        assert result == Decimal("0.2")

    def test_higher_payoff_ratio(self):
        """50% win rate, 2:1 payoff → f* = (2*0.5 - 0.5) / 2 = 0.25."""
        result = kelly_fraction(Decimal("0.5"), Decimal("2"))
        assert result == Decimal("0.25")

    def test_breakeven_edge_returns_zero(self):
        """50% win rate, 1:1 payoff → breakeven, f* = 0."""
        # f = (1*0.5 - 0.5)/1 = 0
        result = kelly_fraction(Decimal("0.5"), Decimal("1"))
        assert result == Decimal("0")

    def test_negative_edge_returns_zero(self):
        """40% win rate, 1:1 payoff → negative edge, f* floored at 0."""
        result = kelly_fraction(Decimal("0.4"), Decimal("1"))
        assert result == Decimal("0")

    def test_zero_payoff_ratio_returns_zero(self):
        """payoff_ratio = 0 is degenerate; must return 0 without dividing by zero."""
        result = kelly_fraction(Decimal("0.7"), Decimal("0"))
        assert result == Decimal("0")

    def test_zero_win_rate_returns_zero(self):
        """win_rate = 0 means the strategy never wins; must return 0."""
        result = kelly_fraction(Decimal("0"), Decimal("2"))
        assert result == Decimal("0")

    def test_returns_decimal_not_float(self):
        """Return type must be Decimal, never float."""
        result = kelly_fraction(Decimal("0.6"), Decimal("1"))
        assert isinstance(result, Decimal)

    def test_negative_payoff_ratio_returns_zero(self):
        """Negative payoff_ratio is nonsensical; guard returns 0."""
        result = kelly_fraction(Decimal("0.6"), Decimal("-1"))
        assert result == Decimal("0")


# ---------------------------------------------------------------------------
# position_fraction tests
# ---------------------------------------------------------------------------


class TestPositionFraction:
    def _cfg(self) -> SizingConfig:
        return SizingConfig()

    def test_cold_start_path(self):
        """Trade count below threshold returns cold_start_fixed_risk * risk_dial."""
        cfg = self._cfg()
        result = position_fraction(
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("2"),
            risk_dial=Decimal("1"),
            concentration_cap=Decimal("0.15"),
            trade_count=0,
            cfg=cfg,
        )
        assert result == cfg.cold_start_fixed_risk * Decimal("1")

    def test_cold_start_boundary(self):
        """trade_count = threshold - 1 still takes cold-start path."""
        cfg = self._cfg()
        result = position_fraction(
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("2"),
            risk_dial=Decimal("1"),
            concentration_cap=Decimal("0.15"),
            trade_count=cfg.cold_start_threshold - 1,
            cfg=cfg,
        )
        assert result == cfg.cold_start_fixed_risk

    def test_normal_path_activates_at_threshold(self):
        """trade_count = threshold takes the Kelly path (larger than cold-start for good edge)."""
        cfg = self._cfg()
        cold_start_val = cfg.cold_start_fixed_risk
        result = position_fraction(
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("2"),
            risk_dial=Decimal("1"),
            concentration_cap=Decimal("0.15"),
            trade_count=cfg.cold_start_threshold,
            cfg=cfg,
        )
        # half-Kelly of (0.6, 2.0): f = (2*0.6 - 0.4)/2 = 0.7; half = 0.35; capped at 0.15
        assert result == Decimal("0.15")

    def test_half_kelly_applied(self):
        """Normal path applies half-Kelly fraction (0.5) to raw kelly."""
        cfg = self._cfg()
        # For (0.6, 1): kelly = 0.2, half = 0.1; cap = 0.15 → 0.1 passes through
        result = position_fraction(
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("1"),
            risk_dial=Decimal("1"),
            concentration_cap=Decimal("0.15"),
            trade_count=cfg.cold_start_threshold,
            cfg=cfg,
        )
        assert result == Decimal("0.1")

    def test_concentration_cap_clamps(self):
        """Half-Kelly above concentration_cap is clamped to cap."""
        cfg = self._cfg()
        # (0.6, 2): raw kelly = 0.7, half = 0.35; cap = 0.15 → 0.15
        result = position_fraction(
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("2"),
            risk_dial=Decimal("1"),
            concentration_cap=Decimal("0.15"),
            trade_count=cfg.cold_start_threshold,
            cfg=cfg,
        )
        assert result == Decimal("0.15")

    def test_risk_dial_scales_normal_path(self):
        """risk_dial = 0.5 halves the normal-path output."""
        cfg = self._cfg()
        # half-Kelly of (0.6, 1) = 0.1, * 0.5 risk dial = 0.05
        result = position_fraction(
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("1"),
            risk_dial=Decimal("0.5"),
            concentration_cap=Decimal("0.15"),
            trade_count=cfg.cold_start_threshold,
            cfg=cfg,
        )
        assert result == Decimal("0.05")

    def test_risk_dial_scales_cold_start(self):
        """risk_dial = 0.5 halves cold-start output."""
        cfg = self._cfg()
        result = position_fraction(
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("2"),
            risk_dial=Decimal("0.5"),
            concentration_cap=Decimal("0.15"),
            trade_count=0,
            cfg=cfg,
        )
        assert result == cfg.cold_start_fixed_risk * Decimal("0.5")

    def test_negative_edge_returns_zero_normal_path(self):
        """Negative Kelly → zero fraction (no bet into losing edge)."""
        cfg = self._cfg()
        result = position_fraction(
            win_rate=Decimal("0.3"),
            payoff_ratio=Decimal("1"),
            risk_dial=Decimal("1"),
            concentration_cap=Decimal("0.15"),
            trade_count=cfg.cold_start_threshold,
            cfg=cfg,
        )
        assert result == Decimal("0")

    def test_default_cfg_used_when_none(self):
        """Passing cfg=None uses the default SizingConfig()."""
        result_none = position_fraction(
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("1"),
            risk_dial=Decimal("1"),
            concentration_cap=Decimal("0.15"),
            trade_count=SizingConfig().cold_start_threshold,
            cfg=None,
        )
        result_explicit = position_fraction(
            win_rate=Decimal("0.6"),
            payoff_ratio=Decimal("1"),
            risk_dial=Decimal("1"),
            concentration_cap=Decimal("0.15"),
            trade_count=SizingConfig().cold_start_threshold,
            cfg=SizingConfig(),
        )
        assert result_none == result_explicit


# ---------------------------------------------------------------------------
# EWMA volatility tests
# ---------------------------------------------------------------------------


class TestEwmaVolatility:
    def test_empty_returns_zero(self):
        assert ewma_volatility([], span=20) == Decimal("0")

    def test_single_return_zero(self):
        assert ewma_volatility([Decimal("0.01")], span=20) == Decimal("0")

    def test_known_uniform_sequence(self):
        """Uniform 1% returns — EWMA vol should converge toward 0.01."""
        returns = [Decimal("0.01")] * 50
        vol = ewma_volatility(returns, span=20)
        # Must be close to 0.01 (within 5% relative tolerance)
        assert abs(float(vol) - 0.01) < 0.0005

    def test_alternating_returns(self):
        """Alternating +1% / -1% — volatility should be non-zero."""
        returns = [Decimal("0.01"), Decimal("-0.01")] * 25
        vol = ewma_volatility(returns, span=20)
        assert vol > Decimal("0")

    def test_returns_decimal_not_float(self):
        returns = [Decimal("0.01"), Decimal("0.02")]
        vol = ewma_volatility(returns, span=20)
        assert isinstance(vol, Decimal)

    def test_two_equal_returns(self):
        """Two identical returns — variance seeds at r[0]^2, then updates."""
        r = Decimal("0.02")
        vol = ewma_volatility([r, r], span=20)
        # Both r^2 values equal; result should be > 0
        assert vol > Decimal("0")

    def test_zero_returns_produce_zero_vol(self):
        """All-zero returns → zero variance → zero volatility."""
        returns = [Decimal("0")] * 10
        vol = ewma_volatility(returns, span=20)
        assert vol == Decimal("0")

    def test_higher_span_slower_decay(self):
        """Higher span gives more weight to history; older spikes linger longer."""
        # Spike at index 0, then zeros — high span retains more of the initial spike
        returns = [Decimal("0.10")] + [Decimal("0")] * 30
        vol_low_span = ewma_volatility(returns, span=5)
        vol_high_span = ewma_volatility(returns, span=50)
        # With a spike at the start and all zeros after, higher span keeps the
        # initial variance longer (alpha is smaller → less weight on new zero values)
        assert vol_high_span > vol_low_span


# ---------------------------------------------------------------------------
# stop_distance tests
# ---------------------------------------------------------------------------


class TestStopDistance:
    def test_stop_is_three_times_vol(self):
        """stop_distance = stop_sigma_multiplier * ewma_volatility."""
        returns = [Decimal("0.01")] * 50
        cfg = SizingConfig()
        vol = ewma_volatility(returns, span=cfg.ewma_span)
        expected = cfg.stop_sigma_multiplier * vol
        result = stop_distance(returns, cfg=cfg)
        assert result == expected

    def test_stop_zero_on_empty_returns(self):
        assert stop_distance([], cfg=SizingConfig()) == Decimal("0")

    def test_stop_zero_on_single_return(self):
        assert stop_distance([Decimal("0.01")], cfg=SizingConfig()) == Decimal("0")

    def test_default_cfg_used_when_none(self):
        returns = [Decimal("0.01")] * 10
        result_none = stop_distance(returns, cfg=None)
        result_explicit = stop_distance(returns, cfg=SizingConfig())
        assert result_none == result_explicit

    def test_custom_multiplier(self):
        """Custom SizingConfig with a different multiplier is respected."""
        returns = [Decimal("0.02")] * 30
        cfg_2x = SizingConfig(stop_sigma_multiplier=Decimal("2"))
        cfg_3x = SizingConfig(stop_sigma_multiplier=Decimal("3"))
        stop_2 = stop_distance(returns, cfg=cfg_2x)
        stop_3 = stop_distance(returns, cfg=cfg_3x)
        # Ratio must be 3:2
        assert abs(float(stop_3) / float(stop_2) - 1.5) < 1e-9


# ---------------------------------------------------------------------------
# Blackout window tests in PortfolioManager
# ---------------------------------------------------------------------------


class TestBlackoutWindow:
    def setup_method(self) -> None:
        self.pm, self.agent = make_pm_with_agent()
        self.req = PositionRequest(
            agent_id=self.agent.agent_id,
            asset="BTC/USDT",
            notional=Decimal("500"),
            direction="long",
        )

    def _active_posture(self) -> MarketPosture:
        """Return a posture with a blackout window that spans the current moment."""
        now = datetime.now(timezone.utc)
        window = BlackoutWindow(
            start=now - timedelta(minutes=5),
            end=now + timedelta(minutes=5),
            reason="FOMC announcement",
        )
        return MarketPosture(risk_dial=Decimal("1"), blackout_windows=[window])

    def _inactive_posture(self) -> MarketPosture:
        """Return a posture with a blackout window entirely in the past."""
        past = datetime.now(timezone.utc) - timedelta(hours=2)
        window = BlackoutWindow(
            start=past - timedelta(minutes=5),
            end=past + timedelta(minutes=5),
            reason="Past event",
        )
        return MarketPosture(risk_dial=Decimal("1"), blackout_windows=[window])

    def test_active_blackout_rejects_position(self):
        """A currently active blackout window blocks the position."""
        decision = self.pm.approve_position(self.req, posture=self._active_posture())
        assert not decision.approved
        assert "blackout" in decision.reason.lower()

    def test_active_blackout_reason_contains_window_reason(self):
        """Rejection reason includes the blackout window's reason string."""
        decision = self.pm.approve_position(self.req, posture=self._active_posture())
        assert "FOMC announcement" in decision.reason

    def test_inactive_blackout_does_not_block(self):
        """A past blackout window does not block the position."""
        decision = self.pm.approve_position(self.req, posture=self._inactive_posture())
        assert decision.approved

    def test_no_posture_does_not_block(self):
        """Omitting posture (None) has no effect on normal approvals."""
        decision = self.pm.approve_position(self.req, posture=None)
        assert decision.approved

    def test_empty_blackout_list_does_not_block(self):
        """A MarketPosture with an empty blackout list passes through cleanly."""
        posture = MarketPosture(risk_dial=Decimal("1"), blackout_windows=[])
        decision = self.pm.approve_position(self.req, posture=posture)
        assert decision.approved

    def test_future_blackout_does_not_block(self):
        """A blackout window in the future does not affect current positions."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        window = BlackoutWindow(
            start=future,
            end=future + timedelta(minutes=30),
            reason="Earnings release",
        )
        posture = MarketPosture(risk_dial=Decimal("1"), blackout_windows=[window])
        decision = self.pm.approve_position(self.req, posture=posture)
        assert decision.approved

    def test_blackout_checked_before_drawdown(self):
        """Blackout rejection fires even when portfolio is also in drawdown."""
        # Trip drawdown kill switch
        self.pm._high_water_mark = Decimal("100000")
        self.pm._current_nav = Decimal("84000")  # ~16% drawdown
        decision = self.pm.approve_position(self.req, posture=self._active_posture())
        # Blackout message returned, not drawdown message (step 0 fires first)
        assert not decision.approved
        assert "blackout" in decision.reason.lower()
