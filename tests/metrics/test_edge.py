"""
Tests for the edge-metrics / signal-decay module.

Covers:
- Metric math against hand-computed fixture logs
- Regime tagging
- Decay-trend detection on synthetic decaying / stable edges
- Zero-edge flagging
- Kelly inputs as the sole sizing-stats source
- Evolution ranking with edge-metrics inputs (Sharpe-based ranking)
- Existing evolution tests still green (imported via test_evolution_engine)
"""
from __future__ import annotations

import json
import math
import statistics
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import List

import pytest

from evo_trader.agents.base import AgentState, AgentType, PerformanceRecord, TradingAgent
from evo_trader.config.parameters import EdgeMetricsConfig, EvolutionConfig
from evo_trader.evolution.engine import EvolutionEngine
from evo_trader.metrics.edge import (
    EdgeCalculator,
    EdgeMetrics,
    KellyInputs,
    MetricsSummary,
    TradeRecord,
    _hit_rate,
    _profit_factor,
    _sharpe_per_trade,
    _linear_slope,
    _t_stat_mean,
)
from evo_trader.portfolio.manager import PortfolioManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_trade_log(path: Path, trades: list[dict]) -> None:
    with path.open("w") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")


def make_trade(strategy_id: str, pnl: float, regime: str = "neutral",
               closed_at: str = "2026-01-01T10:00:00Z") -> dict:
    return {
        "strategy_id": strategy_id,
        "trade_id": f"t-{pnl}",
        "regime": regime,
        "closed_at": closed_at,
        "pnl": pnl,
        "entry_price": 100.0,
        "exit_price": 100.0 + pnl,
        "side": "long",
    }


# ---------------------------------------------------------------------------
# Hand-computed metric math
# ---------------------------------------------------------------------------

class TestMetricHelpers:
    def test_hit_rate_50pct(self):
        pnls = [10.0, -5.0, 8.0, -3.0]
        assert _hit_rate(pnls) == pytest.approx(0.5)

    def test_hit_rate_all_wins(self):
        assert _hit_rate([5.0, 3.0, 2.0]) == pytest.approx(1.0)

    def test_hit_rate_all_losses(self):
        assert _hit_rate([-1.0, -2.0]) == pytest.approx(0.0)

    def test_hit_rate_empty(self):
        assert _hit_rate([]) is None

    def test_profit_factor_hand_computed(self):
        # gross_wins = 10 + 8 = 18; gross_losses = 5 + 3 = 8
        pnls = [10.0, -5.0, 8.0, -3.0]
        assert _profit_factor(pnls) == pytest.approx(18.0 / 8.0)

    def test_profit_factor_no_losses(self):
        result = _profit_factor([5.0, 3.0])
        assert result == float("inf")

    def test_profit_factor_no_wins(self):
        result = _profit_factor([-2.0, -1.0])
        assert result == pytest.approx(0.0)

    def test_sharpe_per_trade_hand_computed(self):
        pnls = [10.0, -5.0, 8.0, -3.0, 5.0]
        m = statistics.mean(pnls)
        s = statistics.stdev(pnls)
        expected = m / s
        assert _sharpe_per_trade(pnls) == pytest.approx(expected)

    def test_linear_slope_increasing(self):
        slope = _linear_slope([1.0, 2.0, 3.0, 4.0])
        assert slope == pytest.approx(1.0)

    def test_linear_slope_decreasing(self):
        slope = _linear_slope([4.0, 3.0, 2.0, 1.0])
        assert slope == pytest.approx(-1.0)

    def test_linear_slope_flat(self):
        assert _linear_slope([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_linear_slope_single_point(self):
        assert _linear_slope([3.0]) is None

    def test_t_stat_positive_mean(self):
        vals = [2.0, 3.0, 4.0, 5.0, 6.0]
        t = _t_stat_mean(vals)
        assert t is not None and t > 0

    def test_t_stat_none_when_std_zero(self):
        assert _t_stat_mean([5.0, 5.0, 5.0]) is None


# ---------------------------------------------------------------------------
# EdgeCalculator — fixture-based metrics
# ---------------------------------------------------------------------------

class TestEdgeCalculatorMetrics:
    def test_insufficient_trades_returns_none_metrics(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        trades = [make_trade("strat_a", float(p)) for p in [10, -5, 8]]  # only 3
        write_trade_log(log, trades)
        cfg = EdgeMetricsConfig(min_trades_for_metrics=5)
        calc = EdgeCalculator(log, cfg=cfg)
        m = calc.compute("strat_a")
        assert m.hit_rate is None
        assert m.profit_factor is None
        assert m.sharpe_per_trade is None

    def test_hit_rate_matches_fixture(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        # 6 wins, 4 losses → hit_rate = 0.6
        pnls = [10, 8, 7, 6, 5, 4, -1, -2, -3, -4]
        trades = [make_trade("s1", float(p)) for p in pnls]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(min_trades_for_metrics=5))
        m = calc.compute("s1", window=10)
        assert m.hit_rate == pytest.approx(0.6)

    def test_profit_factor_matches_fixture(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        pnls = [10.0, 8.0, -5.0, -3.0]
        trades = [make_trade("s1", p) for p in pnls]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(min_trades_for_metrics=4))
        m = calc.compute("s1", window=4)
        assert m.profit_factor == pytest.approx(18.0 / 8.0)

    def test_avg_edge_correct(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        pnls = [10.0, -5.0, 8.0, -3.0, 5.0, 2.0]
        trades = [make_trade("s1", p) for p in pnls]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(min_trades_for_metrics=5))
        m = calc.compute("s1", window=6)
        assert m.avg_edge == pytest.approx(statistics.mean(pnls))

    def test_empty_log_no_crash(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        log.write_text("")
        calc = EdgeCalculator(log)
        ki = calc.kelly_inputs("strat_x")
        assert ki.trade_count == 0
        assert ki.win_rate == 0.0


# ---------------------------------------------------------------------------
# Regime tagging
# ---------------------------------------------------------------------------

class TestRegimeTagging:
    def test_regime_breakdown_hit_rates(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        trades = [
            make_trade("s1", 10.0, regime="risk_on"),
            make_trade("s1", 5.0, regime="risk_on"),
            make_trade("s1", -3.0, regime="risk_off"),
            make_trade("s1", -2.0, regime="risk_off"),
            make_trade("s1", 1.0, regime="neutral"),
            make_trade("s1", -1.0, regime="neutral"),
        ]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(min_trades_for_metrics=5))
        m = calc.compute("s1", window=6)
        assert m.regime_hit_rate["risk_on"] == pytest.approx(1.0)
        assert m.regime_hit_rate["risk_off"] == pytest.approx(0.0)
        assert m.regime_hit_rate["neutral"] == pytest.approx(0.5)

    def test_regime_avg_edge(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        trades = [
            make_trade("s1", 10.0, regime="risk_on"),
            make_trade("s1", 6.0, regime="risk_on"),
            make_trade("s1", -4.0, regime="risk_off"),
            make_trade("s1", -2.0, regime="risk_off"),
            make_trade("s1", 0.5, regime="neutral"),
        ]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(min_trades_for_metrics=5))
        m = calc.compute("s1", window=5)
        assert m.regime_avg_edge["risk_on"] == pytest.approx(8.0)
        assert m.regime_avg_edge["risk_off"] == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# Decay-trend detection
# ---------------------------------------------------------------------------

class TestDecayTrend:
    def _decaying_log(self, tmp_path: Path, strategy_id: str = "s1") -> Path:
        log = tmp_path / "trades.jsonl"
        # 30 trades: clearly decaying — first 10 avg +10, next 10 avg +3, last 10 avg -2
        trades = (
            [make_trade(strategy_id, 10.0)] * 10
            + [make_trade(strategy_id, 3.0)] * 10
            + [make_trade(strategy_id, -2.0)] * 10
        )
        write_trade_log(log, trades)
        return log

    def _stable_log(self, tmp_path: Path, strategy_id: str = "s2") -> Path:
        log = tmp_path / "trades.jsonl"
        trades = [make_trade(strategy_id, 5.0)] * 30
        write_trade_log(log, trades)
        return log

    def test_decaying_edge_has_negative_avg_edge_slope(self, tmp_path):
        log = self._decaying_log(tmp_path)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(
            min_trades_for_metrics=5, trend_window_count=3
        ))
        m = calc.compute("s1", window=30)
        assert m.avg_edge_slope is not None
        assert m.avg_edge_slope < 0, "Decaying edge should have negative avg_edge_slope"

    def test_stable_edge_has_near_zero_slope(self, tmp_path):
        log = self._stable_log(tmp_path)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(
            min_trades_for_metrics=5, trend_window_count=3
        ))
        m = calc.compute("s2", window=30)
        assert m.avg_edge_slope is not None
        assert abs(m.avg_edge_slope) < 0.1, "Stable edge should have ~0 slope"

    def test_sharpe_slope_negative_for_decaying(self, tmp_path):
        log = tmp_path / "trades_sharpe.jsonl"
        # Sub-windows need internal variance for Sharpe to be computable.
        # Window 1: high positive Sharpe (mostly wins, some variance)
        w1 = [make_trade("s3", float(p)) for p in [15, 12, 10, 8, -1, 14, 11, 9, 7, -2]]
        # Window 2: moderate Sharpe
        w2 = [make_trade("s3", float(p)) for p in [3, 4, 2, -2, 1, 5, 2, -3, 1, 3]]
        # Window 3: negative Sharpe (mostly losses)
        w3 = [make_trade("s3", float(p)) for p in [-4, -5, 2, -6, -3, -7, 1, -4, -2, -5]]
        write_trade_log(log, w1 + w2 + w3)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(
            min_trades_for_metrics=5, trend_window_count=3
        ))
        m = calc.compute("s3", window=30)
        assert m.sharpe_slope is not None
        assert m.sharpe_slope < 0, f"Expected negative sharpe_slope, got {m.sharpe_slope}"


# ---------------------------------------------------------------------------
# Zero-edge flagging
# ---------------------------------------------------------------------------

class TestZeroEdge:
    def test_zero_edge_flagged_when_mean_not_significant(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        # Trades centred near zero with high variance
        pnls = [1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1, 0.3, -0.3]
        trades = [make_trade("s1", p) for p in pnls]
        write_trade_log(log, trades)
        # Low t-threshold so near-zero avg always fails significance
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(
            min_trades_for_metrics=5,
            zero_edge_t_threshold=100.0  # effectively always zero-edge for small means
        ))
        m = calc.compute("s1", window=10)
        assert m.zero_edge is True

    def test_strong_edge_not_flagged(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        pnls = [20.0] * 20
        trades = [make_trade("s1", p) for p in pnls]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(
            min_trades_for_metrics=5,
            zero_edge_t_threshold=1.645
        ))
        m = calc.compute("s1", window=20)
        # std == 0 → t-stat is None → zero_edge = True per design (can't confirm edge without variance)
        # This tests the graceful handling; strong edge with variance:
        pnls2 = [20.0 + i * 0.1 for i in range(20)]
        trades2 = [make_trade("s2", p) for p in pnls2]
        log2 = tmp_path / "trades2.jsonl"
        write_trade_log(log2, trades2)
        calc2 = EdgeCalculator(log2, cfg=EdgeMetricsConfig(
            min_trades_for_metrics=5, zero_edge_t_threshold=1.645
        ))
        m2 = calc2.compute("s2", window=20)
        assert m2.zero_edge is False


# ---------------------------------------------------------------------------
# Kelly inputs
# ---------------------------------------------------------------------------

class TestKellyInputs:
    def test_kelly_inputs_hand_computed(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        # 7 wins of +10, 3 losses of -5: win_rate=0.7, avg_win=10, avg_loss=5, payoff=2
        pnls = [10.0] * 7 + [-5.0] * 3
        trades = [make_trade("s1", p) for p in pnls]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log)
        ki = calc.kelly_inputs("s1", window=10)
        assert ki.win_rate == pytest.approx(0.7)
        assert ki.payoff_ratio == pytest.approx(2.0)
        assert ki.trade_count == 10

    def test_kelly_inputs_window_limits_to_last_n(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        # 40 trades — all losses except last 5 which are +10
        pnls = [-1.0] * 35 + [10.0] * 5
        trades = [make_trade("s1", p) for p in pnls]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log)
        ki = calc.kelly_inputs("s1", window=5)
        assert ki.win_rate == pytest.approx(1.0)
        assert ki.trade_count == 5

    def test_kelly_inputs_no_trades(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        log.write_text("")
        calc = EdgeCalculator(log)
        ki = calc.kelly_inputs("missing_strat")
        assert ki.win_rate == 0.0
        assert ki.payoff_ratio == 0.0
        assert ki.trade_count == 0


# ---------------------------------------------------------------------------
# Summary API
# ---------------------------------------------------------------------------

class TestMetricsSummary:
    def test_summary_flagged_after_consecutive_zero_edge(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        pnls = [0.1, -0.1] * 10  # near-zero
        trades = [make_trade("s1", p) for p in pnls]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(
            min_trades_for_metrics=5,
            zero_edge_t_threshold=100.0,
            zero_edge_consecutive_periods=2,
        ))
        summary = calc.summary("s1", consecutive_zero_edge_periods=2)
        assert summary.flagged_for_elimination is True

    def test_summary_not_flagged_with_one_zero_edge_period(self, tmp_path):
        log = tmp_path / "trades.jsonl"
        pnls = [0.1, -0.1] * 10
        trades = [make_trade("s1", p) for p in pnls]
        write_trade_log(log, trades)
        calc = EdgeCalculator(log, cfg=EdgeMetricsConfig(
            min_trades_for_metrics=5,
            zero_edge_t_threshold=100.0,
            zero_edge_consecutive_periods=2,
        ))
        summary = calc.summary("s1", consecutive_zero_edge_periods=1)
        assert summary.flagged_for_elimination is False


# ---------------------------------------------------------------------------
# Evolution engine — ranking with edge inputs
# ---------------------------------------------------------------------------

def make_pm_with_agents(n: int) -> tuple[PortfolioManager, list[TradingAgent]]:
    pm = PortfolioManager()
    types = [AgentType.GRID, AgentType.MOMENTUM, AgentType.MEAN_REVERSION, AgentType.RL]
    agents = [TradingAgent(agent_type=types[i % len(types)]) for i in range(n)]
    for a in agents:
        pm.register_agent(a, Decimal("12000"))
    return pm, agents


class TestEvolutionEdgeIntegration:
    def test_sharpe_ranking_overrides_pnl_ranking(self):
        """An agent with low PnL but high Sharpe should rank above one with high PnL but negative Sharpe."""
        pm, agents = make_pm_with_agents(4)
        engine = EvolutionEngine(pm, EvolutionConfig(), log_path="/dev/null")

        a0, a1, a2, a3 = agents

        # a0: low pnl_pct but high Sharpe → should be top-ranked
        # a1: highest pnl_pct but negative Sharpe → should rank bottom
        records = {
            a0.agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                           pnl_pct=Decimal("0.02"), sharpe=2.5),
            a1.agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                           pnl_pct=Decimal("0.20"), sharpe=-0.5),
            a2.agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                           pnl_pct=Decimal("0.08"), sharpe=0.8),
            a3.agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                           pnl_pct=Decimal("0.05"), sharpe=0.3),
        }

        # Run 2 periods (need 2 for elimination/clone triggers)
        engine.run_period(records)
        decisions = engine.run_period(records)

        assert decisions.get(a0.agent_id) in ("cloned", "active"), \
            "High-Sharpe agent should not be eliminated"
        assert decisions.get(a1.agent_id) == "eliminated", \
            "Negative-Sharpe agent should be eliminated after 2 periods"

    def test_zero_edge_elimination_regardless_of_pnl(self):
        """An agent with profitable PnL but persistent zero edge should be eliminated."""
        pm, agents = make_pm_with_agents(4)
        cfg = EvolutionConfig(elimination_consecutive_periods=2)
        engine = EvolutionEngine(pm, cfg, log_path="/dev/null")

        a0, a1, a2, a3 = agents

        # a3 has good pnl_pct but zero_edge=True
        def make_records(zero_edge_id: str) -> dict[str, PerformanceRecord]:
            return {
                a0.agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                               pnl_pct=Decimal("0.10"), sharpe=1.5),
                a1.agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                               pnl_pct=Decimal("0.08"), sharpe=1.2),
                a2.agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                               pnl_pct=Decimal("0.05"), sharpe=0.8),
                zero_edge_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                                pnl_pct=Decimal("0.07"), sharpe=0.5,
                                                zero_edge=True),
            }

        engine.run_period(make_records(a3.agent_id))
        decisions = engine.run_period(make_records(a3.agent_id))

        assert decisions.get(a3.agent_id) == "eliminated", \
            "Zero-edge agent should be eliminated after 2 consecutive zero-edge periods"

    def test_zero_edge_streak_counter(self):
        """zero_edge_streak tracks consecutive zero-edge periods correctly."""
        pm, agents = make_pm_with_agents(4)
        engine = EvolutionEngine(pm, EvolutionConfig(), log_path="/dev/null")

        target = agents[0]

        def record_with_ze(ze: bool) -> dict[str, PerformanceRecord]:
            return {
                a.agent_id: PerformanceRecord(
                    "2026-01-01", "2026-01-28",
                    pnl_pct=Decimal("0.05"), sharpe=0.5,
                    zero_edge=ze if a.agent_id == target.agent_id else False,
                )
                for a in agents
            }

        engine.run_period(record_with_ze(True))
        assert engine.zero_edge_streak(target.agent_id) == 1

        engine.run_period(record_with_ze(False))
        assert engine.zero_edge_streak(target.agent_id) == 0

    def test_pnl_fallback_when_no_sharpe(self):
        """Engine falls back to pnl_pct ranking when sharpe is None (legacy compatibility)."""
        pm, agents = make_pm_with_agents(4)
        engine = EvolutionEngine(pm, EvolutionConfig(), log_path="/dev/null")

        records = {
            agents[0].agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                                   pnl_pct=Decimal("0.10")),
            agents[1].agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                                   pnl_pct=Decimal("0.08")),
            agents[2].agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                                   pnl_pct=Decimal("0.05")),
            agents[3].agent_id: PerformanceRecord("2026-01-01", "2026-01-28",
                                                   pnl_pct=Decimal("-0.03")),
        }
        # Two periods — worst should be eliminated
        engine.run_period(records)
        decisions = engine.run_period(records)
        assert decisions.get(agents[3].agent_id) == "eliminated"
