"""Tests for the shared starting-population wiring and launchd plist templates."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from evo_trader.config.parameters import CapitalConfig
from evo_trader.portfolio.manager import PortfolioManager
from evo_trader.strategies.base import Bar
from evo_trader.tactical.population import (
    STARTING_AGENTS,
    init_population,
    register_starting_strategies,
)
from evo_trader.tactical.runner import StrategyRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FlatBars:
    """200 identical bars: no strategy has a signal, but each still logs a decision."""

    def get_bars(self, symbol, timeframe_minutes, limit):
        px = Decimal("100")
        return [
            Bar(f"2026-10-15T14:{i % 60:02d}:00+00:00", px, px, px, px, Decimal("1000"))
            for i in range(200)
        ]


class _NoOrders:
    def submit_order(self, symbol, direction, notional):
        raise AssertionError("no orders expected")

    def get_open_positions(self, strategy_id):
        return []

    def close_position(self, strategy_id, symbol, reason):
        raise AssertionError("no closes expected")


def _runner(tmp_path):
    from evo_trader.macro.store import PostureStore

    pm = PortfolioManager()
    agents = init_population(pm, CapitalConfig())
    runner = StrategyRunner(
        portfolio_manager=pm,
        posture_store=PostureStore(),
        bars_client=_FlatBars(),
        order_client=_NoOrders(),
        log_path=tmp_path / "decisions.jsonl",
    )
    return runner, agents


def test_init_population_registers_four_equal_agents():
    pm = PortfolioManager()
    cfg = CapitalConfig()
    agents = init_population(pm, cfg)

    assert len(agents) == len(STARTING_AGENTS) == 4
    assert len({a.agent_id for a in agents}) == 4
    assert all(a.allocated_capital == cfg.initial_agent_allocation for a in agents)
    assert cfg.initial_agent_allocation == Decimal("15000")


def test_register_starting_strategies_one_per_agent(tmp_path):
    runner, agents = _runner(tmp_path)
    register_starting_strategies(runner, agents, "SPY")

    regs = runner._registrations
    assert len(regs) == 4
    assert {r.agent_id for r in regs.values()} == {a.agent_id for a in agents}
    assert {r.strategy.symbol for r in regs.values()} == {"SPY"}
    assert len({type(r.strategy) for r in regs.values()}) == 4


def test_registered_runner_logs_one_decision_per_strategy(tmp_path):
    runner, agents = _runner(tmp_path)
    register_starting_strategies(runner, agents, "SPY")
    runner.run_once()

    lines = (tmp_path / "decisions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4


def test_launchd_templates_do_not_use_unexpanded_home():
    # launchd does not expand ${HOME} in StandardOutPath / StandardErrorPath.
    for plist in (REPO_ROOT / "ops" / "launchd").glob("*.plist"):
        assert "${HOME}" not in plist.read_text(), f"{plist.name} uses ${{HOME}}"
