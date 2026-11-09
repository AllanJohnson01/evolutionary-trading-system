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
    def submit_order(self, symbol, direction, notional, *, strategy_id=None):
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


# ---------------------------------------------------------------------------
# Stable identity tests
# ---------------------------------------------------------------------------


def test_init_population_with_stable_ids_reuses_them():
    pm = PortfolioManager()
    cfg = CapitalConfig()
    stable = {
        "Momentum-Hunter": "aaa-111",
        "Breakout": "bbb-222",
        "Grid-Trader": "ccc-333",
        "Mean-Reversion": "ddd-444",
    }
    agents = init_population(pm, cfg, stable_ids=stable)
    ids = {a.agent_id for a in agents}
    assert ids == set(stable.values()), "stable IDs not reused across restart"


def test_init_population_without_stable_ids_mints_new_uuids():
    pm = PortfolioManager()
    cfg = CapitalConfig()
    agents1 = init_population(pm, cfg, stable_ids=None)
    pm2 = PortfolioManager()
    agents2 = init_population(pm2, cfg, stable_ids=None)
    ids1 = {a.agent_id for a in agents1}
    ids2 = {a.agent_id for a in agents2}
    assert ids1.isdisjoint(ids2), "fresh runs without stable_ids must mint distinct UUIDs"


def test_strategy_ids_are_stable_across_restarts(tmp_path):
    """Restarting with the same stable_ids produces the same strategy_ids in decisions.jsonl."""
    stable = {
        "Momentum-Hunter": "aaa-111",
        "Breakout": "bbb-222",
        "Grid-Trader": "ccc-333",
        "Mean-Reversion": "ddd-444",
    }

    def _run(log_path):
        pm = PortfolioManager()
        agents = init_population(pm, CapitalConfig(), stable_ids=stable)
        runner, _ = _runner(tmp_path)  # builds a fresh runner with _FlatBars / _NoOrders
        # Re-wire to use the stable-id agents
        from evo_trader.macro.store import PostureStore
        r = runner.__class__(
            portfolio_manager=pm,
            posture_store=PostureStore(),
            bars_client=_FlatBars(),
            order_client=_NoOrders(),
            log_path=log_path,
        )
        register_starting_strategies(r, agents, "SPY")
        r.run_once()
        return log_path

    import json
    log1 = _run(tmp_path / "run1.jsonl")
    log2 = _run(tmp_path / "run2.jsonl")

    sids1 = {json.loads(l)["strategy_id"] for l in log1.read_text().splitlines() if l.strip()}
    sids2 = {json.loads(l)["strategy_id"] for l in log2.read_text().splitlines() if l.strip()}
    assert sids1 == sids2, "strategy_ids differ across restarts with same stable_ids"


def test_decisions_log_includes_agent_id(tmp_path):
    """Every decision log entry must carry the stable agent_id field."""
    import json
    runner, agents = _runner(tmp_path)
    register_starting_strategies(runner, agents, "SPY")
    runner.run_once()

    for line in (tmp_path / "decisions.jsonl").read_text().strip().splitlines():
        entry = json.loads(line)
        assert "agent_id" in entry, "agent_id missing from decision log entry"
        assert entry["agent_id"], "agent_id is empty in decision log entry"
