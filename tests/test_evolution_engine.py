"""Tests for evolution engine — elimination, cloning, mutation, and log."""
import tempfile
from decimal import Decimal
from pathlib import Path

import pytest

from evo_trader.agents.base import AgentState, AgentType, PerformanceRecord, TradingAgent
from evo_trader.config.parameters import EvolutionConfig
from evo_trader.evolution.engine import EvolutionEngine
from evo_trader.evolution.transaction_log import EventType, EvolutionLog
from evo_trader.portfolio.manager import PortfolioManager


def make_agents(n: int) -> list[TradingAgent]:
    types = [AgentType.GRID, AgentType.MOMENTUM, AgentType.MEAN_REVERSION, AgentType.RL]
    return [TradingAgent(agent_type=types[i % len(types)]) for i in range(n)]


def make_engine_with_agents(
    n: int = 4, log_path: str = "/dev/null"
) -> tuple[EvolutionEngine, PortfolioManager, list[TradingAgent]]:
    pm = PortfolioManager()
    agents = make_agents(n)
    for agent in agents:
        pm.register_agent(agent, Decimal("12000"))

    cfg = EvolutionConfig()
    engine = EvolutionEngine(pm, cfg, log_path=log_path)
    return engine, pm, agents


def make_perf_records(
    agents: list[TradingAgent], pnl_pcts: list[float]
) -> dict[str, PerformanceRecord]:
    result = {}
    for agent, pnl_pct in zip(agents, pnl_pcts):
        result[agent.agent_id] = PerformanceRecord(
            period_start="2026-01-01",
            period_end="2026-01-28",
            pnl=Decimal(str(pnl_pct * 1000)),
            pnl_pct=Decimal(str(pnl_pct)),
        )
    return result


class TestEvolutionPeriod:
    def test_no_elimination_after_one_bad_period(self):
        engine, pm, agents = make_engine_with_agents()
        records = make_perf_records(agents, [0.10, 0.08, 0.05, -0.02])
        decisions = engine.run_period(records)
        # Bottom agent after only 1 period should NOT be eliminated (needs 2)
        for decision in decisions.values():
            assert decision != "eliminated"

    def test_elimination_after_two_consecutive_bottom_periods(self):
        engine, pm, agents = make_engine_with_agents()

        # Period 1: last agent is worst
        worst = agents[-1]
        records1 = make_perf_records(agents, [0.10, 0.08, 0.05, -0.02])
        engine.run_period(records1)

        # Period 2: same agent still worst
        records2 = make_perf_records(agents, [0.09, 0.07, 0.04, -0.03])
        decisions = engine.run_period(records2)

        assert decisions.get(worst.agent_id) == "eliminated"
        assert worst.state == AgentState.ELIMINATED

    def test_no_clone_after_one_top_period(self):
        engine, pm, agents = make_engine_with_agents()
        records = make_perf_records(agents, [0.20, 0.08, 0.05, 0.02])
        decisions = engine.run_period(records)
        assert decisions.get(agents[0].agent_id) != "cloned"

    def test_clone_after_two_consecutive_top_periods(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name
        engine, pm, agents = make_engine_with_agents(log_path=log_path)

        best = agents[0]
        records = make_perf_records(agents, [0.20, 0.08, 0.05, 0.02])
        engine.run_period(records)
        engine.run_period(records)

        # After 2 top periods, a clone should have been created
        log = EvolutionLog(log_path)
        clone_events = log.events_by_type(EventType.CLONE)
        assert len(clone_events) > 0
        assert clone_events[0].parent_agent_id == best.agent_id

    def test_transaction_log_records_eliminations(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name
        engine, pm, agents = make_engine_with_agents(log_path=log_path)

        worst = agents[-1]
        records = make_perf_records(agents, [0.10, 0.08, 0.05, -0.02])
        engine.run_period(records)
        engine.run_period(records)

        log = EvolutionLog(log_path)
        elim_events = log.events_by_type(EventType.ELIMINATION)
        assert any(e.agent_id == worst.agent_id for e in elim_events)

    def test_max_agent_cap_respected(self):
        engine, pm, agents = make_engine_with_agents()
        best = agents[0]
        records = make_perf_records(agents, [0.20, 0.08, 0.05, 0.02])

        # Run many periods — population should never exceed 8
        for _ in range(10):
            engine.run_period(records)

        from evo_trader.config.parameters import EvolutionConfig
        assert len(pm.active_agents()) <= EvolutionConfig().max_agent_count
