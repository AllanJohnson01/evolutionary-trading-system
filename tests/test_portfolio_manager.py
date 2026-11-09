"""Tests for portfolio manager risk enforcement."""
from datetime import date
from decimal import Decimal

import pytest

from evo_trader.agents.base import AgentType, TradingAgent
from evo_trader.config.parameters import CapitalConfig, RiskConfig
from evo_trader.portfolio.manager import PortfolioManager, PositionRequest


def make_agent(agent_type: AgentType = AgentType.GRID) -> TradingAgent:
    return TradingAgent(agent_type=agent_type)


def make_pm() -> PortfolioManager:
    return PortfolioManager()


class TestAgentRegistration:
    def test_register_agent(self):
        pm = make_pm()
        agent = make_agent()
        # $60K active pool * 25% cap = $15K max per agent (matches equal split across 4)
        pm.register_agent(agent, Decimal("15000"))
        assert pm.agent_allocation(agent.agent_id) == Decimal("15000")

    def test_register_exceeds_cap_raises(self):
        pm = make_pm()
        agent = make_agent()
        # $60K * 25% = $15K cap; $16K exceeds it
        with pytest.raises(ValueError, match="per-agent cap"):
            pm.register_agent(agent, Decimal("16000"))

    def test_register_duplicate_raises(self):
        pm = make_pm()
        agent = make_agent()
        pm.register_agent(agent, Decimal("10000"))
        with pytest.raises(ValueError, match="already registered"):
            pm.register_agent(agent, Decimal("10000"))


class TestPositionApproval:
    def setup_method(self):
        self.pm = make_pm()
        self.agent = make_agent()
        self.pm.register_agent(self.agent, Decimal("15000"))
        self.today = date(2026, 1, 1)

    def test_normal_position_approved(self):
        req = PositionRequest(
            agent_id=self.agent.agent_id,
            asset="BTC/USDT",
            notional=Decimal("1000"),
            direction="long",
        )
        decision = self.pm.approve_position(req, self.today)
        assert decision.approved
        assert decision.approved_notional == Decimal("1000")

    def test_inactive_agent_rejected(self):
        from evo_trader.agents.base import AgentState
        self.agent.state = AgentState.ELIMINATED
        req = PositionRequest(
            agent_id=self.agent.agent_id,
            asset="BTC/USDT",
            notional=Decimal("1000"),
            direction="long",
        )
        decision = self.pm.approve_position(req, self.today)
        assert not decision.approved

    def test_drawdown_kill_switch(self):
        # Simulate max drawdown breach
        self.pm._high_water_mark = Decimal("100000")
        self.pm._current_nav = Decimal("84999")  # 15.001% drawdown
        req = PositionRequest(
            agent_id=self.agent.agent_id,
            asset="BTC/USDT",
            notional=Decimal("1000"),
            direction="long",
        )
        decision = self.pm.approve_position(req, self.today)
        assert not decision.approved
        assert "drawdown" in decision.reason

    def test_daily_loss_limit(self):
        # Simulate 2% daily loss on $100K = $2000
        daily = self.pm._get_daily_stats(self.today)
        daily.realized_pnl = Decimal("-2000")
        req = PositionRequest(
            agent_id=self.agent.agent_id,
            asset="BTC/USDT",
            notional=Decimal("1000"),
            direction="long",
        )
        decision = self.pm.approve_position(req, self.today)
        assert not decision.approved
        assert "Daily loss" in decision.reason

    def test_asset_concentration_partial_approval(self):
        # Agent has $15K, 15% limit = $2250. Request $3000 → partial approval at $2250
        req = PositionRequest(
            agent_id=self.agent.agent_id,
            asset="ETH/USDT",
            notional=Decimal("3000"),
            direction="long",
        )
        decision = self.pm.approve_position(req, self.today)
        assert decision.approved
        assert decision.approved_notional == Decimal("2250")

    def test_asset_concentration_fully_blocked(self):
        # Fill up the asset slot first ($15K * 15% = $2250)
        self.pm.update_position(self.agent.agent_id, "ETH/USDT", Decimal("2250"))
        req = PositionRequest(
            agent_id=self.agent.agent_id,
            asset="ETH/USDT",
            notional=Decimal("100"),
            direction="long",
        )
        decision = self.pm.approve_position(req, self.today)
        assert not decision.approved


class TestElimination:
    def test_eliminate_returns_capital(self):
        pm = make_pm()
        agent = make_agent()
        pool_before_register = pm.active_pool
        pm.register_agent(agent, Decimal("10000"))
        assert pm.active_pool == pool_before_register - Decimal("10000")
        pm.eliminate_agent(agent.agent_id)
        # Capital returned to pool after elimination
        assert pm.active_pool == pool_before_register
        from evo_trader.agents.base import AgentState
        assert agent.state == AgentState.ELIMINATED


class TestRebalance:
    def test_rebalance_equal_weights(self):
        pm = make_pm()
        agents = [make_agent() for _ in range(4)]
        for a in agents:
            # $15K each is exactly the 25% cap ($60K * 25% = $15K)
            pm.register_agent(a, Decimal("15000"))

        # 4 agents at $15K = $60K deployed; pool = $0 → rebalance redistributes $60K
        new_allocs = pm.rebalance(agents)
        assert len(new_allocs) == 4
        # $60K / 4 = $15K, 25% cap on $60K = $15K → equal split at cap
        for alloc in new_allocs.values():
            assert alloc == Decimal("15000")
