"""
Portfolio manager — sole enforcer of risk limits.

Agents MUST NOT self-enforce drawdown or daily loss limits; they will game it.
All position approvals, drawdown checks, and kill switches run here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Optional

from evo_trader.agents.base import AgentState, TradingAgent
from evo_trader.config.parameters import CapitalConfig, RiskConfig

logger = logging.getLogger(__name__)


@dataclass
class PositionRequest:
    agent_id: str
    asset: str
    notional: Decimal       # Proposed position size in base currency
    direction: str          # "long" | "short"


@dataclass
class PositionDecision:
    approved: bool
    reason: str
    approved_notional: Decimal = Decimal("0")


@dataclass
class _DailyStats:
    date: date
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")


class PortfolioManager:
    """
    Owns all capital buckets and enforces risk parameters.

    Buckets:
      - active_trading: split across agent pool (subject to evolution)
      - dca: separate scheduler, not touched by evolution rules
      - cash_reserve: isolated; only releasable for clone seeding or drawdown cover
    """

    def __init__(
        self,
        capital_cfg: CapitalConfig | None = None,
        risk_cfg: RiskConfig | None = None,
    ) -> None:
        self._cap = capital_cfg or CapitalConfig()
        self._risk = risk_cfg or RiskConfig()

        # Capital buckets (immutable after init except cash reserve for seeding)
        self._active_pool: Decimal = self._cap.active_trading_total
        self._dca_pool: Decimal = self._cap.dca_total
        self._cash_reserve: Decimal = self._cap.cash_reserve_total

        # Agent registry: agent_id → agent
        self._agents: dict[str, TradingAgent] = {}

        # Per-agent allocated capital tracking
        self._agent_allocations: dict[str, Decimal] = {}

        # Daily P&L tracking for daily loss limit
        self._daily_stats: dict[str, _DailyStats] = {}   # keyed by date.isoformat()

        # Portfolio high-water mark for drawdown calculation
        self._high_water_mark: Decimal = self._cap.total_capital
        self._current_nav: Decimal = self._cap.total_capital

        # Per-agent, per-asset position tracker: (agent_id, asset) → notional
        self._positions: dict[tuple[str, str], Decimal] = {}

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent: TradingAgent, initial_allocation: Decimal) -> None:
        """Register a new agent and assign it capital from the active pool."""
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")

        # Cap is against the total configured active pool, not the remaining balance,
        # so the limit doesn't drift downward as agents are registered sequentially.
        max_alloc = self._cap.active_trading_total * self._risk.per_agent_allocation_cap
        if initial_allocation > max_alloc:
            raise ValueError(
                f"Requested {initial_allocation} exceeds per-agent cap {max_alloc}"
            )
        if initial_allocation > self._active_pool:
            raise ValueError("Insufficient active pool capital")

        self._agents[agent.agent_id] = agent
        self._agent_allocations[agent.agent_id] = initial_allocation
        agent.allocated_capital = initial_allocation
        self._active_pool -= initial_allocation   # deduct from unallocated pool
        logger.info("Registered agent %s with $%s", agent.agent_id, initial_allocation)

    # ------------------------------------------------------------------
    # Position approval — called before any agent opens a position
    # ------------------------------------------------------------------

    def approve_position(
        self, request: PositionRequest, today: Optional[date] = None
    ) -> PositionDecision:
        """
        Gate for every position an agent wants to open.
        Returns PositionDecision with approved=False if any limit would be breached.
        """
        today = today or date.today()

        agent = self._agents.get(request.agent_id)
        if not agent or not agent.is_active():
            return PositionDecision(
                approved=False,
                reason=f"Agent {request.agent_id} not active",
            )

        # 1. Portfolio-level drawdown kill switch
        if self._current_drawdown() >= self._risk.max_drawdown:
            return PositionDecision(
                approved=False,
                reason=(
                    f"Portfolio drawdown {self._current_drawdown():.1%} >= "
                    f"max {self._risk.max_drawdown:.1%}; all new positions halted"
                ),
            )

        # 2. Daily loss limit check (portfolio-wide)
        daily = self._get_daily_stats(today)
        daily_loss_pct = abs(daily.realized_pnl + daily.unrealized_pnl) / self._cap.total_capital
        if (daily.realized_pnl + daily.unrealized_pnl) < 0 and daily_loss_pct >= self._risk.daily_loss_limit:
            return PositionDecision(
                approved=False,
                reason=(
                    f"Daily loss {daily_loss_pct:.1%} >= limit {self._risk.daily_loss_limit:.1%}"
                ),
            )

        # 3. Per-asset concentration check for this agent
        agent_alloc = self._agent_allocations[request.agent_id]
        current_asset_notional = self._positions.get((request.agent_id, request.asset), Decimal("0"))
        new_total = current_asset_notional + request.notional
        concentration = new_total / agent_alloc if agent_alloc > 0 else Decimal("1")
        if concentration > self._risk.per_asset_concentration:
            max_allowed = agent_alloc * self._risk.per_asset_concentration - current_asset_notional
            if max_allowed <= 0:
                return PositionDecision(
                    approved=False,
                    reason=(
                        f"Asset {request.asset} concentration {concentration:.1%} "
                        f"would exceed {self._risk.per_asset_concentration:.1%} limit"
                    ),
                )
            # Approve partial
            return PositionDecision(
                approved=True,
                reason=f"Partial approval to stay within asset concentration limit",
                approved_notional=max_allowed,
            )

        return PositionDecision(
            approved=True,
            reason="All checks passed",
            approved_notional=request.notional,
        )

    # ------------------------------------------------------------------
    # P&L updates — called after each fill or mark-to-market
    # ------------------------------------------------------------------

    def record_pnl(
        self,
        agent_id: str,
        realized: Decimal,
        unrealized: Decimal,
        today: Optional[date] = None,
    ) -> None:
        today = today or date.today()
        daily = self._get_daily_stats(today)
        daily.realized_pnl += realized
        daily.unrealized_pnl += unrealized

        # Update NAV
        self._current_nav += realized
        if self._current_nav > self._high_water_mark:
            self._high_water_mark = self._current_nav

    def update_position(
        self, agent_id: str, asset: str, notional: Decimal
    ) -> None:
        """Record or update an open position after a fill."""
        self._positions[(agent_id, asset)] = notional

    # ------------------------------------------------------------------
    # Rebalance — portfolio manager reallocates capital among active agents
    # ------------------------------------------------------------------

    def rebalance(self, active_agents: list[TradingAgent]) -> dict[str, Decimal]:
        """
        Equal-weight rebalance across currently active agents.
        Redistributes the total currently deployed capital (sum of agent allocations
        plus unallocated remainder) evenly. Returns new allocation map.
        """
        if not active_agents:
            return {}

        # Total tradeable capital = deployed allocations + unallocated pool remainder
        deployed = sum(self._agent_allocations.get(a.agent_id, Decimal("0")) for a in active_agents)
        total_available = deployed + self._active_pool
        per_agent = total_available / len(active_agents)
        max_alloc = self._cap.active_trading_total * self._risk.per_agent_allocation_cap
        per_agent = min(per_agent, max_alloc)

        new_allocations: dict[str, Decimal] = {}
        for agent in active_agents:
            self._agent_allocations[agent.agent_id] = per_agent
            agent.allocated_capital = per_agent
            new_allocations[agent.agent_id] = per_agent

        # Reset unallocated pool to remainder after full deployment
        self._active_pool = total_available - per_agent * len(active_agents)

        logger.info(
            "Rebalanced %d agents at $%s each", len(active_agents), per_agent
        )
        return new_allocations

    # ------------------------------------------------------------------
    # Cash reserve — only for clone seeding or drawdown cover
    # ------------------------------------------------------------------

    def release_reserve_for_clone(self, amount: Decimal) -> Decimal:
        """
        Pull seeding capital from cash reserve for a new clone agent.
        Returns the amount actually released (may be less if reserve is low).
        """
        available = min(amount, self._cash_reserve)
        self._cash_reserve -= available
        self._active_pool += available
        logger.info("Released $%s from cash reserve for clone seeding", available)
        return available

    # ------------------------------------------------------------------
    # Elimination — remove agent, return its capital to active pool
    # ------------------------------------------------------------------

    def eliminate_agent(self, agent_id: str) -> None:
        agent = self._agents.get(agent_id)
        if not agent:
            return
        agent.state = AgentState.ELIMINATED
        returned = self._agent_allocations.pop(agent_id, Decimal("0"))
        self._active_pool += returned
        # Close all positions for this agent
        to_close = [k for k in self._positions if k[0] == agent_id]
        for k in to_close:
            del self._positions[k]
        logger.info(
            "Eliminated agent %s; returned $%s to active pool", agent_id, returned
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_drawdown(self) -> Decimal:
        if self._high_water_mark == 0:
            return Decimal("0")
        return (self._high_water_mark - self._current_nav) / self._high_water_mark

    def _get_daily_stats(self, today: date) -> _DailyStats:
        key = today.isoformat()
        if key not in self._daily_stats:
            self._daily_stats[key] = _DailyStats(date=today)
        return self._daily_stats[key]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def drawdown(self) -> Decimal:
        return self._current_drawdown()

    @property
    def nav(self) -> Decimal:
        return self._current_nav

    @property
    def active_pool(self) -> Decimal:
        return self._active_pool

    @property
    def cash_reserve(self) -> Decimal:
        return self._cash_reserve

    @property
    def dca_pool(self) -> Decimal:
        return self._dca_pool

    def agent_allocation(self, agent_id: str) -> Decimal:
        return self._agent_allocations.get(agent_id, Decimal("0"))

    def active_agents(self) -> list[TradingAgent]:
        return [a for a in self._agents.values() if a.is_active()]
