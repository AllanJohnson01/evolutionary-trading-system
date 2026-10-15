"""
Starting population: the 4 diverse agents and their strategies.

Single source of truth shared by ``ops/scripts/burn_in.py`` and the tactical
runner's ``--once`` entry point, so both always run the same population.
"""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Callable

from evo_trader.agents.base import AgentState, AgentType, TradingAgent
from evo_trader.config.parameters import CapitalConfig
from evo_trader.portfolio.manager import PortfolioManager
from evo_trader.strategies.base import Strategy
from evo_trader.strategies.breakout import BreakoutStrategy
from evo_trader.strategies.grid import GridStrategy
from evo_trader.strategies.intraday_momentum import IntradayMomentumStrategy
from evo_trader.strategies.mean_reversion import MeanReversionStrategy

if TYPE_CHECKING:
    from evo_trader.tactical.runner import StrategyRunner

logger = logging.getLogger(__name__)

# (name, agent type, strategy factory taking a symbol)
STARTING_AGENTS: list[tuple[str, AgentType, Callable[[str], Strategy]]] = [
    ("Momentum-Hunter", AgentType.MOMENTUM, IntradayMomentumStrategy),
    ("Breakout", AgentType.MOMENTUM, BreakoutStrategy),  # breakout is in the momentum family
    ("Grid-Trader", AgentType.GRID, GridStrategy),
    ("Mean-Reversion", AgentType.MEAN_REVERSION, MeanReversionStrategy),
]


def init_population(
    portfolio_manager: PortfolioManager, capital_cfg: CapitalConfig
) -> list[TradingAgent]:
    """Register the starting agents with equal capital allocation."""
    per_agent = capital_cfg.initial_agent_allocation
    agents: list[TradingAgent] = []

    for name, agent_type, _ in STARTING_AGENTS:
        agent = TradingAgent(
            agent_id=str(uuid.uuid4()),
            agent_type=agent_type,
            state=AgentState.ACTIVE,
            allocated_capital=per_agent,
            generation=0,
        )
        portfolio_manager.register_agent(agent, per_agent)
        logger.info("Registered agent %s (%s) with $%s", name, agent_type.value, per_agent)
        agents.append(agent)

    return agents


def register_starting_strategies(
    runner: "StrategyRunner", agents: list[TradingAgent], symbol: str
) -> None:
    """Register one strategy per agent (same order as ``STARTING_AGENTS``)."""
    for agent, (_, _, factory) in zip(agents, STARTING_AGENTS, strict=True):
        runner.register_strategy(factory(symbol), agent.agent_id)
