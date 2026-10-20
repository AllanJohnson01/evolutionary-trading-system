"""
Starting population: the 4 diverse agents and their strategies.

Single source of truth shared by ``ops/scripts/burn_in.py`` and the tactical
runner's ``--once`` entry point, so both always run the same population.

Agent identities are stable across restarts. ``init_population`` accepts an
optional ``stable_ids`` mapping (name → agent_id string). When supplied the
same IDs are reused; otherwise new UUIDs are minted (first-run case). The
manifest written by ``burn_in.py`` records those IDs so subsequent runs can
reload them.
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

# (name, agent_type, strategy factory)
# The strategy_id prefix passed to each factory must match this table so
# decisions.jsonl family-prefix analysis stays consistent.
STARTING_AGENTS: list[tuple[str, AgentType, Callable[[str], Strategy]]] = [
    ("Momentum-Hunter", AgentType.MOMENTUM, IntradayMomentumStrategy),
    ("Breakout", AgentType.MOMENTUM, BreakoutStrategy),  # breakout is in the momentum family
    ("Grid-Trader", AgentType.GRID, GridStrategy),
    ("Mean-Reversion", AgentType.MEAN_REVERSION, MeanReversionStrategy),
]

# Stable strategy-id prefix per agent name — used when building strategy_id from agent_id.
_STRATEGY_PREFIX: dict[str, str] = {
    "Momentum-Hunter": "intraday_momentum",
    "Breakout": "breakout",
    "Grid-Trader": "grid",
    "Mean-Reversion": "mean_reversion",
}


def _stable_strategy_id(agent_name: str, agent_id: str) -> str:
    """Return a stable strategy_id derived from agent_id, preserving the family prefix."""
    prefix = _STRATEGY_PREFIX.get(agent_name, agent_name.lower().replace("-", "_"))
    # Use first 8 hex chars of agent_id (without dashes) as the suffix.
    suffix = agent_id.replace("-", "")[:8]
    return f"{prefix}_{suffix}"


def init_population(
    portfolio_manager: PortfolioManager,
    capital_cfg: CapitalConfig,
    stable_ids: dict[str, str] | None = None,
) -> list[TradingAgent]:
    """Register the starting agents with equal capital allocation.

    Args:
        portfolio_manager: Enforces risk limits and holds per-agent allocations.
        capital_cfg: Capital configuration (provides per-agent allocation).
        stable_ids: Optional mapping of agent name → agent_id string. When
            supplied the same IDs are reused across restarts; otherwise fresh
            UUIDs are minted (first-run). ``burn_in.py`` writes these into
            ``state/population.json`` and passes them back on subsequent runs.
    """
    per_agent = capital_cfg.initial_agent_allocation
    agents: list[TradingAgent] = []

    for name, agent_type, _ in STARTING_AGENTS:
        agent_id = (stable_ids or {}).get(name) or str(uuid.uuid4())
        agent = TradingAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            state=AgentState.ACTIVE,
            allocated_capital=per_agent,
            generation=0,
        )
        portfolio_manager.register_agent(agent, per_agent)
        logger.info("Registered agent %s (%s) id=%s with $%s", name, agent_type.value, agent_id, per_agent)
        agents.append(agent)

    return agents


def register_starting_strategies(
    runner: "StrategyRunner", agents: list[TradingAgent], symbol: str
) -> None:
    """Register one strategy per agent (same order as ``STARTING_AGENTS``).

    Strategy IDs are derived from the agent's stable ``agent_id`` so that the
    same strategy_id appears in ``decisions.jsonl`` across restarts.
    """
    for agent, (name, _, factory) in zip(agents, STARTING_AGENTS, strict=True):
        sid = _stable_strategy_id(name, agent.agent_id)
        runner.register_strategy(factory(symbol, strategy_id=sid), agent.agent_id)
