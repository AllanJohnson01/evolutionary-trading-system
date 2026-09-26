"""
System-wide parameter constants. All financial limits live here — never hardcode
these values elsewhere. Portfolio manager and evolution engine import from this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final


# ---------------------------------------------------------------------------
# Capital allocation (paper trading baseline: $100K virtual)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapitalConfig:
    total_capital: Decimal = Decimal("100000")

    # Proportions
    active_trading_fraction: Decimal = Decimal("0.60")   # 60% → $60K across 4 agents
    dca_fraction: Decimal = Decimal("0.25")              # 25% → $25K DCA pool
    cash_reserve_fraction: Decimal = Decimal("0.15")     # 15% → $15K reserve

    # Derived values
    @property
    def active_trading_total(self) -> Decimal:
        return self.total_capital * self.active_trading_fraction

    @property
    def dca_total(self) -> Decimal:
        return self.total_capital * self.dca_fraction

    @property
    def cash_reserve_total(self) -> Decimal:
        return self.total_capital * self.cash_reserve_fraction

    @property
    def initial_agent_allocation(self) -> Decimal:
        """Equal split across 4 starting agents."""
        return self.active_trading_total / 4


# ---------------------------------------------------------------------------
# Risk management parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskConfig:
    # Portfolio-level limits (enforced by portfolio manager, never by agents)
    max_drawdown: Decimal = Decimal("0.15")          # 15% — kill switch
    daily_loss_limit: Decimal = Decimal("0.02")      # 2% of total capital per day

    # Agent-level limits
    per_agent_allocation_cap: Decimal = Decimal("0.20")   # 20% of active trading pool
    per_asset_concentration: Decimal = Decimal("0.15")    # 15% of agent's own book


# ---------------------------------------------------------------------------
# Evolution engine rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvolutionConfig:
    rebalance_cadence_days: int = 7                  # Weekly, Monday open
    evaluation_window_weeks: int = 4                 # 4-week performance window
    evaluation_window_days: int = 28                 # Derived from above

    elimination_threshold: Decimal = Decimal("0.25")     # Bottom 25% of agents
    elimination_consecutive_periods: int = 2             # 2 consecutive eval periods

    clone_threshold: Decimal = Decimal("0.25")           # Top 25% of agents
    clone_consecutive_periods: int = 2                   # 2 consecutive eval periods

    # Mutation applied when cloning a top performer
    mutation_rate_min: Decimal = Decimal("0.10")         # 10% parameter variance
    mutation_rate_max: Decimal = Decimal("0.20")         # 20% parameter variance

    # Population bounds
    initial_agent_count: int = 4
    max_agent_count: int = 8


# ---------------------------------------------------------------------------
# Market configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketConfig:
    # Feature flags — set FOREX_ENABLED=true in env to unlock forex
    crypto_enabled: bool = True
    forex_enabled: bool = False   # Phase 2 — gated

    # Primary crypto pairs (sufficient liquidity, varied volatility)
    crypto_pairs: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT")

    # Forex secondary pairs (London/NY overlap session only when enabled)
    forex_pairs: tuple[str, ...] = ("EUR/USD", "GBP/USD")

    # Equities deferred — not in this phase


# ---------------------------------------------------------------------------
# Per-agent configuration schema
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """
    Mutable config for a single trading agent instance. Portfolio manager
    reads per_asset_concentration and allocation_cap to enforce position limits.
    Cloned agents receive mutated copies of this config.
    """
    agent_id: str = ""
    agent_type: str = ""                             # grid | momentum | mean_reversion | rl

    # Capital position — set by portfolio manager on creation/rebalance
    allocated_capital: Decimal = Decimal("0")
    allocation_cap: Decimal = field(
        default_factory=lambda: RiskConfig().per_agent_allocation_cap
    )

    # Per-asset concentration limit for this agent
    per_asset_concentration: Decimal = field(
        default_factory=lambda: RiskConfig().per_asset_concentration
    )

    # Lineage tracking for evolution audit trail
    generation: int = 0
    parent_id: str | None = None


# ---------------------------------------------------------------------------
# Assembled system config singleton
# ---------------------------------------------------------------------------

SYSTEM_CONFIG: Final = {
    "capital": CapitalConfig(),
    "risk": RiskConfig(),
    "evolution": EvolutionConfig(),
    "markets": MarketConfig(),
}
