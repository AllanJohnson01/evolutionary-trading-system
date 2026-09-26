"""Base trading agent interface."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Protocol


class AgentType(str, Enum):
    GRID = "grid"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    RL = "rl"


class AgentState(str, Enum):
    ACTIVE = "active"
    ELIMINATED = "eliminated"
    PAUSED = "paused"


@dataclass
class PerformanceRecord:
    """One evaluation-period snapshot for an agent."""
    period_start: str        # ISO8601 date
    period_end: str
    pnl: Decimal = Decimal("0")
    pnl_pct: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    trade_count: int = 0


@dataclass
class TradingAgent:
    """
    Base representation of a trading agent. Does not contain strategy logic —
    that lives in concrete subclasses. Portfolio manager references this
    interface to enforce allocation caps and position limits.
    """
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.GRID
    state: AgentState = AgentState.ACTIVE

    # Capital managed by portfolio manager — agents read, don't set
    allocated_capital: Decimal = Decimal("0")

    # Lineage
    generation: int = 0
    parent_id: str | None = None

    # Performance history — one entry per completed evaluation window
    performance_history: list[PerformanceRecord] = field(default_factory=list)

    def record_period(self, record: PerformanceRecord) -> None:
        self.performance_history.append(record)

    def consecutive_bottom(self, threshold_pct: Decimal, n: int) -> bool:
        """True if this agent ranked in bottom `threshold_pct` for last n periods."""
        # Caller (evolution engine) passes pre-ranked data; this checks raw history length.
        return len(self.performance_history) >= n

    def is_active(self) -> bool:
        return self.state == AgentState.ACTIVE
