"""
Hard constraint filter — Horizon's own risk limits. Filters candidates
BEFORE scoring; never participates in the optionality score itself. On a
breach, HOLD is always the surviving action — the decision loop (Task 4)
never has to invent a fallback for an empty candidate list.
"""
from __future__ import annotations

from decimal import Decimal

from evo_trader.horizon.config import HorizonConfig
from evo_trader.horizon.market_state import Candidate


def filter_candidates(
    candidates: list[Candidate],
    current_drawdown: Decimal,
    daily_loss_pct: Decimal,
    cfg: HorizonConfig,
) -> list[Candidate]:
    if current_drawdown >= cfg.max_drawdown or daily_loss_pct >= cfg.daily_loss_limit:
        return [Candidate.HOLD]
    return candidates
