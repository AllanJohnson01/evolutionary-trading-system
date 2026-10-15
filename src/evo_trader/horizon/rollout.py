"""
Rollout scoring — implements the causal-entropic-optionality skill's
procedure: enumerate candidates, roll forward tau steps per candidate
(Mode B: single-call virtual rollout), score by reachable-future-diversity,
pick the winner. branch_generator is injected so this stays deterministic
and testable without a live model call; see HorizonAgent (Task 7) for the
production branch_generator that actually invokes the skill.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from evo_trader.horizon.alpaca_client import Bar
from evo_trader.horizon.market_state import Candidate, MarketState


@dataclass(frozen=True)
class RolloutBranch:
    end_bar: Bar
    reachable_candidates: list[Candidate]


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: Candidate
    branches: list[RolloutBranch]
    diversity_score: int


BranchGenerator = Callable[[Candidate, MarketState, int], list[RolloutBranch]]


def score_candidates(
    candidates: list[Candidate],
    state: MarketState,
    tau: int,
    branch_generator: BranchGenerator,
) -> list[ScoredCandidate]:
    """Diversity score = count of distinct reachable branches found by the
    rollout for that candidate — NOT expected value. A candidate that
    forecloses future branches scores worse even if its direct expected
    value looks good, per the skill's core instruction."""
    scored = []
    for candidate in candidates:
        branches = branch_generator(candidate, state, tau)
        scored.append(
            ScoredCandidate(
                candidate=candidate, branches=branches, diversity_score=len(branches)
            )
        )
    return scored


def pick_best(scored: list[ScoredCandidate]) -> Candidate:
    """Ties resolve to the first-listed candidate — deterministic, never
    dependent on iteration order of an unordered collection."""
    best = max(scored, key=lambda s: s.diversity_score)
    return best.candidate
