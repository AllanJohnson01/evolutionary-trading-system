"""Tests for rollout scoring — the causal-entropic-optionality skill's
procedure (enumerate, roll forward, score by diversity, pick), Mode B.
branch_generator is injected so these tests are deterministic and don't
call out to a real model."""
import random
from decimal import Decimal

from evo_trader.horizon.alpaca_client import Bar
from evo_trader.horizon.market_state import Candidate, MarketState
from evo_trader.horizon.rollout import (
    RolloutBranch,
    ScoredCandidate,
    pick_best,
    score_candidates,
)

SOME_BAR = Bar(
    timestamp="2026-01-01T00:00:00",
    open=Decimal("100"), high=Decimal("101"), low=Decimal("99"),
    close=Decimal("100.5"), volume=Decimal("1000"),
)
STATE = MarketState(
    symbol="AAPL", bars=[SOME_BAR], has_open_position=False,
    position_notional=Decimal("0"),
)


def _fixed_branch_generator(branch_counts: dict[Candidate, int]):
    """Returns a branch_generator that reports a fixed number of reachable
    branches per candidate, regardless of state/tau — for deterministic
    diversity-score assertions."""
    def generator(candidate, state, tau):
        count = branch_counts.get(candidate, 0)
        return [
            RolloutBranch(end_bar=SOME_BAR, reachable_candidates=[Candidate.HOLD])
            for _ in range(count)
        ]
    return generator


class TestScoreCandidates:
    def test_scores_each_candidate_by_branch_count(self):
        candidates = [Candidate.ENTER_LONG, Candidate.HOLD]
        generator = _fixed_branch_generator(
            {Candidate.ENTER_LONG: 3, Candidate.HOLD: 1}
        )

        scored = score_candidates(candidates, STATE, tau=5, branch_generator=generator)

        by_candidate = {s.candidate: s for s in scored}
        assert by_candidate[Candidate.ENTER_LONG].diversity_score == 3
        assert by_candidate[Candidate.HOLD].diversity_score == 1

    def test_empty_candidate_list_returns_empty(self):
        generator = _fixed_branch_generator({})
        scored = score_candidates([], STATE, tau=5, branch_generator=generator)
        assert scored == []

    def test_candidate_with_zero_branches_scores_zero(self):
        candidates = [Candidate.EXIT]
        generator = _fixed_branch_generator({})  # no entry for EXIT → 0

        scored = score_candidates(candidates, STATE, tau=5, branch_generator=generator)

        assert scored[0].diversity_score == 0


class TestPickBest:
    def test_picks_highest_diversity_score(self):
        scored = [
            ScoredCandidate(candidate=Candidate.HOLD, branches=[], diversity_score=1),
            ScoredCandidate(candidate=Candidate.ENTER_LONG, branches=[], diversity_score=5),
            ScoredCandidate(candidate=Candidate.ENTER_SHORT, branches=[], diversity_score=2),
        ]
        assert pick_best(scored) == Candidate.ENTER_LONG

    def test_single_candidate_is_picked(self):
        scored = [
            ScoredCandidate(candidate=Candidate.HOLD, branches=[], diversity_score=0),
        ]
        assert pick_best(scored) == Candidate.HOLD

    def test_tie_picks_uniformly_from_tied_set(self):
        # Tie-break must be uniform-random among all tied candidates so that
        # no action type (e.g. ENTER_LONG) is implicitly preferred by list
        # order.  A seeded RNG makes the assertion deterministic.
        scored = [
            ScoredCandidate(candidate=Candidate.HOLD, branches=[], diversity_score=3),
            ScoredCandidate(candidate=Candidate.ENTER_LONG, branches=[], diversity_score=3),
            ScoredCandidate(candidate=Candidate.ENTER_SHORT, branches=[], diversity_score=3),
        ]
        rng = random.Random(0)
        results = {pick_best(scored, rng=rng) for _ in range(60)}
        assert results == {Candidate.HOLD, Candidate.ENTER_LONG, Candidate.ENTER_SHORT}

    def test_tie_seeded_rng_is_deterministic(self):
        scored = [
            ScoredCandidate(candidate=Candidate.HOLD, branches=[], diversity_score=3),
            ScoredCandidate(candidate=Candidate.ENTER_LONG, branches=[], diversity_score=3),
        ]
        result_a = pick_best(scored, rng=random.Random(42))
        result_b = pick_best(scored, rng=random.Random(42))
        assert result_a == result_b

    def test_empty_scored_list_raises(self):
        import pytest
        with pytest.raises(ValueError, match="at least one"):
            pick_best([])
