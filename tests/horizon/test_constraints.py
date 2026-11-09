"""Tests for the hard constraint filter — never a scored factor, always a
pre-scoring filter. Drawdown/daily-loss breach removes ALL candidates except
HOLD; HOLD is always legal so the decision loop never has to invent a
fallback action."""
from decimal import Decimal

from evo_trader.horizon.config import HorizonConfig
from evo_trader.horizon.constraints import filter_candidates
from evo_trader.horizon.market_state import Candidate


class TestFilterCandidates:
    def setup_method(self):
        self.cfg = HorizonConfig()  # max_drawdown=0.15, daily_loss_limit=0.02
        self.all_candidates = [
            Candidate.ENTER_LONG, Candidate.ENTER_SHORT, Candidate.HOLD,
        ]

    def test_passes_through_when_within_limits(self):
        result = filter_candidates(
            self.all_candidates,
            current_drawdown=Decimal("0.05"),
            daily_loss_pct=Decimal("0.01"),
            cfg=self.cfg,
        )
        assert result == self.all_candidates

    def test_drawdown_breach_leaves_only_hold(self):
        result = filter_candidates(
            self.all_candidates,
            current_drawdown=Decimal("0.16"),  # > 0.15 max_drawdown
            daily_loss_pct=Decimal("0.0"),
            cfg=self.cfg,
        )
        assert result == [Candidate.HOLD]

    def test_daily_loss_breach_leaves_only_hold(self):
        result = filter_candidates(
            self.all_candidates,
            current_drawdown=Decimal("0.0"),
            daily_loss_pct=Decimal("0.03"),  # > 0.02 daily_loss_limit
            cfg=self.cfg,
        )
        assert result == [Candidate.HOLD]

    def test_drawdown_at_exact_limit_breaches(self):
        result = filter_candidates(
            self.all_candidates,
            current_drawdown=Decimal("0.15"),  # == max_drawdown
            daily_loss_pct=Decimal("0.0"),
            cfg=self.cfg,
        )
        assert result == [Candidate.HOLD]

    def test_hold_only_candidate_list_still_returns_hold_on_breach(self):
        result = filter_candidates(
            [Candidate.HOLD],
            current_drawdown=Decimal("0.20"),
            daily_loss_pct=Decimal("0.0"),
            cfg=self.cfg,
        )
        assert result == [Candidate.HOLD]

    def test_candidate_list_without_hold_still_returns_hold_on_breach(self):
        # Defensive: even if a caller passes a candidate set that never
        # included HOLD, breach must still produce a safe non-empty result.
        result = filter_candidates(
            [Candidate.ENTER_LONG, Candidate.ENTER_SHORT],
            current_drawdown=Decimal("0.20"),
            daily_loss_pct=Decimal("0.0"),
            cfg=self.cfg,
        )
        assert result == [Candidate.HOLD]
