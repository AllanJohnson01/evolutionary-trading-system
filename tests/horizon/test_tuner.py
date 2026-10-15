"""Tests for the Jev-style tau tuner and execution_mode gate. Paper mode applies
unbounded; live mode always escalates regardless of delta size. Cold start
(too little evidence) declines to propose rather than guessing.

t_c is NOT tuned — score_candidates/pick_best in rollout.py have no competing
objective for t_c to weight against until the real Mode B rollout lands."""
import json
from decimal import Decimal
from pathlib import Path

import pytest

from evo_trader.horizon.config import HorizonConfig
from evo_trader.horizon.tuner import (
    MIN_EVIDENCE_WINDOW,
    JevTuner,
    RolloutRecord,
    TuningDecision,
    TuningLog,
)


def _good_evidence(n: int) -> list[RolloutRecord]:
    """Evidence that matches a default HorizonConfig exactly: 0% miss rate,
    avg diversity 2.0 (>= 1.5) — produces a no-op proposal (no change from
    cfg.tau)."""
    return [
        RolloutRecord(candidate_diversity_scores=[3, 1, 2], price_stayed_in_branch=True)
        for _ in range(n)
    ]


def _bad_evidence(n: int) -> list[RolloutRecord]:
    """Evidence that triggers the tau scoring branch: mostly missed branches
    (miss_rate > 0.3) — against a default HorizonConfig this proposes a
    real, non-zero tau delta, not a no-op."""
    missed = [
        RolloutRecord(candidate_diversity_scores=[1, 1, 1], price_stayed_in_branch=False)
        for _ in range(n)
    ]
    return missed


class TestColdStart:
    def test_insufficient_evidence_returns_none(self):
        tuner = JevTuner()
        cfg = HorizonConfig()
        decision = tuner.propose(cfg, evidence=_good_evidence(MIN_EVIDENCE_WINDOW - 1))
        assert decision is None

    def test_exact_minimum_evidence_with_real_delta_proposes(self):
        tuner = JevTuner()
        cfg = HorizonConfig()
        decision = tuner.propose(cfg, evidence=_bad_evidence(MIN_EVIDENCE_WINDOW))
        assert decision is not None


class TestNoOpProposalReturnsNone:
    """Design decision: when the computed proposal exactly matches the
    current cfg.tau, propose() must return None — the same "nothing to act
    on" signal as cold-start — in both modes. Escalating a no-op in live
    mode every cycle degrades the human gate's signal value; in paper mode
    it's just log clutter."""

    def test_unchanged_proposal_returns_none_in_paper_mode(self):
        tuner = JevTuner()
        cfg = HorizonConfig(execution_mode="paper")
        decision = tuner.propose(cfg, evidence=_good_evidence(MIN_EVIDENCE_WINDOW))
        assert decision is None

    def test_unchanged_proposal_returns_none_in_live_mode(self):
        tuner = JevTuner()
        cfg = HorizonConfig(execution_mode="live")
        decision = tuner.propose(cfg, evidence=_good_evidence(MIN_EVIDENCE_WINDOW))
        assert decision is None


class TestPaperModeAutoApplies:
    def test_proposal_applies_automatically_in_paper_mode(self):
        tuner = JevTuner()
        cfg = HorizonConfig(execution_mode="paper")
        decision = tuner.propose(cfg, evidence=_bad_evidence(MIN_EVIDENCE_WINDOW))
        assert decision.applied is True
        assert "paper" in decision.reason.lower()


class TestLiveModeAlwaysEscalates:
    def test_real_delta_still_escalates_in_live_mode(self):
        tuner = JevTuner()
        cfg = HorizonConfig(execution_mode="live")
        evidence = _bad_evidence(MIN_EVIDENCE_WINDOW)
        decision = tuner.propose(cfg, evidence=evidence)

        assert decision is not None
        # Prove this is a real, non-zero tau delta — not a no-op escalation.
        assert decision.proposed_tau != cfg.tau
        assert decision.applied is False
        assert "approval" in decision.reason.lower()

    def test_live_mode_never_auto_applies_regardless_of_evidence_size(self):
        # Defensive: even a very large evidence window with a real,
        # non-zero proposed delta must not auto-apply once execution_mode
        # is "live" — no code path may bypass the human gate based on
        # evidence quality/quantity.
        tuner = JevTuner()
        cfg = HorizonConfig(execution_mode="live")
        evidence = _bad_evidence(MIN_EVIDENCE_WINDOW * 10)
        decision = tuner.propose(cfg, evidence=evidence)

        assert decision is not None
        assert decision.proposed_tau != cfg.tau
        assert decision.applied is False


class TestTuningLog:
    def test_record_appends_jsonl_with_before_and_after(self, tmp_path: Path):
        log_path = tmp_path / "tuning_log.jsonl"
        log = TuningLog(log_path)
        decision = TuningDecision(
            proposed_tau=12,
            applied=True, reason="paper mode: auto-applied",
            miss_rate=Decimal("0.4"), avg_diversity=Decimal("1.0"),
            evidence_count=5,
        )

        log.record(decision, before_tau=10)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["before_tau"] == 10
        assert entry["proposed_tau"] == 12
        assert entry["applied"] is True
        assert entry["miss_rate"] == "0.4"
        assert entry["avg_diversity"] == "1.0"
        assert entry["evidence_count"] == 5

    def test_record_is_append_only_across_instances(self, tmp_path: Path):
        log_path = tmp_path / "tuning_log.jsonl"
        decision = TuningDecision(
            proposed_tau=11,
            applied=False, reason="live mode: pending approval",
            miss_rate=Decimal("0.4"), avg_diversity=Decimal("1.0"),
            evidence_count=5,
        )

        TuningLog(log_path).record(decision, before_tau=10)
        TuningLog(log_path).record(decision, before_tau=11)

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2


class TestRolloutRecordValidation:
    def test_empty_diversity_scores_raises(self):
        with pytest.raises(ValueError):
            RolloutRecord(candidate_diversity_scores=[], price_stayed_in_branch=True)
