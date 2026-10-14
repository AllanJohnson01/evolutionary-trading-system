"""Tests for the Jev-style tuner and execution_mode gate. Paper mode applies
unbounded; live mode always escalates regardless of delta size. Cold start
(too little evidence) declines to propose rather than guessing."""
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
    return [
        RolloutRecord(candidate_diversity_scores=[3, 1, 2], price_stayed_in_branch=True)
        for _ in range(n)
    ]


class TestColdStart:
    def test_insufficient_evidence_returns_none(self):
        tuner = JevTuner()
        cfg = HorizonConfig()
        decision = tuner.propose(cfg, evidence=_good_evidence(MIN_EVIDENCE_WINDOW - 1))
        assert decision is None

    def test_exact_minimum_evidence_proposes(self):
        tuner = JevTuner()
        cfg = HorizonConfig()
        decision = tuner.propose(cfg, evidence=_good_evidence(MIN_EVIDENCE_WINDOW))
        assert decision is not None


class TestPaperModeAutoApplies:
    def test_proposal_applies_automatically_in_paper_mode(self):
        tuner = JevTuner()
        cfg = HorizonConfig(execution_mode="paper")
        decision = tuner.propose(cfg, evidence=_good_evidence(MIN_EVIDENCE_WINDOW))
        assert decision.applied is True
        assert "paper" in decision.reason.lower()


class TestLiveModeAlwaysEscalates:
    def test_small_delta_still_escalates_in_live_mode(self):
        tuner = JevTuner()
        cfg = HorizonConfig(execution_mode="live")
        decision = tuner.propose(cfg, evidence=_good_evidence(MIN_EVIDENCE_WINDOW))
        assert decision.applied is False
        assert "approval" in decision.reason.lower()

    def test_live_mode_never_auto_applies_regardless_of_evidence_size(self):
        # Defensive: even a very large, very clean evidence window must not
        # auto-apply once execution_mode is "live" — no code path may bypass
        # the human gate based on evidence quality/quantity.
        tuner = JevTuner()
        cfg = HorizonConfig(execution_mode="live")
        decision = tuner.propose(cfg, evidence=_good_evidence(MIN_EVIDENCE_WINDOW * 10))
        assert decision.applied is False


class TestTuningLog:
    def test_record_appends_jsonl_with_before_and_after(self, tmp_path: Path):
        log_path = tmp_path / "tuning_log.jsonl"
        log = TuningLog(log_path)
        decision = TuningDecision(
            proposed_tau=12, proposed_t_c=Decimal("1.1"),
            applied=True, reason="paper mode: auto-applied",
        )

        log.record(decision, before_tau=10, before_t_c=Decimal("1.0"))

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["before_tau"] == 10
        assert entry["proposed_tau"] == 12
        assert entry["applied"] is True

    def test_record_is_append_only_across_instances(self, tmp_path: Path):
        log_path = tmp_path / "tuning_log.jsonl"
        decision = TuningDecision(
            proposed_tau=11, proposed_t_c=Decimal("1.0"),
            applied=False, reason="live mode: pending approval",
        )

        TuningLog(log_path).record(decision, before_tau=10, before_t_c=Decimal("1.0"))
        TuningLog(log_path).record(decision, before_tau=11, before_t_c=Decimal("1.0"))

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
