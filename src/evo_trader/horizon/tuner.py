"""
Jev-style tau tuner, per Jev_First_Standard's "trading system: regime
classification, signal-quality scoring" lane. The execution_mode gate is the
one hard rule in this module: "live" mode NEVER auto-applies, regardless of
evidence size or proposed delta — matching the doctrine's "never as trade
execution authorization... the human gate stays" clause at the point it
actually matters (real capital).

t_c exists in HorizonConfig and is the right design, but is NOT tuned here
yet. score_candidates/pick_best in rollout.py are pure branch-count diversity
with no competing objective — there is nothing for t_c to weight against. The
tuner only adjusts tau until the real Mode B rollout generator replaces the
placeholder and gives t_c a meaningful second axis to balance.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from evo_trader.horizon.config import HorizonConfig

logger = logging.getLogger(__name__)

MIN_EVIDENCE_WINDOW = 5  # fewer decisions than this → decline to propose


@dataclass(frozen=True)
class RolloutRecord:
    candidate_diversity_scores: list[int]
    price_stayed_in_branch: bool

    def __post_init__(self) -> None:
        if not self.candidate_diversity_scores:
            raise ValueError(
                "candidate_diversity_scores must not be empty — a rollout "
                "record with no scored candidates cannot be averaged and "
                "indicates malformed evidence upstream"
            )


@dataclass(frozen=True)
class TuningDecision:
    proposed_tau: int
    applied: bool
    reason: str
    miss_rate: Decimal
    avg_diversity: Decimal
    evidence_count: int


class JevTuner:
    def propose(
        self, cfg: HorizonConfig, evidence: list[RolloutRecord]
    ) -> TuningDecision | None:
        if len(evidence) < MIN_EVIDENCE_WINDOW:
            return None

        proposed_tau, miss_rate, avg_diversity = self._score_evidence(cfg, evidence)

        # No-op proposals never surface as a decision, in either mode: in
        # live mode, escalating nothing-to-approve every cycle degrades the
        # human gate's signal value (a human asked to rubber-stamp a no-op
        # repeatedly stops reading); in paper mode it's just log clutter.
        # This mirrors the cold-start "nothing to act on" signal — Task 7
        # already skips the cycle on `None`.
        if proposed_tau == cfg.tau:
            return None

        if cfg.execution_mode == "paper":
            return TuningDecision(
                proposed_tau=proposed_tau,
                applied=True,
                reason="paper mode: auto-applied, unbounded per Allan's explicit decision",
                miss_rate=miss_rate,
                avg_diversity=avg_diversity,
                evidence_count=len(evidence),
            )

        # execution_mode == "live" (HorizonConfig.__post_init__ already
        # rejects any other value) — always escalate, no exceptions.
        return TuningDecision(
            proposed_tau=proposed_tau,
            applied=False,
            reason="live mode: requires human approval before applying",
            miss_rate=miss_rate,
            avg_diversity=avg_diversity,
            evidence_count=len(evidence),
        )

    def _score_evidence(
        self, cfg: HorizonConfig, evidence: list[RolloutRecord]
    ) -> tuple[int, Decimal, Decimal]:
        """Minimal rollout-quality scoring: if branches consistently missed
        where price actually went, shorten tau (rollouts are stale faster
        than assumed).

        t_c is intentionally not tuned here. score_candidates/pick_best in
        rollout.py are pure branch-count diversity — there is no competing
        objective for t_c to weight against until the real Mode B rollout
        generator replaces the placeholder.

        Returns (proposed_tau, miss_rate, avg_diversity) — the two computed
        evidence metrics are returned alongside the proposal so callers can
        persist the reasoning behind a decision, not just its outcome
        (required for a human to reconstruct the trajectory at the live
        go/no-go gate)."""
        miss_rate = Decimal(1) - (
            Decimal(sum(1 for r in evidence if r.price_stayed_in_branch))
            / Decimal(len(evidence))
        )
        avg_diversity = sum(
            (
                Decimal(sum(r.candidate_diversity_scores))
                / Decimal(len(r.candidate_diversity_scores))
            )
            for r in evidence
        ) / Decimal(len(evidence))

        proposed_tau = cfg.tau
        if miss_rate > Decimal("0.3"):
            proposed_tau = max(1, cfg.tau - 1)

        return proposed_tau, miss_rate, avg_diversity


class TuningLog:
    """Append-only JSONL log — mirrors evo_trader.evolution.transaction_log.
    EvolutionLog's pattern. Every proposed/applied decision lands here
    regardless of execution_mode, so a human reviewing before go-live can
    reconstruct the full tau trajectory."""

    def __init__(self, log_path: Path | str) -> None:
        self._path = Path(log_path)

    def record(self, decision: TuningDecision, before_tau: int) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "before_tau": before_tau,
            "proposed_tau": decision.proposed_tau,
            "applied": decision.applied,
            "reason": decision.reason,
            "miss_rate": str(decision.miss_rate),
            "avg_diversity": str(decision.avg_diversity),
            "evidence_count": decision.evidence_count,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(
            "Tuning decision: applied=%s tau %s->%s",
            decision.applied, before_tau, decision.proposed_tau,
        )
