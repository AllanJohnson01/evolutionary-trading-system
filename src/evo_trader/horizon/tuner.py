"""
Jev-style tau/t_c tuner, per Jev_First_Standard's "trading system: regime
classification, signal-quality scoring" lane. The execution_mode gate is the
one hard rule in this module: "live" mode NEVER auto-applies, regardless of
evidence size or proposed delta — matching the doctrine's "never as trade
execution authorization... the human gate stays" clause at the point it
actually matters (real capital).
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


@dataclass(frozen=True)
class TuningDecision:
    proposed_tau: int
    proposed_t_c: Decimal
    applied: bool
    reason: str


class JevTuner:
    def propose(
        self, cfg: HorizonConfig, evidence: list[RolloutRecord]
    ) -> TuningDecision | None:
        if len(evidence) < MIN_EVIDENCE_WINDOW:
            return None

        proposed_tau, proposed_t_c = self._score_evidence(cfg, evidence)

        if cfg.execution_mode == "paper":
            return TuningDecision(
                proposed_tau=proposed_tau,
                proposed_t_c=proposed_t_c,
                applied=True,
                reason="paper mode: auto-applied, unbounded per Allan's explicit decision",
            )

        # execution_mode == "live" (HorizonConfig.__post_init__ already
        # rejects any other value) — always escalate, no exceptions.
        return TuningDecision(
            proposed_tau=proposed_tau,
            proposed_t_c=proposed_t_c,
            applied=False,
            reason="live mode: requires human approval before applying",
        )

    def _score_evidence(
        self, cfg: HorizonConfig, evidence: list[RolloutRecord]
    ) -> tuple[int, Decimal]:
        """Minimal rollout-quality scoring: if branches consistently missed
        where price actually went, shorten tau (rollouts are stale faster
        than assumed); if diversity scores are consistently low, raise t_c
        slightly (optionality isn't dominating the pick enough to matter).
        This is intentionally simple for v1 — refining the scoring heuristic
        is expected future work, not something this task blocks on."""
        miss_rate = 1 - (
            sum(1 for r in evidence if r.price_stayed_in_branch) / len(evidence)
        )
        avg_diversity = sum(
            sum(r.candidate_diversity_scores) / len(r.candidate_diversity_scores)
            for r in evidence
        ) / len(evidence)

        proposed_tau = cfg.tau
        if miss_rate > 0.3:
            proposed_tau = max(1, cfg.tau - 1)

        proposed_t_c = cfg.t_c
        if avg_diversity < 1.5:
            proposed_t_c = cfg.t_c + Decimal("0.1")

        return proposed_tau, proposed_t_c


class TuningLog:
    """Append-only JSONL log — mirrors evo_trader.evolution.transaction_log.
    EvolutionLog's pattern. Every proposed/applied decision lands here
    regardless of execution_mode, so a human reviewing before go-live can
    reconstruct the full tau/t_c trajectory."""

    def __init__(self, log_path: Path | str) -> None:
        self._path = Path(log_path)

    def record(
        self, decision: TuningDecision, before_tau: int, before_t_c: Decimal
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "before_tau": before_tau,
            "before_t_c": str(before_t_c),
            "proposed_tau": decision.proposed_tau,
            "proposed_t_c": str(decision.proposed_t_c),
            "applied": decision.applied,
            "reason": decision.reason,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(
            "Tuning decision: applied=%s tau %s->%s t_c %s->%s",
            decision.applied, before_tau, decision.proposed_tau,
            before_t_c, decision.proposed_t_c,
        )
