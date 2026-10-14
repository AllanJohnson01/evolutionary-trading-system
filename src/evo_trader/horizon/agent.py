"""
HorizonAgent — the one public entry point. Wires the market state adapter,
rollout scoring, hard constraint filter, ledger, and Jev tuner into the
spec's six-step decision loop. Never imports evo_trader.portfolio.manager or
evo_trader.evolution.engine — that import boundary IS Horizon's isolation
guarantee; do not add either import to this file.
"""
from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path

from evo_trader.horizon.alpaca_client import AlpacaClientError, AlpacaPaperClient, Bar
from evo_trader.horizon.config import HorizonConfig
from evo_trader.horizon.constraints import filter_candidates
from evo_trader.horizon.ledger import HorizonLedger
from evo_trader.horizon.market_state import (
    Candidate,
    MarketState,
    build_candidates,
    fetch_market_state,
)
from evo_trader.horizon.rollout import RolloutBranch, pick_best, score_candidates
from evo_trader.horizon.tuner import JevTuner, RolloutRecord, TuningLog

STARTING_CAPITAL = Decimal("10000")  # Horizon's own isolated paper pool
LOOKBACK_BARS = 50


def _placeholder_branch_generator(
    candidate: Candidate, state: MarketState, tau: int
) -> list[RolloutBranch]:
    """Stand-in for a real Mode B rollout: perturbs the last close price to
    produce 2-3 plausible branches. NOT a production rollout — replacing
    this with an actual model-driven Mode B call (per the
    causal-entropic-optionality skill) is tracked as follow-up work, not
    part of this task. Kept deterministic-ish via a fixed seed per call so
    tests aren't flaky; production replacement will remove the seed.
    """
    last_close = state.bars[-1].close if state.bars else Decimal("100")
    rng = random.Random(hash((candidate, tau)))
    branch_count = rng.randint(2, 3)
    branches = []
    for _ in range(branch_count):
        pct_move = Decimal(str(rng.uniform(-0.02, 0.02)))
        new_close = last_close * (Decimal("1") + pct_move)
        end_bar = Bar(
            timestamp=state.bars[-1].timestamp if state.bars else "",
            open=new_close, high=new_close, low=new_close, close=new_close,
            volume=Decimal("0"),
        )
        branches.append(
            RolloutBranch(end_bar=end_bar, reachable_candidates=[Candidate.HOLD])
        )
    return branches


class HorizonAgent:
    def __init__(
        self,
        client: AlpacaPaperClient,
        cfg: HorizonConfig,
        symbol: str,
        tuning_log_path: Path | str,
    ) -> None:
        self._client = client
        self._cfg = cfg
        self._symbol = symbol
        self._ledger = HorizonLedger(starting_capital=STARTING_CAPITAL)
        self._tuner = JevTuner()
        self._tuning_log = TuningLog(tuning_log_path)
        self._bar_count = 0
        self._rollout_evidence: list[RolloutRecord] = []
        self._position_notional = Decimal("0")

    def run_one_bar(self) -> Candidate:
        self._bar_count += 1

        try:
            state = fetch_market_state(
                self._client,
                symbol=self._symbol,
                lookback_bars=LOOKBACK_BARS,
                has_open_position=self._position_notional != 0,
                position_notional=self._position_notional,
            )
        except AlpacaClientError:
            # Data-fetch failure: never crash the loop, never submit a
            # stale/partial order. HOLD is always safe.
            return Candidate.HOLD

        candidates = build_candidates(has_open_position=state.has_open_position)
        candidates = filter_candidates(
            candidates,
            current_drawdown=self._ledger.drawdown,
            daily_loss_pct=self._ledger.daily_loss_pct(
                today_realized=Decimal("0"), today_unrealized=Decimal("0"),
                starting_capital=STARTING_CAPITAL,
            ),
            cfg=self._cfg,
        )

        scored = score_candidates(
            candidates, state, tau=self._cfg.tau,
            branch_generator=_placeholder_branch_generator,
        )
        action = pick_best(scored)

        self._rollout_evidence.append(
            RolloutRecord(
                candidate_diversity_scores=[s.diversity_score for s in scored],
                price_stayed_in_branch=True,  # placeholder generator always "fits"
            )
        )

        if action != Candidate.HOLD:
            self._submit_action(action)

        if self._bar_count % self._cfg.tuning_cadence_bars == 0:
            self._run_tuning_cycle()

        return action

    def _submit_action(self, action: Candidate) -> None:
        side = "buy" if action in (Candidate.ENTER_LONG, Candidate.RESIZE) else "sell"
        self._client.submit_order(
            self._symbol, side=side, notional=Decimal("100")
        )

    def _run_tuning_cycle(self) -> None:
        """Runs one Jev-tuner cycle and, if it produced a decision, logs it
        and (in paper mode) applies it to the live ``self._cfg``.

        ``JevTuner.propose()`` returns ``None`` in two distinct cases: cold
        start (insufficient evidence) and a no-op proposal (computed
        (tau, t_c) exactly equals the current values). Both mean the same
        thing to this caller — "nothing to act on this cycle" — so both are
        handled identically here: skip the cycle, log nothing, leave
        ``self._cfg`` untouched. Distinguishing *why* `propose()` returned
        `None` is not this method's concern; that reasoning lives entirely
        in `JevTuner`.
        """
        decision = self._tuner.propose(self._cfg, evidence=self._rollout_evidence)
        if decision is None:
            return  # cold start or no-op — nothing to act on, per the tuner
        before_tau, before_t_c = self._cfg.tau, self._cfg.t_c
        self._tuning_log.record(decision, before_tau=before_tau, before_t_c=before_t_c)
        if decision.applied:
            self._cfg.tau = decision.proposed_tau
            self._cfg.t_c = decision.proposed_t_c
