"""
HorizonAgent — the one public entry point. Wires the market state adapter,
rollout scoring, hard constraint filter, ledger, and Jev tuner into the
spec's six-step decision loop. Never imports evo_trader.portfolio.manager or
evo_trader.evolution.engine — that import boundary IS Horizon's isolation
guarantee; do not add either import to this file.
"""
from __future__ import annotations

import random
from collections import deque
from decimal import Decimal
from pathlib import Path
from typing import Literal

from evo_trader.horizon.alpaca_client import AlpacaClientError, AlpacaPaperClient, Bar
from evo_trader.horizon.config import HorizonConfig
from evo_trader.horizon.constraints import filter_candidates
from evo_trader.horizon.ledger import HorizonLedger
from evo_trader.horizon.market_state import (
    BarsProvider,
    Candidate,
    MarketState,
    build_candidates,
    fetch_market_state,
)
from evo_trader.horizon.rollout import RolloutBranch, pick_best, score_candidates
from evo_trader.horizon.tuner import JevTuner, RolloutRecord, TuningLog

STARTING_CAPITAL = Decimal("10000")  # Horizon's own isolated paper pool
LOOKBACK_BARS = 50
# Rolling window size = cadence * this multiplier: 3 cadence-cycles of history.
EVIDENCE_WINDOW_MULTIPLIER = 3
ORDER_NOTIONAL = Decimal("100")

PositionDirection = Literal["long", "short"]


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
        bars_client: BarsProvider | None = None,
    ) -> None:
        self._client = client
        # ``bars_client`` overrides ``client`` for market-data fetches only.
        # ``client`` is still used for order submission via ``_submit_action``.
        self._bars_client: BarsProvider = bars_client if bars_client is not None else client
        self._cfg = cfg
        self._symbol = symbol
        self._ledger = HorizonLedger(starting_capital=STARTING_CAPITAL)
        self._tuner = JevTuner()
        self._tuning_log = TuningLog(tuning_log_path)
        self._bar_count = 0
        self._rollout_evidence: deque[RolloutRecord] = deque(
            maxlen=cfg.tuning_cadence_bars * EVIDENCE_WINDOW_MULTIPLIER
        )
        self._position_notional = Decimal("0")
        self._position_direction: PositionDirection | None = None

    def run_one_bar(self) -> Candidate:
        """Runs one full decision cycle and returns the ``Candidate`` that
        was *chosen* by the decision loop (``pick_best``'s output) — not
        necessarily one that was *executed*.

        ``HOLD`` never submits an order by design. All other candidates
        (``ENTER_LONG``, ``ENTER_SHORT``, ``EXIT``, ``RESIZE``, ``HEDGE``)
        submit an order via ``_submit_action``, which derives the correct
        order side from ``self._position_direction``. A failed submission
        does not retroactively downgrade the returned candidate — that was
        the decision actually made this bar.

        Callers must not assume "returned X" means "X was executed."
        """
        self._bar_count += 1

        try:
            state = fetch_market_state(
                self._bars_client,
                symbol=self._symbol,
                lookback_bars=LOOKBACK_BARS,
                has_open_position=self._position_direction is not None,
                position_notional=self._position_notional,
            )
        except Exception:  # noqa: BLE001 — catches both AlpacaClientError and YFinanceBarsError
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
            try:
                self._submit_action(action)
            except AlpacaClientError:
                # Order-submission failure: the decision was already made
                # (`action`), so unlike the fetch-failure path above we do
                # NOT retroactively downgrade to HOLD — that would misreport
                # what the loop actually chose. Never crash the loop; let
                # tuning-cadence and evidence bookkeeping proceed normally.
                pass

        if self._bar_count % self._cfg.tuning_cadence_bars == 0:
            self._run_tuning_cycle()

        return action

    def _submit_action(self, action: Candidate) -> None:
        """Submits an order for ``action``, updates position state, and
        records the fill with the ledger so risk gates (drawdown,
        daily_loss_pct) work correctly on subsequent bars.

        Order-side mapping:
        - ENTER_LONG  → buy;  position becomes long at ORDER_NOTIONAL
        - ENTER_SHORT → sell; position becomes short at ORDER_NOTIONAL
        - EXIT        → opposite of current direction; clears position
        - RESIZE      → same as current direction; increases notional
        - HEDGE       → opposite of current direction; reduces net exposure

        EXIT/RESIZE/HEDGE all require an open position; if called with no
        open position (``_position_direction is None``) they no-op — this
        mirrors the fail-closed pattern used elsewhere in Horizon.

        P&L at order submission is zero: Alpaca returns only an order-ID
        and status, not a fill price. ``record_fill(0, 0)`` advances the
        ledger's fill counter so the NAV/drawdown bookkeeping path is
        exercised every bar an order fires.

        Raises:
            AlpacaClientError: Propagated from ``submit_order``; callers
                catch this and leave position state unchanged.
        """
        if action == Candidate.ENTER_LONG:
            self._client.submit_order(self._symbol, side="buy", notional=ORDER_NOTIONAL)
            self._position_direction = "long"
            self._position_notional = ORDER_NOTIONAL
            self._ledger.record_fill(Decimal("0"), Decimal("0"))

        elif action == Candidate.ENTER_SHORT:
            self._client.submit_order(self._symbol, side="sell", notional=ORDER_NOTIONAL)
            self._position_direction = "short"
            self._position_notional = ORDER_NOTIONAL
            self._ledger.record_fill(Decimal("0"), Decimal("0"))

        elif action == Candidate.EXIT:
            if self._position_direction is None:
                return
            side = "sell" if self._position_direction == "long" else "buy"
            self._client.submit_order(self._symbol, side=side, notional=self._position_notional)
            self._position_direction = None
            self._position_notional = Decimal("0")
            self._ledger.record_fill(Decimal("0"), Decimal("0"))

        elif action == Candidate.RESIZE:
            if self._position_direction is None:
                return
            side = "buy" if self._position_direction == "long" else "sell"
            self._client.submit_order(self._symbol, side=side, notional=ORDER_NOTIONAL)
            self._position_notional += ORDER_NOTIONAL
            self._ledger.record_fill(Decimal("0"), Decimal("0"))

        elif action == Candidate.HEDGE:
            if self._position_direction is None:
                return
            # Partial hedge: send an opposite-side order at a fixed notional.
            # Does not clear the position — leaves net exposure reduced.
            side = "sell" if self._position_direction == "long" else "buy"
            self._client.submit_order(self._symbol, side=side, notional=ORDER_NOTIONAL)
            self._ledger.record_fill(Decimal("0"), Decimal("0"))

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
        before_tau = self._cfg.tau
        self._tuning_log.record(decision, before_tau=before_tau)
        if decision.applied:
            self._cfg.tau = decision.proposed_tau
