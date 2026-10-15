"""Tests for HorizonAgent — wires all six Horizon modules into one
run_one_bar() call. Alpaca client is mocked throughout; no network calls."""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from evo_trader.horizon.agent import HorizonAgent
from evo_trader.horizon.alpaca_client import AlpacaClientError, Bar
from evo_trader.horizon.config import HorizonConfig
from evo_trader.horizon.market_state import Candidate
from evo_trader.horizon.tuner import RolloutRecord

SOME_BAR = Bar(
    timestamp="2026-01-01T00:00:00",
    open=Decimal("100"), high=Decimal("101"), low=Decimal("99"),
    close=Decimal("100.5"), volume=Decimal("1000"),
)


class TestRunOneBar:
    def test_picks_and_returns_a_candidate_on_happy_path(self, tmp_path):
        mock_client = MagicMock()
        mock_client.get_recent_bars.return_value = [SOME_BAR]

        agent = HorizonAgent(
            client=mock_client,
            cfg=HorizonConfig(),
            symbol="AAPL",
            tuning_log_path=tmp_path / "tuning.jsonl",
        )

        action = agent.run_one_bar()

        assert isinstance(action, Candidate)

    def test_alpaca_failure_does_not_crash_and_returns_hold(self, tmp_path):
        mock_client = MagicMock()
        mock_client.get_recent_bars.side_effect = AlpacaClientError("timeout")

        agent = HorizonAgent(
            client=mock_client,
            cfg=HorizonConfig(),
            symbol="AAPL",
            tuning_log_path=tmp_path / "tuning.jsonl",
        )

        action = agent.run_one_bar()

        assert action == Candidate.HOLD
        mock_client.submit_order.assert_not_called()

    def test_drawdown_breach_forces_hold_and_skips_order(self, tmp_path):
        mock_client = MagicMock()
        mock_client.get_recent_bars.return_value = [SOME_BAR]

        agent = HorizonAgent(
            client=mock_client,
            cfg=HorizonConfig(max_drawdown=Decimal("0.01")),
            symbol="AAPL",
            tuning_log_path=tmp_path / "tuning.jsonl",
        )
        # Force a breach by crashing the ledger's NAV down from starting capital.
        agent._ledger.record_fill(realized=Decimal("-5000"), unrealized=Decimal("0"))

        action = agent.run_one_bar()

        assert action == Candidate.HOLD
        mock_client.submit_order.assert_not_called()

    def test_submit_order_failure_does_not_crash_and_returns_chosen_action(
        self, tmp_path, monkeypatch
    ):
        """submit_order raising AlpacaClientError (e.g. Alpaca's paper API
        rejecting a fractional short-sell order) must not escape
        run_one_bar(): the chosen Candidate is still returned (unlike the
        fetch_market_state failure path, this does NOT retroactively
        downgrade to HOLD), and the rest of the bar's bookkeeping —
        evidence recording and the tuning-cadence check — still runs."""
        mock_client = MagicMock()
        mock_client.get_recent_bars.return_value = [SOME_BAR]
        mock_client.submit_order.side_effect = AlpacaClientError("order rejected")
        tuning_log_path = tmp_path / "tuning.jsonl"

        agent = HorizonAgent(
            client=mock_client,
            cfg=HorizonConfig(tuning_cadence_bars=1),
            symbol="AAPL",
            tuning_log_path=tuning_log_path,
        )
        # Force the chosen action to be a real order-submitting one so
        # _submit_action actually calls submit_order.
        monkeypatch.setattr(
            "evo_trader.horizon.agent.pick_best", lambda scored: Candidate.ENTER_LONG
        )

        action = agent.run_one_bar()

        assert action == Candidate.ENTER_LONG
        mock_client.submit_order.assert_called_once()
        # Evidence recording happened before the submit attempt.
        assert len(agent._rollout_evidence) == 1
        # Tuning cadence check still ran (cadence=1, cold start -> no file
        # yet, but _run_tuning_cycle was invoked without raising).
        assert not tuning_log_path.exists()

    def test_tuner_runs_every_tuning_cadence_bars(self, tmp_path):
        mock_client = MagicMock()
        mock_client.get_recent_bars.return_value = [SOME_BAR]
        tuning_log_path = tmp_path / "tuning.jsonl"

        agent = HorizonAgent(
            client=mock_client,
            cfg=HorizonConfig(tuning_cadence_bars=3),
            symbol="AAPL",
            tuning_log_path=tuning_log_path,
        )

        for _ in range(3):
            agent.run_one_bar()

        # After exactly tuning_cadence_bars calls, no tuning decision should
        # have been logged yet: MIN_EVIDENCE_WINDOW is 5, so with only 3
        # bars of evidence, JevTuner.propose() returns None (cold start),
        # _run_tuning_cycle returns early, and TuningLog.record() — the
        # only thing that ever creates the log file — is never called.
        assert not tuning_log_path.exists()

    def test_tuning_log_grows_once_evidence_window_is_met(self, tmp_path):
        """Pins that _run_tuning_cycle actually writes to the log once real,
        non-no-op evidence clears MIN_EVIDENCE_WINDOW (5).

        NOTE on a deviation from this test's original brief form: the
        brief's version of this test ran 6 bars through the *real*
        placeholder branch_generator and asserted the log existed. Against
        the current (post-Task-6-fix-round) JevTuner, that no longer holds:
        the placeholder generator always yields 2-3 branches per candidate
        (diversity_score 2 or 3, always >= 1.5) and
        RolloutRecord(price_stayed_in_branch=True) always (miss_rate 0), so
        every real cycle computes a no-op proposal
        (proposed_tau == cfg.tau and proposed_t_c == cfg.t_c) and
        JevTuner.propose() now returns None for a no-op exactly like it
        does for cold start (see tuner.py's post-fix-round no-op guard, and
        task-7-brief.md's own callout that this interface changed after the
        brief was written). Driving a real log write through the
        placeholder generator alone is therefore no longer possible without
        making this test itself a fragile, implementation-coupled hack.

        Instead, this test seeds `agent._rollout_evidence` directly with
        evidence that Task 6's own tuner tests already prove triggers a
        real, non-zero (tau, t_c) delta against a default HorizonConfig
        (mostly-missed branches + low diversity), then drives exactly one
        bar at `tuning_cadence_bars=1` so `_run_tuning_cycle` runs against
        that evidence. This still exercises the real `_run_tuning_cycle` ->
        `JevTuner.propose` -> `TuningLog.record` wiring end-to-end; it just
        doesn't rely on the placeholder generator (which was never meant to
        produce bad rollouts) to coincidentally produce bad evidence.
        """
        mock_client = MagicMock()
        mock_client.get_recent_bars.return_value = [SOME_BAR]
        tuning_log_path = tmp_path / "tuning.jsonl"

        agent = HorizonAgent(
            client=mock_client,
            cfg=HorizonConfig(tuning_cadence_bars=1),  # tune every bar
            symbol="AAPL",
            tuning_log_path=tuning_log_path,
        )

        # Seed evidence that clears MIN_EVIDENCE_WINDOW (5) with a real,
        # non-no-op delta: mostly-missed branches (miss_rate > 0.3) and low
        # diversity (avg_diversity < 1.5) against the default
        # HorizonConfig() (tau=10, t_c=Decimal("1.0")).
        agent._rollout_evidence.extend(
            RolloutRecord(
                candidate_diversity_scores=[1, 1, 1], price_stayed_in_branch=False
            )
            for _ in range(5)
        )

        agent.run_one_bar()  # cadence=1, so this triggers _run_tuning_cycle

        assert tuning_log_path.exists()
        lines = tuning_log_path.read_text().strip().splitlines()
        assert len(lines) >= 1


class TestSubmitActionSideMapping:
    """`_submit_action` only knows a correct order side for ENTER_LONG
    (buy) and ENTER_SHORT (sell). RESIZE/EXIT/HEDGE each depend on the
    direction of an existing position, and `self._position_notional` is
    never updated after a fill anywhere in this class, so there is no real
    position-tracking state to derive a correct side from for those three.
    Guessing a side for them would risk a silent wrong-side order in paper
    trading, so `_submit_action` must no-op for all three instead — this
    mirrors the project's existing fail-closed pattern (e.g.
    constraints.py's breach -> HOLD, and the tuner's live-mode gate).

    These tests call `_submit_action` directly rather than going through
    `run_one_bar()`, because which candidate `run_one_bar()` ultimately
    picks depends on the placeholder branch generator's seeded-random
    branch counts and on `build_candidates`'s position-state gating —
    neither of which is a reliable way to force a specific action
    deterministically. Testing `_submit_action` directly against each
    action name is the precise, stable way to pin this behavior.
    """

    def test_enter_long_submits_buy(self, tmp_path):
        mock_client = MagicMock()
        agent = HorizonAgent(
            client=mock_client,
            cfg=HorizonConfig(),
            symbol="AAPL",
            tuning_log_path=tmp_path / "tuning.jsonl",
        )

        agent._submit_action(Candidate.ENTER_LONG)

        mock_client.submit_order.assert_called_once_with(
            "AAPL", side="buy", notional=Decimal("100")
        )

    def test_enter_short_submits_sell(self, tmp_path):
        mock_client = MagicMock()
        agent = HorizonAgent(
            client=mock_client,
            cfg=HorizonConfig(),
            symbol="AAPL",
            tuning_log_path=tmp_path / "tuning.jsonl",
        )

        agent._submit_action(Candidate.ENTER_SHORT)

        mock_client.submit_order.assert_called_once_with(
            "AAPL", side="sell", notional=Decimal("100")
        )

    @pytest.mark.parametrize(
        "action", [Candidate.RESIZE, Candidate.EXIT, Candidate.HEDGE]
    )
    def test_ambiguous_direction_actions_never_submit_an_order(
        self, tmp_path, action
    ):
        mock_client = MagicMock()
        agent = HorizonAgent(
            client=mock_client,
            cfg=HorizonConfig(),
            symbol="AAPL",
            tuning_log_path=tmp_path / "tuning.jsonl",
        )

        agent._submit_action(action)

        mock_client.submit_order.assert_not_called()
