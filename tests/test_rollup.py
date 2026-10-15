"""Tests for the P&L / activity rollup. Fake Alpaca client — no network."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from evo_trader.reporting.rollup import (
    AccountSnapshot,
    RollupError,
    fetch_account_snapshot,
    read_jsonl,
    render_report,
    summarize_horizon,
    summarize_tactical,
)

NOW = datetime(2026, 10, 15, 20, 0, tzinfo=timezone.utc)


def _ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _tactical(sid, cycle="skip", reason="no_signal", order_id=None, hours_ago=1, direction="flat"):
    return {
        "timestamp": _ts(hours_ago),
        "strategy_id": sid,
        "symbol": "SPY",
        "cycle_type": cycle,
        "direction": direction,
        "approval_reason": reason,
        "order_id": order_id,
    }


# ---------------------------------------------------------------------------
# read_jsonl
# ---------------------------------------------------------------------------


def test_read_jsonl_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_read_jsonl_skips_blank_and_corrupt_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n\nnot json\n{"b": 2}\n')
    assert read_jsonl(p) == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------------------
# tactical summary
# ---------------------------------------------------------------------------


def test_summarize_tactical_counts_per_strategy_within_window():
    entries = [
        _tactical("grid_1"),
        _tactical("grid_1", cycle="entry", reason="approved", order_id="o1", direction="long"),
        _tactical("grid_1", hours_ago=100),  # outside 24h window
        _tactical("breakout_1", cycle="entry", reason="drawdown", order_id=None, direction="long"),
        _tactical("breakout_1", cycle="exit", reason="stop", order_id=None),
    ]
    s = summarize_tactical(entries, since=NOW - timedelta(hours=24))

    assert s["grid_1"].cycles == 2
    assert s["grid_1"].orders == 1
    assert s["breakout_1"].entries_denied == 1
    assert s["breakout_1"].exits == 1
    assert s["breakout_1"].orders == 0


def test_summarize_tactical_groups_random_suffixed_ids_by_family():
    # Strategy ids carry a per-process random suffix; one family must aggregate.
    entries = [
        _tactical("mean_reversion_46524405"),
        _tactical("mean_reversion_4e06e454"),
        _tactical("intraday_momentum_e5760bd8"),
    ]
    s = summarize_tactical(entries, since=NOW - timedelta(days=1))
    assert s["mean_reversion"].cycles == 2
    assert s["intraday_momentum"].cycles == 1


def test_summarize_tactical_ignores_entries_without_valid_timestamp():
    bad = _tactical("grid_1")
    bad["timestamp"] = "garbage"
    assert summarize_tactical([bad, {"strategy_id": "x"}], since=NOW - timedelta(days=1)) == {}


# ---------------------------------------------------------------------------
# horizon summary — handles both serializations of the candidate
# ---------------------------------------------------------------------------


def test_summarize_horizon_normalizes_candidate_prefix():
    entries = [
        {"timestamp": _ts(1), "candidate": "HOLD"},
        {"timestamp": _ts(2), "candidate": "Candidate.HOLD"},
        {"timestamp": _ts(3), "candidate": "ENTER_SHORT"},
        {"timestamp": _ts(500), "candidate": "HOLD"},  # outside window
    ]
    counts = summarize_horizon(entries, since=NOW - timedelta(hours=24))
    assert counts == {"HOLD": 2, "ENTER_SHORT": 1}


# ---------------------------------------------------------------------------
# account snapshot
# ---------------------------------------------------------------------------


class _FakeTrading:
    def __init__(self, error=None):
        self._error = error

    def get_account(self):
        if self._error:
            raise self._error
        return SimpleNamespace(
            equity="100500.25", last_equity="100000", cash="90000.10", buying_power="180000"
        )

    def get_all_positions(self):
        return [
            SimpleNamespace(
                symbol="SPY", side=SimpleNamespace(value="long"), qty="10",
                market_value="5000.5", unrealized_pl="120.25", unrealized_plpc="0.0246",
            )
        ]


def test_fetch_account_snapshot_parses_decimals():
    snap = fetch_account_snapshot(_FakeTrading())
    assert snap.equity == Decimal("100500.25")
    assert snap.day_change == Decimal("500.25")
    assert snap.positions[0].symbol == "SPY"
    assert snap.positions[0].unrealized_pl == Decimal("120.25")


def test_fetch_account_snapshot_wraps_errors():
    with pytest.raises(RollupError, match="boom"):
        fetch_account_snapshot(_FakeTrading(error=RuntimeError("boom")))


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def test_render_report_labels_pnl_as_unrealized_and_lists_gaps():
    snap = fetch_account_snapshot(_FakeTrading())
    tactical = summarize_tactical([_tactical("grid_1")], since=NOW - timedelta(days=1))
    text = render_report("daily", NOW, snap, tactical, {"HOLD": 3})

    assert "Daily" in text
    assert "unrealized" in text.lower()
    assert "SPY" in text and "120.25" in text
    assert "grid_1" in text
    assert "HOLD" in text
    # honest about what is not tracked yet
    assert "realized" in text.lower() and "not tracked" in text.lower()


def test_render_report_with_no_positions_or_activity():
    snap = AccountSnapshot(
        equity=Decimal("100000"), last_equity=Decimal("100000"),
        cash=Decimal("100000"), buying_power=Decimal("200000"), positions=[],
    )
    text = render_report("weekly", NOW, snap, {}, {})
    assert "No open positions" in text
    assert "No tactical activity" in text
    assert "No Horizon ticks" in text
