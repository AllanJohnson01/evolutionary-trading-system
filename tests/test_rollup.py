"""Tests for the P&L / activity rollup. Fake Alpaca client — no network."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from evo_trader.reporting.rollup import (
    AccountSnapshot,
    AgentTradeStats,
    RollupError,
    fetch_account_snapshot,
    read_jsonl,
    render_report,
    summarize_horizon,
    summarize_tactical,
    summarize_trade_log,
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
    # report has a realized P&L section
    assert "realized" in text.lower()


def test_render_report_with_no_positions_or_activity():
    snap = AccountSnapshot(
        equity=Decimal("100000"), last_equity=Decimal("100000"),
        cash=Decimal("100000"), buying_power=Decimal("200000"), positions=[],
    )
    text = render_report("weekly", NOW, snap, {}, {})
    assert "No open positions" in text
    assert "No tactical activity" in text
    assert "No Horizon ticks" in text


# ---------------------------------------------------------------------------
# Trade log summarization
# ---------------------------------------------------------------------------


def _trade(agent_id, pnl, hours_ago=1):
    return {
        "timestamp": _ts(hours_ago),
        "agent_id": agent_id,
        "strategy_id": f"strategy_{agent_id[:4]}",
        "symbol": "SPY",
        "side": "long",
        "entry_price": "100.00",
        "exit_price": str(Decimal("100.00") + Decimal(pnl)),
        "entry_time": _ts(hours_ago + 1),
        "exit_time": _ts(hours_ago),
        "size": "10",
        "realized_pnl": str(pnl),
        "exit_reason": "take_profit",
    }


def test_summarize_trade_log_sums_pnl_and_counts_wins():
    entries = [
        _trade("agent-aaa", "50.00"),
        _trade("agent-aaa", "-25.00"),
        _trade("agent-bbb", "100.00"),
        _trade("agent-aaa", "75.00", hours_ago=200),  # outside 24h window
    ]
    stats = summarize_trade_log(entries, since=NOW - timedelta(hours=24))

    assert "agent-aaa" in stats
    assert stats["agent-aaa"].trade_count == 2
    assert stats["agent-aaa"].wins == 1
    assert stats["agent-aaa"].realized_pnl == Decimal("25.00")
    assert stats["agent-bbb"].trade_count == 1
    assert stats["agent-bbb"].wins == 1
    assert stats["agent-bbb"].realized_pnl == Decimal("100.00")


def test_summarize_trade_log_win_rate():
    entries = [
        _trade("a", "10"),
        _trade("a", "20"),
        _trade("a", "-5"),
        _trade("a", "-15"),
    ]
    stats = summarize_trade_log(entries, since=NOW - timedelta(hours=24))
    assert stats["a"].win_rate == Decimal("2") / Decimal("4")


def test_summarize_trade_log_empty_returns_empty():
    assert summarize_trade_log([], since=NOW - timedelta(days=1)) == {}


def test_summarize_trade_log_missing_agent_id_skipped():
    entries = [{"timestamp": _ts(1), "realized_pnl": "50", "symbol": "SPY"}]
    assert summarize_trade_log(entries, since=NOW - timedelta(hours=24)) == {}


def test_summarize_trade_log_bad_pnl_skipped():
    entries = [{"timestamp": _ts(1), "agent_id": "a", "realized_pnl": "not_a_number"}]
    assert summarize_trade_log(entries, since=NOW - timedelta(hours=24)) == {}


def test_render_report_shows_realized_pnl_per_agent():
    snap = AccountSnapshot(
        equity=Decimal("100000"), last_equity=Decimal("100000"),
        cash=Decimal("100000"), buying_power=Decimal("200000"), positions=[],
    )
    stats = {"agent-aaa": AgentTradeStats(agent_id="agent-aaa", trade_count=3, wins=2, realized_pnl=Decimal("150.00"))}
    text = render_report("daily", NOW, snap, {}, {}, trade_stats=stats)
    assert "agent-aaa" in text
    assert "150.00" in text
    assert "3 trades" in text
    assert "66.7%" in text


def test_render_report_no_closed_trades_says_so():
    snap = AccountSnapshot(
        equity=Decimal("100000"), last_equity=Decimal("100000"),
        cash=Decimal("100000"), buying_power=Decimal("200000"), positions=[],
    )
    text = render_report("daily", NOW, snap, {}, {}, trade_stats={})
    assert "No closed trades this window" in text
