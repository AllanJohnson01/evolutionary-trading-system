"""Tests for the dashboard data export. Pure functions — no network, no Alpaca."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from evo_trader.reporting.dashboard_export import (
    append_equity_snapshot,
    build_dashboard_data,
    load_equity_curve,
)
from evo_trader.reporting.rollup import AccountSnapshot

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
MOMENTUM_ID = "11669cf7-aaaa-bbbb-cccc-000000000000"
GRID_ID = "908bcb06-aaaa-bbbb-cccc-000000000000"


def _ts(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _decision(sid, cycle="skip", order_id=None, hours_ago=1):
    return {"timestamp": _ts(hours_ago), "strategy_id": sid, "cycle_type": cycle, "order_id": order_id}


def _trade(agent_id, pnl, hours_ago=1):
    return {"timestamp": _ts(hours_ago), "agent_id": agent_id, "realized_pnl": pnl}


def _snap(equity="100500"):
    return AccountSnapshot(
        equity=Decimal(equity), last_equity=Decimal("100000"),
        cash=Decimal("50000"), buying_power=Decimal("90000"),
    )


def test_agents_use_display_names_and_join_trade_log_by_id_suffix():
    decisions = [
        _decision("intraday_momentum_11669cf7", cycle="entry", order_id="o1"),
        _decision("intraday_momentum_11669cf7", cycle="exit"),
        _decision("grid_908bcb06"),
    ]
    trades = [_trade(MOMENTUM_ID, "100.50"), _trade(MOMENTUM_ID, "-20"), _trade(GRID_ID, "5")]
    data = build_dashboard_data(NOW, 7, decisions, trades, [], None, [])
    by_name = {a["agent_id"]: a for a in data["agents"]}
    mom = by_name["Momentum-Hunter"]
    assert (mom["trade_count"], mom["wins"], mom["realized_pnl"]) == (2, 1, 80.5)
    assert (mom["cycles"], mom["orders"], mom["exits"]) == (2, 1, 1)
    assert by_name["Grid-Trader"]["trade_count"] == 1


def test_agent_with_activity_but_no_trades_has_zero_pnl():
    data = build_dashboard_data(NOW, 7, [_decision("breakout_63ee934a")], [], [], None, [])
    (agent,) = data["agents"]
    assert agent["agent_id"] == "Breakout"
    assert (agent["trade_count"], agent["wins"], agent["realized_pnl"]) == (0, 0, 0.0)


def test_unmapped_trade_log_agent_keeps_raw_id():
    data = build_dashboard_data(NOW, 7, [], [_trade("deadbeef-0000", "3")], [], None, [])
    (agent,) = data["agents"]
    assert agent["agent_id"] == "deadbeef-0000"
    assert agent["cycles"] is None


def test_entries_outside_window_excluded():
    decisions = [_decision("grid_908bcb06", hours_ago=24 * 10)]
    data = build_dashboard_data(NOW, 7, decisions, [_trade(GRID_ID, "9", hours_ago=24 * 10)], [], None, [])
    assert data["agents"] == []


def test_horizon_counts_normalised_and_output_is_json_serialisable():
    horizon = [
        {"timestamp": _ts(1), "candidate": "hold"},
        {"timestamp": _ts(2), "candidate": "Candidate.HOLD"},
        {"timestamp": _ts(3), "candidate": "enter_long"},
    ]
    data = build_dashboard_data(NOW, 7, [], [], horizon, _snap(), [])
    assert data["horizon_decisions"] == {"hold": 2, "enter_long": 1}
    assert data["account"]["equity"] == 100500.0
    json.dumps(data)


def test_account_is_none_without_snapshot():
    data = build_dashboard_data(NOW, 7, [], [], [], None, [])
    assert data["account"] is None
    assert data["as_of"] == NOW.isoformat()


def test_equity_history_roundtrip_keeps_last_value_per_day(tmp_path):
    path = tmp_path / "equity_history.jsonl"
    append_equity_snapshot(path, NOW - timedelta(days=1), Decimal("100000"))
    append_equity_snapshot(path, NOW, Decimal("100100"))
    append_equity_snapshot(path, NOW + timedelta(hours=2), Decimal("100250.5"))
    curve = load_equity_curve(path)
    assert curve == [
        {"date": "2026-10-26", "nav": 100000.0},
        {"date": "2026-10-27", "nav": 100250.5},
    ]


def test_missing_equity_history_gives_empty_curve(tmp_path):
    assert load_equity_curve(tmp_path / "nope.jsonl") == []
