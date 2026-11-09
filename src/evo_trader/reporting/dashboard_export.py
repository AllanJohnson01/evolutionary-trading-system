"""
Build the JSON payload consumed by the trading-agents dashboard.

Reuses the ``rollup`` summarisers so the dashboard and ``pnl_rollup.py`` can
never disagree. Equity history is the one thing the rollup does not keep, so
this module owns a small append-only ``state/equity_history.jsonl``.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from evo_trader.reporting.rollup import (
    AccountSnapshot,
    _parse_ts,
    _strategy_family,
    read_jsonl,
    summarize_horizon,
    summarize_tactical,
    summarize_trade_log,
)
from evo_trader.tactical.population import _STRATEGY_PREFIX

_FAMILY_TO_NAME = {family: name for name, family in _STRATEGY_PREFIX.items()}


def _id_suffix(agent_id: str) -> str:
    """First 8 hex chars of the agent UUID — the suffix ``_stable_strategy_id`` appends."""
    return agent_id.replace("-", "")[:8]


def _suffix_to_family(decisions: list[dict[str, Any]]) -> dict[str, str]:
    """Map ``<8 hex>`` -> strategy family using strategy_ids seen in decisions."""
    out: dict[str, str] = {}
    for e in decisions:
        sid = e.get("strategy_id")
        if not sid:
            continue
        family = _strategy_family(sid)
        if family != sid:
            out[sid[len(family) + 1:]] = family
    return out


def append_equity_snapshot(path: Path, now: datetime, equity: Decimal) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": now.isoformat(), "equity": str(equity)}) + "\n")


def load_equity_curve(path: Path) -> list[dict[str, Any]]:
    """Daily NAV points (last snapshot of each UTC day), oldest first."""
    by_day: dict[str, float] = {}
    for e in sorted(read_jsonl(path), key=lambda e: str(e.get("timestamp"))):
        ts = _parse_ts(e.get("timestamp"))
        try:
            nav = float(Decimal(str(e.get("equity"))))
        except ArithmeticError:
            continue
        if ts is not None:
            by_day[ts.date().isoformat()] = nav
    return [{"date": d, "nav": nav} for d, nav in sorted(by_day.items())]


def build_dashboard_data(
    now: datetime,
    days: int,
    decisions: list[dict[str, Any]],
    trade_log: list[dict[str, Any]],
    horizon: list[dict[str, Any]],
    snapshot: AccountSnapshot | None,
    equity_curve: list[dict[str, Any]],
) -> dict[str, Any]:
    since = now - timedelta(days=days)
    tactical = summarize_tactical(decisions, since)
    trades = summarize_trade_log(trade_log, since)
    suffixes = _suffix_to_family(decisions)

    rows: dict[str, dict[str, Any]] = {}
    for family, act in tactical.items():
        name = _FAMILY_TO_NAME.get(family, family)
        rows[name] = {
            "agent_id": name, "type": family, "trade_count": 0, "wins": 0, "realized_pnl": 0.0,
            "cycles": act.cycles, "orders": act.orders,
            "entries_denied": act.entries_denied, "exits": act.exits,
        }
    for agent_id, stats in trades.items():
        known_family = suffixes.get(_id_suffix(agent_id))
        name = _FAMILY_TO_NAME.get(known_family, known_family) if known_family else agent_id
        row = rows.setdefault(name, {
            "agent_id": name, "type": known_family or "unknown", "trade_count": 0, "wins": 0,
            "realized_pnl": 0.0, "cycles": None, "orders": None, "entries_denied": None, "exits": None,
        })
        row["trade_count"] += stats.trade_count
        row["wins"] += stats.wins
        row["realized_pnl"] += float(stats.realized_pnl)

    account = None
    if snapshot is not None:
        account = {
            "equity": float(snapshot.equity),
            "day_change": float(snapshot.day_change),
            "cash": float(snapshot.cash),
            "buying_power": float(snapshot.buying_power),
            "unrealized_pl": float(sum((p.unrealized_pl for p in snapshot.positions), Decimal("0"))),
            "open_positions": len(snapshot.positions),
        }

    # Logs hold both "hold" and the legacy "Candidate.HOLD" form; fold case so buckets don't split.
    horizon_counts: dict[str, int] = {}
    for key, count in summarize_horizon(horizon, since).items():
        horizon_counts[key.lower()] = horizon_counts.get(key.lower(), 0) + count

    return {
        "as_of": now.isoformat(),
        "period_days": days,
        "starting_equity": equity_curve[0]["nav"] if equity_curve else None,
        "account": account,
        "equity_curve": equity_curve,
        "agents": sorted(rows.values(), key=lambda r: r["agent_id"]),
        "horizon_decisions": horizon_counts,
    }
