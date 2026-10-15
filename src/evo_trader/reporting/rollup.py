"""
Daily / weekly rollup of paper-account P&L and system activity.

Sources, and what each can honestly say:
  * Alpaca paper account  -> equity, cash, open positions, *unrealized* P&L.
  * state/decisions.jsonl -> tactical cycle activity per strategy.
  * state/horizon_decisions.jsonl -> Horizon tick decisions.

Not tracked yet (no data source exists): realized P&L / win rate per strategy
and Horizon NAV (its ledger is in-memory only). The report says so explicitly
rather than implying full strategy performance.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


class RollupError(Exception):
    """Raised for any failure fetching data for the rollup."""


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    side: str
    qty: Decimal
    market_value: Decimal
    unrealized_pl: Decimal
    unrealized_plpc: Decimal


@dataclass(frozen=True)
class AccountSnapshot:
    equity: Decimal
    last_equity: Decimal
    cash: Decimal
    buying_power: Decimal
    positions: list[PositionSnapshot] = field(default_factory=list)

    @property
    def day_change(self) -> Decimal:
        """Equity change since the previous close (Alpaca ``last_equity``)."""
        return self.equity - self.last_equity


@dataclass
class StrategyActivity:
    cycles: int = 0
    orders: int = 0
    entries_denied: int = 0
    exits: int = 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank/corrupt lines. Missing file -> []."""
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _parse_ts(value: Any) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return ts if ts.tzinfo else None


_RANDOM_SUFFIX = re.compile(r"_[0-9a-f]{8}$")


def _strategy_family(strategy_id: str) -> str:
    """Strip the per-process random suffix (``grid_908bcb06`` -> ``grid``)."""
    return _RANDOM_SUFFIX.sub("", strategy_id)


def summarize_tactical(
    entries: list[dict[str, Any]], since: datetime
) -> dict[str, StrategyActivity]:
    """Count tactical activity per strategy family for entries at or after ``since``."""
    out: dict[str, StrategyActivity] = {}
    for e in entries:
        ts = _parse_ts(e.get("timestamp"))
        sid = e.get("strategy_id")
        if ts is None or ts < since or not sid:
            continue
        act = out.setdefault(_strategy_family(sid), StrategyActivity())
        act.cycles += 1
        if e.get("order_id"):
            act.orders += 1
        if e.get("cycle_type") == "entry" and not e.get("order_id"):
            act.entries_denied += 1
        if e.get("cycle_type") == "exit":
            act.exits += 1
    return out


def summarize_horizon(entries: list[dict[str, Any]], since: datetime) -> dict[str, int]:
    """Count Horizon decisions by candidate. Accepts ``HOLD`` and ``Candidate.HOLD``."""
    counts: Counter[str] = Counter()
    for e in entries:
        ts = _parse_ts(e.get("timestamp"))
        cand = e.get("candidate")
        if ts is None or ts < since or not cand:
            continue
        counts[str(cand).removeprefix("Candidate.")] += 1
    return dict(counts)


def fetch_account_snapshot(trading_client: Any) -> AccountSnapshot:
    """Read account + positions from an alpaca-py ``TradingClient``."""
    try:
        acct = trading_client.get_account()
        positions = [
            PositionSnapshot(
                symbol=p.symbol,
                side=p.side.value,
                qty=Decimal(str(p.qty)),
                market_value=Decimal(str(p.market_value)),
                unrealized_pl=Decimal(str(p.unrealized_pl)),
                unrealized_plpc=Decimal(str(p.unrealized_plpc)),
            )
            for p in trading_client.get_all_positions()
        ]
        return AccountSnapshot(
            equity=Decimal(str(acct.equity)),
            last_equity=Decimal(str(acct.last_equity)),
            cash=Decimal(str(acct.cash)),
            buying_power=Decimal(str(acct.buying_power)),
            positions=positions,
        )
    except Exception as exc:
        raise RollupError(f"fetching account snapshot failed: {exc}") from exc


def _money(d: Decimal) -> str:
    sign = "-" if d < 0 else ""
    return f"{sign}${abs(d):,.2f}"


def render_report(
    period: str,
    now: datetime,
    snap: AccountSnapshot,
    tactical: dict[str, StrategyActivity],
    horizon: dict[str, int],
) -> str:
    """Render a human-readable markdown report."""
    lines = [
        f"# {period.capitalize()} rollup — {now:%Y-%m-%d %H:%M} UTC (paper account)",
        "",
        "## Account (from Alpaca)",
        f"- Equity: {_money(snap.equity)} (change since previous close: {_money(snap.day_change)})",
        f"- Cash: {_money(snap.cash)} · Buying power: {_money(snap.buying_power)}",
        "",
        "## Open positions (unrealized P&L, from Alpaca)",
    ]
    if snap.positions:
        for p in snap.positions:
            lines.append(
                f"- {p.symbol} {p.side} {p.qty} · value {_money(p.market_value)} · "
                f"unrealized {_money(p.unrealized_pl)} ({p.unrealized_plpc * 100:.2f}%)"
            )
        total = sum((p.unrealized_pl for p in snap.positions), Decimal("0"))
        lines.append(f"- **Total unrealized: {_money(total)}**")
    else:
        lines.append("- No open positions")

    lines += ["", "## Tactical activity"]
    if tactical:
        for sid, a in sorted(tactical.items()):
            lines.append(
                f"- {sid}: {a.cycles} cycles · {a.orders} orders · "
                f"{a.entries_denied} entries denied · {a.exits} exits"
            )
    else:
        lines.append("- No tactical activity in this period")

    lines += ["", "## Horizon ticks"]
    if horizon:
        lines.append("- " + " · ".join(f"{k}: {v}" for k, v in sorted(horizon.items())))
    else:
        lines.append("- No Horizon ticks in this period")

    lines += [
        "",
        "## Not tracked yet",
        "- Realized P&L and win rate per strategy: not tracked (no trade log exists).",
        "- Horizon NAV / drawdown: not tracked (ledger is in-memory only).",
        "- Tactical strategy exits cannot fire yet (no per-strategy position attribution).",
    ]
    return "\n".join(lines) + "\n"
