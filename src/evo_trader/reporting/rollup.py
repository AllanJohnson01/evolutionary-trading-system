"""
Daily / weekly rollup of paper-account P&L and system activity.

Sources, and what each can honestly say:
  * Alpaca paper account          -> equity, cash, open positions, *unrealized* P&L.
  * state/decisions.jsonl         -> tactical cycle activity per strategy (agent_id included).
  * state/trade_log.jsonl         -> closed-trade outcomes: realized P&L, win/loss per agent.
  * state/horizon_decisions.jsonl -> Horizon tick decisions.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
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


@dataclass
class AgentTradeStats:
    """Realized P&L and win-rate stats for a single agent from ``trade_log.jsonl``."""
    agent_id: str
    trade_count: int = 0
    wins: int = 0
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))

    @property
    def win_rate(self) -> Decimal | None:
        if self.trade_count == 0:
            return None
        return Decimal(self.wins) / Decimal(self.trade_count)


def summarize_trade_log(
    entries: list[dict[str, Any]], since: datetime
) -> dict[str, AgentTradeStats]:
    """Aggregate realized P&L and win rate per agent from ``trade_log.jsonl`` entries.

    Returns a dict keyed by ``agent_id``. Entries outside the window or missing
    ``agent_id`` / ``realized_pnl`` are skipped gracefully.
    """
    out: dict[str, AgentTradeStats] = {}
    for e in entries:
        ts = _parse_ts(e.get("timestamp"))
        agent_id = e.get("agent_id")
        pnl_str = e.get("realized_pnl")
        if ts is None or ts < since or not agent_id or pnl_str is None:
            continue
        try:
            pnl = Decimal(str(pnl_str))
        except InvalidOperation:
            continue
        stats = out.setdefault(agent_id, AgentTradeStats(agent_id=agent_id))
        stats.trade_count += 1
        stats.realized_pnl += pnl
        if pnl > Decimal("0"):
            stats.wins += 1
    return out


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
    """Count tactical activity per strategy family for entries at or after ``since``.

    When ``agent_id`` is present on an entry it is carried onto the
    ``StrategyActivity`` record so the report can associate activity with a
    stable agent identity.
    """
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
    trade_stats: dict[str, AgentTradeStats] | None = None,
) -> str:
    """Render a human-readable markdown report.

    Args:
        period: "daily" or "weekly".
        now: Report timestamp.
        snap: Alpaca account snapshot.
        tactical: Per-strategy-family activity from decisions.jsonl.
        horizon: Horizon tick counts.
        trade_stats: Optional per-agent realized P&L stats from trade_log.jsonl.
    """
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

    # Realized P&L per agent (from trade_log.jsonl)
    lines += ["", "## Realized P&L per agent (closed trades)"]
    if trade_stats:
        for agent_id, stats in sorted(trade_stats.items()):
            win_rate_str = (
                f"{stats.win_rate * 100:.1f}%" if stats.win_rate is not None else "n/a"
            )
            lines.append(
                f"- agent `{agent_id}`: {stats.trade_count} trades · "
                f"realized {_money(stats.realized_pnl)} · win rate {win_rate_str}"
            )
    else:
        lines.append("- No closed trades this window")

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
        "- Horizon NAV / drawdown: not tracked (ledger is in-memory only).",
    ]
    return "\n".join(lines) + "\n"
