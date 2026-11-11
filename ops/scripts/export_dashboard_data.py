"""
Export the trading-agents dashboard data as JSON.

Usage::

    python ops/scripts/export_dashboard_data.py [--days 7] [--out PATH] [--no-alpaca]

Writes dashboard_data.json plus a dashboard_data.js sidecar for docs/dashboard/index.html.
Reads state/*.jsonl and, when ALPACA_API_KEY / ALPACA_SECRET_KEY are set, the
paper account (and appends an equity snapshot to state/equity_history.jsonl so
the equity curve grows over time). Read-only: never places orders.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evo_trader.reporting.dashboard_export import (  # noqa: E402
    append_equity_snapshot,
    build_dashboard_data,
    load_equity_curve,
)
from evo_trader.reporting.rollup import RollupError, fetch_account_snapshot, read_jsonl  # noqa: E402

logger = logging.getLogger("export_dashboard_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dashboard data JSON")
    parser.add_argument("--days", type=int, default=7, help="Activity/P&L window (default 7).")
    parser.add_argument("--state-dir", default=str(_REPO_ROOT / "state"))
    parser.add_argument("--out", default=str(_REPO_ROOT / "docs" / "dashboard" / "dashboard_data.json"))
    parser.add_argument("--no-alpaca", action="store_true", help="Skip the live account snapshot.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    state = Path(args.state_dir)
    equity_path = state / "equity_history.jsonl"
    now = datetime.now(timezone.utc)

    snap = None
    api_key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if args.no_alpaca or not (api_key and secret):
        logger.warning("Skipping Alpaca snapshot (account section will be empty).")
    else:
        try:
            from alpaca.trading.client import TradingClient

            from evo_trader.tactical._alpaca_clients import _require_paper

            _require_paper(os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))
            snap = fetch_account_snapshot(TradingClient(api_key, secret, paper=True))
            append_equity_snapshot(equity_path, now, snap.equity)
        except (RollupError, ValueError) as exc:
            logger.error("Account snapshot failed: %s", exc)
            sys.exit(1)

    data = build_dashboard_data(
        now,
        args.days,
        read_jsonl(state / "decisions.jsonl"),
        read_jsonl(state / "trade_log.jsonl"),
        read_jsonl(state / "horizon_decisions.jsonl"),
        snap,
        load_equity_curve(equity_path),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2)
    out.write_text(payload + "\n", encoding="utf-8")
    # Browsers block fetch() on file:// pages; a plain <script> sidecar lets index.html open by double-click.
    out.with_suffix(".js").write_text(f"window.DASHBOARD_DATA = {payload};\n", encoding="utf-8")
    logger.info("Wrote %s and %s (%d agents)", out, out.with_suffix(".js").name, len(data["agents"]))


if __name__ == "__main__":
    main()
