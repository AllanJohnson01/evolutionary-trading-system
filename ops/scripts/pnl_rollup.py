"""
Daily / weekly paper P&L and activity rollup.

Usage::

    python ops/scripts/pnl_rollup.py --daily
    python ops/scripts/pnl_rollup.py --weekly [--out report.md]

Requires ALPACA_API_KEY, ALPACA_SECRET_KEY (paper account). Read-only: never
places orders. Prints the markdown report to stdout (and to --out if given).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evo_trader.reporting.rollup import (  # noqa: E402
    RollupError,
    fetch_account_snapshot,
    read_jsonl,
    render_report,
    summarize_horizon,
    summarize_tactical,
)

logger = logging.getLogger("pnl_rollup")


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper P&L and activity rollup")
    period = parser.add_mutually_exclusive_group(required=True)
    period.add_argument("--daily", action="store_true", help="Last 24 hours.")
    period.add_argument("--weekly", action="store_true", help="Last 7 days.")
    parser.add_argument("--state-dir", default=str(_REPO_ROOT / "state"))
    parser.add_argument("--out", help="Also write the report to this file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        print("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.", file=sys.stderr)
        sys.exit(1)

    try:
        # Reuse the paper-only guard: refuses any non-paper ALPACA_BASE_URL.
        from alpaca.trading.client import TradingClient

        from evo_trader.tactical._alpaca_clients import _require_paper

        _require_paper(os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"))

        label = "daily" if args.daily else "weekly"
        now = datetime.now(timezone.utc)
        since = now - timedelta(days=1 if args.daily else 7)
        state = Path(args.state_dir)

        snap = fetch_account_snapshot(TradingClient(api_key, secret, paper=True))
        tactical = summarize_tactical(read_jsonl(state / "decisions.jsonl"), since)
        horizon = summarize_horizon(read_jsonl(state / "horizon_decisions.jsonl"), since)
        report = render_report(label, now, snap, tactical, horizon)
    except (RollupError, ValueError) as exc:
        logger.error("Rollup failed: %s", exc)
        sys.exit(1)

    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
