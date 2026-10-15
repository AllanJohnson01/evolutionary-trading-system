"""
Horizon decision-tick runner.

Entry point::

    python -m evo_trader.horizon.runner --once

Runs one full ``HorizonAgent.run_one_bar()`` cycle, appends the result as a
single JSON line to ``state/horizon_decisions.jsonl`` (relative to the working
directory), and exits.

Exit codes:
    0 — cycle completed (even if the decision was HOLD).
    1 — misconfiguration or unhandled exception; message written to stderr.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from evo_trader.horizon.agent import HorizonAgent
from evo_trader.horizon.alpaca_client import AlpacaPaperClient
from evo_trader.horizon.config import HorizonConfig
from evo_trader.horizon.market_state import Candidate

logger = logging.getLogger(__name__)

# Default symbol — override via env var HORIZON_SYMBOL if needed.
_DEFAULT_SYMBOL = "SPY"
_DEFAULT_LOG_PATH = Path("state") / "horizon_decisions.jsonl"
_DEFAULT_TUNING_LOG_PATH = Path("state") / "horizon_tuning.jsonl"


def _build_log_entry(candidate: Candidate) -> dict:
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "candidate": candidate.value,
    }


def _run_once(log_path: Path, tuning_log_path: Path) -> None:
    """Instantiate HorizonAgent, call run_one_bar(), and write the result."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    symbol = os.environ.get("HORIZON_SYMBOL", _DEFAULT_SYMBOL)

    client = AlpacaPaperClient(api_key=api_key, api_secret=secret_key)
    cfg = HorizonConfig()

    agent = HorizonAgent(
        client=client,
        cfg=cfg,
        symbol=symbol,
        tuning_log_path=tuning_log_path,
    )

    logger.info("Starting Horizon decision tick for symbol=%s", symbol)
    candidate: Candidate = agent.run_one_bar()
    logger.info("Horizon decision: %s", candidate.value)

    entry = _build_log_entry(candidate)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    logger.info("Appended decision to %s", log_path)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolutionary trading system — Horizon decision-tick runner"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single Horizon decision tick and exit.",
    )
    parser.add_argument(
        "--log-path",
        default=str(_DEFAULT_LOG_PATH),
        help="Path to the JSONL decisions log (default: state/horizon_decisions.jsonl).",
    )
    parser.add_argument(
        "--tuning-log-path",
        default=str(_DEFAULT_TUNING_LOG_PATH),
        help="Path to the JSONL tuning log (default: state/horizon_tuning.jsonl).",
    )
    args = parser.parse_args()

    if not args.once:
        print(
            "No action flag supplied. Use --once to run a single decision tick.",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        _run_once(
            log_path=Path(args.log_path),
            tuning_log_path=Path(args.tuning_log_path),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Horizon runner failed: {exc}", file=sys.stderr)
        logger.exception("Unhandled exception in Horizon runner")
        sys.exit(1)


if __name__ == "__main__":
    _main()
