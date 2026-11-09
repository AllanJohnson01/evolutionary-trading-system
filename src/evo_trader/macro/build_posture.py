"""
Posture writer harness — CLI entry point for the Market-Analyst agent.

Usage::

    python -m evo_trader.macro.build_posture --posture-json path/to/posture.json
    python -m evo_trader.macro.build_posture --print-context

The harness does NOT call an LLM.  The Market-Analyst agent:

1. Runs ``--print-context`` to receive the assembled inputs (calendar + recent
   price action summary stub) it needs to write a posture.
2. Constructs a posture JSON matching the ``MarketPosture`` schema.
3. Passes the JSON file path via ``--posture-json``.  The harness validates
   the schema and enforces the autonomy bound before writing to disk.

Autonomy bound
--------------
``risk_dial`` may never exceed 1.0 unless the environment variable
``ALLOW_AGGRESSIVE=true`` is set.  With the flag, the ceiling rises to 1.25.
Values above 1.25 are always rejected.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal

from evo_trader.macro.calendar_fetcher import fetch_calendar
from evo_trader.macro.schemas import MarketPosture
from evo_trader.macro.store import PostureStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_NORMAL_DIAL_CAP = Decimal("1.0")
_AGGRESSIVE_DIAL_CAP = Decimal("1.25")


def _dial_cap() -> Decimal:
    return _AGGRESSIVE_DIAL_CAP if os.getenv("ALLOW_AGGRESSIVE", "").lower() == "true" else _NORMAL_DIAL_CAP


def build_context_payload(symbols: list[str] | None = None) -> dict:
    """Assemble the inputs the LLM analyst needs to write a posture."""
    calendar = fetch_calendar(symbols=symbols, use_fixtures=False)
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "event_calendar": calendar.to_dict(),
        "price_action_summary": (
            "TODO: inject recent OHLCV summary for configured symbols here "
            "(to be wired up in Stage 3 operational integration)."
        ),
        "instructions": (
            "Based on the event calendar and price action summary above, produce "
            "a market_posture.json with keys: as_of, regime (risk_on|neutral|risk_off), "
            "risk_dial (Decimal 0.0–1.0), blackout_windows (list of {start, end, reason}), "
            "notes (one-paragraph rationale), valid_until (RFC3339, typically 24h from as_of). "
            "risk_dial 0 = flat, 1.0 = full allocation.  You may not exceed 1.0 without "
            "the ALLOW_AGGRESSIVE environment flag set by Allan."
        ),
    }


def validate_and_write(posture_json_path: str, store: PostureStore) -> int:
    """
    Load, validate, and write a posture JSON file.

    Returns 0 on success, non-zero on validation failure.
    """
    try:
        raw = json.loads(open(posture_json_path, encoding="utf-8").read())  # noqa: WPS515
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Cannot read posture JSON: %s", exc)
        return 1

    try:
        posture = MarketPosture.from_dict(raw)
    except Exception as exc:
        logger.error("Schema validation failed: %s", exc)
        return 1

    cap = _dial_cap()
    if posture.risk_dial > cap:
        logger.error(
            "risk_dial %s exceeds autonomy bound %s — set ALLOW_AGGRESSIVE=true "
            "to raise the cap to 1.25 (requires Allan's sign-off).",
            posture.risk_dial,
            cap,
        )
        return 1

    if posture.risk_dial < Decimal("0"):
        logger.error("risk_dial must be >= 0, got %s", posture.risk_dial)
        return 1

    store.write_posture(posture)
    logger.info("Posture written: regime=%s risk_dial=%s valid_until=%s",
                posture.regime.value, posture.risk_dial, posture.valid_until)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Market posture writer harness for the Market-Analyst agent."
    )
    parser.add_argument(
        "--posture-json",
        metavar="PATH",
        help="Path to posture JSON produced by the LLM analyst; validate and write to store.",
    )
    parser.add_argument(
        "--print-context",
        action="store_true",
        help="Print assembled analyst context (calendar + price action) to stdout and exit.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Symbols to include in context assembly (e.g. AAPL MSFT BTC-USD).",
    )
    parser.add_argument(
        "--posture-path",
        default="state/market_posture.json",
        help="Destination path for the posture file.",
    )
    parser.add_argument(
        "--calendar-path",
        default="state/event_calendar.json",
        help="Destination path for the event calendar file.",
    )

    args = parser.parse_args(argv)
    store = PostureStore(
        posture_path=args.posture_path,
        calendar_path=args.calendar_path,
    )

    if args.print_context:
        payload = build_context_payload(symbols=args.symbols)
        print(json.dumps(payload, indent=2))
        return 0

    if args.posture_json:
        return validate_and_write(args.posture_json, store)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
