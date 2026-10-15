"""
Posture store — read/write market_posture.json and event_calendar.json.

All paths are configurable; defaults point at ``state/`` relative to the repo
root (or wherever the process runs from).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from evo_trader.macro.schemas import (
    BlackoutWindow,
    EventCalendar,
    MarketPosture,
    Regime,
)

logger = logging.getLogger(__name__)

_DEFAULT_POSTURE_PATH = Path("state/market_posture.json")
_DEFAULT_CALENDAR_PATH = Path("state/event_calendar.json")

_STALE_RISK_DIAL = Decimal("0.5")
_STALE_BLACKOUT_HOURS = 24


class PostureStore:
    """
    Thin read/write wrapper around the two posture JSON files.

    Tactical-loop entry point::

        store = PostureStore()
        posture = store.current_posture()

    ``current_posture()`` never raises on a missing or expired file — it
    degrades gracefully as specified in the macro layer design.
    """

    def __init__(
        self,
        posture_path: Path | str = _DEFAULT_POSTURE_PATH,
        calendar_path: Path | str = _DEFAULT_CALENDAR_PATH,
    ) -> None:
        self.posture_path = Path(posture_path)
        self.calendar_path = Path(calendar_path)

    # ------------------------------------------------------------------
    # Public read API (tactical loop)
    # ------------------------------------------------------------------

    def current_posture(self) -> MarketPosture:
        """
        Return the current ``MarketPosture``.

        If the file is missing, unreadable, or stale (``valid_until`` in the
        past), return a *degraded* posture:

        - ``regime = neutral``
        - ``risk_dial = 0.5``
        - A single blackout window covering 24 h from ``valid_until`` (or now
          when there is no stored posture) — fail cautious, never fail open.
        """
        raw = self._read_json(self.posture_path)
        if raw is None:
            logger.warning("market_posture.json missing — returning degraded posture")
            return self._degraded_posture(anchor=None)

        try:
            posture = MarketPosture.from_dict(raw)
        except Exception as exc:
            logger.error("Failed to parse market_posture.json: %s — degrading", exc)
            return self._degraded_posture(anchor=None)

        if self._is_stale(posture.valid_until):
            logger.warning(
                "Posture expired at %s — returning degraded posture", posture.valid_until
            )
            return self._degraded_posture(anchor=posture.valid_until)

        return posture

    def current_calendar(self) -> EventCalendar:
        """
        Return the current ``EventCalendar``.  Returns an empty calendar on
        any error — an empty calendar must never crash the tactical loop.
        """
        now_str = _now_rfc3339()
        raw = self._read_json(self.calendar_path)
        if raw is None:
            logger.warning("event_calendar.json missing — returning empty calendar")
            return EventCalendar.empty(fetched_at=now_str)

        try:
            return EventCalendar.from_dict(raw)
        except Exception as exc:
            logger.error("Failed to parse event_calendar.json: %s — returning empty", exc)
            return EventCalendar.empty(fetched_at=now_str)

    # ------------------------------------------------------------------
    # Public write API (posture writer harness)
    # ------------------------------------------------------------------

    def write_posture(self, posture: MarketPosture) -> None:
        self.posture_path.parent.mkdir(parents=True, exist_ok=True)
        self.posture_path.write_text(posture.to_json(), encoding="utf-8")
        logger.info("Wrote posture as_of=%s to %s", posture.as_of, self.posture_path)

    def write_calendar(self, calendar: EventCalendar) -> None:
        self.calendar_path.parent.mkdir(parents=True, exist_ok=True)
        self.calendar_path.write_text(calendar.to_json(), encoding="utf-8")
        logger.info("Wrote calendar fetched_at=%s to %s", calendar.fetched_at, self.calendar_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> dict | None:  # type: ignore[type-arg]
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return None

    @staticmethod
    def _is_stale(valid_until: str) -> bool:
        try:
            deadline = datetime.fromisoformat(valid_until)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            return datetime.now(tz=timezone.utc) > deadline
        except (ValueError, TypeError):
            return True  # unparseable → treat as stale

    @staticmethod
    def _degraded_posture(anchor: str | None) -> MarketPosture:
        now = _now_rfc3339()
        blackout_start = anchor if anchor else now
        blackout_end_dt = datetime.fromisoformat(blackout_start)
        if blackout_end_dt.tzinfo is None:
            blackout_end_dt = blackout_end_dt.replace(tzinfo=timezone.utc)
        from datetime import timedelta
        blackout_end = (
            blackout_end_dt + timedelta(hours=_STALE_BLACKOUT_HOURS)
        ).isoformat()

        return MarketPosture(
            as_of=now,
            regime=Regime.NEUTRAL,
            risk_dial=_STALE_RISK_DIAL,
            blackout_windows=(
                BlackoutWindow(
                    start=blackout_start,
                    end=blackout_end,
                    reason="stale_posture",
                ),
            ),
            notes="Degraded: posture file missing or expired.",
            valid_until=blackout_end,
        )


def _now_rfc3339() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
