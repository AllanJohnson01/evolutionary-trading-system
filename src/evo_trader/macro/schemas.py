"""Frozen dataclasses for macro posture and event calendar with JSON (de)serialization."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any


class Regime(str, Enum):
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


class ImpactTier(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass(frozen=True)
class BlackoutWindow:
    """A time range during which new entries are forbidden."""
    start: str   # RFC3339
    end: str     # RFC3339
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BlackoutWindow":
        return cls(start=str(d["start"]), end=str(d["end"]), reason=str(d["reason"]))


@dataclass(frozen=True)
class MarketPosture:
    """
    Daily macro assessment emitted by the Market-Analyst agent.

    ``risk_dial`` is in [0.0, 1.0] under normal operation.  The only path to
    values above 1.0 is an explicit ``ALLOW_AGGRESSIVE=true`` environment
    override (enforced by the posture writer harness, not here).  Stale
    posture returned by ``PostureStore.current_posture()`` always has
    ``risk_dial=0.5``.
    """
    as_of: str
    regime: Regime
    risk_dial: Decimal
    blackout_windows: tuple[BlackoutWindow, ...]
    notes: str
    valid_until: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of,
            "regime": self.regime.value,
            "risk_dial": str(self.risk_dial),
            "blackout_windows": [w.to_dict() for w in self.blackout_windows],
            "notes": self.notes,
            "valid_until": self.valid_until,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MarketPosture":
        try:
            risk_dial = Decimal(str(d["risk_dial"]))
        except (InvalidOperation, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid risk_dial: {d.get('risk_dial')!r}") from exc
        return cls(
            as_of=str(d["as_of"]),
            regime=Regime(d["regime"]),
            risk_dial=risk_dial,
            blackout_windows=tuple(
                BlackoutWindow.from_dict(w) for w in d.get("blackout_windows", [])
            ),
            notes=str(d.get("notes", "")),
            valid_until=str(d["valid_until"]),
        )

    @classmethod
    def from_json(cls, raw: str) -> "MarketPosture":
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class CalendarEvent:
    """A single scheduled macro or earnings event."""
    timestamp: str          # RFC3339
    symbol_scope: tuple[str, ...]  # empty = all symbols
    impact_tier: ImpactTier
    description: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "symbol_scope": list(self.symbol_scope),
            "impact_tier": self.impact_tier.value,
            "description": self.description,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalendarEvent":
        return cls(
            timestamp=str(d["timestamp"]),
            symbol_scope=tuple(str(s) for s in d.get("symbol_scope", [])),
            impact_tier=ImpactTier(int(d["impact_tier"])),
            description=str(d.get("description", "")),
            source=str(d.get("source", "")),
        )


@dataclass(frozen=True)
class EventCalendar:
    """Collection of upcoming macro and earnings events."""
    events: tuple[CalendarEvent, ...]
    fetched_at: str   # RFC3339
    fetch_errors: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fetched_at": self.fetched_at,
            "events": [e.to_dict() for e in self.events],
            "fetch_errors": list(self.fetch_errors),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EventCalendar":
        return cls(
            fetched_at=str(d["fetched_at"]),
            events=tuple(CalendarEvent.from_dict(e) for e in d.get("events", [])),
            fetch_errors=tuple(str(e) for e in d.get("fetch_errors", [])),
        )

    @classmethod
    def from_json(cls, raw: str) -> "EventCalendar":
        return cls.from_dict(json.loads(raw))

    @classmethod
    def empty(cls, fetched_at: str) -> "EventCalendar":
        return cls(events=(), fetched_at=fetched_at)
