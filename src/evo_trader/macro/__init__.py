"""
Macro posture layer — daily market regime assessment consumed by the tactical loop.

Tactical-loop-facing read API
------------------------------
    from evo_trader.macro.store import PostureStore

    store = PostureStore()            # default paths: state/market_posture.json
                                      #                 state/event_calendar.json
    posture = store.current_posture() # MarketPosture (may be degraded)

Staleness behaviour
-------------------
If the stored posture's ``valid_until`` is in the past, ``current_posture()``
returns a *degraded* posture: ``risk_dial=0.5`` with a synthetic 24-hour
blackout window starting from the expired ``valid_until`` timestamp.  The
tactical loop must **never** fail open on a stale posture.
"""
from evo_trader.macro.schemas import (
    BlackoutWindow,
    CalendarEvent,
    EventCalendar,
    ImpactTier,
    MarketPosture,
    Regime,
)
from evo_trader.macro.store import PostureStore

__all__ = [
    "BlackoutWindow",
    "CalendarEvent",
    "EventCalendar",
    "ImpactTier",
    "MarketPosture",
    "PostureStore",
    "Regime",
]
