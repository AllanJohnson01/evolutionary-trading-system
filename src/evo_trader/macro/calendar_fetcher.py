"""
Calendar fetcher — pulls upcoming FOMC/CPI/NFP dates and earnings.

Sources:
- Economic events (FOMC, CPI, NFP): scraped from stooq.com/db/is/ which
  provides a plain-text economic calendar without authentication.
- Earnings: fetched from financialmodelingprep.com free tier (no key needed
  for the earnings calendar endpoint on a short lookahead).
- Crypto catalyst events: merged from a manually-maintained
  ``state/crypto_events.json`` file (user-managed; missing file is graceful).

All network calls are wrapped in try/except so a timeout or HTTP error never
crashes the caller — ``fetch_errors`` on the returned ``EventCalendar``
records what went wrong.

**Tests must pass ``use_fixtures=True`` to avoid live network calls.**
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from evo_trader.macro.schemas import (
    CalendarEvent,
    EventCalendar,
    ImpactTier,
)

logger = logging.getLogger(__name__)

_CRYPTO_EVENTS_PATH = Path("state/crypto_events.json")

# Lookahead window for fetching events
_LOOKAHEAD_DAYS = 14

# Fixture paths used in tests (src/evo_trader/macro/ → 4 parents → repo root)
_FIXTURE_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures"


def fetch_calendar(
    symbols: list[str] | None = None,
    lookahead_days: int = _LOOKAHEAD_DAYS,
    use_fixtures: bool = False,
    crypto_events_path: Path = _CRYPTO_EVENTS_PATH,
) -> EventCalendar:
    """
    Build an ``EventCalendar`` for the next ``lookahead_days`` days.

    Parameters
    ----------
    symbols:
        Equity/crypto tickers to include in earnings lookups.  None → no
        earnings fetched.
    lookahead_days:
        How many calendar days ahead to include.
    use_fixtures:
        When True, load pre-recorded JSON fixtures from ``tests/fixtures/``
        instead of making live network calls.  Required for tests.
    crypto_events_path:
        Path to the manually-maintained crypto events JSON.  Missing file is
        tolerated.
    """
    now = datetime.now(tz=timezone.utc)
    fetched_at = now.isoformat()
    events: list[CalendarEvent] = []
    errors: list[str] = []

    # 1. Economic calendar (FOMC / CPI / NFP)
    econ_events, econ_errors = _fetch_economic_events(
        now, lookahead_days, use_fixtures
    )
    events.extend(econ_events)
    errors.extend(econ_errors)

    # 2. Earnings calendar
    if symbols:
        earn_events, earn_errors = _fetch_earnings_events(
            symbols, now, lookahead_days, use_fixtures
        )
        events.extend(earn_events)
        errors.extend(earn_errors)

    # 3. Manually-maintained crypto events
    crypto_events, crypto_errors = _load_crypto_events(crypto_events_path, now, lookahead_days)
    events.extend(crypto_events)
    errors.extend(crypto_errors)

    return EventCalendar(
        events=tuple(events),
        fetched_at=fetched_at,
        fetch_errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Economic events (FOMC, CPI, NFP) — stooq.com/db/is/
# ---------------------------------------------------------------------------

_STOOQ_ECON_FIXTURE = "stooq_econ.json"

_HIGH_IMPACT_KEYWORDS = {"fomc", "federal", "cpi", "pce", "nfp", "payroll", "gdp", "rate decision"}
_MEDIUM_IMPACT_KEYWORDS = {"pmi", "ism", "retail sales", "jolts", "claims", "housing"}


def _fetch_economic_events(
    now: datetime, lookahead_days: int, use_fixtures: bool
) -> tuple[list[CalendarEvent], list[str]]:
    events: list[CalendarEvent] = []
    errors: list[str] = []

    if use_fixtures:
        fixture_path = _FIXTURE_DIR / _STOOQ_ECON_FIXTURE
        if not fixture_path.exists():
            return events, errors
        try:
            raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"econ fixture load error: {exc}")
            return events, errors
        return _parse_stooq_fixture(raw, now, lookahead_days), errors

    # Live fetch from stooq economic calendar (public, no auth)
    url = "https://stooq.com/db/is/"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        events = _parse_stooq_html(html, now, lookahead_days)
    except (urllib.error.URLError, OSError, Exception) as exc:
        errors.append(f"econ calendar fetch error: {exc}")

    return events, errors


def _parse_stooq_fixture(data: list[dict[str, Any]], now: datetime, lookahead_days: int) -> list[CalendarEvent]:
    """Parse pre-recorded fixture data (same schema as _parse_stooq_html output)."""
    cutoff = now + timedelta(days=lookahead_days)
    events: list[CalendarEvent] = []
    for item in data:
        try:
            ts_str = str(item["timestamp"])
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < now or ts > cutoff:
                continue
            events.append(
                CalendarEvent(
                    timestamp=ts.isoformat(),
                    symbol_scope=tuple(item.get("symbol_scope", [])),
                    impact_tier=ImpactTier(int(item.get("impact_tier", 2))),
                    description=str(item.get("description", "")),
                    source=str(item.get("source", "fixture")),
                )
            )
        except Exception:
            continue
    return events


def _parse_stooq_html(html: str, now: datetime, lookahead_days: int) -> list[CalendarEvent]:
    """
    Minimal HTML scrape of stooq economic calendar rows.
    Returns whatever we can parse; silently drops malformed rows.
    """
    import re
    events: list[CalendarEvent] = []
    cutoff = now + timedelta(days=lookahead_days)

    # Stooq table rows have format: <td>YYYY-MM-DD</td><td>Event Name</td>
    row_re = re.compile(r"<tr[^>]*>.*?</tr>", re.DOTALL | re.IGNORECASE)
    date_re = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
    tag_re = re.compile(r"<[^>]+>")

    for row in row_re.finditer(html):
        text = tag_re.sub(" ", row.group()).strip()
        date_m = date_re.search(text)
        if not date_m:
            continue
        try:
            ts = datetime.fromisoformat(date_m.group(1)).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if ts < now or ts > cutoff:
            continue

        desc = text.replace(date_m.group(1), "").strip()[:200]
        lower = desc.lower()
        if any(kw in lower for kw in _HIGH_IMPACT_KEYWORDS):
            tier = ImpactTier.HIGH
        elif any(kw in lower for kw in _MEDIUM_IMPACT_KEYWORDS):
            tier = ImpactTier.MEDIUM
        else:
            tier = ImpactTier.LOW

        events.append(
            CalendarEvent(
                timestamp=ts.isoformat(),
                symbol_scope=(),
                impact_tier=tier,
                description=desc,
                source="stooq.com",
            )
        )
    return events


# ---------------------------------------------------------------------------
# Earnings calendar — financialmodelingprep.com free tier
# ---------------------------------------------------------------------------

_EARNINGS_FIXTURE = "fmp_earnings.json"
_FMP_EARNINGS_URL = (
    "https://financialmodelingprep.com/api/v3/earning_calendar"
    "?from={from_date}&to={to_date}"
)


def _fetch_earnings_events(
    symbols: list[str],
    now: datetime,
    lookahead_days: int,
    use_fixtures: bool,
) -> tuple[list[CalendarEvent], list[str]]:
    events: list[CalendarEvent] = []
    errors: list[str] = []

    if use_fixtures:
        fixture_path = _FIXTURE_DIR / _EARNINGS_FIXTURE
        if not fixture_path.exists():
            return events, errors
        try:
            data = json.loads(fixture_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"earnings fixture load error: {exc}")
            return events, errors
        return _parse_fmp_earnings(data, symbols, now, lookahead_days), errors

    from_date = now.strftime("%Y-%m-%d")
    to_date = (now + timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
    url = _FMP_EARNINGS_URL.format(from_date=from_date, to_date=to_date)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        events = _parse_fmp_earnings(data, symbols, now, lookahead_days)
    except (urllib.error.URLError, OSError, json.JSONDecodeError, Exception) as exc:
        errors.append(f"earnings fetch error: {exc}")

    return events, errors


def _parse_fmp_earnings(
    data: list[dict[str, Any]],
    symbols: list[str],
    now: datetime,
    lookahead_days: int,
) -> list[CalendarEvent]:
    upper_symbols = {s.upper() for s in symbols}
    cutoff = now + timedelta(days=lookahead_days)
    events: list[CalendarEvent] = []

    for item in data:
        try:
            sym = str(item.get("symbol", "")).upper()
            if upper_symbols and sym not in upper_symbols:
                continue
            date_str = str(item.get("date", ""))
            ts = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
            if ts < now or ts > cutoff:
                continue
            events.append(
                CalendarEvent(
                    timestamp=ts.isoformat(),
                    symbol_scope=(sym,),
                    impact_tier=ImpactTier.MEDIUM,
                    description=f"Earnings: {sym}",
                    source="financialmodelingprep.com",
                )
            )
        except Exception:
            continue
    return events


# ---------------------------------------------------------------------------
# Crypto events — manually-maintained JSON
# ---------------------------------------------------------------------------

def _load_crypto_events(
    path: Path, now: datetime, lookahead_days: int
) -> tuple[list[CalendarEvent], list[str]]:
    events: list[CalendarEvent] = []
    errors: list[str] = []

    if not path.exists():
        return events, errors

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"crypto_events load error: {exc}")
        return events, errors

    cutoff = now + timedelta(days=lookahead_days)
    for item in data:
        try:
            ts = datetime.fromisoformat(str(item["timestamp"]))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < now or ts > cutoff:
                continue
            events.append(
                CalendarEvent(
                    timestamp=ts.isoformat(),
                    symbol_scope=tuple(str(s) for s in item.get("symbol_scope", [])),
                    impact_tier=ImpactTier(int(item.get("impact_tier", 2))),
                    description=str(item.get("description", "")),
                    source="crypto_events.json",
                )
            )
        except Exception:
            continue
    return events, errors
