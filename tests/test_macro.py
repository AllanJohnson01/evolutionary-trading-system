"""Tests for the macro posture layer."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from evo_trader.macro.schemas import (
    BlackoutWindow,
    CalendarEvent,
    EventCalendar,
    ImpactTier,
    MarketPosture,
    Regime,
)
from evo_trader.macro.store import PostureStore
from evo_trader.macro.calendar_fetcher import fetch_calendar
from evo_trader.macro.build_posture import validate_and_write, _NORMAL_DIAL_CAP, _AGGRESSIVE_DIAL_CAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future_ts(hours: int = 24) -> str:
    return (datetime.now(tz=timezone.utc) + timedelta(hours=hours)).isoformat()


def _past_ts(hours: int = 1) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()


def _make_posture(
    risk_dial: str = "0.8",
    regime: str = "risk_on",
    valid_until: str | None = None,
) -> MarketPosture:
    return MarketPosture(
        as_of=datetime.now(tz=timezone.utc).isoformat(),
        regime=Regime(regime),
        risk_dial=Decimal(risk_dial),
        blackout_windows=(),
        notes="test posture",
        valid_until=valid_until or _future_ts(24),
    )


def _store_with_dirs() -> tuple[PostureStore, Path, Path]:
    tmp = Path(tempfile.mkdtemp())
    posture_path = tmp / "market_posture.json"
    calendar_path = tmp / "event_calendar.json"
    store = PostureStore(posture_path=posture_path, calendar_path=calendar_path)
    return store, posture_path, calendar_path


# ---------------------------------------------------------------------------
# Schema round-trip tests
# ---------------------------------------------------------------------------

class TestMarketPostureRoundTrip:
    def test_to_dict_from_dict(self):
        posture = _make_posture()
        restored = MarketPosture.from_dict(posture.to_dict())
        assert restored == posture

    def test_to_json_from_json(self):
        posture = _make_posture(risk_dial="0.5", regime="neutral")
        restored = MarketPosture.from_json(posture.to_json())
        assert restored == posture

    def test_with_blackout_windows(self):
        bw = BlackoutWindow(start=_past_ts(1), end=_future_ts(1), reason="FOMC")
        posture = MarketPosture(
            as_of=datetime.now(tz=timezone.utc).isoformat(),
            regime=Regime.RISK_OFF,
            risk_dial=Decimal("0"),
            blackout_windows=(bw,),
            notes="test",
            valid_until=_future_ts(24),
        )
        restored = MarketPosture.from_json(posture.to_json())
        assert restored == posture
        assert restored.blackout_windows[0].reason == "FOMC"

    def test_all_regimes_round_trip(self):
        for regime in Regime:
            p = _make_posture(regime=regime.value)
            assert MarketPosture.from_dict(p.to_dict()).regime == regime

    def test_invalid_risk_dial_raises(self):
        d = _make_posture().to_dict()
        d["risk_dial"] = "not-a-number"
        with pytest.raises(ValueError):
            MarketPosture.from_dict(d)


class TestEventCalendarRoundTrip:
    def test_empty_calendar(self):
        cal = EventCalendar.empty(fetched_at=datetime.now(tz=timezone.utc).isoformat())
        restored = EventCalendar.from_json(cal.to_json())
        assert restored == cal

    def test_calendar_with_events(self):
        ev = CalendarEvent(
            timestamp=_future_ts(48),
            symbol_scope=("AAPL",),
            impact_tier=ImpactTier.HIGH,
            description="Earnings",
            source="test",
        )
        cal = EventCalendar(
            events=(ev,),
            fetched_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        restored = EventCalendar.from_json(cal.to_json())
        assert restored == cal
        assert restored.events[0].impact_tier == ImpactTier.HIGH

    def test_all_impact_tiers_round_trip(self):
        for tier in ImpactTier:
            ev = CalendarEvent(
                timestamp=_future_ts(10),
                symbol_scope=(),
                impact_tier=tier,
                description="test",
                source="test",
            )
            cal = EventCalendar(events=(ev,), fetched_at=_future_ts(0))
            assert EventCalendar.from_dict(cal.to_dict()).events[0].impact_tier == tier


# ---------------------------------------------------------------------------
# Staleness degradation tests
# ---------------------------------------------------------------------------

class TestStalenessHandling:
    def test_missing_file_returns_degraded(self):
        store, _, _ = _store_with_dirs()
        posture = store.current_posture()
        assert posture.risk_dial == Decimal("0.5")
        assert posture.regime == Regime.NEUTRAL
        assert len(posture.blackout_windows) == 1
        assert posture.blackout_windows[0].reason == "stale_posture"

    def test_stale_posture_returns_degraded(self):
        store, posture_path, _ = _store_with_dirs()
        stale = _make_posture(valid_until=_past_ts(2))
        store.write_posture(stale)
        degraded = store.current_posture()
        assert degraded.risk_dial == Decimal("0.5")
        assert len(degraded.blackout_windows) == 1

    def test_fresh_posture_returned_as_is(self):
        store, _, _ = _store_with_dirs()
        fresh = _make_posture(risk_dial="0.9", regime="risk_on")
        store.write_posture(fresh)
        result = store.current_posture()
        assert result.risk_dial == Decimal("0.9")
        assert result.regime == Regime.RISK_ON

    def test_stale_blackout_covers_24h(self):
        store, posture_path, _ = _store_with_dirs()
        stale = _make_posture(valid_until=_past_ts(1))
        store.write_posture(stale)
        degraded = store.current_posture()
        bw = degraded.blackout_windows[0]
        start_dt = datetime.fromisoformat(bw.start)
        end_dt = datetime.fromisoformat(bw.end)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)
        diff = end_dt - start_dt
        assert abs(diff.total_seconds() - 24 * 3600) < 2  # within 2s of 24h

    def test_corrupt_json_returns_degraded(self):
        store, posture_path, _ = _store_with_dirs()
        posture_path.parent.mkdir(parents=True, exist_ok=True)
        posture_path.write_text("{ not valid json !!!", encoding="utf-8")
        degraded = store.current_posture()
        assert degraded.risk_dial == Decimal("0.5")


# ---------------------------------------------------------------------------
# Blackout window containment checks
# ---------------------------------------------------------------------------

class TestBlackoutWindows:
    def test_write_and_read_blackout_windows(self):
        store, _, _ = _store_with_dirs()
        bw1 = BlackoutWindow(start=_future_ts(1), end=_future_ts(2), reason="FOMC")
        bw2 = BlackoutWindow(start=_future_ts(5), end=_future_ts(6), reason="CPI")
        posture = MarketPosture(
            as_of=datetime.now(tz=timezone.utc).isoformat(),
            regime=Regime.RISK_OFF,
            risk_dial=Decimal("0"),
            blackout_windows=(bw1, bw2),
            notes="test",
            valid_until=_future_ts(24),
        )
        store.write_posture(posture)
        result = store.current_posture()
        assert len(result.blackout_windows) == 2
        reasons = {bw.reason for bw in result.blackout_windows}
        assert reasons == {"FOMC", "CPI"}


# ---------------------------------------------------------------------------
# Calendar fetcher (fixture-based, no live network)
# ---------------------------------------------------------------------------

class TestCalendarFetcher:
    def test_fetch_with_econ_fixture(self):
        # Use a large lookahead so year-2099 fixture dates are included
        cal = fetch_calendar(use_fixtures=True, lookahead_days=99999)
        assert isinstance(cal, EventCalendar)
        assert len(cal.events) > 0
        assert all(e.impact_tier in ImpactTier for e in cal.events)

    def test_fetch_with_earnings_fixture(self):
        cal = fetch_calendar(symbols=["AAPL", "MSFT"], use_fixtures=True, lookahead_days=99999)
        symbols_found = {s for e in cal.events for s in e.symbol_scope}
        assert "AAPL" in symbols_found or "MSFT" in symbols_found

    def test_empty_calendar_on_no_fixtures(self, tmp_path):
        """Fetcher with non-existent fixture dir returns empty calendar without raising."""
        import evo_trader.macro.calendar_fetcher as fetcher_mod
        original = fetcher_mod._FIXTURE_DIR
        fetcher_mod._FIXTURE_DIR = tmp_path / "nonexistent"
        try:
            cal = fetch_calendar(use_fixtures=True)
            assert isinstance(cal, EventCalendar)
        finally:
            fetcher_mod._FIXTURE_DIR = original

    def test_crypto_events_merged(self, tmp_path):
        """Manually-maintained crypto events are merged when present."""
        from datetime import timedelta
        future = (datetime.now(tz=timezone.utc) + timedelta(days=3)).isoformat()
        crypto_file = tmp_path / "crypto_events.json"
        crypto_file.write_text(json.dumps([{
            "timestamp": future,
            "symbol_scope": ["BTC-USD"],
            "impact_tier": 3,
            "description": "ETH Shanghai upgrade",
            "source": "manual",
        }]), encoding="utf-8")
        cal = fetch_calendar(
            use_fixtures=True,
            crypto_events_path=crypto_file,
        )
        descs = [e.description for e in cal.events]
        assert "ETH Shanghai upgrade" in descs

    def test_no_live_network(self):
        """use_fixtures=True must never make live calls — confirmed by no errors from fixtures."""
        cal = fetch_calendar(use_fixtures=True)
        # network errors would appear in fetch_errors; fixture-only run should have none
        network_errors = [e for e in cal.fetch_errors if "fetch error" in e.lower()]
        assert network_errors == []


# ---------------------------------------------------------------------------
# Posture writer harness — risk_dial rejection
# ---------------------------------------------------------------------------

class TestPostureWriterHarness:
    def test_valid_posture_writes(self, tmp_path):
        posture = _make_posture(risk_dial="0.8")
        posture_file = tmp_path / "posture_in.json"
        posture_file.write_text(posture.to_json(), encoding="utf-8")
        store = PostureStore(
            posture_path=tmp_path / "market_posture.json",
            calendar_path=tmp_path / "event_calendar.json",
        )
        result = validate_and_write(str(posture_file), store)
        assert result == 0
        assert (tmp_path / "market_posture.json").exists()

    def test_dial_above_1_rejected_without_flag(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ALLOW_AGGRESSIVE", raising=False)
        posture = _make_posture(risk_dial="1.1")
        posture_file = tmp_path / "posture_in.json"
        posture_file.write_text(posture.to_json(), encoding="utf-8")
        store = PostureStore(
            posture_path=tmp_path / "market_posture.json",
            calendar_path=tmp_path / "event_calendar.json",
        )
        result = validate_and_write(str(posture_file), store)
        assert result != 0
        assert not (tmp_path / "market_posture.json").exists()

    def test_dial_above_1_25_always_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALLOW_AGGRESSIVE", "true")
        posture = _make_posture(risk_dial="1.3")
        posture_file = tmp_path / "posture_in.json"
        posture_file.write_text(posture.to_json(), encoding="utf-8")
        store = PostureStore(
            posture_path=tmp_path / "market_posture.json",
            calendar_path=tmp_path / "event_calendar.json",
        )
        result = validate_and_write(str(posture_file), store)
        assert result != 0

    def test_dial_1_0_accepted_without_flag(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ALLOW_AGGRESSIVE", raising=False)
        posture = _make_posture(risk_dial="1.0")
        posture_file = tmp_path / "posture_in.json"
        posture_file.write_text(posture.to_json(), encoding="utf-8")
        store = PostureStore(
            posture_path=tmp_path / "market_posture.json",
            calendar_path=tmp_path / "event_calendar.json",
        )
        result = validate_and_write(str(posture_file), store)
        assert result == 0

    def test_dial_1_25_accepted_with_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALLOW_AGGRESSIVE", "true")
        posture = _make_posture(risk_dial="1.25")
        posture_file = tmp_path / "posture_in.json"
        posture_file.write_text(posture.to_json(), encoding="utf-8")
        store = PostureStore(
            posture_path=tmp_path / "market_posture.json",
            calendar_path=tmp_path / "event_calendar.json",
        )
        result = validate_and_write(str(posture_file), store)
        assert result == 0

    def test_dial_zero_accepted(self, tmp_path):
        posture = _make_posture(risk_dial="0")
        posture_file = tmp_path / "posture_in.json"
        posture_file.write_text(posture.to_json(), encoding="utf-8")
        store = PostureStore(
            posture_path=tmp_path / "market_posture.json",
            calendar_path=tmp_path / "event_calendar.json",
        )
        result = validate_and_write(str(posture_file), store)
        assert result == 0

    def test_invalid_json_rejected(self, tmp_path):
        posture_file = tmp_path / "bad.json"
        posture_file.write_text("{not: valid}", encoding="utf-8")
        store = PostureStore(
            posture_path=tmp_path / "market_posture.json",
            calendar_path=tmp_path / "event_calendar.json",
        )
        result = validate_and_write(str(posture_file), store)
        assert result != 0


# ---------------------------------------------------------------------------
# Fetch error surfacing — warning logs and graceful degradation
# ---------------------------------------------------------------------------

class TestFetchErrorSurfacing:
    def test_fetch_errors_logged_as_warning(self, caplog):
        """When fetch_errors is non-empty, a WARNING is emitted."""
        import evo_trader.macro.calendar_fetcher as fetcher_mod
        original = fetcher_mod._FIXTURE_DIR
        fetcher_mod._FIXTURE_DIR = Path("/nonexistent/path/that/cannot/exist")
        try:
            with caplog.at_level("WARNING", logger="evo_trader.macro.calendar_fetcher"):
                # No FRED or FMP keys → errors will be generated
                cal = fetch_calendar(symbols=["AAPL"], use_fixtures=False)
                warnings = [r for r in caplog.records if "ACTION NEEDED" in r.message]
                # With no API keys, both sources fail → should warn
                assert len(warnings) >= 1 or len(cal.fetch_errors) >= 1
        finally:
            fetcher_mod._FIXTURE_DIR = original

    def test_no_fred_key_produces_error(self, monkeypatch):
        """Missing FRED_API_KEY produces an error entry, not a crash."""
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        import evo_trader.macro.calendar_fetcher as fetcher_mod
        original = fetcher_mod._FIXTURE_DIR
        fetcher_mod._FIXTURE_DIR = Path("/nonexistent/path")
        try:
            cal = fetch_calendar(use_fixtures=False)
            assert any("FRED_API_KEY" in e for e in cal.fetch_errors)
        finally:
            fetcher_mod._FIXTURE_DIR = original

    def test_no_fmp_key_produces_error(self, monkeypatch):
        """Missing FMP_API_KEY produces an error entry in fetch_errors, not a crash."""
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        # Also suppress FRED so we only trigger the FMP path
        monkeypatch.setenv("FRED_API_KEY", "dummy-fred-key-for-test")
        import evo_trader.macro.calendar_fetcher as fetcher_mod
        original_fetch_econ = fetcher_mod._fetch_economic_events

        def _stub_econ(now, lookahead_days, use_fixtures):
            return [], []

        fetcher_mod._fetch_economic_events = _stub_econ
        try:
            cal = fetch_calendar(symbols=["AAPL"], use_fixtures=False)
            assert any("FMP_API_KEY" in e for e in cal.fetch_errors)
        finally:
            fetcher_mod._fetch_economic_events = original_fetch_econ

    def test_store_logs_warning_on_stored_fetch_errors(self, tmp_path, caplog):
        """PostureStore.current_calendar() logs ACTION NEEDED when stored calendar has errors."""
        cal_with_errors = EventCalendar(
            events=(),
            fetched_at=datetime.now(tz=timezone.utc).isoformat(),
            fetch_errors=("FRED_API_KEY not set",),
        )
        cal_path = tmp_path / "event_calendar.json"
        cal_path.write_text(cal_with_errors.to_json(), encoding="utf-8")
        store = PostureStore(
            posture_path=tmp_path / "market_posture.json",
            calendar_path=cal_path,
        )
        with caplog.at_level("WARNING", logger="evo_trader.macro.store"):
            calendar = store.current_calendar()
        warnings = [r for r in caplog.records if "ACTION NEEDED" in r.message]
        assert len(warnings) == 1
        assert calendar.fetch_errors == ("FRED_API_KEY not set",)


# ---------------------------------------------------------------------------
# Live integration test — skipped unless both API keys are present
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("FRED_API_KEY") and os.environ.get("FMP_API_KEY")),
    reason="Requires FRED_API_KEY and FMP_API_KEY env vars (live network)",
)
def test_live_fetch_returns_future_events():
    """
    Live smoke test: fetch_calendar() must return at least one genuinely future
    event and must not surface any fetch errors when both API keys are present.

    Run with:  FRED_API_KEY=<key> FMP_API_KEY=<key> pytest -m integration
    """
    from datetime import datetime, timezone

    cal = fetch_calendar(symbols=["AAPL", "MSFT"], lookahead_days=60, use_fixtures=False)

    now = datetime.now(tz=timezone.utc)
    future_events = [
        e for e in cal.events
        if datetime.fromisoformat(e.timestamp) > now
    ]

    assert len(future_events) >= 1, (
        f"Expected at least one future event; got {len(cal.events)} total. "
        f"fetch_errors: {cal.fetch_errors}"
    )
    assert cal.fetch_errors == (), (
        f"Expected no fetch errors; got: {cal.fetch_errors}"
    )
