"""
Tests for StrategyRunner (tactical execution loop).

Uses lightweight inline mock classes — no unittest.mock.
No live network calls.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from evo_trader.agents.base import AgentType, TradingAgent
from evo_trader.macro.schemas import (
    BlackoutWindow as MacroBlackoutWindow,
    MarketPosture as MacroPosture,
    Regime,
)
from evo_trader.macro.store import PostureStore
from evo_trader.portfolio.manager import PortfolioManager
from evo_trader.strategies.base import Bar, Position
from evo_trader.strategies.intraday_momentum import (
    IntradayMomentumParams,
    IntradayMomentumStrategy,
)
from evo_trader.strategies.mean_reversion import (
    MeanReversionParams,
    MeanReversionStrategy,
)
from evo_trader.tactical.runner import StrategyRunner


# ---------------------------------------------------------------------------
# Inline mock clients
# ---------------------------------------------------------------------------


class MockBarsClient:
    """Returns pre-loaded bars per symbol; no network access."""

    def __init__(self, bars_by_symbol: Dict[str, List[Bar]]) -> None:
        self._bars_by_symbol = bars_by_symbol

    def get_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        limit: int,
    ) -> List[Bar]:
        return self._bars_by_symbol.get(symbol, [])


class MockOrderClient:
    """Records orders and positions in-memory."""

    def __init__(self) -> None:
        self.submitted_orders: List[tuple[str, str, Decimal]] = []
        self.closed_positions: List[tuple[str, str, str]] = []
        self._open_positions: Dict[str, List[Position]] = {}

    def submit_order(
        self,
        symbol: str,
        direction: str,
        notional: Decimal,
    ) -> str:
        self.submitted_orders.append((symbol, direction, notional))
        return "test_order_id"

    def get_open_positions(self, strategy_id: str) -> List[Position]:
        return self._open_positions.get(strategy_id, [])

    def close_position(
        self,
        strategy_id: str,
        symbol: str,
        reason: str,
    ) -> None:
        self.closed_positions.append((strategy_id, symbol, reason))

    def set_open_positions(
        self,
        strategy_id: str,
        positions: List[Position],
    ) -> None:
        self._open_positions[strategy_id] = positions


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TRADE_COUNT_NORMAL = 30  # above cold_start_threshold (30) triggers Kelly path
_WIN_RATE = Decimal("0.6")
_PAYOFF_RATIO = Decimal("1.5")
_AGENT_ALLOCATION = Decimal("12000")


def _make_posture(
    risk_dial: Decimal = Decimal("1.0"),
    blackout_windows: tuple[MacroBlackoutWindow, ...] = (),
) -> MacroPosture:
    """Build a fresh, non-stale MacroPosture."""
    now = datetime.now(tz=timezone.utc)
    valid_until = (now + timedelta(hours=12)).isoformat()
    return MacroPosture(
        as_of=now.isoformat(),
        regime=Regime.NEUTRAL,
        risk_dial=risk_dial,
        blackout_windows=blackout_windows,
        notes="test posture",
        valid_until=valid_until,
    )


def _make_bar(
    close: Decimal,
    high: Optional[Decimal] = None,
    low: Optional[Decimal] = None,
    volume: Decimal = Decimal("10000"),
) -> Bar:
    if high is None:
        high = close * Decimal("1.002")
    if low is None:
        low = close * Decimal("0.998")
    return Bar(
        timestamp="2024-01-01T10:00:00Z",
        open=close,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _declining_then_spike_bars() -> List[Bar]:
    """
    Build a bar sequence that reliably triggers a long signal for IntradayMomentum.

    50 declining bars push RSI well below 55 (avg_loss >> avg_gain).
    9 additional declining bars set bars[-10].close to a known value.
    The final bar spikes 8% above bars[-10].close with 5× normal volume.
    Paired with _momentum_strategy_wide_thresholds(), this reliably fires a long.
    """
    bars: List[Bar] = []
    price = Decimal("100")
    for i in range(50):
        price = price * Decimal("0.99")
        bars.append(Bar(
            timestamp=f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00Z",
            open=price, high=price * Decimal("1.001"),
            low=price * Decimal("0.999"), close=price, volume=Decimal("10000"),
        ))
    for i in range(9):
        price = price * Decimal("0.995")
        bars.append(Bar(
            timestamp=f"2024-01-01T{(50 + i) // 60:02d}:{(50 + i) % 60:02d}:00Z",
            open=price, high=price * Decimal("1.001"),
            low=price * Decimal("0.999"), close=price, volume=Decimal("10000"),
        ))
    spike_close = bars[-1].close * Decimal("1.08")
    bars.append(Bar(
        timestamp="2024-01-01T01:00:00Z",
        open=bars[-1].close, high=spike_close * Decimal("1.001"),
        low=bars[-1].close, close=spike_close, volume=Decimal("50000"),
    ))
    return bars


def _momentum_strategy_wide_thresholds(
    symbol: str = "AAPL",
    strategy_id: str = "test_momentum",
) -> IntradayMomentumStrategy:
    """Return an IntradayMomentumStrategy with thresholds calibrated for _declining_then_spike_bars."""
    params = IntradayMomentumParams(
        lookback_bars=10,
        rsi_oversold=Decimal("55"),   # RSI after 50+ declines is ~54; this threshold fires
        momentum_threshold=Decimal("0.001"),
        min_volume_ratio=Decimal("1.5"),
        atr_multiplier_stop=Decimal("2.0"),
        atr_multiplier_tp=Decimal("3.0"),
        atr_multiplier_trail=Decimal("1.5"),
        kelly_cap=Decimal("0.15"),
    )
    return IntradayMomentumStrategy(
        symbol=symbol, params=params, strategy_id=strategy_id
    )


def make_runner(
    strategies: list,
    bars_by_symbol: Dict[str, List[Bar]],
    open_positions_by_strategy: Optional[Dict[str, List[Position]]] = None,
    risk_dial: Decimal = Decimal("1.0"),
    tmp_dir: Optional[str] = None,
) -> tuple[StrategyRunner, MockOrderClient]:
    """
    Build a fully wired StrategyRunner with mock clients.

    Each strategy is registered with trade_count=30 (Kelly path), WIN_RATE=0.6,
    PAYOFF_RATIO=1.5 unless a different trade_count is passed as an attribute on
    the strategy object via the 'override_trade_count' attribute (not standard —
    caller uses the returned runner to call register_strategy again with custom args).

    Returns:
        (runner, mock_order_client)
    """
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()

    tmp_path = Path(tmp_dir)
    posture_path = tmp_path / "market_posture.json"
    calendar_path = tmp_path / "event_calendar.json"
    log_path = tmp_path / "decisions.jsonl"

    posture = _make_posture(risk_dial=risk_dial)
    store = PostureStore(posture_path=posture_path, calendar_path=calendar_path)
    store.write_posture(posture)

    pm = PortfolioManager()
    mock_order = MockOrderClient()

    if open_positions_by_strategy:
        for sid, positions in open_positions_by_strategy.items():
            mock_order.set_open_positions(sid, positions)

    bars_client = MockBarsClient(bars_by_symbol)

    runner = StrategyRunner(
        portfolio_manager=pm,
        posture_store=store,
        bars_client=bars_client,
        order_client=mock_order,
        log_path=log_path,
    )

    for strategy in strategies:
        agent = TradingAgent(agent_type=AgentType.MOMENTUM)
        pm.register_agent(agent, _AGENT_ALLOCATION)
        runner.register_strategy(
            strategy=strategy,
            agent_id=agent.agent_id,
            trade_count=_TRADE_COUNT_NORMAL,
            win_rate=_WIN_RATE,
            payoff_ratio=_PAYOFF_RATIO,
        )

    return runner, mock_order


# ---------------------------------------------------------------------------
# TestStrategyRunner
# ---------------------------------------------------------------------------


class TestStrategyRunner:

    # 1. Signal fires → order submitted
    def test_run_once_submits_order_on_signal(self) -> None:
        strategy = _momentum_strategy_wide_thresholds(symbol="AAPL")
        bars = _declining_then_spike_bars()
        runner, mock_order = make_runner(
            strategies=[strategy],
            bars_by_symbol={"AAPL": bars},
        )
        runner.run_once()
        assert len(mock_order.submitted_orders) == 1
        _, direction, _ = mock_order.submitted_orders[0]
        assert direction == "long"

    # 2. No signal → no order
    def test_run_once_skips_entry_on_no_signal(self) -> None:
        strategy = IntradayMomentumStrategy(
            symbol="AAPL",
            params=IntradayMomentumParams(lookback_bars=10),
            strategy_id="flat_momentum",
        )
        # Flat bars produce no extreme RSI or momentum
        flat_bars = []
        for i in range(15):
            flat_bars.append(
                Bar(
                    timestamp=f"2024-01-01T00:{i:02d}:00Z",
                    open=Decimal("100"),
                    high=Decimal("100.1"),
                    low=Decimal("99.9"),
                    close=Decimal("100"),
                    volume=Decimal("5000"),
                )
            )
        runner, mock_order = make_runner(
            strategies=[strategy],
            bars_by_symbol={"AAPL": flat_bars},
        )
        runner.run_once()
        assert len(mock_order.submitted_orders) == 0

    # 3. Pre-loaded long position at stop; bar.low triggers stop → position closed
    def test_run_once_closes_position_on_stop(self) -> None:
        strategy = _momentum_strategy_wide_thresholds(symbol="AAPL")
        position = Position(
            strategy_id=strategy.strategy_id,
            symbol="AAPL",
            direction="long",
            entry_price=Decimal("105"),
            size=Decimal("10"),
            stop_price=Decimal("100"),
            take_profit_price=Decimal("120"),
            trailing_stop_distance=Decimal("0.02"),
            entry_time="2024-01-01T00:00:00Z",
        )
        # Build bars where the last bar's low breaches the stop
        bars = _declining_then_spike_bars()
        # Overwrite the last bar: low = 99 (below stop_price=100)
        last = bars[-1]
        bars[-1] = Bar(
            timestamp=last.timestamp,
            open=last.open,
            high=last.high,
            low=Decimal("99"),
            close=last.close,
            volume=last.volume,
        )
        runner, mock_order = make_runner(
            strategies=[strategy],
            bars_by_symbol={"AAPL": bars},
            open_positions_by_strategy={strategy.strategy_id: [position]},
        )
        runner.run_once()
        assert len(mock_order.closed_positions) == 1
        sid, sym, reason = mock_order.closed_positions[0]
        assert reason == "hard_stop"

    # 4. EOD flatten for equity intraday — monkeypatched _is_near_market_close
    def test_run_once_closes_eod_for_equity_intraday(self, monkeypatch) -> None:
        strategy = _momentum_strategy_wide_thresholds(symbol="AAPL")
        position = Position(
            strategy_id=strategy.strategy_id,
            symbol="AAPL",
            direction="long",
            entry_price=Decimal("105"),
            size=Decimal("10"),
            stop_price=Decimal("95"),
            take_profit_price=Decimal("120"),
            trailing_stop_distance=Decimal("0.02"),
            entry_time="2024-01-01T00:00:00Z",
        )
        bars = []
        for i in range(15):
            bars.append(
                Bar(
                    timestamp=f"2024-01-01T00:{i:02d}:00Z",
                    open=Decimal("105"),
                    high=Decimal("106"),
                    low=Decimal("104"),
                    close=Decimal("105"),
                    volume=Decimal("5000"),
                )
            )
        monkeypatch.setattr(
            "evo_trader.tactical.runner._is_near_market_close",
            lambda _dt: True,
        )
        runner, mock_order = make_runner(
            strategies=[strategy],
            bars_by_symbol={"AAPL": bars},
            open_positions_by_strategy={strategy.strategy_id: [position]},
        )
        runner.run_once()
        assert len(mock_order.closed_positions) >= 1
        reasons = [r for _sid, _sym, r in mock_order.closed_positions]
        assert "eod_flatten" in reasons

    # 5. Blackout window suppresses entry
    def test_blackout_window_suppresses_entry(self) -> None:
        strategy = _momentum_strategy_wide_thresholds(symbol="AAPL")
        bars = _declining_then_spike_bars()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            posture_path = tmp_path / "market_posture.json"
            log_path = tmp_path / "decisions.jsonl"

            now = datetime.now(tz=timezone.utc)
            blackout = MacroBlackoutWindow(
                start=(now - timedelta(minutes=5)).isoformat(),
                end=(now + timedelta(minutes=30)).isoformat(),
                reason="test_blackout",
            )
            posture = MacroPosture(
                as_of=now.isoformat(),
                regime=Regime.NEUTRAL,
                risk_dial=Decimal("1.0"),
                blackout_windows=(blackout,),
                notes="blackout test",
                valid_until=(now + timedelta(hours=12)).isoformat(),
            )
            store = PostureStore(
                posture_path=posture_path,
                calendar_path=tmp_path / "calendar.json",
            )
            store.write_posture(posture)

            pm = PortfolioManager()
            mock_order = MockOrderClient()
            bars_client = MockBarsClient({"AAPL": bars})

            runner = StrategyRunner(
                portfolio_manager=pm,
                posture_store=store,
                bars_client=bars_client,
                order_client=mock_order,
                log_path=log_path,
            )
            agent = TradingAgent(agent_type=AgentType.MOMENTUM)
            pm.register_agent(agent, _AGENT_ALLOCATION)
            runner.register_strategy(
                strategy=strategy,
                agent_id=agent.agent_id,
                trade_count=_TRADE_COUNT_NORMAL,
                win_rate=_WIN_RATE,
                payoff_ratio=_PAYOFF_RATIO,
            )
            runner.run_once()

        assert len(mock_order.submitted_orders) == 0
        # Confirm decision log records the blackout rejection
        if log_path.exists():
            entries = [
                json.loads(line)
                for line in log_path.read_text().splitlines()
                if line.strip()
            ]
            approval_reasons = [e.get("approval_reason", "") for e in entries]
            assert any(
                "blackout" in (r or "").lower() for r in approval_reasons
            )

    # 6. 30-min strategy skips on 15-min tick (minute=15 → not 0 or 30)
    def test_30min_strategy_skips_on_15min_tick(self, monkeypatch) -> None:
        strategy = MeanReversionStrategy(
            symbol="BTC/USDT",
            params=MeanReversionParams(lookback_bars=20, entry_z=Decimal("2.0")),
            strategy_id="test_mean_rev_skip",
        )
        # Patch datetime.now to return minute=15
        fixed_dt = datetime(2024, 1, 1, 15, 15, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(
            "evo_trader.tactical.runner.datetime",
            _make_datetime_mock(fixed_dt),
        )
        # Even if bars exist, cadence gate fires before get_bars — but the runner
        # calls get_bars after the cadence check, so no bars need to be provided.
        runner, mock_order = make_runner(
            strategies=[strategy],
            bars_by_symbol={"BTC/USDT": []},
        )
        runner.run_once()
        assert len(mock_order.submitted_orders) == 0

    # 7. Decision log is written after run_once
    def test_decision_log_written(self) -> None:
        strategy = _momentum_strategy_wide_thresholds(symbol="AAPL")
        bars = _declining_then_spike_bars()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_path = tmp_path / "decisions.jsonl"

            runner, _ = make_runner(
                strategies=[strategy],
                bars_by_symbol={"AAPL": bars},
                tmp_dir=tmp_dir,
            )
            runner.run_once()

            assert log_path.exists()
            lines = [l for l in log_path.read_text().splitlines() if l.strip()]
            assert len(lines) >= 1
            # Each line must parse as valid JSON
            for line in lines:
                obj = json.loads(line)
                assert "strategy_id" in obj

    # 8. Cold start uses fixed fraction
    def test_cold_start_uses_fixed_fraction(self) -> None:
        """Strategy with trade_count=0 → notional ≈ cold_start_fixed_risk * allocation."""
        strategy = _momentum_strategy_wide_thresholds(symbol="AAPL", strategy_id="cold_start")
        bars = _declining_then_spike_bars()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            posture_path = tmp_path / "market_posture.json"
            log_path = tmp_path / "decisions.jsonl"

            posture = _make_posture(risk_dial=Decimal("1.0"))
            store = PostureStore(
                posture_path=posture_path,
                calendar_path=tmp_path / "cal.json",
            )
            store.write_posture(posture)

            pm = PortfolioManager()
            mock_order = MockOrderClient()
            runner = StrategyRunner(
                portfolio_manager=pm,
                posture_store=store,
                bars_client=MockBarsClient({"AAPL": bars}),
                order_client=mock_order,
                log_path=log_path,
            )
            agent = TradingAgent(agent_type=AgentType.MOMENTUM)
            pm.register_agent(agent, _AGENT_ALLOCATION)
            runner.register_strategy(
                strategy=strategy,
                agent_id=agent.agent_id,
                trade_count=0,  # cold start
                win_rate=_WIN_RATE,
                payoff_ratio=_PAYOFF_RATIO,
            )
            runner.run_once()

        assert len(mock_order.submitted_orders) == 1
        _sym, _dir, notional = mock_order.submitted_orders[0]
        # cold_start_fixed_risk=0.01, risk_dial=1.0 → fraction=0.01
        # notional = 0.01 * 12000 = 120
        from evo_trader.config.parameters import SizingConfig
        cfg = SizingConfig()
        expected = cfg.cold_start_fixed_risk * Decimal("1.0") * _AGENT_ALLOCATION
        assert abs(notional - expected) < Decimal("1")

    # 9. risk_dial=0.5 produces half the notional compared to risk_dial=1.0
    def test_posture_risk_dial_scales_notional(self) -> None:
        strategy_full = _momentum_strategy_wide_thresholds(
            symbol="AAPL", strategy_id="dial_full"
        )
        strategy_half = _momentum_strategy_wide_thresholds(
            symbol="AAPL", strategy_id="dial_half"
        )
        bars = _declining_then_spike_bars()

        def _run_with_dial(dial: Decimal, strategy: IntradayMomentumStrategy) -> Decimal:
            with tempfile.TemporaryDirectory() as tmp_dir:
                runner, mock_order = make_runner(
                    strategies=[strategy],
                    bars_by_symbol={"AAPL": bars},
                    risk_dial=dial,
                    tmp_dir=tmp_dir,
                )
                runner.run_once()
            if mock_order.submitted_orders:
                return mock_order.submitted_orders[0][2]
            return Decimal("0")

        notional_full = _run_with_dial(Decimal("1.0"), strategy_full)
        notional_half = _run_with_dial(Decimal("0.5"), strategy_half)

        assert notional_full > Decimal("0")
        assert notional_half > Decimal("0")
        # Allow 1% tolerance for floating-point artefacts in Decimal math
        ratio = notional_full / notional_half
        assert abs(ratio - Decimal("2")) < Decimal("0.05")

    # 10. Agent with zero allocation produces no order
    def test_zero_allocation_skips_entry(self) -> None:
        """When agent_allocation returns 0, notional=0 and no order is submitted."""
        strategy = _momentum_strategy_wide_thresholds(
            symbol="AAPL", strategy_id="zero_alloc"
        )
        bars = _declining_then_spike_bars()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            posture_path = tmp_path / "market_posture.json"
            log_path = tmp_path / "decisions.jsonl"

            posture = _make_posture(risk_dial=Decimal("1.0"))
            store = PostureStore(
                posture_path=posture_path,
                calendar_path=tmp_path / "cal.json",
            )
            store.write_posture(posture)

            pm = PortfolioManager()
            mock_order = MockOrderClient()
            runner = StrategyRunner(
                portfolio_manager=pm,
                posture_store=store,
                bars_client=MockBarsClient({"AAPL": bars}),
                order_client=mock_order,
                log_path=log_path,
            )
            # Register with a bogus agent_id that has no capital in the PM
            # (we intentionally skip pm.register_agent for this agent_id)
            runner.register_strategy(
                strategy=strategy,
                agent_id="unregistered_agent_id",
                trade_count=_TRADE_COUNT_NORMAL,
                win_rate=_WIN_RATE,
                payoff_ratio=_PAYOFF_RATIO,
            )
            runner.run_once()

        assert len(mock_order.submitted_orders) == 0


# ---------------------------------------------------------------------------
# Helper: datetime mock for cadence tests
# ---------------------------------------------------------------------------


class _DatetimeMock:
    """Minimal datetime shim that overrides ``now()`` with a fixed value."""

    def __init__(self, fixed: datetime) -> None:
        self._fixed = fixed

    def now(self, tz=None) -> datetime:
        if tz is not None:
            return self._fixed.astimezone(tz)
        return self._fixed

    def fromisoformat(self, s: str) -> datetime:
        return datetime.fromisoformat(s)

    def __getattr__(self, name: str):
        return getattr(datetime, name)


def _make_datetime_mock(fixed_dt: datetime) -> _DatetimeMock:
    return _DatetimeMock(fixed_dt)
