"""
Tactical execution loop.

Entry point::

    python -m evo_trader.tactical.runner --once

``StrategyRunner`` orchestrates the per-bar cycle:
  1. Load macro posture
  2. For each registered strategy: check exits → evaluate entry → size → approve → submit

All client I/O is injected through Protocol interfaces so the runner can be
fully exercised in tests without live network calls.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Protocol

from evo_trader.macro.schemas import MarketPosture as MacroPosture
from evo_trader.macro.store import PostureStore
from evo_trader.portfolio.manager import (
    BlackoutWindow as PMBlackoutWindow,
    MarketPosture as PMPosture,
    PortfolioManager,
    PositionRequest,
)
from evo_trader.portfolio.sizing import position_fraction
from evo_trader.strategies.base import Bar, Position, Signal, Strategy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cold-start defaults used when a strategy has no edge metrics yet
# ---------------------------------------------------------------------------
_COLD_WIN_RATE = Decimal("0.5")
_COLD_PAYOFF_RATIO = Decimal("1.0")


# ---------------------------------------------------------------------------
# Client interfaces (injectable, mockable in tests)
# ---------------------------------------------------------------------------


class BarsClient(Protocol):
    """Provides historical price bars for a symbol."""

    def get_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        limit: int,
    ) -> list[Bar]:
        """Fetch recent price bars.

        Args:
            symbol: Tradeable instrument identifier.
            timeframe_minutes: Bar interval in minutes (e.g. 15 or 30).
            limit: Maximum number of bars to return (newest last).

        Returns:
            List of Bar objects in chronological order (oldest first).
        """
        ...


class OrderClient(Protocol):
    """Submits orders and queries open positions on the broker/exchange."""

    def submit_order(
        self,
        symbol: str,
        direction: str,
        notional: Decimal,
    ) -> str:
        """Submit a market order.

        Args:
            symbol: Tradeable instrument identifier.
            direction: "long" or "short".
            notional: Dollar-notional to deploy.

        Returns:
            Broker-assigned order ID string.
        """
        ...

    def get_open_positions(self, strategy_id: str) -> list[Position]:
        """Return all open positions tracked for a given strategy.

        Args:
            strategy_id: Strategy whose positions should be returned.

        Returns:
            List of Position objects (may be empty).
        """
        ...

    def close_position(
        self,
        strategy_id: str,
        symbol: str,
        reason: str,
    ) -> None:
        """Close an open position.

        Args:
            strategy_id: Owning strategy.
            symbol: Instrument to flatten.
            reason: Human-readable reason for the closure.
        """
        ...


# ---------------------------------------------------------------------------
# Internal per-strategy registration record
# ---------------------------------------------------------------------------


@dataclass
class _StrategyRegistration:
    strategy: Strategy
    agent_id: str
    trade_count: int = 0
    win_rate: Optional[Decimal] = None
    payoff_ratio: Optional[Decimal] = None


# ---------------------------------------------------------------------------
# Decision log entry schema
# ---------------------------------------------------------------------------


def _build_log_entry(
    *,
    strategy_id: str,
    symbol: str,
    cycle_type: str,
    direction: str,
    signal_strength: Optional[Decimal],
    posture: MacroPosture,
    fraction_computed: Optional[Decimal],
    notional_requested: Optional[Decimal],
    approved: Optional[bool],
    approved_notional: Optional[Decimal],
    approval_reason: Optional[str],
    order_id: Optional[str],
    rsi: Optional[Decimal] = None,
    z_score: Optional[Decimal] = None,
    stop_distance: Optional[Decimal] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a single decision-log JSON object."""
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "strategy_id": strategy_id,
        "symbol": symbol,
        "cycle_type": cycle_type,
        "direction": direction,
        "signal_strength": str(signal_strength) if signal_strength is not None else None,
        "posture_regime": posture.regime.value,
        "posture_risk_dial": str(posture.risk_dial),
        "fraction_computed": str(fraction_computed) if fraction_computed is not None else None,
        "notional_requested": str(notional_requested) if notional_requested is not None else None,
        "approved": approved,
        "approved_notional": str(approved_notional) if approved_notional is not None else None,
        "approval_reason": approval_reason,
        "order_id": order_id,
        "rsi": str(rsi) if rsi is not None else None,
        "z_score": str(z_score) if z_score is not None else None,
        "stop_distance": str(stop_distance) if stop_distance is not None else None,
        "extra": extra or {},
    }


# ---------------------------------------------------------------------------
# StrategyRunner
# ---------------------------------------------------------------------------


class StrategyRunner:
    """Orchestrates the per-bar tactical execution cycle.

    Each call to ``run_once()`` performs one full cycle across all registered
    strategies: exit management for open positions, then signal evaluation and
    order submission for new entries.

    All broker and market-data interactions are delegated to injected client
    objects so the runner is fully testable without live network access.

    Args:
        portfolio_manager: Enforces risk limits and approves positions.
        posture_store: Supplies the current macro posture.
        bars_client: Provides historical price bars.
        order_client: Submits orders and queries open positions.
        log_path: File path for the JSONL decision log.
        edge_metrics_source: Optional callable ``(strategy_id) -> (win_rate, payoff_ratio)``
            that supplies live edge metrics from the performance database.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        posture_store: PostureStore,
        bars_client: BarsClient,
        order_client: OrderClient,
        log_path: Path | str,
        edge_metrics_source: Optional[Any] = None,
    ) -> None:
        self._pm = portfolio_manager
        self._posture_store = posture_store
        self._bars_client = bars_client
        self._order_client = order_client
        self._log_path = Path(log_path)
        self._edge_metrics_source = edge_metrics_source
        self._registrations: dict[str, _StrategyRegistration] = {}

    def register_strategy(
        self,
        strategy: Strategy,
        agent_id: str,
        trade_count: int = 0,
        win_rate: Optional[Decimal] = None,
        payoff_ratio: Optional[Decimal] = None,
    ) -> None:
        """Register a strategy with its associated agent and optional edge metrics.

        Args:
            strategy: Concrete strategy instance.
            agent_id: ID of the TradingAgent that owns this strategy.
            trade_count: Completed trade count; controls cold-start vs Kelly path.
            win_rate: Historical win rate; uses cold-start default when None.
            payoff_ratio: Historical payoff ratio; uses cold-start default when None.
        """
        self._registrations[strategy.strategy_id] = _StrategyRegistration(
            strategy=strategy,
            agent_id=agent_id,
            trade_count=trade_count,
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
        )
        logger.info(
            "Registered strategy %s for agent %s (trades=%d)",
            strategy.strategy_id,
            agent_id,
            trade_count,
        )

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """Execute one full tactical cycle across all registered strategies.

        Processing order: 15-minute cadence strategies before 30-minute ones.
        """
        posture = self._posture_store.current_posture()
        now_utc = datetime.now(tz=timezone.utc)
        is_near_close = _is_near_market_close(now_utc)

        # Convert macro posture → PM posture (blackout windows need datetime objects)
        pm_posture = _macro_to_pm_posture(posture)

        # Sort: 15-min strategies first, then 30-min
        registrations_sorted = sorted(
            self._registrations.values(),
            key=lambda r: r.strategy.cadence_minutes,
        )

        for reg in registrations_sorted:
            strategy = reg.strategy
            # Skip 30-min strategies on 15-min ticks
            # (Simple gate: 30-min strategies only run when the minute is 0 or 30)
            if strategy.cadence_minutes == 30:
                minute = now_utc.minute
                if minute not in (0, 30):
                    self._append_log(
                        _build_log_entry(
                            strategy_id=strategy.strategy_id,
                            symbol=strategy.symbol,
                            cycle_type="skip",
                            direction="flat",
                            signal_strength=None,
                            posture=posture,
                            fraction_computed=None,
                            notional_requested=None,
                            approved=None,
                            approved_notional=None,
                            approval_reason="cadence_skip",
                            order_id=None,
                        )
                    )
                    continue

            bars = self._bars_client.get_bars(
                strategy.symbol,
                strategy.cadence_minutes,
                limit=200,
            )

            # --- Exit management ---
            open_positions = self._order_client.get_open_positions(strategy.strategy_id)
            last_bar = bars[-1] if bars else None
            for position in open_positions:
                if last_bar is None:
                    continue
                decision = strategy.check_exits(position, last_bar, is_near_close)
                if decision.should_exit:
                    self._order_client.close_position(
                        strategy.strategy_id,
                        position.symbol,
                        decision.reason,
                    )
                    self._append_log(
                        _build_log_entry(
                            strategy_id=strategy.strategy_id,
                            symbol=position.symbol,
                            cycle_type="exit",
                            direction=position.direction,
                            signal_strength=None,
                            posture=posture,
                            fraction_computed=None,
                            notional_requested=None,
                            approved=True,
                            approved_notional=None,
                            approval_reason=decision.reason,
                            order_id=None,
                            extra={"exit_reason": decision.reason},
                        )
                    )
                    logger.info(
                        "Closed %s %s — reason: %s",
                        strategy.strategy_id,
                        position.symbol,
                        decision.reason,
                    )

            # --- Entry signal ---
            if not bars:
                continue

            signal: Signal | None = strategy.generate_signal(bars)

            if signal is None or signal.direction == "flat":
                self._append_log(
                    _build_log_entry(
                        strategy_id=strategy.strategy_id,
                        symbol=strategy.symbol,
                        cycle_type="skip",
                        direction="flat",
                        signal_strength=Decimal("0"),
                        posture=posture,
                        fraction_computed=None,
                        notional_requested=None,
                        approved=None,
                        approved_notional=None,
                        approval_reason="no_signal",
                        order_id=None,
                    )
                )
                continue

            # Resolve edge metrics
            win_rate, payoff_ratio = self._resolve_edge_metrics(reg)

            fraction = position_fraction(
                win_rate=win_rate,
                payoff_ratio=payoff_ratio,
                risk_dial=posture.risk_dial,
                concentration_cap=strategy.params.kelly_cap,
                trade_count=reg.trade_count,
            )

            agent_allocation = self._pm.agent_allocation(reg.agent_id)
            notional = fraction * agent_allocation

            if notional <= Decimal("0"):
                self._append_log(
                    _build_log_entry(
                        strategy_id=strategy.strategy_id,
                        symbol=signal.symbol,
                        cycle_type="entry",
                        direction=signal.direction,
                        signal_strength=signal.strength,
                        posture=posture,
                        fraction_computed=fraction,
                        notional_requested=notional,
                        approved=False,
                        approved_notional=Decimal("0"),
                        approval_reason="zero_notional",
                        order_id=None,
                        stop_distance=signal.stop_distance,
                    )
                )
                continue

            request = PositionRequest(
                agent_id=reg.agent_id,
                asset=signal.symbol,
                notional=notional,
                direction=signal.direction,
            )
            pm_decision = self._pm.approve_position(request, posture=pm_posture)

            if not pm_decision.approved:
                self._append_log(
                    _build_log_entry(
                        strategy_id=strategy.strategy_id,
                        symbol=signal.symbol,
                        cycle_type="entry",
                        direction=signal.direction,
                        signal_strength=signal.strength,
                        posture=posture,
                        fraction_computed=fraction,
                        notional_requested=notional,
                        approved=False,
                        approved_notional=Decimal("0"),
                        approval_reason=pm_decision.reason,
                        order_id=None,
                        stop_distance=signal.stop_distance,
                    )
                )
                logger.info(
                    "Position denied for %s: %s", strategy.strategy_id, pm_decision.reason
                )
                continue

            try:
                order_id = self._order_client.submit_order(
                    signal.symbol,
                    signal.direction,
                    pm_decision.approved_notional,
                )
            except Exception as exc:
                logger.error(
                    "Order submission failed for %s: %s", strategy.strategy_id, exc
                )
                self._append_log(
                    _build_log_entry(
                        strategy_id=strategy.strategy_id,
                        symbol=signal.symbol,
                        cycle_type="entry",
                        direction=signal.direction,
                        signal_strength=signal.strength,
                        posture=posture,
                        fraction_computed=fraction,
                        notional_requested=notional,
                        approved=True,
                        approved_notional=pm_decision.approved_notional,
                        approval_reason=pm_decision.reason,
                        order_id=None,
                        stop_distance=signal.stop_distance,
                        extra={"order_error": str(exc)},
                    )
                )
                continue

            self._append_log(
                _build_log_entry(
                    strategy_id=strategy.strategy_id,
                    symbol=signal.symbol,
                    cycle_type="entry",
                    direction=signal.direction,
                    signal_strength=signal.strength,
                    posture=posture,
                    fraction_computed=fraction,
                    notional_requested=notional,
                    approved=True,
                    approved_notional=pm_decision.approved_notional,
                    approval_reason=pm_decision.reason,
                    order_id=order_id,
                    stop_distance=signal.stop_distance,
                )
            )
            logger.info(
                "Order submitted: %s %s %s notional=%s order_id=%s",
                signal.direction,
                signal.symbol,
                strategy.strategy_id,
                pm_decision.approved_notional,
                order_id,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_edge_metrics(
        self, reg: _StrategyRegistration
    ) -> tuple[Decimal, Decimal]:
        """Return (win_rate, payoff_ratio) for a strategy registration.

        Priority: live edge_metrics_source > registered values > cold defaults.
        """
        if self._edge_metrics_source is not None:
            try:
                result = self._edge_metrics_source(reg.strategy.strategy_id)
                if result is not None:
                    return result
            except Exception as exc:
                logger.warning(
                    "Edge metrics source failed for %s: %s",
                    reg.strategy.strategy_id,
                    exc,
                )

        win_rate = reg.win_rate if reg.win_rate is not None else _COLD_WIN_RATE
        payoff_ratio = (
            reg.payoff_ratio if reg.payoff_ratio is not None else _COLD_PAYOFF_RATIO
        )
        return win_rate, payoff_ratio

    def _append_log(self, entry: dict[str, Any]) -> None:
        """Append a JSON object as a single line to the decision log."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_near_market_close(now_utc: datetime) -> bool:
    """Return True when the current time (UTC) is within 15 minutes of 16:00 ET.

    Uses a fixed UTC offset: ET = UTC-4 (EDT, April–October) as an
    approximation.  A full timezone-aware implementation should use
    ``zoneinfo.ZoneInfo("America/New_York")``.
    """
    # Approximate ET offset: UTC-4 (EDT) for most of the trading year
    et_hour = (now_utc.hour - 4) % 24
    et_minute = now_utc.minute
    et_total_minutes = et_hour * 60 + et_minute
    close_total_minutes = 16 * 60  # 16:00 ET
    return close_total_minutes - 15 <= et_total_minutes < close_total_minutes


def _macro_to_pm_posture(macro: MacroPosture) -> PMPosture:
    """Convert a macro-layer MarketPosture to a portfolio-manager MarketPosture.

    The macro schema stores blackout windows as RFC3339 strings; the PM expects
    ``datetime`` objects.

    Args:
        macro: Macro-layer posture from PostureStore.

    Returns:
        PM-compatible MarketPosture.
    """
    pm_windows: list[PMBlackoutWindow] = []
    for w in macro.blackout_windows:
        try:
            start = datetime.fromisoformat(w.start)
            end = datetime.fromisoformat(w.end)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            pm_windows.append(PMBlackoutWindow(start=start, end=end, reason=w.reason))
        except (ValueError, TypeError) as exc:
            logger.warning("Could not parse blackout window %s: %s", w, exc)

    return PMPosture(
        risk_dial=macro.risk_dial,
        blackout_windows=pm_windows,
    )


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    """Parse CLI args and run one tactical cycle."""
    parser = argparse.ArgumentParser(
        description="Evolutionary trading system — tactical runner"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single tactical cycle and exit.",
    )
    parser.add_argument(
        "--log-path",
        default="state/decisions.jsonl",
        help="Path to the JSONL decision log.",
    )
    args = parser.parse_args()

    if not args.once:
        print(
            "No action flag supplied.  Use --once to run a single cycle.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Attempt to build real clients from environment variables
    # ------------------------------------------------------------------
    import os

    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    try:
        from evo_trader.tactical._alpaca_clients import (  # type: ignore[import]
            AlpacaOrderClient,
        )
        from evo_trader.tactical._yfinance_client import YFinanceBarsClient

        if not alpaca_key or not alpaca_secret:
            raise EnvironmentError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables are required"
                " for order execution."
            )

        # Market data: yfinance (free, no API key needed).
        bars_client: BarsClient = YFinanceBarsClient()
        # Order execution: Alpaca (still requires credentials).
        order_client: OrderClient = AlpacaOrderClient(
            api_key=alpaca_key,
            secret_key=alpaca_secret,
            base_url=alpaca_base_url,
        )
    except (ImportError, EnvironmentError) as exc:
        print(
            f"Cannot configure clients: {exc}\n"
            "Set ALPACA_API_KEY / ALPACA_SECRET_KEY to run against live/paper accounts.",
            file=sys.stderr,
        )
        sys.exit(1)

    from evo_trader.config.parameters import CapitalConfig
    from evo_trader.tactical.population import (
        init_population,
        register_starting_strategies,
    )

    portfolio_manager = PortfolioManager()
    agents = init_population(portfolio_manager, CapitalConfig())
    posture_store = PostureStore()
    log_path = Path(args.log_path)

    runner = StrategyRunner(
        portfolio_manager=portfolio_manager,
        posture_store=posture_store,
        bars_client=bars_client,
        order_client=order_client,
        log_path=log_path,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    register_starting_strategies(runner, agents, os.getenv("TACTICAL_SYMBOL", "SPY"))
    runner.run_once()
    print("Tactical cycle complete.")


if __name__ == "__main__":
    _main()
