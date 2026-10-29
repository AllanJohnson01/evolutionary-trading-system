"""
Edge metrics and signal-decay module.

Reads a JSONL trade log (one closed-trade record per line) and computes
rolling, regime-tagged per-strategy metrics used by:
  - Kelly sizing inputs (the ONLY sanctioned source)
  - Evolution engine evaluation (risk-adjusted ranking)
  - Weekly system review (decay flags, regime breakdown)

Trade log schema (each line is JSON):
  {
    "strategy_id": str,
    "trade_id": str,
    "regime": str,          # e.g. "risk_on" | "neutral" | "risk_off"
    "closed_at": str,       # ISO8601
    "pnl": float,           # realised PnL in dollar terms
    "entry_price": float,
    "exit_price": float,
    "side": "long" | "short"
  }

All metric computation is pure Python over the JSONL data; no external
dependencies beyond the standard library are required.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """A single closed trade as read from the JSONL log."""
    strategy_id: str
    trade_id: str
    regime: str
    closed_at: str
    pnl: float
    entry_price: float
    exit_price: float
    side: str  # "long" | "short"

    @classmethod
    def from_dict(cls, d: dict) -> "TradeRecord":
        return cls(
            strategy_id=str(d["strategy_id"]),
            trade_id=str(d.get("trade_id", "")),
            regime=str(d.get("regime", "neutral")),
            closed_at=str(d.get("closed_at", "")),
            pnl=float(d["pnl"]),
            entry_price=float(d.get("entry_price", 0.0)),
            exit_price=float(d.get("exit_price", 0.0)),
            side=str(d.get("side", "long")),
        )


@dataclass
class EdgeMetrics:
    """
    Rolling metrics snapshot for one strategy over one evaluation window.

    All slope fields are in units-per-sub-window (positive = improving,
    negative = decaying). None means insufficient data.
    """
    strategy_id: str
    trade_count: int

    hit_rate: Optional[float]              # fraction of trades that were winners
    profit_factor: Optional[float]         # gross_wins / gross_losses
    sharpe_per_trade: Optional[float]      # mean(pnl) / std(pnl), per-trade
    sharpe_daily: Optional[float]          # annualised daily Sharpe (NaN → None)
    avg_edge: Optional[float]              # mean(pnl) — average edge per trade

    # Decay detection: linear slope of each metric across trend_window_count sub-windows
    hit_rate_slope: Optional[float]
    profit_factor_slope: Optional[float]
    sharpe_slope: Optional[float]
    avg_edge_slope: Optional[float]

    # Zero-edge flag: True when avg_edge is statistically indistinguishable from 0
    zero_edge: bool

    # Regime breakdown: metric values keyed by regime string
    regime_hit_rate: dict[str, float] = field(default_factory=dict)
    regime_avg_edge: dict[str, float] = field(default_factory=dict)


@dataclass
class KellyInputs:
    """
    The ONLY sanctioned source of win_rate / payoff_ratio for position sizing.

    win_rate: fraction of last-N closed trades that were wins
    payoff_ratio: avg_win_pnl / avg_loss_pnl (absolute), over last-N trades
    trade_count: number of closed trades used
    """
    strategy_id: str
    win_rate: float
    payoff_ratio: float
    trade_count: int


@dataclass
class MetricsSummary:
    """Full summary for one strategy, as consumed by weekly review and Evolution-Controller."""
    strategy_id: str
    metrics: EdgeMetrics
    kelly_inputs: KellyInputs
    # Consecutive periods with zero_edge flagged (set by caller tracking state)
    consecutive_zero_edge_periods: int = 0
    flagged_for_elimination: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_slope(values: Sequence[float]) -> Optional[float]:
    """Least-squares slope for evenly-spaced y values. Returns None if < 2 points."""
    n = len(values)
    if n < 2:
        return None
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def _t_stat_mean(values: Sequence[float]) -> Optional[float]:
    """One-sample t-stat: mean / (std / sqrt(n)). Returns None if std == 0 or n < 2."""
    n = len(values)
    if n < 2:
        return None
    m = statistics.mean(values)
    s = statistics.stdev(values)
    if s == 0:
        return None
    return m / (s / math.sqrt(n))


def _sharpe_per_trade(pnls: Sequence[float]) -> Optional[float]:
    if len(pnls) < 2:
        return None
    m = statistics.mean(pnls)
    s = statistics.stdev(pnls)
    if s == 0:
        return None
    return m / s


def _sharpe_daily(pnls: Sequence[float], annualise: float = 252.0) -> Optional[float]:
    """Daily Sharpe from daily-bucketed PnL. Returns None if fewer than 2 days."""
    if len(pnls) < 2:
        return None
    s = _sharpe_per_trade(pnls)
    if s is None:
        return None
    return s * math.sqrt(annualise)


def _profit_factor(pnls: Sequence[float]) -> Optional[float]:
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    if gross_losses == 0:
        return None if gross_wins == 0 else float("inf")
    return gross_wins / gross_losses


def _hit_rate(pnls: Sequence[float]) -> Optional[float]:
    if not pnls:
        return None
    return sum(1 for p in pnls if p > 0) / len(pnls)


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------

class EdgeCalculator:
    """
    Computes rolling edge metrics for all strategies from a JSONL trade log.

    Usage:
        calc = EdgeCalculator(log_path="state/trade_log.jsonl")
        metrics = calc.compute("strategy_momentum", window=30)
        kelly = calc.kelly_inputs("strategy_momentum")
        summary = calc.summary("strategy_momentum")
    """

    def __init__(
        self,
        log_path: str | Path,
        cfg=None,
    ) -> None:
        from evo_trader.config.parameters import EdgeMetricsConfig
        self._cfg: EdgeMetricsConfig = cfg or EdgeMetricsConfig()
        self._trades: list[TradeRecord] = self._load(Path(log_path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        strategy_id: str,
        window: Optional[int] = None,
    ) -> EdgeMetrics:
        """
        Compute rolling metrics for strategy_id over the last `window` closed trades.
        Defaults to cfg.rolling_window_trades.
        """
        window = window or self._cfg.rolling_window_trades
        trades = self._last_n(strategy_id, window)
        return self._compute_metrics(strategy_id, trades)

    def kelly_inputs(
        self,
        strategy_id: str,
        window: int = 30,
    ) -> KellyInputs:
        """
        Return (win_rate, payoff_ratio, trade_count) over last `window` closed trades.

        This is the ONLY sanctioned source for position_fraction inputs.
        Calling code must NOT compute these values itself.
        """
        trades = self._last_n(strategy_id, window)
        pnls = [t.pnl for t in trades]
        n = len(pnls)

        if n == 0:
            return KellyInputs(strategy_id=strategy_id, win_rate=0.0, payoff_ratio=0.0, trade_count=0)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / n
        avg_win = statistics.mean(wins) if wins else 0.0
        avg_loss = abs(statistics.mean(losses)) if losses else 0.0
        payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        return KellyInputs(
            strategy_id=strategy_id,
            win_rate=win_rate,
            payoff_ratio=payoff_ratio,
            trade_count=n,
        )

    def summary(
        self,
        strategy_id: str,
        consecutive_zero_edge_periods: int = 0,
    ) -> MetricsSummary:
        """
        Full summary for weekly review and Evolution-Controller.
        consecutive_zero_edge_periods must be tracked externally (e.g. by EvolutionEngine).
        """
        metrics = self.compute(strategy_id)
        ki = self.kelly_inputs(strategy_id)
        flagged = (
            metrics.zero_edge
            and consecutive_zero_edge_periods >= self._cfg.zero_edge_consecutive_periods
        )
        return MetricsSummary(
            strategy_id=strategy_id,
            metrics=metrics,
            kelly_inputs=ki,
            consecutive_zero_edge_periods=consecutive_zero_edge_periods,
            flagged_for_elimination=flagged,
        )

    def all_strategy_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for t in self._trades:
            seen[t.strategy_id] = None
        return list(seen)

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> list[TradeRecord]:
        if not path.exists():
            return []
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(TradeRecord.from_dict(json.loads(line)))
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
        return records

    def _last_n(self, strategy_id: str, n: int) -> list[TradeRecord]:
        filtered = [t for t in self._trades if t.strategy_id == strategy_id]
        return filtered[-n:] if len(filtered) > n else filtered

    def _compute_metrics(
        self,
        strategy_id: str,
        trades: list[TradeRecord],
    ) -> EdgeMetrics:
        pnls = [t.pnl for t in trades]
        n = len(pnls)
        min_n = self._cfg.min_trades_for_metrics

        if n < min_n:
            return EdgeMetrics(
                strategy_id=strategy_id,
                trade_count=n,
                hit_rate=None,
                profit_factor=None,
                sharpe_per_trade=None,
                sharpe_daily=None,
                avg_edge=None,
                hit_rate_slope=None,
                profit_factor_slope=None,
                sharpe_slope=None,
                avg_edge_slope=None,
                zero_edge=False,
            )

        hr = _hit_rate(pnls)
        pf = _profit_factor(pnls)
        spt = _sharpe_per_trade(pnls)
        avg_e = statistics.mean(pnls)

        # Daily Sharpe: bucket trades by date prefix (YYYY-MM-DD)
        daily_buckets: dict[str, list[float]] = {}
        for t in trades:
            day = t.closed_at[:10] if len(t.closed_at) >= 10 else "unknown"
            daily_buckets.setdefault(day, []).append(t.pnl)
        daily_pnls = [sum(v) for v in daily_buckets.values()]
        sd = _sharpe_daily(daily_pnls, self._cfg.sharpe_annualise_factor)

        # Trend / decay slopes across trend_window_count sub-windows
        wc = self._cfg.trend_window_count
        sub_size = max(1, n // wc)
        sub_windows = [trades[i * sub_size:(i + 1) * sub_size] for i in range(wc)]
        sub_windows = [w for w in sub_windows if len(w) >= 2]

        hr_slope = pf_slope = sh_slope = ae_slope = None
        if len(sub_windows) >= 2:
            hr_vals = [_hit_rate([t.pnl for t in w]) or 0.0 for w in sub_windows]
            pf_vals = [_profit_factor([t.pnl for t in w]) or 0.0 for w in sub_windows]
            sh_vals = [_sharpe_per_trade([t.pnl for t in w]) or 0.0 for w in sub_windows]
            ae_vals = [statistics.mean([t.pnl for t in w]) for w in sub_windows]
            hr_slope = _linear_slope(hr_vals)
            pf_slope = _linear_slope(pf_vals)
            sh_slope = _linear_slope(sh_vals)
            ae_slope = _linear_slope(ae_vals)

        # Zero-edge detection via t-test on average edge
        t_stat = _t_stat_mean(pnls)
        zero_edge = (t_stat is None) or (abs(t_stat) < self._cfg.zero_edge_t_threshold)

        # Regime breakdown
        regimes: dict[str, list[float]] = {}
        for t in trades:
            regimes.setdefault(t.regime, []).append(t.pnl)

        regime_hr: dict[str, float] = {}
        regime_ae: dict[str, float] = {}
        for r, rpnls in regimes.items():
            rh = _hit_rate(rpnls)
            if rh is not None:
                regime_hr[r] = rh
            regime_ae[r] = statistics.mean(rpnls)

        return EdgeMetrics(
            strategy_id=strategy_id,
            trade_count=n,
            hit_rate=hr,
            profit_factor=pf,
            sharpe_per_trade=spt,
            sharpe_daily=sd,
            avg_edge=avg_e,
            hit_rate_slope=hr_slope,
            profit_factor_slope=pf_slope,
            sharpe_slope=sh_slope,
            avg_edge_slope=ae_slope,
            zero_edge=zero_edge,
            regime_hit_rate=regime_hr,
            regime_avg_edge=regime_ae,
        )
