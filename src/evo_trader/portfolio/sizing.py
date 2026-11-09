"""
Position sizing: half-Kelly formula and cold-start path.

Kelly inputs (win_rate, payoff_ratio) are always externally supplied by the
edge-metrics module — strategies never compute their own sizing inputs.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from evo_trader.config.parameters import SizingConfig


def kelly_fraction(win_rate: Decimal, payoff_ratio: Decimal) -> Decimal:
    """
    Standard Kelly criterion: f* = (b*p - q) / b
    where b = payoff_ratio, p = win_rate, q = 1 - p.

    Returns 0 if edge is zero or negative (never bet into a losing edge).

    Args:
        win_rate: Probability of a winning trade in [0, 1].
        payoff_ratio: Average win / average loss (must be > 0 for a valid bet).

    Returns:
        Kelly fraction as a Decimal, floored at 0.
    """
    if payoff_ratio <= 0 or win_rate <= 0:
        return Decimal("0")
    q = Decimal("1") - win_rate
    f = (payoff_ratio * win_rate - q) / payoff_ratio
    return max(Decimal("0"), f)


def position_fraction(
    win_rate: Decimal,
    payoff_ratio: Decimal,
    risk_dial: Decimal,
    concentration_cap: Decimal,
    trade_count: int,
    cfg: Optional[SizingConfig] = None,
) -> Decimal:
    """
    Returns the fraction of allocated capital to deploy for a single position.

    Cold-start path (trade_count < cold_start_threshold):
        fixed_risk x risk_dial

    Normal path:
        0.5 x kelly(win_rate, payoff_ratio), clamped to [0, concentration_cap], x risk_dial

    Args:
        win_rate: Historical win rate for this strategy.
        payoff_ratio: Historical payoff ratio (avg win / avg loss).
        risk_dial: Portfolio-level risk scalar in [0, 1]; 1.0 = full risk on.
        concentration_cap: Maximum fraction of agent capital for a single position.
        trade_count: Number of completed trades (determines cold-start vs. normal path).
        cfg: Optional SizingConfig; defaults to SizingConfig() if None.

    Returns:
        Fraction of allocated capital to deploy, as a Decimal.
    """
    cfg = cfg or SizingConfig()

    if trade_count < cfg.cold_start_threshold:
        return cfg.cold_start_fixed_risk * risk_dial

    raw = cfg.half_kelly_fraction * kelly_fraction(win_rate, payoff_ratio)
    clamped = min(max(Decimal("0"), raw), concentration_cap)
    return clamped * risk_dial
