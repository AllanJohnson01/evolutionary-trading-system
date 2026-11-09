"""
Volatility-scaled stop distances.

Uses EWMA realized volatility (not Gaussian assumptions) with a configurable
sigma multiplier to set stop distances that reflect actual fat-tail risk.
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Sequence

from evo_trader.config.parameters import SizingConfig


def ewma_volatility(
    returns: Sequence[Decimal],
    span: int,
) -> Decimal:
    """
    Compute EWMA realized volatility from a sequence of bar returns.

    Uses the standard EWMA formula: variance_t = (1-alpha)*r_t^2 + alpha*variance_{t-1}
    where alpha = 1 - 2/(span+1).

    Seeded with the squared first return. The remaining returns update the
    running variance iteratively.

    Args:
        returns: Sequence of bar-level returns as Decimals (e.g. 0.01 = 1%).
        span: EWMA span in bars; controls the decay rate alpha.

    Returns:
        EWMA realized volatility (std dev) as a Decimal. Returns 0 if fewer
        than 2 returns are supplied.
    """
    if len(returns) < 2:
        return Decimal("0")

    alpha = Decimal("2") / (Decimal(span) + Decimal("1"))
    one_minus_alpha = Decimal("1") - alpha

    # Seed with the squared first return
    variance = returns[0] ** 2
    for r in returns[1:]:
        variance = one_minus_alpha * variance + alpha * (r ** 2)

    variance_float = float(variance)
    if variance_float <= 0:
        return Decimal("0")
    # Cast back to Decimal immediately — no floats escape this function
    return Decimal(str(math.sqrt(variance_float)))


def stop_distance(
    returns: Sequence[Decimal],
    cfg: SizingConfig | None = None,
) -> Decimal:
    """
    Compute stop distance as sigma_multiplier x EWMA_volatility.

    This is the fraction of price to move against entry before the position
    is considered stopped out. Returns 0 if volatility cannot be estimated.

    Args:
        returns: Sequence of bar-level returns used to estimate realized vol.
        cfg: Optional SizingConfig; defaults to SizingConfig() if None.

    Returns:
        Stop distance as a fraction of price (Decimal). Multiply by entry
        price to get the absolute stop level.
    """
    cfg = cfg or SizingConfig()
    vol = ewma_volatility(returns, span=cfg.ewma_span)
    return cfg.stop_sigma_multiplier * vol
