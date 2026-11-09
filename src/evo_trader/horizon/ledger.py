"""
Horizon's standalone paper ledger. Deliberately does NOT import from
evo_trader.portfolio.manager — Horizon's isolation depends on this module
having zero code path back into PortfolioManager's buckets.
"""
from __future__ import annotations

from decimal import Decimal


class HorizonLedger:
    def __init__(self, starting_capital: Decimal) -> None:
        if starting_capital <= 0:
            raise ValueError(
                f"starting_capital must be positive, got {starting_capital}"
            )
        self._nav: Decimal = starting_capital
        self._high_water_mark: Decimal = starting_capital

    def record_fill(self, realized: Decimal, unrealized: Decimal) -> None:
        """Applies a fill's P&L to NAV and updates the high-water mark.

        NAV moves only on ``realized``, matching the realized-only NAV
        convention of the existing ``PortfolioManager.record_pnl``.
        ``unrealized`` is accepted for interface symmetry with that method but
        currently has no effect on NAV. This is asymmetric with
        ``daily_loss_pct``, which does include unrealized P&L, via its own
        separate parameters rather than through this method.
        """
        self._nav += realized
        if self._nav > self._high_water_mark:
            self._high_water_mark = self._nav

    @property
    def nav(self) -> Decimal:
        return self._nav

    @property
    def drawdown(self) -> Decimal:
        if self._high_water_mark == 0:
            return Decimal("0")
        return (self._high_water_mark - self._nav) / self._high_water_mark

    def daily_loss_pct(
        self,
        today_realized: Decimal,
        today_unrealized: Decimal,
        starting_capital: Decimal,
    ) -> Decimal:
        """Returns the day's loss as a positive fraction of starting_capital,
        or exactly 0 for a flat or positive day — never negative."""
        if starting_capital == 0:
            return Decimal("0")
        pnl = today_realized + today_unrealized
        if pnl >= 0:
            return Decimal("0")
        return abs(pnl) / starting_capital
