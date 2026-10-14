"""Tests for HorizonLedger — standalone NAV/drawdown tracking, structurally
independent of PortfolioManager. Mirrors PortfolioManager's drawdown formula
((high_water_mark - nav) / high_water_mark) but as Horizon's own instance."""
from decimal import Decimal

from evo_trader.horizon.ledger import HorizonLedger


class TestHorizonLedgerInitialState:
    def test_nav_starts_at_capital(self):
        ledger = HorizonLedger(starting_capital=Decimal("10000"))
        assert ledger.nav == Decimal("10000")

    def test_drawdown_starts_at_zero(self):
        ledger = HorizonLedger(starting_capital=Decimal("10000"))
        assert ledger.drawdown == Decimal("0")


class TestRecordFill:
    def test_gain_increases_nav_and_high_water_mark(self):
        ledger = HorizonLedger(starting_capital=Decimal("10000"))
        ledger.record_fill(realized=Decimal("500"), unrealized=Decimal("0"))
        assert ledger.nav == Decimal("10500")
        assert ledger.drawdown == Decimal("0")

    def test_loss_after_gain_computes_drawdown_from_high_water_mark(self):
        ledger = HorizonLedger(starting_capital=Decimal("10000"))
        ledger.record_fill(realized=Decimal("1000"), unrealized=Decimal("0"))  # NAV 11000, HWM 11000
        ledger.record_fill(realized=Decimal("-550"), unrealized=Decimal("0"))  # NAV 10450
        # drawdown = (11000 - 10450) / 11000
        assert ledger.drawdown == (Decimal("11000") - Decimal("10450")) / Decimal("11000")

    def test_high_water_mark_does_not_fall_on_loss(self):
        ledger = HorizonLedger(starting_capital=Decimal("10000"))
        ledger.record_fill(realized=Decimal("1000"), unrealized=Decimal("0"))
        ledger.record_fill(realized=Decimal("-2000"), unrealized=Decimal("0"))
        ledger.record_fill(realized=Decimal("100"), unrealized=Decimal("0"))
        # HWM should still be 11000 from the first gain, not reset by the loss
        assert ledger.nav == Decimal("9100")
        assert ledger.drawdown == (Decimal("11000") - Decimal("9100")) / Decimal("11000")


class TestDailyLossPct:
    def test_zero_pnl_is_zero_pct(self):
        ledger = HorizonLedger(starting_capital=Decimal("10000"))
        pct = ledger.daily_loss_pct(
            today_realized=Decimal("0"), today_unrealized=Decimal("0"),
            starting_capital=Decimal("10000"),
        )
        assert pct == Decimal("0")

    def test_loss_is_positive_pct_of_starting_capital(self):
        ledger = HorizonLedger(starting_capital=Decimal("10000"))
        pct = ledger.daily_loss_pct(
            today_realized=Decimal("-300"), today_unrealized=Decimal("0"),
            starting_capital=Decimal("10000"),
        )
        assert pct == Decimal("0.03")

    def test_gain_is_zero_pct_not_negative(self):
        # A gain should never read as a "loss percentage" — must not go negative.
        ledger = HorizonLedger(starting_capital=Decimal("10000"))
        pct = ledger.daily_loss_pct(
            today_realized=Decimal("300"), today_unrealized=Decimal("0"),
            starting_capital=Decimal("10000"),
        )
        assert pct == Decimal("0")


class TestZeroCapitalEdgeCase:
    def test_drawdown_with_zero_high_water_mark_is_zero_not_divide_error(self):
        # Defensive: starting_capital=0 would otherwise divide by zero.
        ledger = HorizonLedger(starting_capital=Decimal("0"))
        assert ledger.drawdown == Decimal("0")
