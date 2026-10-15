"""Tests for HorizonConfig — verifies defaults and execution_mode validation."""
from decimal import Decimal

import pytest

from evo_trader.horizon.config import HorizonConfig


class TestHorizonConfigDefaults:
    def setup_method(self):
        self.cfg = HorizonConfig()

    def test_default_tau(self):
        assert self.cfg.tau == 10

    def test_default_t_c(self):
        assert self.cfg.t_c == Decimal("1.0")

    def test_default_tuning_cadence(self):
        assert self.cfg.tuning_cadence_bars == 20

    def test_default_execution_mode(self):
        assert self.cfg.execution_mode == "paper"

    def test_default_risk_limits(self):
        assert self.cfg.max_drawdown == Decimal("0.15")
        assert self.cfg.daily_loss_limit == Decimal("0.02")


class TestHorizonConfigValidation:
    def test_rejects_invalid_execution_mode(self):
        with pytest.raises(ValueError, match="execution_mode"):
            HorizonConfig(execution_mode="live_but_typo")

    def test_accepts_live_mode(self):
        cfg = HorizonConfig(execution_mode="live")
        assert cfg.execution_mode == "live"

    def test_rejects_non_positive_tau(self):
        with pytest.raises(ValueError, match="tau"):
            HorizonConfig(tau=0)

    def test_rejects_non_positive_tuning_cadence(self):
        with pytest.raises(ValueError, match="tuning_cadence_bars"):
            HorizonConfig(tuning_cadence_bars=0)
