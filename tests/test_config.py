"""Tests for parameter config — verify all Frogman-Architect locked values."""
from decimal import Decimal

import pytest

from evo_trader.config.parameters import (
    CapitalConfig,
    EvolutionConfig,
    MarketConfig,
    RiskConfig,
)


class TestCapitalConfig:
    def setup_method(self):
        self.cfg = CapitalConfig()

    def test_total_capital(self):
        assert self.cfg.total_capital == Decimal("100000")

    def test_bucket_fractions_sum_to_one(self):
        total = (
            self.cfg.active_trading_fraction
            + self.cfg.dca_fraction
            + self.cfg.cash_reserve_fraction
        )
        assert total == Decimal("1.0")

    def test_active_trading_total(self):
        assert self.cfg.active_trading_total == Decimal("60000")

    def test_dca_total(self):
        assert self.cfg.dca_total == Decimal("25000")

    def test_cash_reserve_total(self):
        assert self.cfg.cash_reserve_total == Decimal("15000")

    def test_initial_agent_allocation(self):
        # 4 agents, equal split of $60K
        assert self.cfg.initial_agent_allocation == Decimal("15000")


class TestRiskConfig:
    def setup_method(self):
        self.cfg = RiskConfig()

    def test_max_drawdown(self):
        assert self.cfg.max_drawdown == Decimal("0.15")

    def test_daily_loss_limit(self):
        assert self.cfg.daily_loss_limit == Decimal("0.02")

    def test_per_agent_allocation_cap(self):
        assert self.cfg.per_agent_allocation_cap == Decimal("0.20")

    def test_per_asset_concentration(self):
        assert self.cfg.per_asset_concentration == Decimal("0.15")


class TestEvolutionConfig:
    def setup_method(self):
        self.cfg = EvolutionConfig()

    def test_rebalance_cadence(self):
        assert self.cfg.rebalance_cadence_days == 7

    def test_evaluation_window(self):
        assert self.cfg.evaluation_window_weeks == 4
        assert self.cfg.evaluation_window_days == 28

    def test_elimination_threshold(self):
        assert self.cfg.elimination_threshold == Decimal("0.25")

    def test_elimination_consecutive_periods(self):
        assert self.cfg.elimination_consecutive_periods == 2

    def test_clone_threshold(self):
        assert self.cfg.clone_threshold == Decimal("0.25")

    def test_clone_consecutive_periods(self):
        assert self.cfg.clone_consecutive_periods == 2

    def test_mutation_rate_range(self):
        assert self.cfg.mutation_rate_min == Decimal("0.10")
        assert self.cfg.mutation_rate_max == Decimal("0.20")

    def test_max_agent_count(self):
        assert self.cfg.max_agent_count == 8

    def test_initial_agent_count(self):
        assert self.cfg.initial_agent_count == 4


class TestMarketConfig:
    def setup_method(self):
        self.cfg = MarketConfig()

    def test_crypto_enabled_by_default(self):
        assert self.cfg.crypto_enabled is True

    def test_forex_disabled_by_default(self):
        assert self.cfg.forex_enabled is False

    def test_crypto_pairs(self):
        assert "BTC/USDT" in self.cfg.crypto_pairs
        assert "ETH/USDT" in self.cfg.crypto_pairs
        assert "SOL/USDT" in self.cfg.crypto_pairs

    def test_forex_pairs(self):
        assert "EUR/USD" in self.cfg.forex_pairs
        assert "GBP/USD" in self.cfg.forex_pairs
