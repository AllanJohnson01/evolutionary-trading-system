"""Tests for market selector and forex feature flag."""
import os

import pytest

from evo_trader.markets.selector import Market, MarketSelector


class TestMarketSelector:
    def test_crypto_pairs_by_default(self):
        sel = MarketSelector()
        pairs = sel.active_pairs()
        symbols = [p.symbol for p in pairs]
        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols
        assert "SOL/USDT" in symbols

    def test_forex_disabled_by_default(self):
        sel = MarketSelector()
        pairs = sel.active_pairs()
        markets = {p.market for p in pairs}
        assert Market.FOREX not in markets

    def test_forex_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("FOREX_ENABLED", "true")
        sel = MarketSelector()
        assert sel.is_forex_enabled()
        pairs = sel.active_pairs()
        markets = {p.market for p in pairs}
        assert Market.FOREX in markets

    def test_forex_env_false_overrides_config(self, monkeypatch):
        monkeypatch.setenv("FOREX_ENABLED", "false")
        from evo_trader.config.parameters import MarketConfig
        cfg = MarketConfig(forex_enabled=True)
        sel = MarketSelector(config=cfg)
        assert not sel.is_forex_enabled()

    def test_equities_raises(self):
        sel = MarketSelector()
        with pytest.raises(NotImplementedError, match="deferred"):
            sel.pairs_for_market(Market.EQUITIES)

    def test_asset_pair_parsing(self):
        sel = MarketSelector()
        pairs = sel.pairs_for_market(Market.CRYPTO)
        btc = next(p for p in pairs if p.symbol == "BTC/USDT")
        assert btc.base == "BTC"
        assert btc.quote == "USDT"

    def test_active_markets_default(self):
        sel = MarketSelector()
        assert sel.active_markets() == [Market.CRYPTO]
