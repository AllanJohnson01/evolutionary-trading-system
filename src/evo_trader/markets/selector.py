"""
Market selector — returns the tradeable asset pairs based on feature flags.

Crypto: always enabled.
Forex: gated behind FOREX_ENABLED env var (Phase 2).
Equities: deferred — not in this phase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

from evo_trader.config.parameters import MarketConfig


class Market(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    EQUITIES = "equities"


@dataclass(frozen=True)
class AssetPair:
    symbol: str
    market: Market
    base: str
    quote: str

    @classmethod
    def from_symbol(cls, symbol: str, market: Market) -> "AssetPair":
        parts = symbol.split("/")
        if len(parts) != 2:
            raise ValueError(f"Expected 'BASE/QUOTE', got: {symbol!r}")
        return cls(symbol=symbol, market=market, base=parts[0], quote=parts[1])


class MarketSelector:
    """
    Returns currently active asset pairs based on configuration and env flags.

    Forex is unlocked by setting the environment variable FOREX_ENABLED=true.
    Equities are deferred and will raise NotImplementedError.
    """

    def __init__(self, config: MarketConfig | None = None) -> None:
        self._cfg = config or MarketConfig()

    def is_forex_enabled(self) -> bool:
        """
        Forex is gated. Override in config OR set FOREX_ENABLED=true in env.
        Env var takes precedence over config to allow runtime unlocking.
        """
        env_val = os.environ.get("FOREX_ENABLED", "").lower()
        if env_val in ("1", "true", "yes"):
            return True
        if env_val in ("0", "false", "no"):
            return False
        return self._cfg.forex_enabled

    def active_pairs(self) -> list[AssetPair]:
        """Return all currently tradeable pairs."""
        pairs: list[AssetPair] = []

        if self._cfg.crypto_enabled:
            for sym in self._cfg.crypto_pairs:
                pairs.append(AssetPair.from_symbol(sym, Market.CRYPTO))

        if self.is_forex_enabled():
            for sym in self._cfg.forex_pairs:
                pairs.append(AssetPair.from_symbol(sym, Market.FOREX))

        return pairs

    def pairs_for_market(self, market: Market) -> list[AssetPair]:
        if market == Market.EQUITIES:
            raise NotImplementedError("Equities are deferred to a future phase")
        return [p for p in self.active_pairs() if p.market == market]

    def active_markets(self) -> list[Market]:
        markets: list[Market] = []
        if self._cfg.crypto_enabled:
            markets.append(Market.CRYPTO)
        if self.is_forex_enabled():
            markets.append(Market.FOREX)
        return markets
