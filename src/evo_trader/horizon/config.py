"""
Horizon's own configuration — deliberately separate from
evo_trader.config.parameters. PortfolioManager and EvolutionEngine must
never read this module; Horizon's isolation depends on that staying true.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

VALID_EXECUTION_MODES = ("paper", "live")


@dataclass
class HorizonConfig:
    """
    tau: rollout horizon, in bars.
    t_c: optionality-objective strength (how much the entropic-force score
         dominates vs. hard constraints — constraints always filter first,
         regardless of t_c).
    tuning_cadence_bars: how often the Jev tuner re-evaluates tau/t_c.
    execution_mode: "paper" (tuner adjustments auto-apply, unbounded) or
         "live" (every adjustment requires human approval before applying).
    max_drawdown / daily_loss_limit: Horizon's own hard risk limits — not
         shared with evo_trader.config.parameters.RiskConfig, so Horizon can
         diverge from the shared agents' limits without touching that module.
    """
    tau: int = 10
    t_c: Decimal = Decimal("1.0")
    tuning_cadence_bars: int = 20
    execution_mode: str = "paper"
    max_drawdown: Decimal = Decimal("0.15")
    daily_loss_limit: Decimal = Decimal("0.02")

    def __post_init__(self) -> None:
        if self.execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {VALID_EXECUTION_MODES}, "
                f"got {self.execution_mode!r}"
            )
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")
        if self.tuning_cadence_bars <= 0:
            raise ValueError(
                f"tuning_cadence_bars must be positive, got {self.tuning_cadence_bars}"
            )
