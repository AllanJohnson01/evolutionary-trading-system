"""
Evolution engine — runs evaluation cycles, eliminates losers,
clones winners with mutation, and keeps the population bounded.
"""
from __future__ import annotations

import logging
import random
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Optional

from evo_trader.agents.base import AgentState, AgentType, TradingAgent, PerformanceRecord
from evo_trader.config.parameters import EvolutionConfig
from evo_trader.evolution.transaction_log import (
    EvolutionEvent,
    EventType,
    EvolutionLog,
)
from evo_trader.portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)


def _ranking_score(record: Optional[PerformanceRecord]) -> float:
    """
    Risk-adjusted ranking key.

    Prefers Sharpe when available (risk-adjusted edge); falls back to pnl_pct
    so legacy callers that don't supply Sharpe still work correctly.
    """
    if record is None:
        return 0.0
    if record.sharpe is not None:
        return record.sharpe
    return float(record.pnl_pct)


class EvolutionEngine:
    """
    Runs one evaluation period at a time. Caller drives the cadence
    (e.g., every 7 days); this class handles the ranking and decisions.

    All capital moves go through portfolio_manager — this engine only
    decides WHO gets eliminated/cloned and tells the portfolio manager
    to act.

    Edge-metrics integration:
        - Ranking uses per-period Sharpe (risk-adjusted edge) when the
          PerformanceRecord carries it; falls back to pnl_pct otherwise.
        - A zero_edge flag on the record increments a per-agent counter;
          agents with zero_edge for >=cfg consecutive periods are flagged
          for elimination regardless of PnL rank.
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        evolution_cfg: Optional[EvolutionConfig] = None,
        log_path: Path | str = "evolution_log.jsonl",
    ) -> None:
        self._pm = portfolio_manager
        self._cfg = evolution_cfg or EvolutionConfig()
        self._log = EvolutionLog(log_path)
        self._period_number: int = 0

        # Consecutive-period counters: agent_id → count of bottom/top periods
        self._consecutive_bottom: dict[str, int] = {}
        self._consecutive_top: dict[str, int] = {}

        # Zero-edge streak counters: agent_id → consecutive periods with zero_edge=True
        self._consecutive_zero_edge: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public: zero-edge streak read-out (for MetricsSummary / weekly review)
    # ------------------------------------------------------------------

    def zero_edge_streak(self, agent_id: str) -> int:
        """Current consecutive zero-edge period count for an agent."""
        return self._consecutive_zero_edge.get(agent_id, 0)

    # ------------------------------------------------------------------
    # Main entry point — call once per evaluation period
    # ------------------------------------------------------------------

    def run_period(
        self, period_record: dict[str, PerformanceRecord]
    ) -> dict[str, str]:
        """
        Evaluate all active agents for this period.

        Args:
            period_record: agent_id → PerformanceRecord for the period just ended.
                           Set PerformanceRecord.sharpe and .zero_edge from the
                           edge-metrics module before calling for full ranking.

        Returns:
            dict of agent_id → decision ("eliminated" | "cloned" | "active")
        """
        self._period_number += 1
        active = self._pm.active_agents()
        if not active:
            logger.warning("No active agents to evaluate")
            return {}

        # Attach period records to agents
        for agent in active:
            if agent.agent_id in period_record:
                agent.record_period(period_record[agent.agent_id])

        # Update zero-edge streaks
        for agent in active:
            rec = period_record.get(agent.agent_id)
            if rec is not None and rec.zero_edge:
                self._consecutive_zero_edge[agent.agent_id] = (
                    self._consecutive_zero_edge.get(agent.agent_id, 0) + 1
                )
            else:
                self._consecutive_zero_edge[agent.agent_id] = 0

        # Rank by risk-adjusted edge (Sharpe when available, else pnl_pct)
        ranked = sorted(
            active,
            key=lambda a: _ranking_score(period_record.get(a.agent_id)),
            reverse=True,
        )
        total = len(ranked)

        self._log_period_evaluation(ranked, period_record, total)

        decisions: dict[str, str] = {}

        # Identify bottom and top performers
        bottom_n = max(1, int(total * float(self._cfg.elimination_threshold)))
        top_n = max(1, int(total * float(self._cfg.clone_threshold)))

        bottom_ids = {a.agent_id for a in ranked[-bottom_n:]}
        top_ids = {a.agent_id for a in ranked[:top_n]}

        # Update consecutive bottom/top counters
        for agent in active:
            aid = agent.agent_id
            if aid in bottom_ids:
                self._consecutive_bottom[aid] = self._consecutive_bottom.get(aid, 0) + 1
                self._consecutive_top[aid] = 0
            elif aid in top_ids:
                self._consecutive_top[aid] = self._consecutive_top.get(aid, 0) + 1
                self._consecutive_bottom[aid] = 0
            else:
                self._consecutive_bottom[aid] = 0
                self._consecutive_top[aid] = 0

        # Agents flagged for zero-edge regardless of PnL rank
        zero_edge_condemned: set[str] = {
            agent.agent_id
            for agent in active
            if self._consecutive_zero_edge.get(agent.agent_id, 0)
            >= self._cfg.elimination_consecutive_periods
        }

        # Eliminate: bottom performers with enough consecutive bad periods,
        # plus any zero-edge condemned agents not already in bottom
        eliminated_ids: set[str] = set()

        for agent in ranked[-bottom_n:]:
            aid = agent.agent_id
            consec = self._consecutive_bottom.get(aid, 0)
            if consec >= self._cfg.elimination_consecutive_periods:
                self._eliminate(agent, period_record.get(aid), ranked.index(agent), total,
                                reason="bottom_pnl")
                eliminated_ids.add(aid)
                decisions[aid] = "eliminated"

        for agent in active:
            aid = agent.agent_id
            if aid in eliminated_ids:
                continue
            if aid in zero_edge_condemned:
                self._eliminate(agent, period_record.get(aid), ranked.index(agent), total,
                                reason="zero_edge")
                eliminated_ids.add(aid)
                decisions[aid] = "eliminated"

        # Clone top performers (only if under max agent count and not already eliminated)
        current_active_count = len([a for a in active if a.agent_id not in eliminated_ids])
        new_clones: list[TradingAgent] = []

        for agent in ranked[:top_n]:
            aid = agent.agent_id
            if aid in eliminated_ids:
                continue
            consec = self._consecutive_top.get(aid, 0)
            if consec >= self._cfg.clone_consecutive_periods:
                if current_active_count + len(new_clones) < self._cfg.max_agent_count:
                    clone = self._clone(agent, period_record.get(aid), ranked.index(agent), total)
                    new_clones.append(clone)
                    decisions[aid] = "cloned"
                    self._consecutive_top[aid] = 0

        # Register clones with portfolio manager
        for clone in new_clones:
            seed_amount = self._pm.release_reserve_for_clone(
                Decimal("5000")  # $5K seed from cash reserve per clone
            )
            if seed_amount > 0:
                self._pm.register_agent(clone, seed_amount)
                decisions[clone.agent_id] = "seeded"

        # Mark remaining active agents
        for agent in active:
            if agent.agent_id not in decisions:
                decisions[agent.agent_id] = "active"

        logger.info(
            "Period %d complete: %d agents, %d eliminated, %d clones",
            self._period_number, total, len(eliminated_ids), len(new_clones),
        )
        return decisions

    # ------------------------------------------------------------------
    # Elimination
    # ------------------------------------------------------------------

    def _eliminate(
        self,
        agent: TradingAgent,
        record: Optional[PerformanceRecord],
        rank: int,
        total: int,
        reason: str = "bottom_pnl",
    ) -> None:
        trigger_metric = "sharpe" if (record and record.sharpe is not None) else "pnl_pct"
        trigger_value = (
            str(record.sharpe) if (record and record.sharpe is not None)
            else (str(record.pnl_pct) if record else "0")
        )
        self._log.record(EvolutionEvent(
            event_type=EventType.ELIMINATION,
            period_number=self._period_number,
            agent_id=agent.agent_id,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            agent_rank=rank + 1,
            total_agents=total,
            pnl_pct=str(record.pnl_pct) if record else "0",
            notes=(
                f"zero_edge for {self._consecutive_zero_edge.get(agent.agent_id, 0)} "
                f"consecutive periods"
                if reason == "zero_edge"
                else f"Bottom {self._cfg.elimination_threshold:.0%} for "
                     f"{self._cfg.elimination_consecutive_periods} consecutive periods"
            ),
        ))
        self._pm.eliminate_agent(agent.agent_id)

    # ------------------------------------------------------------------
    # Cloning with mutation
    # ------------------------------------------------------------------

    def _clone(
        self,
        source: TradingAgent,
        record: Optional[PerformanceRecord],
        rank: int,
        total: int,
    ) -> TradingAgent:
        mutation_rate = Decimal(str(random.uniform(
            float(self._cfg.mutation_rate_min),
            float(self._cfg.mutation_rate_max),
        )))

        clone = TradingAgent(
            agent_id=str(uuid.uuid4()),
            agent_type=source.agent_type,
            state=AgentState.ACTIVE,
            generation=source.generation + 1,
            parent_id=source.agent_id,
        )

        trigger_metric = "sharpe" if (record and record.sharpe is not None) else "pnl_pct"
        trigger_value = (
            str(record.sharpe) if (record and record.sharpe is not None)
            else (str(record.pnl_pct) if record else "0")
        )

        mutated: dict[str, str] = {
            "mutation_rate": str(mutation_rate),
            "source_generation": str(source.generation),
        }

        self._log.record(EvolutionEvent(
            event_type=EventType.CLONE,
            period_number=self._period_number,
            agent_id=clone.agent_id,
            parent_agent_id=source.agent_id,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            mutated_parameters=mutated,
            mutation_rate=str(mutation_rate),
            agent_rank=rank + 1,
            total_agents=total,
            pnl_pct=str(record.pnl_pct) if record else "0",
            notes=f"Top {self._cfg.clone_threshold:.0%} for "
                  f"{self._cfg.clone_consecutive_periods} consecutive periods",
        ))
        return clone

    # ------------------------------------------------------------------
    # Period evaluation log entries
    # ------------------------------------------------------------------

    def _log_period_evaluation(
        self,
        ranked: list[TradingAgent],
        period_record: dict[str, PerformanceRecord],
        total: int,
    ) -> None:
        for rank, agent in enumerate(ranked):
            rec = period_record.get(agent.agent_id)
            self._log.record(EvolutionEvent(
                event_type=EventType.PERIOD_EVALUATION,
                period_number=self._period_number,
                agent_id=agent.agent_id,
                agent_rank=rank + 1,
                total_agents=total,
                pnl_pct=str(rec.pnl_pct) if rec else "0",
                notes=f"Period {self._period_number} rank {rank + 1}/{total}",
            ))

    @property
    def transaction_log(self) -> EvolutionLog:
        return self._log

    @property
    def period_number(self) -> int:
        return self._period_number
