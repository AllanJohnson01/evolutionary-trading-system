"""
Evolution transaction log — immutable audit trail for every elimination,
clone, and mutation event. Required for understanding what the system learns.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    ELIMINATION = "elimination"
    CLONE = "clone"
    MUTATION = "mutation"
    REBALANCE = "rebalance"
    PERIOD_EVALUATION = "period_evaluation"


@dataclass
class EvolutionEvent:
    event_type: EventType
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    period_number: int = 0

    # Subject agent(s)
    agent_id: str = ""
    parent_agent_id: str | None = None

    # Trigger metrics
    trigger_metric: str = ""           # e.g. "pnl_pct", "drawdown"
    trigger_value: str = ""            # str so Decimal serializes cleanly

    # Mutation details (clone events)
    mutated_parameters: dict[str, Any] = field(default_factory=dict)
    mutation_rate: str = ""            # e.g. "0.15"

    # Evaluation snapshot
    agent_rank: int | None = None
    total_agents: int | None = None
    pnl_pct: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


class EvolutionLog:
    """
    Append-only event log. Persists to JSONL on disk so history survives restarts.
    Each line is one JSON event object.
    """

    def __init__(self, log_path: Path | str = "evolution_log.jsonl") -> None:
        self._path = Path(log_path)
        self._events: list[EvolutionEvent] = []
        self._load_existing()

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        # Reconstruct EvolutionEvent from dict
                        data["event_type"] = EventType(data["event_type"])
                        self._events.append(EvolutionEvent(**data))
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning("Skipping malformed log entry: %s", e)

    def record(self, event: EvolutionEvent) -> None:
        self._events.append(event)
        # Append immediately — log is authoritative even if process crashes
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
        logger.info(
            "Evolution event: %s agent=%s", event.event_type.value, event.agent_id
        )

    def events_for_agent(self, agent_id: str) -> list[EvolutionEvent]:
        return [e for e in self._events if e.agent_id == agent_id]

    def events_by_type(self, event_type: EventType) -> list[EvolutionEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def all_events(self) -> list[EvolutionEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)
