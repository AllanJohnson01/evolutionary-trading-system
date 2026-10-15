"""
Daily self-review for Horizon. Run via:
    python -m evo_trader.horizon.self_review

Re-reads Horizon's own instructions and tuning log; updates memory.md and
plans.md; runs the Jev tuning proposal.

Written ONLY by this module — no other agent or module writes to
horizon/state/memory.md or horizon/state/plans.md. Horizon's own
instructions are the only authority in this loop (per the spec's
"self-governing" requirement).
"""
from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from evo_trader.horizon.config import HorizonConfig
from evo_trader.horizon.ledger import HorizonLedger
from evo_trader.horizon.tuner import JevTuner, RolloutRecord, TuningDecision, TuningLog

logger = logging.getLogger(__name__)

_STATE_DIR = Path(__file__).parent / "state"
_PROCEDURE_PATH = Path(__file__).parent / "PROCEDURE.md"

STARTING_CAPITAL = Decimal("10000")


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


def _load_tuning_log(state_dir: Path) -> list[dict[str, Any]]:
    log_path = state_dir / "tuning_log.jsonl"
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _build_self_review_prompt(
    procedure_text: str,
    ledger: HorizonLedger,
    tuning_history: list[dict[str, Any]],
    existing_memory: str,
    existing_plans: str,
    config: HorizonConfig,
) -> str:
    tuning_str = json.dumps(tuning_history[-10:], indent=2)
    return f"""You are Horizon, running your daily self-review (after 00:00 UTC).
Your only authority is your own instructions below — no other agent or module may direct this review.

## Your instructions (the causal-entropic-optionality procedure)
{procedure_text}

## Current configuration
- τ: {config.tau}
- T_c: {config.t_c}
- execution_mode: {config.execution_mode}
- tuning_cadence_bars: {config.tuning_cadence_bars}

## Ledger summary
- Current NAV: {ledger.nav}
- Max drawdown so far: {ledger.drawdown}

## Recent tuning history (last 10 entries)
{tuning_str}

## Existing memory
{existing_memory or "(empty)"}

## Existing plans
{existing_plans or "(empty)"}

## Task
1. Reflect on what you have learned from your tuning history and drawdown trajectory.
2. Update your memory.md — what patterns, observations, or lessons are worth carrying forward?
3. Update your plans.md — what adjustments or intentions guide your next decisions?
4. Be concise. Memory and plans are for YOUR future use, not a report to others.

Return JSON with this exact structure:
{{
  "memory_md": "<full updated content of memory.md in markdown>",
  "plans_md": "<full updated content of plans.md in markdown>"
}}
"""


def run_self_review(
    config: HorizonConfig,
    llm: LLMClient,
    state_dir: Path | None = None,
) -> dict[str, str]:
    """
    Execute Horizon's daily self-review. Updates memory.md and plans.md.
    Also triggers a Jev tuning cycle using any accumulated evidence.

    Returns dict with keys 'memory_md' and 'plans_md'.
    """
    state_dir = state_dir or _STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)

    procedure_text = _load_text(_PROCEDURE_PATH) or "(procedure not found)"
    ledger = HorizonLedger(starting_capital=STARTING_CAPITAL)
    tuning_history = _load_tuning_log(state_dir)
    existing_memory = _load_text(state_dir / "memory.md")
    existing_plans = _load_text(state_dir / "plans.md")

    prompt = _build_self_review_prompt(
        procedure_text=procedure_text,
        ledger=ledger,
        tuning_history=tuning_history,
        existing_memory=existing_memory,
        existing_plans=existing_plans,
        config=config,
    )

    response = llm.complete(prompt)

    # Parse LLM response
    memory_md = existing_memory
    plans_md = existing_plans
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
        data = json.loads(response[start:end])
        memory_md = str(data.get("memory_md", existing_memory))
        plans_md = str(data.get("plans_md", existing_plans))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Self-review: failed to parse LLM response: %s — keeping existing files", exc)

    if memory_md:
        (state_dir / "memory.md").write_text(memory_md, encoding="utf-8")
    if plans_md:
        (state_dir / "plans.md").write_text(plans_md, encoding="utf-8")

    # Run one Jev tuning cycle with whatever evidence the rollout deque had.
    # The self-review has no live rollout evidence, so we construct minimal
    # synthetic evidence from the tuning log to let the tuner assess history.
    evidence: deque[RolloutRecord] = deque()
    for entry in tuning_history[-config.tuning_cadence_bars:]:
        # Reconstruct a RolloutRecord from persisted tuning log entries.
        # Miss rate was persisted; invert it to approximate branch hits.
        miss_rate_str = entry.get("miss_rate", "0.5")
        try:
            miss_rate = float(miss_rate_str)
        except (ValueError, TypeError):
            miss_rate = 0.5
        stayed = miss_rate < 0.5
        evidence.append(RolloutRecord(
            candidate_diversity_scores=[2],  # neutral diversity placeholder
            price_stayed_in_branch=stayed,
        ))

    tuner = JevTuner()
    tuning_log = TuningLog(state_dir / "tuning_log.jsonl")
    decision = tuner.propose(config, evidence=evidence)
    if decision is not None:
        before_tau = config.tau
        tuning_log.record(decision, before_tau=before_tau)
        if decision.applied:
            config.tau = decision.proposed_tau
            logger.info(
                "Self-review tuning applied: tau %d → %d",
                before_tau, decision.proposed_tau,
            )
        else:
            logger.info(
                "Self-review tuning escalated (live mode): proposed tau=%d",
                decision.proposed_tau,
            )

    logger.info("Self-review complete: memory.md and plans.md updated")
    return {"memory_md": memory_md, "plans_md": plans_md}


if __name__ == "__main__":
    import os
    from anthropic import Anthropic

    logging.basicConfig(level=logging.INFO)

    class _AnthropicClient:
        def __init__(self) -> None:
            self._client = Anthropic()

        def complete(self, prompt: str) -> str:
            msg = self._client.messages.create(
                model=os.getenv("HORIZON_MODEL", "claude-sonnet-5-5"),
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

    cfg = HorizonConfig()
    llm_client = _AnthropicClient()
    run_self_review(config=cfg, llm=llm_client)
