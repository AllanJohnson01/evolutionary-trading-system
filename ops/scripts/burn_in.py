"""
Paper burn-in script — FRO-701.

Initialises the 4-agent starting population, runs one full tactical cycle,
and runs Horizon's first decision tick.  Designed to be executed ONCE as the
burn-in gate before handing control to the scheduled launchd jobs.

Usage::

    python ops/scripts/burn_in.py

Requires:
    ALPACA_API_KEY, ALPACA_SECRET_KEY env vars (Alpaca paper account).
    ALPACA_BASE_URL defaults to https://paper-api.alpaca.markets

Outputs logged to stdout (INFO) and to state/burn_in.log.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path so we can run from any directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from evo_trader.config.parameters import CapitalConfig
from evo_trader.portfolio.manager import PortfolioManager
from evo_trader.tactical.population import (
    STARTING_AGENTS,
    init_population,
    register_starting_strategies,
)


def _load_stable_ids(manifest_path: Path) -> dict[str, str] | None:
    """Read per-agent stable IDs from an existing population manifest.

    Returns a name→agent_id dict when the manifest exists and contains them,
    or None on first run (manifest missing or pre-dates the agent_id field).
    """
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        ids = {
            a["name"]: a["agent_id"]
            for a in data.get("agents", [])
            if "agent_id" in a and "name" in a
        }
        return ids if ids else None
    except (json.JSONDecodeError, KeyError):
        return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_REPO_ROOT / "state" / "burn_in.log"),
    ],
)
logger = logging.getLogger("burn_in")


def run_tactical_cycle() -> dict:
    """Run one tactical cycle via the runner module and return a summary dict."""
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not alpaca_key or not alpaca_secret:
        raise EnvironmentError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.")

    from evo_trader.macro.store import PostureStore
    from evo_trader.portfolio.manager import PortfolioManager
    from evo_trader.tactical.runner import StrategyRunner

    try:
        from evo_trader.tactical._alpaca_clients import AlpacaBarsClient, AlpacaOrderClient  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "alpaca-py not installed or _alpaca_clients not built — "
            "run: pip install alpaca-py"
        ) from exc

    bars_client = AlpacaBarsClient(
        api_key=alpaca_key,
        secret_key=alpaca_secret,
        base_url=alpaca_base_url,
    )
    order_client = AlpacaOrderClient(
        api_key=alpaca_key,
        secret_key=alpaca_secret,
        base_url=alpaca_base_url,
    )

    pm = PortfolioManager()
    capital_cfg = CapitalConfig()
    # Reload stable IDs so strategy_id in decisions.jsonl is consistent across runs.
    manifest_path = _REPO_ROOT / "state" / "population.json"
    stable_ids = _load_stable_ids(manifest_path)
    agents = init_population(pm, capital_cfg, stable_ids=stable_ids)

    posture_store = PostureStore()
    log_path = _REPO_ROOT / "state" / "decisions.jsonl"

    runner = StrategyRunner(
        portfolio_manager=pm,
        posture_store=posture_store,
        bars_client=bars_client,
        order_client=order_client,
        log_path=log_path,
    )

    symbol = os.getenv("BURN_IN_SYMBOL", "SPY")
    register_starting_strategies(runner, agents, symbol)

    runner.run_once()
    logger.info("Tactical cycle complete — decisions logged to %s", log_path)

    # Return tail of decisions log for reporting
    entries: list[dict] = []
    if log_path.exists():
        lines = log_path.read_text().strip().splitlines()
        for line in lines[-10:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return {"decisions_log": str(log_path), "last_entries": entries}


def run_horizon_tick() -> dict:
    """Run one Horizon decision tick and return a summary dict."""
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    alpaca_base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    symbol = os.getenv("HORIZON_SYMBOL", "SPY")

    if not alpaca_key or not alpaca_secret:
        raise EnvironmentError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.")

    from evo_trader.horizon.agent import HorizonAgent
    from evo_trader.horizon.alpaca_client import AlpacaPaperClient
    from evo_trader.horizon.config import HorizonConfig

    client = AlpacaPaperClient(
        api_key=alpaca_key,
        api_secret=alpaca_secret,
    )
    cfg = HorizonConfig()
    tuning_log_path = _REPO_ROOT / "state" / "horizon_tuning.jsonl"
    agent = HorizonAgent(
        client=client, cfg=cfg, symbol=symbol, tuning_log_path=tuning_log_path
    )

    candidate = agent.run_one_bar()
    logger.info("Horizon tick complete — candidate: %s", candidate)

    # Append to decisions log
    decisions_path = _REPO_ROOT / "state" / "horizon_decisions.jsonl"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    with decisions_path.open("a") as f:
        f.write(
            json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "candidate": str(candidate),
            }) + "\n"
        )

    return {"candidate": str(candidate), "decisions_log": str(decisions_path)}


def main() -> None:
    logger.info("=== EVO-TRADER PAPER BURN-IN ===")
    logger.info("Repo root: %s", _REPO_ROOT)

    # 1. Init population
    logger.info("--- Step 1: Initialising paper population ---")
    capital_cfg = CapitalConfig()
    pm = PortfolioManager()

    manifest_path = _REPO_ROOT / "state" / "population.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Reload stable IDs if manifest already has them (subsequent runs preserve identity).
    stable_ids = _load_stable_ids(manifest_path)
    if stable_ids:
        logger.info("Reloading stable agent IDs from %s", manifest_path)
    else:
        logger.info("First run — minting new stable agent IDs")

    agents = init_population(pm, capital_cfg, stable_ids=stable_ids)
    logger.info("Population: %d agents, $%s active trading capital", len(agents), capital_cfg.active_trading_total)

    # Persist population manifest (includes agent_id so restarts reuse the same identity).
    agent_by_name = {name: agent for agent, (name, _, _) in zip(agents, STARTING_AGENTS)}
    manifest = {
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "capital_config": {
            "total_capital": str(capital_cfg.total_capital),
            "active_trading_total": str(capital_cfg.active_trading_total),
            "initial_agent_allocation": str(capital_cfg.initial_agent_allocation),
        },
        "agents": [
            {
                "name": name,
                "agent_id": agent_by_name[name].agent_id,
                "agent_type": atype.value,
                "allocated_capital": str(capital_cfg.initial_agent_allocation),
            }
            for name, atype, _ in STARTING_AGENTS
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Population manifest written to %s", manifest_path)

    # 2. Tactical cycle
    logger.info("--- Step 2: Running one tactical cycle ---")
    try:
        tactical_result = run_tactical_cycle()
        logger.info("Tactical cycle OK — %d decision entries logged", len(tactical_result["last_entries"]))
    except Exception as exc:
        logger.error("Tactical cycle FAILED: %s", exc)
        sys.exit(1)

    # 3. Horizon tick
    logger.info("--- Step 3: Running Horizon first decision tick ---")
    try:
        horizon_result = run_horizon_tick()
        logger.info("Horizon tick OK — candidate: %s", horizon_result["candidate"])
    except Exception as exc:
        logger.error("Horizon tick FAILED: %s", exc)
        sys.exit(1)

    # 4. Summary
    logger.info("=== BURN-IN COMPLETE ===")
    logger.info("Population manifest:    %s", manifest_path)
    logger.info("Tactical decisions log: %s", tactical_result["decisions_log"])
    logger.info("Horizon decisions log:  %s", horizon_result["decisions_log"])
    logger.info(
        "\nHand-off: launchd jobs are ready to load.\n"
        "  Load tactical scheduler:  launchctl load -w ~/Library/LaunchAgents/com.evo-trader.tactical.plist\n"
        "  Load Horizon tick:        launchctl load -w ~/Library/LaunchAgents/com.evo-trader.horizon-tick.plist\n"
        "  Load Horizon self-review: launchctl load -w ~/Library/LaunchAgents/com.evo-trader.horizon-review.plist"
    )


if __name__ == "__main__":
    main()
