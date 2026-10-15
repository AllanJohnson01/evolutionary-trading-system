# Operations Manual — Evolutionary Trading System

Paper trading only. Nothing in this document authorises live capital deployment.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [One-Time Setup](#one-time-setup)
4. [Scheduled Components](#scheduled-components)
5. [Start / Stop / Recovery](#start--stop--recovery)
6. [Multica Wakeup Rules](#multica-wakeup-rules)
7. [Log Locations](#log-locations)
8. [State Files](#state-files)
9. [Reporting Templates](#reporting-templates)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

Three cadence layers run concurrently:

| Layer | Component | Cadence | Mechanism |
|-------|-----------|---------|-----------|
| Tactical | Strategy runner (`evo_trader.tactical.runner`) | Every 15 min | launchd plist |
| Macro | Market-Analyst (Multica agent) | Daily 07:30 ET + Sunday 18:00 ET | Multica wakeup |
| Strategic | Evolution-Controller (Multica agent) | Monday 07:30 ET | Multica wakeup |
| Horizon | Horizon decision tick (`evo_trader.horizon.runner`) | Every 1 hour | launchd plist |
| Horizon | Horizon daily self-review (`evo_trader.horizon.self_review`) | Daily 00:15 local | launchd plist |

Horizon is isolated from the portfolio manager and evolution engine — it runs its own paper ledger. The tactical loop reads `state/market_posture.json` written by Market-Analyst.

---

## Prerequisites

- macOS with launchd (the scripts are `launchd` plist jobs)
- Python 3.11+ installed in a virtualenv at `REPO_ROOT/venv/` (adjust paths below if different)
- `pip install -e .` run from `REPO_ROOT` to install `evo-trader` in editable mode
- `pip install alpaca-py anthropic` for live Alpaca and Anthropic clients
- Alpaca paper account with `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` set in your shell profile (or in the plist `EnvironmentVariables` blocks)
- `ANTHROPIC_API_KEY` set for Horizon self-review and Market-Analyst runs

---

## One-Time Setup

### 1. Copy the plists to LaunchAgents

```bash
REPO_ROOT="$(pwd)"   # run from repo root
mkdir -p ~/Library/LaunchAgents

for f in ops/launchd/*.plist; do
    cp "$f" ~/Library/LaunchAgents/
done
```

### 2. Edit each plist — fill in the `FILL_IN` placeholders

Open each plist and substitute:

| Placeholder | Value |
|-------------|-------|
| `FILL_IN` (ProgramArguments first item) | Path to your venv Python, e.g. `/Users/yourname/code/evolutionary-trading-system/venv/bin/python3` |
| `FILL_IN` (WorkingDirectory) | Absolute path to repo root, e.g. `/Users/yourname/code/evolutionary-trading-system` |
| `FILL_IN` (ALPACA_API_KEY) | Your Alpaca paper API key |
| `FILL_IN` (ALPACA_SECRET_KEY) | Your Alpaca paper secret key |

The `ALPACA_BASE_URL` is pre-filled with `https://paper-api.alpaca.markets` — leave it unless you know otherwise.

Example sed one-liner (adjust paths):

```bash
VENV_PYTHON="/Users/yourname/code/evolutionary-trading-system/venv/bin/python3"
REPO_PATH="/Users/yourname/code/evolutionary-trading-system"
ALPACA_KEY="your-key-here"
ALPACA_SECRET="your-secret-here"

for plist in ~/Library/LaunchAgents/com.evo-trader.*.plist; do
    sed -i '' \
        "s|FILL_IN_PYTHON|$VENV_PYTHON|g;
         s|FILL_IN_REPO|$REPO_PATH|g;
         s|FILL_IN_ALPACA_KEY|$ALPACA_KEY|g;
         s|FILL_IN_ALPACA_SECRET|$ALPACA_SECRET|g" \
        "$plist"
done
```

### 3. Create the log directory

```bash
mkdir -p ~/Library/Logs/evo-trader
```

### 4. Run the burn-in script

```bash
cd REPO_ROOT
ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python ops/scripts/burn_in.py
```

This initialises the 4-agent paper population, runs one tactical cycle end-to-end, and runs Horizon's first decision tick. Review the output before loading launchd jobs.

### 5. Load the launchd jobs

```bash
launchctl load -w ~/Library/LaunchAgents/com.evo-trader.tactical.plist
launchctl load -w ~/Library/LaunchAgents/com.evo-trader.horizon-tick.plist
launchctl load -w ~/Library/LaunchAgents/com.evo-trader.horizon-review.plist
```

Verify they are loaded:

```bash
launchctl list | grep evo-trader
```

---

## Scheduled Components

### Tactical runner — every 15 minutes

- **Entry point:** `python -m evo_trader.tactical.runner --once`
- **plist label:** `com.evo-trader.tactical`
- **plist file:** `ops/launchd/com.evo-trader.tactical.plist`
- **What it does:** reads `state/market_posture.json`, iterates all registered strategies, checks exits → evaluates entries → sizes positions via half-Kelly × risk_dial → submits paper orders via Alpaca → appends decisions to `state/decisions.jsonl`
- **Log:** `~/Library/Logs/evo-trader/tactical.log`
- **Error log:** `~/Library/Logs/evo-trader/tactical-error.log`

### Horizon decision tick — every 1 hour

- **Entry point:** `python -m evo_trader.horizon.runner --once`
- **plist label:** `com.evo-trader.horizon-tick`
- **plist file:** `ops/launchd/com.evo-trader.horizon-tick.plist`
- **What it does:** runs one Horizon causal-entropic-optionality decision bar; submits paper order if warranted; appends to `state/horizon_decisions.jsonl`
- **Log:** `~/Library/Logs/evo-trader/horizon-tick.log`

### Horizon daily self-review — 00:15 local time

- **Entry point:** `python -m evo_trader.horizon.self_review`
- **plist label:** `com.evo-trader.horizon-review`
- **plist file:** `ops/launchd/com.evo-trader.horizon-review.plist`
- **Timezone note:** fires at 00:15 **local machine time**. If your machine is not in UTC, adjust the `StartCalendarInterval` `Hour` value: UTC offset = local hour where you want 00:15 UTC to land. Example: machine is US/Eastern (UTC-5): set `Hour: 19` `Minute: 15` for 00:15 UTC.
- **What it does:** re-reads Horizon's own instructions + tuning log; updates `src/evo_trader/horizon/state/memory.md` and `plans.md`; runs τ/T_c tuning proposal
- **Log:** `~/Library/Logs/evo-trader/horizon-review.log`

### Market-Analyst — daily 07:30 ET (Multica wakeup)

- **Multica wakeup ID:** `01a140a5-4c58-7205-829b-ddecd36f5474`
- **Schedule:** `30 12 * * *` UTC (= 07:30 America/New_York), timezone: `America/New_York`
- **Agent:** `83e0acc0-c1de-44e2-993c-45008ea6bff3` (Market-Analyst)
- **Issue:** FRO-701 (`01a13cd8-69b7-734a-86d7-69d0d6dbf87a`)
- **Expires:** 2027-01-07 (12 weeks from creation — Ava re-arms at weekly review)
- **What it does:** reads overnight/weekend market action + event calendar + headlines; writes and validates `state/market_posture.json`; posts posture summary comment on FRO-701

### Market-Analyst — weekly calendar rebuild (Sunday 18:00 ET)

- **Multica wakeup ID:** `01a140a5-5ec8-77c5-89f3-99f9d9963095`
- **Schedule:** `0 18 * * 0` America/New_York
- **Agent:** `83e0acc0-c1de-44e2-993c-45008ea6bff3` (Market-Analyst)
- **Issue:** FRO-701 (`01a13cd8-69b7-734a-86d7-69d0d6dbf87a`)
- **Expires:** 2027-01-07
- **What it does:** fetches upcoming FOMC, CPI, NFP, earnings, crypto catalysts; writes `state/event_calendar.json`

### Evolution-Controller — weekly cycle (Monday 07:30 ET)

- **Multica wakeup ID:** `01a140a5-9f49-74a5-b590-406bd5ff5580`
- **Schedule:** `30 7 * * 1` America/New_York
- **Agent:** `34995ba2-5f65-4a9c-8a3a-61a345e05e17` (Evolution-Controller)
- **Issue:** FRO-701 (`01a13cd8-69b7-734a-86d7-69d0d6dbf87a`)
- **Expires:** 2027-01-07
- **What it does:** runs `EvolutionEngine.run_period()` with edge-metrics records; eliminates bottom performers, clones top performers with mutation; posts cycle report using `docs/templates/evolution-cycle-report.md`

---

## Start / Stop / Recovery

### Stop all launchd jobs

```bash
launchctl unload ~/Library/LaunchAgents/com.evo-trader.tactical.plist
launchctl unload ~/Library/LaunchAgents/com.evo-trader.horizon-tick.plist
launchctl unload ~/Library/LaunchAgents/com.evo-trader.horizon-review.plist
```

### Start (or restart) individual jobs

```bash
# Restart tactical runner
launchctl unload ~/Library/LaunchAgents/com.evo-trader.tactical.plist
launchctl load -w ~/Library/LaunchAgents/com.evo-trader.tactical.plist

# Run one tactical cycle manually for diagnostics
cd REPO_ROOT && python -m evo_trader.tactical.runner --once

# Run one Horizon tick manually
cd REPO_ROOT && python -m evo_trader.horizon.runner --once

# Run Horizon self-review manually
cd REPO_ROOT && python -m evo_trader.horizon.self_review
```

### Check job status

```bash
launchctl list | grep evo-trader
# PID column is non-zero while running; last exit code shown in third column
```

### Recovery from crashed tactical cycle

1. Check `~/Library/Logs/evo-trader/tactical-error.log` for the error.
2. If the posture file is stale (past `valid_until`), Market-Analyst's wakeup will fire again at 07:30 ET. You can force a posture refresh manually by triggering the wakeup:
   ```bash
   multica issue wakeup trigger 01a13cd8-69b7-734a-86d7-69d0d6dbf87a --wakeup-id 01a140a5-4c58-7205-829b-ddecd36f5474
   ```
3. If `state/decisions.jsonl` is corrupted, it can be cleared — it is an append log and does not affect live positions.
4. If Alpaca paper positions are out of sync with local state, reconcile via the Alpaca paper UI; there is no auto-reconciliation in paper mode.

### Recovery from missed Horizon ticks

Horizon is stateless per bar — each tick starts fresh from the Alpaca API. Missed ticks are lost; they are logged as gaps in `state/horizon_decisions.jsonl`. To catch up, run ticks manually:

```bash
for i in $(seq 1 N_MISSED); do python -m evo_trader.horizon.runner --once; done
```

### Re-arming expired Multica wakeups

All wakeup rules expire 2027-01-07 (12 weeks from 2026-10-15). Ava re-arms them at her weekly review. To re-arm manually:

```bash
# Re-arm Market-Analyst daily posture
multica issue wakeup update 01a13cd8-69b7-734a-86d7-69d0d6dbf87a \
  --wakeup-id 01a140a5-4c58-7205-829b-ddecd36f5474 \
  --expires-in 2016h

# Re-arm Market-Analyst weekly calendar
multica issue wakeup update 01a13cd8-69b7-734a-86d7-69d0d6dbf87a \
  --wakeup-id 01a140a5-5ec8-77c5-89f3-99f9d9963095 \
  --expires-in 2016h

# Re-arm Evolution-Controller weekly
multica issue wakeup update 01a13cd8-69b7-734a-86d7-69d0d6dbf87a \
  --wakeup-id 01a140a5-9f49-74a5-b590-406bd5ff5580 \
  --expires-in 2016h
```

---

## Multica Wakeup Rules

| Rule | Wakeup ID | Agent | Schedule | Expires |
|------|-----------|-------|----------|---------|
| Market-Analyst daily posture | `01a140a5-4c58-7205-829b-ddecd36f5474` | Market-Analyst | Daily 07:30 ET | 2027-01-07 |
| Market-Analyst weekly calendar | `01a140a5-5ec8-77c5-89f3-99f9d9963095` | Market-Analyst | Sunday 18:00 ET | 2027-01-07 |
| Evolution-Controller weekly cycle | `01a140a5-9f49-74a5-b590-406bd5ff5580` | Evolution-Controller | Monday 07:30 ET | 2027-01-07 |

All rules are on issue FRO-701 (`01a13cd8-69b7-734a-86d7-69d0d6dbf87a`).

---

## Log Locations

| Log | Path |
|-----|------|
| Tactical runner stdout | `~/Library/Logs/evo-trader/tactical.log` |
| Tactical runner stderr | `~/Library/Logs/evo-trader/tactical-error.log` |
| Horizon decision tick | `~/Library/Logs/evo-trader/horizon-tick.log` |
| Horizon daily self-review | `~/Library/Logs/evo-trader/horizon-review.log` |
| Burn-in log | `state/burn_in.log` |

---

## State Files

| File | Written by | Purpose |
|------|-----------|---------|
| `state/market_posture.json` | Market-Analyst | Current regime, risk_dial, blackout windows |
| `state/event_calendar.json` | Market-Analyst | Upcoming macro events |
| `state/decisions.jsonl` | Tactical runner | Per-cycle trade decisions (append-only) |
| `state/population.json` | Burn-in script | Starting population manifest |
| `state/horizon_decisions.jsonl` | Horizon runner | Per-tick candidate decisions |
| `state/horizon_tuning.jsonl` | Horizon | Jev tuning proposals and decisions |
| `src/evo_trader/horizon/state/ledger.json` | Horizon | Horizon's isolated paper ledger |
| `src/evo_trader/horizon/state/memory.md` | Horizon self-review only | Horizon's learnings |
| `src/evo_trader/horizon/state/plans.md` | Horizon self-review only | Horizon's current thinking |

---

## Reporting Templates

| Template | Path | Used by |
|----------|------|---------|
| Evolution cycle report | `docs/templates/evolution-cycle-report.md` | Evolution-Controller (Monday) |
| Weekly system review | `docs/templates/weekly-system-review.md` | Ava (Monday, after evolution cycle) |

---

## Troubleshooting

**Tactical runner exits with "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"**  
→ The plist `EnvironmentVariables` block has not been filled in. Re-edit the plist and reload.

**Posture is stale; tactical runner degraded to risk_dial=0.5**  
→ Market-Analyst's daily wakeup may have failed. Check FRO-701 comments for error. Trigger manually if needed (see Recovery section).

**Horizon self-review fails with Anthropic auth error**  
→ `ANTHROPIC_API_KEY` is not set in the plist `EnvironmentVariables` block. Add it and reload the `horizon-review` plist.

**launchctl list shows last exit code non-zero (e.g. 1)**  
→ Check the error log for that component (see Log Locations). The most common causes are missing env vars, network errors to Alpaca, and stale Python imports (run `pip install -e .` again).

**Evolution-Controller weekly cycle does not appear to have run**  
→ Check the wakeup status: `multica issue wakeup get 01a13cd8-69b7-734a-86d7-69d0d6dbf87a --wakeup-id 01a140a5-9f49-74a5-b590-406bd5ff5580 --output json`. Check `fire_count`, `last_error`, and `next_fire_at`.
