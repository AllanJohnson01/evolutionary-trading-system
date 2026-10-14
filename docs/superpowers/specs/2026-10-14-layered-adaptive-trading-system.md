## Layered adaptive trading system — design (FRO-253)

Status: designed and dispatched by Ava (2026-10-14) under Allan's directive to
"design this yourself and execute it" (FRO-253 comment, 2026-10-14). Paper
trading only — nothing in this design authorizes live capital.

### Source principles

From the article Allan asked reviewed (x.com/velesxbt/status/2076983445511676253 —
an Ed Thorp thread plus "Bell Labs Solved Trading in the 1950s"), six layers of
a durable trading edge, plus one sizing rule:

1. **Signal** — verify a data channel still contains exploitable information
   (Shannon, Simons). Edges decay; measure them.
2. **Distribution** — assume fat tails; never confuse risk (quantifiable) with
   uncertainty (not) (Mandelbrot, Knight).
3. **Sizing** — size positions to survive the tail (Kelly, Thorp). The
   article's operative rule: **half-Kelly** — most of the growth, far smaller
   drawdowns. "The sizing matters more than the signal."
4. **Compounding** — hold through drawdowns; time in a positive-edge system
   beats reactive churn (Hamming, Munger).
5. **Discipline** — guard against self-deception; separate the decision-maker
   from the risk enforcer (Feynman, Marks).
6. **Reality** — models have a shelf life; markets adapt, so the system must
   too (Derman, Lo).

The existing architecture already embodies 4–6 structurally (weekly evolution
cadence with 2-period confirmation; PortfolioManager as sole risk enforcer;
the evolutionary loop itself as institutionalized model mortality). The gaps
this design closes: 1 (edge-decay measurement), 2 (fat-tail-aware stops and
event blackouts), 3 (Kelly sizing — today only caps exist), plus Allan's asks:
a macro layer, frequent tactical decisions, at least one day-trading bot, and
a weekly whole-system review.

### Architecture: three cadence layers

```
┌────────────────────────────────────────────────────────────────────┐
│ STRATEGIC — weekly                                                  │
│  Evolution-Controller: evolution cycle (rank → eliminate → clone)   │
│  Ava: whole-system review (report → vault + FRO-253)                │
└───────────────┬────────────────────────────────────────────────────┘
                │ population & parameter changes
┌───────────────▼────────────────────────────────────────────────────┐
│ MACRO — daily + event calendar (LLM: Market-Analyst)                │
│  market_posture.json: regime, risk dial 0.0–1.0, blackout windows   │
│  event calendar: FOMC / CPI / NFP / earnings / crypto catalysts     │
└───────────────┬────────────────────────────────────────────────────┘
                │ posture read each cycle
┌───────────────▼────────────────────────────────────────────────────┐
│ TACTICAL — every 15/30 min (programmatic, no LLM in the loop)       │
│  strategy runner: manage exits (TP / trailing / stop) → new entries │
│  → half-Kelly × posture sizing → PortfolioManager approval → order  │
└────────────────────────────────────────────────────────────────────┘

  ISOLATED, PRESERVED: Horizon (causal-entropic optionality agent)
  — own ledger, own memory, own self-review; never registered with
  PortfolioManager or EvolutionEngine (per approved 2026-10-14 spec).
```

**Why the fast loop is programmatic:** Allan's requirement is frequent
(15–30 min) buy/sell decisions, not frequent LLM judgment. An LLM in a
96-cycles/day loop adds cost and nondeterminism without edge; the article's
Discipline layer argues for mechanical execution of a measured edge. LLMs sit
where judgment pays: macro interpretation (daily) and system review (weekly).
Allan explicitly allowed this: "some of these agents might be more
programmatically run; I don't really care."

### Macro layer (Market-Analyst, LLM)

- **Weekly calendar build (Sunday):** assemble the coming week's scheduled
  events — FOMC/rate decisions, CPI, NFP, major earnings for tradeable
  symbols, large token unlocks/upgrades for crypto pairs — into
  `state/event_calendar.json` with timestamps and an impact tier.
- **Daily posture (pre-market, ~07:30 ET; crypto runs 24/7 so the posture
  carries a 24h validity):** read overnight/weekend action, the calendar, and
  macro headlines; emit `state/market_posture.json`:

```json
{
  "as_of": "RFC3339",
  "regime": "risk_on | neutral | risk_off",
  "risk_dial": 0.0,
  "blackout_windows": [{"start": "...", "end": "...", "reason": "FOMC"}],
  "notes": "one-paragraph rationale",
  "valid_until": "RFC3339"
}
```

- **Posture semantics (Distribution layer):** `risk_dial` multiplies position
  size (see Sizing). Blackout windows forbid new entries 30 min before to
  30 min after tier-1 events — scheduled uncertainty is not a tradeable risk.
  Stale posture (past `valid_until`) degrades to `risk_dial = 0.5`, no new
  entries in unknown-blackout periods — fail cautious, never fail open.
- **Autonomy bound:** posture may cut risk to 0 on its own; it may never push
  sizing above the neutral baseline by more than 1.25× without Allan's
  explicit sign-off on the aggressive setting.

### Tactical layer (programmatic runner)

Every cycle, per strategy: refresh bars → mark open positions and apply exit
logic (take-profit targets, trailing stops, hard stops) → evaluate entry
signals → size → `PortfolioManager.approve_position` → submit paper order →
log the trade with full context (signal values, posture, sizing inputs) for
the edge-metrics module.

Starting population (diverse by construction — styles, cadences, and markets
deliberately spread; Allan: "try a lot of different ones"):

| Bot | Style | Cadence | Market/hours | Notes |
|---|---|---|---|---|
| Momentum-Hunter | **intraday momentum day trader** | 15 min | equities RTH + crypto | flattens equity positions before close — Allan's required day trader |
| Breakout | **intraday range-breakout day trader** | 15 min | crypto 24/7 | opening-range / volatility breakout |
| Grid-Trader | market-making grid | 15 min | crypto 24/7 | sideways-regime earner |
| Mean-Reversion | swing mean reversion | 30 min | crypto + equities | multi-day holds, z-score entries |
| RL-Ensemble | FinRL ensemble | 30 min | crypto | joins the population once the FinRL research track (FRO-8) productionizes; not a launch blocker |

No static values anywhere: every strategy parameter (lookbacks, grid spacing,
z-thresholds, TP/trail/stop multipliers, Kelly cap) lives in the agent's
mutable config — modulated daily by posture, adjusted intraday by exit logic,
and mutated weekly by evolution. Parameters are data, not code.

### Sizing (the article's core rule)

- Per entry: `fraction = 0.5 × kelly(rolling_win_rate, rolling_payoff_ratio)`
  over the strategy's last 30 closed trades, clamped to
  `[0, per_asset_concentration]`, then `× risk_dial`.
- Cold start (<30 trades): fixed-fraction risk of 1% of the agent's
  allocation per trade, `× risk_dial`.
- Kelly inputs come from the edge-metrics module, never self-reported by the
  strategy (Discipline: the bet-sizer and the bettor are separated).

### Distribution / fat tails

- Stops sized from EWMA realized volatility with a 3σ tail buffer — never
  from Gaussian assumptions or fixed percentages.
- Event blackouts (above) as hard entry filters.
- Existing kill switches unchanged: 15% max-drawdown halt, 2% daily loss
  limit, enforced only by PortfolioManager.

### Signal layer / edge metrics

Per strategy, rolling and regime-tagged: hit rate, profit factor, Sharpe,
average edge per trade, and the **trend** of each (decay detection). Feeds:
(a) Kelly sizing inputs, (b) the evolution engine's evaluation — ranking uses
risk-adjusted edge, not raw PnL% alone, (c) the weekly review's "is this
channel still live?" question. A strategy whose edge is statistically
indistinguishable from zero for 2 consecutive windows is flagged for
elimination regardless of lucky PnL.

### Strategic layer — weekly review (cadence: Ava's decision)

- **Monday 07:30 ET — Evolution-Controller:** runs
  `EvolutionEngine.run_period` with the edge-metrics records, executes
  eliminations/clones/rebalance per existing `EvolutionConfig` (unchanged:
  2-consecutive-period confirmation, bottom-25% elimination, top-25% cloning
  with 10–20% parameter mutation, population 4–8), posts its cycle report.
- **Monday 08:30 ET — Ava (whole-system review, armed on FRO-253):** reviews
  the evolution cycle's decisions, edge-decay flags, posture accuracy vs. what
  actually happened, risk events (blackout hits, kill-switch proximity),
  Horizon's health (observe only), and infrastructure faults. Output: a dated
  report to the vault (`output/reviews/`) and a comment on FRO-253. Anything
  requiring a judgment the system shouldn't make alone (e.g., raising
  aggressiveness, retiring a whole strategy family, go-live questions) is put
  to Allan as a recommendation.
- During the build phase the same Monday review covers build progress instead.

### Horizon — preserved, self-governing

Per the approved 2026-10-14 Horizon spec, unchanged: pure causal-entropic
optionality objective, permanent structural isolation from PortfolioManager
and EvolutionEngine, own paper ledger, own risk limits, Jev tuning loop with
paper-mode autonomy. This design adds only what Allan asked for on 2026-10-14,
plus cadences (Ava's call):

- **Persistent memory, owned solely by Horizon:** `horizon/state/` —
  `memory.md` (what it has learned), `plans.md` (current thinking and intended
  adjustments), tuning log, ledger. No other agent or module writes there.
- **Decision cadence:** 1-hour bars (τ = 10 → a 10-hour optionality horizon;
  ~24 Mode B rollout calls/day — meaningful horizon at sane cost).
- **Daily self-review (after 00:00 UTC):** Horizon re-reads its own
  instructions, trades, and tuning log; updates `memory.md` / `plans.md`;
  applies τ/T_c adjustments per the spec's paper-mode autonomy. Its own
  instructions are the only authority in this loop.
- **Weekly oversight:** Ava's system review reads Horizon's state and reports
  on it — observe and report only; Ava never edits Horizon's memory, params,
  or instructions. Divergence worth acting on goes to Allan.

### Build plan (staged sub-issues under FRO-253, assigned to Forge)

- **Stage 1 (parallel):** 1a market-posture layer (schema, calendar fetcher,
  posture writer harness); 1b Kelly sizing + fat-tail risk in
  PortfolioManager; 1c Horizon per approved spec + memory + daily self-review.
- **Stage 2:** 2a tactical loop runner + strategy set above; 2b edge-metrics
  module + evolution-engine integration.
- **Stage 3:** 3 operational wiring — schedulers for the tactical loop, arm
  Market-Analyst daily/weekly crons and Evolution-Controller weekly cycle,
  start the paper burn-in, reporting templates.

### Gates left to Allan (nothing below executes without his go-ahead)

- Any live-money transition (per Horizon spec's go-live criteria, extended to
  the whole system: minimum paper duration, Sharpe/drawdown bars, posture
  accuracy record).
- Raising posture aggressiveness above the 1.25× neutral cap.
- Retiring or adding strategy families outside evolution's normal clone/kill.
- DCA pool deployment (untouched by this design).
