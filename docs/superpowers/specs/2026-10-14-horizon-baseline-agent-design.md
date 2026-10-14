## Horizon: isolated causal-entropic baseline agent — design spec

Status: approved by Allan (chat, 2026-10-14) — ready for implementation planning.

### Purpose

A fourth trading agent, "Horizon," whose decision rule is pure future-optionality
maximization — the procedure in the `causal-entropic-optionality` skill, itself a
direct translation of Wissner-Gross & Freer, "Causal Entropic Forces" (Phys. Rev.
Lett. 110, 168702, 2013). Horizon carries **no predictive, fundamental, or technical
trading content** — it exists solely as a stable comparison point against the other
(P&L-objective) agents: Grid, Momentum, Mean-Reversion, RL.

Because its only value is as a *stable* reference, Horizon is architecturally
excluded from the evolution engine's capital allocation, cloning, and elimination —
permanently, not as a policy flag that could be toggled off by mistake.

### Non-goals

- Not a fifth competitor for capital allocation.
- Not a replacement for, or new implementation of, the causal-entropic math — that
  already exists as the `causal-entropic-optionality` skill. This spec only covers
  wiring that skill's procedure to live market state and an isolated paper ledger.
- Not live-money trading. This phase is Alpaca **paper** trading only. A live-money
  phase is a distinct, later decision gated on criteria defined below, and changes
  the tuning-autonomy behavior (see Execution mode gate).

### Relationship to existing code

The repo (`evolutionary-trading-system`) has no code on `dev` yet. An unmerged
branch (`agent/forge/665dc279a491`, FRO-9) already implements the shared
abstractions this spec builds on:

- `evo_trader.agents.base.TradingAgent` / `AgentType` / `AgentState` —
  the agent registry's shape. Horizon does **not** need a new `AgentType` member
  unless/until it's merged into the shared registry; see Isolation below.
- `evo_trader.portfolio.manager.PortfolioManager` — sole enforcer of capital buckets,
  position approval, and risk limits for anything registered into it via
  `register_agent`. **The isolation mechanism is simply never calling
  `register_agent` for Horizon** — it is structurally invisible to allocation,
  rebalance, and elimination, not merely exempted by a flag that could be
  misconfigured.
- `evo_trader.evolution.engine.EvolutionEngine.run_period` — only acts on agents
  explicitly passed into it. Horizon is never passed in.
- `evo_trader.config.parameters` — frozen-dataclass config idiom
  (`CapitalConfig`, `RiskConfig`, `EvolutionConfig`, `AgentConfig`). Horizon's
  config follows the same shape (see `HorizonConfig` below) for consistency, even
  though it's never read by `PortfolioManager`.

This spec assumes FRO-9 lands on `dev` first, or Horizon's build takes a private
copy of the three files it depends on (`base.py`, and read-only references to the
`RiskConfig` shape) and reconciles on merge. Either way, Horizon must not require
changes to `PortfolioManager`, `EvolutionEngine`, or `AgentConfig` to exist — if the
design turns out to need one, that's a signal the isolation boundary leaked and
needs re-examination before proceeding.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         HORIZON AGENT                         │
│                                                                │
│  ┌────────────────┐   ┌───────────────────┐   ┌────────────┐ │
│  │ Market state    │ → │ Decision loop       │ → │ Hard       │ │
│  │ adapter         │   │ (causal-entropic-   │   │ constraint │ │
│  │ (Alpaca →       │   │  optionality skill, │   │ filter     │ │
│  │  candidates +   │   │  Mode B rollout)    │   │ (own risk  │ │
│  │  rollout state) │   │                     │   │  params)   │ │
│  └────────────────┘   └──────────┬──────────┘   └─────┬──────┘ │
│                                   │                     │       │
│                                   ▼                     ▼       │
│                        ┌─────────────────┐   ┌────────────────┐│
│                        │ Jev tuning loop │   │ Horizon ledger  ││
│                        │ (τ, T_c deltas) │   │ (own paper      ││
│                        └─────────────────┘   │  account, own   ││
│                                               │  NAV/drawdown)  ││
│                                               └────────────────┘│
└──────────────────────────────────────────────────────────────┘
                 ▲                                    │
                 │ execution_mode (paper|live)         │ telemetry
                 │ gates tuner autonomy, never         │ (same shape as
                 │ Horizon's trading itself            │ other 3 agents)
```

Horizon never calls into `PortfolioManager` or `EvolutionEngine`. It owns its own
ledger end to end.

### Components

#### 1. Market state adapter

Translates Alpaca paper-account state into the skill's vocabulary:

- **Candidate actions** at the current decision point: `enter_long`, `enter_short`,
  `exit`, `resize`, `hold`, `hedge` — constrained to what's actually legal given
  current position (e.g., `exit` only offered if a position is open).
- **Rollout state**: price history + current bar, sufficient to let the decision
  loop construct plausible branches over the next τ bars. No fundamental or
  technical-indicator features — only what's needed to define reachable future
  price paths (OHLCV).
- Alpaca paper REST + data feed client (keys required — confirm Allan already has
  an Alpaca paper account before build starts; not yet confirmed as of this spec).

#### 2. Decision loop

Calls the `causal-entropic-optionality` skill's procedure once per bar:

1. Enumerate candidates (adapter output).
2. Roll forward τ bars per candidate — **Mode B (single-call virtual rollout)**
   at launch, per the skill's own guidance to start cheap/fast for
   frequent decision points. Mode A (parallel dispatched rollout workers) is an
   explicit future option if Mode B's rollouts prove too shallow, not part of this
   phase's scope.
3. Score each candidate by reachable-future-diversity, not direct expected value.
4. Apply hard constraints (component 3).
5. Act: submit the winning candidate as an Alpaca paper order.
6. Repeat fresh next bar — no state carried forward as "locked in."

#### 3. Hard constraint filter

Horizon's own risk parameters — not shared with `PortfolioManager`, since Horizon
has its own capital pool. Values default to the same figures already in
`RiskConfig` (15% max drawdown kill switch, 2% daily loss limit) unless Allan sets
Horizon-specific numbers; this spec does not assume they must match the shared
agents' values, only that Horizon must have *some* enforced limits before paper
trading starts. Constraints filter the candidate list before scoring (illegal/
unsafe candidates removed) — they are never just another scored factor.

#### 4. Horizon ledger

A standalone paper-trading account and position/NAV tracker, structurally separate
from `PortfolioManager`'s buckets. Same metrics tracked as the other three agents
will need (Sharpe, max drawdown, win rate, trade log) for future comparability, but
stored and computed independently. No code path from this ledger back into
`PortfolioManager`'s `_active_pool`, `_cash_reserve`, or `_agents` registry.

#### 5. Jev tuning loop

A planning-agent classifier, per the `Jev_First_Standard` doctrine's "trading
system: regime classification, signal-quality scoring, continue/retry/stop
decisions" lane:

- **Input state**: recent rollout history — for each of the last N decisions, the
  branches that were enumerated vs. what actually happened, plus Horizon's
  resulting NAV trajectory.
- **Output**: proposed `τ` and `T_c` deltas (Score-type questions, bounded to the
  skill's stated parameter roles — τ = rollout horizon depth, T_c = how strongly
  optionality dominates vs. constraints) plus a rollout-quality signal (did price
  action stay within the branches assumed, as a Noul/Score).
- **Cadence**: every N bars or once per trading day — tighter than the originally
  discussed weekly cadence, since paper trading has no capital-loss consequence to
  protect against.
- **Logging**: every proposed and applied adjustment logged with before/after
  `(τ, T_c)` values and the rollout evidence that produced the proposal. This log
  is the record Allan reviews at the live-trading go/no-go gate (see below) — it
  must be reconstructable into "what was Horizon's objective on day X," since an
  unbounded tuner can otherwise drift the agent into a state that's hard to
  interpret after the fact.

`HorizonConfig` (new, same shape as `AgentConfig`/`RiskConfig` in
`evo_trader.config.parameters`):

```python
@dataclass
class HorizonConfig:
    tau: int = 10              # rollout horizon, bars
    t_c: Decimal = Decimal("1.0")   # optionality-objective strength
    tuning_cadence_bars: int = <TBD — set at build time, not blocking this spec>
    execution_mode: str = "paper"   # "paper" | "live"
```

#### 6. Execution mode gate

A single `execution_mode` flag (`paper` | `live`) — same mechanism already used
elsewhere in this workspace (Jev P0's `execution_mode` switch on `create_issue` →
`run_only`), applied here as the trade-execution-authorization gate the doctrine
requires before real capital is at risk:

- `execution_mode = "paper"` (default, this phase): Jev's proposed τ/T_c
  adjustments apply **automatically, unbounded** — no band, no step-size cap, no
  human approval required per adjustment. This is a deliberate, explicit decision
  by Allan (chat, 2026-10-14), made specifically because paper trading has no
  capital-loss consequence; it is not the default recommendation for a system
  with real capital at stake, and must not be read as precedent for the live-mode
  case below.
- `execution_mode = "live"`: every proposed adjustment escalates for Allan's
  explicit approval before applying — no code change required to enact this, the
  flag alone changes the tuner's behavior. This satisfies `Jev_First_Standard`'s
  "never as trade execution authorization... the human gate stays" clause at the
  one point where it actually matters (real capital).
- The flag governs **tuning autonomy only**. It is unrelated to, and does not
  affect, Horizon's permanent exclusion from `PortfolioManager`/`EvolutionEngine`
  — that exclusion holds in both modes.

### Go-live criteria (decide before the paper run starts, not after)

Not fully specified yet — flagged here as an open decision for Allan, not a gap in
this spec's scope:

- Minimum paper-run duration before live is even considered.
- Sharpe / max-drawdown bar Horizon must clear.
- What the τ/T_c trajectory must look like (e.g., converged and stable for N days
  vs. still wandering) — reviewed directly from the tuning log in component 5.

### Testing

Following the repo's existing `pytest` convention (`tests/` mirrors `src/`):

- Market state adapter: unit tests against recorded Alpaca bar fixtures — no live
  network calls in tests.
- Decision loop: unit tests mocking the skill's procedure at the module boundary
  (consistent with the Jev integration plan's own correction: "mock at the module
  boundary... rather than rely on an SDK-provided mock model").
- Hard constraint filter: unit tests for each limit (drawdown kill switch, daily
  loss limit) independent of the decision loop.
- Horizon ledger: unit tests for NAV/drawdown computation, isolated from
  `PortfolioManager` (no shared fixtures — this independence is itself a thing
  worth testing, e.g., asserting Horizon's ledger never imports from
  `evo_trader.portfolio.manager`).
- Jev tuning loop: unit tests for both `execution_mode` branches — paper mode
  applies without approval; live mode always escalates regardless of proposed
  delta size.

### Open items (not blocking this spec, but unresolved)

1. Alpaca paper API keys — confirm Allan has these before build starts.
2. `HorizonConfig.tuning_cadence_bars` concrete value — set at build time.
3. Horizon's own risk-limit values, if Allan wants them to differ from the shared
   `RiskConfig` defaults.
4. Go-live criteria (above) — Allan's call, needed before the paper run
   concludes, not before it starts.
