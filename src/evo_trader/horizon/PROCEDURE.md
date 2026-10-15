# Causal Entropic Optionality — Procedure (vendored)

Source: `causal-entropic-optionality` skill (Wissner-Gross & Freer, PRL 110, 168702, 2013).
This file is Horizon's authoritative reference for its decision objective.
Do not edit. Update by re-vendoring from the skill.

## Core idea

Push the system toward whichever present state keeps the **largest diversity of future
paths open**, evaluated out to a horizon τ.

Formally: `F(X₀, τ) = T_c ∇_X S_c(X, τ)|_{X₀}`, where `S_c` is the path
entropy (log-diversity of accessible future paths) of macrostate X over
horizon τ, and `T_c` is a strength parameter. Two free parameters only:
horizon τ (how far ahead to look) and strength T_c (how much this objective
dominates vs. other goals/constraints).

## The procedure (run at each decision point)

1. **Enumerate candidate actions** at the current state (typically 3-7).
2. **Roll forward τ steps per candidate** — for each candidate, work out what states
   become reachable across a few plausible branches. Depth and branching are judgment
   calls: deeper/wider for high-stakes/slow-changing domains.
3. **Score each candidate by reachable-future diversity**, not by expected value.
   A candidate that forecloses future branches scores worse than one that keeps many
   viable continuations open.
4. **Apply hard constraints** (risk limits, capital constraints) as filters before or
   after scoring — these always take precedence over the optionality score.
5. **Pick the action maximizing future diversity** subject to constraints, and act.
6. **Repeat fresh at the next decision point.** Don't treat step 5's winner as locked
   in — re-enumerate and re-score as new information arrives.

## Execution mode B — single-LLM virtual rollout (Horizon's default)

For each candidate, write out (even briefly) the 2-3 most plausible next branches
and what they in turn open up or foreclose. This must be done per candidate, not
skipped in favor of intuition — the value is in comparing enumerated futures.

Keep breadth × depth small enough to stay legible. Conclude with an explicit
comparison across candidates before picking.

## Applying to trading

- **Candidate actions**: legal moves at this decision point (`enter_long`, `enter_short`,
  `exit`, `resize`, `hold`, `hedge`)
- **"Future diversity"**: breadth of viable follow-up positions/strategies reachable
  from each candidate, given plausible market branches — not raw expected return.
- **Hard constraints**: risk limits, capital constraints — always filter/override the
  optionality score, never compete with it as just another scored factor.
