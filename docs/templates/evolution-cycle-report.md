# Evolution Cycle Report — Week of {{WEEK_START_DATE}}

**Run by:** Evolution-Controller  
**Cycle number:** {{CYCLE_NUMBER}}  
**Period:** {{PERIOD_START}} → {{PERIOD_END}}  
**Timestamp:** {{RUN_TIMESTAMP_UTC}} UTC

---

## Population Summary

| Agent | Type | Allocation | Action This Cycle |
|-------|------|-----------|-------------------|
| {{AGENT_NAME}} | {{AGENT_TYPE}} | ${{ALLOCATION}} | {{eliminated / cloned / retained / new}} |

**Population size:** {{N_BEFORE}} → {{N_AFTER}} agents  
**Active capital deployed:** ${{ACTIVE_CAPITAL}}

---

## Eliminations

{{#each eliminated_agents}}
- **{{name}}** (gen {{generation}}) — {{reason}}  
  Last 2 periods: PnL {{period_1_pnl}}% / {{period_2_pnl}}%, edge trend: {{edge_trend}}
{{/each}}

_None_ ← delete this line if any eliminations occurred

---

## Clones

{{#each cloned_agents}}
- **{{parent_name}}** → **{{clone_name}}** (gen {{clone_generation}})  
  Mutations: {{mutated_params}}
{{/each}}

_None_ ← delete this line if no cloning occurred

---

## Edge-Decay Flags

Strategies with edge statistically indistinguishable from zero for ≥2 consecutive windows:

{{#each edge_decay_flags}}
- **{{strategy_name}}**: hit_rate {{hit_rate}}, profit_factor {{profit_factor}}, trend {{edge_trend}} — **FLAGGED FOR REVIEW**
{{/each}}

_None_ ← delete this line if no flags

---

## Capital Rebalance

| Agent | Before | After |
|-------|--------|-------|
| {{AGENT_NAME}} | ${{BEFORE}} | ${{AFTER}} |

---

## Notes for Ava's Review

{{Free-text observations for the weekly system review. Flag anything requiring Allan's judgment (e.g., retiring a strategy family, go-live discussion, raising aggressiveness).}}
