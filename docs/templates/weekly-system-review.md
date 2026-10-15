# Weekly System Review — Week of {{WEEK_START_DATE}}

**Reviewer:** Ava  
**Review timestamp:** {{REVIEW_TIMESTAMP_UTC}} UTC  
**Evolution cycle ref:** {{EVOLUTION_CYCLE_REPORT_LINK}}

---

## 1. Evolution Cycle Summary

Brief narrative of the Evolution-Controller's Monday run:

- Population changes: {{N_ELIMINATED}} eliminated, {{N_CLONED}} cloned, population now {{N_TOTAL}}
- Overall fitness trend: {{improving / stable / declining}}
- Capital deployed: ${{ACTIVE_CAPITAL}}

---

## 2. Edge-Decay Flags

Strategies flagged by the Evolution-Controller for statistically negligible edge:

{{#each edge_decay_flags}}
- **{{strategy_name}}** — {{status}} (flagged for {{N_PERIODS}} consecutive periods)
{{/each}}

**Recommendation:** {{Ava's read — eliminate next cycle / watch one more period / investigate signal}}

---

## 3. Posture Accuracy

How did Market-Analyst's daily posture calls compare to what actually happened?

| Date | Regime Called | Actual Outcome | Blackout Triggered? | Verdict |
|------|--------------|----------------|---------------------|---------|
| {{DATE}} | {{risk_on/neutral/risk_off}} | {{price action summary}} | {{yes/no}} | {{accurate/miss/overcautious}} |

---

## 4. Risk Events

- **Kill-switch proximity:** Max drawdown reached {{MAX_DD}}% (limit: 15%). {{comfortable / watch / near limit}}
- **Daily loss limit hits this week:** {{N_HITS}} (limit: 2% daily)
- **Blackout windows activated:** {{N_BLACKOUTS}}
- **Posture stale-degradation events:** {{N_STALE}} (posture expired, fell back to risk_dial=0.5)

---

## 5. Horizon Health (observe only — no edits)

- **Decision tick firing rate:** {{N_TICKS}} ticks this week (expected 168)
- **Ledger NAV:** ${{HORIZON_NAV}} (started $10,000)
- **Horizon drawdown:** {{HORIZON_DD}}%
- **τ changes this week:** {{N_TAU_CHANGES}} ({{detail}})
- **Anomalies / missed ticks:** {{none / describe}}

---

## 6. Infrastructure Faults

- **Tactical scheduler missed cycles:** {{N_MISSED}} (launchctl list output attached if >0)
- **Horizon tick missed:** {{N_MISSED}}
- **Error log excerpts:** {{none / paste relevant lines}}

---

## 7. Recommendations for Allan

Items requiring judgment beyond what the system should decide alone:

{{#each recommendations}}
- [ ] **{{title}}**: {{rationale}}
{{/each}}

_None this week_ ← delete if recommendations exist

---

## 8. Build Phase Notes (during burn-in period only)

{{Observations on the paper trading burn-in: data quality, execution latency, unexpected edge cases encountered. Remove this section once the system is past the first month of stable paper operation.}}
