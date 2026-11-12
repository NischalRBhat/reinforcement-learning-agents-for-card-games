# Evaluation: bluff

Episodes per matchup: 1000 (held-out test seeds)

| Matchup | Metric | Value | 95% CI | Mean return |
|---|---|---|---|---|
| agent vs_heuristic | win_rate | 0.7280 | ±0.0272 | 0.6373 |
| agent vs_random | win_rate | 0.5020 | ±0.0309 | 0.3360 |
| reference: heuristic_vs_heuristic | win_rate | 0.2550 | ±0.0248 | 0.0067 |

## Head-to-head: this agent vs the base agent (250 duplicate deals)

| | win_rate |
|---|---|
| this agent | 0.4850 |
| base agent | 0.0150 |
| difference | +0.4700 ± 0.0147 |

Better: **this agent**

**Gate:** {"threshold": 0.3} → PASSED
