# Evaluation: rummy

Episodes per matchup: 1000 (held-out test seeds)

| Matchup | Metric | Value | 95% CI | Mean return |
|---|---|---|---|---|
| agent vs_heuristic | win_rate | 0.5630 | ±0.0284 | 0.0663 |
| agent vs_random | win_rate | 0.9990 | ±0.0020 | 0.7812 |
| reference: heuristic_vs_heuristic | win_rate | 0.4970 | ±0.0034 | 0.0000 |

## Head-to-head: this agent vs the base agent (500 duplicate deals)

| | win_rate |
|---|---|
| this agent | 0.5130 |
| base agent | 0.4850 |
| difference | +0.0280 ± 0.0448 |

Better: **this agent** (within the margin: not a clear difference)

**Gate:** {"threshold": 0.55} → PASSED
