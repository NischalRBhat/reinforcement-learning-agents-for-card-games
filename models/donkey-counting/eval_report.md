# Evaluation: donkey

Episodes per matchup: 2000 (held-out test seeds)

| Matchup | Metric | Value | 95% CI | Mean return |
|---|---|---|---|---|
| agent vs_heuristic | donkey_rate | 0.1325 | ±0.0133 | 0.1567 |
| agent vs_random | donkey_rate | 0.0215 | ±0.0063 | 0.3045 |
| reference: heuristic_vs_heuristic | donkey_rate | 0.2500 | ±0.0000 | -0.0000 |

## Head-to-head: this agent vs the base agent (500 duplicate deals)

| | donkey_rate |
|---|---|
| this agent | 0.2405 |
| base agent | 0.2595 |
| difference | -0.0190 ± 0.0272 |

Better: **this agent** (within the margin: not a clear difference)

**Gate:** {"threshold": 0.18} → PASSED
