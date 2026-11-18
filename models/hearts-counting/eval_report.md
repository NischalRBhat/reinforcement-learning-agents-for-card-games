# Evaluation: hearts

Episodes per matchup: 2000 (held-out test seeds)

| Matchup | Metric | Value | 95% CI | Mean return |
|---|---|---|---|---|
| agent vs_heuristic | mean_points | 4.9945 | ±0.2946 | 0.0762 |
| agent vs_random | mean_points | 2.2460 | ±0.2305 | 0.1766 |
| reference: heuristic_vs_heuristic | mean_points | 6.7860 | ±0.1673 | -0.0000 |

## Head-to-head: this agent vs the base agent (500 duplicate deals)

| | mean_points |
|---|---|
| this agent | 6.6130 |
| base agent | 6.9850 |
| difference | -0.3720 ± 0.4588 |

Better: **this agent** (within the margin: not a clear difference)

**Gate:** {"threshold": 6.0} → PASSED
