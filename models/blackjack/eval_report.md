# Evaluation: blackjack

Episodes per matchup: 50000 (held-out test seeds)

| Matchup | Metric | Value | 95% CI | Mean return |
|---|---|---|---|---|
| agent solo | ev | -0.0006 | ±0.0096 | -0.0006 |
| reference: optimal | ev | 0.0013 | ±0.0100 | 0.0013 |
| reference: heuristic | ev | -0.0497 | ±0.0086 | -0.0497 |
| reference: random | ev | -0.4314 | ±0.0104 | -0.4314 |

**Gate:** {"baseline": "optimal", "max_gap": 0.01} → PASSED
