# Evaluation: klondike

Episodes per matchup: 500 (held-out test seeds)

| Matchup | Metric | Value | 95% CI | Mean return |
|---|---|---|---|---|
| agent solo | win_rate | 0.3940 | ±0.0429 | 0.4491 |
| reference: heuristic | win_rate | 0.5120 | ±0.0439 | 0.5571 |
| reference: random | win_rate | 0.3880 | ±0.0428 | 0.4413 |

Solver on the test deals: {'not winnable': 11, 'unknown': 109, 'winnable': 380} (proven winnable: 76.0%)

| Player | Win rate on proven-winnable deals |
|---|---|
| agent | 0.4842 |
| heuristic | 0.6158 |
| random | 0.4632 |

**Gate:** {"threshold": 0.6} → NOT MET
