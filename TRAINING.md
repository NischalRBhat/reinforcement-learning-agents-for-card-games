# Training guide

Run these from the project root in PowerShell. They use the project venv directly, so no activation is needed.

## 0. One-time check

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not e2e" -q      # ~1 min, all tests should pass
.\.venv\Scripts\python.exe -m card_games_rl.training.benchmark
```

## 1. Full training (all games in parallel, 3-hour budget)

```powershell
.\.venv\Scripts\python.exe -m card_games_rl.training.train --all --profile full --time-budget 3h
```

Add `--games klondike poker` (for example) to train only some games.

Canasta is the slowest game to train (long hands, four seats), so give it its own longer run:

```powershell
.\.venv\Scripts\python.exe -m card_games_rl.training.train --game canasta --profile full --time-budget 6h
```

Each game's config chooses its method (`algorithm` in `training/configs/<game>.yaml`):

| Game | Method | What happens | Device |
|---|---|---|---|
| Blackjack | MaskablePPO | warm start from the System bot, then PPO | CPU |
| Klondike | MaskablePPO | warm start from the System bot, then PPO | GPU if available |
| Bluff, Hearts, Donkey | MaskablePPO self-play | warm start, then PPO against a pool of past versions plus System bots | CPU |
| Poker | **Monte Carlo CFR** | 4 worker processes; merged, evaluated against the System bot and saved every 20 minutes | CPU |
| Canasta | MaskablePPO self-play | warm start, then PPO; partners share one policy; card counting built into the observation | CPU |
| Rummy | **look-ahead search** | no weights: validated against the System bot and registered in about 3 minutes | CPU |

For the PPO games:
1. **Warm start (a few minutes):** behaviour cloning from the System player. The cloned model is evaluated and saved as the first `best.zip`.
2. **PPO self-play** until the time budget or step cap. The best checkpoint by validation metric is saved continuously, so `Ctrl+C` always leaves a usable model.

For Poker, the best merged strategy is saved to `models/poker/cfr_strategy.pkl.gz`, and `Ctrl+C` keeps the last saved progress (workers save every minute).

- **The time budget is a total per game.** `--resume` only uses what's left of it (tracked in `checkpoints/state.json`, or `cfr/state.json` for Poker).
- **Device:** `--device auto` (the default) uses whichever device the benchmark measured as faster for each game, and the CPU if none was measured. Environment workers don't load PyTorch, so only the trainer processes use the GPU.

To train just some games, or continue after a stop:

```powershell
.\.venv\Scripts\python.exe -m card_games_rl.training.train --all --games poker hearts --profile full --time-budget 2h
.\.venv\Scripts\python.exe -m card_games_rl.training.train --game klondike --profile full --time-budget 1h --resume
```

## 1b. Card-counting agents (a second agent per game)

Donkey, Hearts and Bluff each get a second PPO agent that also sees card-counting features. Rummy gets a counting version of its search. They train into `models/<game>-counting/` and never replace the current agents, so the two can be compared:

```powershell
.\.venv\Scripts\python.exe -m card_games_rl.training.train --all --variant counting --profile full --time-budget 3h
.\.venv\Scripts\python.exe -m card_games_rl.training.evaluate --all --variant counting
```

The evaluation reports each counting agent against the System bot (same gates) and **head-to-head against the current agent** on duplicate deals. The features are described in `docs/ENV_DESIGN.md` §9.

## 2. Monitor (optional, second terminal)

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs     # open http://localhost:6006
Get-Content .\runs\hearts\progress.log -Wait                      # one game's progress
```

## 3. Evaluate on held-out deals

```powershell
.\.venv\Scripts\python.exe -m card_games_rl.training.evaluate --all
```

This prints a summary table and writes `models/<game>/eval_report.md`, which shows the agent against every baseline with 95% confidence intervals and whether the acceptance gate was met. Multi-player games use duplicate deals: every deal is played once from each seat, so card luck cancels out. Poker is tested on 20,000 hands. Klondike's report adds the win rate on deals the solver proved winnable; the first run solves the test deals once (a few minutes) and caches them.

## What gets saved

| Path | Contents |
|---|---|
| `models/<game>/best.zip` | best PPO policy so far (used by evaluation and the web app) |
| `models/poker/cfr_strategy.pkl.gz` | best merged CFR strategy (`models/poker/cfr/` holds the workers' resumable state) |
| `models/<game>/meta.json` | observation version, timesteps, validation metric, gate status |
| `models/<game>/checkpoints/latest.zip` | for `--resume` |
| `models/<game>/pool/` | self-play opponent snapshots (competitive games) |
| `runs/<game>/` | TensorBoard logs and `progress.log` |

## Acceptance gates

These are checked by `evaluate`; the details are in `docs/ENV_DESIGN.md`.

| Game | Metric | Gate |
|---|---|---|
| Blackjack | EV per hand | within 0.01 of the exact optimal policy (≈ −0.5%) |
| Klondike | win rate | ≥ 60% (random ≈ 40%, heuristic ≈ 51%) |
| Poker | bb/100 vs heuristic | ≥ +10 |
| Rummy | win rate vs heuristic | ≥ 55% |
| Bluff | win rate vs 3 heuristics | ≥ 30% (fair share 25%) |
| Hearts | points per hand vs 3 heuristics | ≤ 6.0 (fair share 6.5) |
| Donkey | share of games as the Donkey vs 3 heuristics | ≤ 18% (fair share 25%) |
| Canasta | points per hand above a System partnership | ≥ +150 (System vs System = 0) |

## Hyperparameters

Each game's network size, PPO settings, and smoke/full profiles live in `card_games_rl/training/configs/<game>.yaml`. Change them there rather than in code.
