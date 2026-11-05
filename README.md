# Card Games RL

Eight card games with correct rules engines, AI agents trained by reinforcement learning, game-theory solving and search, and a browser table where you can play them, play against them or watch them play each other.

**Games:** Blackjack · Klondike Solitaire · Texas Hold'em (heads-up no-limit) · Rummy · Bluff · Hearts · Donkey · Canasta

Each game has a hand-written **System** bot as a baseline and a trained **agent**. The agent's method is chosen to suit the game:

| Game | Players | Agent | Held-out test result (95% CI) | Target |
|---|---|---|---|---|
| Blackjack | 1 vs dealer | PPO (masked) | −0.0006 ± 0.0096 EV/hand; optimal play scores +0.0013 on the same hands | ✅ within 0.01 of optimal |
| Klondike Solitaire | 1 | PPO (masked), GPU | 39.4% ± 4.3 of deals won (System bot 51.2%) | ❌ ≥ 60% |
| Texas Hold'em | 2 | Monte Carlo CFR | −14.2 ± 24.8 bb/100 vs the System bot | ❌ ≥ +10 |
| Rummy | 2 | Look-ahead search | 54.9% ± 2.9 wins vs the System bot (bot vs itself 49.7%) | ❌ ≥ 55% (inside the margin) |
| Bluff | 4 | PPO self-play | 89.9% ± 1.8 wins vs 3 System bots (fair share 25%) | ✅ ≥ 30% |
| Hearts | 4 | PPO self-play | 5.55 ± 0.32 points per hand vs 3 System bots (bot vs itself 6.79) | ✅ ≤ 6.0 |
| Donkey | 4 | PPO self-play | the Donkey in 16.5% ± 1.5 of games vs 3 System bots (fair share 25%) | ✅ ≤ 18% |
| Canasta | 4 (partnerships) | PPO self-play + card counting | not trained yet | target +150 points/hand |

These come from `evaluate --all` on test deals that training never saw. Multi-player games use **duplicate deals**: each deal is replayed with the agent in every seat, so the luck of the cards cancels out. Poker scored +25.9 bb/100 on its validation hands during training. Its test result is lower and very noisy (hands in which both players randomise vary a lot), so it needs a longer run or more test hands to be settled. Klondike's agent still plays worse than the System bot.

The full rules, how to play each game, and how each bot and agent works are in **[docs/GAMES.md](docs/GAMES.md)**.

---

## Features

- **Rules engines** for all eight games: pure Python, seeded and deterministic, with masked legal actions and hidden information enforced per player. One engine drives training, evaluation, the API and the UI.
- **Gymnasium environments** with a learner seat and configurable opponents, including a self-play pool of past agent versions.
- **Agents:**
  - MaskablePPO with behaviour-cloning warm starts;
  - external-sampling MCCFR for poker;
  - an exact look-ahead search for Rummy;
  - an exact dynamic-programming optimal strategy as the Blackjack reference;
  - a perfect-information Klondike solver.
- **Rewards** are the real game score plus potential-based shaping only, which cannot be farmed or change the best strategy.
- **Training pipeline:**
  - YAML configs with smoke and full profiles;
  - all games trained in parallel;
  - a time budget that holds across `--resume`;
  - TensorBoard logs;
  - evaluation reports with 95% confidence intervals and acceptance gates.
- **Web app:**
  - Human vs System, Human vs Agent (side-by-side duels for Blackjack and Klondike), and Agent vs Agent with any mix of seats;
  - an agent insight panel showing the agent's top choices with probabilities;
  - in-app rules, playback controls, scoreboard, move history and keyboard shortcuts.
- **Tests:** about 270 in all: rules scenarios, randomized fuzzing of game invariants, environment checks, training smoke runs, API play-throughs, and browser end-to-end tests.

## Card-counting agents

Donkey, Hearts, Bluff and Rummy each have a **second agent that counts cards**. Using only what a player may legally know, it works out:
- which suits are still safe to lead;
- the chance a card ends up picking up the trick (Donkey) or winning it (Hearts);
- how many copies of a claimed rank can still exist (Bluff);
- what the opponent is collecting (Rummy).

The counting agents are trained alongside the current ones, not instead of them. In the app you can seat both at one table with the **Agent vs Counting agent** mode, and `evaluate --variant counting` reports a head-to-head on duplicate deals. Details: [docs/ENV_DESIGN.md](docs/ENV_DESIGN.md) §9.

## Quick start

Requires Python 3.10+ (developed on 3.14, Windows). Commands are for PowerShell from the project folder.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe launch_card_games.py        # opens http://127.0.0.1:8000
```

`launch_card_games.bat` does the same with a double-click. On macOS or Linux, use `.venv/bin/python` instead.

Trained model weights are not stored in the repository. Until you train, agent seats are played by the System bot and labelled "(System)". Rummy's search agent works straight away, because it has no weights.

## Playing

1. Pick a game in the left rail. Its start page shows a short how-to-play and the agent's results.
2. Choose a mode. For the four-player games, set each seat to *You*, *System bot*, *Trained agent* or *Random*.
3. Press **Start game**.

| | Human vs System | Human vs Agent | Agent vs Agent |
|---|---|---|---|
| Blackjack | You vs the dealer | Duel: you and the agent on the same shoe | Watch the agent |
| Klondike | Play solo | Duel on the same deal | Watch the agent |
| Poker, Rummy | You vs the System bot | You vs the agent | Agent vs agent |
| Bluff, Hearts, Donkey, Canasta | You + 3 System bots | You + 3 agents | 4 agents (any mix) |

**How to play** (button or H) opens each game's full rules beside the table. **Agent insight** shows what each computer player chose and the agent's top alternatives. At a table with you, those alternatives are hidden, because they could reveal the agent's cards.

**Shortcuts:** Space pauses or resumes, S steps one move, N deals the next hand, R restarts, H opens How to play, and ? shows help.

## Training and evaluation

```powershell
# quick check that the pipeline works (minutes)
.\.venv\Scripts\python.exe -m card_games_rl.training.train --game hearts --profile smoke

# full training of every game in parallel, 3-hour budget per game
.\.venv\Scripts\python.exe -m card_games_rl.training.train --all --profile full --time-budget 3h

# held-out evaluation and reports (models/<game>/eval_report.md)
.\.venv\Scripts\python.exe -m card_games_rl.training.evaluate --all
```

[TRAINING.md](TRAINING.md) covers options, resuming, monitoring, saved files and the acceptance gates.

## How it works

```
                ┌──────────── rules engine (one per game) ────────────┐
                │  legal actions · hidden info · score · observation  │
                └──────┬──────────────────┬───────────────────┬───────┘
                       │                  │                   │
          Gymnasium env + opponents   evaluation        FastAPI sessions
          (self-play pool, bots)      (duplicate deals)  (seat controllers)
                       │                  │                   │
              PPO / CFR / search ──► model registry ──► browser table
```

- **Engines** (`card_games_rl/games/<game>/rules.py`) implement a common `Game` contract (`core/game.py`). Every action is masked, and card actions use card ids (0–51), so an action means the same thing whatever the hand size. Observations contain only what that player may know.
- **Environments** (`envs/`) put the learner in one seat and play the other seats with opponent policies. Rewards from opponents' turns are credited correctly.
- **Agents** (`agents/`) load from `models/<game>/` through a registry. The registry refuses a model trained on an older observation layout.
- **The API** (`api/`) runs seat-based sessions. Each seat is controlled by a human, the System bot, the agent or a random player. All game logic stays in the engines.
- **The web app** (`web/`) is plain HTML, CSS and JavaScript modules with no build step.

The full technical specification (observation layouts, action spaces, reward tables, algorithm choices) is [docs/ENV_DESIGN.md](docs/ENV_DESIGN.md).

## Project structure

```
card_games_rl/
  core/          cards, Game contract, Policy interface, seeds, counting maths
  encoders.py    observation encoders: base, and base + card counting
  games/<game>/  rules.py (engine + observation), bots.py (System bot), counting.py (counting features), plus blackjack/optimal.py,
                 klondike/solver.py, poker/cfr.py, rummy/search.py
  envs/          Gymnasium single-agent wrapper, self-play opponent pool
  agents/        trained-model policy, model registry
  training/      configs/<game>.yaml, train.py, evaluate.py, benchmark.py
  api/           app.py (routes + static files), sessions.py (play sessions)
web/             index.html, styles.css, js/ (app shell, guides, one renderer per game)
tests/           rules, fuzz invariants, envs, training, API, e2e (browser)
docs/            GAMES.md (the games), ENV_DESIGN.md (technical spec)
models/          trained agents (weights are git-ignored; metadata and reports are kept)
```

## Web API

| Endpoint | Purpose |
|---|---|
| `GET /api/games` | Catalog: modes, seat controllers, agent method, results and target |
| `POST /api/sessions` | Start a game: `{game, mode, seats?, seed?, reveal?}`, e.g. `seats: [["human","agent","system","random"]]` |
| `GET /api/sessions/{id}` | Current state, with hidden cards removed for the viewer |
| `POST /api/sessions/{id}/action` | Human move: `{board, action}`, or `{board, cards, claim_rank?}` for Bluff |
| `POST /api/sessions/{id}/advance` | Play up to `steps` computer moves; each includes the agent's top choices |
| `POST /api/sessions/{id}/new-round` | Next deal (Hearts rotates the pass, Poker alternates the button) |
| `POST /api/sessions/{id}/undo`, `/solve` | Klondike: undo a move; check whether the finished deal was winnable |

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not e2e" -q     # unit, fuzz, env, training, API (~2–3 min)
.\.venv\Scripts\python.exe -m pytest -m e2e              # browser tests through Playwright
```

The browser tests use the installed Microsoft Edge on Windows. Elsewhere, run `python -m playwright install chromium` once.

## Adding a game

1. Create `card_games_rl/games/<id>/` with a `Game` subclass (`rules.py`) and a `HeuristicPolicy` (`bots.py`), and export `GAME_CLASS` and `HeuristicPolicy` from its `__init__.py`.
2. Register a `GameSpec` in `card_games_rl/games/__init__.py`.
3. Add `card_games_rl/training/configs/<id>.yaml` and a metric and baselines entry in `training/evaluate.py`.
4. Add `web/js/renderers/<id>.js`, register it in `web/js/renderers/index.js`, and add its rules to `web/js/guide.js`.
5. The generic test suites then cover it automatically (invariants, API play-through, browser). Add rules tests for its special cases.
