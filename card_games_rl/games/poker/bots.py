"""Hold'em baselines: an equity/pot-odds heuristic and a calling station."""
from __future__ import annotations

import json
import random
from pathlib import Path

from card_games_rl.core import Game, Policy
from card_games_rl.core.cards import rank_of, suit_of

from .evaluator import evaluate
from .rules import ALL_IN, CALL, FOLD, RAISE_HALF, RAISE_POT


_PREFLOP_TABLE = Path(__file__).with_name("preflop_equity.json")
# 169 preflop classes, precomputed (1,500 Monte Carlo samples each) so workers start instantly.
_PREFLOP_CACHE: dict[tuple[int, int, bool], float] = {
    (int(h), int(l), bool(int(s))): v
    for key, v in (json.loads(_PREFLOP_TABLE.read_text()) if _PREFLOP_TABLE.exists() else {}).items()
    for h, l, s in [key.split(",")]
}


def preflop_equity(hole: list[int]) -> float:
    """Equity vs a random hand for one of the 169 preflop classes (cached, deterministic)."""
    high, low = sorted((rank_of(c) for c in hole), reverse=True)
    key = (high, low, suit_of(hole[0]) == suit_of(hole[1]))
    if key not in _PREFLOP_CACHE:
        _PREFLOP_CACHE[key] = _monte_carlo(hole, [], random.Random(hash(key) & 0xFFFF), 1500)
    return _PREFLOP_CACHE[key]


def equity_vs_random(hole: list[int], board: list[int], rng: random.Random, samples: int = 64) -> float:
    """Win probability (ties = ½) against one uniformly random hand."""
    if not board:
        return preflop_equity(hole)
    return _monte_carlo(hole, board, rng, samples)


def _monte_carlo(hole: list[int], board: list[int], rng: random.Random, samples: int) -> float:
    used = set(hole) | set(board)
    remaining = [c for c in range(52) if c not in used]
    missing = 5 - len(board)
    wins = 0.0
    for _ in range(samples):
        drawn = rng.sample(remaining, 2 + missing)
        full = board + drawn[2:]
        mine, theirs = evaluate(hole + full), evaluate(drawn[:2] + full)
        wins += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
    return wins / samples


class HeuristicPolicy(Policy):
    """Equity-driven player.

    * Facing a bet: call when equity beats the pot odds (with a small margin
      that grows with bet size, as opponents' betting ranges are stronger
      than random), otherwise fold.
    * Value-raise strong equity, occasionally bluff-raise when checked to.
    """

    name = "heuristic"

    def __init__(self, seed: int | None = None, samples: int = 64):
        self.rng = random.Random(seed)
        self.samples = samples

    def act(self, game: Game, player: int) -> int:
        legal = game.legal_actions()
        equity = equity_vs_random(game.hole[player], game.board, self.rng, self.samples)
        call = game.to_call(player)
        pot_odds = call / (game.pot + call) if call else 0.0
        value_raise = RAISE_POT if RAISE_POT in legal else (RAISE_HALF if RAISE_HALF in legal else None)
        if equity > 0.82 and ALL_IN in legal and game.board_visible >= 3:
            return ALL_IN
        if equity > 0.68 and value_raise is not None:
            return value_raise
        if call == 0:
            if equity < 0.35 and RAISE_HALF in legal and self.rng.random() < 0.12:
                return RAISE_HALF  # bluff
            return CALL
        margin = 0.04 + 0.12 * min(1.0, call / max(1, game.pot - call))
        if equity >= pot_odds + margin:
            return CALL
        return FOLD


class CallingStation(Policy):
    name = "calling_station"

    def __init__(self, seed: int | None = None):
        pass

    def act(self, game: Game, player: int) -> int:
        return CALL
