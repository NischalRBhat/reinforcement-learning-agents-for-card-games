"""Rule-based Canasta player (the "System" opponent and partner).

* **Draw:** take the pile when it is worth it (four or more cards, or it extends a meld);
  otherwise draw from the stock.
* **Meld:** lay down natural cards whenever possible. Spend wild cards only to complete a
  canasta, to reach the opening minimum, or to go out.
* **Discard:** a black three first (it blocks the next player), then the least useful
  natural card: never one the opponents can add to an unfrozen pile, preferring
  singletons and low values. Wild cards are thrown only to freeze a big pile against
  opponents who have melded.
"""
from __future__ import annotations

import random

from card_games_rl.core import Game, Policy

from .rules import (BLACK3, CANASTA_SIZE, DISCARD_BASE, DRAW, JOKER, MELD_BASE, TAKE_ADD, TAKE_MIXED,
                    TAKE_PAIR, TWO, VALUES, WILD, CanastaGame, decode_meld)


class HeuristicPolicy(Policy):
    name = "system"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, game: Game, player: int) -> int:
        legal = game.legal_actions()
        if game.phase == "draw":
            return self._draw(game, legal)
        melds = [a for a in legal if MELD_BASE <= a < DISCARD_BASE]
        best, best_score = None, 0.0
        for action in melds:
            score = self._meld_score(game, player, action)
            if score > best_score:
                best, best_score = action, score
        if best is not None:
            return best
        discards = [a for a in legal if a >= DISCARD_BASE]
        if discards:
            return max(discards, key=lambda a: self._discard_score(game, player, a - DISCARD_BASE))
        return melds[0] if melds else legal[0]

    @staticmethod
    def _draw(game: CanastaGame, legal: list[int]) -> int:
        takes = [a for a in (TAKE_ADD, TAKE_PAIR, TAKE_MIXED) if a in legal]
        size = len(game.pile)
        if takes and (TAKE_ADD in takes or size >= 4 or DRAW not in legal):
            if TAKE_MIXED in takes and len(takes) == 1 and size < 6 and DRAW in legal:
                return DRAW  # not worth a wild card
            return takes[0]
        return DRAW if DRAW in legal else takes[0]

    @staticmethod
    def _meld_score(game: CanastaGame, player: int, action: int) -> float:
        kind, n, j, d = decode_meld(action)
        team = game.team(player)
        if kind == BLACK3:
            return 1000.0  # only legal when it lets the player go out
        hand_after = len(game.hands[player]) - (n + j + d)
        existing = game.melds[team].get(kind)
        size_after = (game.meld_size(existing) if existing else 0) + n + j + d
        initial = not game.team_melded[team]
        if hand_after == 0:
            return 900.0  # goes out
        if j or d:
            completes = existing is not None and game.meld_size(existing) < CANASTA_SIZE <= size_after
            if not (initial or completes):
                return 0.0
            return 50.0 + n * VALUES[kind] - 0.2 * (j * VALUES[JOKER] + d * VALUES[TWO])
        score = 10.0 + n * VALUES[kind] + (40.0 if size_after >= CANASTA_SIZE else 0.0)
        return score + (5.0 if initial else 0.0)

    @staticmethod
    def _discard_score(game: CanastaGame, player: int, kind: int) -> float:
        opponents = 1 - game.team(player)
        own = game.team(player)
        count = game.count(player, kind)
        if kind == BLACK3:
            return 100.0
        if kind in WILD:
            # Freeze a big pile when the opponents could otherwise take it.
            return 40.0 if len(game.pile) >= 10 and game.team_melded[opponents] else -200.0
        score = -VALUES[kind]
        if kind in game.melds[opponents] and not game.pile_frozen:
            score -= 300.0
        if kind in game.melds[own]:
            score -= 30.0
        score += 15.0 if count == 1 else -10.0 * (count - 1)
        return score
