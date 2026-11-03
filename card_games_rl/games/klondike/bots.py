"""Priority-rule Klondike player used as the baseline."""
from __future__ import annotations

import random

from card_games_rl.core import Game, Policy

from .rules import (DRAW, FOUNDATION_TO_TABLEAU, RECYCLE, TABLEAU_TO_FOUNDATION, TABLEAU_TO_TABLEAU,
                    WASTE_TO_FOUNDATION, WASTE_TO_TABLEAU, decode_move, krank)


class HeuristicPolicy(Policy):
    """Priorities: foundation > reveal a hidden card > free a foundation card >
    play the waste > draw/recycle; explores tableau moves when stalled."""

    name = "heuristic"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, game: Game, player: int) -> int:
        legal = game.legal_actions()
        foundation = [a for a in legal if a == WASTE_TO_FOUNDATION or TABLEAU_TO_FOUNDATION <= a < FOUNDATION_TO_TABLEAU]
        if foundation:
            return foundation[0]
        reveals, frees, others = [], [], []
        for action in (a for a in legal if a >= TABLEAU_TO_TABLEAU):
            src, dst, n = decode_move(action)
            up, pile = game.face_up(src), game.tableau[src]
            if n == up and game.hidden[src] > 0:
                reveals.append((game.hidden[src], action))
            elif n < up and game.can_found(pile[-n - 1]):
                frees.append(action)
            elif not (n == up and krank(pile[-n]) == 13 and not game.tableau[dst]):
                others.append(action)
        if reveals:
            return max(reveals)[1]
        if frees:
            return frees[0]
        waste_moves = [a for a in legal if WASTE_TO_TABLEAU <= a < TABLEAU_TO_FOUNDATION]
        if waste_moves:
            non_empty = [a for a in waste_moves if game.tableau[a - WASTE_TO_TABLEAU]]
            if non_empty or krank(game.waste[-1]) == 13:
                return (non_empty or waste_moves)[0]
        if others and game.stall > 30 and self.rng.random() < 0.3:
            return self.rng.choice(others)
        if DRAW in legal:
            return DRAW
        if RECYCLE in legal:
            return RECYCLE
        return self.rng.choice(legal)
