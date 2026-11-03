"""Greedy-deadwood Rummy player (the "System" opponent)."""
from __future__ import annotations

from card_games_rl.core import Game, Policy

from .melds import solver_for
from .rules import DISCARD, DRAW_DISCARD, DRAW_STOCK


class HeuristicPolicy(Policy):
    """Take the discard only if it lowers deadwood; discard to minimise deadwood."""

    name = "heuristic"

    def __init__(self, seed: int | None = None):
        pass

    def act(self, game: Game, player: int) -> int:
        legal = game.legal_actions()
        hand = game.hands[player]
        if game.phase == "draw":
            if DRAW_DISCARD in legal:
                top = game.discard[-1]
                current = solver_for(hand).solve()
                after, card = solver_for(hand + [top]).best_discard(forbidden=top)
                if after < current:
                    return DRAW_DISCARD
            return DRAW_STOCK if DRAW_STOCK in legal else DRAW_DISCARD
        _, card = solver_for(hand).best_discard(forbidden=game.drawn_from_discard)
        return DISCARD + card
