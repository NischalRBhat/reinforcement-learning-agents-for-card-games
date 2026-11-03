"""Blackjack baselines. The real "System" opponent is the dealer inside the engine."""
from __future__ import annotations

from card_games_rl.core import Game, Policy

from .rules import HIT, NO_INSURE, STAND, hand_value


class HeuristicPolicy(Policy):
    """Mimic the dealer: hit below 17, never double/split/surrender/insure."""

    name = "heuristic"

    def __init__(self, seed: int | None = None):
        pass

    def act(self, game: Game, player: int) -> int:
        if game.insurance_phase:
            return NO_INSURE
        return HIT if hand_value(game.hands[game.active])[0] < 17 else STAND
