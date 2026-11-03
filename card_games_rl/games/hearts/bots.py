"""Rule-based Hearts player (the "System" opponent)."""
from __future__ import annotations

import random

from card_games_rl.core import Game, Policy
from card_games_rl.core.cards import HEARTS, SPADES, rank_of, suit_of

from .rules import QUEEN_SPADES, card_points


class HeuristicPolicy(Policy):
    name = "heuristic"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    @staticmethod
    def _pass_danger(card: int, hand: list[int]) -> float:
        suit, rank = suit_of(card), rank_of(card)
        if card == QUEEN_SPADES:
            return 100
        if suit == SPADES and rank > 12:
            return 80 if QUEEN_SPADES not in hand else 40
        suit_len = sum(suit_of(c) == suit for c in hand)
        base = 30 + rank if suit == HEARTS else rank
        return base + (13 - suit_len) * 1.5

    def act(self, game: Game, player: int) -> int:
        legal = game.legal_actions()
        hand = game.hands[player]
        if game.passing:
            return max(legal, key=lambda c: self._pass_danger(c, hand))
        if len(legal) == 1:
            return legal[0]
        if not game.trick:
            return self._lead(game, legal, hand)
        lead = suit_of(game.trick[0][1])
        winning = max((c for _, c in game.trick if suit_of(c) == lead), key=rank_of)
        points_on_table = sum(card_points(c) for _, c in game.trick)
        if suit_of(legal[0]) == lead and all(suit_of(c) == lead for c in legal):
            if QUEEN_SPADES in legal and rank_of(winning) > 12:
                return QUEEN_SPADES
            under = [c for c in legal if rank_of(c) < rank_of(winning)]
            if under:
                return max(under, key=rank_of)  # duck as high as possible
            if len(game.trick) == 3 and points_on_table == 0:
                safe = [c for c in legal if c != QUEEN_SPADES]
                return max(safe or legal, key=rank_of)  # we win anyway: shed the highest
            safe = [c for c in legal if c != QUEEN_SPADES]
            return min(safe or legal, key=rank_of)
        # Void in the led suit: dump danger.
        if QUEEN_SPADES in legal:
            return QUEEN_SPADES
        high_spades = [c for c in legal if suit_of(c) == SPADES and rank_of(c) > 12]
        if high_spades:
            return max(high_spades, key=rank_of)
        hearts = [c for c in legal if suit_of(c) == HEARTS]
        if hearts:
            return max(hearts, key=rank_of)
        return max(legal, key=rank_of)

    @staticmethod
    def _lead(game: Game, legal: list[int], hand: list[int]) -> int:
        def key(card: int):
            suit = suit_of(card)
            suit_len = sum(suit_of(c) == suit for c in hand)
            risky_spade = suit == SPADES and (rank_of(card) >= 12 or QUEEN_SPADES in hand)
            return (risky_spade, suit == HEARTS, rank_of(card), suit_len)

        return min(legal, key=key)
