"""Card-counting features for Hearts (docs/ENV_DESIGN.md §9).

Public information only: the player's own hand, every card played, the cards it
passed (it knows who holds them until they are played), and the suits opponents
have shown they are out of. Unseen cards are assumed spread at random over the
opponents' unknown hand slots.

* Per suit (6 × 4): own / unseen / played counts, the chance a lead gets cut by
  someone out of the suit, and how many unseen cards are above the player's lowest
  (room to duck) and below its highest (risk of winning) in that suit.
* Q♠ (7): where it is (own hand, played, passed by the player, unseen) and the
  chance each opponent holds it.
* Points (5): hearts unseen, points still out, the leading opponent's points, a
  moon-shot warning (one opponent has taken every point so far), points in the
  current trick.
* Per card (52): the chance that playing it now wins the trick.
"""
from __future__ import annotations

import numpy as np

from card_games_rl.core.cards import HEARTS, rank_of, suit_of
from card_games_rl.core.counting import p_none

from .rules import PASS_NONE, PASS_OFFSETS, QUEEN_SPADES, card_points

COUNTING_VERSION = 1
SIZE = 6 * 4 + 7 + 5 + 52
OVERTAKE_CHANCE = 0.5   # how often a later player holding a higher card plays it


class _Belief:
    def __init__(self, game, player: int):
        self.game, self.player = game, player
        hand = set(game.hands[player])
        self.known = {q: set() for q in range(4)}   # cards the player knows an opponent holds
        if not game.passing and game.direction != PASS_NONE:
            target = (player + PASS_OFFSETS[game.direction]) % 4
            self.known[target] = {c for c in game.pass_picks[player] if c not in game.played}
        located = hand | game.played | self.known[(player + 1) % 4] | self.known[(player + 2) % 4] | self.known[(player + 3) % 4]
        self.unseen = [c for c in range(52) if c not in located]
        self.by_suit = [[c for c in self.unseen if suit_of(c) == s] for s in range(4)]

    def slots(self, q: int) -> int:
        return len(self.game.hands[q]) - len(self.known[q])

    def pool(self, q: int) -> int:
        return sum(len(self.by_suit[s]) for s in range(4) if not self.game.voids[q][s])

    def p_holds(self, q: int, cards: list[int], suit: int) -> float:
        """Chance opponent ``q`` holds at least one of ``cards`` (all of ``suit``)."""
        if self.known[q] & set(cards):
            return 1.0
        if self.game.voids[q][suit] or not self.game.hands[q]:
            return 0.0
        unseen = [c for c in cards if c in self.by_suit[suit]]
        return 1.0 - p_none(self.pool(q), len(unseen), self.slots(q))

    def p_has_above(self, q: int, suit: int, rank: int) -> float:
        cards = [c for c in range(52) if suit_of(c) == suit and rank_of(c) > rank]
        return self.p_holds(q, cards, suit)


def features(game, player: int) -> np.ndarray:
    belief = _Belief(game, player)
    hand = game.hands[player]
    others = [(player + i) % 4 for i in (1, 2, 3)]

    suits = []
    for s in range(4):
        mine = [rank_of(c) for c in hand if suit_of(c) == s]
        unseen = belief.by_suit[s]
        all_cut = 1.0 - float(np.prod([belief.p_has_above(q, s, 1) for q in others]))
        above_low = sum(rank_of(c) > min(mine) for c in unseen) / max(1, len(unseen)) if mine else 0.0
        below_high = sum(rank_of(c) < max(mine) for c in unseen) / max(1, len(unseen)) if mine else 0.0
        suits += [len(mine) / 13, len(unseen) / 13, sum(suit_of(c) == s for c in game.played) / 13,
                  all_cut, above_low, below_high]

    where = [QUEEN_SPADES in hand, QUEEN_SPADES in game.played,
             any(QUEEN_SPADES in belief.known[q] for q in others), QUEEN_SPADES in belief.unseen]
    queen = [float(x) for x in where] + [belief.p_holds(q, [QUEEN_SPADES], 3) for q in others]

    hearts_unseen = len(belief.by_suit[HEARTS])
    points_out = 26 - sum(card_points(c) for c in game.played) - sum(card_points(c) for c in hand)
    taken = [game.points[q] for q in others]
    total_taken = sum(game.points)
    moon_risk = float(total_taken > 0 and max(taken) == total_taken)
    in_trick = sum(card_points(c) for _, c in game.trick)
    points = [hearts_unseen / 13, points_out / 26, max(taken) / 26, moon_risk, in_trick / 26]

    win = np.zeros(52, dtype=np.float32)
    if not game.passing and not game.done and game.current_player == player:
        led = suit_of(game.trick[0][1]) if game.trick else None
        current_high = max((rank_of(c) for _, c in game.trick if suit_of(c) == led), default=0)
        later = [(player + i) % 4 for i in range(1, 4 - len(game.trick))]
        for card in game.legal_actions():
            suit, rank = suit_of(card), rank_of(card)
            if game.trick and (suit != led or rank < current_high):
                continue  # off-suit or under the current winner: cannot win
            p = 1.0
            for q in later:
                p *= 1.0 - OVERTAKE_CHANCE * belief.p_has_above(q, suit, rank)
            win[card] = p

    return np.concatenate([np.asarray(suits + queen + points, np.float32), win]).astype(np.float32)
