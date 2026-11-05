"""Card-counting features for Donkey (docs/ENV_DESIGN.md §9).

Only public information is used: the player's own hand, discarded tricks, the current
trick, cards everyone saw an opponent pick up, and the suits opponents have shown
they are out of. Unseen cards are assumed spread at random over the opponents'
unknown hand slots.

* Per suit (6 × 4): own / unseen / discarded counts, *lead safety* (the chance every
  remaining opponent still holds the suit, so a lead is discarded), the expected
  number of opponents out of it, and how low the player's lowest card sits among the
  unseen ones.
* Per card (52): the chance that playing it now ends with this player picking up the
  trick. Walks the players still to play: each may cut (the current highest card
  picks up) or follow and overtake.
* 2 scalars: unseen cards, own cards the others know about.
"""
from __future__ import annotations

import numpy as np

from card_games_rl.core.cards import rank_of, suit_of
from card_games_rl.core.counting import p_none

COUNTING_VERSION = 1
SIZE = 6 * 4 + 52 + 2
OVERTAKE_CHANCE = 0.5   # how often a follower holding a higher card plays it


class _Belief:
    """What ``player`` can infer about where the unseen cards are."""

    def __init__(self, game, player: int):
        self.game, self.player = game, player
        seen = set(game.hands[player]) | set(game.discarded) | {c for _, c in game.trick}
        for q in range(4):
            if q != player:
                seen |= game.known[q]
        self.unseen = [c for c in range(52) if c not in seen]
        self.by_suit = [[c for c in self.unseen if suit_of(c) == s] for s in range(4)]

    def slots(self, q: int) -> int:
        return len(self.game.hands[q]) - len(self.game.known[q])

    def pool(self, q: int) -> int:
        return sum(len(self.by_suit[s]) for s in range(4) if not self.game.void[q][s])

    def p_has(self, q: int, suit: int, above: int = 1) -> float:
        """Chance opponent ``q`` holds a card of ``suit`` ranked above ``above``."""
        game = self.game
        if any(suit_of(c) == suit and rank_of(c) > above for c in game.known[q]):
            return 1.0
        if game.void[q][suit] or not game.hands[q]:
            return 0.0
        special = sum(rank_of(c) > above for c in self.by_suit[suit])
        return 1.0 - p_none(self.pool(q), special, self.slots(q))

    def pickup_risk(self, card: int, still_to_play: list[int], current_high: int) -> float:
        """Chance this player ends up highest in the led suit when someone cuts."""
        suit, rank = suit_of(card), rank_of(card)
        if rank < current_high:
            return 0.0
        risk, still_high = 0.0, 1.0
        for q in still_to_play:
            has = self.p_has(q, suit)
            risk += still_high * (1.0 - has)                      # q cannot follow: q cuts
            beats = self.p_has(q, suit, rank) / has if has > 0 else 0.0
            still_high *= has * (1.0 - OVERTAKE_CHANCE * min(1.0, beats))
        return risk


def features(game, player: int) -> np.ndarray:
    belief = _Belief(game, player)
    hand = game.hands[player]
    others = [q for q in range(4) if q != player and game.hands[q]]

    suits = []
    for s in range(4):
        mine = [rank_of(c) for c in hand if suit_of(c) == s]
        unseen = belief.by_suit[s]
        holds = [belief.p_has(q, s) for q in others]
        lowest = min(mine) if mine else None
        below = sum(rank_of(c) < lowest for c in unseen) / max(1, len(unseen)) if lowest else 0.0
        suits += [len(mine) / 13, len(unseen) / 13, sum(suit_of(c) == s for c in game.discarded) / 13,
                  float(np.prod(holds)) if holds else 1.0, sum(1 - h for h in holds) / 3, below]

    risk = np.zeros(52, dtype=np.float32)
    if game.current_player == player and not game.done:
        order = game.trick_players
        position = len(game.trick)
        still_to_play = [q for q in order[position + 1:] if q != player]
        led = game.led_suit
        current_high = max((rank_of(c) for _, c in game.trick if suit_of(c) == led), default=0)
        for card in game.legal_actions():
            if game.trick and suit_of(card) != led:
                continue  # a cut ends the trick and never picks up
            leading = not game.trick
            risk[card] = belief.pickup_risk(card, [q for q in order if q != player] if leading else still_to_play,
                                            0 if leading else current_high)

    scalars = [len(belief.unseen) / 52, len(game.known[player]) / 52]
    return np.concatenate([np.asarray(suits, np.float32), risk, np.asarray(scalars, np.float32)]).astype(np.float32)
