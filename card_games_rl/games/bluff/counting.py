"""Card-counting features for Bluff (docs/ENV_DESIGN.md §9).

Public information only: the player's own hand, its own cards in the pile and among
discarded piles, and the cards it saw go to other players after a call. Every other
card is *unlocated*; unlocated cards are assumed spread at random over the other
players' unknown hand slots.

* Per rank (13): how many copies are unlocated, so could be in another hand.
* The chance the last claim is true: the claimant holding at least that many copies
  among its unknown cards, after the copies other players already claimed this round.
* Claim inflation: copies claimed by others this round beyond those that can exist.
* Exposure (4): if the player claims 1–4 cards of the round's rank, the chance the
  next player holds enough copies to know for certain it is a lie.
* 1 scalar: unlocated cards in total.
"""
from __future__ import annotations

import numpy as np

from card_games_rl.core.cards import rank_of
from card_games_rl.core.counting import p_at_least

from .rules import rank_index

COUNTING_VERSION = 1
SIZE = 13 + 1 + 1 + 4 + 1


def features(game, player: int) -> np.ndarray:
    located = [0] * 13
    mine = list(game.hands[player]) + list(game.contrib[player]) + list(game.own_discarded[player])
    known = {q: game.known[player][q] for q in range(4) if q != player}
    for card in mine + [c for cards in known.values() for c in cards]:
        located[rank_index(rank_of(card))] += 1
    unlocated = [max(0, 4 - n) for n in located]
    pool = sum(unlocated)

    def slots(q: int) -> int:
        return max(0, len(game.hands[q]) - len(known[q]))

    rank = game.round_rank
    p_true, inflation, exposure = 0.0, 0.5, [0.0] * 4
    if rank is not None:
        r = rank_index(rank)
        claim = game.last_claim
        if claim and claim[0] != player:
            claimant, cards = claim
            earlier = sum(len(c) for q, c in game.claims[:-1] if q not in (player, claimant))
            copies = max(0, unlocated[r] - earlier)
            p_true = p_at_least(pool, copies, slots(claimant) + len(cards), len(cards))
        others_claimed = game.claimed_by_others(player)
        inflation = (np.clip((others_claimed - unlocated[r]) / 4, -1.0, 1.0) + 1.0) / 2
        nxt = (player + 1) % 4
        while not game.hands[nxt] and nxt != player:
            nxt = (nxt + 1) % 4
        if nxt != player:
            exposure = [p_at_least(pool, unlocated[r], slots(nxt), 5 - n) for n in range(1, 5)]

    return np.asarray([u / 4 for u in unlocated] + [p_true, inflation] + exposure + [pool / 52],
                      dtype=np.float32)

