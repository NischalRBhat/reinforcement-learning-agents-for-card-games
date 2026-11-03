"""Rule-based Donkey player (the "System" opponent)."""
from __future__ import annotations

import random

from card_games_rl.core import Game, Policy
from card_games_rl.core.cards import rank_of, suit_of


class HeuristicPolicy(Policy):
    """Shed cards safely and make others pick up.

    * Lead: the lowest card of a suit that no remaining opponent is known to be void
      in (so the trick is likely discarded), preferring suits it holds most of.
    * Follow: if a later player is known to be void in the led suit (a cut is coming),
      stay under the current highest card; otherwise dump the highest card of the suit.
    * Cut: throw the highest card, the hardest one to get rid of later.
    """

    name = "heuristic"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, game: Game, player: int) -> int:
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]
        others = [p for p in game.active() if p != player]
        if not game.trick:
            hand = game.hands[player]
            safe = [c for c in legal if not any(game.void[p][suit_of(c)] for p in others)]
            pool = safe or legal
            by_suit = {s: sum(suit_of(c) == s for c in hand) for s in range(4)}
            return min(pool, key=lambda c: (-by_suit[suit_of(c)], rank_of(c)))
        led = game.led_suit
        if suit_of(legal[0]) != led:  # cutting
            return max(legal, key=lambda c: (rank_of(c), c))
        remaining = game.trick_players[len(game.trick) + 1:]
        cut_coming = any(game.void[p][led] for p in remaining)
        if cut_coming:
            top = max(rank_of(c) for _, c in game.trick if suit_of(c) == led)
            under = [c for c in legal if rank_of(c) < top]
            return max(under, key=rank_of) if under else min(legal, key=rank_of)
        return max(legal, key=rank_of)
