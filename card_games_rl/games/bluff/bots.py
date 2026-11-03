"""Rule-based Bluff player (the "System" opponent), Indian rules."""
from __future__ import annotations

import random
from collections import Counter

from card_games_rl.core import Game, Policy
from card_games_rl.core.cards import rank_of

from .rules import CALL, DECLARE, PASS, PLAY_COMBOS, rank_index


class HeuristicPolicy(Policy):
    """Honest by default, bluffs sometimes, and calls with a risk-aware probability.

    * Start a round with the rank it holds most of, and play all of those cards.
    * Respond with every true card it has; without one, bluff a single card now
      and then (less often as its hand shrinks, because a caught bluff costs the pile).
    * Call for sure when the claim is impossible or the claimant is on their last card;
      otherwise call more when the round's claims look inflated, less when the pile is big.
    """

    name = "heuristic"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    @staticmethod
    def lie_probability(game: Game, player: int) -> float:
        """Calibrated chance that the last claim is false.

        Starts from the context: a round's first claim is rarely a lie (starters pick
        ranks they hold), added claims often are (measured ~9% vs ~39% among bots).
        That prior is updated with the claimant's record of being caught, and set
        high when the round's claims exceed the cards that could exist, assuming
        the earlier claims were honest.
        """
        claimant, played = game.last_claim
        first = len(game.claims) == 1
        prior = 0.1 if first else 0.35
        stats = game.stats[claimant]
        weight = 4.0
        p = (stats["caught"] + prior * weight) / (stats["called"] + weight)
        earlier = game.claimed_by_others(player) - len(played)
        if len(played) > game.max_true(player) - max(0, earlier):
            p = max(p, 0.85)
        return p

    @classmethod
    def call_probability(cls, game: Game, player: int) -> float:
        claimant, _ = game.last_claim
        if game.certain_lie(player) or not game.hands[claimant]:
            return 1.0
        # Calling wrongly costs the whole pile, so a bigger pile needs more certainty.
        threshold = min(0.9, 0.6 + 0.015 * len(game.pile))
        if len(game.hands[claimant]) <= 2:
            threshold -= 0.15
        return 0.9 if cls.lie_probability(game, player) > threshold else 0.03

    def act(self, game: Game, player: int) -> int:
        legal = game.legal_actions()
        if game.phase == "declare":
            counts = Counter(rank_of(c) for c in game.hands[player])
            best = max(counts.values())
            rank = self.rng.choice(sorted(r for r, n in counts.items() if n == best))
            return DECLARE + rank_index(rank)
        if CALL in legal and self.rng.random() < self.call_probability(game, player):
            return CALL
        hand = len(game.hands[player])
        true_count = min(4, game.count_rank(player, game.round_rank))
        if true_count:
            t, b = true_count, 0
            if t < 4 and hand > 6 and self.rng.random() < 0.2:
                b = 1  # pad an honest play
        elif game.phase == "lead" or self.rng.random() < (0.35 if hand > 3 else 0.15):
            t, b = 0, 1
        else:
            return PASS if PASS in legal else legal[0]
        combo = PLAY_COMBOS.index((t, b))
        return combo if combo in legal else (PASS if PASS in legal else legal[0])
