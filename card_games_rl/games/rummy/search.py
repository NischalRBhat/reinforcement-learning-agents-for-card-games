"""Look-ahead search agent for Rummy (docs/GAMES.md §4).

The greedy System bot minimises deadwood *now*. This agent looks one draw ahead
and at the opponent:

* **Outs:** each candidate discard is scored by the expected deadwood after the
  next stock draw, averaged over every card the player has not seen, with a large
  bonus for draws that go out. Among discards with equal deadwood, it keeps the
  cards most likely to complete melds.
* **Danger:** a discard that would fit the cards the opponent is known to hold
  (taken from the discard pile) gets a penalty, so it doesn't feed their melds.
* **Draw:** take the discard only if the resulting hand's value beats the expected
  value of drawing from the stock.

It is exact and deterministic given what the player knows (no sampling noise),
uses only information the player has, and needs no training. Monte Carlo rollouts
with greedy continuations were tried first and did not beat the greedy bot: the
outcome noise swamped the small differences between candidate moves.
"""
from __future__ import annotations

from card_games_rl.core import Game, Policy
from card_games_rl.core.cards import rank_of, suit_of

from .melds import low_rank, solver_for
from .rules import DISCARD, DRAW_DISCARD, DRAW_STOCK, RummyGame

GOING_OUT_BONUS = 60.0   # deadwood-equivalent value of going out on the next draw
DANGER_PENALTY = 3.0     # per way a discard fits the opponent's known cards (tuned: 0, 3, 6, 12, 25)


class LookaheadPolicy(Policy):
    name = "lookahead"

    def __init__(self, seed: int | None = None, max_discards: int = 4):
        self.max_discards = max_discards

    # ----- evaluation ------------------------------------------------------
    @staticmethod
    def unseen(game: RummyGame, player: int) -> list[int]:
        seen = set(game.hands[player]) | set(game.discard) | set(game.known[player])
        return [c for c in range(52) if c not in seen]

    @staticmethod
    def hand_value(hand: list[int], unseen: list[int]) -> float:
        """Expected deadwood (lower is better) of a 10-card hand after one more
        stock draw and the best discard; going out counts as a large gain."""
        if not unseen:
            return float(solver_for(hand).solve())
        total = 0.0
        for card in unseen:
            after, _ = solver_for(hand + [card]).best_discard()
            total += -GOING_OUT_BONUS if after == 0 else after
        return min(total / len(unseen), float(solver_for(hand).solve()))

    @staticmethod
    def danger(card: int, opponent_cards: set[int]) -> int:
        """Ways ``card`` fits the opponent's known cards: same rank (sets) or the
        same suit within two ranks (runs)."""
        hits = 0
        for known in opponent_cards:
            if rank_of(known) == rank_of(card):
                hits += 1
            elif suit_of(known) == suit_of(card) and abs(low_rank(known) - low_rank(card)) <= 2:
                hits += 1
        return hits

    def discard_scores(self, game: RummyGame, player: int) -> dict[int, float]:
        hand = game.hands[player]
        legal = game.legal_actions()
        solver = solver_for(hand)
        full = (1 << len(hand)) - 1
        # Shortlist by immediate deadwood, then score the shortlist exactly.
        shortlist = sorted((solver.solve(full & ~(1 << i)), card) for i, card in enumerate(solver.hand)
                           if DISCARD + card in legal)
        cutoff = shortlist[min(self.max_discards, len(shortlist)) - 1][0]
        unseen = self.unseen(game, player)
        opponent_cards = {c for c in game.known[player] if c in game.hands[1 - player]}
        scores = {}
        for _, card in (s for s in shortlist if s[0] <= cutoff):
            rest = [c for c in hand if c != card]
            if solver_for(rest).solve() == 0:
                scores[DISCARD + card] = -1000.0  # goes out right now
                continue
            scores[DISCARD + card] = self.hand_value(rest, unseen) + DANGER_PENALTY * self.danger(card, opponent_cards)
        return scores

    # ----- policy ----------------------------------------------------------
    def act(self, game: Game, player: int) -> int:
        legal = game.legal_actions()
        if game.phase == "draw":
            if DRAW_DISCARD not in legal:
                return legal[0]
            if DRAW_STOCK not in legal:
                return DRAW_DISCARD
            hand, top = game.hands[player], game.discard[-1]
            unseen = self.unseen(game, player)
            after_take, _ = solver_for(hand + [top]).best_discard(forbidden=top)
            if after_take == 0:
                return DRAW_DISCARD
            # Value of taking: keep the top card, drop the best card to drop (shortlisted
            # by immediate deadwood), then one more draw.
            taken = hand + [top]
            solver = solver_for(taken)
            full = (1 << len(taken)) - 1
            drops = sorted((solver.solve(full & ~(1 << i)), card) for i, card in enumerate(solver.hand) if card != top)
            take = min(self.hand_value([c for c in taken if c != card], unseen) for _, card in drops[:3])
            return DRAW_DISCARD if take < self.hand_value(hand, unseen) else DRAW_STOCK
        scores = self.discard_scores(game, player)
        return min(scores, key=scores.get)

    def action_probabilities(self, game: Game, player: int) -> dict[int, float]:
        """A deterministic search, not a probability model: the chosen move gets 1."""
        return {self.act(game, player): 1.0}
