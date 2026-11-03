"""Meld detection and minimum deadwood for basic rummy (Ace low only)."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations

from card_games_rl.core.cards import rank_of, suit_of


def low_rank(card: int) -> int:
    """Ace = 1 .. King = 13."""
    rank = rank_of(card)
    return 1 if rank == 14 else rank


def points(card: int) -> int:
    return min(low_rank(card), 10)


def is_meld(cards: list[int]) -> bool:
    if len(cards) < 3:
        return False
    ranks = [low_rank(c) for c in cards]
    if len(set(ranks)) == 1:
        return len(cards) <= 4
    if len({suit_of(c) for c in cards}) != 1:
        return False
    ordered = sorted(ranks)
    return all(b == a + 1 for a, b in zip(ordered, ordered[1:]))


def all_melds(hand: list[int]) -> list[int]:
    """Every possible meld in ``hand`` as a bitmask over hand positions."""
    melds = []
    by_rank, by_suit = defaultdict(list), defaultdict(list)
    for i, card in enumerate(hand):
        by_rank[low_rank(card)].append(i)
        by_suit[suit_of(card)].append(i)
    for indices in by_rank.values():
        for size in (3, 4):
            for group in combinations(indices, size):
                melds.append(sum(1 << i for i in group))
    for indices in by_suit.values():
        ordered = sorted(indices, key=lambda i: low_rank(hand[i]))
        for start in range(len(ordered)):
            mask, previous = 1 << ordered[start], low_rank(hand[ordered[start]])
            length = 1
            for i in ordered[start + 1:]:
                if low_rank(hand[i]) != previous + 1:
                    break
                mask |= 1 << i
                previous += 1
                length += 1
                if length >= 3:
                    melds.append(mask)
    return melds


class DeadwoodSolver:
    """Minimum deadwood for any subset of one hand, memoised by bitmask."""

    def __init__(self, hand: list[int]):
        self.hand = list(hand)
        self.melds = all_melds(self.hand)
        self.by_card = [[m for m in self.melds if m >> i & 1] for i in range(len(self.hand))]
        self.memo: dict[int, int] = {0: 0}

    def solve(self, mask: int | None = None) -> int:
        if mask is None:
            mask = (1 << len(self.hand)) - 1
        if mask in self.memo:
            return self.memo[mask]
        first = (mask & -mask).bit_length() - 1
        best = points(self.hand[first]) + self.solve(mask & ~(1 << first))
        for meld in self.by_card[first]:
            if meld & mask == meld:
                best = min(best, self.solve(mask & ~meld))
        self.memo[mask] = best
        return best

    def best_discard(self, forbidden: int | None = None) -> tuple[int, int]:
        """(deadwood, card) for the discard that leaves the lowest deadwood."""
        full = (1 << len(self.hand)) - 1
        options = [(self.solve(full & ~(1 << i)), points(c), c) for i, c in enumerate(self.hand) if c != forbidden]
        deadwood, _, card = min(options, key=lambda o: (o[0], -o[1]))
        return deadwood, card

    def meld_cards(self) -> int:
        """Number of cards that belong to at least one possible meld."""
        return sum(1 for melds in self.by_card if melds)


def deadwood(hand: list[int]) -> int:
    return solver_for(hand).solve() if hand else 0


@lru_cache(maxsize=8192)
def _cached_solver(hand: tuple[int, ...]) -> DeadwoodSolver:
    return DeadwoodSolver(list(hand))


def solver_for(cards) -> DeadwoodSolver:
    """Memoised solver for a hand (order-independent). ``solver.hand`` is sorted,
    so iterate ``solver.hand`` (not the caller's list) when using bit positions."""
    return _cached_solver(tuple(sorted(cards)))
