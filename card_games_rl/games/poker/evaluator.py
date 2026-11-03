"""Hand evaluation and equity for hold'em.

``evaluate`` returns a tuple that compares correctly between hands:
``(category, tiebreakers...)`` with categories 0 = high card … 8 = straight flush.
"""
from __future__ import annotations

import random
from collections import Counter
from itertools import combinations

from card_games_rl.core.cards import rank_of, suit_of

CATEGORY_NAMES = ["High card", "Pair", "Two pair", "Three of a kind", "Straight", "Flush",
                  "Full house", "Four of a kind", "Straight flush"]


def _straight_high(ranks: set[int]) -> int:
    if 14 in ranks:
        ranks = ranks | {1}
    for high in range(14, 4, -1):
        if all(high - i in ranks for i in range(5)):
            return high
    return 0


def evaluate(cards: list[int]) -> tuple:
    ranks = [rank_of(c) for c in cards]
    suits = Counter(suit_of(c) for c in cards)
    flush_suit = next((s for s, n in suits.items() if n >= 5), None)
    if flush_suit is not None:
        flush_ranks = sorted((rank_of(c) for c in cards if suit_of(c) == flush_suit), reverse=True)
        straight_flush = _straight_high(set(flush_ranks))
        if straight_flush:
            return (8, straight_flush)
    counts = Counter(ranks)
    groups = sorted(counts.items(), key=lambda rc: (rc[1], rc[0]), reverse=True)
    if groups[0][1] == 4:
        quad = groups[0][0]
        return (7, quad, max(r for r in ranks if r != quad))
    trips = sorted((r for r, n in counts.items() if n >= 3), reverse=True)
    pairs = sorted((r for r, n in counts.items() if n >= 2), reverse=True)
    if trips and len(pairs) >= 2:
        return (6, trips[0], max(r for r in pairs if r != trips[0]))
    if flush_suit is not None:
        return (5, *flush_ranks[:5])
    straight = _straight_high(set(ranks))
    if straight:
        return (4, straight)
    desc = sorted(ranks, reverse=True)
    if trips:
        return (3, trips[0], *[r for r in desc if r != trips[0]][:2])
    if len(pairs) >= 2:
        high, low = pairs[0], pairs[1]
        return (2, high, low, max(r for r in desc if r not in (high, low)))
    if pairs:
        return (1, pairs[0], *[r for r in desc if r != pairs[0]][:3])
    return (0, *desc[:5])


def equity(hole: list[int], opponent: list[int], board: list[int], samples: int = 400,
           rng: random.Random | None = None) -> float:
    """Probability (ties count half) that ``hole`` beats ``opponent`` once the board completes.

    Exact enumeration when 2 or fewer cards are missing, Monte Carlo otherwise.
    """
    used = set(hole) | set(opponent) | set(board)
    remaining = [c for c in range(52) if c not in used]
    missing = 5 - len(board)
    if missing == 0:
        boards = [()]
    elif missing <= 2:
        boards = combinations(remaining, missing)
    else:
        rng = rng or random.Random(0)
        boards = (tuple(rng.sample(remaining, missing)) for _ in range(samples))
    wins = total = 0.0
    for extra in boards:
        full = board + list(extra)
        mine, theirs = evaluate(hole + full), evaluate(opponent + full)
        wins += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        total += 1
    return wins / total
