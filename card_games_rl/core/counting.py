"""Probability helpers for card counting (docs/ENV_DESIGN.md §9).

The counting features reason about *unseen* cards: cards the player has not seen and
cannot locate. Each opponent holds some of them in its unknown hand slots. Treating
the unseen cards as dealt uniformly at random into those slots (consistent with what
is publicly known, such as voids) gives simple closed forms.
"""
from __future__ import annotations


def p_none(pool: int, special: int, draws: int) -> float:
    """Chance that ``draws`` cards taken from ``pool`` contain none of ``special`` cards."""
    if special <= 0 or draws <= 0:
        return 1.0
    if special > pool - draws:
        return 0.0
    p = 1.0
    for i in range(draws):
        p *= (pool - special - i) / (pool - i)
    return p


def p_at_least(pool: int, special: int, draws: int, k: int) -> float:
    """Chance that ``draws`` cards from ``pool`` contain at least ``k`` of ``special`` cards."""
    if k <= 0:
        return 1.0
    if special < k or draws < k or pool <= 0:
        return 0.0
    from math import comb

    total = comb(pool, draws)
    if total == 0:
        return 0.0
    hit = sum(comb(special, j) * comb(pool - special, draws - j) for j in range(k, min(special, draws) + 1))
    return hit / total
