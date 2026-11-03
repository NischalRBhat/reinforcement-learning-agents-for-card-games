"""Exact expectimax strategy for the Blackjack variant (infinite-deck model).

Used as the oracle baseline: the RL agent is judged by how close its EV gets
to this policy's EV. Dealer outcomes are conditioned on the dealer not having
blackjack, because decisions only happen after the peek.
"""
from __future__ import annotations

from functools import lru_cache

from card_games_rl.core import Game, Policy

from .rules import DOUBLE, HIT, NO_INSURE, SPLIT, STAND, SURRENDER, card_points, hand_value

# Draw probabilities by hard value 1 (Ace) .. 10.
PROB = {v: 1 / 13 for v in range(1, 10)} | {10: 4 / 13}


def _add(total: int, soft: bool, value: int) -> tuple[int, bool]:
    """Add a card (hard value) to a (best total, soft) hand.

    A hard hand that contains an Ace already has hard total >= 12, so its Ace
    can never count as 11 again; only soft hands or a new Ace can be soft.
    """
    hard = (total - 10 if soft else total) + value
    if (soft or value == 1) and hard + 10 <= 21:
        return hard + 10, True
    return hard, False


@lru_cache(maxsize=None)
def dealer_distribution(up: int) -> dict[int, float]:
    """Final dealer total (17..21, 22 = bust) given the up card, no dealer blackjack."""

    @lru_cache(maxsize=None)
    def play(total: int, soft: bool) -> tuple[tuple[int, float], ...]:
        if total > 21:
            return ((22, 1.0),)
        if total >= 17:
            return ((total, 1.0),)
        result: dict[int, float] = {}
        for value, p in PROB.items():
            for final, q in play(*_add(total, soft, value)):
                result[final] = result.get(final, 0.0) + p * q
        return tuple(result.items())

    start = (11, True) if up == 1 else (up, False)
    result: dict[int, float] = {}
    norm = 0.0
    for value, p in PROB.items():
        if (up == 1 and value == 10) or (up == 10 and value == 1):
            continue  # would be blackjack: excluded by the peek
        norm += p
        for final, q in play(*_add(*start, value)):
            result[final] = result.get(final, 0.0) + p * q
    return {k: v / norm for k, v in result.items()}


@lru_cache(maxsize=None)
def ev_stand(total: int, up: int) -> float:
    if total > 21:
        return -1.0
    ev = 0.0
    for final, p in dealer_distribution(up).items():
        ev += p * (1.0 if final == 22 or total > final else 0.0 if total == final else -1.0)
    return ev


@lru_cache(maxsize=None)
def ev_hit(total: int, soft: bool, up: int) -> float:
    """EV of hitting now and then playing hit/stand optimally."""
    ev = 0.0
    for value, p in PROB.items():
        new_total, new_soft = _add(total, soft, value)
        ev += p * (-1.0 if new_total > 21 else ev_hit_stand(new_total, new_soft, up))
    return ev


@lru_cache(maxsize=None)
def ev_hit_stand(total: int, soft: bool, up: int) -> float:
    if total >= 21:
        return ev_stand(total, up)
    return max(ev_stand(total, up), ev_hit(total, soft, up))


@lru_cache(maxsize=None)
def ev_double(total: int, soft: bool, up: int) -> float:
    ev = 0.0
    for value, p in PROB.items():
        ev += p * ev_stand(_add(total, soft, value)[0], up)
    return 2 * ev


@lru_cache(maxsize=None)
def ev_split(pair_value: int, up: int) -> float:
    """Two hands, no resplit, DAS allowed; split aces get one card and stand."""
    start = (11, True) if pair_value == 1 else (pair_value, False)
    ev = 0.0
    for value, p in PROB.items():
        total, soft = _add(*start, value)
        if pair_value == 1:
            ev += p * ev_stand(total, up)
        else:
            ev += p * max(ev_hit_stand(total, soft, up), ev_double(total, soft, up))
    return 2 * ev


def action_values(game: Game) -> dict[int, float]:
    """EV of each legal action in the current blackjack state."""
    legal = game.legal_actions()
    if game.insurance_phase:
        return {NO_INSURE: 0.0}  # insurance is -EV without card counting
    hand = game.hands[game.active]
    total, soft = hand_value(hand)
    up = card_points(game.up_card)
    values = {HIT: ev_hit(total, soft, up) if total < 21 else -1.0, STAND: ev_stand(total, up)}
    if DOUBLE in legal:
        values[DOUBLE] = ev_double(total, soft, up)
    if SPLIT in legal:
        values[SPLIT] = ev_split(card_points(hand[0]), up)
    if SURRENDER in legal:
        values[SURRENDER] = -0.5
    return values


class OptimalPolicy(Policy):
    name = "optimal"

    def __init__(self, seed: int | None = None):
        pass

    def act(self, game: Game, player: int) -> int:
        values = action_values(game)
        return max(values, key=values.get)
