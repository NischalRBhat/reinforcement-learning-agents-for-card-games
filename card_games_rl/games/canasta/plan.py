"""Planner actions for the Canasta agent (docs/ENV_DESIGN.md §10).

The engine splits a turn into many small actions: draw, every meld and every wild card
added, then the discard. A player makes about 150 of them per hand, which made the
first agent hard to train. The agent instead makes the strategic choices and the
System bot's meld rules carry out the details:

| Plan action | Meaning |
|---|---|
| 0–3 | draw from the stock / take the pile (natural pair, natural + wild, add to a meld): as in the engine |
| 4 ``MELD`` | the meld the System bot would make now (naturals, and wild cards only to open, complete a canasta or go out) |
| 5 ``MELD_WILD`` | the meld that uses wild cards to grow a meld furthest towards a canasta |
| 6–20 | discard one card of a kind: as in the engine |

Choosing a discard while melds are still possible is how the agent holds cards back.
Every legal engine move is reachable, so nothing is lost: any natural meld is a positive
System meld and any meld with wild cards is a ``MELD_WILD`` candidate.

The agent also sees *planner features*: what the System bot would do now, what the two
meld options would lay down, and the bot's discard choice. They come from public
information and the agent's own hand only.
"""
from __future__ import annotations

import numpy as np

from .bots import HeuristicPolicy
from .rules import (CANASTA_SIZE, DISCARD_BASE, DRAW, JOKER, KINDS, MELD_BASE, TAKE_ADD, TWO, VALUES,
                    CanastaGame, decode_meld)

PLAN_VERSION = 1
MELD, MELD_WILD, DISCARD = 4, 5, 6
NUM_PLAN_ACTIONS = DISCARD + KINDS
MELD_FEATURES = 6
SIZE = NUM_PLAN_ACTIONS + 2 * MELD_FEATURES + KINDS

_SYSTEM = HeuristicPolicy()


def _melds(game: CanastaGame) -> list[int]:
    return [a for a in game.legal_actions() if MELD_BASE <= a < DISCARD_BASE]


def system_meld(game: CanastaGame) -> int | None:
    """The System bot's meld now, or None when it would not meld."""
    player = game.current_player
    best, best_score = None, 0.0
    for action in _melds(game):
        score = _SYSTEM._meld_score(game, player, action)
        if score > best_score:
            best, best_score = action, score
    return best


def wild_meld(game: CanastaGame) -> int | None:
    """The meld with wild cards that grows a meld furthest: completing a canasta first,
    then the biggest meld, then the cheapest wild cards (2s before jokers)."""
    team = game.team(game.current_player)

    def key(action: int):
        kind, n, j, d = decode_meld(action)
        existing = game.melds[team].get(kind)
        size = (game.meld_size(existing) if existing else 0) + n + j + d
        return (size >= CANASTA_SIZE, size, -(j * VALUES[JOKER] + d * VALUES[TWO]), n)

    options = [a for a in _melds(game) if sum(decode_meld(a)[2:]) > 0]
    return max(options, key=key) if options else None


def system_discard(game: CanastaGame) -> int | None:
    """The kind the System bot would discard if it stopped melding now."""
    player = game.current_player
    kinds = [a - DISCARD_BASE for a in game.legal_actions() if a >= DISCARD_BASE]
    return max(kinds, key=lambda k: _SYSTEM._discard_score(game, player, k)) if kinds else None


def legal(game: CanastaGame) -> list[int]:
    engine = game.legal_actions()
    if game.phase == "draw":
        return [a for a in engine if a <= TAKE_ADD]
    actions = []
    if system_meld(game) is not None:
        actions.append(MELD)
    if wild_meld(game) is not None:
        actions.append(MELD_WILD)
    return actions + [DISCARD + a - DISCARD_BASE for a in engine if a >= DISCARD_BASE]


def mask(game: CanastaGame) -> np.ndarray:
    out = np.zeros(NUM_PLAN_ACTIONS, dtype=bool)
    out[legal(game)] = True
    return out


def to_game(game: CanastaGame, action: int) -> int:
    """Plan action -> engine action."""
    if action <= TAKE_ADD:
        return action
    if action == MELD:
        return system_meld(game)
    if action == MELD_WILD:
        return wild_meld(game)
    return DISCARD_BASE + action - DISCARD


def from_game(game: CanastaGame, action: int) -> int:
    """Engine action -> the plan action that covers it (for imitating the System bot)."""
    if action <= TAKE_ADD:
        return action
    if action >= DISCARD_BASE:
        return DISCARD + action - DISCARD_BASE
    if action == system_meld(game) or not sum(decode_meld(action)[2:]):
        return MELD
    return MELD_WILD


def _meld_features(game: CanastaGame, action: int | None) -> list[float]:
    if action is None:
        return [0.0] * MELD_FEATURES
    player = game.current_player
    kind, n, j, d = decode_meld(action)
    existing = game.melds[game.team(player)].get(kind)
    size = (game.meld_size(existing) if existing else 0) + n + j + d
    added = n + j + d
    return [1.0, added / 7, (j + d) / 3, float(existing is None), float(size >= CANASTA_SIZE),
            float(added == len(game.hands[player]))]


def features(game: CanastaGame, player: int) -> np.ndarray:
    """Planner features for the player to move (zeros when it is someone else's turn)."""
    out = np.zeros(SIZE, dtype=np.float32)
    if game.done or game.current_player != player:
        return out
    out[from_game(game, _SYSTEM.act(game, player))] = 1.0
    offset = NUM_PLAN_ACTIONS
    if game.phase == "play":
        out[offset:offset + MELD_FEATURES] = _meld_features(game, system_meld(game))
        out[offset + MELD_FEATURES:offset + 2 * MELD_FEATURES] = _meld_features(game, wild_meld(game))
        discard = system_discard(game)
        if discard is not None:
            out[offset + 2 * MELD_FEATURES + discard] = 1.0
    return out


__all__ = ["DRAW", "MELD", "MELD_WILD", "DISCARD", "NUM_PLAN_ACTIONS", "PLAN_VERSION", "SIZE", "features",
           "from_game", "legal", "mask", "to_game"]
