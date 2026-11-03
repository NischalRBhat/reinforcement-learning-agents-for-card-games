"""Generic invariant checks for any ``Game`` (docs/ENV_DESIGN.md §8)."""
from __future__ import annotations

import random
from collections import Counter

import numpy as np

from card_games_rl.core import Game


def check_state(game: Game) -> None:
    legal = game.legal_actions()
    mask = game.action_mask()
    assert sorted(np.flatnonzero(mask).tolist()) == sorted(legal), "mask != legal_actions"
    assert len(set(legal)) == len(legal), "duplicate legal actions"
    if game.done:
        assert legal == [] and game.current_player == -1
    else:
        assert legal, "non-terminal state without legal actions"
        assert 0 <= game.current_player < game.num_players
    for player in range(game.num_players):
        obs = game.observation(player)
        assert obs.shape == (game.obs_size,) and obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        assert obs.min() >= -1e-6 and obs.max() <= 1 + 1e-6, f"obs out of [0,1]: {obs.min()} {obs.max()}"
    try:
        deck = game.all_cards()
    except NotImplementedError:
        return
    expected = Counter(range(52)) if len(deck) <= 52 else None
    counts = Counter(deck)
    if expected is not None:
        assert counts == expected, "card conservation violated"
    else:
        assert set(counts.values()) == {len(deck) // 52}, "card conservation violated (multi-deck)"


def random_playout(game: Game, rng: random.Random, max_steps: int = 5000, check_every: int = 1) -> list[int]:
    actions = []
    check_state(game)
    for step in range(max_steps):
        if game.done:
            break
        action = rng.choice(game.legal_actions())
        actions.append(action)
        game.step(action)
        if step % check_every == 0:
            check_state(game)
    assert game.done, f"episode did not finish within {max_steps} steps"
    check_state(game)
    if game.zero_sum:
        assert abs(sum(game.returns())) < 1e-6, f"returns not zero-sum: {game.returns()}"
    return actions
