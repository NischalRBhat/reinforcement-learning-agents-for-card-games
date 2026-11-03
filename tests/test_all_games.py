"""Invariants every registered game must satisfy."""
import json
import random

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from card_games_rl.core import IllegalAction
from card_games_rl.envs import SingleAgentEnv, random_opponents
from card_games_rl.games import GAME_SPECS, heuristic_policy, make_game

from .fuzz import random_playout

GAMES = sorted(GAME_SPECS)


@pytest.mark.parametrize("game_id", GAMES)
def test_random_playouts_keep_invariants(game_id):
    rng = random.Random(1)
    game = make_game(game_id)
    for episode in range(60):
        game.reset(episode)
        random_playout(game, rng, check_every=3)


@pytest.mark.parametrize("game_id", GAMES)
def test_same_seed_same_trajectory(game_id):
    def run(seed):
        game, rng = make_game(game_id, seed=seed), random.Random(99)
        actions = random_playout(game, rng)
        return actions, game.returns()

    assert run(5) == run(5)


@pytest.mark.parametrize("game_id", GAMES)
def test_illegal_action_rejected(game_id):
    game = make_game(game_id, seed=3)
    illegal = [a for a in range(game.num_actions) if a not in game.legal_actions()]
    if illegal:
        with pytest.raises(IllegalAction):
            game.step(illegal[0])


@pytest.mark.parametrize("game_id", GAMES)
def test_heuristic_policy_only_plays_legal_moves(game_id):
    game = make_game(game_id)
    for seed in range(15):
        game.reset(seed)
        bot = heuristic_policy(game_id, seed)
        for _ in range(5000):
            if game.done:
                break
            game.step(bot.act(game, game.current_player))
        assert game.done


@pytest.mark.parametrize("game_id", GAMES)
def test_views_are_json_serialisable(game_id):
    game = make_game(game_id, seed=11)
    rng = random.Random(0)
    for _ in range(30):
        for viewer in [None, *range(game.num_players)]:
            json.dumps(game.view(viewer))
        if game.done:
            break
        game.step(rng.choice(game.legal_actions()))


@pytest.mark.parametrize("game_id", GAMES)
def test_gym_env_passes_check_env(game_id):
    env = SingleAgentEnv(game_id, seed=0)
    env.illegal_action_mode = "random"  # check_env samples unmasked actions
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("game_id", GAMES)
def test_rewards_sum_to_final_score_without_shaping(game_id):
    """ΔS rewards telescope to the final score (shaping/step cost disabled)."""
    env = SingleAgentEnv(game_id, opponents=random_opponents(0), seed=0, shaping_weight=0.0)
    env.game.step_cost = 0.0
    rng = np.random.default_rng(0)
    for episode in range(20):
        obs, info = env.reset(seed=episode)
        start = env.game.score(env.seat)
        total, done = 0.0, False
        while not done:
            legal = np.flatnonzero(env.action_masks())
            obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
            total += reward
            done = terminated or truncated
        assert abs(total - (info["final_score"] - start)) < 1e-6


@pytest.mark.parametrize("game_id", GAMES)
def test_shaping_telescopes(game_id):
    """With Φ(terminal)=0 the shaping term sums to −Φ(s0): it cannot be farmed."""
    env = SingleAgentEnv(game_id, opponents=random_opponents(0), seed=0)
    env.game.step_cost = 0.0
    rng = np.random.default_rng(1)
    for episode in range(10):
        env.reset(seed=episode)
        start_score, start_phi = env.game.score(env.seat), env.game.potential(env.seat)
        total, done = 0.0, False
        while not done:
            legal = np.flatnonzero(env.action_masks())
            _, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
            total += reward
            done = terminated or truncated
        expected = (info["final_score"] - start_score) - env.shaping_weight * start_phi
        assert abs(total - expected) < 1e-6
