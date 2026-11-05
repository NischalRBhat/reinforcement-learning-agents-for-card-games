"""Card-counting agents: the base observations stay frozen, counting features use only
public information, and variant agents train, load and play without touching the base."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from card_games_rl.agents.registry import best_model_path, load_agent, model_dir, read_meta, write_meta
from card_games_rl.core.counting import p_at_least, p_none
from card_games_rl.encoders import COUNTING, COUNTING_ENCODERS, agent_key, get_encoder, variants_for
from card_games_rl.envs import SingleAgentEnv
from card_games_rl.games import GAME_SPECS, heuristic_policy, make_game

# Fingerprints of the engines' own observations on fixed deals. They must never change
# without an OBS_VERSION bump: every trained base agent depends on them.
BASE_OBSERVATIONS = {
    "blackjack": "370895f7f77382f6",
    "bluff": "47328bb91892e2f8",
    "canasta": "39e6dbad3006cdc9",
    "donkey": "656f3acb49ad6936",
    "hearts": "ca4ab6b8422e34af",
    "klondike": "05752cf32e37a0e3",
    "poker": "f07a3cfe4706ff35",
    "rummy": "56959c6fc1fd4767",
}


@pytest.mark.parametrize("game_id", sorted(GAME_SPECS))
def test_base_observations_are_unchanged(game_id):
    digest = hashlib.sha256()
    bot = heuristic_policy(game_id, 0)
    for seed in range(3):
        game = make_game(game_id, seed=seed)
        for _ in range(60):
            if game.done:
                break
            player = game.current_player
            digest.update(np.round(game.observation(player), 5).astype(np.float32).tobytes())
            game.step(bot.act(game, player))
    assert digest.hexdigest()[:16] == BASE_OBSERVATIONS[game_id], "base observation changed: bump OBS_VERSION"


def test_probability_helpers():
    assert p_none(10, 0, 3) == 1.0 and p_none(10, 8, 3) == 0.0
    assert abs(p_none(4, 1, 2) - 0.5) < 1e-12              # one special card among 4, draw 2
    assert abs(p_at_least(4, 1, 2, 1) - 0.5) < 1e-12
    assert p_at_least(52, 4, 13, 5) == 0.0


def _play(game_id, seed, steps):
    game = make_game(game_id, seed=seed)
    bot = heuristic_policy(game_id, seed)
    for _ in range(steps):
        if game.done:
            break
        game.step(bot.act(game, game.current_player))
    return game


@pytest.mark.parametrize("game_id", COUNTING_ENCODERS)
def test_counting_features_are_bounded_and_extend_the_base(game_id):
    encoder, base = get_encoder(game_id, COUNTING), get_encoder(game_id)
    for seed in range(8):
        game = make_game(game_id, seed=seed)
        bot = heuristic_policy(game_id, seed)
        while not game.done:
            player = game.current_player
            x = encoder(game, player)
            raw = np.concatenate([game.observation(player), __import__(
                f"card_games_rl.games.{game_id}.counting", fromlist=["features"]).features(game, player)])
            assert x.shape == (encoder.size,) and encoder.size > base.size
            assert np.isfinite(raw).all() and raw.min() >= 0.0 and raw.max() <= 1.0 + 1e-6
            assert np.array_equal(x[:base.size], base(game, player))
            game.step(bot.act(game, player))


def _shuffle_hidden(game_id, game, player, rng):
    """Re-deal the cards ``player`` cannot locate among the other hands (same sizes)."""
    others = [q for q in range(4) if q != player]
    if game_id == "donkey":
        fixed = {q: game.known[q] for q in others}
    elif game_id == "hearts":
        from card_games_rl.games.hearts.counting import _Belief
        fixed = _Belief(game, player).known
    else:  # bluff
        fixed = {q: game.known[player][q] & set(game.hands[q]) for q in others}
    hidden = [c for q in others for c in game.hands[q] if c not in fixed[q]]
    rng.shuffle(hidden)
    for q in others:
        keep = [c for c in game.hands[q] if c in fixed[q]]
        take = len(game.hands[q]) - len(keep)
        game.hands[q] = sorted(keep + hidden[:take])
        hidden = hidden[take:]
    game.invalidate()


@pytest.mark.parametrize("game_id", COUNTING_ENCODERS)
def test_counting_features_never_peek_at_hidden_cards(game_id):
    module = __import__(f"card_games_rl.games.{game_id}.counting", fromlist=["features"])
    rng = random.Random(0)
    checked = 0
    for seed in range(12):
        game = _play(game_id, seed, steps=rng.randrange(5, 40))
        if game.done or getattr(game, "passing", False):
            continue
        player = game.current_player
        before = module.features(game, player)
        _shuffle_hidden(game_id, game, player, rng)
        assert np.allclose(module.features(game, player), before), f"{game_id} seed {seed}"
        checked += 1
    assert checked >= 5


@pytest.mark.parametrize("game_id", COUNTING_ENCODERS)
def test_counting_env_passes_gymnasium_checks(game_id):
    env = SingleAgentEnv(game_id, seed=0, variant=COUNTING)
    assert env.observation_space.shape == (get_encoder(game_id, COUNTING).size,)
    env.illegal_action_mode = "random"
    check_env(env, skip_render_check=True)


def test_variants_and_folders():
    assert variants_for("donkey") == ["base", "counting"] and variants_for("poker") == ["base"]
    assert agent_key("donkey", COUNTING) == "donkey-counting" and agent_key("donkey") == "donkey"
    with pytest.raises(ValueError):
        get_encoder("poker", COUNTING)


def test_counting_meta_is_versioned_and_kept_apart(tmp_path):
    write_meta("hearts", {"algorithm": "MaskablePPO"}, root=tmp_path, variant=COUNTING)
    (model_dir("hearts", tmp_path, COUNTING) / "best.zip").write_bytes(b"x")
    meta = read_meta("hearts", tmp_path, COUNTING)
    assert meta["variant"] == COUNTING and meta["counting_version"] == get_encoder("hearts", COUNTING).version[1]
    assert read_meta("hearts", tmp_path) is None                   # the base agent's folder is untouched
    assert best_model_path("hearts", tmp_path, COUNTING) is not None
    meta["counting_version"] = 999                                  # features changed since training
    (model_dir("hearts", tmp_path, COUNTING) / "meta.json").write_text(__import__("json").dumps(meta))
    with pytest.warns(UserWarning):
        assert best_model_path("hearts", tmp_path, COUNTING) is None


def test_counting_agent_trains_loads_and_plays_without_touching_the_base(tmp_path):
    from card_games_rl.agents.model_policy import ModelPolicy
    from card_games_rl.training.train import train

    train("donkey", "smoke", model_root=tmp_path, run_root=tmp_path / "runs", n_envs=1, variant=COUNTING,
          overrides={"timesteps": 256, "pretrain_episodes": 2, "eval_every": 256, "eval_episodes": 4,
                     "snapshot_every": 128})
    assert (tmp_path / "donkey-counting" / "best.zip").exists() and not (tmp_path / "donkey").exists()
    agent = load_agent("donkey", root=tmp_path, variant=COUNTING)
    assert isinstance(agent, ModelPolicy) and agent.encoder.size == get_encoder("donkey", COUNTING).size
    game = make_game("donkey", seed=1)
    assert agent.act(game, game.current_player) in game.legal_actions()
    assert load_agent("donkey", root=tmp_path) is None


def test_rummy_counting_agent_is_a_registered_search(tmp_path):
    from card_games_rl.games.rummy.search import CountingLookaheadPolicy
    from card_games_rl.training.train import train

    train("rummy", "smoke", model_root=tmp_path, run_root=tmp_path / "runs", variant=COUNTING,
          overrides={"eval_episodes": 4})
    assert isinstance(load_agent("rummy", root=tmp_path, variant=COUNTING), CountingLookaheadPolicy)


def test_head_to_head_swaps_seats_on_duplicate_deals():
    from card_games_rl.training.evaluate import head_to_head

    bot = heuristic_policy("donkey", 0)
    result = head_to_head("donkey", bot, bot, deals=6)
    assert result["deals"] == 6 and abs(result["difference"]) < 1e-9   # identical players: no edge
    assert {"a", "b", "ci95", "a_better", "metric"} <= set(result)
