import numpy as np
import pytest

from card_games_rl.core import RandomPolicy
from card_games_rl.envs import OpponentPool, SingleAgentEnv, heuristic_opponents
from card_games_rl.games import make_game


def test_learner_seat_is_randomised_in_multiplayer_games():
    env = SingleAgentEnv("hearts", seed=0)
    seats = {env.reset(seed=s)[1]["seat"] for s in range(40)}
    assert seats == {0, 1, 2, 3}


def test_fixed_learner_seat():
    env = SingleAgentEnv("poker", seed=0, learner_seat=1)
    for s in range(10):
        env.reset(seed=s)
        assert env.seat == 1 and env.game.current_player == 1


def test_illegal_actions_raise_by_default():
    env = SingleAgentEnv("blackjack", seed=0)
    env.reset(seed=0)
    illegal = int(np.flatnonzero(~env.action_masks())[0])
    with pytest.raises(ValueError):
        env.step(illegal)


def test_blackjack_env_never_starts_on_a_finished_hand():
    env = SingleAgentEnv("blackjack", seed=0)
    for s in range(300):
        env.reset(seed=s)
        assert not env.game.done


def test_empty_pool_falls_back_to_heuristic_and_random(tmp_path):
    pool = OpponentPool("hearts", tmp_path / "pool", seed=0)
    game = make_game("hearts", seed=0)
    names = {p.name for _ in range(50) for p in pool.sample(game, 0).values()}
    assert names <= {"heuristic", "random"} and "heuristic" in names


def test_hearts_reward_on_opponent_completed_trick():
    """Points taken when an opponent closes the trick reach the learner (old bug)."""
    env = SingleAgentEnv("hearts", opponents=heuristic_opponents("hearts", 0), seed=0, learner_seat=0)
    rng = np.random.default_rng(0)
    for episode in range(30):
        env.reset(seed=episode)
        done = False
        while not done:
            before = env.game.points[0]
            _, reward, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
            done = terminated or truncated
            gained = env.game.points[0] - before
            if gained and not done:
                assert reward < 0


def test_training_seeds_never_hit_validation_or_test_range():
    from card_games_rl.core.seeds import RESERVED_END, VALIDATION_SEEDS, is_reserved, training_seed

    rng = np.random.default_rng(0)
    seeds = [training_seed(rng) for _ in range(200_000)]
    assert not any(is_reserved(s) for s in seeds)
    assert min(seeds) < VALIDATION_SEEDS and max(seeds) > RESERVED_END  # both sides of the gap are used
    env = SingleAgentEnv("hearts", seed=0)
    for s in range(50):
        env.reset(seed=s)


def test_pool_survives_pruned_or_corrupt_snapshots(tmp_path):
    """Regression: a worker picked a snapshot the trainer had just pruned and crashed."""
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (pool_dir / "snap_0000000100.npz").write_bytes(b"not an npz")     # corrupt / half-written
    pool = OpponentPool("hearts", pool_dir, mix={"recent": 1.0}, seed=0)
    game = make_game("hearts", seed=0)
    pool.sample(game, 0)                                               # loads the corrupt file
    pool.snapshots.append(pool_dir / "snap_0000000200.npz")            # stale entry: file never existed
    for _ in range(20):
        opponents = pool.sample(game, 0)
        assert all(p.name in {"heuristic", "random", "agent"} for p in opponents.values())


def test_pool_ignores_snapshots_from_an_older_observation_layout(tmp_path):
    import numpy as np

    from card_games_rl.envs import OpponentPool
    from card_games_rl.games import make_game

    # A snapshot whose first layer expects 98 inputs (old Bluff), not the current 129.
    np.savez(tmp_path / "snap_0000001000.npz", layers=np.int64(1), activation=np.array("tanh"),
             W0=np.zeros((8, 98), np.float32), b0=np.zeros(8, np.float32),
             Wa=np.zeros((29, 8), np.float32), ba=np.zeros(29, np.float32))
    pool = OpponentPool("bluff", tmp_path, mix={"recent": 1.0}, seed=0)
    game = make_game("bluff", seed=0)
    for _ in range(3):
        opponents = pool.sample(game, 0)
        assert all(policy is pool.heuristic for policy in opponents.values())
    assert pool.snapshots == []
