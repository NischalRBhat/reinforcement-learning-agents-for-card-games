"""Training pipeline, registry and evaluation (tiny runs into a temp folder)."""
import json

import numpy as np
import pytest

from card_games_rl.agents.registry import best_model_path, load_agent, read_meta
from card_games_rl.envs import OpponentPool
from card_games_rl.games import GAME_SPECS, make_game
from card_games_rl.training.config import load_config, parse_duration
from card_games_rl.training.evaluate import gate_passed, make_baseline, play
from card_games_rl.training.train import train

TINY = {"timesteps": 1024, "n_envs": 1, "eval_every": 512, "eval_episodes": 10, "snapshot_every": 512}


@pytest.mark.parametrize("game_id", sorted(GAME_SPECS))
def test_every_game_has_smoke_and_full_profiles(game_id):
    for profile in ("smoke", "full"):
        run = load_config(game_id, profile)["run"]
        assert run["timesteps"] > 0 and run["n_envs"] >= 1 and run["eval_episodes"] > 0


def test_parse_duration():
    assert parse_duration("3h") == 10800 and parse_duration("90m") == 5400 and parse_duration("45s") == 45
    assert parse_duration(None) is None


@pytest.fixture(scope="module")
def trained_hearts(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    runs = tmp_path_factory.mktemp("runs")
    path = train("hearts", "smoke", model_root=root, run_root=runs, overrides=TINY, torch_threads=1)
    return root, path


def test_training_writes_model_meta_and_snapshots(trained_hearts):
    root, path = trained_hearts
    assert path.exists()
    meta = read_meta("hearts", root)
    assert meta["obs_version"] == make_game("hearts").OBS_VERSION
    assert meta["validation"]["metric"] == "mean_points" and "passed" in meta["gate"]
    assert list((root / "hearts" / "pool").glob("snap_*.npz"))
    assert (root / "hearts" / "checkpoints" / "latest.zip").exists()


def test_loaded_agent_plays_only_legal_moves_with_probabilities(trained_hearts):
    root, _ = trained_hearts
    agent = load_agent("hearts", root=root)
    game = make_game("hearts", seed=1)
    while not game.done:
        player = game.current_player
        probs = agent.action_probabilities(game, player)
        assert set(probs) <= set(game.legal_actions()) and abs(sum(probs.values()) - 1) < 1e-4
        action = agent.act(game, player)
        assert action in game.legal_actions()
        game.step(action)


def test_incompatible_obs_version_is_rejected(trained_hearts):
    root, _ = trained_hearts
    meta_file = root / "hearts" / "meta.json"
    original = meta_file.read_text()
    try:
        data = json.loads(original)
        data["obs_version"] = 999
        meta_file.write_text(json.dumps(data))
        with pytest.warns(UserWarning):
            assert best_model_path("hearts", root) is None
    finally:
        meta_file.write_text(original)


def test_pool_loads_snapshots_as_opponents(trained_hearts):
    root, _ = trained_hearts
    pool = OpponentPool("hearts", root / "hearts" / "pool", mix={"recent": 1.0}, seed=0)
    game = make_game("hearts", seed=0)
    opponents = pool.sample(game, 0)
    assert {p.name for p in opponents.values()} == {"agent"}
    while not game.done:
        p = game.current_player
        policy = opponents.get(p) or make_baseline("hearts", "random")
        game.step(policy.act(game, p))


def test_gate_logic():
    higher = {"value": 0.6, "higher_is_better": True}
    lower = {"value": 5.5, "higher_is_better": False}
    assert gate_passed("rummy", higher, {"threshold": 0.55})
    assert gate_passed("hearts", lower, {"threshold": 6.0})
    assert not gate_passed("hearts", {"value": 6.5, "higher_is_better": False}, {"threshold": 6.0})
    assert gate_passed("blackjack", {"value": -0.012}, {"max_gap": 0.01}, reference={"value": -0.005})
    assert not gate_passed("blackjack", {"value": -0.03}, {"max_gap": 0.01}, reference={"value": -0.005})


def test_play_rotates_seats_and_reports_ci():
    result = play("hearts", make_baseline("hearts", "heuristic"), make_baseline("hearts", "random"), 40)
    assert result["metric"] == "mean_points" and result["ci95"] > 0 and result["episodes"] == 40


def test_sampling_never_returns_zero_probability_action():
    """Regression: float32 rounding made random.choices return the last (illegal) index."""
    from card_games_rl.agents.model_policy import sample_legal

    class EdgeRng:  # worst case: random() just below 1.0
        def random(self):
            return 1.0 - 2**-53

    probs = np.zeros(52, dtype=np.float32)
    probs[[3, 17, 20]] = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    assert sample_legal(probs, EdgeRng()) == 20
    rng = __import__("random").Random(0)
    assert {sample_legal(probs, rng) for _ in range(5000)} == {3, 17, 20}


def test_resume_counts_total_steps_and_keeps_schedules(tmp_path):
    """Regression: on --resume the cap is a total (SB3 would add it), and nothing re-runs."""
    from sb3_contrib import MaskablePPO

    root, runs = tmp_path / "models", tmp_path / "runs"
    train("blackjack", "smoke", model_root=root, run_root=runs, overrides=TINY, torch_threads=1)
    checkpoint = root / "blackjack" / "checkpoints" / "latest.zip"
    first = MaskablePPO.load(str(checkpoint)).num_timesteps
    best_before = read_meta("blackjack", root)["validation"]["value"]

    train("blackjack", "smoke", resume=True, model_root=root, run_root=runs, overrides=TINY, torch_threads=1)
    assert MaskablePPO.load(str(checkpoint)).num_timesteps == first  # cap already reached: no training

    train("blackjack", "smoke", resume=True, model_root=root, run_root=runs,
          overrides={**TINY, "timesteps": 2048}, torch_threads=1)
    resumed = MaskablePPO.load(str(checkpoint)).num_timesteps
    assert 2048 <= resumed < 2048 + 512 + 1   # total cap, not 1024 + 2048
    assert read_meta("blackjack", root)["validation"]["value"] >= best_before  # best never replaced by worse


def test_time_budget_is_total_across_resumes(tmp_path):
    from sb3_contrib import MaskablePPO

    root, runs = tmp_path / "models", tmp_path / "runs"
    train("blackjack", "smoke", model_root=root, run_root=runs, overrides=TINY, torch_threads=1)
    state = json.loads((root / "blackjack" / "checkpoints" / "state.json").read_text())
    assert state["trained_seconds"] > 0 and state["timesteps"] >= 1024
    checkpoint = root / "blackjack" / "checkpoints" / "latest.zip"
    before = MaskablePPO.load(str(checkpoint)).num_timesteps
    # Budget smaller than the time already used: a resume must not train at all.
    train("blackjack", "smoke", resume=True, time_budget="0.01s", model_root=root, run_root=runs,
          overrides={**TINY, "timesteps": 10_000}, torch_threads=1)
    assert MaskablePPO.load(str(checkpoint)).num_timesteps == before


def test_numpy_snapshot_matches_torch_policy(trained_hearts, tmp_path):
    from card_games_rl.agents.numpy_policy import NumpyPolicy, export_policy

    root, _ = trained_hearts
    torch_agent = load_agent("hearts", root=root)
    export_policy(torch_agent.model, tmp_path / "snap.npz")
    numpy_agent = NumpyPolicy(tmp_path / "snap.npz")
    game = make_game("hearts", seed=4)
    for _ in range(20):
        player = game.current_player
        np.testing.assert_allclose(numpy_agent.probabilities(game, player),
                                   torch_agent.probabilities(game, player), atol=1e-5)
        game.step(torch_agent.act(game, player))


def test_env_workers_never_import_torch():
    import subprocess
    import sys

    code = ("import sys; from card_games_rl.envs.worker import build_env; "
            "env = build_env({'game_id': 'hearts', 'seed': 0, 'opponents': 'pool', 'pool_dir': 'nonexistent'}); "
            "env.reset(seed=0); print('torch' in sys.modules, 'stable_baselines3' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def test_light_vec_env_matches_api_and_returns_masks():
    from card_games_rl.training.config import load_config
    from card_games_rl.training.train import make_vec_env

    config = load_config("hearts", "smoke")
    env = make_vec_env("hearts", config, 2, 0, __import__("pathlib").Path("nonexistent"))
    try:
        obs = env.reset()
        assert obs.shape == (2, make_game("hearts").obs_size)
        for _ in range(40):
            masks = np.stack(env.env_method("action_masks"))
            actions = [int(np.flatnonzero(m)[0]) for m in masks]
            obs, rewards, dones, infos = env.step(np.asarray(actions))
            for done, info in zip(dones, infos):
                if done:
                    assert "terminal_observation" in info and "episode" in info
    finally:
        env.close()
