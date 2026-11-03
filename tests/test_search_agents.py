"""Non-PPO agents: the Rummy look-ahead search and Poker CFR, end to end through the registry."""
import random

import numpy as np
import pytest

from card_games_rl.agents.registry import load_agent, read_meta
from card_games_rl.games import heuristic_policy, make_game
from card_games_rl.games.poker.cfr import NUM_ACTIONS, CFRPolicy, MCCFR, merge_strategies, save_strategy
from card_games_rl.games.rummy.search import LookaheadPolicy
from card_games_rl.training.train import train


# ----- Rummy look-ahead ------------------------------------------------------
def test_lookahead_plays_whole_games_legally():
    agent, opponent = LookaheadPolicy(), heuristic_policy("rummy", 0)
    for seed in range(3):
        game = make_game("rummy", seed=seed)
        while not game.done:
            player = game.current_player
            action = (agent if player == 0 else opponent).act(game, player)
            assert action in game.legal_actions()
            game.step(action)


def test_lookahead_does_not_peek_at_hidden_cards():
    """Swapping the opponent's unknown cards with stock cards must not change its choice."""
    rng = random.Random(0)
    checked = 0
    for seed in range(30):
        game = make_game("rummy", seed=seed)
        player = game.current_player
        hidden = [c for c in game.hands[1 - player] if c not in game.known[player]]
        if len(game.stock) < len(hidden):
            continue
        before = LookaheadPolicy().act(game, player)
        swap = rng.sample(game.stock, len(hidden))
        game.stock = [c for c in game.stock if c not in swap] + hidden
        game.hands[1 - player] = sorted([c for c in game.hands[1 - player] if c not in hidden] + swap)
        game.invalidate()
        assert LookaheadPolicy().act(game, player) == before
        checked += 1
    assert checked > 10


def test_search_agent_is_registered_and_loaded(tmp_path):
    train("rummy", "smoke", model_root=tmp_path, run_root=tmp_path / "runs", overrides={"eval_episodes": 6})
    meta = read_meta("rummy", tmp_path)
    assert meta["algorithm"] == "search" and "passed" in meta["gate"]
    assert isinstance(load_agent("rummy", root=tmp_path), LookaheadPolicy)


# ----- Poker CFR -------------------------------------------------------------
def test_mccfr_strategies_are_distributions_over_legal_actions(tmp_path):
    solver = MCCFR(seed=1)
    for _ in range(40):
        solver.iterate()
    assert solver.regrets and all(r.shape == (NUM_ACTIONS,) and (r >= 0).all() for r in solver.regrets.values())
    path = tmp_path / "worker_0.pkl.gz"
    solver.save(path)
    assert MCCFR.load(path).iterations == 40
    strategy, iterations = merge_strategies([path, path])
    assert iterations == 80
    assert all(abs(p.sum() - 1) < 1e-5 for p in strategy.values())


def test_cfr_policy_plays_legally_and_falls_back_when_unseen():
    solver = MCCFR(seed=2)
    for _ in range(30):
        solver.iterate()
    strategy, _ = merge_strategies_from(solver)
    agent = CFRPolicy(strategy=strategy, seed=0)
    for seed in range(20):
        game = make_game("poker", seed=seed)
        while not game.done:
            player = game.current_player
            assert agent.act(game, player) in game.legal_actions()
            game.step(agent.act(game, player))
    empty = CFRPolicy(strategy={}, seed=0)
    game = make_game("poker", seed=3)
    assert empty.action_probabilities(game, game.current_player) is None
    assert empty.act(game, game.current_player) in game.legal_actions()


def merge_strategies_from(solver):
    return {k: v / v.sum() for k, v in solver.strategy_sum.items() if v.sum() > 0}, solver.iterations


def test_cfr_training_writes_strategy_meta_and_loads(tmp_path):
    train("poker", "smoke", model_root=tmp_path, run_root=tmp_path / "runs", n_envs=1,
          overrides={"iterations": 60, "eval_episodes": 8, "eval_every_seconds": 999})
    meta = read_meta("poker", tmp_path)
    assert meta["algorithm"] == "cfr" and meta["timesteps"] == 60 and meta["info_sets"] > 0
    assert (tmp_path / "poker" / "cfr_strategy.pkl.gz").exists()
    agent = load_agent("poker", root=tmp_path)
    assert isinstance(agent, CFRPolicy) and not agent.deterministic
