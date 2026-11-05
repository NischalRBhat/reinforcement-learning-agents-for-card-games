"""Behaviour-cloning warm start: imitate the System player before PPO.

PPO from a random network spends millions of steps rediscovering basics (e.g.
which Rummy discard keeps melds together). Instead, the policy head is first
trained to imitate a teacher (the heuristic, or the exact optimal policy for
Blackjack) and the value head to predict the teacher's returns, from games
against heuristic opponents. PPO then starts near teacher strength and spends
its budget on *beating* it.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from card_games_rl.core import Policy
from card_games_rl.envs import SingleAgentEnv, heuristic_opponents
from card_games_rl.games import GAME_SPECS, heuristic_policy


def teacher_for(game_id: str, seed: int = 0) -> Policy:
    if game_id == "blackjack":
        from card_games_rl.games.blackjack import OptimalPolicy
        return OptimalPolicy()
    return heuristic_policy(game_id, seed)


def collect(game_id: str, config: dict[str, Any], episodes: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Teacher trajectories from the learner's point of view (same rewards as training)."""
    opponents = heuristic_opponents(game_id, seed) if GAME_SPECS[game_id].max_players > 1 else None
    env = SingleAgentEnv(game_id, opponents=opponents, seed=seed, game_kwargs=config.get("game_kwargs") or {},
                         variant=config.get("variant", "base"))
    teacher = teacher_for(game_id, seed)
    gamma = config["ppo"]["gamma"]
    observations, masks, actions, returns = [], [], [], []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + episode)
        rewards, done = [], False
        while not done:
            action = teacher.act(env.game, env.seat)
            observations.append(obs)
            masks.append(env.action_masks())
            actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
        running, episode_returns = 0.0, []
        for reward in reversed(rewards):
            running = reward + gamma * running
            episode_returns.append(running)
        returns.extend(reversed(episode_returns))
    return {"obs": np.asarray(observations, np.float32), "masks": np.asarray(masks, bool),
            "actions": np.asarray(actions, np.int64), "returns": np.asarray(returns, np.float32)}


def pretrain(model, data: dict[str, np.ndarray], epochs: int = 8, batch_size: int = 512,
             learning_rate: float = 1e-3, seed: int = 0) -> dict[str, float]:
    """Cross-entropy on teacher actions + MSE on returns; returns final accuracy and losses."""
    import torch

    policy = model.policy
    device = policy.device
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    n = len(data["actions"])
    policy.set_training_mode(True)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            obs = torch.as_tensor(data["obs"][idx], device=device)
            acts = torch.as_tensor(data["actions"][idx], device=device)
            rets = torch.as_tensor(data["returns"][idx], device=device)
            values, log_prob, _ = policy.evaluate_actions(obs, acts, action_masks=data["masks"][idx])
            loss = -log_prob.mean() + 0.5 * torch.nn.functional.mse_loss(values.flatten(), rets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
    policy.set_training_mode(False)
    with torch.no_grad():
        sample = slice(0, min(n, 20_000))
        obs = torch.as_tensor(data["obs"][sample], device=device)
        dist = policy.get_distribution(obs, action_masks=data["masks"][sample])
        predicted = dist.distribution.probs.argmax(dim=1).cpu().numpy()
        values = policy.predict_values(obs).flatten().cpu().numpy()
    return {"samples": float(n), "accuracy": float((predicted == data["actions"][sample]).mean()),
            "value_mse": float(np.mean((values - data["returns"][sample]) ** 2))}
