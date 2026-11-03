"""Torch-free inference for self-play snapshots.

Env workers only need forward passes of frozen opponents. Doing that in numpy
keeps PyTorch (≈0.5 GB RSS per process with the CUDA build) out of the
workers entirely, which is what makes GPU training fit in RAM.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

from card_games_rl.core import Game, Policy
from card_games_rl.core.policy import sample_legal


def export_policy(model, path: str | Path) -> None:
    """Write the actor of a MaskablePPO model to ``path`` (.npz) atomically."""
    import torch

    policy = model.policy
    weights: dict[str, np.ndarray] = {}
    activation = None
    layers = 0
    for module in policy.mlp_extractor.policy_net:
        if isinstance(module, torch.nn.Linear):
            weights[f"W{layers}"] = module.weight.detach().cpu().numpy()
            weights[f"b{layers}"] = module.bias.detach().cpu().numpy()
            layers += 1
        elif activation is None:
            activation = type(module).__name__.lower()
    weights["Wa"] = policy.action_net.weight.detach().cpu().numpy()
    weights["ba"] = policy.action_net.bias.detach().cpu().numpy()
    path = Path(path)
    temp = path.with_name("_tmp_" + path.name)  # must not match the pool's snap_*.npz glob
    with open(temp, "wb") as stream:
        np.savez(stream, layers=np.int64(layers), activation=np.array(activation or "tanh"), **weights)
    os.replace(temp, path)


class NumpyPolicy(Policy):
    name = "agent"

    _ACTIVATIONS = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0.0)}

    def __init__(self, path: str | Path, deterministic: bool = False, seed: int | None = None):
        with np.load(path) as data:
            layers = int(data["layers"])
            self.layers = [(data[f"W{i}"].astype(np.float32), data[f"b{i}"].astype(np.float32)) for i in range(layers)]
            self.head = (data["Wa"].astype(np.float32), data["ba"].astype(np.float32))
            self.activation = self._ACTIVATIONS[str(data["activation"])]
        self.deterministic = deterministic
        self.rng = random.Random(seed)

    def probabilities(self, game: Game, player: int) -> np.ndarray:
        h = np.clip(game.observation(player), 0.0, 1.0).astype(np.float32)
        for weight, bias in self.layers:
            h = self.activation(weight @ h + bias)
        logits = (self.head[0] @ h + self.head[1]).astype(np.float64)
        mask = game.action_mask()
        logits = np.where(mask, logits, -np.inf)
        logits -= logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def act(self, game: Game, player: int) -> int:
        probs = self.probabilities(game, player)
        return int(np.argmax(probs)) if self.deterministic else sample_legal(probs, self.rng)

    def action_probabilities(self, game: Game, player: int) -> dict[int, float]:
        probs = self.probabilities(game, player)
        return {int(a): float(probs[a]) for a in np.flatnonzero(probs > 0)}
