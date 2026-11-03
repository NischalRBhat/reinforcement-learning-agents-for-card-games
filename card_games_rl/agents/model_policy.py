"""Trained MaskablePPO policy usable at any seat (training pool, evaluation, API)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from card_games_rl.core import Game, Policy
from card_games_rl.core.policy import sample_legal  # noqa: F401  (re-exported)


class ModelPolicy(Policy):
    name = "agent"

    def __init__(self, path: str | Path | None = None, deterministic: bool = True, seed: int | None = None, model=None):
        if model is None:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(str(path), device="cpu")
        self.path = Path(path) if path is not None else None
        self.model = model
        self.deterministic = deterministic
        self.rng = random.Random(seed)

    @classmethod
    def from_model(cls, model, deterministic: bool = True) -> "ModelPolicy":
        """Wrap a live (training) model without saving it first."""
        return cls(model=model, deterministic=deterministic)

    def probabilities(self, game: Game, player: int) -> np.ndarray:
        observation = np.clip(game.observation(player), 0.0, 1.0)
        mask = game.action_mask()
        obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor, action_masks=mask[None, :])
            probs = distribution.distribution.probs[0].cpu().numpy()
        probs = np.where(mask, probs, 0.0)
        total = probs.sum()
        return probs / total if total > 0 else mask / mask.sum()

    def act(self, game: Game, player: int) -> int:
        probs = self.probabilities(game, player)
        if self.deterministic:
            return int(np.argmax(probs))
        return sample_legal(probs, self.rng)

    def action_probabilities(self, game: Game, player: int) -> dict[int, float]:
        probs = self.probabilities(game, player)
        return {int(a): float(probs[a]) for a in np.flatnonzero(probs > 0)}
