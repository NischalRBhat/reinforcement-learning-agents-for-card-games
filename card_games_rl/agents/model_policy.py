"""Trained MaskablePPO policy usable at any seat (training pool, evaluation, API)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from card_games_rl.core import Game, Policy
from card_games_rl.core.policy import engine_probabilities, sample_legal  # noqa: F401  (re-exported)
from card_games_rl.encoders import get_encoder


class ModelPolicy(Policy):
    name = "agent"

    def __init__(self, path: str | Path | None = None, deterministic: bool = True, seed: int | None = None, model=None,
                 encoder=None):
        if model is None:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(str(path), device="cpu")
        self.path = Path(path) if path is not None else None
        self.model = model
        self.deterministic = deterministic
        self.rng = random.Random(seed)
        self.encoder = encoder  # None: the game's base encoder

    @classmethod
    def from_model(cls, model, deterministic: bool = True, encoder=None) -> "ModelPolicy":
        """Wrap a live (training) model without saving it first."""
        return cls(model=model, deterministic=deterministic, encoder=encoder)

    def _encoder(self, game: Game):
        return self.encoder or get_encoder(game.game_id)

    def probabilities(self, game: Game, player: int) -> np.ndarray:
        """Probabilities over the agent's own actions (planner actions for planned games)."""
        encoder = self._encoder(game)
        observation = encoder(game, player)
        mask = encoder.mask(game)
        obs_tensor, _ = self.model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor, action_masks=mask[None, :])
            probs = distribution.distribution.probs[0].cpu().numpy()
        probs = np.where(mask, probs, 0.0)
        total = probs.sum()
        return probs / total if total > 0 else mask / mask.sum()

    def act(self, game: Game, player: int) -> int:
        probs = self.probabilities(game, player)
        action = int(np.argmax(probs)) if self.deterministic else sample_legal(probs, self.rng)
        return self._encoder(game).to_game(game, action)

    def action_probabilities(self, game: Game, player: int) -> dict[int, float]:
        """Probabilities of engine actions (planner actions are translated)."""
        return engine_probabilities(self._encoder(game), game, self.probabilities(game, player))
