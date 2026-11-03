"""Policies choose actions for a seat; bots, trained models and humans share this shape."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np

from .game import Game


def sample_legal(probs: np.ndarray, rng: random.Random) -> int:
    """Sample an action with probability > 0.

    Only legal (non-zero) entries are considered, in float64, and the index is
    clamped. ``random.choices`` on float32 weights can round ``random() * total``
    up to ``total`` (~1 in 16M draws) and then returns the *last* index even if
    its weight is zero, which surfaced as an illegal move in long training runs.
    """
    legal = np.flatnonzero(probs > 0)
    cumulative = np.cumsum(probs[legal], dtype=np.float64)
    index = int(np.searchsorted(cumulative, rng.random() * cumulative[-1], side="right"))
    return int(legal[min(index, len(legal) - 1)])


class Policy(ABC):
    name: str = "policy"

    @abstractmethod
    def act(self, game: Game, player: int) -> int:
        ...

    def action_probabilities(self, game: Game, player: int) -> dict[int, float] | None:
        """Optional: probability per legal action (used by the UI insight panel)."""
        return None


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, game: Game, player: int) -> int:
        return self.rng.choice(game.legal_actions())

    def action_probabilities(self, game: Game, player: int) -> dict[int, float]:
        legal = game.legal_actions()
        return {a: 1.0 / len(legal) for a in legal}
