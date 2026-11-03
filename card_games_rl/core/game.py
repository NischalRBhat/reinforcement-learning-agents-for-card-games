"""Engine contract shared by every card game (see docs/ENV_DESIGN.md §1.1)."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class IllegalAction(ValueError):
    """Raised when an action is not legal in the current state."""


class Game(ABC):
    """Pure-Python, seeded, deterministic N-seat card game.

    Subclasses implement the rules; training, evaluation, the API and the UI
    all drive games exclusively through this interface.
    """

    game_id: str = ""
    name: str = ""
    num_players: int = 1
    num_actions: int = 0
    obs_size: int = 0
    OBS_VERSION: int = 1
    # Weight of potential-based shaping used by the RL wrapper (λ in the spec).
    shaping_weight: float = 0.0
    # Small per-decision cost charged to the learner (only Klondike uses it).
    step_cost: float = 0.0
    # True when returns always sum to zero across seats.
    zero_sum: bool = False

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.is_terminal = False
        self.is_truncated = False
        self.history: list[tuple[int, int]] = []  # (player, action)
        self._legal: list[int] | None = None  # cache for the current state

    # ----- lifecycle -------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        self.is_terminal = False
        self.is_truncated = False
        self.history = []
        self._legal = None
        self._deal()

    @abstractmethod
    def _deal(self) -> None:
        """Set up a fresh episode using ``self.rng``."""

    @property
    def done(self) -> bool:
        return self.is_terminal or self.is_truncated

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Seat to act, or -1 when the episode is over."""

    def legal_actions(self) -> list[int]:
        """Legal actions for ``current_player`` (cached per state; treat as read-only)."""
        if self.done:
            return []
        if self._legal is None:
            self._legal = self._legal_actions()
        return self._legal

    def invalidate(self) -> None:
        """Drop cached state-derived data; call after mutating state outside ``step``."""
        self._legal = None

    @abstractmethod
    def _legal_actions(self) -> list[int]:
        """Compute the legal actions for the current state."""

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=bool)
        mask[self.legal_actions()] = True
        return mask

    def step(self, action: int) -> None:
        action = int(action)
        if self.done:
            raise IllegalAction("Episode is over; call reset().")
        if action not in self.legal_actions():
            raise IllegalAction(f"Action {action} ({self.action_label(action)}) is not legal")
        player = self.current_player
        self.history.append((player, action))
        self._legal = None
        self._apply(action)

    @abstractmethod
    def _apply(self, action: int) -> None:
        """Apply a legal action for ``current_player``."""

    # ----- rewards ---------------------------------------------------------
    @abstractmethod
    def score(self, player: int) -> float:
        """Realized utility so far; equals the final return at the end."""

    def returns(self) -> list[float]:
        return [self.score(p) for p in range(self.num_players)]

    def potential(self, player: int) -> float:
        """Shaping potential Φ (0 unless overridden; the wrapper forces Φ=0 at the end)."""
        return 0.0

    # ----- observation / presentation --------------------------------------
    @abstractmethod
    def observation(self, player: int) -> np.ndarray:
        """float32 vector of length ``obs_size`` with only what ``player`` may know."""

    @abstractmethod
    def action_label(self, action: int) -> str:
        ...

    def public_label(self, action: int) -> str:
        """What other players see of this action (override when the label reveals hidden cards)."""
        return self.action_label(action)

    @abstractmethod
    def view(self, viewer: int | None) -> dict[str, Any]:
        """JSON-friendly state for the UI; hides cards ``viewer`` cannot see.

        ``viewer=None`` means spectator: all private cards are hidden.
        """

    def all_cards(self) -> list[int]:
        """Every card in every zone; used by tests to check card conservation."""
        raise NotImplementedError

    def winners(self) -> list[int]:
        """Seats with the best final return (empty while the game runs)."""
        if not self.done:
            return []
        values = self.returns()
        best = max(values)
        return [p for p, v in enumerate(values) if abs(v - best) < 1e-9]
