"""Gymnasium view of a multi-seat ``Game`` from one learner seat.

Other seats are driven by opponent policies and advanced automatically, so
from the learner's point of view every ``step`` goes from one of its decisions
to the next. Rewards follow docs/ENV_DESIGN.md §1.2:

    r = ΔS + λ·ΔΦ − step_cost,   Φ(terminal) = 0

and include everything that happened during opponents' turns.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from card_games_rl.core.seeds import training_seed
from card_games_rl.encoders import BASE, get_encoder
from card_games_rl.games import make_game

from .opponents import OpponentSampler, heuristic_opponents


class SingleAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        game_id: str,
        opponents: OpponentSampler | None = None,
        seed: int | None = None,
        shaping_weight: float | None = None,
        learner_seat: int | None = None,
        game_kwargs: dict | None = None,
        variant: str = BASE,
    ):
        super().__init__()
        self.game_id = game_id
        self.game = make_game(game_id, seed=seed, **(game_kwargs or {}))
        # What the learner sees: the engine's observation, or that plus card counting.
        self.encoder = get_encoder(game_id, variant)
        self.opponent_sampler = opponents or heuristic_opponents(game_id, seed)
        self.shaping_weight = self.game.shaping_weight if shaping_weight is None else shaping_weight
        self.fixed_seat = learner_seat
        # Masked agents never send illegal actions. "random" (replace with a
        # random legal move) exists only for unmasked tools such as check_env.
        self.illegal_action_mode = "raise"
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.encoder.size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.game.num_actions)
        self.seat = 0
        self.opponents = {}
        self._score = 0.0
        self._potential = 0.0
        self.episode_return, self.episode_length = 0.0, 0

    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        return self.encoder(self.game, self.seat)

    def _advance_opponents(self) -> None:
        game = self.game
        while not game.done and game.current_player != self.seat:
            player = game.current_player
            game.step(self.opponents[player].act(game, player))

    def _phi(self) -> float:
        return 0.0 if self.game.done else self.game.potential(self.seat)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Some deals end before the learner decides (e.g. blackjack naturals);
        # they carry no learning signal, so re-deal.
        for _ in range(1000):
            self.game.reset(training_seed(self.np_random))
            players = self.game.num_players
            self.seat = self.fixed_seat if self.fixed_seat is not None else int(self.np_random.integers(players))
            self.opponents = self.opponent_sampler.sample(self.game, self.seat) if players > 1 else {}
            # Reseed the opponents' own randomness from the env's, so an episode is
            # fully reproducible from the reset seed.
            for policy in {id(p): p for p in self.opponents.values()}.values():
                if hasattr(policy, "rng"):
                    policy.rng.seed(int(self.np_random.integers(2**31)))
            self._advance_opponents()
            if not self.game.done:
                break
        else:  # pragma: no cover - would indicate a broken game
            raise RuntimeError(f"{self.game_id}: could not reach a learner decision")
        self._score = self.game.score(self.seat)
        self._potential = self._phi()
        self.episode_return = 0.0
        self.episode_length = 0
        return self._observe(), {"seat": self.seat}

    def step(self, action):
        action = int(action)
        if not self.game.action_mask()[action]:
            if self.illegal_action_mode == "raise":
                raise ValueError(f"{self.game_id}: illegal action {action}; use action masks")
            action = int(self.np_random.choice(self.game.legal_actions()))
        self.game.step(action)
        self._advance_opponents()
        score, potential = self.game.score(self.seat), self._phi()
        reward = (score - self._score) + self.shaping_weight * (potential - self._potential) - self.game.step_cost
        self._score, self._potential = score, potential
        self.episode_return += reward
        self.episode_length += 1
        terminated, truncated = self.game.is_terminal, self.game.is_truncated and not self.game.is_terminal
        info = {"seat": self.seat}
        if terminated or truncated:
            info["final_score"] = score
            info["won"] = self.seat in self.game.winners()
            # SB3 reads episode return/length from here (no torch-importing Monitor needed).
            # No wall-clock field: infos must be deterministic for a given seed.
            info["episode"] = {"r": self.episode_return, "l": self.episode_length}
        return self._observe(), float(reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self.game.action_mask()
