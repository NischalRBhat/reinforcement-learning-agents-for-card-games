"""Encoders: how an agent sees a game and which actions it chooses from.

* ``base``: the engine's own ``observation()``. Every existing agent uses it, and it
  never changes without an ``OBS_VERSION`` bump.
* ``counting``: the base observation followed by card-counting features computed
  from public information only (``games/<game>/counting.py``, with its own
  ``COUNTING_VERSION``).

A *variant* names an agent kind. Variant agents live in their own model folder
(``models/<game>-<variant>/``), so they never replace the base agent. Rummy's
counting agent is a search policy and needs no encoder.

*Planned* games (``games/<game>/plan.py``, Canasta) give their agents a smaller set of
strategic actions that the encoder translates to engine actions, plus planner features
in the observation. Everything outside the agent (opponents, API, evaluation) keeps
exchanging engine actions.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from card_games_rl.core import Game
from card_games_rl.games import game_class

BASE, COUNTING = "base", "counting"
VARIANTS = (BASE, COUNTING)
COUNTING_ENCODERS = ("donkey", "hearts", "bluff")   # games with counting features (PPO agents)
COUNTING_SEARCH = ("rummy",)                        # games whose counting agent is a search policy
PLANNED = ("canasta",)                              # agents act through planner actions


def variants_for(game_id: str) -> list[str]:
    return [BASE, COUNTING] if game_id in COUNTING_ENCODERS + COUNTING_SEARCH else [BASE]


def agent_key(game_id: str, variant: str = BASE) -> str:
    """Folder name of an agent: ``donkey`` or ``donkey-counting``."""
    return game_id if variant == BASE else f"{game_id}-{variant}"


def check_variant(game_id: str, variant: str) -> None:
    if variant not in variants_for(game_id):
        raise ValueError(f"{game_id} has no {variant!r} agent; choose from {variants_for(game_id)}")


def counting_module(game_id: str):
    return importlib.import_module(f"card_games_rl.games.{game_id}.counting")


def plan_module(game_id: str):
    return importlib.import_module(f"card_games_rl.games.{game_id}.plan")


@dataclass(frozen=True)
class Encoder:
    game_id: str
    variant: str
    size: int
    version: tuple[int, int]   # (OBS_VERSION, COUNTING_VERSION or PLAN_VERSION, else 0)
    num_actions: int           # the agent's action count (engine actions unless planned)
    planned: bool = False

    def __call__(self, game: Game, player: int) -> np.ndarray:
        base = game.observation(player)
        if self.variant == COUNTING:
            base = np.concatenate([base, counting_module(self.game_id).features(game, player)])
        if self.planned:
            base = np.concatenate([base, plan_module(self.game_id).features(game, player)])
        return np.clip(base, 0.0, 1.0).astype(np.float32, copy=False)

    # ----- the agent's actions ---------------------------------------------
    def mask(self, game: Game) -> np.ndarray:
        return plan_module(self.game_id).mask(game) if self.planned else game.action_mask()

    def legal(self, game: Game) -> list[int]:
        return plan_module(self.game_id).legal(game) if self.planned else game.legal_actions()

    def to_game(self, game: Game, action: int) -> int:
        """Agent action -> engine action."""
        return plan_module(self.game_id).to_game(game, action) if self.planned else action

    def from_game(self, game: Game, action: int) -> int:
        """Engine action -> agent action (to imitate a teacher that plays engine actions)."""
        return plan_module(self.game_id).from_game(game, action) if self.planned else action


@lru_cache(maxsize=None)
def get_encoder(game_id: str, variant: str = BASE) -> Encoder:
    cls = game_class(game_id)
    if game_id in PLANNED:
        check_variant(game_id, variant)
        plan = plan_module(game_id)
        return Encoder(game_id, BASE, cls.obs_size + plan.SIZE, (cls.OBS_VERSION, plan.PLAN_VERSION),
                       plan.NUM_PLAN_ACTIONS, planned=True)
    if variant == BASE or game_id not in COUNTING_ENCODERS:
        check_variant(game_id, variant)
        return Encoder(game_id, BASE, cls.obs_size, (cls.OBS_VERSION, 0), cls.num_actions)
    module = counting_module(game_id)
    return Encoder(game_id, COUNTING, cls.obs_size + module.SIZE, (cls.OBS_VERSION, module.COUNTING_VERSION),
                   cls.num_actions)
