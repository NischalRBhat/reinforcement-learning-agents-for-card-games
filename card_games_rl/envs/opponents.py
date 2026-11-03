"""Opponent samplers: who plays the non-learner seats in an episode."""
from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from card_games_rl.core import Game, Policy, RandomPolicy
from card_games_rl.games import game_class, heuristic_policy


class OpponentSampler:
    def sample(self, game: Game, learner_seat: int) -> dict[int, Policy]:
        raise NotImplementedError


class FixedOpponents(OpponentSampler):
    """Every opponent seat uses a policy from the same factory."""

    def __init__(self, factory: Callable[[], Policy]):
        self.policy = factory()

    def sample(self, game: Game, learner_seat: int) -> dict[int, Policy]:
        return {p: self.policy for p in range(game.num_players) if p != learner_seat}


def heuristic_opponents(game_id: str, seed: int | None = None) -> FixedOpponents:
    return FixedOpponents(lambda: heuristic_policy(game_id, seed))


def random_opponents(seed: int | None = None) -> FixedOpponents:
    return FixedOpponents(lambda: RandomPolicy(seed))


class OpponentPool(OpponentSampler):
    """Self-play pool: frozen snapshots plus heuristic and random players.

    Snapshots are ``snap_*.npz`` actor weights (numpy, no torch) that the trainer writes to ``pool_dir``. The
    directory is re-scanned every ``refresh_every`` episodes so subprocess
    workers pick up new snapshots without any IPC. Mix shares follow
    docs/ENV_DESIGN.md §1.4.
    """

    DEFAULT_MIX = {"recent": 0.50, "old": 0.30, "heuristic": 0.15, "random": 0.05}

    def __init__(self, game_id: str, pool_dir: str | Path, mix: dict[str, float] | None = None,
                 recent_k: int = 3, refresh_every: int = 100, cache_size: int = 8, seed: int | None = None):
        self.game_id = game_id
        self.pool_dir = Path(pool_dir)
        self.mix = mix or dict(self.DEFAULT_MIX)
        self.recent_k = recent_k
        self.refresh_every = refresh_every
        self.cache_size = cache_size
        self.rng = random.Random(seed)
        self.heuristic = heuristic_policy(game_id, seed)
        self.random = RandomPolicy(seed)
        self.snapshots: list[Path] = []
        self._incompatible: set[Path] = set()  # snapshots from an older observation layout
        self._episodes = 0
        self._cache: OrderedDict[Path, Policy] = OrderedDict()

    def _refresh(self) -> None:
        # Names are zero-padded step counts (snap_0000250000.zip), so sorting by
        # name is chronological and needs no stat() on files that may vanish.
        if self.pool_dir.exists():
            self.snapshots = sorted(p for p in self.pool_dir.glob("snap_*.npz") if p not in self._incompatible)

    def _load(self, path: Path) -> Policy:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        from card_games_rl.agents.numpy_policy import NumpyPolicy

        try:
            policy = NumpyPolicy(path, deterministic=False, seed=self.rng.randrange(2**31))
        except Exception:  # noqa: BLE001 - pruned or unreadable snapshot: never crash a worker
            # The trainer prunes old snapshots; this worker's list was stale.
            self._refresh()
            return self.heuristic
        if policy.layers[0][0].shape[1] != game_class(self.game_id).obs_size:
            # Left over from before an observation change (OBS_VERSION bump): never use it.
            self._incompatible.add(path)
            self._refresh()
            return self.heuristic
        self._cache[path] = policy
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return policy

    def _pick(self) -> Policy:
        kinds = list(self.mix)
        kind = self.rng.choices(kinds, weights=[self.mix[k] for k in kinds])[0]
        recent, old = self.snapshots[-self.recent_k:], self.snapshots[:-self.recent_k]
        if kind == "old" and not old:
            kind = "recent"
        if kind == "recent" and not recent:
            kind = "heuristic"
        if kind == "recent":
            return self._load(self.rng.choice(recent))
        if kind == "old":
            return self._load(self.rng.choice(old))
        return self.heuristic if kind == "heuristic" else self.random

    def sample(self, game: Game, learner_seat: int) -> dict[int, Policy]:
        if self._episodes % self.refresh_every == 0:
            self._refresh()
        self._episodes += 1
        return {p: self._pick() for p in range(game.num_players) if p != learner_seat}
