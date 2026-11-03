"""Lightweight subprocess VecEnv for SB3 training.

Unlike SB3's ``SubprocVecEnv``, the workers run ``card_games_rl.envs.worker``,
which never imports torch or stable-baselines3. With the CUDA build of
PyTorch that saves roughly 0.5 GB RSS per worker, so GPU training fits in RAM.
Action masks come back with every reset/step, so ``env_method("action_masks")``
(called by MaskablePPO before every action) costs no extra round trip.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from card_games_rl.envs import worker
from card_games_rl.games import game_class


class LightVecEnv(VecEnv):
    def __init__(self, specs: list[dict[str, Any]]):
        game = game_class(specs[0]["game_id"])
        observation_space = spaces.Box(0.0, 1.0, shape=(game.obs_size,), dtype=np.float32)
        super().__init__(len(specs), observation_space, spaces.Discrete(game.num_actions))
        context = mp.get_context("spawn")
        self.remotes, self.processes = [], []
        for spec in specs:
            parent, child = context.Pipe()
            process = context.Process(target=worker.run, args=(child, spec), daemon=True)
            process.start()
            child.close()
            self.remotes.append(parent)
            self.processes.append(process)
        self.masks = [np.ones(game.num_actions, dtype=bool)] * len(specs)
        self.closed = False

    def reset(self):
        for remote, seed in zip(self.remotes, self._seeds):
            remote.send(("reset", seed))
        results = [remote.recv() for remote in self.remotes]
        self._reset_seeds()
        self._reset_options()
        observations, self.masks, self.reset_infos = (list(x) for x in zip(*results))
        return np.stack(observations)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        observations, rewards, dones, infos, masks = zip(*results)
        self.masks = list(masks)
        return np.stack(observations), np.asarray(rewards, np.float32), np.asarray(dones, bool), list(infos)

    def close(self) -> None:
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close",))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
        self.closed = True

    def _indices(self, indices) -> list[int]:
        if indices is None:
            return list(range(self.num_envs))
        return [indices] if isinstance(indices, int) else list(indices)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list[Any]:
        targets = self._indices(indices)
        if method_name == "action_masks":
            return [self.masks[i] for i in targets]
        for i in targets:
            self.remotes[i].send(("call", method_name, args, kwargs))
        return [self.remotes[i].recv() for i in targets]

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        targets = self._indices(indices)
        if attr_name == "render_mode":  # queried by VecEnv.__init__ before workers exist
            return [None] * len(targets)
        for i in targets:
            self.remotes[i].send(("get", attr_name))
        return [self.remotes[i].recv() for i in targets]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        raise NotImplementedError("LightVecEnv does not support set_attr")

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False] * len(self._indices(indices))
