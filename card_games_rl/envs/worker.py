"""Env worker process. Imports numpy/gymnasium only, never torch.

Protocol (multiprocessing Pipe), one env per worker:
  ("reset", seed)   -> (obs, mask, info)
  ("step", action)  -> (obs, reward, done, info, mask)   auto-resets like SB3 VecEnvs
  ("call", name, args, kwargs) -> result of env.<name>(*args, **kwargs)
  ("get", name)     -> getattr(env, name)
  ("close",)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_env(spec: dict[str, Any]):
    from .opponents import OpponentPool, heuristic_opponents
    from .single_agent import SingleAgentEnv

    game_id, seed = spec["game_id"], spec["seed"]
    kind = spec.get("opponents")
    if kind == "pool":
        opponents = OpponentPool(game_id, Path(spec["pool_dir"]), mix=spec.get("opponent_mix"), seed=seed,
                                 variant=spec.get("variant", "base"))
    elif kind == "heuristic":
        opponents = heuristic_opponents(game_id, seed)
    else:
        opponents = None
    return SingleAgentEnv(game_id, opponents=opponents, seed=seed, game_kwargs=spec.get("game_kwargs") or {},
                          variant=spec.get("variant", "base"))


def run(remote, spec: dict[str, Any]) -> None:
    env = build_env(spec)
    try:
        while True:
            command, *payload = remote.recv()
            if command == "step":
                obs, reward, terminated, truncated, info = env.step(int(payload[0]))
                done = terminated or truncated
                if done:
                    info["terminal_observation"] = obs
                    info["TimeLimit.truncated"] = truncated and not terminated
                    obs, reset_info = env.reset()
                remote.send((obs, reward, done, info, env.action_masks()))
            elif command == "reset":
                obs, info = env.reset(seed=payload[0])
                remote.send((obs, env.action_masks(), info))
            elif command == "call":
                name, args, kwargs = payload
                remote.send(getattr(env, name)(*args, **kwargs))
            elif command == "get":
                remote.send(getattr(env, payload[0]))
            elif command == "close":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        remote.close()
