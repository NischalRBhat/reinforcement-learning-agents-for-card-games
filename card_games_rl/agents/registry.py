"""Model registry: ``models/<agent>/best.zip`` plus ``meta.json``.

``<agent>`` is the game id for the base agent and ``<game>-<variant>`` for variant
agents such as ``donkey-counting`` (see ``card_games_rl.encoders``), so a variant never
replaces the base agent. The metadata pins the observation versions a model was
trained on, so a model from an older encoding is never silently loaded.
"""
from __future__ import annotations

import importlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from card_games_rl.encoders import BASE, agent_key, get_encoder

MODEL_ROOT = Path(__file__).resolve().parents[2] / "models"


def model_dir(game_id: str, root: Path | None = None, variant: str = BASE) -> Path:
    return (root or MODEL_ROOT) / agent_key(game_id, variant)


def meta_path(game_id: str, root: Path | None = None, variant: str = BASE) -> Path:
    return model_dir(game_id, root, variant) / "meta.json"


def read_meta(game_id: str, root: Path | None = None, variant: str = BASE) -> dict[str, Any] | None:
    path = meta_path(game_id, root, variant)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_meta(game_id: str, data: dict[str, Any], root: Path | None = None, variant: str = BASE) -> None:
    path = meta_path(game_id, root, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_version, features_version = get_encoder(game_id, variant).version
    payload = {"game": game_id, "variant": variant, "obs_version": obs_version,
               "updated_at": datetime.now().isoformat(timespec="seconds"), **data}
    if features_version:  # counting features or planner actions
        payload["features_version"] = features_version
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


ARTIFACTS = {"MaskablePPO": "best.zip", "cfr": "cfr_strategy.pkl.gz", "search": "meta.json"}


def best_model_path(game_id: str, root: Path | None = None, variant: str = BASE) -> Path | None:
    """Path of the usable trained agent's artifact, or None (missing or incompatible).

    PPO agents are ``best.zip``, CFR agents a strategy table, and search agents
    (no learned weights) only their ``meta.json``.
    """
    meta = read_meta(game_id, root, variant)
    if meta is None:
        return None
    path = model_dir(game_id, root, variant) / ARTIFACTS.get(meta.get("algorithm", "MaskablePPO"), "best.zip")
    if not path.exists():
        return None
    current = get_encoder(game_id, variant).version
    # "counting_version" is the older name of "features_version".
    trained = (meta.get("obs_version"), meta.get("features_version", meta.get("counting_version", 0)))
    if trained != current:
        warnings.warn(f"{agent_key(game_id, variant)}: model trained on observation v{trained}, "
                      f"engine is v{current}; retrain")
        return None
    return path


def load_agent(game_id: str, deterministic: bool | None = None, root: Path | None = None, variant: str = BASE):
    """Load the trained agent for ``game_id`` (None if not available).

    ``deterministic=None`` uses the setting the agent was evaluated with.
    """
    path = best_model_path(game_id, root, variant)
    if path is None:
        return None
    meta = read_meta(game_id, root, variant)
    if deterministic is None:
        deterministic = meta.get("deterministic", True)
    algorithm = meta.get("algorithm", "MaskablePPO")
    if algorithm == "search":
        module, name = meta["policy"].split(":")
        return getattr(importlib.import_module(module), name)()
    if algorithm == "cfr":
        from card_games_rl.games.poker.cfr import CFRPolicy

        return CFRPolicy(path, deterministic=deterministic)
    from .model_policy import ModelPolicy

    return ModelPolicy(path, deterministic=deterministic, encoder=get_encoder(game_id, variant))
