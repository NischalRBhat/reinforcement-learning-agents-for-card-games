"""Model registry: ``models/<game>/best.zip`` plus ``meta.json``.

The metadata pins the observation version a model was trained on, so a model
from an older encoding is never silently loaded (the old project's failure mode).
"""
from __future__ import annotations

import importlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from card_games_rl.games import game_class

MODEL_ROOT = Path(__file__).resolve().parents[2] / "models"


def model_dir(game_id: str, root: Path | None = None) -> Path:
    return (root or MODEL_ROOT) / game_id


def meta_path(game_id: str, root: Path | None = None) -> Path:
    return model_dir(game_id, root) / "meta.json"


def read_meta(game_id: str, root: Path | None = None) -> dict[str, Any] | None:
    path = meta_path(game_id, root)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_meta(game_id: str, data: dict[str, Any], root: Path | None = None) -> None:
    path = meta_path(game_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"game": game_id, "obs_version": game_class(game_id).OBS_VERSION,
               "updated_at": datetime.now().isoformat(timespec="seconds"), **data}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


ARTIFACTS = {"MaskablePPO": "best.zip", "cfr": "cfr_strategy.pkl.gz", "search": "meta.json"}


def best_model_path(game_id: str, root: Path | None = None) -> Path | None:
    """Path of the usable trained agent's artifact, or None (missing or incompatible).

    PPO agents are ``best.zip``, CFR agents a strategy table, and search agents
    (no learned weights) only their ``meta.json``.
    """
    meta = read_meta(game_id, root)
    if meta is None:
        return None
    path = model_dir(game_id, root) / ARTIFACTS.get(meta.get("algorithm", "MaskablePPO"), "best.zip")
    if not path.exists():
        return None
    expected = game_class(game_id).OBS_VERSION
    if meta.get("obs_version") != expected:
        warnings.warn(f"{game_id}: model trained on obs v{meta.get('obs_version')}, engine is v{expected}; retrain")
        return None
    return path


def load_agent(game_id: str, deterministic: bool | None = None, root: Path | None = None):
    """Load the trained agent for ``game_id`` (None if not available).

    ``deterministic=None`` uses the setting the agent was evaluated with.
    """
    path = best_model_path(game_id, root)
    if path is None:
        return None
    meta = read_meta(game_id, root)
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

    return ModelPolicy(path, deterministic=deterministic)
