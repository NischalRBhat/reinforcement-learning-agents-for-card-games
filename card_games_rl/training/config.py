"""Training configuration: ``configs/<game>.yaml`` with smoke/full profiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_config(game_id: str, profile: str | None = None) -> dict[str, Any]:
    path = CONFIG_DIR / f"{game_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No training config for {game_id!r} ({path})")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if profile is not None:
        if profile not in config["profiles"]:
            raise ValueError(f"{game_id}: unknown profile {profile!r}; choose from {list(config['profiles'])}")
        config["run"] = dict(config["profiles"][profile], profile=profile)
    return config


def parse_duration(text: str | None) -> float | None:
    """'3h', '90m', '45s', '1.5h' → seconds."""
    if not text:
        return None
    text = text.strip().lower()
    units = {"h": 3600, "m": 60, "s": 1}
    if text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)
