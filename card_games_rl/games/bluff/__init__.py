from .bots import HeuristicPolicy
from .rules import BluffGame

GAME_CLASS = BluffGame

__all__ = ["GAME_CLASS", "BluffGame", "HeuristicPolicy"]
