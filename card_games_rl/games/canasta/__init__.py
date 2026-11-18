from .bots import HeuristicPolicy
from .rules import CanastaGame

GAME_CLASS = CanastaGame

__all__ = ["GAME_CLASS", "CanastaGame", "HeuristicPolicy"]
