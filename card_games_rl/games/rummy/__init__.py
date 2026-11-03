from .bots import HeuristicPolicy
from .rules import RummyGame

GAME_CLASS = RummyGame

__all__ = ["GAME_CLASS", "RummyGame", "HeuristicPolicy"]
