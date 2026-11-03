from .bots import HeuristicPolicy
from .rules import HeartsGame

GAME_CLASS = HeartsGame

__all__ = ["GAME_CLASS", "HeartsGame", "HeuristicPolicy"]
