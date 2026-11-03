from .bots import HeuristicPolicy
from .rules import DonkeyGame

GAME_CLASS = DonkeyGame

__all__ = ["GAME_CLASS", "DonkeyGame", "HeuristicPolicy"]
