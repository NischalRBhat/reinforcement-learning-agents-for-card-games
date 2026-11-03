from .bots import HeuristicPolicy
from .rules import KlondikeGame

GAME_CLASS = KlondikeGame

__all__ = ["GAME_CLASS", "KlondikeGame", "HeuristicPolicy"]
