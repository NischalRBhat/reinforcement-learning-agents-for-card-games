from .bots import CallingStation, HeuristicPolicy
from .rules import PokerGame

GAME_CLASS = PokerGame

__all__ = ["GAME_CLASS", "PokerGame", "HeuristicPolicy", "CallingStation"]
