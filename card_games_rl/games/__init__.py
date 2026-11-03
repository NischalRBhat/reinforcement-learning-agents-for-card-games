"""Game registry: the only place that knows which games exist.

Each game package exposes ``GAME_CLASS`` and ``HeuristicPolicy``. Adding a game
means adding a package and one ``GameSpec`` entry here.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from card_games_rl.core import Game, Policy, RandomPolicy


@dataclass(frozen=True)
class GameSpec:
    game_id: str
    name: str
    category: str  # solo | dealer | competitive
    description: str
    min_players: int
    max_players: int
    modes: tuple[str, ...] = field(default=("human_vs_system", "human_vs_agent", "agent_vs_agent"))

    @property
    def module(self):
        return importlib.import_module(f"card_games_rl.games.{self.game_id}")


GAME_SPECS: dict[str, GameSpec] = {
    spec.game_id: spec
    for spec in (
        GameSpec("blackjack", "Blackjack", "dealer", "6-deck shoe, dealer stands on soft 17, 3:2, double, split, surrender, insurance.", 1, 1),
        GameSpec("klondike", "Klondike Solitaire", "solo", "Draw-1 Klondike with unlimited recycles and auto safe moves.", 1, 1),
        GameSpec("poker", "Texas Hold'em", "competitive", "Heads-up no-limit hold'em, 100 big blinds.", 2, 2),
        GameSpec("rummy", "Rummy", "competitive", "Two-player basic 10-card rummy: sets and runs, go out to win.", 2, 2),
        GameSpec("bluff", "Bluff", "competitive", "Four-player Cheat: claim ranks face-down, call out liars.", 4, 4),
        GameSpec("hearts", "Hearts", "competitive", "Four-player Black Lady hearts with passing and shooting the moon.", 4, 4),
        GameSpec("donkey", "Donkey", "competitive", "Gadha Ladam: follow suit to discard, cut to make others pick up; don't be the last one holding cards.", 4, 4),
    )
}


def get_spec(game_id: str) -> GameSpec:
    if game_id not in GAME_SPECS:
        raise ValueError(f"Unknown game {game_id!r}; choose from {sorted(GAME_SPECS)}")
    return GAME_SPECS[game_id]


def game_class(game_id: str) -> type[Game]:
    return get_spec(game_id).module.GAME_CLASS


def make_game(game_id: str, seed: int | None = None, **kwargs) -> Game:
    game = game_class(game_id)(seed=seed, **kwargs)
    game.reset(seed)
    return game


def heuristic_policy(game_id: str, seed: int | None = None) -> Policy:
    return get_spec(game_id).module.HeuristicPolicy(seed=seed)


def random_policy(game_id: str, seed: int | None = None) -> Policy:
    return RandomPolicy(seed=seed)


__all__ = ["GAME_SPECS", "GameSpec", "get_spec", "game_class", "make_game", "heuristic_policy", "random_policy"]
