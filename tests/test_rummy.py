import numpy as np
import pytest

from card_games_rl.core import RandomPolicy
from card_games_rl.core.cards import cards, parse
from card_games_rl.games import make_game
from card_games_rl.games.rummy import HeuristicPolicy
from card_games_rl.games.rummy.melds import deadwood, is_meld
from card_games_rl.games.rummy.rules import DISCARD, DRAW_DISCARD, DRAW_STOCK


@pytest.mark.parametrize("text,valid", [
    ("7S 7H 7D", True), ("7S 7H 7D 7C", True), ("7S 7H", False),
    ("AS 2S 3S", True), ("QS KS AS", False), ("4H 5H 6H 7H", True), ("4H 5D 6H", False),
])
def test_meld_validation(text, valid):
    assert is_meld(cards(text)) is valid


def test_deadwood_minimises_over_overlapping_melds():
    assert deadwood(cards("AS 2S 3S 7H 7D 7C KC QD 5H 9S")) == 34
    assert deadwood(cards("7S 7H 7D 8S 9S")) == 14     # run 7-8-9S beats the set
    assert deadwood(cards("AS 2S 3S 4S 5S 6S 7S 8S 9S 10S")) == 0


def rig(hands, discard="2D", stock=None, turn=0):
    game = make_game("rummy", seed=0)
    game.hands = [sorted(cards(h)) for h in hands]
    game.discard = cards(discard)
    used = set(game.hands[0] + game.hands[1] + game.discard)
    game.stock = cards(stock) if stock is not None else [c for c in range(52) if c not in used]
    game.turn, game.phase, game.drawn_from_discard = turn, "draw", None
    game.restocked, game.turns, game.known = False, [0, 0], [set(), set()]
    return game


def test_cannot_discard_the_card_just_taken():
    game = rig(["AS 2S 3S 7H 7D 7C KC QD 5H 9S", "4C 4D 6C 6D 8C 8D 10C 10D JC JD"], discard="4S")
    game.step(DRAW_DISCARD)
    assert DISCARD + parse("4S") not in game.legal_actions()
    assert parse("4S") in game.known[1]


def test_going_out_wins_and_scores_opponent_deadwood():
    game = rig(["AS 2S 3S 7H 7D 7C 9C 10C JC KD", "4C 4D 6C 6D 8C 8D 10H 10D QH QD"], stock="5H QC")
    game.step(DRAW_STOCK)           # takes QC (top of stock = last element), completing 9-10-J-Q clubs
    game.step(DISCARD + parse("KD"))
    assert game.done and game.winner == 0
    opp = deadwood(game.hands[1])
    assert game.score(0) == pytest.approx(0.5 + 0.5 * min(opp, 100) / 100) and abs(sum(game.returns())) < 1e-9


def test_stock_restocks_once_then_draw():
    game = rig(["AS 3S 5S 7H 9D JC KC 2D 4H 6H", "8C 10C QC 2C 4C 6C 8D 10D QD KH"], discard="3C", stock="5D")
    game.step(DRAW_STOCK)
    game.step(DISCARD + parse("KC"))
    assert game.restocked and game.stock == cards("3C") and game.discard == cards("KC")
    game.step(DRAW_STOCK)
    game.step(DISCARD + parse("KH"))
    assert game.done and game.winner is None
    assert abs(sum(game.returns())) < 1e-9


def test_observation_hides_opponent_hand():
    a, b = make_game("rummy", seed=4), make_game("rummy", seed=4)
    b.hands[1][0], b.stock[0] = b.stock[0], b.hands[1][0]
    b.hands[1].sort()
    player = a.current_player
    np.testing.assert_array_equal(a.observation(player), b.observation(player))


@pytest.mark.slow
def test_heuristic_beats_random():
    wins, games = 0, 150
    game = make_game("rummy")
    bot, rand = HeuristicPolicy(), RandomPolicy(0)
    for seed in range(games):
        game.reset(seed)
        while not game.done:
            p = game.current_player
            game.step((bot if p == 0 else rand).act(game, p))
        wins += game.winner == 0
    assert wins / games > 0.8
