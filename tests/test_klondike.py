import random

import numpy as np
import pytest

from card_games_rl.core import RandomPolicy
from card_games_rl.core.cards import cards, parse
from card_games_rl.games import make_game
from card_games_rl.games.klondike import HeuristicPolicy
from card_games_rl.games.klondike.rules import (DRAW, FOUNDATION_TO_TABLEAU, RECYCLE, STALL_LIMIT,
                                                TABLEAU_TO_FOUNDATION, WASTE_TO_TABLEAU, decode_move,
                                                encode_move)


def rig(columns, hidden=None, stock="", waste="", foundation=(0, 0, 0, 0), auto=False):
    game = make_game("klondike", seed=0, auto_safe_moves=auto)
    game.tableau = [cards(c) for c in columns] + [[] for _ in range(7 - len(columns))]
    game.hidden = list(hidden or [0] * 7) + [0] * (7 - len(hidden or [0] * 7))
    game.stock, game.waste = cards(stock), cards(waste)
    game.foundation = list(foundation)
    game.seen = set(game.waste)
    game.won = game.is_terminal = game.is_truncated = False
    game.steps = game.stall = game.revealed = 0
    game.start_tracking()
    return game


def test_initial_deal_layout():
    game = make_game("klondike", seed=4, auto_safe_moves=False)
    assert [len(p) for p in game.tableau] == [1, 2, 3, 4, 5, 6, 7]
    assert game.hidden == [0, 1, 2, 3, 4, 5, 6]
    assert len(game.stock) == 24 and game.waste == []


def test_only_kings_move_to_empty_column():
    game = rig(["5H", "", "KS"], waste="QD")
    game.tableau[1] = []
    legal = game.legal_actions()
    assert WASTE_TO_TABLEAU + 2 in legal          # QD on KS
    assert WASTE_TO_TABLEAU + 1 not in legal      # Q not into empty
    king_column = rig(["5H", ""], waste="KD")
    king_column.tableau[1] = []
    assert WASTE_TO_TABLEAU + 1 in king_column.legal_actions()


def test_tableau_requires_alternating_colours_descending():
    game = rig(["9H", "8D", "8S"])
    legal = game.legal_actions()
    assert encode_move(2, 0, 1) in legal       # 8S on 9H
    assert encode_move(1, 0, 1) not in legal   # 8D on 9H: same colour


def test_sequence_move_and_auto_flip():
    game = rig(["3C 10H 9S", "JS"], hidden=[1, 0])
    action = encode_move(0, 1, 2)                # move 10H 9S onto JS
    assert action in game.legal_actions()
    game.step(action)
    assert cards("JS 10H 9S") == game.tableau[1]
    assert game.hidden[0] == 0 and game.revealed == 1


def test_foundation_builds_up_by_suit_from_ace():
    game = rig(["2H", "AH"])
    legal = game.legal_actions()
    assert TABLEAU_TO_FOUNDATION + 1 in legal and TABLEAU_TO_FOUNDATION + 0 not in legal
    game.step(TABLEAU_TO_FOUNDATION + 1)
    assert TABLEAU_TO_FOUNDATION + 0 in game.legal_actions()


def test_draw_and_recycle_preserve_order():
    game = rig(["KS"], stock="2C 3C 4C")
    for _ in range(3):
        game.step(DRAW)
    assert game.waste == cards("4C 3C 2C") and RECYCLE in game.legal_actions()
    game.step(RECYCLE)
    assert game.stock == cards("2C 3C 4C") and game.waste == []


def test_foundation_to_tableau_only_when_not_auto_safe():
    unsafe = rig(["6S"], foundation=(0, 0, 5, 0), auto=True)   # 5H, spades/clubs empty -> unsafe
    assert FOUNDATION_TO_TABLEAU + 2 * 7 + 0 in unsafe.legal_actions()
    safe = rig(["3S"], foundation=(2, 0, 2, 2), auto=True)     # 2H is always safe
    assert FOUNDATION_TO_TABLEAU + 2 * 7 + 0 not in safe.legal_actions()


def test_auto_safe_moves_play_aces_and_twos():
    game = rig(["5C", "AD"], hidden=[0, 0], stock="9S", auto=True)
    game.tableau[1] = cards("7S AD")
    game.step(DRAW)
    assert game.foundation[1] == 1 and game.tableau[1] == cards("7S")


def test_move_encoding_roundtrip():
    seen = set()
    for src in range(7):
        for dst in range(7):
            if src != dst:
                for n in range(1, 14):
                    action = encode_move(src, dst, n)
                    assert decode_move(action) == (src, dst, n)
                    seen.add(action)
    assert len(seen) == 546 and max(seen) == 590


def test_observation_hides_face_down_cards():
    a = rig(["2C 3D KH", "4S"], hidden=[2, 0])
    b = rig(["5C 6D KH", "4S"], hidden=[2, 0])
    for game, swap in ((a, "2C 3D"), (b, "5C 6D")):
        rest = [c for c in range(52) if c not in game.all_cards()]
        game.stock = rest  # unseen stock
    np.testing.assert_array_equal(a.observation(0), b.observation(0))


def test_stall_truncation():
    game = rig(["KS"], stock="2C 3C")
    game.is_deadlocked = lambda *args, **kwargs: False  # test the safety net on its own
    for _ in range(STALL_LIMIT + 10):  # the first draws reveal new cards, which is progress
        if game.done:
            break
        game.step(DRAW if DRAW in game.legal_actions() else RECYCLE)
    assert game.is_truncated and not game.is_terminal


def test_win_detection_and_score():
    game = rig(["KS"], foundation=(13, 13, 13, 12))
    game.step(TABLEAU_TO_FOUNDATION)
    assert game.won and game.is_terminal and game.score(0) == pytest.approx(1.0)


@pytest.mark.slow
def test_heuristic_beats_random():
    def play(policy, games=150):
        wins, found = 0, 0
        game = make_game("klondike")
        for seed in range(games):
            game.reset(seed)
            while not game.done:
                game.step(policy.act(game, 0))
            wins += game.won
            found += sum(game.foundation)
        return wins / games, found / games

    heuristic, rand = play(HeuristicPolicy(0)), play(RandomPolicy(0))
    assert heuristic[1] > rand[1]


# ----- repetition, dead ends and the solver --------------------------------
from card_games_rl.games.klondike.rules import DEADLOCK_CHECK_EVERY  # noqa: E402
from card_games_rl.games.klondike.solver import NOT_WINNABLE, WON, solve  # noqa: E402


def test_moving_a_card_straight_back_is_hidden_while_other_moves_exist():
    game = rig(["10S 9H", "10C"], stock="4D")
    game.step(encode_move(0, 1, 1))                      # 9H onto 10C
    assert encode_move(1, 0, 1) not in game.legal_actions()  # back to a seen position
    assert DRAW in game.legal_actions()


def test_repeats_are_allowed_when_every_move_repeats():
    game = rig(["10S 9H", "10C"])
    game.step(encode_move(0, 1, 1))
    assert game.legal_actions() == [encode_move(1, 0, 1)]


def test_dead_end_ends_the_game_and_the_solver_agrees():
    # KS sits on a hidden 2C; the stock holds two cards that can never be played.
    game = rig(["2C KS"], hidden=[1], stock="5D 9H")
    for _ in range(DEADLOCK_CHECK_EVERY + 6):
        if game.done:
            break
        game.step(game.legal_actions()[0])
    assert game.deadlocked and game.is_terminal and not game.won
    assert solve(game).verdict == NOT_WINNABLE


def test_live_position_is_not_deadlocked():
    game = rig(["2C KS", "AC"], hidden=[1, 0], stock="5D 9H")
    assert not game.is_deadlocked()


def test_solver_finds_a_win_from_a_nearly_finished_game():
    game = rig(["KS QH", "KH"], foundation=(13, 13, 11, 12))
    assert solve(game).verdict == WON
