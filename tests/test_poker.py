import random

import numpy as np
import pytest

from card_games_rl.core import RandomPolicy
from card_games_rl.core.cards import cards
from card_games_rl.games import make_game
from card_games_rl.games.poker import CallingStation, HeuristicPolicy
from card_games_rl.games.poker.evaluator import equity, evaluate
from card_games_rl.games.poker.rules import (ALL_IN, CALL, FOLD, RAISE_2POT, RAISE_HALF, RAISE_POT,
                                             STACK)


def test_evaluator_category_order_and_tiebreaks():
    hands = [
        "2C 3C 4C 5C 6C",   # straight flush
        "9S 9H 9D 9C 2H",   # quads
        "8S 8H 8D 3C 3H",   # full house
        "2H 7H 9H JH KH",   # flush
        "AH 2C 3D 4S 5H",   # wheel straight
        "7S 7H 7D KC 2H",   # trips
        "7S 7H 5D 5C 2H",   # two pair
        "7S 7H 5D 4C 2H",   # pair
        "AS KH 5D 4C 2H",   # high card
    ]
    values = [evaluate(cards(h)) for h in hands]
    assert values == sorted(values, reverse=True)
    assert evaluate(cards("AH 2C 3D 4S 5H"))[1] == 5
    assert evaluate(cards("AS AH KD 4C 2H")) > evaluate(cards("AS AH QD JC 10H"))  # kicker
    assert evaluate(cards("8S 8H 8D 3C 3H 3D 2C"))[:3] == (6, 8, 3)  # two trips -> full house


def test_equity_aces_vs_kings():
    assert 0.76 < equity(cards("AS AH"), cards("KS KH"), [], samples=2000) < 0.88
    assert equity(cards("AS AH"), cards("KS KH"), cards("2C 7D 9H 3S 4C")) == 1.0


def new(button=0, seed=0):
    return make_game("poker", seed=seed, button=button)


def test_blinds_and_preflop_order():
    game = new(button=0)
    assert game.committed == [1, 2] and game.current_player == 0
    assert game.min_raise_to(0) == 4
    assert FOLD in game.legal_actions()


def test_small_blind_limp_gives_big_blind_option_then_flop_bb_first():
    game = new(button=0)
    game.step(CALL)                    # SB completes
    assert game.current_player == 1 and FOLD not in game.legal_actions()
    game.step(CALL)                    # BB checks
    assert game.street == 1 and len(game.board) == 3 and game.current_player == 1


def test_fold_awards_pot():
    game = new(button=0)
    game.step(RAISE_POT)
    game.step(FOLD)
    assert game.done and game.chip_result() == [2, -2]


def test_raise_reraise_call_advances_street():
    game = new(button=1)
    game.step(RAISE_HALF)
    game.step(RAISE_POT)
    assert game.street == 0
    game.step(CALL)
    assert game.street == 1 and game.street_bet == [0, 0]


def test_raise_sizes_are_deduplicated_and_capped_by_stack():
    game = new(button=0)
    targets = game.raise_targets(0)
    assert len(set(targets.values())) == len(targets)
    assert targets[ALL_IN] == STACK
    assert all(v < STACK for a, v in targets.items() if a != ALL_IN)


def test_all_in_and_call_runs_out_board():
    game = new(button=0, seed=3)
    game.step(ALL_IN)
    game.step(CALL)
    assert game.done and game.board_visible == 5 and game.showdown
    result = game.chip_result()
    assert sum(result) == 0 and abs(result[0]) in (0, STACK)


def test_no_reraise_against_all_in():
    game = new(button=0)
    game.step(ALL_IN)
    assert game.legal_actions() == [FOLD, CALL]


def test_observation_hides_opponent_hole_cards():
    a, b = new(seed=5), new(seed=5)
    b.hole[1], b.deck[:2] = b.deck[:2], b.hole[1]
    np.testing.assert_array_equal(a.observation(0), b.observation(0))


def test_allin_ev_is_zero_sum_and_bounded():
    game = make_game("poker", seed=9, button=0, allin_ev=True)
    game.step(ALL_IN)
    game.step(CALL)
    assert game.ev_net is not None and abs(sum(game.ev_net)) < 1e-9 and abs(game.ev_net[0]) <= 1


def _bb_per_100(hero, villain, hands=3000):
    game, total = make_game("poker"), 0
    for seed in range(hands):
        game.reset(seed)
        while not game.done:
            p = game.current_player
            game.step((hero if p == 0 else villain).act(game, p))
        total += game.chip_result()[0]
    return total / 2 / hands * 100


@pytest.mark.slow
def test_heuristic_beats_random_and_calling_station():
    assert _bb_per_100(HeuristicPolicy(0), RandomPolicy(1)) > 20
    assert _bb_per_100(HeuristicPolicy(0), CallingStation()) > 0


def test_last_actions_are_tagged_by_street():
    from card_games_rl.games import make_game
    from card_games_rl.games.poker.rules import CALL

    game = make_game("poker", seed=3, button=0)
    game.step(CALL)   # the button completes the small blind
    game.step(CALL)   # the big blind checks: on to the flop
    actions = game.view(None)["last_actions"]
    assert actions[0]["text"] == "calls 1" and actions[1]["text"] == "checks"
    assert not actions[0]["current"] and not actions[1]["current"] and actions[0]["street"] == "preflop"
    game.step(CALL)   # big blind checks on the flop
    assert game.view(None)["last_actions"][1] == {"text": "checks", "street": "flop", "current": True}
