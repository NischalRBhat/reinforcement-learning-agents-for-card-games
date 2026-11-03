import random

import pytest

from card_games_rl.core.cards import cards
from card_games_rl.games import make_game
from card_games_rl.games.blackjack import HeuristicPolicy, OptimalPolicy
from card_games_rl.games.blackjack.optimal import action_values
from card_games_rl.games.blackjack.rules import (DOUBLE, HIT, INSURE, NO_INSURE, SPLIT, STAND, SURRENDER,
                                                 card_points)


def rig(player: str, dealer: str, draws: str = ""):
    """Deal a fixed hand; ``draws`` are the next cards off the shoe, in order."""
    game = make_game("blackjack", seed=0)
    game.hands = [cards(player)]
    game.dealer = cards(dealer)
    game.shoe = cards("2C") * 50 + list(reversed(cards(draws)))
    game.wagers, game.active, game.net = [1.0], 0, 0.0
    game.split_done = game.split_aces = game.surrendered = False
    game.insurance, game.is_terminal = 0.0, False
    game.insurance_phase = card_points(game.dealer[0]) == 1
    if not game.insurance_phase:
        game._peek()
    return game


def test_player_blackjack_pays_3_to_2():
    game = rig("AS KD", "9C 7H")
    assert game.done and game.score(0) == 1.5


def test_dealer_blackjack_with_ten_up_ends_hand():
    game = rig("9S 8D", "KC AH")
    assert game.done and game.score(0) == -1.0


def test_insurance_offered_with_ace_and_pays_when_dealer_has_blackjack():
    game = rig("9S 8D", "AC KH")
    assert game.legal_actions() == [INSURE, NO_INSURE]
    game.step(INSURE)
    assert game.done and game.score(0) == pytest.approx(0.0)  # -1 hand +1 insurance


def test_insurance_lost_when_no_dealer_blackjack():
    game = rig("10S 8D", "AC 7H")
    game.step(INSURE)
    assert not game.done
    game.step(STAND)  # 18 vs soft 18: push, insurance -0.5
    assert game.score(0) == pytest.approx(-0.5)


def test_surrender_loses_half_and_is_not_double_counted():
    game = rig("10S 6D", "9C 9H")
    game.step(SURRENDER)
    assert game.done and game.score(0) == -0.5


def test_dealer_stands_on_soft_17():
    game = rig("10S 8D", "AC 6H", draws="5C")
    game.step(NO_INSURE)
    game.step(STAND)
    assert len(game.dealer) == 2 and game.score(0) == 1.0


def test_double_takes_exactly_one_card_and_doubles_wager():
    game = rig("6S 5D", "9C 7H", draws="KC 10D")  # player gets K -> 21; dealer 16 draws 10 -> bust
    game.step(DOUBLE)
    assert game.done and len(game.hands[0]) == 3 and game.score(0) == 2.0


def test_split_aces_get_one_card_each_and_21_is_not_blackjack():
    game = rig("AS AD", "9C 8H", draws="KC 5D")
    game.step(SPLIT)
    assert game.done
    assert [len(h) for h in game.hands] == [2, 2]
    assert game.score(0) == pytest.approx(1.0 - 1.0)  # 21 beats 17 (+1), 16 loses (-1)


def test_double_after_split_allowed_and_no_resplit_or_surrender():
    game = rig("8S 8D", "9C 8H", draws="3C 8C")
    game.step(SPLIT)
    legal = game.legal_actions()
    assert DOUBLE in legal and SPLIT not in legal and SURRENDER not in legal


def test_bust_loses_even_if_dealer_busts():
    game = rig("10S 6D", "10C 6H", draws="9C")
    game.step(HIT)
    assert game.done and game.score(0) == -1.0 and len(game.dealer) == 2


@pytest.mark.parametrize("player,dealer,expected", [
    ("10S 6D", "10C 7H", SURRENDER),  # hard 16 v 10: late surrender
    ("6S 5D", "6C 7H", DOUBLE),       # 11 v 6
    ("10S 2D", "4C 7H", STAND),       # 12 v 4
    ("10S 2D", "2C 7H", HIT),         # 12 v 2
    ("AS 7D", "9C 7H", HIT),          # soft 18 v 9
    ("AS 7D", "2C 7H", STAND),        # soft 18 v 2 (S17)
    ("8S 8D", "9C 7H", SPLIT),        # 8s v 9
    ("AS AD", "10C 7H", SPLIT),       # aces v 10
    ("10S 10D", "6C 7H", STAND),      # never split tens
])
def test_optimal_policy_matches_basic_strategy(player, dealer, expected):
    game = rig(player, dealer)
    assert OptimalPolicy().act(game, 0) == expected


def _play(policy, hands=20000, seed=123):
    game, total = make_game("blackjack"), 0.0
    for i in range(hands):
        game.reset(seed + i)
        while not game.done:
            game.step(policy.act(game, 0))
        total += game.score(0)
    return total / hands


@pytest.mark.slow
def test_optimal_ev_is_close_to_theory_and_beats_dealer_mimic():
    optimal = _play(OptimalPolicy())
    mimic = _play(HeuristicPolicy())
    assert -0.03 < optimal < 0.015       # theory ≈ -0.5%
    assert optimal > mimic + 0.02         # mimic-the-dealer ≈ -5.5%


def test_action_values_cover_legal_actions():
    game = rig("8S 8D", "9C 7H")
    assert set(action_values(game)) == set(game.legal_actions())
