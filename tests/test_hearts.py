import numpy as np
import pytest

from card_games_rl.core import RandomPolicy
from card_games_rl.core.cards import cards, parse
from card_games_rl.games import make_game
from card_games_rl.games.hearts import HeuristicPolicy
from card_games_rl.games.hearts.rules import PASS_LEFT, PASS_NONE, QUEEN_SPADES, TWO_CLUBS


def rig(hands, leader=None, tricks_played=1, hearts_broken=False):
    game = make_game("hearts", seed=0, pass_direction=PASS_NONE)
    game.hands = [sorted(cards(h)) for h in hands]
    game.tricks_played = tricks_played
    game.hearts_broken = hearts_broken
    game.leader = leader if leader is not None else 0
    game.trick = []
    return game


def test_passing_moves_three_cards_in_direction():
    game = make_game("hearts", seed=1, pass_direction=PASS_LEFT)
    before = [list(h) for h in game.hands]
    picks = []
    for seat in range(4):
        assert game.current_player == seat
        chosen = game.legal_actions()[:3]
        picks.append(chosen)
        for card in chosen:
            game.step(card)
    assert not game.passing
    for seat in range(4):
        target = (seat + 1) % 4
        assert all(c in game.hands[target] for c in picks[seat])
        assert all(c not in game.hands[seat] for c in picks[seat])
        assert len(game.hands[seat]) == 13
    assert game.received[1] == picks[0] and before != game.hands


def test_first_lead_is_two_of_clubs_by_its_holder():
    game = make_game("hearts", seed=2, pass_direction=PASS_NONE)
    assert TWO_CLUBS in game.hands[game.current_player]
    assert game.legal_actions() == [TWO_CLUBS]


def test_must_follow_suit():
    game = rig(["5C 9H", "7C KH", "2D", "3D"])
    game.step(parse("5C"))
    assert game.legal_actions() == [parse("7C")]


def test_no_points_on_first_trick_unless_unavoidable():
    game = rig(["2C", "QS 9H 4D", "5H 6H", "3C"], tricks_played=0)
    game.step(TWO_CLUBS)
    assert game.legal_actions() == [parse("4D")]       # void in clubs: no QS, no hearts
    game.step(parse("4D"))
    assert sorted(game.legal_actions()) == sorted(cards("5H 6H"))  # only hearts left: allowed


def test_hearts_cannot_be_led_until_broken():
    game = rig(["5H 9C", "2C", "3C", "4C"])
    assert game.legal_actions() == [parse("9C")]
    only_hearts = rig(["5H 9H", "2C", "3C", "4C"])
    assert sorted(only_hearts.legal_actions()) == sorted(cards("5H 9H"))
    broken = rig(["5H 9C", "2C", "3C", "4C"], hearts_broken=True)
    assert sorted(broken.legal_actions()) == sorted(cards("5H 9C"))


def test_trick_winner_takes_points_and_leads():
    game = rig(["KS 2D", "QS 3D", "5H 4D", "AS 6D"], tricks_played=11)
    for card in ("KS", "QS", "5H", "AS"):
        game.step(parse(card))
    assert game.points == [0, 0, 0, 14] and game.leader == 3 and game.current_player == 3


def test_shooting_the_moon():
    game = rig(["2D", "3D", "4D", "5D"], tricks_played=12)
    game.points = [0, 0, 26, 0]
    for card in ("2D", "3D", "4D", "5D"):
        game.step(parse(card))
    assert game.done and game.hand_points() == [26, 26, 0, 26]
    assert game.score(2) == pytest.approx((78 / 4) / 26)
    assert abs(sum(game.returns())) < 1e-9


def test_observation_hides_opponent_hands():
    a = make_game("hearts", seed=7, pass_direction=PASS_NONE)
    b = make_game("hearts", seed=7, pass_direction=PASS_NONE)
    me = (a.current_player + 1) % 4
    x, y = [p for p in range(4) if p != me][:2]
    card_x, card_y = next(c for c in b.hands[x] if c != TWO_CLUBS), next(c for c in b.hands[y] if c != TWO_CLUBS)
    b.hands[x].remove(card_x); b.hands[y].remove(card_y)
    b.hands[x].append(card_y); b.hands[y].append(card_x)
    np.testing.assert_array_equal(a.observation(me), b.observation(me))


def _average_points(seat_policy, hands=300):
    game, total = make_game("hearts"), 0
    others = RandomPolicy(1)
    for seed in range(hands):
        game.reset(seed)
        while not game.done:
            p = game.current_player
            game.step((seat_policy if p == 0 else others).act(game, p))
        total += game.hand_points()[0]
    return total / hands


@pytest.mark.slow
def test_heuristic_beats_random_players():
    assert _average_points(HeuristicPolicy(0)) < _average_points(RandomPolicy(0)) - 1.5
