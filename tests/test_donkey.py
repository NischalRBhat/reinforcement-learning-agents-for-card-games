"""Donkey rules: AS lead, follow suit, discard vs pick-up, going out, the Donkey, stalemates."""
import numpy as np
import pytest

from card_games_rl.core import RandomPolicy
from card_games_rl.core.cards import cards, parse
from card_games_rl.games import make_game
from card_games_rl.games.donkey import HeuristicPolicy
from card_games_rl.games.donkey.rules import ACE_SPADES, MAX_TRICKS


def rig(hands, leader=0, first=False):
    game = make_game("donkey", seed=0)
    game.hands = [sorted(cards(h)) for h in hands]
    game.leader = leader
    game.trick = []
    game.trick_players = game._order_from(leader)
    game.discarded = []
    game.tricks_played = 0 if first else 1
    game.invalidate()
    return game


def test_ace_of_spades_holder_leads_it_first():
    game = make_game("donkey", seed=5)
    holder = next(p for p in range(4) if ACE_SPADES in game.hands[p])
    assert game.current_player == holder and game.legal_actions() == [ACE_SPADES]


def test_players_must_follow_suit_and_may_cut_when_void():
    game = rig(["5S 9H", "7S 2D", "3D 4D", "8S KC"])
    game.step(parse("5S"))
    assert game.legal_actions() == cards("7S")
    game.step(parse("7S"))
    assert set(game.legal_actions()) == set(cards("3D 4D"))  # no spades: any card


def test_completed_trick_is_discarded_and_highest_card_leads():
    game = rig(["5S 9H", "7S 2D", "3S 4D", "QS KC"])
    for card in ("5S", "7S", "3S", "QS"):
        game.step(parse(card))
    assert sorted(game.discarded) == sorted(cards("5S 7S 3S QS"))
    assert game.current_player == 3 and game.last_trick["type"] == "discarded"


def test_cut_stops_the_trick_and_highest_led_card_picks_everything_up():
    game = rig(["5S 9H", "JS 2D", "3D 4D", "QS KC"])
    game.step(parse("5S"))
    game.step(parse("JS"))
    game.step(parse("4D"))  # P3 cuts; P4 never plays
    assert sorted(game.hands[1]) == sorted(cards("2D 5S JS 4D"))
    assert game.current_player == 1 and game.void[2][3] and not game.trick
    assert game.known[1] == set(cards("5S JS 4D"))  # everyone saw what P2 picked up


def test_finished_trick_keeps_every_play_for_the_table():
    """The last card of a trick (the cut, or the last follower) must still be shown after it ends."""
    game = rig(["5S 9H", "JS 2D", "3D 4D", "QS KC"])
    for card in ("5S", "JS", "4D"):
        game.step(parse(card))
    last = game.view(0)["last_trick"]
    assert [(p["player"], p["card"]["label"]) for p in last["plays"]] == [(0, "5S"), (1, "JS"), (2, "4D")]
    assert last["type"] == "picked" and last["cutter"] == 2 and last["picker"] == 1 and game.view(0)["trick"] == []

    game = rig(["5S 9H", "7S 2D", "3S 4D", "QS KC"])
    for card in ("5S", "7S", "3S", "QS"):
        game.step(parse(card))
    assert [p["player"] for p in game.view(None)["last_trick"]["plays"]] == [0, 1, 2, 3]


def test_picking_up_a_suit_clears_a_known_void():
    game = rig(["5S 9H", "JS 2D", "3D 4D", "QS KC"])
    game.void[1][2] = True  # P2 was known void in hearts
    game.step(parse("9H"))
    game.step(parse("2D"))  # P2 cuts: P1 (9H) picks up 9H and 2D
    assert not game.void[0][2] and game.void[1][2]


def test_going_out_and_the_lead_skips_players_without_cards():
    game = rig(["5S", "2S 2D", "3S 4D", "KS KC"])
    for card in ("5S", "2S", "3S", "KS"):
        game.step(parse(card))
    assert game.out_order == [0] and game.current_player == 3
    game.step(parse("KC"))  # P4 leads clubs; P2 and P3 cut... P2 first
    assert game.current_player == 1


def test_last_player_with_cards_is_the_donkey():
    game = rig(["5S", "2S", "3S", "KS QC"])
    for card in ("5S", "2S", "3S", "KS"):
        game.step(parse(card))
    assert game.is_terminal and game.donkey == 3 and game.losers == [3]
    assert game.returns() == pytest.approx([1 / 3, 1 / 3, 1 / 3, -1.0])
    assert game.winners() == [0, 1, 2]


def test_everyone_out_on_the_last_discard_is_a_draw():
    game = rig(["5S", "2S", "3S", "KS"])
    for card in ("5S", "2S", "3S", "KS"):
        game.step(parse(card))
    assert game.is_terminal and game.donkey is None and game.returns() == [0, 0, 0, 0]


def test_repeated_position_stops_the_game_and_most_cards_loses():
    game = rig(["5S 9H", "JS 2D", "3D 4D", "QS KC"])
    game.step(parse("5S"))
    game.step(parse("JS"))
    after = [sorted(game.hands[0]), sorted(game.hands[1] + cards("5S JS 4D")), cards("3D"), sorted(game.hands[3])]
    game.positions[(tuple(tuple(h) for h in after), 1)] = 2  # seen twice already
    game.step(parse("4D"))
    assert game.stopped and game.is_terminal and game.donkey == 1  # P2 holds the most cards
    assert abs(sum(game.returns())) < 1e-9


def test_trick_cap_is_a_forced_stop():
    game = rig(["5S 9H", "JS 2D", "3D 4D", "QS KC"])
    game.tricks_played = MAX_TRICKS - 1
    for card in ("5S", "JS", "4D"):
        game.step(parse(card))
    assert game.stopped and game.is_terminal


def test_observation_hides_hands_but_shows_public_knowledge():
    game = make_game("donkey", seed=2)
    before = game.observation(0)
    game.hands[1], game.hands[2] = game.hands[2], game.hands[1]
    assert np.array_equal(before, game.observation(0))
    game.known[1] = {game.hands[1][0]}
    assert not np.array_equal(before, game.observation(0))


def test_shedding_raises_the_potential_and_picking_up_lowers_it():
    game = rig(["5S 9H", "JS 2D", "3D 4D", "QS KC"])
    start = [game.potential(p) for p in range(4)]
    for card in ("5S", "JS", "4D"):
        game.step(parse(card))
    assert game.potential(1) < start[1] and game.potential(2) > start[2]


def _donkey_rate(policies, seat, games=300):
    rate = 0.0
    for seed in range(games):
        game = make_game("donkey", seed=seed)
        while not game.done:
            game.step(policies[game.current_player].act(game, game.current_player))
        rate += 1 / len(game.losers) if seat in game.losers else 0.0
    return rate / games


def test_heuristic_is_rarely_the_donkey_against_random_players():
    assert _donkey_rate([HeuristicPolicy(0), *[RandomPolicy(s) for s in range(3)]], seat=0) < 0.12
