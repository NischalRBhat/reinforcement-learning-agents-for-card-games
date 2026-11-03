"""Bluff rules (Indian variant): declare, add / pass / call, discard on all-pass."""
import random

import numpy as np
import pytest

from card_games_rl.core import IllegalAction, RandomPolicy
from card_games_rl.core.cards import cards, rank_of
from card_games_rl.games import make_game
from card_games_rl.games.bluff import HeuristicPolicy
from card_games_rl.games.bluff.rules import CALL, DECLARE, MAX_ACTIONS, PASS, PLAY_COMBOS, rank_index


def rig(hands, starter=0):
    game = make_game("bluff", seed=0)
    game.hands = [sorted(cards(h)) for h in hands]
    game.own_discarded = [[] for _ in range(4)]
    game.discarded = 0
    game.known = [[set() for _ in range(4)] for _ in range(4)]
    game._new_round(starter)
    game.invalidate()
    return game


def combo(t, b):
    return PLAY_COMBOS.index((t, b))


def declare(rank):
    return DECLARE + rank_index(rank)


def test_round_starts_with_a_free_rank_choice_then_a_forced_play():
    game = rig(["7S 7H 2C", "4C 5C", "6C 8C", "9C 10C"])
    assert game.legal_actions() == [declare(r) for r in range(2, 15)]
    game.step(declare(13))  # Kings, holding none: bluffing the rank is allowed
    legal = game.legal_actions()
    assert CALL not in legal and PASS not in legal
    assert combo(0, 1) in legal and combo(1, 0) not in legal


def test_legal_plays_respect_true_and_bluff_counts():
    game = rig(["AS AH 5C", "4C", "6C", "8C"])
    game.step(declare(14))
    legal = {PLAY_COMBOS[a] for a in game.legal_actions()}
    assert (2, 1) in legal and (1, 1) in legal and (3, 0) not in legal and (0, 2) not in legal


def test_call_is_only_possible_right_after_a_claim():
    game = rig(["7S 7H 2C", "7C 5C", "6C 8C", "9C 10C"])
    game.step(declare(7))
    game.step(combo(2, 0))
    assert {CALL, PASS, combo(1, 0)} <= set(game.legal_actions())  # P2: call, pass or add a 7
    game.step(PASS)
    assert CALL not in game.legal_actions() and PASS in game.legal_actions()  # P3 can't call P1 now


def test_true_claim_called_caller_takes_pile_claimant_starts():
    game = rig(["7S 7H 2C", "4C 5C", "6C 8C", "9C 10C"])
    game.step(declare(7))
    game.step(combo(2, 0))
    game.step(CALL)
    assert sorted(game.hands[1]) == sorted(cards("4C 5C 7S 7H"))
    assert game.reveal["truthful"] and game.reveal["loser"] == 1
    assert game.phase == "declare" and game.current_player == 0 and not game.pile


def test_caught_lie_claimant_takes_pile_caller_starts():
    game = rig(["2C 3C 4C", "4D 5D", "6C 8C", "9C 10C"])
    game.step(declare(9))
    game.step(combo(0, 2))
    game.step(CALL)
    assert len(game.hands[0]) == 3 and not game.reveal["truthful"]
    assert game.phase == "declare" and game.current_player == 1


def test_everyone_passing_discards_the_pile_and_next_starter_rotates():
    game = rig(["7S 2C 3C", "4C 5C", "6C 8C", "9C 10C"], starter=0)
    game.step(declare(7))
    game.step(combo(1, 0))
    for _ in range(3):
        game.step(PASS)
    assert game.discarded == 1 and not game.pile and game.own_discarded[0] == cards("7S")
    assert game.phase == "declare" and game.current_player == 1  # the player after the last starter


def test_passing_then_adding_keeps_the_round_going():
    game = rig(["7S 2C 3C", "7C 5C", "7D 8C", "9C 10C"])
    game.step(declare(7))
    game.step(combo(1, 0))   # P1: one 7
    game.step(PASS)          # P2 passes
    game.step(combo(1, 0))   # P3 adds a 7: passes reset, P4 may now call P3
    assert game.passes == 0 and CALL in game.legal_actions() and game.current_player == 3


def test_last_card_claim_wins_when_the_next_player_does_not_call():
    game = rig(["7S", "4C 5C", "6C 8C", "9C 10C"])
    game.step(declare(7))
    game.step(combo(1, 0))
    assert game.legal_actions() == [CALL, PASS]
    game.step(PASS)
    assert game.is_terminal and game.winner == 0


def test_last_card_claim_called_true_still_wins_and_lie_does_not():
    honest = rig(["7S", "4C 5C", "6C 8C", "9C 10C"])
    honest.step(declare(7))
    honest.step(combo(1, 0))
    honest.step(CALL)
    assert honest.winner == 0 and sorted(honest.all_cards()) == sorted(cards("7S 4C 5C 6C 8C 9C 10C"))
    liar = rig(["2S", "4C 5C", "6C 8C", "9C 10C"])
    liar.step(declare(7))
    liar.step(combo(0, 1))
    liar.step(CALL)
    assert not liar.done and liar.hands[0] == cards("2S")


def test_exact_card_play_for_humans():
    game = rig(["7S 7H 2C", "4C 5C", "6C 8C", "9C 10C"])
    with pytest.raises(IllegalAction):
        game.step_cards(cards("7S"))  # starting a round needs a rank
    game.step_cards(cards("7S 2C"), rank=7)
    assert game.round_rank == 7 and game.hands[0] == cards("7H") and game.history[-1] == (0, combo(1, 1))
    with pytest.raises(IllegalAction):
        game.step_cards(cards("4C"), rank=5)  # later plays must claim the round's rank
    game.step_cards(cards("4C"))
    assert game.hands[1] == cards("5C")


def test_human_cannot_add_cards_when_the_claimant_is_out():
    game = rig(["7S", "4C 5C", "6C 8C", "9C 10C"])
    game.step(declare(7))
    game.step(combo(1, 0))
    with pytest.raises(IllegalAction):
        game.step_cards(cards("4C"))


def test_bluff_cards_are_the_most_expendable():
    game = rig(["9S 9H 9D 3C KH", "4C", "6C", "8C"])
    game.step(declare(5))
    assert set(game.cards_for(0, combo(0, 2))) == set(cards("3C KH"))  # singletons, not the 9s


def test_certain_lie_signal_is_never_wrong():
    policies = [HeuristicPolicy(seed=i) for i in range(4)]
    for seed in range(60):
        game = make_game("bluff", seed=seed)
        while not game.done:
            player = game.current_player
            if game.can_call and game.certain_lie(player):
                assert any(rank_of(c) != game.round_rank for c in game.last_claim[1])
            game.step(policies[player].act(game, player))


def test_observation_hides_opponent_hands():
    game = make_game("bluff", seed=3)
    before = game.observation(0)
    game.hands[1], game.hands[2] = game.hands[2], game.hands[1]  # same sizes, different cards
    assert np.array_equal(before, game.observation(0))


def test_truncation_is_zero_sum():
    game = make_game("bluff", seed=1)
    game.actions_taken = MAX_ACTIONS - 1
    game.step(game.legal_actions()[0])
    assert game.is_truncated and abs(sum(game.returns())) < 1e-9


def test_cards_are_conserved_and_games_end():
    policies = [HeuristicPolicy(seed=1), RandomPolicy(2), HeuristicPolicy(seed=3), RandomPolicy(4)]
    for seed in range(40):
        game = make_game("bluff", seed=seed)
        while not game.done:
            game.step(policies[game.current_player].act(game, game.current_player))
            assert sorted(game.all_cards()) == list(range(52))
        assert game.is_terminal


def _win_rate(policies, seat, games=200):
    wins = 0
    for seed in range(games):
        game = make_game("bluff", seed=seed)
        while not game.done:
            game.step(policies[game.current_player].act(game, game.current_player))
        wins += game.winner == seat
    return wins / games


def test_heuristic_beats_random():
    random_players = [RandomPolicy(s) for s in range(3)]
    assert _win_rate([HeuristicPolicy(seed=0), *random_players], seat=0) > 0.30  # fair share 25%


class NeverCaller(HeuristicPolicy):
    """Dumps as many cards as possible and never calls: the exploit of the old variant."""

    def act(self, game, player):
        legal = [a for a in game.legal_actions() if a != CALL]
        plays = [a for a in legal if a < CALL]
        return max(plays, key=lambda a: sum(PLAY_COMBOS[a])) if plays else super().act(game, player)


def test_dumping_cards_without_calling_does_not_beat_the_bots():
    rate = _win_rate([NeverCaller(seed=0), *[HeuristicPolicy(seed=s) for s in (1, 2, 3)]], seat=0)
    assert rate < 0.30
