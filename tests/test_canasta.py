"""Canasta rules: dealing, melds, the opening minimum, the pile, going out and scoring."""
from __future__ import annotations

import random

import numpy as np
import pytest

from card_games_rl.core.cards import make_card
from card_games_rl.games import heuristic_policy, make_game
from card_games_rl.games.canasta.rules import (BLACK3, DISCARD_BASE, DRAW, JOKER, RED3, TAKE_ADD, TAKE_MIXED,
                                               TAKE_PAIR, TWO, CanastaGame, decode_meld, kind_of, meld_action,
                                               minimum_for)


class Cards:
    """Hands out distinct card ids by kind (both decks, then jokers)."""

    def __init__(self):
        self.used: set[int] = set()

    def take(self, kind: int, n: int = 1) -> list[int]:
        out = []
        for card in range(108):
            if card not in self.used and kind_of(card) == kind:
                out.append(card)
                self.used.add(card)
                if len(out) == n:
                    return out
        raise ValueError(f"not enough cards of kind {kind}")


def rig(hands: dict[int, list[int]], pile: list[int] | None = None, turn: int = 0, phase: str = "play",
        melded: tuple[bool, bool] = (True, True), team_scores=(0, 0)) -> CanastaGame:
    game = make_game("canasta", seed=0, team_scores=list(team_scores))
    used = {c for h in hands.values() for c in h} | set(pile or [])
    game.hands = [list(hands.get(p, [])) for p in range(4)]
    game.red_threes = [[] for _ in range(4)]
    game.melds = [{}, {}]
    game.pile = list(pile or [])
    game.pile_frozen = any(kind_of(c) in (JOKER, TWO, RED3) for c in game.pile)
    game.stock = [c for c in range(108) if c not in used]
    game.team_melded = list(melded)
    game.turn, game.phase = turn, phase
    game.turn_value, game.pile_pending = 0, []
    game.invalidate()
    return game


def test_deal_gives_eleven_cards_and_lays_out_red_threes():
    for seed in range(20):
        game = make_game("canasta", seed=seed)
        assert all(len(h) == 11 for h in game.hands)
        assert not any(kind_of(c) == RED3 for h in game.hands for c in h)
        assert sorted(game.all_cards()) == list(range(108))
        if game.pile_frozen:
            assert any(kind_of(c) in (JOKER, TWO, RED3) for c in game.pile)


def test_opening_minimum_follows_the_match_score():
    assert [minimum_for(s) for s in (-10, 0, 1495, 1500, 2995, 3000)] == [15, 50, 50, 90, 90, 120]


def test_meld_needs_two_naturals_three_cards_and_at_most_three_wilds():
    c = Cards()
    hand = c.take(9, 3) + c.take(JOKER, 2) + c.take(TWO, 2) + c.take(5, 1) + c.take(6, 2)
    game = rig({0: hand})
    legal = set(game.legal_actions())
    assert meld_action(9, 3) in legal
    assert meld_action(9, 2, 1, 0) in legal                 # two naturals + a wild
    assert meld_action(9, 2, 2, 1) in legal                 # three wild cards is the limit
    assert meld_action(9, 1, 1, 1) not in legal             # only one natural
    assert meld_action(9, 2) not in legal                   # only two cards
    assert meld_action(5, 1, 1, 1) not in legal
    game.melds[0][9] = {"nat": c.take(9, 2), "jok": c.take(JOKER, 1), "two": c.take(TWO, 1)}
    game.invalidate()
    legal = set(game.legal_actions())
    assert meld_action(9, 0, 1, 0) in legal                 # a third wild card may be added
    assert meld_action(9, 0, 1, 1) not in legal             # a fourth may not


def test_opening_meld_must_be_able_to_reach_the_minimum():
    c = Cards()
    # 4s are worth 5 each: 4-4-4 is 15, far below 50 with nothing else to meld.
    game = rig({0: c.take(4, 3) + c.take(5, 1) + c.take(7, 1)}, melded=(False, True))
    assert meld_action(4, 3) not in game.legal_actions()
    # With two aces and a joker in hand the minimum is reachable, so starting is fine.
    c2 = Cards()
    hand = c2.take(4, 3) + c2.take(14, 2) + c2.take(JOKER, 1) + c2.take(7, 1) + c2.take(8, 2)
    game = rig({0: hand}, melded=(False, True))
    assert meld_action(4, 3) in game.legal_actions()
    game.step(meld_action(4, 3))
    assert not any(a >= DISCARD_BASE for a in game.legal_actions())   # must finish the opening first
    game.step(meld_action(14, 2, 1, 0))
    assert game.team_melded[0] and any(a >= DISCARD_BASE for a in game.legal_actions())


def test_frozen_pile_needs_a_natural_pair_and_unfrozen_allows_more():
    c = Cards()
    pile = c.take(8, 1) + c.take(TWO, 1) + c.take(10, 1)        # a wild card inside: frozen
    hand = c.take(10, 1) + c.take(JOKER, 1) + c.take(6, 3)
    game = rig({0: hand}, pile=pile, phase="draw")
    assert game.legal_actions() == [DRAW]                      # one natural + wild is not enough when frozen
    c2 = Cards()
    pile = c2.take(8, 1) + c2.take(10, 1)
    game = rig({0: c2.take(10, 1) + c2.take(JOKER, 1) + c2.take(6, 3)}, pile=pile, phase="draw")
    game.melds[0][10] = {"nat": c2.take(10, 3), "jok": [], "two": []}
    game.invalidate()
    assert set(game.legal_actions()) == {DRAW, TAKE_MIXED, TAKE_ADD}
    game.step(TAKE_ADD)
    assert len(game.melds[0][10]["nat"]) == 4 and game.phase == "play" and not game.pile
    assert sum(kind_of(c) == 8 for c in game.hands[0]) == 1    # the rest of the pile is in hand


def test_black_three_or_wild_on_top_cannot_be_taken():
    c = Cards()
    for top in (BLACK3, JOKER):
        game = rig({0: c.take(9, 2) + c.take(6, 2)}, pile=c.take(9, 1) + c.take(top, 1), phase="draw")
        assert game.legal_actions() == [DRAW]


def test_taking_a_frozen_pile_before_melding_needs_the_minimum_from_hand():
    c = Cards()
    top = c.take(14, 1)                                        # an ace on top: A-A-A is 60 >= 50
    game = rig({0: c.take(14, 2) + c.take(6, 2)}, pile=c.take(5, 3) + top, phase="draw", melded=(False, False))
    assert TAKE_PAIR in game.legal_actions()
    game.step(TAKE_PAIR)
    assert game.team_melded[0] and len(game.hands[0]) == 5    # two 6s + three 5s from the pile
    c2 = Cards()
    top = c2.take(5, 1)                                        # 5-5-5 is only 15 and nothing else melds
    game = rig({0: c2.take(5, 2) + c2.take(9, 1) + c2.take(12, 1)}, pile=c2.take(4, 2) + top, phase="draw",
               melded=(False, False))
    assert TAKE_PAIR not in game.legal_actions()


def test_discarding_a_wild_card_freezes_the_pile():
    c = Cards()
    game = rig({0: c.take(JOKER, 1) + c.take(7, 3)})
    game.step(DISCARD_BASE + JOKER)
    assert game.pile_frozen and game.turn == 1


def test_going_out_needs_a_canasta_and_scores_the_bonus():
    c = Cards()
    queens = c.take(12, 7)
    game = rig({0: c.take(11, 3) + c.take(13, 1), 1: c.take(5, 2), 2: c.take(6, 2), 3: c.take(7, 2)})
    assert meld_action(11, 3) not in game.legal_actions()      # no canasta: may not go down to one card
    game.melds[0][12] = {"nat": queens, "jok": [], "two": []}  # a natural canasta of queens
    game.prior_meld = True                                     # not a concealed hand
    game.invalidate()
    game.step(meld_action(11, 3))
    game.step(DISCARD_BASE + 13)
    assert game.done and game.went_out == 0 and not game.concealed
    melded = 7 * 10 + 3 * 10
    assert game.hand_scores[0] == melded + 500 + 100 - 2 * 5    # the partner still holds two 6s
    assert game.hand_scores[1] == -(2 * 5 + 2 * 5)
    assert abs(sum(game.returns())) < 1e-9 and game.winners() == [0, 2]


def test_black_threes_meld_only_when_going_out():
    c = Cards()
    game = rig({0: c.take(BLACK3, 3) + c.take(6, 3)})
    assert meld_action(BLACK3, 3) not in game.legal_actions()   # no canasta
    game.melds[0][10] = {"nat": [x for x in range(108) if kind_of(x) == 10][:7], "jok": [], "two": []}
    game.invalidate()
    assert meld_action(BLACK3, 3) not in game.legal_actions()   # three 6s would remain
    game.step(meld_action(6, 3))
    assert meld_action(BLACK3, 3) in game.legal_actions()
    game.step(meld_action(BLACK3, 3))
    assert game.done and game.went_out == 0


def test_hand_ends_when_the_stock_runs_out_and_the_pile_cannot_be_taken():
    c = Cards()
    game = rig({0: c.take(9, 2), 1: c.take(5, 3), 2: c.take(6, 2), 3: c.take(7, 2)})
    game.stock = []
    game.step(DISCARD_BASE + 9)                                  # P2 cannot draw or take a 9
    assert game.done and game.went_out is None


def test_system_partnership_beats_random_players():
    totals = 0
    for seed in range(30):
        game = make_game("canasta", seed=seed)
        bots = {p: heuristic_policy("canasta", p) for p in (0, 2)}
        rng = random.Random(seed)
        while not game.done:
            p = game.current_player
            game.step(bots[p].act(game, p) if p in bots else rng.choice(game.legal_actions()))
        totals += game.hand_scores[0] - game.hand_scores[1]
    assert totals / 30 > 300


def test_observation_has_counting_features_without_hidden_cards():
    game = make_game("canasta", seed=3)
    bot = heuristic_policy("canasta", 0)
    for _ in range(40):
        game.step(bot.act(game, game.current_player))
    player = game.current_player
    before = game.observation(player)
    others = [q for q in range(4) if q != player]
    hidden = [c for q in others for c in game.hands[q] if not game.known[q][kind_of(c)]]
    rng = random.Random(1)
    rng.shuffle(hidden)
    for q in others:  # re-deal unknown cards among the other hands (sizes kept, known kinds kept)
        keep = [c for c in game.hands[q] if game.known[q][kind_of(c)]]
        take = len(game.hands[q]) - len(keep)
        game.hands[q], hidden = keep + hidden[:take], hidden[take:]
    game.invalidate()
    assert (game.observation(player) == before).all()


def test_planner_covers_every_turn_and_reproduces_the_system_bot():
    from card_games_rl.encoders import get_encoder
    from card_games_rl.games.canasta import plan

    encoder = get_encoder("canasta")
    assert encoder.planned and encoder.num_actions == plan.NUM_PLAN_ACTIONS
    bot = heuristic_policy("canasta", 0)
    for seed in range(15):
        rng = random.Random(seed)
        game = make_game("canasta", seed=seed)
        while not game.done:
            player = game.current_player
            options = plan.legal(game)
            assert options and list(np.flatnonzero(plan.mask(game))) == sorted(options)
            assert all(encoder.to_game(game, a) in game.legal_actions() for a in options)
            teacher = bot.act(game, player)
            assert encoder.to_game(game, encoder.from_game(game, teacher)) == teacher   # copying is exact
            features = encoder(game, player)
            assert features.shape == (encoder.size,) and features.min() >= 0 and features.max() <= 1
            game.step(encoder.to_game(game, rng.choice(options)) if player % 2 == 0 else teacher)


def test_planner_lets_the_agent_hold_back_melds():
    from card_games_rl.games.canasta import plan

    c = Cards()
    game = rig({0: c.take(9, 3) + c.take(JOKER, 1) + c.take(6, 2) + c.take(7, 1)})
    options = plan.legal(game)
    assert plan.MELD in options and plan.MELD_WILD in options
    assert plan.DISCARD + 7 in options                       # discard instead of melding the 9s
    assert decode_meld(plan.to_game(game, plan.MELD)) == (9, 3, 0, 0)
    assert decode_meld(plan.to_game(game, plan.MELD_WILD))[2] == 1


@pytest.mark.parametrize("seed", range(3))
def test_views_hide_other_hands(seed):
    game = make_game("canasta", seed=seed)
    view = game.view(0)
    assert isinstance(view["hands"][0], list) and all(isinstance(view["hands"][q], int) for q in (1, 2, 3))
