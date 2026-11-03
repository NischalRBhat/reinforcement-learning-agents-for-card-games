"""Heads-up no-limit Texas Hold'em engine, one hand per episode (docs/ENV_DESIGN.md §4)."""
from __future__ import annotations

from typing import Any

import numpy as np

from card_games_rl.core import Game
from card_games_rl.core.cards import card_json, new_deck, one_hot, one_hot_index, rank_of, suit_of

from .evaluator import CATEGORY_NAMES, equity, evaluate

FOLD, CALL, RAISE_HALF, RAISE_POT, RAISE_2POT, ALL_IN = range(6)
RAISE_FRACTIONS = {RAISE_HALF: 0.5, RAISE_POT: 1.0, RAISE_2POT: 2.0}
STACK, SMALL_BLIND, BIG_BLIND = 200, 1, 2
STREETS = ["preflop", "flop", "turn", "river"]


class PokerGame(Game):
    game_id = "poker"
    name = "Texas Hold'em"
    num_players = 2
    num_actions = 6
    obs_size = 146
    OBS_VERSION = 1
    zero_sum = True

    def __init__(self, seed: int | None = None, allin_ev: bool = False, button: int | None = None):
        super().__init__(seed)
        self.allin_ev = allin_ev  # training-only variance reduction (see spec)
        self.fixed_button = button

    def _deal(self) -> None:
        deck = new_deck(self.rng)
        self.hole = [[deck.pop(), deck.pop()], [deck.pop(), deck.pop()]]
        self.board_cards = [deck.pop() for _ in range(5)]  # revealed street by street
        self.deck = deck
        self.board_visible = 0
        self.button = self.fixed_button if self.fixed_button is not None else self.rng.randrange(2)
        self.stacks = [STACK, STACK]
        self.committed = [0, 0]
        self.street_bet = [0, 0]
        self.street = 0
        self.raises = [[0, 0] for _ in range(4)]
        self.has_acted = [[False, False] for _ in range(4)]
        self.acted = [False, False]
        self.can_raise = [True, True]
        self.last_raise = BIG_BLIND
        self.folded: int | None = None
        self.showdown = False
        self.ev_net: list[float] | None = None
        self.log: list[str] = []
        self._post(self.button, SMALL_BLIND)
        self._post(1 - self.button, BIG_BLIND)
        self.to_act = self.button  # small blind acts first preflop

    def clone(self) -> "PokerGame":
        """Fast copy for tree search (CFR). Cards are shared: they never change within a hand."""
        other = object.__new__(PokerGame)
        other.__dict__.update(self.__dict__)
        other.stacks, other.committed, other.street_bet = list(self.stacks), list(self.committed), list(self.street_bet)
        other.raises = [list(r) for r in self.raises]
        other.has_acted = [list(h) for h in self.has_acted]
        other.acted, other.can_raise = list(self.acted), list(self.can_raise)
        other.log, other.history = [], []
        other._legal = None
        return other

    # ----- helpers ---------------------------------------------------------
    @property
    def board(self) -> list[int]:
        return self.board_cards[:self.board_visible]

    @property
    def pot(self) -> int:
        return sum(self.committed)

    def _post(self, player: int, amount: int) -> None:
        amount = min(amount, self.stacks[player])
        self.stacks[player] -= amount
        self.committed[player] += amount
        self.street_bet[player] += amount

    def to_call(self, player: int) -> int:
        return max(0, self.street_bet[1 - player] - self.street_bet[player])

    def min_raise_to(self, player: int) -> int:
        return self.street_bet[1 - player] + self.last_raise

    def raise_targets(self, player: int) -> dict[int, int]:
        """Legal raise actions → total street bet they raise to (deduplicated)."""
        opponent = 1 - player
        call = self.to_call(player)
        if not self.can_raise[player] or self.stacks[player] <= call or self.stacks[opponent] == 0:
            return {}
        max_to = self.street_bet[player] + self.stacks[player]
        targets: dict[int, int] = {}
        for action, fraction in RAISE_FRACTIONS.items():
            target = self.street_bet[opponent] + int(round(fraction * (self.pot + call)))
            target = max(target, self.min_raise_to(player))
            if target < max_to and target not in targets.values():
                targets[action] = target
        targets[ALL_IN] = max_to
        return targets

    # ----- contract --------------------------------------------------------
    @property
    def current_player(self) -> int:
        return -1 if self.done else self.to_act

    def _legal_actions(self) -> list[int]:
        if self.done:
            return []
        player = self.to_act
        legal = [FOLD] if self.to_call(player) > 0 else []
        legal.append(CALL)
        legal.extend(sorted(self.raise_targets(player)))
        return legal

    def _apply(self, action: int) -> None:
        player, opponent = self.to_act, 1 - self.to_act
        name = ["P1", "P2"][player]
        if action == FOLD:
            self.folded = player
            self.log.append(f"{name} folds")
            self.is_terminal = True
            return
        self.has_acted[self.street][player] = True
        if action == CALL:
            call = min(self.to_call(player), self.stacks[player])
            self._post(player, call)
            self.log.append(f"{name} {'calls ' + str(call) if call else 'checks'}")
            self.acted[player] = True
        else:
            target = self.raise_targets(player)[action]
            raise_size = target - self.street_bet[opponent]
            self._post(player, target - self.street_bet[player])
            if raise_size >= self.last_raise:
                self.last_raise = raise_size
                self.can_raise[opponent] = True
            else:  # incomplete all-in raise does not reopen the betting
                self.can_raise[opponent] = not self.acted[opponent]
            self.raises[self.street][player] += 1
            self.acted = [False, False]
            self.acted[player] = True
            self.log.append(f"{name} {'all-in' if self.stacks[player] == 0 else 'raises'} to {target}")
        if action == CALL:
            if self.street_bet[0] != self.street_bet[1]:  # called all-in for less
                self._run_out()
                return
            if all(self.acted):
                self._next_street()
                return
        self.to_act = opponent

    def _next_street(self) -> None:
        if 0 in self.stacks:
            self._run_out()
            return
        self.street += 1
        if self.street == 4:
            self._showdown()
            return
        self.board_visible = {1: 3, 2: 4, 3: 5}[self.street]
        self.street_bet = [0, 0]
        self.acted = [False, False]
        self.can_raise = [True, True]
        self.last_raise = BIG_BLIND
        self.to_act = 1 - self.button  # big blind acts first postflop

    def _run_out(self) -> None:
        if self.allin_ev and self.board_visible < 5:
            share = equity(self.hole[0], self.hole[1], self.board, rng=self.rng)
            contested = 2 * min(self.committed)
            ev0 = share * contested - min(self.committed)
            self.ev_net = [ev0 / STACK, -ev0 / STACK]
        self.board_visible = 5
        self._showdown()

    def _showdown(self) -> None:
        self.showdown = True
        self.is_terminal = True

    def chip_result(self) -> list[int]:
        """Net chips won by each player (only meaningful once the hand is over)."""
        if not self.is_terminal:
            return [0, 0]
        if self.folded is not None:
            loser = self.folded
            won = self.committed[loser]
            return [won if p != loser else -won for p in range(2)]
        contested = min(self.committed)
        values = [evaluate(self.hole[p] + self.board_cards) for p in range(2)]
        if values[0] == values[1]:
            return [0, 0]
        winner = 0 if values[0] > values[1] else 1
        return [contested if p == winner else -contested for p in range(2)]

    def score(self, player: int) -> float:
        if not self.is_terminal:
            return 0.0
        if self.ev_net is not None:
            return self.ev_net[player]
        return self.chip_result()[player] / STACK

    def all_cards(self) -> list[int]:
        return self.deck + self.hole[0] + self.hole[1] + self.board_cards

    # ----- observation -----------------------------------------------------
    def made_category(self, player: int) -> int:
        cards = self.hole[player] + self.board
        if len(cards) >= 5:
            return evaluate(cards)[0]
        return int(rank_of(cards[0]) == rank_of(cards[1]))

    def observation(self, player: int) -> np.ndarray:
        opponent = 1 - player
        hole = self.hole[player]
        high, low = sorted((rank_of(c) for c in hole), reverse=True)
        history = []
        for street in range(4):
            for seat in (player, opponent):
                history += [min(self.raises[street][seat], 4) / 4, float(self.has_acted[street][seat])]
        money = [
            self.pot / (2 * STACK), self.stacks[player] / STACK, self.stacks[opponent] / STACK,
            self.to_call(player) / STACK, self.street_bet[player] / STACK, self.street_bet[opponent] / STACK,
            min(self.min_raise_to(player), STACK) / STACK,
        ]
        preflop = [float(high == low), float(suit_of(hole[0]) == suit_of(hole[1])), high / 14, low / 14, (high - low) / 12]
        return np.concatenate([
            one_hot(hole), one_hot(self.board), one_hot_index(self.street, 4),
            np.asarray([float(self.button == player)], np.float32), np.asarray(money, np.float32),
            np.asarray(history, np.float32), one_hot_index(self.made_category(player), 9),
            np.asarray(preflop, np.float32),
        ]).astype(np.float32)

    def action_label(self, action: int) -> str:
        player = self.to_act
        if action == FOLD:
            return "Fold"
        if action == CALL:
            call = min(self.to_call(player), self.stacks[player])
            return f"Call {call}" if call else "Check"
        target = self.raise_targets(player).get(action)
        names = {RAISE_HALF: "½ pot", RAISE_POT: "pot", RAISE_2POT: "2× pot"}
        if action == ALL_IN:
            return f"All-in ({target})" if target else "All-in"
        return f"Raise to {target} ({names[action]})" if target else f"Raise {names[action]}"

    def view(self, viewer: int | None) -> dict[str, Any]:
        reveal = self.showdown and self.folded is None
        legal = self.legal_actions() if viewer is not None and viewer == self.current_player else []
        targets = self.raise_targets(self.to_act) if legal else {}
        return {
            "game": self.game_id,
            "done": self.done,
            "street": STREETS[min(self.street, 3)] if not self.showdown else "showdown",
            "current_player": self.current_player,
            "button": self.button,
            "hole": [[card_json(c) for c in self.hole[p]] if p == viewer or reveal else None for p in range(2)],
            "board": [card_json(c) for c in self.board],
            "pot": self.pot,
            "stacks": list(self.stacks),
            "street_bets": list(self.street_bet),
            "to_call": self.to_call(self.to_act) if not self.done else 0,
            "hand_names": [CATEGORY_NAMES[evaluate(self.hole[p] + self.board_cards)[0]] if reveal else None for p in range(2)],
            "result": self.chip_result() if self.done else None,
            "folded": self.folded,
            "log": self.log[-12:],
            "legal_actions": [{"action": a, "label": self.action_label(a), "amount": targets.get(a)} for a in legal],
        }
