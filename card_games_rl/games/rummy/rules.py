"""Basic 10-card rummy for two players (docs/ENV_DESIGN.md §5)."""
from __future__ import annotations

from typing import Any

import numpy as np

from card_games_rl.core import Game
from card_games_rl.core.cards import card_json, label, new_deck, one_hot

from .melds import deadwood, solver_for

DRAW_STOCK, DRAW_DISCARD, DISCARD = 0, 1, 2  # DISCARD + card id
HAND_SIZE = 10
MAX_TURNS = 100


class RummyGame(Game):
    game_id = "rummy"
    name = "Rummy"
    num_players = 2
    num_actions = 2 + 52
    obs_size = 320
    OBS_VERSION = 2  # v2: meld-aware features (per-card discard value, meld membership, take gain)
    zero_sum = True
    shaping_weight = 1.0

    def _deal(self) -> None:
        deck = new_deck(self.rng)
        self.dealer = self.rng.randrange(2)
        self.hands = [sorted(deck.pop() for _ in range(HAND_SIZE)) for _ in range(2)]
        self.discard = [deck.pop()]
        self.stock = deck
        self.turn = 1 - self.dealer
        self.phase = "draw"
        self.drawn_from_discard: int | None = None
        self.restocked = False
        self.turns = [0, 0]
        self.known = [set(), set()]  # known[p] = cards p knows the opponent holds
        self.winner: int | None = None
        self.final_deadwood: list[int] | None = None
        self.last_event = ""

    # ----- contract --------------------------------------------------------
    @property
    def current_player(self) -> int:
        return -1 if self.done else self.turn

    def _legal_actions(self) -> list[int]:
        if self.done:
            return []
        if self.phase == "draw":
            return ([DRAW_STOCK] if self.stock else []) + ([DRAW_DISCARD] if self.discard else [])
        return [DISCARD + c for c in self.hands[self.turn] if c != self.drawn_from_discard]

    def _apply(self, action: int) -> None:
        player, opponent = self.turn, 1 - self.turn
        hand = self.hands[player]
        if action == DRAW_STOCK:
            hand.append(self.stock.pop())
            self.drawn_from_discard = None
            self.phase = "discard"
            self.last_event = f"P{player + 1} drew from the stock"
            return
        if action == DRAW_DISCARD:
            card = self.discard.pop()
            hand.append(card)
            self.drawn_from_discard = card
            self.known[opponent].add(card)
            self.phase = "discard"
            self.last_event = f"P{player + 1} took {label(card)}"
            return
        card = action - DISCARD
        hand.remove(card)
        hand.sort()
        self.discard.append(card)
        self.known[opponent].discard(card)
        self.drawn_from_discard = None
        self.turns[player] += 1
        self.phase = "draw"
        self.last_event = f"P{player + 1} discarded {label(card)}"
        if deadwood(hand) == 0:
            self.winner = player
            self._finish()
            return
        if not self.stock:
            if self.restocked:
                self._finish()
                return
            self.stock = self.discard[:-1][::-1]
            self.discard = self.discard[-1:]
            self.restocked = True
        if min(self.turns) >= MAX_TURNS:
            self._finish()
            return
        self.turn = opponent

    def _finish(self) -> None:
        self.final_deadwood = [deadwood(h) for h in self.hands]
        self.is_terminal = True

    # ----- rewards ---------------------------------------------------------
    def score(self, player: int) -> float:
        if not self.is_terminal:
            return 0.0
        dw = self.final_deadwood
        if self.winner is None:
            return (dw[1 - player] - dw[player]) / 200
        value = 0.5 + 0.5 * min(dw[1 - self.winner], 100) / 100
        return value if player == self.winner else -value

    def winners(self) -> list[int]:
        if not self.done:
            return []
        if self.winner is not None:
            return [self.winner]
        return super().winners()

    def hand_deadwood(self, player: int) -> int:
        """Deadwood of the hand, or of the best 10 cards when holding 11."""
        hand = self.hands[player]
        solver = solver_for(hand)
        if len(hand) > HAND_SIZE:
            forbidden = self.drawn_from_discard if player == self.turn else None
            return solver.best_discard(forbidden)[0]
        return solver.solve()

    def potential(self, player: int) -> float:
        return -0.5 * self.hand_deadwood(player) / 100

    def all_cards(self) -> list[int]:
        return self.hands[0] + self.hands[1] + self.stock + self.discard

    # ----- observation -----------------------------------------------------
    def observation(self, player: int) -> np.ndarray:
        hand = self.hands[player]
        solver = solver_for(hand)
        full = (1 << len(hand)) - 1
        # Meld-aware features (v2): an MLP learns meld arithmetic from raw one-hots
        # very slowly, so expose what the player can compute from its own hand.
        discard_value = np.ones(52, dtype=np.float32)    # deadwood left after discarding c (1 = n/a)
        in_meld = np.zeros(52, dtype=np.float32)          # c can be part of some meld
        for i, card in enumerate(solver.hand):  # solver.hand is sorted; bit i <-> solver.hand[i]
            in_meld[card] = float(bool(solver.by_card[i]))
            if len(hand) > HAND_SIZE and card != self.drawn_from_discard:
                discard_value[card] = min(solver.solve(full & ~(1 << i)), 100) / 100
        take_gain = 0.0
        if self.phase == "draw" and player == self.turn and self.discard:
            top = self.discard[-1]
            after, _ = solver_for(hand + [top]).best_discard(forbidden=top)
            take_gain = max(0, solver.solve() - after) / 100
        scalars = [
            min(self.hand_deadwood(player), 100) / 100,
            len(self.stock) / 31,
            self.turns[player] / MAX_TURNS,
            len(self.hands[1 - player]) / 11,
            solver.meld_cards() / 11,
            min(take_gain, 1.0),
        ]
        return np.concatenate([
            one_hot(hand), one_hot(self.discard[-1:]), one_hot(self.discard[:-1]), one_hot(self.known[player]),
            np.asarray([float(self.phase == "draw"), float(self.phase == "discard")], np.float32),
            discard_value, in_meld, np.asarray(scalars, np.float32),
        ]).astype(np.float32)

    def action_label(self, action: int) -> str:
        if action == DRAW_STOCK:
            return "Draw from stock"
        if action == DRAW_DISCARD:
            return f"Take {label(self.discard[-1])}" if self.discard else "Take discard"
        return f"Discard {label(action - DISCARD)}"

    def view(self, viewer: int | None) -> dict[str, Any]:
        legal = self.legal_actions() if viewer is not None and viewer == self.current_player else []
        return {
            "game": self.game_id,
            "done": self.done,
            "phase": "complete" if self.done else self.phase,
            "current_player": self.current_player,
            "hands": [[card_json(c) for c in h] if p == viewer or self.done else len(h) for p, h in enumerate(self.hands)],
            "discard_top": card_json(self.discard[-1]) if self.discard else None,
            "discard_count": len(self.discard),
            "stock_count": len(self.stock),
            "deadwood": self.hand_deadwood(viewer) if viewer is not None else None,
            "final_deadwood": self.final_deadwood,
            "winner": self.winner,
            "drawn_from_discard": card_json(self.drawn_from_discard) if self.drawn_from_discard is not None and viewer == self.turn else None,
            "last_event": self.last_event,
            "legal_actions": [{"action": a, "label": self.action_label(a)} for a in legal],
        }
