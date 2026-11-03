"""Blackjack engine (docs/ENV_DESIGN.md §2)."""
from __future__ import annotations

from typing import Any

import numpy as np

from card_games_rl.core import Game
from card_games_rl.core.cards import card_json, new_deck, one_hot_index, rank_of

HIT, STAND, DOUBLE, SPLIT, SURRENDER, INSURE, NO_INSURE = range(7)
ACTION_NAMES = ["Hit", "Stand", "Double", "Split", "Surrender", "Take insurance", "Decline insurance"]


def card_points(card: int) -> int:
    """Hard value: Ace = 1, faces = 10."""
    rank = rank_of(card)
    return 1 if rank == 14 else min(rank, 10)


def value_index(card: int) -> int:
    """0..9 for 2..10, Ace → 9 (the order used in observations / the DP table)."""
    points = card_points(card)
    return 9 if points == 1 else points - 2


def hand_value(cards: list[int]) -> tuple[int, bool]:
    total = sum(card_points(c) for c in cards)
    if any(card_points(c) == 1 for c in cards) and total + 10 <= 21:
        return total + 10, True
    return total, False


def is_blackjack(cards: list[int]) -> bool:
    return len(cards) == 2 and hand_value(cards)[0] == 21


class BlackjackGame(Game):
    game_id = "blackjack"
    name = "Blackjack"
    num_players = 1
    num_actions = 7
    obs_size = 31
    OBS_VERSION = 1
    DECKS = 6

    def _deal(self) -> None:
        self.shoe = new_deck(self.rng, self.DECKS)
        draw = self.shoe.pop
        first, up, second, hole = draw(), draw(), draw(), draw()
        self.hands = [[first, second]]
        self.dealer = [up, hole]
        self.wagers = [1.0]
        self.active = 0
        self.split_done = False
        self.split_aces = False
        self.surrendered = False
        self.insurance = 0.0
        self.insurance_phase = card_points(up) == 1
        self.net = 0.0
        self.outcomes: list[str] = []
        if not self.insurance_phase:
            self._peek()

    # ----- helpers ---------------------------------------------------------
    @property
    def up_card(self) -> int:
        return self.dealer[0]

    def _peek(self) -> None:
        """Dealer peeks (Ace/ten up); naturals end the hand immediately."""
        if is_blackjack(self.dealer) or is_blackjack(self.hands[0]):
            self._resolve()

    def _draw(self) -> int:
        return self.shoe.pop()

    def _next_hand(self) -> None:
        self.active += 1
        while self.active < len(self.hands) and hand_value(self.hands[self.active])[0] == 21:
            self.active += 1  # 21 auto-stands
        if self.active >= len(self.hands):
            self.active = len(self.hands) - 1
            self._resolve()

    def _resolve(self) -> None:
        dealer_bj = is_blackjack(self.dealer)
        live = not self.surrendered and any(hand_value(h)[0] <= 21 for h in self.hands)
        if live and not dealer_bj and not (len(self.hands) == 1 and is_blackjack(self.hands[0])):
            while hand_value(self.dealer)[0] < 17:  # stands on soft 17
                self.dealer.append(self._draw())
        dealer_total = hand_value(self.dealer)[0]
        net, outcomes = 0.0, []
        for hand, wager in zip(self.hands, self.wagers):
            total = hand_value(hand)[0]
            natural = is_blackjack(hand) and not self.split_done
            if self.surrendered:
                result, outcome = -0.5 * wager, "surrender"
            elif total > 21:
                result, outcome = -wager, "bust"
            elif natural and not dealer_bj:
                result, outcome = 1.5 * wager, "blackjack"
            elif dealer_bj:
                result, outcome = (0.0, "push") if natural else (-wager, "dealer blackjack")
            elif dealer_total > 21 or total > dealer_total:
                result, outcome = wager, "win"
            elif total == dealer_total:
                result, outcome = 0.0, "push"
            else:
                result, outcome = -wager, "lose"
            net += result
            outcomes.append(outcome)
        if self.insurance:
            net += 2 * self.insurance if dealer_bj else -self.insurance
        self.net = net
        self.outcomes = outcomes
        self.is_terminal = True

    # ----- contract --------------------------------------------------------
    @property
    def current_player(self) -> int:
        return -1 if self.done else 0

    def can_split(self) -> bool:
        hand = self.hands[self.active]
        return not self.split_done and len(hand) == 2 and card_points(hand[0]) == card_points(hand[1])

    def can_surrender(self) -> bool:
        return not self.split_done and len(self.hands[0]) == 2

    def _legal_actions(self) -> list[int]:
        if self.done:
            return []
        if self.insurance_phase:
            return [INSURE, NO_INSURE]
        legal = [HIT, STAND]
        if len(self.hands[self.active]) == 2:
            legal.append(DOUBLE)
        if self.can_split():
            legal.append(SPLIT)
        if self.can_surrender():
            legal.append(SURRENDER)
        return legal

    def _apply(self, action: int) -> None:
        if action in (INSURE, NO_INSURE):
            self.insurance = 0.5 if action == INSURE else 0.0
            self.insurance_phase = False
            self._peek()
            return
        hand = self.hands[self.active]
        if action == HIT:
            hand.append(self._draw())
            if hand_value(hand)[0] >= 21:
                self._next_hand()
        elif action == STAND:
            self._next_hand()
        elif action == DOUBLE:
            self.wagers[self.active] *= 2
            hand.append(self._draw())
            self._next_hand()
        elif action == SPLIT:
            first, second = hand
            self.hands = [[first, self._draw()], [second, self._draw()]]
            self.wagers = [1.0, 1.0]
            self.split_done = True
            self.active = 0
            if card_points(first) == 1:
                self.split_aces = True  # one card each, then stand
                self._resolve()
            elif hand_value(self.hands[0])[0] == 21:
                self._next_hand()
        elif action == SURRENDER:
            self.surrendered = True
            self._resolve()

    def score(self, player: int) -> float:
        return self.net if self.is_terminal else 0.0

    def all_cards(self) -> list[int]:
        return self.shoe + self.dealer + [c for h in self.hands for c in h]

    # ----- observation -----------------------------------------------------
    def observation(self, player: int) -> np.ndarray:
        hand = self.hands[self.active]
        total, soft = hand_value(hand)
        pair = one_hot_index(value_index(hand[0]), 10) if self.can_split() else np.zeros(10, np.float32)
        other = self.hands[1 - self.active] if len(self.hands) == 2 else []
        flags = [
            float(self.insurance_phase),
            float(len(hand) == 2 and not self.done),
            float(self.can_split()),
            float(self.can_surrender()),
            float(self.split_done),
            float(self.active),
            float(self.split_aces),
        ]
        head = [min(total, 21) / 21, float(soft), min(len(hand), 8) / 8]
        tail = [min(hand_value(other)[0], 21) / 21 if other else 0.0]
        return np.concatenate([
            np.asarray(head, np.float32), pair, one_hot_index(value_index(self.up_card), 10),
            np.asarray(flags, np.float32), np.asarray(tail, np.float32),
        ]).astype(np.float32)

    def action_label(self, action: int) -> str:
        return ACTION_NAMES[action]

    def view(self, viewer: int | None) -> dict[str, Any]:
        show_hole = self.done
        dealer_cards = [card_json(c) for c in self.dealer] if show_hole else [card_json(self.up_card), None]
        dealer_total = hand_value(self.dealer)[0] if show_hole else hand_value([self.up_card])[0]
        return {
            "game": self.game_id,
            "done": self.done,
            "phase": "insurance" if self.insurance_phase else ("complete" if self.done else "player"),
            "dealer": {"cards": dealer_cards, "total": dealer_total},
            "hands": [
                {"cards": [card_json(c) for c in h], "total": hand_value(h)[0], "soft": hand_value(h)[1],
                 "wager": w, "active": i == self.active and not self.done,
                 "outcome": self.outcomes[i] if self.done else None}
                for i, (h, w) in enumerate(zip(self.hands, self.wagers))
            ],
            "insurance": self.insurance,
            "net": self.net if self.done else None,
            "legal_actions": [{"action": a, "label": ACTION_NAMES[a]} for a in self.legal_actions()],
        }
