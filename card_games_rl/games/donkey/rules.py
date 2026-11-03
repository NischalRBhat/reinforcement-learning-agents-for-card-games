"""Donkey (Gadha Ladam / Kazhutha) engine, 4 players (docs/GAMES.md §7).

The AS holder leads it. Players must follow the led suit if they can.

* Everyone still in the game follows: the trick is discarded (out of play) and the
  highest card leads next (or the next player clockwise with cards, if they are out).
* Someone cannot follow: they play any card (a "cut"), the trick stops at once, and
  whoever has played the highest card of the led suit picks up every card in it,
  then leads next.

A player with no cards left is safe. The last player holding cards is the Donkey.
Two players can avoid ending forever, so a position (all hands + leader) seen for
the third time stops the game, as does a 300-trick cap: the player holding the most
cards is then the Donkey (a tie for most means no Donkey).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from card_games_rl.core import Game
from card_games_rl.core.cards import card_json, label, make_card, new_deck, one_hot, one_hot_index, rank_of, suit_of

ACE_SPADES = make_card(14, 3)
MAX_TRICKS = 300
REPETITIONS = 3


class DonkeyGame(Game):
    game_id = "donkey"
    name = "Donkey"
    num_players = 4
    num_actions = 52
    obs_size = 501
    OBS_VERSION = 1
    zero_sum = True
    shaping_weight = 1.0

    def _deal(self) -> None:
        deck = new_deck(self.rng)
        self.hands = [sorted(deck[i * 13:(i + 1) * 13]) for i in range(4)]
        self.leader = next(p for p in range(4) if ACE_SPADES in self.hands[p])
        self.trick: list[tuple[int, int]] = []
        self.trick_players = self._order_from(self.leader)
        self.discarded: list[int] = []
        self.known = [set() for _ in range(4)]          # public: picked-up cards still held
        self.void = [[False] * 4 for _ in range(4)]     # public: void[p][suit], shown by cutting
        self.out_order: list[int] = []
        self.tricks_played = 0
        self.last_trick: dict[str, Any] | None = None
        self.donkey: int | None = None
        self.losers: list[int] = []
        self.stopped = False                            # ended by repetition or the trick cap
        self.positions: Counter = Counter()

    # ----- helpers ---------------------------------------------------------
    def _order_from(self, seat: int) -> list[int]:
        return [(seat + i) % 4 for i in range(4) if self.hands[(seat + i) % 4]]

    @property
    def led_suit(self) -> int | None:
        return suit_of(self.trick[0][1]) if self.trick else None

    def highest_led(self) -> int | None:
        """Seat holding the highest card of the led suit in the current trick."""
        led = [(rank_of(c), p) for p, c in self.trick if suit_of(c) == self.led_suit]
        return max(led)[1] if led else None

    def active(self) -> list[int]:
        return [p for p in range(4) if self.hands[p]]

    # ----- contract --------------------------------------------------------
    @property
    def current_player(self) -> int:
        return -1 if self.done else self.trick_players[len(self.trick)]

    def _legal_actions(self) -> list[int]:
        hand = self.hands[self.current_player]
        if not self.trick:
            return [ACE_SPADES] if self.tricks_played == 0 else list(hand)
        follow = [c for c in hand if suit_of(c) == self.led_suit]
        return follow or list(hand)

    def _apply(self, card: int) -> None:
        player = self.current_player
        self.hands[player].remove(card)
        self.known[player].discard(card)
        self.trick.append((player, card))
        led = self.led_suit
        cards = [c for _, c in self.trick]
        if suit_of(card) != led:  # a cut: the highest card of the led suit picks everything up
            self.void[player][led] = True
            picker = self.highest_led()
            self.hands[picker] = sorted(self.hands[picker] + cards)
            self.known[picker] |= set(cards)
            for suit in {suit_of(c) for c in cards}:
                self.void[picker][suit] = False
            self.last_trick = {"type": "picked", "picker": picker, "cutter": player, "cards": cards,
                               "plays": list(self.trick)}
            next_leader = picker
        elif len(self.trick) == len(self.trick_players):  # everyone followed: discard
            next_leader = self.highest_led()
            self.discarded.extend(cards)
            self.last_trick = {"type": "discarded", "winner": next_leader, "cards": cards, "plays": list(self.trick)}
        else:
            return
        self._finish_trick(next_leader)

    def _finish_trick(self, next_leader: int) -> None:
        self.trick = []
        self.tricks_played += 1
        for p in [*self._order_from(next_leader), *range(4)]:
            if not self.hands[p] and p not in self.out_order:
                self.out_order.append(p)
        active = self.active()
        if len(active) <= 1:
            self.losers = active
            self.donkey = active[0] if active else None
            self.is_terminal = True
            return
        if not self.hands[next_leader]:
            next_leader = next(p for p in ((next_leader + i) % 4 for i in range(1, 4)) if self.hands[p])
        self.leader = next_leader
        self.trick_players = self._order_from(next_leader)
        key = (tuple(tuple(h) for h in self.hands), next_leader)
        self.positions[key] += 1
        if self.positions[key] >= REPETITIONS or self.tricks_played >= MAX_TRICKS:
            most = max(len(h) for h in self.hands)
            self.losers = [p for p in range(4) if len(self.hands[p]) == most]
            self.donkey = self.losers[0] if len(self.losers) == 1 else None
            self.stopped = self.is_terminal = True

    # ----- rewards ---------------------------------------------------------
    def score(self, player: int) -> float:
        """Zero-sum: the Donkey -1, everyone else +1/3; players tied for most cards share -1."""
        if not self.is_terminal or not self.losers or len(self.losers) == 4:
            return 0.0
        k = len(self.losers)
        return -1.0 / k if player in self.losers else 1.0 / (4 - k)

    def potential(self, player: int) -> float:
        """Every card shed is good: own card count against the table average."""
        sizes = [len(h) for h in self.hands]
        others = (sum(sizes) - sizes[player]) / 3
        return 0.5 * (others - sizes[player]) / 52

    def winners(self) -> list[int]:
        if not self.done or not self.losers:
            return []
        return [p for p in range(4) if p not in self.losers]

    def all_cards(self) -> list[int]:
        return [c for h in self.hands for c in h] + [c for _, c in self.trick] + self.discarded

    # ----- observation -----------------------------------------------------
    def observation(self, player: int) -> np.ndarray:
        rel = [(player + i) % 4 for i in range(4)]
        trick = np.zeros((4, 52), dtype=np.float32)
        for p, c in self.trick:
            trick[(p - player) % 4, c] = 1.0
        highest = self.highest_led()
        return np.concatenate([
            one_hot(self.hands[player]),
            one_hot(self.discarded),
            trick.ravel(),
            *[one_hot(self.known[q]) for q in rel[1:]],
            np.asarray([float(v) for q in rel[1:] for v in self.void[q]], np.float32),
            np.asarray([len(self.hands[q]) / 52 for q in rel], np.float32),
            np.asarray([float(bool(self.hands[q])) for q in rel], np.float32),
            one_hot_index(self.led_suit if self.trick else -1, 4),
            np.asarray([float(not self.trick), len(self.trick) / 3, float(self.tricks_played == 0),
                        self.tricks_played / MAX_TRICKS, len(self.active()) / 4], np.float32),
            one_hot_index((highest - player) % 4 if highest is not None else -1, 4),
        ]).astype(np.float32)

    def action_label(self, action: int) -> str:
        return f"{'Lead' if not self.trick else 'Play'} {label(action)}"

    def view(self, viewer: int | None) -> dict[str, Any]:
        legal = self.legal_actions() if viewer is not None and viewer == self.current_player else []
        last = self.last_trick
        return {
            "game": self.game_id,
            "done": self.done,
            "current_player": self.current_player,
            "leader": self.leader,
            "hands": [[card_json(c) for c in h] if p == viewer or self.done else len(h) for p, h in enumerate(self.hands)],
            "known": [[card_json(c) for c in sorted(k)] for k in self.known],
            "voids": [[s for s in range(4) if self.void[p][s]] for p in range(4)],
            "trick": [{"player": p, "card": card_json(c)} for p, c in self.trick],
            "led_suit": self.led_suit,
            "discarded": len(self.discarded),
            "last_trick": {**last, "cards": [card_json(c) for c in last["cards"]],
                           "plays": [{"player": p, "card": card_json(c)} for p, c in last["plays"]]} if last else None,
            "out_order": self.out_order,
            "tricks_played": self.tricks_played,
            "donkey": self.donkey,
            "losers": self.losers if self.done else [],
            "stopped": self.stopped,
            "legal_actions": [{"action": a, "label": self.action_label(a)} for a in legal],
        }
