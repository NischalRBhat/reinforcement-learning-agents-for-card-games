"""Hearts (Black Lady) engine, one hand per episode (docs/ENV_DESIGN.md §7)."""
from __future__ import annotations

from typing import Any

import numpy as np

from card_games_rl.core import Game
from card_games_rl.core.cards import HEARTS, card_json, label, make_card, new_deck, one_hot, one_hot_index, rank_of, suit_of

TWO_CLUBS = make_card(2, 0)
QUEEN_SPADES = make_card(12, 3)
PASS_LEFT, PASS_RIGHT, PASS_ACROSS, PASS_NONE = range(4)
PASS_OFFSETS = {PASS_LEFT: 1, PASS_RIGHT: 3, PASS_ACROSS: 2}
PASS_NAMES = ["left", "right", "across", "none"]


def card_points(card: int) -> int:
    return 13 if card == QUEEN_SPADES else int(suit_of(card) == HEARTS)


class HeartsGame(Game):
    game_id = "hearts"
    name = "Hearts"
    num_players = 4
    num_actions = 52
    obs_size = 446
    OBS_VERSION = 1
    zero_sum = True

    def __init__(self, seed: int | None = None, pass_direction: int | None = None):
        super().__init__(seed)
        self.fixed_direction = pass_direction

    def _deal(self) -> None:
        deck = new_deck(self.rng)
        self.hands = [sorted(deck[i * 13:(i + 1) * 13]) for i in range(4)]
        self.direction = self.fixed_direction if self.fixed_direction is not None else self.rng.randrange(4)
        self.passing = self.direction != PASS_NONE
        self.passer = 0
        self.pass_picks: list[list[int]] = [[] for _ in range(4)]
        self.received: list[list[int]] = [[] for _ in range(4)]
        self.trick: list[tuple[int, int]] = []
        self.last_trick: list[tuple[int, int]] = []
        self.last_winner = -1
        self.tricks_played = 0
        self.points = [0, 0, 0, 0]
        self.taken: list[list[int]] = [[] for _ in range(4)]
        self.played: set[int] = set()
        self.hearts_broken = False
        self.voids = [[False] * 4 for _ in range(4)]
        self.leader = 0
        if not self.passing:
            self._start_play()

    def _start_play(self) -> None:
        self.passing = False
        self.leader = next(p for p in range(4) if TWO_CLUBS in self.hands[p])

    # ----- contract --------------------------------------------------------
    @property
    def current_player(self) -> int:
        if self.done:
            return -1
        if self.passing:
            return self.passer
        return (self.leader + len(self.trick)) % 4

    def _legal_actions(self) -> list[int]:
        if self.done:
            return []
        player = self.current_player
        hand = self.hands[player]
        if self.passing:
            return [c for c in hand if c not in self.pass_picks[player]]
        if not self.trick:
            if self.tricks_played == 0:
                return [TWO_CLUBS]
            if not self.hearts_broken:
                return [c for c in hand if suit_of(c) != HEARTS] or list(hand)
            return list(hand)
        lead = suit_of(self.trick[0][1])
        follow = [c for c in hand if suit_of(c) == lead]
        if follow:
            return follow
        if self.tricks_played == 0:  # no points on the first trick unless unavoidable
            clean = [c for c in hand if card_points(c) == 0]
            return clean or [c for c in hand if c != QUEEN_SPADES] or list(hand)
        return list(hand)

    def _apply(self, card: int) -> None:
        if self.passing:
            self.pass_picks[self.passer].append(card)
            if len(self.pass_picks[self.passer]) == 3:
                self.passer += 1
                if self.passer == 4:
                    self._exchange()
            return
        player = self.current_player
        self.hands[player].remove(card)
        if self.trick and suit_of(card) != suit_of(self.trick[0][1]):
            self.voids[player][suit_of(self.trick[0][1])] = True
        self.trick.append((player, card))
        self.played.add(card)
        if suit_of(card) == HEARTS:
            self.hearts_broken = True
        if len(self.trick) == 4:
            self._finish_trick()

    def _exchange(self) -> None:
        offset = PASS_OFFSETS[self.direction]
        for player in range(4):
            target = (player + offset) % 4
            for card in self.pass_picks[player]:
                self.hands[player].remove(card)
            self.received[target] = list(self.pass_picks[player])
        for player in range(4):
            self.hands[player] = sorted(self.hands[player] + self.received[player])
        self._start_play()

    def _finish_trick(self) -> None:
        lead = suit_of(self.trick[0][1])
        winner = max((c for c in self.trick if suit_of(c[1]) == lead), key=lambda pc: rank_of(pc[1]))[0]
        self.points[winner] += sum(card_points(c) for _, c in self.trick)
        self.taken[winner].extend(c for _, c in self.trick)
        self.last_trick, self.last_winner = self.trick, winner
        self.trick = []
        self.leader = winner
        self.tricks_played += 1
        if self.tricks_played == 13:
            self.is_terminal = True

    # ----- scoring ---------------------------------------------------------
    def moon_shooter(self) -> int | None:
        if self.is_terminal and 26 in self.points:
            return self.points.index(26)
        return None

    def hand_points(self) -> list[int]:
        """Points charged for this hand (shoot-the-moon applied at the end)."""
        shooter = self.moon_shooter()
        if shooter is not None:
            return [0 if p == shooter else 26 for p in range(4)]
        return list(self.points)

    def score(self, player: int) -> float:
        points = self.hand_points()
        return (sum(points) / 4 - points[player]) / 26

    def winners(self) -> list[int]:
        if not self.done:
            return []
        points = self.hand_points()
        return [p for p in range(4) if points[p] == min(points)]

    def all_cards(self) -> list[int]:
        return [c for h in self.hands for c in h] + [c for _, c in self.trick] + [c for t in self.taken for c in t]

    # ----- observation -----------------------------------------------------
    def observation(self, player: int) -> np.ndarray:
        rel = [(player + i) % 4 for i in range(4)]
        trick_blocks = np.zeros((4, 52), dtype=np.float32)
        for seat, card in self.trick:
            trick_blocks[(seat - player) % 4, card] = 1.0
        voids = [float(self.voids[q][s]) for q in rel[1:] for s in range(4)]
        scalars = [
            float(self.hearts_broken), float(QUEEN_SPADES in self.played),
            float(self.passing), float(not self.passing),
            len(self.pass_picks[player]) / 3 if self.passing else 1.0,
            self.tricks_played / 13,
        ]
        leader = one_hot_index((self.leader - player) % 4, 4) if not self.passing else np.zeros(4, np.float32)
        return np.concatenate([
            one_hot(self.hands[player]), one_hot(self.played), trick_blocks.ravel(),
            one_hot(self.pass_picks[player]), one_hot(self.received[player]),
            one_hot_index(self.direction, 4), np.asarray([self.points[q] / 26 for q in rel], np.float32),
            np.asarray(voids, np.float32), np.asarray(scalars, np.float32), leader,
        ]).astype(np.float32)

    def action_label(self, action: int) -> str:
        return f"{'Pass' if self.passing else 'Play'} {label(action)}"

    def view(self, viewer: int | None) -> dict[str, Any]:
        legal = self.legal_actions() if viewer is not None and viewer == self.current_player else []
        return {
            "game": self.game_id,
            "done": self.done,
            "phase": "complete" if self.done else ("passing" if self.passing else "play"),
            "pass_direction": PASS_NAMES[self.direction],
            "current_player": self.current_player,
            "hands": [[card_json(c) for c in h] if p == viewer or self.done else len(h) for p, h in enumerate(self.hands)],
            "pass_picks": [card_json(c) for c in self.pass_picks[viewer]] if viewer is not None else [],
            "received": [card_json(c) for c in self.received[viewer]] if viewer is not None else [],
            "trick": [{"player": p, "card": card_json(c)} for p, c in self.trick],
            "last_trick": [{"player": p, "card": card_json(c)} for p, c in self.last_trick],
            "last_winner": self.last_winner,
            "leader": self.leader,
            "points": list(self.points),
            "hand_points": self.hand_points() if self.done else None,
            "moon_shooter": self.moon_shooter(),
            "hearts_broken": self.hearts_broken,
            "tricks_played": self.tricks_played,
            "legal_actions": [{"action": a, "label": self.action_label(a)} for a in legal],
        }
