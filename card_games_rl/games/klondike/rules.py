"""Klondike solitaire engine, draw-1 (docs/ENV_DESIGN.md §3)."""
from __future__ import annotations

from typing import Any

import numpy as np

from card_games_rl.core import Game
from card_games_rl.core.cards import SUITS, card_json, is_red, label, make_card, new_deck, rank_of, suit_of

DRAW, RECYCLE, WASTE_TO_FOUNDATION = 0, 1, 2
WASTE_TO_TABLEAU = 3          # + column (7)
TABLEAU_TO_FOUNDATION = 10    # + column (7)
FOUNDATION_TO_TABLEAU = 17    # + suit*7 + column (28)
TABLEAU_TO_TABLEAU = 45       # + (src*6 + dst')*13 + (n-1)  (546)
NUM_ACTIONS = 591

STALL_LIMIT = 200
STEP_LIMIT = 1000
HIDDEN_TOTAL = 21
DEADLOCK_CHECK_EVERY = 8      # run the dead-end search every N moves without progress
DEADLOCK_SEARCH_LIMIT = 4000  # positions; beyond this the game is assumed alive

# Location categories used in the observation (12 per card).
LOC_WASTE_TOP, LOC_WASTE, LOC_STOCK_SEEN, LOC_FOUNDATION, LOC_UNKNOWN = 7, 8, 9, 10, 11


def krank(card: int) -> int:
    """Klondike rank: Ace = 1 .. King = 13."""
    rank = rank_of(card)
    return 1 if rank == 14 else rank


def card_for(suit: int, k: int) -> int:
    return make_card(14 if k == 1 else k, suit)


def encode_move(src: int, dst: int, n: int) -> int:
    dst_index = dst if dst < src else dst - 1
    return TABLEAU_TO_TABLEAU + (src * 6 + dst_index) * 13 + (n - 1)


def decode_move(action: int) -> tuple[int, int, int]:
    pair, n_minus_one = divmod(action - TABLEAU_TO_TABLEAU, 13)
    src, dst_index = divmod(pair, 6)
    dst = dst_index if dst_index < src else dst_index + 1
    return src, dst, n_minus_one + 1


class KlondikeGame(Game):
    game_id = "klondike"
    name = "Klondike Solitaire"
    num_players = 1
    num_actions = NUM_ACTIONS
    obs_size = 52 * 12 + 15
    OBS_VERSION = 1
    shaping_weight = 1.0
    step_cost = 0.001

    def __init__(self, seed: int | None = None, auto_safe_moves: bool = True):
        super().__init__(seed)
        self.auto_safe_moves = auto_safe_moves

    def _deal(self) -> None:
        deck = new_deck(self.rng)
        self.tableau: list[list[int]] = []
        self.hidden: list[int] = []
        for column in range(7):
            self.tableau.append([deck.pop() for _ in range(column + 1)])
            self.hidden.append(column)
        self.stock = deck
        self.waste: list[int] = []
        self.foundation = [0, 0, 0, 0]
        self.seen: set[int] = set()
        self.revealed = 0
        self.steps = 0
        self.stall = 0
        self.best_progress = 0
        self.won = False
        self.deadlocked = False
        self.auto_moves: list[str] = []
        self._run_auto_moves()
        self.start_tracking()

    # ----- progress, repetition and dead ends ------------------------------
    def _progress(self) -> int:
        """Monotone measure of new information: foundation cards, revealed cards, stock cards seen."""
        return sum(self.foundation) + self.revealed + len(self.seen)

    def start_tracking(self) -> None:
        """(Re)start repetition tracking from the current position."""
        self.best_progress = self._progress()
        self.visited = {self.position_key()}

    def position_key(self) -> tuple:
        return (tuple(map(tuple, self.tableau)), tuple(self.hidden), tuple(self.stock), tuple(self.waste),
                tuple(self.foundation))

    def _clone(self) -> "KlondikeGame":
        """Cheap copy of the card state for look-ahead (shares the rng and history)."""
        other = object.__new__(KlondikeGame)
        other.__dict__.update(self.__dict__)
        other.tableau = [list(p) for p in self.tableau]
        other.hidden, other.stock, other.waste = list(self.hidden), list(self.stock), list(self.waste)
        other.foundation, other.seen = list(self.foundation), set(self.seen)
        return other

    def _successor(self, action: int) -> "KlondikeGame":
        nxt = self._clone()
        nxt._move(action)
        return nxt

    def is_deadlocked(self, limit: int = DEADLOCK_SEARCH_LIMIT) -> bool:
        """True when no sequence of moves can make progress (reveal a card, see a new
        stock card, or reach a new best foundation count).

        Every position searched has the same face-down cards as the current one, so
        the search uses only information the player has. Beyond ``limit`` positions
        the game is assumed to be alive.
        """
        seen = {self.position_key()}
        frontier = [self]
        while frontier:
            state = frontier.pop()
            for action in state._candidate_moves():
                nxt = state._successor(action)
                if nxt._progress() > self.best_progress:
                    return False
                key = nxt.position_key()
                if key not in seen:
                    if len(seen) >= limit:
                        return False
                    seen.add(key)
                    frontier.append(nxt)
        return True

    # ----- rule helpers ----------------------------------------------------

    def can_found(self, card: int) -> bool:
        return self.foundation[suit_of(card)] == krank(card) - 1

    def can_place(self, card: int, column: int) -> bool:
        pile = self.tableau[column]
        if not pile:
            return krank(card) == 13
        top = pile[-1]
        return krank(top) == krank(card) + 1 and is_red(top) != is_red(card)

    def face_up(self, column: int) -> int:
        return len(self.tableau[column]) - self.hidden[column]

    def _is_safe(self, card: int) -> bool:
        k = krank(card)
        if k <= 2:
            return True
        opposite = [s for s in range(4) if (s in (1, 2)) != is_red(card)]
        return all(self.foundation[s] >= k - 1 for s in opposite)

    def _flip(self, column: int) -> None:
        if self.tableau[column] and self.hidden[column] == len(self.tableau[column]):
            self.hidden[column] -= 1
            self.revealed += 1

    def _to_foundation(self, card: int) -> None:
        self.foundation[suit_of(card)] += 1

    def _run_auto_moves(self) -> None:
        self.auto_moves = []
        if not self.auto_safe_moves:
            return
        moved = True
        while moved:
            moved = False
            if self.waste and self.can_found(self.waste[-1]) and self._is_safe(self.waste[-1]):
                card = self.waste.pop()
                self._to_foundation(card)
                self.auto_moves.append(label(card))
                moved = True
            for column in range(7):
                pile = self.tableau[column]
                if pile and self.face_up(column) and self.can_found(pile[-1]) and self._is_safe(pile[-1]):
                    card = pile.pop()
                    self._to_foundation(card)
                    self._flip(column)
                    self.auto_moves.append(label(card))
                    moved = True

    # ----- contract --------------------------------------------------------
    @property
    def current_player(self) -> int:
        return -1 if self.done else 0

    def _legal_actions(self) -> list[int]:
        """Rule-legal moves, minus those that recreate a position already seen since the
        last progress, which removes back-and-forth loops.

        Repeats are only a preference: if every move repeats a position, all are
        allowed again. Blocking them outright can cut the only path to a win (verified
        with the solver), so the game ends only through the exact dead-end search,
        having no moves at all, or the move limits.
        """
        if self.done:
            return []
        moves = self._candidate_moves()
        fresh = [a for a in moves if self._successor(a).position_key() not in self.visited]
        return fresh or moves

    def _candidate_moves(self) -> list[int]:
        legal = []
        if self.stock:
            legal.append(DRAW)
        elif self.waste:
            legal.append(RECYCLE)
        if self.waste:
            top = self.waste[-1]
            if self.can_found(top):
                legal.append(WASTE_TO_FOUNDATION)
            legal.extend(WASTE_TO_TABLEAU + c for c in range(7) if self.can_place(top, c))
        for column in range(7):
            if self.tableau[column] and self.can_found(self.tableau[column][-1]):
                legal.append(TABLEAU_TO_FOUNDATION + column)
        for suit in range(4):
            if self.foundation[suit]:
                card = card_for(suit, self.foundation[suit])
                if self.auto_safe_moves and self._is_safe(card):
                    continue  # it would bounce straight back to the foundation
                legal.extend(FOUNDATION_TO_TABLEAU + suit * 7 + c for c in range(7) if self.can_place(card, c))
        for src in range(7):
            pile, up = self.tableau[src], self.face_up(src)
            for n in range(1, up + 1):
                card = pile[-n]
                for dst in range(7):
                    if dst == src or not self.can_place(card, dst):
                        continue
                    if n == len(pile) and not self.tableau[dst]:
                        continue  # king-led column into an empty column: no-op
                    legal.append(encode_move(src, dst, n))
        return legal

    def _apply(self, action: int) -> None:
        self._move(action)
        self.steps += 1
        progress = self._progress()
        if progress > self.best_progress:
            self.best_progress, self.stall = progress, 0
            self.visited = set()
        else:
            self.stall += 1
        self.visited.add(self.position_key())
        if sum(self.foundation) == 52:
            self.won = self.is_terminal = True
        elif not self.legal_actions():
            self.deadlocked = self.is_terminal = True
        elif self.stall % DEADLOCK_CHECK_EVERY == 0 and self.stall and self.is_deadlocked():
            self.deadlocked = self.is_terminal = True
        elif self.stall >= STALL_LIMIT or self.steps >= STEP_LIMIT:
            self.is_truncated = True

    def _move(self, action: int) -> None:
        """Move the cards (and run auto moves) without any bookkeeping."""
        if action == DRAW:
            card = self.stock.pop()
            self.waste.append(card)
            self.seen.add(card)
        elif action == RECYCLE:
            self.stock = self.waste[::-1]
            self.waste = []
        elif action == WASTE_TO_FOUNDATION:
            self._to_foundation(self.waste.pop())
        elif action < TABLEAU_TO_FOUNDATION:
            self.tableau[action - WASTE_TO_TABLEAU].append(self.waste.pop())
        elif action < FOUNDATION_TO_TABLEAU:
            column = action - TABLEAU_TO_FOUNDATION
            self._to_foundation(self.tableau[column].pop())
            self._flip(column)
        elif action < TABLEAU_TO_TABLEAU:
            suit, column = divmod(action - FOUNDATION_TO_TABLEAU, 7)
            self.tableau[column].append(card_for(suit, self.foundation[suit]))
            self.foundation[suit] -= 1
        else:
            src, dst, n = decode_move(action)
            moving = self.tableau[src][-n:]
            del self.tableau[src][-n:]
            self.tableau[dst].extend(moving)
            self._flip(src)
        self._run_auto_moves()

    def score(self, player: int) -> float:
        return 0.5 * sum(self.foundation) / 52 + (0.5 if self.won else 0.0)

    def potential(self, player: int) -> float:
        return 0.25 * self.revealed / HIDDEN_TOTAL

    def all_cards(self) -> list[int]:
        found = [card_for(s, k) for s in range(4) for k in range(1, self.foundation[s] + 1)]
        return [c for pile in self.tableau for c in pile] + self.stock + self.waste + found

    # ----- observation -----------------------------------------------------
    def observation(self, player: int) -> np.ndarray:
        locations = np.zeros((52, 12), dtype=np.float32)
        locations[:, LOC_UNKNOWN] = 1.0

        def place(card: int, loc: int) -> None:
            locations[card, LOC_UNKNOWN] = 0.0
            locations[card, loc] = 1.0

        for column, pile in enumerate(self.tableau):
            for card in pile[self.hidden[column]:]:
                place(card, column)
        for i, card in enumerate(self.waste):
            place(card, LOC_WASTE_TOP if i == len(self.waste) - 1 else LOC_WASTE)
        for card in self.stock:
            if card in self.seen:
                place(card, LOC_STOCK_SEEN)
        for suit in range(4):
            for k in range(1, self.foundation[suit] + 1):
                place(card_for(suit, k), LOC_FOUNDATION)
        globals_ = [h / 6 for h in self.hidden] + [
            len(self.stock) / 24, len(self.waste) / 24, *[f / 13 for f in self.foundation],
            self.stall / STALL_LIMIT, self.steps / STEP_LIMIT,
        ]
        return np.concatenate([locations.ravel(), np.asarray(globals_, np.float32)])

    def action_label(self, action: int) -> str:
        if action == DRAW:
            return "Draw from stock"
        if action == RECYCLE:
            return "Recycle waste into stock"
        if action == WASTE_TO_FOUNDATION:
            return "Waste → foundation"
        if action < TABLEAU_TO_FOUNDATION:
            return f"Waste → column {action - WASTE_TO_TABLEAU + 1}"
        if action < FOUNDATION_TO_TABLEAU:
            return f"Column {action - TABLEAU_TO_FOUNDATION + 1} → foundation"
        if action < TABLEAU_TO_TABLEAU:
            suit, column = divmod(action - FOUNDATION_TO_TABLEAU, 7)
            return f"{SUITS[suit]} foundation → column {column + 1}"
        src, dst, n = decode_move(action)
        return f"Move {n} card{'s' if n > 1 else ''} column {src + 1} → column {dst + 1}"

    def describe_action(self, action: int) -> dict[str, Any]:
        """Structured source/destination for drag-and-drop in the UI."""
        if action == DRAW or action == RECYCLE:
            return {"from": "stock", "to": "waste" if action == DRAW else "stock"}
        if action == WASTE_TO_FOUNDATION:
            return {"from": "waste", "to": "foundation"}
        if action < TABLEAU_TO_FOUNDATION:
            return {"from": "waste", "to": "tableau", "dst": action - WASTE_TO_TABLEAU}
        if action < FOUNDATION_TO_TABLEAU:
            return {"from": "tableau", "src": action - TABLEAU_TO_FOUNDATION, "n": 1, "to": "foundation"}
        if action < TABLEAU_TO_TABLEAU:
            suit, column = divmod(action - FOUNDATION_TO_TABLEAU, 7)
            return {"from": "foundation", "suit": suit, "to": "tableau", "dst": column}
        src, dst, n = decode_move(action)
        return {"from": "tableau", "src": src, "n": n, "to": "tableau", "dst": dst}

    def view(self, viewer: int | None) -> dict[str, Any]:
        return {
            "game": self.game_id,
            "done": self.done,
            "won": self.won,
            "deadlocked": self.deadlocked,
            "tableau": [[card_json(c) if i >= self.hidden[col] else None for i, c in enumerate(pile)]
                        for col, pile in enumerate(self.tableau)],
            "waste": [card_json(c) for c in self.waste[-3:]],
            "waste_count": len(self.waste),
            "stock_count": len(self.stock),
            "foundation": [card_json(card_for(s, self.foundation[s])) if self.foundation[s] else None for s in range(4)],
            "foundation_count": sum(self.foundation),
            "steps": self.steps,
            "stall": self.stall,
            "auto_moves": self.auto_moves,
            "legal_actions": [{"action": a, "label": self.action_label(a), **self.describe_action(a)}
                              for a in self.legal_actions()],
        }
