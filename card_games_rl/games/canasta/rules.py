"""Canasta, 4 players in two partnerships (docs/GAMES.md §8, docs/ENV_DESIGN.md §10).

Classic (Hoyle) partnership rules. Partners sit opposite: seats 0 and 2 against 1 and 3.

* **Cards:** two 52-card decks plus 4 jokers (108). Jokers and 2s are wild. Each player
  gets 11 cards.
* **Turn:** draw from the stock, or take the whole discard pile; meld; discard one card.
* **Melds:** 3 or more cards of one rank (4 to Ace), with at least 2 natural cards and at
  most 3 wild cards. A partnership has one meld per rank. Seven or more cards make a
  *canasta*: natural (no wild cards) +500, mixed +300.
* **Threes:** red threes are bonus cards, laid out at once and replaced (+100 each, +800
  for all four; negative if the side never melded). A black three on top of the pile
  stops the next player taking it, and black threes may only be melded when going out.
* **Initial meld:** a side's first melds in a hand must be worth a minimum, set by its
  match score: 15 below zero, 50 up to 1,495, 90 up to 2,995, 120 from 3,000.
* **The pile:** to take it, the top card must be melded at once with cards from the hand.
  It is *frozen* for everyone once a wild card or red three is in it, and for a side that
  has not melded yet. A frozen pile needs a natural pair matching the top card. An
  unfrozen pile can also be taken with one natural and one wild card, or by adding the
  top card to the side's meld of that rank. Before a side has melded, the top card plus
  cards from the hand must meet the minimum before the rest of the pile is picked up.
* **Going out:** needs at least one canasta in the side: meld every card, or all but one
  and discard it. +100, or +200 *concealed* (the player melds the whole hand in one turn
  having made no meld before).
* **End of hand:** someone goes out, or the stock runs out and the player to draw cannot
  take the pile. Each side scores its melded cards, canasta bonuses, red threes and the
  going-out bonus, minus the cards left in its hands. Card values: joker 50, 2 and Ace 20,
  King to 8 10, 7 to 4 and black 3 5. The match is played to 5,000.

Cards are ints 0..107: ``deck * 52 + card`` for the two decks, 104..107 for the jokers.
Melding and discarding work with *kinds*: joker, 2, red 3, black 3, then ranks 4..Ace.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from card_games_rl.core import Game
from card_games_rl.core.cards import card_json as card52_json
from card_games_rl.core.cards import rank_label, rank_of, suit_of
from card_games_rl.core.counting import p_at_least

NUM_CARDS = 108
HAND_SIZE = 11
JOKER, TWO, RED3, BLACK3 = 0, 1, 2, 3          # kinds 4..14 are the ranks 4..Ace
KINDS = 15
COPIES = [4, 8, 4, 4] + [8] * 11
VALUES = [50, 20, 0, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 10, 20]
WILD = (JOKER, TWO)
TARGETS = list(range(4, 15)) + [BLACK3]         # meld targets: ranks 4..Ace, then black threes
WILD_COMBOS = [(j, d) for j in range(4) for d in range(4) if j + d <= 3]  # (jokers, twos) added

DRAW, TAKE_PAIR, TAKE_MIXED, TAKE_ADD = range(4)
MELD_BASE = 4
MELD_ACTIONS = len(TARGETS) * 4 * len(WILD_COMBOS)   # target × naturals 0..3 × wild combo
DISCARD_BASE = MELD_BASE + MELD_ACTIONS
NUM_ACTIONS = DISCARD_BASE + KINDS
CANASTA_SIZE = 7
NATURAL_CANASTA, MIXED_CANASTA = 500, 300
MATCH_TARGET = 5000
MAX_ACTIONS = 800
SCORE_SCALE = 2000.0
TEAM_NAMES = ("North–South", "East–West")


def kind_of(card: int) -> int:
    if card >= 104:
        return JOKER
    base = card % 52
    rank = rank_of(base)
    if rank == 2:
        return TWO
    if rank == 3:
        return RED3 if suit_of(base) in (1, 2) else BLACK3
    return rank


def kind_label(kind: int) -> str:
    return {JOKER: "Joker", TWO: "2", RED3: "red 3", BLACK3: "black 3"}.get(kind) or rank_label(kind)


def kind_plural(kind: int) -> str:
    return {JOKER: "jokers", TWO: "2s", RED3: "red 3s", BLACK3: "black 3s"}.get(kind) or f"{rank_label(kind)}s"


def card_json(card: int) -> dict[str, Any]:
    if card >= 104:
        return {"id": card, "rank": "JK", "suit": "X", "label": "Joker", "kind": JOKER}
    data = card52_json(card % 52)
    return {**data, "id": card, "kind": kind_of(card)}


def minimum_for(score: int) -> int:
    if score < 0:
        return 15
    if score < 1500:
        return 50
    return 90 if score < 3000 else 120


def decode_meld(action: int) -> tuple[int, int, int, int]:
    """Meld action -> (kind, naturals, jokers, twos)."""
    index = action - MELD_BASE
    combo = index % len(WILD_COMBOS)
    rest = index // len(WILD_COMBOS)
    jokers, twos = WILD_COMBOS[combo]
    return TARGETS[rest // 4], rest % 4, jokers, twos


def meld_action(kind: int, naturals: int, jokers: int = 0, twos: int = 0) -> int:
    return MELD_BASE + (TARGETS.index(kind) * 4 + naturals) * len(WILD_COMBOS) + WILD_COMBOS.index((jokers, twos))


class CanastaGame(Game):
    game_id = "canasta"
    name = "Canasta"
    num_players = 4
    num_actions = NUM_ACTIONS
    obs_size = 239
    OBS_VERSION = 1
    zero_sum = True
    shaping_weight = 1.0
    unique_cards = NUM_CARDS

    def __init__(self, seed: int | None = None, team_scores: list[int] | None = None, dealer: int | None = None):
        super().__init__(seed)
        self.team_scores = list(team_scores or [0, 0])
        self.fixed_dealer = dealer

    # ----- setup -----------------------------------------------------------
    def _deal(self) -> None:
        deck = list(range(NUM_CARDS))
        self.rng.shuffle(deck)
        self.stock = deck
        self.hands: list[list[int]] = [[] for _ in range(4)]
        self.red_threes: list[list[int]] = [[] for _ in range(4)]
        self.known = [Counter() for _ in range(4)]      # public: kinds seen picked up with the pile
        self.melds: list[dict[int, dict[str, list[int]]]] = [{}, {}]
        self.team_melded = [False, False]
        self.melded_cards = [False] * 4                 # has this player melded this hand
        self.pile: list[int] = []
        self.pile_frozen = False
        self.pile_pending: list[int] = []
        self.turn_value = 0
        self.prior_meld = False                         # current player had melded before this turn
        self.went_out: int | None = None
        self.concealed = False
        self.hand_scores: list[int] | None = None
        self.actions_taken = 0
        self.last_event = ""
        for _ in range(HAND_SIZE):
            for p in range(4):
                self._draw_into(p)
        self.dealer = self.fixed_dealer if self.fixed_dealer is not None else self.rng.randrange(4)
        while self.stock:                               # the upcard; wild cards and red threes freeze the pile
            card = self.stock.pop()
            self.pile.append(card)
            if kind_of(card) not in (JOKER, TWO, RED3):
                break
            self.pile_frozen = True
        self.turn = (self.dealer + 1) % 4
        self.phase = "draw"
        self._begin_turn()

    def _draw_into(self, player: int) -> int | None:
        """Draw one card; red threes are laid out and replaced. Returns the card kept."""
        while self.stock:
            card = self.stock.pop()
            if kind_of(card) == RED3:
                self.red_threes[player].append(card)
                continue
            self.hands[player].append(card)
            return card
        return None

    # ----- helpers ---------------------------------------------------------
    @staticmethod
    def team(player: int) -> int:
        return player % 2

    def team_seats(self, player: int) -> list[int]:
        return [player % 2, player % 2 + 2]

    def minimum(self, team: int) -> int:
        return minimum_for(self.team_scores[team])

    def count(self, player: int, kind: int) -> int:
        return sum(kind_of(c) == kind for c in self.hands[player])

    def cards_of(self, player: int, kind: int, n: int) -> list[int]:
        return sorted(c for c in self.hands[player] if kind_of(c) == kind)[:n]

    @staticmethod
    def meld_size(meld: dict[str, list[int]]) -> int:
        return len(meld["nat"]) + len(meld["jok"]) + len(meld["two"])

    @staticmethod
    def meld_wilds(meld: dict[str, list[int]]) -> int:
        return len(meld["jok"]) + len(meld["two"])

    def canastas(self, team: int) -> list[str]:
        return ["natural" if not self.meld_wilds(m) else "mixed"
                for m in self.melds[team].values() if self.meld_size(m) >= CANASTA_SIZE]

    def top_kind(self) -> int | None:
        return kind_of(self.pile[-1]) if self.pile else None

    def _max_meld_value(self, counts: Counter, melds: dict[int, dict[str, list[int]]]) -> int:
        """Most points the hand ``counts`` could still meld (no black threes): used to keep
        an initial meld reachable once started."""
        value, need_wild = 0, []
        slots = {k: 3 - self.meld_wilds(m) for k, m in melds.items() if k != BLACK3}
        for kind in range(4, 15):
            n = counts[kind]
            if kind in melds:
                value += n * VALUES[kind]
            elif n >= 3:
                value += n * VALUES[kind]
                slots[kind] = 3
            elif n == 2:
                need_wild.append(kind)
        wilds = sorted([JOKER] * counts[JOKER] + [TWO] * counts[TWO], key=lambda k: -VALUES[k])
        for kind in sorted(need_wild, key=lambda k: -VALUES[k]):
            if not wilds:
                break
            value += 2 * VALUES[kind] + VALUES[wilds.pop(0)]
            slots[kind] = 2
        free = sum(max(0, s) for s in slots.values())
        value += sum(VALUES[w] for w in wilds[:free])
        return value

    # ----- contract --------------------------------------------------------
    @property
    def current_player(self) -> int:
        return -1 if self.done else self.turn

    def _take_options(self, player: int) -> list[int]:
        if not self.pile:
            return []
        top = self.top_kind()
        if top in (JOKER, TWO, RED3, BLACK3):
            return []
        team = self.team(player)
        naturals = self.count(player, top)
        frozen = self.pile_frozen or not self.team_melded[team]
        options = []
        if naturals >= 2 and self._take_reaches_minimum(player, top):
            options.append(TAKE_PAIR)
        if not frozen:
            has_meld = top in self.melds[team]
            wilds = self.count(player, JOKER) + self.count(player, TWO)
            if naturals >= 1 and wilds and (not has_meld or self.meld_wilds(self.melds[team][top]) < 3):
                options.append(TAKE_MIXED)
            if has_meld:
                options.append(TAKE_ADD)
        return options

    def _take_reaches_minimum(self, player: int, top: int) -> bool:
        team = self.team(player)
        if self.team_melded[team]:
            return True
        counts = Counter(kind_of(c) for c in self.hands[player])
        counts[top] -= 2
        start = 3 * VALUES[top]
        melds = {top: {"nat": [0, 0, 0], "jok": [], "two": []}}
        return start + self._max_meld_value(counts, melds) >= self.minimum(team)

    def _meld_legal(self, player: int, kind: int, n: int, j: int, d: int) -> bool:
        team = self.team(player)
        added = n + j + d
        if added == 0 or self.count(player, kind) < n or self.count(player, JOKER) < j or self.count(player, TWO) < d:
            return False
        existing = self.melds[team].get(kind)
        size_after = (self.meld_size(existing) if existing else 0) + added
        if kind == BLACK3:  # only when going out, 3-4 black threes, no wild cards
            if j or d or existing or n < 3 or self.pile_pending:
                return False
            return bool(self.canastas(team)) and len(self.hands[player]) - added <= 1
        if existing is None:
            if n < 2 or added < 3 or j + d > 3:
                return False
        elif self.meld_wilds(existing) + j + d > 3:
            return False
        hand_after = len(self.hands[player]) - added
        if hand_after <= 1 and not self.pile_pending:
            if not self.canastas(team) and size_after < CANASTA_SIZE:
                return False  # cannot run out of cards without a canasta
        if not self.team_melded[team]:
            value = n * VALUES[kind] + j * VALUES[JOKER] + d * VALUES[TWO]
            counts = Counter(kind_of(c) for c in self.hands[player])
            counts[kind] -= n
            counts[JOKER] -= j
            counts[TWO] -= d
            melds = dict(self.melds[team])
            melds[kind] = {"nat": [0] * ((len(existing["nat"]) if existing else 0) + n),
                           "jok": [0] * ((len(existing["jok"]) if existing else 0) + j),
                           "two": [0] * ((len(existing["two"]) if existing else 0) + d)}
            if self.turn_value + value + self._max_meld_value(counts, melds) < self.minimum(team):
                return False
        return True

    def _legal_actions(self) -> list[int]:
        player = self.turn
        if self.phase == "draw":
            return ([DRAW] if self.stock else []) + self._take_options(player)
        team = self.team(player)
        counts = Counter(kind_of(c) for c in self.hands[player])
        legal = []
        for kind in TARGETS:  # only ranks the player holds, or can add wild cards to
            if not counts[kind] and kind not in self.melds[team]:
                continue
            for n in range(min(3, counts[kind]) + 1):
                for j, d in WILD_COMBOS:
                    if j <= counts[JOKER] and d <= counts[TWO] and self._meld_legal(player, kind, n, j, d):
                        legal.append(meld_action(kind, n, j, d))
        initial_pending = not self.team_melded[team] and (self.turn_value > 0 or self.pile_pending)
        last_card = len(self.hands[player]) == 1
        if not initial_pending and (not last_card or self.canastas(team)):
            legal += sorted({DISCARD_BASE + kind_of(c) for c in self.hands[player]})
        return legal

    def _begin_turn(self) -> None:
        """Start ``self.turn``'s turn, ending the hand if the stock is out and the pile can't be taken."""
        self.phase = "draw"
        self.turn_value = 0
        self.pile_pending = []
        self.prior_meld = self.melded_cards[self.turn]
        self.invalidate()
        if not self.stock and not self._take_options(self.turn):
            self._end_hand()
        elif not self._legal_actions():
            self._end_hand()

    def _apply(self, action: int) -> None:
        player = self.turn
        team = self.team(player)
        self.actions_taken += 1
        if action == DRAW:
            card = self._draw_into(player)
            self.phase = "play"
            self.last_event = f"P{player + 1} drew" + ("" if card is not None else " a red 3 and the stock ran out")
        elif action in (TAKE_PAIR, TAKE_MIXED, TAKE_ADD):
            self._take(player, action)
        elif action < DISCARD_BASE:
            kind, n, j, d = decode_meld(action)
            cards = self.cards_of(player, kind, n) + self.cards_of(player, JOKER, j) + self.cards_of(player, TWO, d)
            self._lay(player, kind, cards)
            self.last_event = f"P{player + 1} {self.action_label(action).lower()}"
            if not self.hands[player] and not self.pile_pending:
                self._go_out(player)
                return
        else:
            kind = action - DISCARD_BASE
            card = self.cards_of(player, kind, 1)[0]
            self.hands[player].remove(card)
            if self.known[player][kind]:
                self.known[player][kind] -= 1
            self.pile.append(card)
            if kind in WILD:
                self.pile_frozen = True
            self.last_event = f"P{player + 1} discarded {kind_label(kind)}"
            if not self.hands[player]:
                self._go_out(player)
                return
            self.turn = (player + 1) % 4
            self._begin_turn()
        if not self.done and (self.actions_taken >= MAX_ACTIONS or not self._legal_actions()):
            # The action cap, or a rare dead end (e.g. one card left, no canasta and an
            # empty stock): the hand is scored as if the stock had run out.
            self._end_hand()

    def _lay(self, player: int, kind: int, cards: list[int]) -> None:
        team = self.team(player)
        meld = self.melds[team].setdefault(kind, {"nat": [], "jok": [], "two": []})
        for card in cards:
            if card in self.hands[player]:  # the pile's top card comes from the table
                self.hands[player].remove(card)
            card_kind = kind_of(card)
            meld["jok" if card_kind == JOKER else "two" if card_kind == TWO else "nat"].append(card)
            if self.known[player][card_kind]:
                self.known[player][card_kind] -= 1
        self.melded_cards[player] = True
        if not self.team_melded[team]:
            self.turn_value += sum(VALUES[kind_of(c)] for c in cards)
            if self.turn_value >= self.minimum(team):
                self.team_melded[team] = True
                self._collect_pending(player)

    def _take(self, player: int, action: int) -> None:
        top_card = self.pile.pop()
        top = kind_of(top_card)
        rest, self.pile = self.pile, []
        self.pile_frozen = False
        if action == TAKE_PAIR:
            used = self.cards_of(player, top, 2)
        elif action == TAKE_MIXED:
            wild = self.cards_of(player, TWO, 1) or self.cards_of(player, JOKER, 1)
            used = self.cards_of(player, top, 1) + wild
        else:
            used = []
        for card in rest:  # the whole pile is seen by everyone
            if kind_of(card) == RED3:
                self.red_threes[player].append(card)
        rest = [c for c in rest if kind_of(c) != RED3]
        self.pile_pending = rest
        self._lay(player, top, [top_card] + used)
        if self.team_melded[self.team(player)]:
            self._collect_pending(player)
        self.phase = "play"
        self.last_event = f"P{player + 1} took the pile ({len(rest) + 1} cards)"

    def _collect_pending(self, player: int) -> None:
        for card in self.pile_pending:
            self.hands[player].append(card)
            self.known[player][kind_of(card)] += 1
        self.pile_pending = []

    def _go_out(self, player: int) -> None:
        self.went_out = player
        self.concealed = not self.prior_meld  # the whole hand melded in one turn, nothing melded before
        self.last_event = f"P{player + 1} went out" + (" concealed" if self.concealed else "")
        self._end_hand()

    def _end_hand(self) -> None:
        if self.pile_pending:  # a pile picked up but not yet released still belongs to the taker
            self._collect_pending(self.turn)
        scores = []
        for team in (0, 1):
            melded = sum(VALUES[kind_of(c)] for m in self.melds[team].values() for part in m.values() for c in part)
            bonus = sum(NATURAL_CANASTA if kind == "natural" else MIXED_CANASTA for kind in self.canastas(team))
            reds = sum(len(self.red_threes[p]) for p in self.team_seats(team))
            red_points = 800 if reds == 4 else 100 * reds
            if not self.melds[team]:
                red_points = -red_points
            out = 0
            if self.went_out is not None and self.team(self.went_out) == team:
                out = 200 if self.concealed else 100
            left = sum(VALUES[kind_of(c)] for p in self.team_seats(team) for c in self.hands[p])
            scores.append(melded + bonus + red_points + out - left)
        self.hand_scores = scores
        self.is_terminal = True

    # ----- rewards ---------------------------------------------------------
    def score(self, player: int) -> float:
        if not self.is_terminal or self.hand_scores is None:
            return 0.0
        team = self.team(player)
        return (self.hand_scores[team] - self.hand_scores[1 - team]) / SCORE_SCALE

    def _progress(self, team: int) -> float:
        melded = sum(VALUES[kind_of(c)] for m in self.melds[team].values() for part in m.values() for c in part)
        bonus = sum(NATURAL_CANASTA if kind == "natural" else MIXED_CANASTA for kind in self.canastas(team))
        reds = sum(len(self.red_threes[p]) for p in self.team_seats(team)) * 100
        # No credit for melds still short of a canasta: it taught early agents to lay cards
        # down too soon, which shows the opponents your ranks and spends the pairs you
        # need to take the pile.
        return melded + bonus + (reds if self.melds[team] else 0)

    def potential(self, player: int) -> float:
        """Public progress (melded cards, canastas, red threes) against the other side."""
        team = self.team(player)
        return 0.5 * (self._progress(team) - self._progress(1 - team)) / SCORE_SCALE

    def winners(self) -> list[int]:
        if not self.done or self.hand_scores is None:
            return []
        a, b = self.hand_scores
        teams = [0, 1] if a == b else [0] if a > b else [1]
        return [p for p in range(4) if self.team(p) in teams]

    def all_cards(self) -> list[int]:
        return (self.stock + self.pile + self.pile_pending + [c for h in self.hands for c in h]
                + [c for r in self.red_threes for c in r]
                + [c for team in self.melds for m in team.values() for part in m.values() for c in part])

    # ----- observation -----------------------------------------------------
    def unseen_counts(self, player: int) -> list[int]:
        """Copies of each kind ``player`` cannot locate (in the stock or unknown hand slots)."""
        located = Counter(kind_of(c) for c in self.hands[player])
        located.update(kind_of(c) for c in self.pile + self.pile_pending)
        located.update(kind_of(c) for r in self.red_threes for c in r)
        located.update(kind_of(c) for team in self.melds for m in team.values() for part in m.values() for c in part)
        for q in range(4):
            if q != player:
                located.update(self.known[q])
        return [max(0, COPIES[k] - located[k]) for k in range(KINDS)]

    def discard_danger(self, player: int, unseen: list[int]) -> list[float]:
        """For each kind: the chance the next player (an opponent) could take the pile if
        ``player`` discarded it now."""
        nxt = (player + 1) % 4
        team = self.team(nxt)
        pool = sum(unseen)
        slots = max(0, len(self.hands[nxt]) - sum(self.known[nxt].values()))
        danger = []
        for kind in range(KINDS):
            if kind in (JOKER, TWO, RED3, BLACK3):
                danger.append(0.0)
                continue
            frozen = self.pile_frozen or not self.team_melded[team]
            if not frozen and kind in self.melds[team]:
                danger.append(1.0)
                continue
            known_nat = self.known[nxt][kind]
            pair = p_at_least(pool, unseen[kind], slots, max(0, 2 - known_nat))
            if frozen:
                danger.append(pair)
                continue
            wild_known = self.known[nxt][JOKER] + self.known[nxt][TWO]
            one = p_at_least(pool, unseen[kind], slots, max(0, 1 - known_nat))
            wild = p_at_least(pool, unseen[JOKER] + unseen[TWO], slots, max(0, 1 - wild_known))
            danger.append(1 - (1 - pair) * (1 - one * wild))
        return danger

    def observation(self, player: int) -> np.ndarray:
        rel = [(player + i) % 4 for i in range(4)]
        team, other = self.team(player), 1 - self.team(player)
        hand = Counter(kind_of(c) for c in self.hands[player])

        def melds(t: int) -> list[float]:
            out = []
            for kind in TARGETS:
                m = self.melds[t].get(kind)
                if m is None:
                    out += [0.0, 0.0, 0.0, 0.0]
                else:
                    size = self.meld_size(m)
                    out += [len(m["nat"]) / 8, self.meld_wilds(m) / 3, float(size >= CANASTA_SIZE),
                            float(size >= CANASTA_SIZE and not self.meld_wilds(m))]
            return out

        top = np.zeros(KINDS, np.float32)
        if self.pile:
            top[self.top_kind()] = 1.0
        pile_kinds = Counter(kind_of(c) for c in self.pile)
        unseen = self.unseen_counts(player)
        known = [min(1.0, self.known[q][k] / 8) for q in rel[1:] for k in range(KINDS)]
        reds = [sum(len(self.red_threes[p]) for p in self.team_seats(t)) / 4 for t in (team, other)]
        phase = [float(self.phase == "draw"), float(self.phase == "play"),
                 float(not self.team_melded[team] and (self.turn_value > 0 or bool(self.pile_pending)))]
        vector = [
            *[min(1.0, hand[k] / 8) for k in range(KINDS)],
            *melds(team), *melds(other), *reds,
            min(1.0, len(self.pile) / 60), *top, float(self.pile_frozen), float(self.top_kind() == BLACK3),
            *[min(1.0, pile_kinds[k] / 8) for k in range(KINDS)],
            *[min(1.0, len(self.hands[q]) / 40) for q in rel],
            len(self.stock) / NUM_CARDS,
            float(self.team_melded[team]), float(self.team_melded[other]),
            self.minimum(team) / 120, self.minimum(other) / 120,
            min(1.0, len(self.canastas(team)) / 6), min(1.0, len(self.canastas(other)) / 6),
            float(np.clip(self.team_scores[team] / MATCH_TARGET, 0, 1)), float(np.clip(self.team_scores[other] / MATCH_TARGET, 0, 1)),
            *phase, min(1.0, self.turn_value / 120), min(1.0, len(self.pile_pending) / 60),
            # Card counting: unseen copies, what others are known to hold, and discard danger.
            *[u / 8 for u in unseen], *known, *self.discard_danger(player, unseen),
        ]
        return np.asarray(vector, dtype=np.float32)

    # ----- presentation ----------------------------------------------------
    def action_label(self, action: int) -> str:
        player = self.turn if not self.done else 0
        team = self.team(player)
        if action == DRAW:
            return "Draw from the stock"
        top = self.top_kind()
        if action == TAKE_PAIR:
            return f"Take the pile (pair of {kind_plural(top)})" if top is not None else "Take the pile"
        if action == TAKE_MIXED:
            return f"Take the pile ({kind_label(top)} + wild)" if top is not None else "Take the pile"
        if action == TAKE_ADD:
            return f"Take the pile (add to your {kind_plural(top)})" if top is not None else "Take the pile"
        if action < DISCARD_BASE:
            kind, n, j, d = decode_meld(action)
            parts = [f"{n} × {kind_label(kind)}" if n else None, f"{j} joker{'s' * (j > 1)}" if j else None,
                     f"{d} × 2" if d else None]
            verb = "Add" if kind in self.melds[team] else "Meld"
            return f"{verb} {' + '.join(p for p in parts if p)}" + (f" to {kind_plural(kind)}" if verb == "Add" and not n else "")
        return f"Discard {kind_label(action - DISCARD_BASE)}"

    def view(self, viewer: int | None) -> dict[str, Any]:
        legal = self.legal_actions() if viewer is not None and viewer == self.current_player else []

        def meld_view(kind: int, m: dict[str, list[int]]) -> dict[str, Any]:
            size = self.meld_size(m)
            canasta = None if size < CANASTA_SIZE else ("natural" if not self.meld_wilds(m) else "mixed")
            return {"kind": kind, "label": kind_label(kind), "plural": kind_plural(kind),
                    "cards": [card_json(c) for c in sorted(m["nat"]) + m["two"] + m["jok"]],
                    "naturals": len(m["nat"]), "wilds": self.meld_wilds(m), "size": size, "canasta": canasta}

        def sort_key(card: int) -> tuple[int, int]:
            kind = kind_of(card)
            return (kind if kind >= 3 else 20 + kind, card)

        return {
            "game": self.game_id,
            "done": self.done,
            "phase": "complete" if self.done else self.phase,
            "current_player": self.current_player,
            "dealer": self.dealer,
            "hands": [[card_json(c) for c in sorted(h, key=sort_key)] if p == viewer or self.done else len(h)
                      for p, h in enumerate(self.hands)],
            "hand_counts": [len(h) for h in self.hands],
            "melds": [[meld_view(k, m) for k, m in sorted(self.melds[t].items(), key=lambda kv: -kv[0])] for t in (0, 1)],
            "red_threes": [[card_json(c) for c in r] for r in self.red_threes],
            "pile_top": card_json(self.pile[-1]) if self.pile else None,
            "pile_count": len(self.pile),
            "pile_frozen": self.pile_frozen,
            "pile_blocked": self.top_kind() == BLACK3,
            "pile_pending": len(self.pile_pending),
            "stock_count": len(self.stock),
            "team_melded": list(self.team_melded),
            "minimums": [self.minimum(0), self.minimum(1)],
            "turn_value": self.turn_value,
            "canastas": [len(self.canastas(0)), len(self.canastas(1))],
            "team_scores": list(self.team_scores),
            "team_names": list(TEAM_NAMES),
            "hand_scores": self.hand_scores,
            "went_out": self.went_out,
            "concealed": self.concealed,
            "last_event": self.last_event,
            "legal_actions": [self._action_json(a) for a in legal],
        }

    def _action_json(self, action: int) -> dict[str, Any]:
        """A legal action for the UI, with what it does in structured form."""
        data: dict[str, Any] = {"action": action, "label": self.action_label(action)}
        if action in (TAKE_PAIR, TAKE_MIXED, TAKE_ADD):
            data["type"] = "take"
        elif action == DRAW:
            data["type"] = "draw"
        elif action < DISCARD_BASE:
            kind, n, j, d = decode_meld(action)
            data.update(type="meld", kind=kind, naturals=n, jokers=j, twos=d,
                        adds=kind in self.melds[self.team(self.turn)])
        else:
            data.update(type="discard", kind=action - DISCARD_BASE)
        return data
