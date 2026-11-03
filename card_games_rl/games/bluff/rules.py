"""Bluff engine, 4 players, Indian rules (docs/ENV_DESIGN.md §6, docs/GAMES.md §5).

A round starts with the starter declaring any rank (they need not hold it) and
playing 1-4 cards face down. Each following player either adds 1-4 cards claiming
the same rank, passes, or (right after a claim) calls bluff on it:

* call, claim true  -> the caller takes the pile; the claimant starts the next round
* call, claim false -> the claimant takes the pile; the caller starts the next round
* everyone else passes after a claim -> the pile is discarded (out of play) and the
  player after the previous round's starter starts the next round

A player wins by emptying their hand and surviving the next player's decision.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from card_games_rl.core import Game, IllegalAction
from card_games_rl.core.cards import card_json, new_deck, one_hot_index, rank_label, rank_of

# (true cards, bluff cards) for every play of 1..4 cards, ordered by size then t.
PLAY_COMBOS = [(t, n - t) for n in range(1, 5) for t in range(n + 1)]
CALL, PASS = len(PLAY_COMBOS), len(PLAY_COMBOS) + 1
DECLARE = PASS + 1             # + rank index (13): the starter names the round's rank
NUM_ACTIONS = DECLARE + 13
RANKS = list(range(2, 15))     # 2..A (Ace = 14)
MAX_ACTIONS = 1000
PHASES = ("declare", "lead", "respond")


def rank_index(rank: int) -> int:
    """0..12 in rank order 2..A, used for actions and observation vectors."""
    return rank - 2


class BluffGame(Game):
    game_id = "bluff"
    name = "Bluff"
    num_players = 4
    num_actions = NUM_ACTIONS
    obs_size = 129
    OBS_VERSION = 2
    zero_sum = True
    shaping_weight = 1.0

    def _deal(self) -> None:
        deck = new_deck(self.rng)
        self.hands = [sorted(deck[i * 13:(i + 1) * 13]) for i in range(4)]
        self.round_starter = self.rng.randrange(4)
        self.turn = self.round_starter
        self.phase = "declare"
        self.round_rank: int | None = None
        self.pile: list[int] = []
        self.claims: list[tuple[int, list[int]]] = []   # this round: (claimant, cards)
        self.can_call = False                           # the previous action was a claim
        self.passes = 0                                 # consecutive passes since the last claim
        self.discarded = 0
        self.contrib = [[] for _ in range(4)]           # each seat's own cards in the pile
        self.own_discarded = [[] for _ in range(4)]     # each seat's own cards discarded
        self.known: list[list[set[int]]] = [[set() for _ in range(4)] for _ in range(4)]  # known[p][q]
        # Public behaviour statistics per seat, for opponent modelling.
        self.stats = [{"call_chances": 0, "calls": 0, "called": 0, "caught": 0, "claims": 0} for _ in range(4)]
        self.reveal: dict[str, Any] | None = None
        self.last_event = f"P{self.round_starter + 1} starts"
        self.actions_taken = 0
        self.winner: int | None = None

    # ----- helpers ---------------------------------------------------------
    def count_rank(self, player: int, rank: int) -> int:
        return sum(rank_of(c) == rank for c in self.hands[player])

    @property
    def last_claim(self) -> tuple[int, list[int]] | None:
        return self.claims[-1] if self.claims else None

    def _expendable(self, player: int, rank: int) -> list[int]:
        """Cards to bluff with, most expendable first: ranks held in the fewest copies."""
        counts = Counter(rank_of(c) for c in self.hands[player])
        others = [c for c in self.hands[player] if rank_of(c) != rank]
        return sorted(others, key=lambda c: (counts[rank_of(c)], rank_of(c), c))

    def cards_for(self, player: int, combo: int, rank: int | None = None) -> list[int]:
        rank = self.round_rank if rank is None else rank
        t, b = PLAY_COMBOS[combo]
        true_cards = [c for c in self.hands[player] if rank_of(c) == rank][:t]
        return true_cards + self._expendable(player, rank)[:b]

    def max_true(self, player: int, rank: int | None = None) -> int:
        """Most cards of ``rank`` the other players could truly hold, from certain knowledge
        only: the player's own hand, own cards in the pile, and own discarded cards."""
        rank = self.round_rank if rank is None else rank
        if rank is None:
            return 4
        mine = (self.count_rank(player, rank) + sum(rank_of(c) == rank for c in self.contrib[player])
                + sum(rank_of(c) == rank for c in self.own_discarded[player]))
        return max(0, 4 - mine)

    def claimed_by_others(self, player: int) -> int:
        return sum(len(cards) for claimant, cards in self.claims if claimant != player)

    def certain_lie(self, player: int) -> bool:
        """The last claim cannot be true given what ``player`` holds or has played."""
        claim = self.last_claim
        return bool(claim) and claim[0] != player and len(claim[1]) > self.max_true(player)

    # ----- contract --------------------------------------------------------
    @property
    def current_player(self) -> int:
        return -1 if self.done else self.turn

    def _combos(self, player: int, rank: int) -> list[int]:
        hand = self.hands[player]
        true_count = self.count_rank(player, rank)
        other_count = len(hand) - true_count
        return [i for i, (t, b) in enumerate(PLAY_COMBOS) if t <= true_count and b <= other_count]

    def _legal_actions(self) -> list[int]:
        if self.phase == "declare":
            return [DECLARE + rank_index(r) for r in RANKS]
        if self.phase == "lead":
            return self._combos(self.turn, self.round_rank)
        claimant = self.last_claim[0] if self.last_claim else None
        if self.can_call and claimant is not None and not self.hands[claimant]:
            return [CALL, PASS]  # the claimant is out of cards: call it or they win
        return self._combos(self.turn, self.round_rank) + ([CALL] if self.can_call else []) + [PASS]

    def can_play_cards(self) -> bool:
        return not self.done and (self.phase == "declare" or any(a < CALL for a in self.legal_actions()))

    def _apply(self, action: int) -> None:
        player = self.turn
        if action >= DECLARE:
            self.round_rank = RANKS[action - DECLARE]
            self.phase = "lead"
        elif action == CALL:
            self._call(player)
        elif action == PASS:
            self._pass(player)
        else:
            self._play(player, self.cards_for(player, action))
        self._tick()

    def step_cards(self, played: list[int], rank: int | None = None) -> None:
        """Human play: exactly these cards, face down. ``rank`` names the round's rank
        when starting a round; later plays must claim the round's rank."""
        played = [int(c) for c in played]
        player = self.turn
        if self.done or (self.phase == "respond" and PASS not in self.legal_actions()) \
                or (self.phase == "respond" and self.legal_actions() == [CALL, PASS]):
            raise IllegalAction("You can't play cards now")
        hand = self.hands[player]
        if not 1 <= len(played) <= 4 or len(set(played)) != len(played) or any(c not in hand for c in played):
            raise IllegalAction("Play 1-4 distinct cards from your hand")
        if self.phase == "declare":
            if rank not in RANKS:
                raise IllegalAction("Name the rank you are claiming")
            self.history.append((player, DECLARE + rank_index(rank)))
            self.round_rank = rank
            self.phase = "lead"
        elif rank is not None and rank != self.round_rank:
            raise IllegalAction(f"This round's rank is {rank_label(self.round_rank)}")
        t = sum(rank_of(c) == self.round_rank for c in played)
        self.history.append((player, PLAY_COMBOS.index((t, len(played) - t))))
        self.invalidate()
        self._play(player, played)
        self._tick()

    def _tick(self) -> None:
        self.actions_taken += 1
        if not self.done and self.actions_taken >= MAX_ACTIONS:
            self.is_truncated = True

    def _next(self, seat: int) -> int:
        return (seat + 1) % 4

    def _claimant_survives(self) -> bool:
        """A claimant with an empty hand wins once the next player declines to call."""
        claim = self.last_claim
        if self.can_call and claim and not self.hands[claim[0]]:
            self.winner = claim[0]
            self.is_terminal = True
            self.last_event = f"P{claim[0] + 1} played their last card and nobody called"
            return True
        return False

    def _play(self, player: int, played: list[int]) -> None:
        if self.phase == "respond" and self._claimant_survives():
            return
        if self.can_call:
            self.stats[player]["call_chances"] += 1
        for card in played:
            self.hands[player].remove(card)
        self.pile.extend(played)
        self.contrib[player].extend(played)
        # Opponents assume honest play: known cards of the claimed rank leave the hand.
        for p in range(4):
            known = self.known[p][player]
            for card in sorted(c for c in known if rank_of(c) == self.round_rank)[:len(played)]:
                known.discard(card)
        self.claims.append((player, played))
        self.stats[player]["claims"] += 1
        self.can_call, self.passes = True, 0
        self.phase = "respond"
        self.reveal = None
        self.last_event = f"P{player + 1} claims {len(played)} × {rank_label(self.round_rank)}"
        self.turn = self._next(player)

    def _pass(self, player: int) -> None:
        if self._claimant_survives():
            return
        if self.can_call:
            self.stats[player]["call_chances"] += 1
        self.can_call = False
        self.passes += 1
        self.last_event = f"P{player + 1} passes"
        if self.passes >= 3:  # everyone else passed: the pile leaves the game
            for p in range(4):
                self.own_discarded[p].extend(self.contrib[p])
            self.discarded += len(self.pile)
            self.last_event = f"Everyone passed: {len(self.pile)} cards discarded"
            self._new_round(self._next(self.round_starter))
        else:
            self.turn = self._next(player)

    def _call(self, caller: int) -> None:
        claimant, played = self.last_claim
        truthful = all(rank_of(c) == self.round_rank for c in played)
        loser = caller if truthful else claimant
        self.stats[caller]["call_chances"] += 1
        self.stats[caller]["calls"] += 1
        self.stats[claimant]["called"] += 1
        self.stats[claimant]["caught"] += int(not truthful)
        for p in range(4):
            if p != loser:  # everyone knows the revealed cards and their own pile cards went to the loser
                self.known[p][loser] |= set(self.contrib[p]) | set(played)
        self.hands[loser] = sorted(self.hands[loser] + self.pile)
        self.reveal = {"caller": caller, "claimant": claimant, "cards": list(played), "truthful": truthful,
                       "loser": loser, "pile": len(self.pile)}
        self.last_event = (f"P{caller + 1} called P{claimant + 1}: "
                           f"{'true' if truthful else 'a lie'}, P{loser + 1} takes {len(self.pile)} cards")
        self._new_round(claimant if truthful else caller)
        if truthful and not self.hands[claimant]:
            self.winner = claimant
            self.is_terminal = True

    def _new_round(self, starter: int) -> None:
        self.pile = []
        self.contrib = [[] for _ in range(4)]
        self.claims = []
        self.round_rank = None
        self.can_call, self.passes = False, 0
        self.round_starter = self.turn = starter
        self.phase = "declare"

    # ----- rewards ---------------------------------------------------------
    def score(self, player: int) -> float:
        if self.is_terminal:
            return 1.0 if player == self.winner else -1 / 3
        if self.is_truncated:
            sizes = [len(h) for h in self.hands]
            return (sum(sizes) / 4 - sizes[player]) / 52
        return 0.0

    def potential(self, player: int) -> float:
        sizes = [len(h) for h in self.hands]
        opponents = (sum(sizes) - sizes[player]) / 3
        return 0.5 * (opponents - sizes[player]) / 52

    def all_cards(self) -> list[int]:
        # Discarded cards are out of play; track them through the seats' own records.
        out = [c for cards in self.own_discarded for c in cards]
        return [c for h in self.hands for c in h] + self.pile + out

    # ----- observation -----------------------------------------------------
    def observation(self, player: int) -> np.ndarray:
        def rank_counts(cards) -> np.ndarray:
            vector = np.zeros(13, dtype=np.float32)
            for card in cards:
                vector[rank_index(rank_of(card))] += 0.25
            return vector

        rel = [(player + i) % 4 for i in range(4)]
        claim = self.last_claim
        rank = self.round_rank
        rank_onehot = one_hot_index(rank_index(rank), 13) if rank else np.zeros(13, np.float32)
        own_rank = self.count_rank(player, rank) / 4 if rank else 0.0
        max_true = self.max_true(player)
        others_claimed = self.claimed_by_others(player)

        def rate(numerator: int, denominator: int) -> float:
            return (numerator + 1) / (denominator + 2)  # smoothed, 0.5 with no evidence

        behaviour = []
        for q in rel[1:]:
            s = self.stats[q]
            behaviour += [rate(s["calls"], s["call_chances"]), rate(s["caught"], s["called"]), min(s["claims"], 50) / 50]
        return np.concatenate([
            rank_counts(self.hands[player]),
            one_hot_index(PHASES.index(self.phase), 3),
            rank_onehot,
            np.asarray([own_rank], np.float32),
            np.asarray([len(self.hands[q]) / 52 for q in rel], np.float32),
            np.asarray([len(self.pile) / 52, self.discarded / 52, min(len(self.claims) / 8, 1.0),
                        min(sum(len(c) for _, c in self.claims) / 16, 1.0), min(others_claimed / 16, 1.0)], np.float32),
            one_hot_index(len(claim[1]) - 1, 4) if claim else np.zeros(4, np.float32),
            one_hot_index((claim[0] - player) % 4, 4) if claim else np.zeros(4, np.float32),
            np.asarray([float(self.can_call), max_true / 4, float(self.certain_lie(player)),
                        min(max(others_claimed - max_true, 0), 4) / 4, self.passes / 3], np.float32),
            rank_counts(self.contrib[player]),
            rank_counts(self.own_discarded[player]),
            *[rank_counts(self.known[player][q]) for q in rel[1:]],
            np.asarray(behaviour, np.float32),
            np.asarray([float(bool(claim) and not self.hands[claim[0]]), self.actions_taken / MAX_ACTIONS,
                        float(self.round_starter == player)], np.float32),
        ]).astype(np.float32)

    def action_label(self, action: int) -> str:
        if action == CALL:
            return "Call bluff"
        if action == PASS:
            return "Pass"
        if action >= DECLARE:
            return f"Start a round of {rank_label(RANKS[action - DECLARE])}s"
        t, b = PLAY_COMBOS[action]
        verb = "Claim" if self.phase == "lead" else "Add"
        return f"{verb} {t + b} × {rank_label(self.round_rank)} ({t} true, {b} bluff)"

    def public_label(self, action: int) -> str:
        if action < CALL:
            return self.claim_label(sum(PLAY_COMBOS[action]))
        return self.action_label(action)

    def claim_label(self, count: int, rank: int | None = None) -> str:
        verb = "Add" if self.phase == "respond" else "Claim"
        return f"{verb} {count} × {rank_label(rank or self.round_rank)}"

    def view(self, viewer: int | None) -> dict[str, Any]:
        legal = self.legal_actions() if viewer is not None and viewer == self.current_player else []
        claim = self.last_claim
        return {
            "game": self.game_id,
            "done": self.done,
            "phase": "complete" if self.done else self.phase,
            "current_player": self.current_player,
            "turn": self.turn,
            "round_starter": self.round_starter,
            "round_rank": rank_label(self.round_rank) if self.round_rank else None,
            "ranks": [{"rank": r, "label": rank_label(r)} for r in RANKS],
            "hands": [[card_json(c) for c in h] if p == viewer or self.done else len(h) for p, h in enumerate(self.hands)],
            "pile_count": len(self.pile),
            "discarded": self.discarded,
            "claims": [{"player": p, "count": len(cards)} for p, cards in self.claims],
            "last_claim": {"claimant": claim[0], "count": len(claim[1])} if claim else None,
            "can_call": self.can_call,
            "reveal": {**self.reveal, "cards": [card_json(c) for c in self.reveal["cards"]]} if self.reveal else None,
            "last_event": self.last_event,
            "winner": self.winner,
            "actions_taken": self.actions_taken,
            "legal_actions": [{"action": a, "label": self.action_label(a)} for a in legal],
            "can_play_exact": bool(legal) and self.can_play_cards(),
        }
