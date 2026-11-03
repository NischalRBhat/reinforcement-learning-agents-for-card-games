"""Perfect-information Klondike solver ("thoughtful solitaire").

Answers "was this deal winnable?" for reports and the UI. It sees the face-down
cards, so it is never used to choose moves during play.

With draw-1 and unlimited recycles, every stock/waste card can be reached by
drawing, and recycling keeps their order. The stock and waste therefore form one
ring in draw order, and any ring card can be played directly; this removes the
long draw chains from the search. The search is depth-first with a table of
positions already explored and a node budget.
"""
from __future__ import annotations

from dataclasses import dataclass

from card_games_rl.core.cards import is_red, suit_of

from .rules import KlondikeGame, card_for, krank

WON, NOT_WINNABLE, UNKNOWN = "winnable", "not winnable", "unknown"


@dataclass
class SolveResult:
    verdict: str          # winnable | not winnable | unknown (budget exhausted)
    nodes: int
    moves: int | None = None  # length of the solution found (macro moves)


def _can_place(card: int, pile: tuple[int, ...]) -> bool:
    if not pile:
        return krank(card) == 13
    top = pile[-1]
    return krank(top) == krank(card) + 1 and is_red(top) != is_red(card)


def _safe(card: int, foundation: tuple[int, ...]) -> bool:
    k = krank(card)
    if k <= 2:
        return True
    opposite = [s for s in range(4) if (s in (1, 2)) != is_red(card)]
    return all(foundation[s] >= k - 1 for s in opposite)


class Solver:
    def __init__(self, max_nodes: int = 200_000):
        self.max_nodes = max_nodes

    def solve(self, game: KlondikeGame) -> SolveResult:
        tableau = tuple(tuple(p) for p in game.tableau)
        hidden = tuple(game.hidden)
        ring = tuple(game.waste) + tuple(reversed(game.stock))
        start = self._settle(list(map(list, tableau)), list(hidden), list(game.foundation), list(ring))
        self.nodes = 0
        self.seen: set = set()
        stack = [(start, 0)]
        while stack:
            state, depth = stack.pop()
            tableau, hidden, foundation, ring = state
            if sum(foundation) == 52:
                return SolveResult(WON, self.nodes, depth)
            key = (tuple(sorted(zip(hidden, tableau))), foundation, ring)
            if key in self.seen:
                continue
            self.seen.add(key)
            self.nodes += 1
            if self.nodes >= self.max_nodes:
                return SolveResult(UNKNOWN, self.nodes)
            # Push in reverse priority so the most promising move is explored first.
            for child in reversed(self._children(state)):
                stack.append((child, depth + 1))
        return SolveResult(NOT_WINNABLE, self.nodes)

    # ----- move generation -------------------------------------------------
    def _settle(self, tableau, hidden, foundation, ring):
        """Flip exposed cards and play safe cards to the foundations."""
        moved = True
        while moved:
            moved = False
            for c, pile in enumerate(tableau):
                if pile and hidden[c] == len(pile):
                    hidden[c] -= 1
                if pile and hidden[c] < len(pile):
                    card = pile[-1]
                    if foundation[suit_of(card)] == krank(card) - 1 and _safe(card, foundation):
                        pile.pop()
                        foundation[suit_of(card)] += 1
                        moved = True
            for i, card in enumerate(ring):
                if foundation[suit_of(card)] == krank(card) - 1 and _safe(card, foundation):
                    ring.pop(i)
                    foundation[suit_of(card)] += 1
                    moved = True
                    break
            for c, pile in enumerate(tableau):
                if pile and hidden[c] == len(pile):
                    hidden[c] -= 1
        return tuple(tuple(p) for p in tableau), tuple(hidden), tuple(foundation), tuple(ring)

    def _children(self, state):
        tableau, hidden, foundation, ring = state
        found, reveal, ring_moves, other, back = [], [], [], [], []

        def make(new_tableau=None, new_hidden=None, new_foundation=None, new_ring=None):
            return self._settle([list(p) for p in (new_tableau or tableau)], list(new_hidden or hidden),
                                list(new_foundation or foundation), list(new_ring if new_ring is not None else ring))

        # tableau top -> foundation
        for c, pile in enumerate(tableau):
            if pile and hidden[c] < len(pile):
                card = pile[-1]
                if foundation[suit_of(card)] == krank(card) - 1:
                    t = list(tableau); t[c] = pile[:-1]
                    f = list(foundation); f[suit_of(card)] += 1
                    found.append(make(t, None, f))
        # ring card -> foundation / tableau
        for i, card in enumerate(ring):
            rest = ring[:i] + ring[i + 1:]
            if foundation[suit_of(card)] == krank(card) - 1:
                f = list(foundation); f[suit_of(card)] += 1
                found.append(make(None, None, f, rest))
            for c, pile in enumerate(tableau):
                if _can_place(card, pile):
                    t = list(tableau); t[c] = pile + (card,)
                    ring_moves.append(make(t, None, None, rest))
        # tableau runs -> tableau. Splitting a run part-way only helps if the card it
        # uncovers can then go to a foundation or take another card; other splits are
        # pruned (standard dominance pruning, which makes "no win found" not a proof).
        heads = set(ring) | {p[h] for p, h in zip(tableau, hidden) if h < len(p)}
        for src, pile in enumerate(tableau):
            for start in range(hidden[src], len(pile)):
                card = pile[start]
                if start > hidden[src]:
                    under = pile[start - 1]
                    useful = foundation[suit_of(under)] == krank(under) - 1 or any(
                        h != card and krank(h) == krank(under) - 1 and is_red(h) != is_red(under) for h in heads)
                    if not useful:
                        continue
                for dst, target in enumerate(tableau):
                    if dst == src or not _can_place(card, target):
                        continue
                    if start == 0 and not target:
                        continue  # whole column into an empty column
                    t = list(tableau); t[src] = pile[:start]; t[dst] = target + pile[start:]
                    (reveal if start == hidden[src] and start > 0 else other).append(make(t))
        # foundation -> tableau
        for s in range(4):
            if foundation[s]:
                card = card_for(s, foundation[s])
                for c, pile in enumerate(tableau):
                    if _can_place(card, pile):
                        t = list(tableau); t[c] = pile + (card,)
                        f = list(foundation); f[s] -= 1
                        back.append(make(t, None, f))
        return found + reveal + ring_moves + other + back


def solve(game: KlondikeGame, max_nodes: int = 200_000) -> SolveResult:
    return Solver(max_nodes).solve(game)
