"""Counterfactual regret minimisation for heads-up hold'em (docs/GAMES.md §3).

**Abstraction** (what the strategy can tell apart):

* *Cards:* a strength bucket per street, from the hand's equity against a random
  hand (preflop from the 169-class table, later streets by Monte Carlo).
* *Betting:* the street, this street's action sequence, the pot size at the start
  of the street (bucketed) and whether the player has the button. Earlier streets'
  details are forgotten (imperfect recall, standard for poker abstractions).
* *Actions:* the engine's own sizes. To keep the tree small, training uses fold,
  check/call, ½-pot, pot and all-in, with at most ``MAX_RAISES`` raises per street.

**Training** is external-sampling Monte Carlo CFR with regret matching+ and
linearly weighted averaging. Several worker processes train independently; their
average strategies are summed. In a two-player zero-sum game a mix of
equilibrium strategies is still an equilibrium strategy.

At play time, information sets the training never reached fall back to the
System bot's equity heuristic.
"""
from __future__ import annotations

import gzip
import pickle
import random
import time
from pathlib import Path

import numpy as np

from card_games_rl.core import Game, Policy
from card_games_rl.core.policy import sample_legal

from .bots import HeuristicPolicy, equity_vs_random, preflop_equity
from .rules import ALL_IN, CALL, FOLD, RAISE_2POT, RAISE_HALF, RAISE_POT, PokerGame

BUCKETS = 10
EQUITY_SAMPLES = 64
MAX_RAISES = 3
TRAIN_ACTIONS = (FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN)
POT_EDGES = (4, 8, 16, 32, 64, 128)
NUM_ACTIONS = 6
ABSTRACTION_VERSION = 1


def bucket_of(equity: float) -> int:
    return min(BUCKETS - 1, int(equity * BUCKETS))


def preflop_bucket(hole: list[int]) -> int:
    # Preflop equities span ~0.30-0.85; stretch them over the buckets.
    return min(BUCKETS - 1, max(0, int((preflop_equity(hole) - 0.30) / 0.55 * BUCKETS)))


def street_buckets(hole: list[int], board_cards: list[int], rng: random.Random) -> list[int]:
    """Strength bucket on each street (preflop, flop, turn, river)."""
    return [preflop_bucket(hole)] + [bucket_of(equity_vs_random(hole, board_cards[:n], rng, EQUITY_SAMPLES))
                                     for n in (3, 4, 5)]


def pot_bucket(chips: int) -> int:
    return sum(chips > edge for edge in POT_EDGES)


class Tracker:
    """Per-street action sequences and the pot at each street's start, for info-set keys."""

    __slots__ = ("street", "actions", "street_pot")

    def __init__(self, street: int = 0, actions: str = "", street_pot: int = 3):
        self.street, self.actions, self.street_pot = street, actions, street_pot

    def after(self, game: PokerGame, action: int, before_street: int) -> "Tracker":
        if game.done or game.street == before_street:
            return Tracker(self.street, self.actions + str(action), self.street_pot)
        return Tracker(game.street, "", game.pot)


def info_key(game: PokerGame, player: int, bucket: int, tracker: Tracker) -> str:
    return f"{tracker.street}|{bucket}|{int(game.button == player)}|{pot_bucket(tracker.street_pot)}|{tracker.actions}"


def train_legal(game: PokerGame) -> list[int]:
    legal = game.legal_actions()
    if game.raises[game.street][0] + game.raises[game.street][1] >= MAX_RAISES:
        return [a for a in legal if a in (FOLD, CALL, ALL_IN)]
    return [a for a in legal if a in TRAIN_ACTIONS]


class MCCFR:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.regrets: dict[str, np.ndarray] = {}
        self.strategy_sum: dict[str, np.ndarray] = {}
        self.iterations = 0

    @staticmethod
    def _match(regret: np.ndarray, legal: list[int]) -> np.ndarray:
        strategy = np.zeros(NUM_ACTIONS)
        positive = np.maximum(regret[legal], 0.0)
        total = positive.sum()
        strategy[legal] = positive / total if total > 0 else 1.0 / len(legal)
        return strategy

    def iterate(self) -> None:
        seed = self.rng.randrange(2**62)
        game = PokerGame(seed=seed)  # the sampled deal fixes the board: real showdowns, no all-in EV
        game.reset(seed)
        buckets = [street_buckets(game.hole[p], game.board_cards, self.rng) for p in range(2)]
        self.iterations += 1
        for traverser in (0, 1):
            self._traverse(game.clone(), traverser, buckets, Tracker())

    def _traverse(self, game: PokerGame, traverser: int, buckets, tracker: Tracker) -> float:
        if game.done:
            return game.score(traverser)
        player = game.current_player
        legal = train_legal(game)
        key = info_key(game, player, buckets[player][tracker.street], tracker)
        regret = self.regrets.get(key)
        if regret is None:
            regret = self.regrets[key] = np.zeros(NUM_ACTIONS)
        strategy = self._match(regret, legal)
        if player == traverser:
            values = np.zeros(NUM_ACTIONS)
            for action in legal:
                child = game.clone()
                street = child.street
                child.step(action)
                values[action] = self._traverse(child, traverser, buckets, tracker.after(child, action, street))
            node = float(strategy @ values)
            regret[legal] = np.maximum(regret[legal] + values[legal] - node, 0.0)  # regret matching+
            return node
        # Opponent node: accumulate the (linearly weighted) average strategy, sample one action.
        total = self.strategy_sum.get(key)
        if total is None:
            total = self.strategy_sum[key] = np.zeros(NUM_ACTIONS)
        total += self.iterations * strategy
        action = sample_legal(strategy, self.rng)
        street = game.street
        game.step(action)
        return self._traverse(game, traverser, buckets, tracker.after(game, action, street))

    # ----- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        tmp = path.with_name("_tmp_" + path.name)
        with gzip.open(tmp, "wb", compresslevel=3) as f:
            pickle.dump({"regrets": self.regrets, "strategy_sum": self.strategy_sum,
                         "iterations": self.iterations}, f, protocol=pickle.HIGHEST_PROTOCOL)
        for attempt in range(20):  # the trainer may be reading the old file (Windows locks)
            try:
                tmp.replace(path)
                return
            except PermissionError:
                time.sleep(0.5)

    @classmethod
    def load(cls, path: Path, seed: int = 0) -> "MCCFR":
        solver = cls(seed)
        data = _read(path)
        solver.regrets, solver.strategy_sum, solver.iterations = data["regrets"], data["strategy_sum"], data["iterations"]
        return solver


def _read(path: Path) -> dict:
    for attempt in range(20):  # a worker may be replacing the file right now (Windows locks)
        try:
            with gzip.open(path, "rb") as f:
                return pickle.load(f)
        except (PermissionError, EOFError, OSError):
            time.sleep(0.5)
    raise RuntimeError(f"could not read {path}")


def merge_strategies(paths: list[Path]) -> tuple[dict[str, np.ndarray], int]:
    """Average strategy per info set over every worker, and the total iterations."""
    totals: dict[str, np.ndarray] = {}
    iterations = 0
    for path in paths:
        data = _read(path)
        iterations += data["iterations"]
        for key, value in data["strategy_sum"].items():
            norm = value.sum()
            if norm > 0:
                totals[key] = totals.get(key, 0) + value / norm
    return {k: (v / v.sum()).astype(np.float32) for k, v in totals.items()}, iterations


def save_strategy(strategy: dict[str, np.ndarray], path: Path, info: dict) -> None:
    tmp = path.with_name("_tmp_" + path.name)
    with gzip.open(tmp, "wb") as f:
        pickle.dump({"strategy": strategy, "abstraction_version": ABSTRACTION_VERSION, **info}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Playing
# ---------------------------------------------------------------------------
class CFRPolicy(Policy):
    """Plays the trained average strategy; unreached situations use the System bot."""

    name = "cfr"

    def __init__(self, path: str | Path | None = None, strategy: dict[str, np.ndarray] | None = None,
                 deterministic: bool = False, seed: int | None = None):
        if strategy is None:
            with gzip.open(path, "rb") as f:
                strategy = pickle.load(f)["strategy"]
        self.strategy = strategy
        self.deterministic = deterministic
        self.rng = random.Random(seed)
        self.fallback = HeuristicPolicy(seed=seed)
        self._buckets: dict[tuple, list[int]] = {}
        self.lookups = self.misses = 0

    def _tracker(self, game: PokerGame) -> Tracker:
        """Rebuild this street's action sequence from the hand's history."""
        replay = PokerGame(seed=0, button=game.button)
        replay.reset(0)
        replay.hole, replay.board_cards = game.hole, game.board_cards
        tracker = Tracker()
        for _, action in game.history:
            street = replay.street
            replay.step(action)
            tracker = tracker.after(replay, action, street)
        return tracker

    def _bucket(self, game: PokerGame, player: int, street: int) -> int:
        key = (tuple(game.hole[player]), tuple(game.board_cards))
        if key not in self._buckets:
            if len(self._buckets) > 256:
                self._buckets.clear()
            self._buckets[key] = street_buckets(game.hole[player], game.board_cards, random.Random(hash(key)))
        return self._buckets[key][street]

    def probabilities(self, game: Game, player: int) -> np.ndarray | None:
        tracker = self._tracker(game)
        key = info_key(game, player, self._bucket(game, player, tracker.street), tracker)
        self.lookups += 1
        entry = self.strategy.get(key)
        legal = game.legal_actions()
        if entry is None:
            self.misses += 1
            return None
        probs = np.zeros(NUM_ACTIONS)
        probs[legal] = entry[legal]
        if probs.sum() <= 0:
            self.misses += 1
            return None
        return probs / probs.sum()

    def act(self, game: Game, player: int) -> int:
        probs = self.probabilities(game, player)
        if probs is None:
            return self.fallback.act(game, player)
        return int(np.argmax(probs)) if self.deterministic else sample_legal(probs, self.rng)

    def action_probabilities(self, game: Game, player: int) -> dict[int, float] | None:
        probs = self.probabilities(game, player)
        return None if probs is None else {int(a): float(probs[a]) for a in np.flatnonzero(probs > 0)}


# ---------------------------------------------------------------------------
# Training workers
# ---------------------------------------------------------------------------
def worker(path: str, seed: int, deadline: float, max_iterations: int, save_every: float) -> None:
    """Run MCCFR until the deadline or iteration cap, saving periodically (resumable)."""
    file = Path(path)
    solver = MCCFR.load(file, seed) if file.exists() else MCCFR(seed)
    last_save = time.time()
    while time.time() < deadline and solver.iterations < max_iterations:
        solver.iterate()
        if time.time() - last_save > save_every:
            solver.save(file)
            last_save = time.time()
    solver.save(file)
