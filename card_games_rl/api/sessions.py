"""Seat-based play sessions shared by every play mode.

A session holds one or two *boards*. A board is one game plus a controller per
seat: ``human``, ``system`` (the heuristic bot), ``agent`` (the trained model) or
``random``. Two boards with the same deal make a *duel*, used for the solo and
dealer games (Blackjack, Klondike) where "human vs agent" means playing the same
cards side by side. All rules live in the engines; this module only routes moves.
"""
from __future__ import annotations

import copy
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from card_games_rl.agents.registry import best_model_path, load_agent, meta_path, model_dir, read_meta
from card_games_rl.core import Game, IllegalAction, Policy
from card_games_rl.core.seeds import VALIDATION_SEEDS
from card_games_rl.games import get_spec, heuristic_policy, make_game, random_policy

CONTROLLERS = ("human", "system", "agent", "random")
MODES = ("human_vs_system", "human_vs_agent", "agent_vs_agent")
ALGORITHM_NAMES = {"MaskablePPO": "Reinforcement learning (PPO)", "cfr": "Counterfactual regret minimisation",
                   "search": "Look-ahead search"}
# The reference result shown next to the agent's: what it should be compared with.
BASELINE_REFERENCE = {"blackjack": "optimal", "klondike": "heuristic"}
BASELINE_LABELS = {"optimal": "Optimal strategy", "heuristic": "System bot", "heuristic_vs_heuristic": "System vs System"}
HEARTS_MATCH_POINTS = 100
MAX_ADVANCE = 2000
SESSION_TTL = 3600.0
MAX_SESSIONS = 200
TOP_K = 3


class SessionError(Exception):
    """A request that is valid JSON but not allowed in the session's state."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Trained agents
# ---------------------------------------------------------------------------
class AgentCache:
    """Loads each game's trained model once and shares it across sessions."""

    def __init__(self, root: Path | None = None, loader: Callable[..., Policy | None] = load_agent):
        self.root = root
        self.loader = loader
        self._agents: dict[str, Policy | None] = {}
        self._reports: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, game_id: str) -> Policy | None:
        with self._lock:
            if game_id not in self._agents:
                self._agents[game_id] = self.loader(game_id, root=self.root)
            return self._agents[game_id]

    def _report(self, game_id: str) -> dict[str, Any] | None:
        """The held-out test report from ``evaluate``, if it was written after the current model."""
        path = model_dir(game_id, self.root) / "eval_report.json"
        meta = meta_path(game_id, self.root)
        if not path.exists() or not meta.exists() or path.stat().st_mtime < meta.stat().st_mtime:
            return None
        stamp = path.stat().st_mtime
        cached = self._reports.get(game_id)
        if cached is None or cached[0] != stamp:
            try:
                cached = self._reports[game_id] = (stamp, json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                return None
        return cached[1]

    def status(self, game_id: str) -> dict[str, Any]:
        """What the UI shows about a game's agent: method, headline result, target and a baseline."""
        meta = read_meta(game_id, self.root) or {}
        report = self._report(game_id)
        result = (report or {}).get("primary") or meta.get("validation") or {}
        gate = (report or {}).get("gate") or meta.get("gate") or {}
        algorithm = meta.get("algorithm", "MaskablePPO") if meta else None
        baseline = None
        if report:
            name = BASELINE_REFERENCE.get(game_id, "heuristic_vs_heuristic")
            reference = report.get("references", {}).get(name)
            if reference:
                baseline = {"label": BASELINE_LABELS[name], "value": reference["value"]}
        elif isinstance(meta.get("reference"), dict) and "value" in meta["reference"]:
            baseline = {"label": BASELINE_LABELS["optimal"], "value": meta["reference"]["value"]}
        return {
            "trained": best_model_path(game_id, self.root) is not None,
            "algorithm": algorithm,
            "algorithm_name": ALGORITHM_NAMES.get(algorithm, algorithm),
            "timesteps": meta.get("timesteps"),
            "updated_at": meta.get("updated_at"),
            "source": "test" if report else "validation" if result else None,
            "episodes": result.get("deals") or result.get("episodes"),
            "metric": result.get("metric"),
            "value": result.get("value"),
            "ci95": result.get("ci95"),
            "higher_is_better": result.get("higher_is_better", True),
            "target": gate.get("threshold"),
            "max_gap": gate.get("max_gap"),
            "gate_passed": gate.get("passed"),
            "baseline": baseline,
        }


# ---------------------------------------------------------------------------
# Boards and sessions
# ---------------------------------------------------------------------------
def default_seats(game_id: str, mode: str) -> list[list[str]]:
    """Controllers per board for a mode (one list per board)."""
    spec = get_spec(game_id)
    n = spec.max_players
    if mode not in MODES:
        raise SessionError(f"Unknown mode {mode!r}; choose from {list(MODES)}", 422)
    if mode == "human_vs_system":
        return [["human"] + ["system"] * (n - 1)]
    if mode == "agent_vs_agent":
        return [["agent"] * n]
    if spec.category in ("solo", "dealer"):  # duel: same deal, side by side
        return [["human"], ["agent"]]
    return [["human"] + ["agent"] * (n - 1)]


@dataclass
class Board:
    game: Game
    seats: list[str]
    policies: list[Policy | None]
    agent_fallback: list[bool]
    moves: list[dict[str, Any]] = field(default_factory=list)
    settled: bool = False
    rounds: int = 0
    totals: list[float] = field(default_factory=list)   # sum of returns (game score units)
    wins: list[int] = field(default_factory=list)
    points: list[float] = field(default_factory=list)   # chips (poker) / match points (hearts)
    undo_stack: list[tuple[Game, int]] = field(default_factory=list)  # (game before a human move, moves length)
    solve_result: dict[str, Any] | None = None

    @property
    def human_seat(self) -> int | None:
        return self.seats.index("human") if "human" in self.seats else None

    @property
    def human_turn(self) -> bool:
        game = self.game
        return not game.done and self.seats[game.current_player] == "human"


class Session:
    def __init__(self, game_id: str, mode: str, agents: AgentCache, seats: list[list[str]] | None = None,
                 seed: int | None = None, reveal: bool | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.spec = get_spec(game_id)
        self.game_id = game_id
        self.mode = mode
        self.agents = agents
        self.base_seed = seed if seed is not None else random.randrange(VALIDATION_SEEDS)
        self.round = 0
        self.match_over = False
        self.lock = threading.Lock()
        self.touched = time.monotonic()
        board_seats = seats or default_seats(game_id, mode)
        self._validate_seats(board_seats)
        # Spectators see every hand by default; a human seat sees only its own.
        self.reveal = reveal if reveal is not None else all("human" not in s for s in board_seats)
        self.boards = [self._make_board(s) for s in board_seats]
        # Undo only where nobody else is affected: a human playing a solo game alone.
        self.undo_allowed = self.spec.category == "solo" and len(self.boards) == 1 and "human" in board_seats[0]

    # ----- setup -----------------------------------------------------------
    def _validate_seats(self, board_seats: list[list[str]]) -> None:
        n = self.spec.max_players
        if not 1 <= len(board_seats) <= 2:
            raise SessionError("A session has one board, or two for a duel", 422)
        if len(board_seats) == 2 and self.spec.category not in ("solo", "dealer"):
            raise SessionError("Duels are only for solo and dealer games", 422)
        for seats in board_seats:
            if len(seats) != n:
                raise SessionError(f"{self.spec.name} needs {n} seat(s), got {len(seats)}", 422)
            bad = [c for c in seats if c not in CONTROLLERS]
            if bad:
                raise SessionError(f"Unknown controller(s) {bad}; choose from {list(CONTROLLERS)}", 422)
            if seats.count("human") > 1:
                raise SessionError("At most one human seat per board", 422)

    def _game_kwargs(self) -> dict[str, Any]:
        # Match play: Hearts rotates the pass direction, Poker alternates the button.
        if self.game_id == "hearts":
            return {"pass_direction": self.round % 4}
        if self.game_id == "poker":
            return {"button": (self.base_seed + self.round) % 2}
        return {}

    def _new_game(self) -> Game:
        return make_game(self.game_id, seed=self.base_seed + self.round, **self._game_kwargs())

    def _policy(self, controller: str, seat: int) -> tuple[Policy | None, bool]:
        """Policy for a seat and whether an untrained agent fell back to the System bot."""
        seed = self.base_seed * 7 + seat
        if controller == "human":
            return None, False
        if controller == "random":
            return random_policy(self.game_id, seed), False
        if controller == "agent":
            agent = self.agents.get(self.game_id)
            if agent is not None:
                return agent, False
        return heuristic_policy(self.game_id, seed), controller == "agent"

    def _make_board(self, seats: list[str]) -> Board:
        policies, fallback = zip(*(self._policy(c, i) for i, c in enumerate(seats)))
        n = len(seats)
        return Board(self._new_game(), list(seats), list(policies), list(fallback),
                     totals=[0.0] * n, wins=[0] * n, points=[0.0] * n)

    # ----- play ------------------------------------------------------------
    def board(self, index: int) -> Board:
        if not 0 <= index < len(self.boards):
            raise SessionError(f"No board {index}", 422)
        return self.boards[index]

    def human_action(self, board_index: int, action: int | None = None, cards: list[int] | None = None,
                     claim_rank: int | None = None) -> None:
        board = self.board(board_index)
        if not board.human_turn:
            raise SessionError("It is not your turn")
        game = board.game
        seat = game.current_player
        if self.undo_allowed:
            board.undo_stack.append((copy.deepcopy(game), len(board.moves)))
            del board.undo_stack[:-200]
        try:
            if cards is not None:
                if not hasattr(game, "step_cards"):
                    raise SessionError(f"{self.spec.name} does not take card lists", 422)
                label = game.claim_label(len(cards), claim_rank)
                game.step_cards(cards, **({"rank": claim_rank} if claim_rank is not None else {}))
                board.moves.append({"seat": seat, "controller": "human", "action": None, "label": label})
            elif action is not None:
                label = game.action_label(action) if 0 <= action < game.num_actions else str(action)
                game.step(action)
                board.moves.append({"seat": seat, "controller": "human", "action": action, "label": label})
            else:
                raise SessionError("Send an action or a card list", 422)
        except IllegalAction as exc:
            if self.undo_allowed:
                board.undo_stack.pop()
            raise SessionError(str(exc), 422) from exc
        self._settle(board)

    def undo(self, board_index: int) -> None:
        board = self.board(board_index)
        if not self.undo_allowed:
            raise SessionError("Undo is only available when playing a solo game alone")
        if not board.undo_stack:
            raise SessionError("Nothing to undo")
        if board.settled:  # the finished deal was already booked; take it back
            board.settled = False
            board.rounds -= 1
            board.wins[0] -= int(board.game.view(None).get("won", False))
            board.totals[0] -= board.game.returns()[0]
        board.game, moves = board.undo_stack.pop()
        del board.moves[moves:]
        board.solve_result = None

    def solve(self, board_index: int) -> dict[str, Any]:
        """Was this deal winnable? Solves the starting deal with every card visible."""
        if self.game_id != "klondike":
            raise SessionError("Only Klondike has a solver", 422)
        board = self.board(board_index)
        if not board.game.done:
            raise SessionError("Finish the deal first (the answer would be a hint)")
        if board.solve_result is None:
            from card_games_rl.games.klondike.solver import solve

            result = solve(make_game("klondike", seed=self.base_seed + self.round), max_nodes=50_000)
            board.solve_result = {"verdict": result.verdict, "nodes": result.nodes}
        return board.solve_result

    def advance(self, board_index: int | None = None, steps: int = 1) -> list[dict[str, Any]]:
        """Play up to ``steps`` non-human moves; stops at a human turn or the end of the deal."""
        indices = [board_index] if board_index is not None else range(len(self.boards))
        played = []
        for index in indices:
            board = self.board(index)
            while len(played) < min(steps, MAX_ADVANCE) and not board.game.done and not board.human_turn:
                played.append({"board": index, **self._bot_move(board)})
            if played:
                break
        return played

    def _bot_move(self, board: Board) -> dict[str, Any]:
        game = board.game
        seat = game.current_player
        policy = board.policies[seat]
        probs = policy.action_probabilities(game, seat)
        action = policy.act(game, seat)
        top = None
        if probs:
            # The alternatives an agent considered can reveal its hidden cards (e.g. "Play QS"),
            # so a table with a human only sees the chosen move; spectators see the top 3.
            show_all = self.reveal or self.spec.category != "competitive"
            best = sorted(probs.items(), key=lambda kv: -kv[1])[:TOP_K] if show_all else [(action, probs.get(action, 0.0))]
            top = [{"action": a, "label": game.action_label(a) if show_all else game.public_label(a),
                    "prob": round(p, 4)} for a, p in best]
        label = game.action_label(action) if self.reveal else game.public_label(action)
        # A trained agent without probabilities met a situation outside its strategy (CFR) and used the System rule.
        rule_move = board.seats[seat] == "agent" and not board.agent_fallback[seat] and probs is None
        move = {"seat": seat, "controller": board.seats[seat], "fallback": board.agent_fallback[seat],
                "rule_move": rule_move, "action": action, "label": label, "top": top}
        game.step(action)
        board.moves.append(move)
        self._settle(board)
        return move

    def _settle(self, board: Board) -> None:
        """Book the result once when a deal ends."""
        game = board.game
        if not game.done or board.settled:
            return
        board.settled = True
        board.rounds += 1
        returns = game.returns()
        final = game.view(None)
        single = len(board.seats) == 1
        for seat, value in enumerate(returns):
            board.totals[seat] += value
        if single:
            board.wins[0] += int(final["won"] if "won" in final else returns[0] > 0)
        else:
            for seat in game.winners():
                board.wins[seat] += 1
        if final.get("hand_points"):  # Hearts: match to 100
            for seat, pts in enumerate(final["hand_points"]):
                board.points[seat] += pts
            self.match_over = max(board.points) >= HEARTS_MATCH_POINTS
        elif "donkey" in final:  # Donkey: times each seat was the Donkey
            for seat in final["losers"]:
                board.points[seat] += 1 / len(final["losers"])
        elif final.get("result"):  # Poker: running chip count
            for seat, chips in enumerate(final["result"]):
                board.points[seat] += chips

    def new_round(self) -> None:
        if not all(b.game.done for b in self.boards):
            raise SessionError("Finish the current deal first")
        self.round += 1
        if self.match_over:
            for board in self.boards:
                board.points = [0.0] * len(board.seats)
            self.match_over = False
        for board in self.boards:
            board.game = self._new_game()
            board.moves = []
            board.settled = False
            board.undo_stack = []
            board.solve_result = None

    # ----- presentation ----------------------------------------------------
    def _board_view(self, board: Board) -> dict[str, Any]:
        game = board.game
        viewer = board.human_seat if board.human_seat is not None else (0 if len(board.seats) == 1 else None)
        view = game.view(viewer)
        if viewer is None and self.reveal:
            for seat in range(len(board.seats)):
                own = game.view(seat)
                for key in ("hands", "hole", "passes"):  # each seat's private part, from its own view
                    if key in view and isinstance(view[key], list):
                        view[key][seat] = own[key][seat]
        if not board.human_turn:
            view["legal_actions"] = []
            if "can_play_exact" in view:
                view["can_play_exact"] = False
        return {
            "seats": board.seats,
            "fallback": board.agent_fallback,
            "human_seat": board.human_seat,
            "human_turn": board.human_turn,
            "current_player": game.current_player,
            "done": game.done,
            "truncated": game.is_truncated,
            "returns": [round(r, 4) for r in game.returns()] if game.done else None,
            "winners": game.winners(),
            "view": view,
            "moves": board.moves[-30:],
            "undo_available": self.undo_allowed and bool(board.undo_stack),
            "solve_result": board.solve_result,
            "stats": {"rounds": board.rounds, "totals": [round(t, 4) for t in board.totals],
                      "wins": board.wins, "points": board.points},
        }

    def state(self) -> dict[str, Any]:
        self.touched = time.monotonic()
        return {
            "id": self.id,
            "game": self.game_id,
            "name": self.spec.name,
            "mode": self.mode,
            "seed": self.base_seed,
            "round": self.round,
            "reveal": self.reveal,
            "duel": len(self.boards) == 2,
            "match_over": self.match_over,
            "match_target": HEARTS_MATCH_POINTS if self.game_id == "hearts" else None,
            "agent": self.agents.status(self.game_id),
            "boards": [self._board_view(b) for b in self.boards],
        }


class SessionStore:
    def __init__(self, agents: AgentCache):
        self.agents = agents
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, **kwargs) -> Session:
        session = Session(agents=self.agents, **kwargs)
        with self._lock:
            self._purge()
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise SessionError("Session not found (it may have expired)", 404)
        session.touched = time.monotonic()
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _purge(self) -> None:
        now = time.monotonic()
        for sid in [s for s, v in self._sessions.items() if now - v.touched > SESSION_TTL]:
            del self._sessions[sid]
        while len(self._sessions) >= MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda v: v.touched)
            del self._sessions[oldest.id]

    def __len__(self) -> int:
        return len(self._sessions)
