"""FastAPI backend: game catalog, seat-based sessions and the static web UI.

Run with ``python -m uvicorn card_games_rl.api.app:app`` or ``launch_card_games.py``.
"""
from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from card_games_rl.encoders import COUNTING, variants_for
from card_games_rl.games import GAME_SPECS

from .sessions import CONTROLLERS, MODES, AgentCache, SessionError, SessionStore

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

MODE_LABELS = {
    "solo": {"human_vs_system": "Play", "human_vs_agent": "Duel vs Agent", "agent_vs_agent": "Watch Agent"},
    "dealer": {"human_vs_system": "Play vs Dealer", "human_vs_agent": "Duel vs Agent", "agent_vs_agent": "Watch Agent"},
    "competitive": {"human_vs_system": "Human vs System", "human_vs_agent": "Human vs Agent", "agent_vs_agent": "Agent vs Agent",
                    "agent_vs_counting": "Agent vs Counting agent"},
}


class CreateSession(BaseModel):
    game: str
    mode: str = "human_vs_system"
    seats: list[list[str]] | None = Field(None, description="Controllers per board; overrides the mode's default")
    seed: int | None = None
    reveal: bool | None = None


class ActionRequest(BaseModel):
    board: int = 0
    action: int | None = None
    cards: list[int] | None = Field(None, description="Exact cards to play (Bluff)")
    claim_rank: int | None = Field(None, description="Rank named when starting a Bluff round (2-14, Ace = 14)")


class BoardRequest(BaseModel):
    board: int = 0


class AdvanceRequest(BaseModel):
    board: int | None = None
    steps: int = Field(1, ge=1)


def create_app(model_root: Path | None = None, agents: AgentCache | None = None, web_dir: Path | None = WEB_DIR,
               preload: bool = True) -> FastAPI:
    agents = agents or AgentCache(model_root)
    store = SessionStore(agents)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Load every trained model in the background so the first game starts instantly.
        if preload:
            threading.Thread(target=lambda: [agents.get(g) for g in GAME_SPECS], daemon=True).start()
        yield

    app = FastAPI(title="Card Games RL", lifespan=lifespan)
    app.state.store = store
    app.state.agents = agents

    @app.middleware("http")
    async def revalidate_static(request, call_next):
        # Without this, browsers reuse old copies of the JS/CSS after an update. "no-cache"
        # still allows caching, but makes the browser check the ETag first (a cheap 304).
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    def run(session_id: str, fn) -> dict[str, Any]:
        try:
            session = store.get(session_id)
            with session.lock:
                result = fn(session)
                state = session.state()
            if isinstance(result, list):
                state["advanced"] = result
            return state
        except SessionError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "sessions": len(store)}

    def modes(spec) -> list[str]:
        counting = COUNTING in variants_for(spec.game_id) and spec.category == "competitive"
        return list(spec.modes) + (["agent_vs_counting"] if counting else [])

    @app.get("/api/games")
    def games() -> list[dict[str, Any]]:
        return [
            {"id": spec.game_id, "name": spec.name, "category": spec.category, "description": spec.description,
             "players": spec.max_players,
             "modes": [{"id": m, "label": MODE_LABELS[spec.category][m]} for m in modes(spec)],
             "controllers": [c for c in CONTROLLERS if c != "counting" or COUNTING in variants_for(spec.game_id)],
             "agent": agents.status(spec.game_id), "agents": agents.statuses(spec.game_id)}
            for spec in GAME_SPECS.values()
        ]

    @app.post("/api/sessions")
    def create_session(req: CreateSession) -> dict[str, Any]:
        if req.game not in GAME_SPECS:
            raise HTTPException(404, f"Unknown game {req.game!r}")
        if req.mode not in MODES:
            raise HTTPException(422, f"Unknown mode {req.mode!r}")
        try:
            session = store.create(game_id=req.game, mode=req.mode, seats=req.seats, seed=req.seed, reveal=req.reveal)
        except SessionError as exc:
            raise HTTPException(exc.status, str(exc)) from exc
        with session.lock:
            return session.state()

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return run(session_id, lambda s: None)

    @app.post("/api/sessions/{session_id}/action")
    def action(session_id: str, req: ActionRequest) -> dict[str, Any]:
        return run(session_id, lambda s: s.human_action(req.board, req.action, req.cards, req.claim_rank))

    @app.post("/api/sessions/{session_id}/advance")
    def advance(session_id: str, req: AdvanceRequest) -> dict[str, Any]:
        return run(session_id, lambda s: s.advance(req.board, req.steps))

    @app.post("/api/sessions/{session_id}/new-round")
    def new_round(session_id: str) -> dict[str, Any]:
        return run(session_id, lambda s: s.new_round())

    @app.post("/api/sessions/{session_id}/undo")
    def undo(session_id: str, req: BoardRequest) -> dict[str, Any]:
        return run(session_id, lambda s: s.undo(req.board))

    @app.post("/api/sessions/{session_id}/solve")
    def solve(session_id: str, req: BoardRequest) -> dict[str, Any]:
        return run(session_id, lambda s: s.solve(req.board))

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, Any]:
        store.delete(session_id)
        return {"ok": True}

    if web_dir is not None and web_dir.exists():
        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(web_dir / "index.html", headers={"Cache-Control": "no-cache"})

        app.mount("/", StaticFiles(directory=web_dir), name="web")

    return app


app = create_app()
