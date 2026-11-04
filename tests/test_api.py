"""API tests: every game x mode plays to the end, hidden info stays hidden,
errors map to the right status codes, and trained models are really used."""
from __future__ import annotations

import random

import pytest
from fastapi.testclient import TestClient

from card_games_rl.agents.registry import write_meta
from card_games_rl.api.app import create_app
from card_games_rl.api.sessions import MODES, AgentCache
from card_games_rl.core import Policy
from card_games_rl.games import GAME_SPECS, heuristic_policy


class SpyPolicy(Policy):
    """Stands in for a trained model and counts how often it is consulted."""

    name = "spy"

    def __init__(self, game_id: str):
        self.inner = heuristic_policy(game_id, seed=0)
        self.calls = 0

    def act(self, game, player):
        self.calls += 1
        return self.inner.act(game, player)

    def action_probabilities(self, game, player):
        legal = game.legal_actions()
        return {a: 1 / len(legal) for a in legal}


@pytest.fixture(scope="module")
def spies():
    return {}


@pytest.fixture(scope="module")
def client(spies):
    def loader(game_id, root=None):
        return spies.setdefault(game_id, SpyPolicy(game_id))

    app = create_app(agents=AgentCache(loader=loader))
    return TestClient(app)


def create(client, game, mode="human_vs_system", **extra):
    response = client.post("/api/sessions", json={"game": game, "mode": mode, "seed": 11, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def play_to_end(client, state, rng):
    sid = state["id"]
    for _ in range(5000):
        if all(b["done"] for b in state["boards"]):
            return state
        for index, board in enumerate(state["boards"]):
            if board["done"]:
                continue
            if board["human_turn"]:
                legal = board["view"]["legal_actions"]
                assert legal, f"{state['game']}: human turn without legal actions"
                response = client.post(f"/api/sessions/{sid}/action",
                                       json={"board": index, "action": rng.choice(legal)["action"]})
            else:
                response = client.post(f"/api/sessions/{sid}/advance", json={"board": index, "steps": 50})
                assert response.json()["advanced"], "advance made no progress"
            assert response.status_code == 200, response.text
            state = response.json()
            break
    raise AssertionError("game did not finish")


def test_catalog_lists_all_games_with_modes_and_model_status(client):
    games = client.get("/api/games").json()
    assert {g["id"] for g in games} == set(GAME_SPECS)
    for g in games:
        assert [m["id"] for m in g["modes"]] == list(MODES)
        assert {"trained", "metric", "value", "gate_passed"} <= set(g["agent"])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("game", sorted(GAME_SPECS))
def test_every_game_and_mode_plays_to_completion(client, game, mode):
    state = create(client, game, mode)
    category = GAME_SPECS[game].category
    if mode == "human_vs_agent" and category in ("solo", "dealer"):
        assert state["duel"] and [b["seats"] for b in state["boards"]] == [["human"], ["agent"]]
    state = play_to_end(client, state, random.Random(0))
    for board in state["boards"]:
        assert board["stats"]["rounds"] == 1 and board["returns"] is not None
    # Next deal of the match starts cleanly.
    nxt = client.post(f"/api/sessions/{state['id']}/new-round").json()
    assert nxt["round"] == 1 and not any(b["done"] for b in nxt["boards"])


def test_agent_seats_consult_the_agent_and_report_probabilities(client, spies):
    state = create(client, "hearts", "agent_vs_agent")
    before = spies["hearts"].calls
    state = client.post(f"/api/sessions/{state['id']}/advance", json={"steps": 5}).json()
    moves = state["advanced"]
    assert len(moves) == 5 and spies["hearts"].calls == before + 5
    assert all(m["controller"] == "agent" and not m["fallback"] and len(m["top"]) <= 3 for m in moves)


@pytest.mark.parametrize("game", ["hearts", "bluff", "rummy", "poker"])
def test_human_never_sees_opponent_cards(client, game):
    state = create(client, game)
    view = state["boards"][0]["view"]
    key = "hole" if game == "poker" else "hands"
    assert isinstance(view[key][0], list) and view[key][0]
    for other in view[key][1:]:
        assert other is None or isinstance(other, int)


def test_spectator_sees_all_hands_only_when_revealed(client):
    shown = create(client, "hearts", "agent_vs_agent")["boards"][0]["view"]["hands"]
    hidden = create(client, "hearts", "agent_vs_agent", reveal=False)["boards"][0]["view"]["hands"]
    assert all(isinstance(h, list) and len(h) == 13 for h in shown)
    assert all(h == 13 for h in hidden)


def test_legal_actions_hidden_when_not_human_turn(client):
    state = create(client, "blackjack", "agent_vs_agent")
    assert state["boards"][0]["view"]["legal_actions"] == []


def test_error_statuses(client):
    assert client.post("/api/sessions", json={"game": "chess"}).status_code == 404
    assert client.post("/api/sessions", json={"game": "hearts", "mode": "nope"}).status_code == 422
    assert client.post("/api/sessions", json={"game": "hearts", "seats": [["human"]]}).status_code == 422
    assert client.post("/api/sessions", json={"game": "hearts", "seats": [["human"] * 4]}).status_code == 422
    assert client.get("/api/sessions/missing").status_code == 404

    state = create(client, "blackjack")
    sid = state["id"]
    legal = {a["action"] for a in state["boards"][0]["view"]["legal_actions"]}
    illegal = next(a for a in range(7) if a not in legal)
    assert client.post(f"/api/sessions/{sid}/action", json={"action": illegal}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/action", json={}).status_code == 422
    assert client.post(f"/api/sessions/{sid}/new-round").status_code == 409

    watch = create(client, "poker", "agent_vs_agent")
    assert client.post(f"/api/sessions/{watch['id']}/action", json={"action": 1}).status_code == 409


def test_bluff_human_plays_exact_cards_and_names_the_rank(client):
    state = create(client, "bluff", seats=[["human", "system", "system", "system"]])
    sid = state["id"]
    for _ in range(300):
        board = state["boards"][0]
        if board["human_turn"] and board["view"]["can_play_exact"]:
            break
        if board["human_turn"]:  # only call/pass available
            state = client.post(f"/api/sessions/{sid}/action", json={"action": 15}).json()
        else:
            state = client.post(f"/api/sessions/{sid}/advance", json={"steps": 1}).json()
    view = state["boards"][0]["view"]
    hand = view["hands"][0]
    pick = [hand[0]["id"], hand[-1]["id"]]
    body = {"cards": pick, **({"claim_rank": 9} if view["phase"] == "declare" else {})}
    after = client.post(f"/api/sessions/{sid}/action", json=body).json()
    remaining = {c["id"] for c in after["boards"][0]["view"]["hands"][0]}
    assert not remaining & set(pick)
    assert after["boards"][0]["moves"][-1]["label"].startswith(("Claim 2 ×", "Add 2 ×"))
    assert client.post(f"/api/sessions/{sid}/action", json={"cards": [999]}).status_code in (409, 422)


def test_humans_never_see_whether_bots_bluffed(client):
    state = create(client, "bluff", seats=[["human", "agent", "agent", "agent"]])
    sid = state["id"]
    for _ in range(60):
        board = state["boards"][0]
        if board["done"]:
            break
        if board["human_turn"]:
            state = client.post(f"/api/sessions/{sid}/action", json={"action": board["view"]["legal_actions"][-1]["action"]}).json()
        else:
            state = client.post(f"/api/sessions/{sid}/advance", json={"steps": 1}).json()
            for move in state["advanced"]:
                assert "true" not in move["label"] and len(move["top"]) == 1
                assert "true" not in move["top"][0]["label"]


def test_hearts_match_points_and_poker_chips_accumulate(client):
    hearts = play_to_end(client, create(client, "hearts", "agent_vs_agent"), random.Random(1))
    points = hearts["boards"][0]["stats"]["points"]
    assert sum(points) in (26, 78)  # 78 when someone shoots the moon
    poker = play_to_end(client, create(client, "poker", "agent_vs_agent"), random.Random(1))
    assert sum(poker["boards"][0]["stats"]["points"]) == 0


def test_untrained_agent_falls_back_to_system(tmp_path):
    client = TestClient(create_app(model_root=tmp_path))
    assert not any(g["agent"]["trained"] for g in client.get("/api/games").json())
    state = client.post("/api/sessions", json={"game": "rummy", "mode": "agent_vs_agent", "seed": 3}).json()
    assert state["boards"][0]["fallback"] == [True, True]
    moves = client.post(f"/api/sessions/{state['id']}/advance", json={}).json()["advanced"]
    assert moves[0]["fallback"]


def test_trained_model_is_loaded_and_used(tmp_path, monkeypatch):
    from sb3_contrib import MaskablePPO

    from card_games_rl.agents.model_policy import ModelPolicy
    from card_games_rl.envs import SingleAgentEnv

    env = SingleAgentEnv("hearts", seed=0)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, batch_size=32, device="cpu",
                        policy_kwargs={"net_arch": [32]})
    (tmp_path / "hearts").mkdir()
    model.save(tmp_path / "hearts" / "best.zip")
    write_meta("hearts", {"deterministic": True, "validation": {"metric": "mean_points", "value": 6.1}}, root=tmp_path)

    calls = []
    original = ModelPolicy.act
    monkeypatch.setattr(ModelPolicy, "act", lambda self, g, p: calls.append(p) or original(self, g, p))

    client = TestClient(create_app(model_root=tmp_path))
    hearts = next(g for g in client.get("/api/games").json() if g["id"] == "hearts")
    assert hearts["agent"]["trained"] and hearts["agent"]["value"] == 6.1
    state = client.post("/api/sessions", json={"game": "hearts", "mode": "human_vs_agent", "seed": 5}).json()
    assert state["boards"][0]["fallback"] == [False] * 4
    sid = state["id"]
    while not state["boards"][0]["human_turn"] or not calls:
        if state["boards"][0]["human_turn"]:
            action = state["boards"][0]["view"]["legal_actions"][0]["action"]
            state = client.post(f"/api/sessions/{sid}/action", json={"action": action}).json()
        else:
            state = client.post(f"/api/sessions/{sid}/advance", json={"steps": 1}).json()
            move = state["advanced"][0]
            assert move["controller"] == "agent" and abs(sum(t["prob"] for t in move["top"])) <= 1.0001
    assert calls


def test_agent_status_prefers_a_fresh_test_report(tmp_path):
    import json
    import os

    write_meta("rummy", {"algorithm": "search", "policy": "card_games_rl.games.rummy.search:LookaheadPolicy",
                         "validation": {"metric": "win_rate", "value": 0.54, "ci95": 0.04, "deals": 500},
                         "gate": {"threshold": 0.55, "passed": False}}, root=tmp_path)
    cache = AgentCache(tmp_path)
    status = cache.status("rummy")
    assert status["trained"] and status["source"] == "validation" and status["value"] == 0.54
    assert status["algorithm_name"] == "Look-ahead search" and status["target"] == 0.55 and status["baseline"] is None

    report = tmp_path / "rummy" / "eval_report.json"
    report.write_text(json.dumps({
        "primary": {"metric": "win_rate", "value": 0.56, "ci95": 0.02, "deals": 2000, "higher_is_better": True},
        "references": {"heuristic_vs_heuristic": {"value": 0.5}},
        "gate": {"threshold": 0.55, "passed": True}}), encoding="utf-8")
    meta_time = (tmp_path / "rummy" / "meta.json").stat().st_mtime
    os.utime(report, (meta_time + 5, meta_time + 5))
    status = cache.status("rummy")
    assert status["source"] == "test" and status["value"] == 0.56 and status["gate_passed"]
    assert status["episodes"] == 2000 and status["baseline"] == {"label": "System vs System", "value": 0.5}

    os.utime(report, (meta_time - 5, meta_time - 5))  # retrained after the report: the report is stale
    assert cache.status("rummy")["source"] == "validation"


def test_web_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200 and "<html" in response.text.lower()


def test_klondike_undo_restores_the_previous_position(client):
    state = create(client, "klondike")
    sid, before = state["id"], state["boards"][0]["view"]
    assert not state["boards"][0]["undo_available"]
    after = client.post(f"/api/sessions/{sid}/action", json={"action": before["legal_actions"][0]["action"]}).json()
    assert after["boards"][0]["undo_available"] and after["boards"][0]["moves"]
    undone = client.post(f"/api/sessions/{sid}/undo", json={"board": 0}).json()
    view = undone["boards"][0]["view"]
    assert (view["tableau"], view["stock_count"], view["waste"]) == (before["tableau"], before["stock_count"], before["waste"])
    assert not undone["boards"][0]["moves"] and not undone["boards"][0]["undo_available"]
    assert client.post(f"/api/sessions/{sid}/undo", json={"board": 0}).status_code == 409


def test_undo_is_refused_in_duels_and_multiplayer_games(client):
    for game, mode in [("klondike", "human_vs_agent"), ("hearts", "human_vs_system"), ("blackjack", "human_vs_system")]:
        state = create(client, game, mode)
        assert client.post(f"/api/sessions/{state['id']}/undo", json={"board": 0}).status_code == 409


def test_klondike_solver_answers_only_after_the_deal(client):
    state = create(client, "klondike", "agent_vs_agent")
    sid = state["id"]
    assert client.post(f"/api/sessions/{sid}/solve", json={"board": 0}).status_code == 409
    state = play_to_end(client, state, random.Random(0))
    solved = client.post(f"/api/sessions/{sid}/solve", json={"board": 0}).json()
    assert solved["boards"][0]["solve_result"]["verdict"] in ("winnable", "not winnable", "unknown")
    hearts = create(client, "hearts")
    assert client.post(f"/api/sessions/{hearts['id']}/solve", json={"board": 0}).status_code == 422


def test_hearts_spectators_see_every_pass_after_the_exchange(client):
    state = create(client, "hearts", "agent_vs_agent", seed=2)
    sid = state["id"]
    while state["boards"][0]["view"]["phase"] == "passing":
        state = client.post(f"/api/sessions/{sid}/advance", json={"steps": 1}).json()
    passes = state["boards"][0]["view"]["passes"]
    if passes:  # hands without passing have no summary
        assert all(p["cards"] and len(p["cards"]) == 3 for p in passes)


def test_humans_never_see_which_cards_bots_pass_in_hearts(client):
    state = create(client, "hearts", "human_vs_agent", seed=6)  # round 0 passes left
    sid = state["id"]
    for card in state["boards"][0]["view"]["legal_actions"][:3]:
        state = client.post(f"/api/sessions/{sid}/action", json={"action": card["action"]}).json()
    moves = []
    while state["boards"][0]["view"]["phase"] == "passing":
        state = client.post(f"/api/sessions/{sid}/advance", json={"steps": 1}).json()
        moves += state["advanced"]
    assert len(moves) == 9
    for move in moves:
        assert move["label"] == "Pass a card"
        assert all(t["label"] == "Pass a card" for t in move["top"] or [])
