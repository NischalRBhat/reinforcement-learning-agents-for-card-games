"""Head-to-head evaluation of trained agents against baselines.

    python -m card_games_rl.training.evaluate --all
    python -m card_games_rl.training.evaluate --game hearts --episodes 2000
    python -m card_games_rl.training.evaluate --all --variant counting   # + head-to-head vs the base agent

Writes ``models/<game>/eval_report.{json,md}`` and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from card_games_rl.core import Game, Policy, RandomPolicy
from card_games_rl.core.seeds import TEST_SEEDS, VALIDATION_SEEDS  # disjoint from training seeds
from card_games_rl.games import GAME_SPECS, heuristic_policy, make_game

from .config import load_config


@dataclass(frozen=True)
class Metric:
    name: str
    value: Callable[[Game, int], float]
    higher_is_better: bool = True


METRICS: dict[str, Metric] = {
    "blackjack": Metric("ev", lambda g, s: g.score(s)),
    "klondike": Metric("win_rate", lambda g, s: float(g.won)),
    "poker": Metric("bb_per_100", lambda g, s: g.chip_result()[s] / 2 * 100),
    "rummy": Metric("win_rate", lambda g, s: float(g.winner == s)),
    "bluff": Metric("win_rate", lambda g, s: float(g.winner == s)),
    "hearts": Metric("mean_points", lambda g, s: float(g.hand_points()[s]), higher_is_better=False),
    # Share of being the Donkey (players tied for most cards at a forced stop share it); fair share 0.25.
    "donkey": Metric("donkey_rate", lambda g, s: 1 / len(g.losers) if s in g.losers else 0.0, higher_is_better=False),
    # Partnership points per hand above the other side (System vs System = 0).
    "canasta": Metric("point_diff", lambda g, s: float(g.hand_scores[g.team(s)] - g.hand_scores[1 - g.team(s)])),
}

BASELINES: dict[str, list[str]] = {
    "blackjack": ["optimal", "heuristic", "random"],
    "klondike": ["heuristic", "random"],
    "poker": ["heuristic", "calling_station", "random"],
    "rummy": ["heuristic", "random"],
    "bluff": ["heuristic", "random"],
    "hearts": ["heuristic", "random"],
    "donkey": ["heuristic", "random"],
    "canasta": ["heuristic", "random"],
}


def make_baseline(game_id: str, name: str, seed: int = 0) -> Policy:
    if name == "random":
        return RandomPolicy(seed)
    if name == "heuristic":
        return heuristic_policy(game_id, seed)
    if name == "optimal" and game_id == "blackjack":
        from card_games_rl.games.blackjack import OptimalPolicy
        return OptimalPolicy()
    if name == "calling_station" and game_id == "poker":
        from card_games_rl.games.poker import CallingStation
        return CallingStation()
    raise ValueError(f"Unknown baseline {name!r} for {game_id}")


def play(game_id: str, learner: Policy, opponent: Policy | None, episodes: int,
         seed_offset: int = TEST_SEEDS) -> dict[str, Any]:
    """Play ``episodes`` games; ``learner`` takes one seat and ``opponent`` the others.

    Duplicate format: each deal is replayed with the learner in every seat, so luck
    of the cards cancels out; the 95% CI is computed over per-deal averages.
    """
    metric = METRICS[game_id]
    game = make_game(game_id)
    seats = game.num_players
    deal_values: list[list[float]] = []
    returns = []
    for episode in range(episodes):
        deal, seat = divmod(episode, seats)
        game.reset(seed_offset + deal)
        # Partnership games: the learner plays both seats of its side.
        learner_seats = set(game.team_seats(seat)) if hasattr(game, "team_seats") else {seat}
        while not game.done:
            player = game.current_player
            policy = learner if player in learner_seats else opponent
            game.step(policy.act(game, player))
        if seat == 0:
            deal_values.append([])
        deal_values[-1].append(metric.value(game, seat))
        returns.append(game.score(seat))
    per_deal = [sum(v) / len(v) for v in deal_values]
    n = len(per_deal)
    mean = sum(per_deal) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in per_deal) / max(1, n - 1))
    return {
        "metric": metric.name, "value": mean, "ci95": 1.96 * std / math.sqrt(n),
        "mean_return": sum(returns) / episodes, "episodes": episodes, "deals": n,
        "higher_is_better": metric.higher_is_better, "per_deal": per_deal,
    }


def head_to_head(game_id: str, agent_a: Policy, agent_b: Policy, deals: int,
                 seed_offset: int = TEST_SEEDS) -> dict[str, Any]:
    """Agent A against agent B on duplicate deals.

    Two players: each deal is played twice, A in seat 0 then seat 1. Four players: A
    takes two opposite seats and B the other two, then the seats swap. Card luck
    cancels out, so the per-deal difference (A minus B, in the game's metric) measures
    which agent is stronger. ``a_better`` says whether that difference favours A.
    """
    metric = METRICS[game_id]
    game = make_game(game_id)
    seats = game.num_players
    arrangements = [[0], [1]] if seats == 2 else [[0, 2], [1, 3]]
    a_values, b_values, diffs = [], [], []
    for deal in range(deals):
        a_total = b_total = 0.0
        for a_seats in arrangements:
            game.reset(seed_offset + deal)
            while not game.done:
                player = game.current_player
                game.step((agent_a if player in a_seats else agent_b).act(game, player))
            a_total += sum(metric.value(game, s) for s in a_seats) / len(a_seats)
            b_total += sum(metric.value(game, s) for s in range(seats) if s not in a_seats) / (seats - len(a_seats))
        a_values.append(a_total / len(arrangements))
        b_values.append(b_total / len(arrangements))
        diffs.append(a_values[-1] - b_values[-1])
    n = len(diffs)
    mean = sum(diffs) / n
    std = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(1, n - 1))
    sign = 1 if metric.higher_is_better else -1
    return {"metric": metric.name, "higher_is_better": metric.higher_is_better, "deals": n,
            "a": sum(a_values) / n, "b": sum(b_values) / n, "difference": mean, "ci95": 1.96 * std / math.sqrt(n),
            "a_better": sign * mean > 0}


def klondike_solvability(deals: int, seed_offset: int = TEST_SEEDS, max_nodes: int = 30_000) -> list[str]:
    """Solver verdict for each test deal, cached on disk (the deals never change)."""
    from card_games_rl.games.klondike.solver import solve

    cache_path = Path(__file__).with_name("klondike_solvability.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    changed = False
    for deal in range(deals):
        key = str(seed_offset + deal)
        if key not in cache:
            cache[key] = solve(make_game("klondike", seed=seed_offset + deal), max_nodes).verdict
            changed = True
    if changed:
        cache_path.write_text(json.dumps(cache, indent=0, sort_keys=True), encoding="utf-8")
    return [cache[str(seed_offset + d)] for d in range(deals)]


def winnable_summary(result: dict[str, Any], verdicts: list[str]) -> dict[str, Any]:
    from card_games_rl.games.klondike.solver import WON

    winnable = [v for v, verdict in zip(result["per_deal"], verdicts) if verdict == WON]
    return {"proven_winnable_share": len(winnable) / len(verdicts),
            "win_rate_on_winnable": sum(winnable) / len(winnable) if winnable else None,
            "verdicts": {k: verdicts.count(k) for k in sorted(set(verdicts))}}


def gate_passed(game_id: str, result: dict[str, Any], gate: dict[str, Any], reference: dict[str, Any] | None = None) -> bool:
    if "max_gap" in gate:  # e.g. blackjack: within max_gap of the optimal policy
        return reference is not None and result["value"] >= reference["value"] - gate["max_gap"]
    if result["higher_is_better"]:
        return result["value"] >= gate["threshold"]
    return result["value"] <= gate["threshold"]


def evaluate_agent(game_id: str, agent: Policy, episodes: int, seed_offset: int = TEST_SEEDS) -> dict[str, Any]:
    """Full report: agent vs every baseline, baseline references, and the gate."""
    config = load_config(game_id)
    solo = GAME_SPECS[game_id].max_players == 1
    report: dict[str, Any] = {"game": game_id, "episodes": episodes, "matchups": {}, "references": {}}
    for name in BASELINES[game_id]:
        baseline = make_baseline(game_id, name)
        if solo:  # single-player: compare as alternative players on the same deals
            report["references"][name] = play(game_id, baseline, None, episodes, seed_offset)
        else:
            report["matchups"][f"vs_{name}"] = play(game_id, agent, baseline, episodes, seed_offset)
    if solo:
        report["matchups"]["solo"] = play(game_id, agent, None, episodes, seed_offset)
        primary = report["matchups"]["solo"]
    else:
        report["references"]["heuristic_vs_heuristic"] = play(game_id, make_baseline(game_id, "heuristic", 1),
                                                              make_baseline(game_id, "heuristic", 2), episodes, seed_offset)
        primary = report["matchups"][f"vs_{config['evaluation']['opponent']}"]
    gate = config["evaluation"]["gate"]
    reference = report["references"].get(gate.get("baseline", ""))
    if game_id == "klondike":
        verdicts = klondike_solvability(primary["deals"], seed_offset)
        report["solvability"] = {name: winnable_summary(r, verdicts)
                                 for name, r in [("agent", primary), *report["references"].items()]}
    report["primary"] = primary
    report["gate"] = {**gate, "passed": gate_passed(game_id, primary, gate, reference)}
    return report


def report_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Evaluation: {report['game']}", "", f"Episodes per matchup: {report['episodes']} (held-out test seeds)", "",
             "| Matchup | Metric | Value | 95% CI | Mean return |", "|---|---|---|---|---|"]
    for section in ("matchups", "references"):
        for name, r in report[section].items():
            label = f"agent {name}" if section == "matchups" else f"reference: {name}"
            lines.append(f"| {label} | {r['metric']} | {r['value']:.4f} | ±{r['ci95']:.4f} | {r['mean_return']:.4f} |")
    if "solvability" in report:
        first = next(iter(report["solvability"].values()))
        lines += ["", f"Solver on the test deals: {first['verdicts']} "
                  f"(proven winnable: {first['proven_winnable_share']:.1%})", "",
                  "| Player | Win rate on proven-winnable deals |", "|---|---|"]
        lines += [f"| {name} | {s['win_rate_on_winnable']:.4f} |" for name, s in report["solvability"].items()]
    if "head_to_head" in report:
        h = report["head_to_head"]
        lines += ["", f"## Head-to-head: this agent vs the base agent ({h['deals']} duplicate deals)", "",
                  f"| | {h['metric']} |", "|---|---|", f"| this agent | {h['a']:.4f} |", f"| base agent | {h['b']:.4f} |",
                  f"| difference | {h['difference']:+.4f} ± {h['ci95']:.4f} |", "",
                  f"Better: **{'this agent' if h['a_better'] else 'base agent'}**"
                  + ("" if abs(h['difference']) > h['ci95'] else " (within the margin: not a clear difference)")]
    gate = report["gate"]
    lines += ["", f"**Gate:** {json.dumps({k: v for k, v in gate.items() if k != 'passed'})} → "
              f"{'PASSED' if gate['passed'] else 'NOT MET'}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained card-game agents")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--game", choices=sorted(GAME_SPECS))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--episodes", type=int, default=None, help="Default: the full profile's eval_episodes")
    parser.add_argument("--variant", default="base", choices=["base", "counting"],
                        help="Which agents to evaluate; counting also plays them head-to-head against the base agents")
    args = parser.parse_args()

    from card_games_rl.agents.registry import load_agent, model_dir
    from card_games_rl.encoders import BASE, agent_key, variants_for

    games = sorted(GAME_SPECS) if args.all else [args.game]
    games = [g for g in games if args.variant in variants_for(g)]
    rows, duels = [], []
    for game_id in games:
        key = agent_key(game_id, args.variant)
        agent = load_agent(game_id, variant=args.variant)
        if agent is None:
            print(f"{key}: no compatible trained model in models/{key}/ - train it first")
            continue
        full = load_config(game_id, "full")["run"]
        episodes = args.episodes or full.get("test_episodes", full["eval_episodes"])
        print(f"{key}: evaluating over {episodes} episodes per matchup...", flush=True)
        report = evaluate_agent(game_id, agent, episodes)
        base = load_agent(game_id) if args.variant != BASE else None
        if base is not None:
            deals = max(1, episodes // GAME_SPECS[game_id].max_players)
            print(f"{key}: head-to-head against the base agent over {deals} duplicate deals...", flush=True)
            report["head_to_head"] = head_to_head(game_id, agent, base, deals)
            h = report["head_to_head"]
            duels.append(f"| {game_id} | {h['metric']} | {h['a']:.4f} | {h['b']:.4f} | {h['difference']:+.4f} +/-{h['ci95']:.4f} |"
                         f" {'counting' if h['a_better'] else 'base'}{'' if abs(h['difference']) > h['ci95'] else ' (not clear)'} |")
        out = model_dir(game_id, variant=args.variant)
        (out / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "eval_report.md").write_text(report_markdown(report), encoding="utf-8")
        p = report["primary"]
        rows.append(f"| {key} | {p['metric']} | {p['value']:.4f} +/-{p['ci95']:.4f} | {'PASSED' if report['gate']['passed'] else 'NOT MET'} |")
    if rows:
        print("\n| Agent | Metric | Agent vs System | Gate |\n|---|---|---|---|\n" + "\n".join(rows))
    if duels:
        print("\n| Game | Metric | Counting agent | Base agent | Difference | Better |\n|---|---|---|---|---|---|\n"
              + "\n".join(duels))


if __name__ == "__main__":
    main()
