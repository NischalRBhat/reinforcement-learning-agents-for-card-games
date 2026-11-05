"""Training for agents that are not PPO networks, selected by ``algorithm`` in the config.

* ``search`` (Rummy): a look-ahead search policy with no learned weights. "Training"
  validates it against the System bot and registers it, so it shows up like any
  trained agent.
* ``cfr`` (Poker): Monte Carlo CFR in parallel worker processes. Their strategies
  are merged and evaluated periodically, and the best one is kept. It follows the
  PPO conventions: ``--time-budget`` is a total across ``--resume`` runs, and
  Ctrl+C keeps the last saved progress.
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Callable

from card_games_rl.agents.registry import write_meta
from card_games_rl.core.seeds import VALIDATION_SEEDS

from .config import parse_duration
from .evaluate import gate_passed, make_baseline, play


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "per_deal"}


def train_search(game_id: str, config: dict[str, Any], out_dir: Path, log: Callable[[str], None], seed: int) -> Path:
    run, evaluation = config["run"], config["evaluation"]
    module, name = config["search_policy"].split(":")
    agent = getattr(importlib.import_module(module), name)(seed=seed)
    log(f"search agent {name}: no weights to learn; validating over {run['eval_episodes']} games...")
    result = play(game_id, agent, make_baseline(game_id, evaluation["opponent"], seed), run["eval_episodes"],
                  VALIDATION_SEEDS)
    gate = evaluation["gate"]
    write_meta(game_id, {"algorithm": "search", "policy": config["search_policy"], "profile": run["profile"],
                         "timesteps": 0, "seed": seed, "deterministic": True, "validation": _summary(result),
                         "reference": None, "gate": {**gate, "passed": gate_passed(game_id, result, gate)}},
               root=out_dir.parent, variant=config.get("variant", "base"))
    log(f"validation: {result['metric']} {result['value']:.4f} +/-{result['ci95']:.4f} -> registered in {out_dir}")
    return out_dir / "meta.json"


def train_cfr(game_id: str, config: dict[str, Any], out_dir: Path, log: Callable[[str], None],
              time_budget: str | None, seed: int, resume: bool, n_envs: int | None) -> Path:
    from card_games_rl.games.poker.cfr import CFRPolicy, merge_strategies, save_strategy, worker

    run, evaluation = config["run"], config["evaluation"]
    cfr_dir = out_dir / "cfr"
    cfr_dir.mkdir(parents=True, exist_ok=True)
    workers = n_envs or run["n_envs"]
    files = [cfr_dir / f"worker_{i}.pkl.gz" for i in range(workers)]
    state_file = cfr_dir / "state.json"
    if not resume:  # a fresh run starts from empty regret tables
        for old in cfr_dir.glob("worker_*.pkl.gz"):
            old.unlink()
        state_file.unlink(missing_ok=True)
    used = json.loads(state_file.read_text())["trained_seconds"] if resume and state_file.exists() else 0.0
    budget = parse_duration(time_budget)
    start = time.time()
    deadline = start + budget - used if budget else float("inf")
    per_worker = run["iterations"] // workers
    context = mp.get_context("spawn")
    processes = [context.Process(target=worker, daemon=True,
                                 args=(str(f), seed * 1000 + i, deadline, per_worker, config["cfr"]["save_every"]))
                 for i, f in enumerate(files)]
    for process in processes:
        process.start()
    log(f"start: CFR with {workers} workers, up to {run['iterations']:,} iterations, "
        f"budget={time_budget or 'none'} (already trained {used / 3600:.2f}h)")

    gate = evaluation["gate"]
    best: dict[str, Any] | None = None
    if resume:
        from card_games_rl.agents.registry import read_meta

        previous = read_meta(game_id, out_dir.parent)
        best = previous.get("validation") if previous and previous.get("algorithm") == "cfr" else None

    def evaluate() -> None:
        nonlocal best
        present = [f for f in files if f.exists()]
        if not present:
            return
        strategy, iterations = merge_strategies(present)
        agent = CFRPolicy(strategy=strategy, deterministic=evaluation["deterministic"], seed=seed)
        result = play(game_id, agent, make_baseline(game_id, evaluation["opponent"], seed), run["eval_episodes"],
                      VALIDATION_SEEDS)
        coverage = 1 - agent.misses / max(1, agent.lookups)
        improved = best is None or result["value"] > best["value"]
        if improved:
            best = _summary(result)
            save_strategy(strategy, out_dir / "cfr_strategy.pkl.gz", {"iterations": iterations})
            write_meta(game_id, {"algorithm": "cfr", "profile": run["profile"], "timesteps": iterations, "seed": seed,
                                 "deterministic": evaluation["deterministic"], "validation": best, "reference": None,
                                 "info_sets": len(strategy), "coverage": round(coverage, 4),
                                 "gate": {**gate, "passed": gate_passed(game_id, result, gate)}}, root=out_dir.parent)
        state_file.write_text(json.dumps({"trained_seconds": used + time.time() - start, "iterations": iterations}))
        status = "new best" if improved else f"best {best['value']:.2f}"
        log(f"eval @ {iterations:,} iterations: {result['metric']} {result['value']:.2f} +/-{result['ci95']:.2f} "
            f"({status}); {len(strategy):,} info sets, {coverage:.0%} of decisions covered")

    next_eval = time.time() + run["eval_every_seconds"]
    try:
        while any(p.is_alive() for p in processes):
            time.sleep(2)
            if time.time() >= next_eval:
                evaluate()
                next_eval = time.time() + run["eval_every_seconds"]
    except KeyboardInterrupt:
        log("interrupted: keeping the last saved progress")
        for process in processes:
            process.terminate()
    evaluate()
    log(f"done: best {best['metric']} = {best['value']:.2f}" if best else "done: nothing to evaluate")
    return out_dir / "cfr_strategy.pkl.gz"
