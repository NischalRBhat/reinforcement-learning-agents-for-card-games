"""Train MaskablePPO agents (docs/ENV_DESIGN.md §1.4).

    python -m card_games_rl.training.train --game hearts --profile smoke
    python -m card_games_rl.training.train --all --profile full --time-budget 3h

Outputs per game:
  models/<game>/best.zip + meta.json    best checkpoint by validation metric
  models/<game>/checkpoints/latest.zip  for --resume
  models/<game>/pool/*.zip              self-play snapshots (competitive games)
  runs/<game>/                          TensorBoard logs + progress.log
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from card_games_rl.games import GAME_SPECS

from .config import load_config, parse_duration

ROOT = Path(__file__).resolve().parents[2]
MAX_POOL_SNAPSHOTS = 30
DEVICE_FILE = Path(__file__).resolve().parent / "device_benchmark.json"


def env_spec(game_id: str, config: dict[str, Any], rank: int, seed: int, pool_dir: Path) -> dict[str, Any]:
    """Picklable description of one training env (built inside a torch-free worker)."""
    return {"game_id": game_id, "seed": seed + rank, "opponents": config["opponents"],
            "opponent_mix": config.get("opponent_mix"), "game_kwargs": config.get("game_kwargs") or {},
            "pool_dir": str(pool_dir)}


def _next_multiple(steps: int, interval: int) -> int:
    return (steps // interval + 1) * interval


def summary(result: dict | None) -> dict | None:
    """An evaluation result without its per-deal values (kept out of meta.json)."""
    return None if result is None else {k: v for k, v in result.items() if k != "per_deal"}


def build_callback(game_id: str, config: dict[str, Any], out_dir: Path, log, deadline: float | None, seed: int,
                   start_steps: int = 0, previous_best: dict[str, Any] | None = None,
                   trained_seconds: float = 0.0):
    """``start_steps``/``previous_best`` let a resumed run continue its schedules
    and keep the best model unless it is actually beaten."""
    from stable_baselines3.common.callbacks import BaseCallback

    from card_games_rl.agents.model_policy import ModelPolicy
    from card_games_rl.agents.numpy_policy import export_policy
    from card_games_rl.agents.registry import write_meta

    from .evaluate import VALIDATION_SEEDS, gate_passed, make_baseline, play

    run = config["run"]
    evaluation = config["evaluation"]
    solo = GAME_SPECS[game_id].max_players == 1
    pool_dir, ckpt_dir = out_dir / "pool", out_dir / "checkpoints"

    class TrainingCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.start = time.time()
            self.best: dict[str, Any] | None = previous_best
            self.reference: dict[str, Any] | None = None
            self.log_every = max(2048, run["eval_every"] // 5)
            self.next_eval = _next_multiple(start_steps, run["eval_every"])
            self.next_log = _next_multiple(start_steps, self.log_every)
            if not run["snapshot_every"]:
                self.next_snapshot = None
            elif start_steps:
                self.next_snapshot = _next_multiple(start_steps, run["snapshot_every"])
            else:
                # First snapshot early (10% of the interval) so the self-play pool fills
                # quickly and the slower heuristic opponents stop dominating.
                self.next_snapshot = max(1, run["snapshot_every"] // 10)
            self.last_eval_step = start_steps
            self.stopped_by_budget = False

        def _elapsed(self) -> str:
            return str(timedelta(seconds=int(time.time() - self.start)))

        def evaluate(self) -> None:
            steps = self.num_timesteps
            agent = ModelPolicy.from_model(self.model, deterministic=evaluation["deterministic"])
            opponent = None if solo else make_baseline(game_id, evaluation["opponent"], seed)
            result = play(game_id, agent, opponent, run["eval_episodes"], VALIDATION_SEEDS)
            gate = evaluation["gate"]
            if "baseline" in gate and self.reference is None:
                self.reference = play(game_id, make_baseline(game_id, gate["baseline"]), None,
                                      run["eval_episodes"], VALIDATION_SEEDS)
            sign = 1 if result["higher_is_better"] else -1
            improved = self.best is None or sign * result["value"] > sign * self.best["value"]
            if improved:
                self.best = result
                self.model.save(str(out_dir / "best.zip"))
                passed = gate_passed(game_id, result, gate, self.reference)
                write_meta(game_id, {
                    "algorithm": "MaskablePPO", "profile": run["profile"], "timesteps": int(steps), "seed": seed,
                    "deterministic": evaluation["deterministic"],
                    "net_arch": config["policy"]["net_arch"], "validation": summary(result),
                    "reference": summary(self.reference), "gate": {**gate, "passed": passed},
                }, root=out_dir.parent)
            self.model.save(str(ckpt_dir / "latest.zip"))
            # Cumulative training time, so --time-budget is a total across resumes.
            (ckpt_dir / "state.json").write_text(json.dumps(
                {"trained_seconds": trained_seconds + time.time() - self.start, "timesteps": int(steps)}))
            status = "new best" if improved else f"best {self.best['value']:.4f}"
            optimal = f" | optimal {self.reference['value']:.4f}" if self.reference else ""
            log(f"eval @ {steps:,}: {result['metric']} {result['value']:.4f} +/-{result['ci95']:.4f} ({status}){optimal}")
            self.last_eval_step = steps

        def _on_step(self) -> bool:
            steps = self.num_timesteps
            if self.next_snapshot is not None and steps >= self.next_snapshot:
                # Actor weights as .npz (numpy inference in torch-free workers), written atomically.
                export_policy(self.model, pool_dir / f"snap_{steps:010d}.npz")
                for old in sorted(pool_dir.glob("snap_*.npz"))[:-MAX_POOL_SNAPSHOTS]:
                    old.unlink(missing_ok=True)
                self.next_snapshot += run["snapshot_every"]
            if steps >= self.next_log:
                episodes = self.model.ep_info_buffer
                ret = sum(e["r"] for e in episodes) / len(episodes) if episodes else float("nan")
                fps = (steps - start_steps) / max(1e-6, time.time() - self.start)
                log(f"{steps:,} steps | {fps:,.0f} steps/s | recent episode return {ret:+.4f} | {self._elapsed()} elapsed")
                self.next_log += self.log_every
            if steps >= self.next_eval:
                self.evaluate()
                self.next_eval += run["eval_every"]
            if deadline is not None and time.time() >= deadline:
                self.stopped_by_budget = True
                log("time budget reached; stopping")
                return False
            return True

        def _on_training_end(self) -> None:
            if self.last_eval_step != self.num_timesteps:
                self.evaluate()

    return TrainingCallback()


def resolve_device(game_id: str, requested: str = "auto") -> str:
    """'cpu' | 'cuda' | 'auto'. Auto uses the faster device measured by
    ``python -m card_games_rl.training.benchmark --training`` (CPU if unmeasured)."""
    import torch

    if requested == "auto":
        choices = json.loads(DEVICE_FILE.read_text()) if DEVICE_FILE.exists() else {}
        requested = choices.get(game_id, {}).get("best", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print(f"[{game_id}] CUDA requested but not available; using CPU", flush=True)
        return "cpu"
    return requested


def make_vec_env(game_id: str, config: dict[str, Any], n_envs: int, seed: int, pool_dir: Path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    from card_games_rl.envs.worker import build_env

    from .vec_env import LightVecEnv

    specs = [env_spec(game_id, config, rank, seed * 1000, pool_dir) for rank in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv([lambda: build_env(specs[0])])
    return LightVecEnv(specs)


def build_model(config: dict[str, Any], vec_env, device: str, seed: int, run_dir: Path | None):
    from sb3_contrib import MaskablePPO

    return MaskablePPO("MlpPolicy", vec_env, seed=seed, device=device, verbose=0,
                       tensorboard_log=str(run_dir) if run_dir else None,
                       policy_kwargs={"net_arch": config["policy"]["net_arch"]}, **config["ppo"])


def train(game_id: str, profile: str = "smoke", time_budget: str | None = None, seed: int = 0,
          resume: bool = False, n_envs: int | None = None, torch_threads: int = 2,
          model_root: Path | None = None, run_root: Path | None = None,
          overrides: dict[str, Any] | None = None, device: str = "auto") -> Path:
    """Train one game. ``overrides`` patch the profile (e.g. tiny runs in tests)."""
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(torch_threads)
    device = resolve_device(game_id, device)
    config = load_config(game_id, profile)
    config["run"].update(overrides or {})
    run = config["run"]
    out_dir = (model_root or ROOT / "models") / game_id
    run_dir = (run_root or ROOT / "runs") / game_id
    for path in (out_dir / "pool", out_dir / "checkpoints", run_dir):
        path.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "progress.log"

    def log(message: str) -> None:
        line = f"[{game_id}] {message}"
        print(line, flush=True)
        with log_file.open("a", encoding="utf-8") as stream:
            stream.write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")

    algorithm = config.get("algorithm", "ppo")
    if algorithm in ("search", "cfr"):
        from .non_neural import train_cfr, train_search

        if algorithm == "search":
            return train_search(game_id, config, out_dir, log, seed)
        return train_cfr(game_id, config, out_dir, log, time_budget, seed, resume, n_envs)

    envs = n_envs or run["n_envs"]
    vec_env = make_vec_env(game_id, config, envs, seed, out_dir / "pool")
    checkpoint = out_dir / "checkpoints" / "latest.zip"
    if resume and checkpoint.exists():
        model = MaskablePPO.load(str(checkpoint), env=vec_env, device=device, tensorboard_log=str(run_dir))
        log(f"resumed from {checkpoint} at {model.num_timesteps:,} steps")
    else:
        model = build_model(config, vec_env, device, seed, run_dir)
    warm_started = False
    if model.num_timesteps == 0:
        episodes = run.get("pretrain_episodes", 0)
        if episodes:
            from .pretrain import collect, pretrain

            log(f"behaviour-cloning warm start: collecting {episodes:,} teacher episodes...")
            stats = pretrain(model, collect(game_id, config, episodes, seed), seed=seed)
            log(f"warm start done: {stats['samples']:,.0f} samples, imitation accuracy {stats['accuracy']:.3f}, "
                f"value MSE {stats['value_mse']:.4f}")
            warm_started = True
    state_file = out_dir / "checkpoints" / "state.json"
    used = json.loads(state_file.read_text())["trained_seconds"] if resume and state_file.exists() else 0.0
    budget = parse_duration(time_budget)
    time_left = budget - used if budget else None  # the budget is a total per game, across resumes
    deadline = time.time() + time_left if budget else None
    log(f"start: profile={profile} device={device} timesteps={run['timesteps']:,} envs={envs} "
        f"budget={time_budget or 'none'} (already trained {used / 3600:.2f}h)")
    from card_games_rl.agents.registry import read_meta

    start = model.num_timesteps
    previous = read_meta(game_id, out_dir.parent) if start else None
    callback = build_callback(game_id, config, out_dir, log, deadline, seed, start,
                              previous["validation"] if previous else None, used)
    if warm_started:  # the cloned policy becomes the first best.zip, so PPO can only improve on it
        callback.init_callback(model)
        callback.evaluate()
    # The profile's `timesteps` is a total; SB3 treats it as "more steps" when not resetting.
    remaining = run["timesteps"] - start
    try:
        if time_left is not None and time_left <= 0:
            log(f"time budget {time_budget} already used ({used / 3600:.2f}h); nothing to train")
        elif remaining > 0:
            model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=not start,
                        tb_log_name="ppo", progress_bar=False)
        else:
            log(f"already at {start:,} steps (cap {run['timesteps']:,}); nothing to train")
    finally:
        vec_env.close()
    best = callback.best
    summary = f"best {best['metric']} = {best['value']:.4f}" if best else "no evaluation yet"
    log(f"done: {model.num_timesteps:,} steps; {summary} -> {out_dir / 'best.zip'}")
    return out_dir / "best.zip"


def train_all(args: argparse.Namespace) -> int:
    """One process per game, in parallel; output streams to the console with [game] prefixes."""
    games = sorted(GAME_SPECS) if args.games is None else args.games
    cores = os.cpu_count() or 4
    threads = max(1, cores // (2 * len(games)))
    processes = {}
    for game_id in games:
        command = [sys.executable, "-m", "card_games_rl.training.train", "--game", game_id, "--profile", args.profile,
                   "--seed", str(args.seed), "--torch-threads", str(threads)]
        if args.time_budget:
            command += ["--time-budget", args.time_budget]
        if args.resume:
            command.append("--resume")
        if args.n_envs:
            command += ["--n-envs", str(args.n_envs)]
        command += ["--device", args.device]
        processes[game_id] = subprocess.Popen(command, cwd=ROOT)
    failed = [g for g, p in processes.items() if p.wait() != 0]
    print("\nAll training processes finished." + (f" FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train card-game agents with MaskablePPO")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--game", choices=sorted(GAME_SPECS))
    group.add_argument("--all", action="store_true", help="Train every game in parallel processes")
    parser.add_argument("--games", nargs="+", choices=sorted(GAME_SPECS), help="With --all: only these games")
    parser.add_argument("--profile", default="smoke", choices=["smoke", "full"])
    parser.add_argument("--time-budget", default=None, help="Wall-clock cap per game, e.g. 3h, 90m")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Continue from models/<game>/checkpoints/latest.zip")
    parser.add_argument("--n-envs", type=int, default=None, help="Override the profile's parallel env count")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto = faster device per game from the training benchmark (CPU if not measured)")
    args = parser.parse_args()
    if args.all:
        return train_all(args)
    train(args.game, args.profile, args.time_budget, args.seed, args.resume, args.n_envs, args.torch_threads,
          device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
