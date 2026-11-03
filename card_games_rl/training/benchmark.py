"""Throughput benchmarks.

    python -m card_games_rl.training.benchmark                 # raw engine steps/s
    python -m card_games_rl.training.benchmark --training      # PPO steps/s on CPU vs GPU

``--training`` runs all games concurrently (as ``train --all`` does), once on
CPU and once on CUDA, and writes the faster device per game to
``training/device_benchmark.json``; ``train --device auto`` then uses it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from card_games_rl.envs import SingleAgentEnv
from card_games_rl.games import GAME_SPECS

ROOT = Path(__file__).resolve().parents[2]
CUDA_MIN_GAIN = 1.05  # pick CUDA only if it is at least 5% faster
MAX_CUDA_GAMES = 2    # GPU memory limit: CUDA contexts per 4 GB card


def measure_engine(game_id: str, seconds: float) -> float:
    env = SingleAgentEnv(game_id, seed=0)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        _, _, terminated, truncated, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        steps += 1
        if terminated or truncated:
            env.reset()
    return steps / (time.perf_counter() - start)


def measure_training(game_id: str, device: str, seconds: float, threads: int) -> float:
    """PPO steps/s with the full profile, excluding the first 25% (start-up / warm-up)."""
    import torch
    from stable_baselines3.common.callbacks import BaseCallback

    from .config import load_config
    from .train import build_model, make_vec_env

    torch.set_num_threads(threads)
    config = load_config(game_id, "full")
    marks: list[tuple[float, int]] = []

    class Stop(BaseCallback):
        def _on_step(self) -> bool:
            marks.append((time.perf_counter(), self.num_timesteps))
            return marks[-1][0] - marks[0][0] < seconds

    with tempfile.TemporaryDirectory() as pool:
        vec_env = make_vec_env(game_id, config, config["run"]["n_envs"], 0, Path(pool))
        try:
            build_model(config, vec_env, device, 0, None).learn(total_timesteps=10**9, callback=Stop())
        finally:
            vec_env.close()
    start = marks[0][0] + 0.25 * (marks[-1][0] - marks[0][0])
    warm = next(m for m in marks if m[0] >= start)
    return (marks[-1][1] - warm[1]) / max(1e-6, marks[-1][0] - warm[0])


def run_concurrently(devices: dict[str, str], seconds: float) -> dict[str, float]:
    """Run every game at once (each on its assigned device); steps/s per game (0 = failed)."""
    threads = max(1, (os.cpu_count() or 4) // (2 * len(devices)))
    processes = {
        g: subprocess.Popen([sys.executable, "-m", "card_games_rl.training.benchmark", "--worker", g, d,
                             str(seconds), str(threads)], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
        for g, d in devices.items()
    }
    results = {}
    for game_id, process in processes.items():
        output, errors = process.communicate()
        line = next((l for l in output.splitlines() if l.startswith("RESULT ")), None)
        results[game_id] = float(line.split()[1]) if line else 0.0
        if not line:
            last = errors.strip().splitlines()[-1] if errors.strip() else "no output"
            print(f"  {game_id} on {devices[game_id]} failed: {last}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure environment / training throughput")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--training", action="store_true", help="Benchmark PPO training on CPU vs CUDA")
    parser.add_argument("--worker", nargs=4, metavar=("GAME", "DEVICE", "SECONDS", "THREADS"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        game_id, device, seconds, threads = args.worker
        print(f"RESULT {measure_training(game_id, device, float(seconds), int(threads)):.1f}", flush=True)
        return
    if not args.training:
        for game_id in sorted(GAME_SPECS):
            print(f"{game_id:10s} {measure_engine(game_id, args.seconds or 3):10,.0f} env steps/s")
        return

    import torch

    from .train import DEVICE_FILE

    seconds = args.seconds or 90
    games = sorted(GAME_SPECS)
    print(f"Pass 1: all games on CPU (~{seconds:.0f}s)...", flush=True)
    cpu = run_concurrently({g: "cpu" for g in games}, seconds)
    cuda = {g: 0.0 for g in games}
    if torch.cuda.is_available():
        # Each process needs its own CUDA context (~0.5-0.7 GB), so a 4 GB laptop
        # GPU fits ~2 at once. Test games on the GPU in pairs while the rest
        # keep the CPU busy, mirroring real `train --all` contention.
        pairs = [games[i:i + MAX_CUDA_GAMES] for i in range(0, len(games), MAX_CUDA_GAMES)]
        for n, pair in enumerate(pairs, start=2):
            print(f"Pass {n}: {', '.join(pair)} on CUDA, others on CPU (~{seconds:.0f}s)...", flush=True)
            result = run_concurrently({g: ("cuda" if g in pair else "cpu") for g in games}, seconds)
            cuda.update({g: result[g] for g in pair})
    gains = sorted(((cuda[g] / cpu[g] if cpu[g] else 0.0, g) for g in games), reverse=True)
    on_gpu = {g for gain, g in gains[:MAX_CUDA_GAMES] if gain >= CUDA_MIN_GAIN}
    summary = {}
    print(f"\n{'game':10s} {'cpu':>10s} {'cuda':>10s}   best")
    for game_id in games:
        best = "cuda" if game_id in on_gpu else "cpu"
        summary[game_id] = {"cpu": cpu[game_id], "cuda": cuda[game_id], "best": best}
        print(f"{game_id:10s} {cpu[game_id]:10,.0f} {cuda[game_id]:10,.0f}   {best}")
    DEVICE_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved to {DEVICE_FILE.relative_to(ROOT)}; 'train --device auto' will use these choices.")


if __name__ == "__main__":
    main()
