"""Start the Card Games RL backend and open the web table once it is ready."""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = os.environ.get("CARD_GAMES_HOST", "127.0.0.1")
PORT = int(os.environ.get("CARD_GAMES_PORT", "8000"))
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
STARTUP_TIMEOUT = 60.0


def python_executable() -> str:
    """Prefer the project venv, whichever Python ran this script."""
    return str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


def server_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1) as response:
            return response.status == 200
    except OSError:
        return False


def main() -> int:
    url = f"http://{HOST}:{PORT}"
    if server_ready(url):
        print(f"Card Games RL is already running at {url}")
        webbrowser.open(url)
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [python_executable(), "-m", "uvicorn", "card_games_rl.api.app:app", "--host", HOST, "--port", str(PORT)]
    print(f"Starting Card Games RL at {url} with {command[0]}")
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while not server_ready(url):
            if process.poll() is not None:
                print(f"\nThe server exited with code {process.returncode}; see the error above.")
                return process.returncode or 1
            if time.monotonic() > deadline:
                print(f"\nThe server did not answer within {STARTUP_TIMEOUT:.0f}s.")
                process.terminate()
                return 1
            time.sleep(0.25)
        webbrowser.open(url)
        print(f"Opened {url}. Keep this window open; press Ctrl+C to stop the server.")
        return process.wait()
    except KeyboardInterrupt:
        print("Stopping the server...")
        process.terminate()
        process.wait(timeout=8)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
