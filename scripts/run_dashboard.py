"""Launch the Streamlit dashboard for a given configuration."""
import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from turnaround.config.loader import ENV_HOME

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "turnaround" / "dashboard" / "app.py"


def main(argv: Optional[List[str]] = None) -> int:
    """Start ``streamlit run`` on the dashboard app.

    Args:
        argv: Argument list (None uses ``sys.argv[1:]``).
    Returns:
        The Streamlit process exit code.
    Logic:
        Export the chosen config path as ``TURNAROUND_CONFIG`` and the ``src`` directory on PYTHONPATH,
        then run ``python -m streamlit run <app.py> --server.port <port>``.
    Side effects:
        Spawns a subprocess and blocks until it exits.
    """
    p = argparse.ArgumentParser(description="Turnaround dashboard")
    p.add_argument("--config", default=None, help="YAML config used by the pipeline (default: configs/default.yaml)")
    p.add_argument("--port", type=int, default=8501)
    args = p.parse_args(argv)
    env = dict(os.environ)
    if args.config:
        env["TURNAROUND_CONFIG"] = str(Path(args.config).resolve())
    src = str(APP_PATH.parents[2])
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.port", str(args.port)]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
