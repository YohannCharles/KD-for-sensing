#!/usr/bin/env python3
"""Build the local TSPC-V2 CSV/Markdown summary without touching outer test."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tools/configs/tspc_v2/stage_c_joint.yaml")
    args = parser.parse_args()
    command = [
        "conda",
        "run",
        "-n",
        "kd_mm_beam",
        "--no-capture-output",
        "python",
        str(ROOT / "tools/run_tspc_v2.py"),
        "--config",
        str(args.config),
        "summarize",
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
