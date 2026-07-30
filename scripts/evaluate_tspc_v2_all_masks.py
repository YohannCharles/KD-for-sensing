#!/usr/bin/env python3
"""Invoke the local V2 all-mask evaluator in the required Conda environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tools/configs/tspc_v2/stage_c_joint.yaml")
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
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
        "evaluate",
        "--method",
        args.method,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.checkpoint is not None:
        command.extend(("--checkpoint", str(args.checkpoint)))
    if args.limit is not None:
        command.extend(("--limit", str(args.limit)))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
