#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import summarize_scene31_bc_next


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    forwarded = ["--manifest", args.manifest, "--out", args.out, "--name-prefix", ""]
    for root in _roots(args):
        forwarded.extend(["--root", str(root)])
    for metrics in args.metrics:
        forwarded.extend(["--metrics", metrics])
    return summarize_scene31_bc_next.main(forwarded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31 apples/BC/weak-beamsoft fresh eval metrics.")
    parser.add_argument("--bc-root", default="outputs/scene31_bc_next_lmdb")
    parser.add_argument("--weak-root", default="outputs/scene31_beamsoft_weak_lmdb")
    parser.add_argument("--uniform-root", default="outputs/scene31_next_round")
    parser.add_argument("--metrics", action="append", default=[])
    parser.add_argument("--manifest", default="configs/scene31/next_round/experiment_manifest.csv")
    parser.add_argument("--out", default="outputs/scene31_beamsoft_weak_lmdb/summary")
    return parser


def _roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    for value, extras in (
        (args.bc_root, ("fresh_eval_main", "fresh_eval_main/apples_uniform")),
        (args.weak_root, ("fresh_eval",)),
        (args.uniform_root, ("p0_fresh_eval",)),
    ):
        root = Path(value)
        roots.append(root)
        for extra in extras:
            candidate = root / extra
            if candidate.exists():
                roots.append(candidate)
    out: list[Path] = []
    for root in roots:
        if root not in out:
            out.append(root)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
