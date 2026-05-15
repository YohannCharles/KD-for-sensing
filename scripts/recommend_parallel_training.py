#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config
from kd_sensing.engine.throughput_recommendations import recommend_parallel_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recommend overrides for background parallel training runs.")
    parser.add_argument("--config", "-c", required=True, help="Training YAML config.")
    parser.add_argument("--parallel-runs", type=int, default=4, help="Number of concurrent training runs.")
    parser.add_argument("--cpu-count", type=int, help="CPU cores available to the concurrent runs.")
    parser.add_argument("--cache-min-coverage", type=float, default=0.95, help="Coverage required before read_only cache.")
    parser.add_argument("--no-cache-check", action="store_true", help="Skip LiDAR cache coverage checks.")
    parser.add_argument("--output", help="Write JSON recommendation to this path.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args, unknown = build_parser().parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    cfg = load_config(args.config, overrides)
    result = recommend_parallel_training(
        cfg,
        config_path=args.config,
        parallel_runs=args.parallel_runs,
        cpu_count=args.cpu_count,
        cache_min_coverage=args.cache_min_coverage,
        check_cache=not args.no_cache_check,
    )
    payload = json.dumps(result, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return result


if __name__ == "__main__":
    main()
