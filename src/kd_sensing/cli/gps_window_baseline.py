from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.baselines.gps_window.runner import run_gps_window_baseline
from kd_sensing.baselines.gps_window.types import normalize_scenarios
from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run GPS window baseline beam prediction.")
    parser.add_argument("--config", "-c", default="configs/baselines/gps_window_smoke.yaml")
    parser.add_argument("--scenes", default=None, help="Comma-separated scenes to evaluate.")
    parser.add_argument("--source-scenes", default=None, help="Comma-separated source calibration scenes.")
    parser.add_argument("--target-scenes", default=None, help="Comma-separated target scenes.")
    parser.add_argument("--sweep", action="store_true", help="Run deterministic parameter grid.")
    parser.add_argument("--execute", action="store_true", help="Execute predictions instead of writing plan only.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    result = run_main(argv)
    print(json.dumps(result, indent=2))
    return 0


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    cfg = _load_baseline_config(args.config, overrides)
    return run_gps_window_baseline(
        cfg,
        scenes=normalize_scenarios(args.scenes),
        source_scenes=normalize_scenarios(args.source_scenes),
        target_scenes=normalize_scenarios(args.target_scenes),
        execute=bool(args.execute),
        sweep=bool(args.sweep),
        output_dir=args.output_dir,
    )


def _load_baseline_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    payload = safe_load_yaml(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"GPS baseline config must be a mapping: {path}")
    if overrides:
        payload = deep_merge(payload, parse_overrides(overrides))
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

