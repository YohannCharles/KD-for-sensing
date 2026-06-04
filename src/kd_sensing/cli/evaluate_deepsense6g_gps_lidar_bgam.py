from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.engine.deepsense6g_gps_lidar_bgam import evaluate_deepsense6g_gps_lidar_bgam


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a DeepSense6G GPS+LiDAR BGAM checkpoint.")
    parser.add_argument("--config", "-c", default="configs/deepsense6g_gps_lidar_bgam.yaml")
    parser.add_argument("--ckpt", "--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--support-ratio", type=float, default=None)
    parser.add_argument("--label-space", default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--bgam-mode", default=None)
    parser.add_argument("--debug-masks", action="store_true")
    parser.add_argument("--override", "-o", action="append", default=[])
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = _load_config(args.config, list(args.override or []) + [item for item in unknown if "=" in item])
    return evaluate_deepsense6g_gps_lidar_bgam(
        cfg,
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        support_ratio=args.support_ratio,
        label_space=args.label_space,
        topk=args.topk,
        bgam_mode=args.bgam_mode,
        debug_masks=args.debug_masks if args.debug_masks else None,
    )


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


def _load_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    payload = safe_load_yaml(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"BGAM config must be a mapping: {path}")
    if overrides:
        payload = deep_merge(payload, parse_overrides(overrides))
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
