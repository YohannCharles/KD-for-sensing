from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.mmw_town_gps_lidar_bgam_manifest import build_mmw_town_gps_lidar_bgam_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare MMW Town GPS+LiDAR BGAM manifest.")
    parser.add_argument("--config", "-c", default="configs/mmw_town_gps_lidar_bgam.yaml")
    parser.add_argument("--label-space", choices=("mapping_enabled", "mapping_disabled"), default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument("--override", "-o", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args, unknown = build_parser().parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    cfg = _load_config(args.config, overrides)
    result = build_mmw_town_gps_lidar_bgam_manifest(
        cfg,
        label_space=args.label_space,
        topk=args.topk,
        output_dir=args.output_dir,
        input_manifest=args.input_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _load_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    payload = safe_load_yaml(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"MMW Town BGAM manifest config must be a mapping: {path}")
    return deep_merge(payload, parse_overrides(overrides)) if overrides else payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
