from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.deepsense6g_camera_residual import build_camera_residual_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DeepSense6G camera residual manifest.")
    parser.add_argument("--config", "-c", default="configs/deepsense6g_camera_residual.yaml")
    parser.add_argument("--support-ratio", type=float, default=None)
    parser.add_argument("--label-space", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--override", "-o", action="append", default=[])
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    cfg = _load_config(args.config, overrides)
    return build_camera_residual_manifest(
        cfg,
        support_ratio=args.support_ratio,
        label_space=args.label_space,
        output_dir=args.output_dir,
    )


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


def _load_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    payload = safe_load_yaml(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Camera residual config must be a mapping: {path}")
    if overrides:
        payload = deep_merge(payload, parse_overrides(overrides))
    return payload


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
