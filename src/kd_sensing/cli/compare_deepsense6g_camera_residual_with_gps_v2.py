from __future__ import annotations

import argparse
import json
from typing import Any

from kd_sensing.engine.deepsense6g_camera_residual import compare_deepsense6g_camera_residual_with_gps_v2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare DeepSense6G camera residual with GPS v2 baselines.")
    parser.add_argument("--gps-v2-root", default="outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep")
    parser.add_argument("--camera-root", default="outputs/analysis/deepsense6g_camera_residual/r15/mapping_disabled")
    parser.add_argument("--support-ratio", type=float, default=0.15)
    parser.add_argument("--label-space", default="mapping_disabled")
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return compare_deepsense6g_camera_residual_with_gps_v2(
        gps_v2_root=args.gps_v2_root,
        camera_root=args.camera_root,
        support_ratio=args.support_ratio,
        label_space=args.label_space,
    )


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
