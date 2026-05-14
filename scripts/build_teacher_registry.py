#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.modalities import MODALITY_ORDER  # noqa: E402
from kd_sensing.utils.teacher_registry import (  # noqa: E402
    build_teacher_registry,
    parse_key_value_floats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a teacher reliability registry for teacher-prior CRAF.")
    parser.add_argument("--teacher-root", required=True, help="Root containing per-modality teacher run directories.")
    parser.add_argument(
        "--output",
        default="outputs/scene31/teacher_registry.json",
        help="Output teacher registry JSON path.",
    )
    parser.add_argument("--scene", default="31", help="Scene id or slug. Defaults to 31.")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(MODALITY_ORDER),
        help="Modalities to include. Defaults to image radar gps lidar mmwave.",
    )
    parser.add_argument(
        "--prior-mode",
        choices=("manual", "metric"),
        default="metric",
        help="Prior source: manual values or validation metric-derived values.",
    )
    parser.add_argument(
        "--manual-prior",
        action="append",
        default=[],
        help="Manual prior as key=value, comma-separated key=value, or JSON object. May be repeated.",
    )
    parser.add_argument(
        "--metric-prior-weights",
        action="append",
        default=[],
        help="Metric prior weights as key=value, comma-separated key=value, or JSON object.",
    )
    parser.add_argument("--prior-min", type=float, default=0.05, help="Minimum prior after clamping.")
    parser.add_argument("--prior-max", type=float, default=0.95, help="Maximum prior after clamping.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_teacher_registry(
        teacher_root=args.teacher_root,
        output_path=args.output,
        scene=args.scene,
        modalities=args.modalities,
        prior_mode=args.prior_mode,
        manual_prior=parse_key_value_floats(args.manual_prior),
        metric_prior_weights=parse_key_value_floats(args.metric_prior_weights),
        prior_min=args.prior_min,
        prior_max=args.prior_max,
    )
    print(f"Wrote teacher registry: {args.output}")
    print(f"Modalities: {', '.join(registry['modalities'])}")
    print(f"Prior mode: {registry['prior_mode']}")


if __name__ == "__main__":
    main()
