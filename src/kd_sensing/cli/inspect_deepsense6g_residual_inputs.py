from __future__ import annotations

import argparse
import json
from typing import Any

from kd_sensing.data.deepsense6g_residual import inspect_residual_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect DeepSense6G GPS v2 artifacts for residual fusion.")
    parser.add_argument("--gps-sweep-root", default="outputs/analysis/deepsense6g_gps_adapter_v2_support_sweep")
    parser.add_argument("--label-space", default="mapping_disabled")
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return inspect_residual_inputs(args.gps_sweep_root, label_space=args.label_space)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
