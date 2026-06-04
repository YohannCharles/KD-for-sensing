from __future__ import annotations

import argparse
import json
from typing import Any

from kd_sensing.engine.deepsense6g_residual_fusion import plot_deepsense6g_residual_fusion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot DeepSense6G residual fusion diagnostics.")
    parser.add_argument("--results-dir", default="outputs/analysis/deepsense6g_residual_fusion/r15/mapping_disabled")
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return plot_deepsense6g_residual_fusion(args.results_dir)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
