from __future__ import annotations

import argparse
import json
from typing import Any

from kd_sensing.engine.deepsense6g_top8_selector import plot_deepsense6g_top8_selector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot DeepSense6G GPS Top8 selector diagnostics.")
    parser.add_argument("--results-dir", default="outputs/analysis/deepsense6g_top8_selector/r15/mapping_disabled")
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    return plot_deepsense6g_top8_selector(args.results_dir)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
