from __future__ import annotations

import argparse

from kd_sensing.cli.common import print_result
from kd_sensing.diagnostics.raymobtime_analysis import analyze_raymobtime_modality_imbalance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Raymobtime s008 modality imbalance from experiment runs.")
    parser.add_argument("--exp-dir", action="append", default=[], help="Experiment directory. Can be repeated.")
    parser.add_argument("--exp-root", help="Discover experiment runs under this directory.")
    parser.add_argument("--output-dir", help="Directory for analysis CSV/JSON outputs.")
    parser.add_argument("--matrix-config", help="Optional diagnosis matrix YAML for planned-run coverage checks.")
    return parser


def run(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    return analyze_raymobtime_modality_imbalance(
        exp_dirs=args.exp_dir,
        exp_root=args.exp_root,
        output_dir=args.output_dir,
        matrix_config=args.matrix_config,
    )


def main(argv: list[str] | None = None) -> None:
    result = run(argv)
    print_result(result)


def console_main(argv: list[str] | None = None) -> None:
    main(argv)


if __name__ == "__main__":
    console_main()
