from __future__ import annotations

import argparse

from kd_sensing.cli.common import print_result
from kd_sensing.diagnostics.raymobtime_analysis import analyze_raymobtime_modality_imbalance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Raymobtime s008 modality imbalance from experiment runs.")
    parser.add_argument("--exp-dir", action="append", required=True, help="Experiment directory. Can be repeated.")
    parser.add_argument("--output-dir", help="Directory for analysis CSV/JSON outputs.")
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = analyze_raymobtime_modality_imbalance(exp_dirs=args.exp_dir, output_dir=args.output_dir)
    print_result(result)
    return result


if __name__ == "__main__":
    main()
