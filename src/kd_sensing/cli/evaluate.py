from __future__ import annotations

import argparse

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.engine.evaluator import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a KD sensing model from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
    parser.add_argument("--weights", help="Model weights or checkpoint path to evaluate.")
    parser.add_argument("--output-dir", help="Directory for metrics and report outputs.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = load_cli_config(args, unknown)
    cfg.setdefault("runtime", {})["cli_config_path"] = args.config
    result = evaluate(cfg, weights=args.weights, output_dir=args.output_dir)
    print_result(result)
    return result


if __name__ == "__main__":
    main()
