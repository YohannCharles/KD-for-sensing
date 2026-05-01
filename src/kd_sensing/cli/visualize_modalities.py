from __future__ import annotations

import argparse

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.diagnostics import visualize_modalities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize processed DeepSense6G modality tensors.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
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
    result = visualize_modalities(cfg)
    print_result(result)
    return result


if __name__ == "__main__":
    main()
