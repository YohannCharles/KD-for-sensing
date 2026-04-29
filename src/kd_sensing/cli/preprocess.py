from __future__ import annotations

import argparse

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.registries import PREPROCESSORS, import_default_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KD sensing preprocessing from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to a preprocessing YAML config.")
    parser.add_argument(
        "--action",
        choices=["radar_fft_csv", "sequence_csv", "lidar_bev_cache", "image_motion_cache"],
        help="Preprocessor name. Defaults to preprocessing.type from the config.",
    )
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
    import_default_components()
    pre_cfg = dict(cfg.get("preprocessing", {}))
    if args.action:
        pre_cfg["type"] = args.action
    if "type" not in pre_cfg:
        parser.error("Preprocessing config must provide preprocessing.type or --action.")
    runner = PREPROCESSORS.build(pre_cfg)
    result = runner.run()
    payload = {"result": str(result)}
    print_result(payload)
    return payload


if __name__ == "__main__":
    main()
