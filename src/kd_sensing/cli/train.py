from __future__ import annotations

import argparse

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.engine.trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a KD sensing model from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one synthetic epoch by overriding dataset, epochs, workers, and output settings.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = load_cli_config(args, unknown)
    if args.dry_run:
        cfg["data"]["dataset"]["type"] = "synthetic"
        cfg["data"]["dataset"]["length"] = 2
        cfg["data"]["dataloader"]["train_batch_size"] = 1
        cfg["data"]["dataloader"]["test_batch_size"] = 1
        cfg["data"]["dataloader"]["num_workers"] = 0
        cfg["training"]["epochs"] = 1
        cfg["training"]["use_early_stopping"] = False
        cfg["distillation"]["type"] = "no_kd"
        cfg["distillation"]["teacher_model_name"] = None
        cfg["output"]["run_name"] = f"{cfg['experiment']['name']}_dry_run"
    result = train(cfg)
    print_result(result)
    return result


if __name__ == "__main__":
    main()
