import argparse

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.engine.trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a sensing or beam prediction model from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", help="Resume from an explicit checkpoint path.")
    parser.add_argument("--auto-resume", "--auto_resume", action="store_true", help="Resume from this run's checkpoints/last.pth.")
    parser.add_argument("--num-workers", "--num_workers", type=int, help="Set train/test DataLoader workers.")
    parser.add_argument("--prefetch-factor", "--prefetch_factor", type=int, help="Set train/test DataLoader prefetch factor.")
    parser.add_argument("--persistent-workers", "--persistent_workers", action="store_true", help="Enable persistent DataLoader workers.")
    parser.add_argument("--no-persistent-workers", "--no_persistent_workers", action="store_true", help="Disable persistent DataLoader workers.")
    parser.add_argument("--pin-memory", "--pin_memory", action="store_true", help="Enable DataLoader pin_memory.")
    parser.add_argument("--no-pin-memory", "--no_pin_memory", action="store_true", help="Disable DataLoader pin_memory.")
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


def run(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = load_cli_config(args, unknown)
    _apply_training_cli_shortcuts(cfg, args)
    cfg.setdefault("runtime", {})["cli_config_path"] = args.config
    if args.dry_run:
        cfg["data"]["dataset"]["type"] = "synthetic"
        for key in ("train_scenes", "test_scenes", "eval_scenes", "validation_scenes"):
            cfg["data"]["dataset"].pop(key, None)
        cfg["data"]["dataset"]["length"] = 2
        cfg["data"]["dataloader"]["train_batch_size"] = 1
        cfg["data"]["dataloader"]["test_batch_size"] = 1
        cfg["data"]["dataloader"]["num_workers"] = 0
        cfg["training"]["epochs"] = 1
        cfg["training"]["use_early_stopping"] = False
        cfg["output"]["run_name"] = f"{cfg['experiment']['name']}_dry_run"
    result = train(cfg)
    print_result(result)
    return result


def _apply_training_cli_shortcuts(cfg: dict, args: argparse.Namespace) -> None:
    training = cfg.setdefault("training", {})
    loader = cfg.setdefault("data", {}).setdefault("dataloader", {})
    if args.resume and args.auto_resume:
        raise ValueError("Use either --resume PATH or --auto-resume, not both.")
    if args.resume:
        training["resume"] = args.resume
    elif args.auto_resume:
        training["resume"] = True
    if args.num_workers is not None:
        loader["num_workers"] = int(args.num_workers)
        loader["train_num_workers"] = int(args.num_workers)
        loader["test_num_workers"] = int(args.num_workers)
    if args.prefetch_factor is not None:
        loader["prefetch_factor"] = int(args.prefetch_factor)
        loader["train_prefetch_factor"] = int(args.prefetch_factor)
        loader["test_prefetch_factor"] = int(args.prefetch_factor)
    if args.persistent_workers and args.no_persistent_workers:
        raise ValueError("Use either --persistent-workers or --no-persistent-workers, not both.")
    if args.persistent_workers or args.no_persistent_workers:
        value = bool(args.persistent_workers)
        loader["persistent_workers"] = value
        loader["train_persistent_workers"] = value
        loader["test_persistent_workers"] = value
    if args.pin_memory and args.no_pin_memory:
        raise ValueError("Use either --pin-memory or --no-pin-memory, not both.")
    if args.pin_memory or args.no_pin_memory:
        value = bool(args.pin_memory)
        loader["pin_memory"] = value
        loader["train_pin_memory"] = value
        loader["test_pin_memory"] = value


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
