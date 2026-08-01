import argparse

from kd_sensing.cli.common import bind_cli_mmw_protocol, load_cli_config, parse_cli_args, print_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train U0, DeepSense6G, or a retained baseline from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", help="Resume from an explicit checkpoint path.")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from checkpoints/last.pth.")
    parser.add_argument("--split-seed", type=int, help="MMW trajectory split seed; defaults to the config value (0).")
    parser.add_argument("--train-seed", type=int, help="Model initialization, shuffling, and dropout seed.")
    parser.add_argument("--evaluate-test", action="store_true", help="Explicitly load and evaluate the sealed MMW test split.")
    parser.add_argument("--num-workers", type=int, help="Set DataLoader workers.")
    parser.add_argument("--prefetch-factor", type=int, help="Set DataLoader prefetch factor.")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--override", "-o", action="append", default=[], help="Override dotted config key=value. Can be repeated.")
    return parser


def run(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parse_cli_args(parser, argv)

    cfg = load_cli_config(args, unknown, parser=parser)
    _apply_runtime_args(cfg, args)
    bind_cli_mmw_protocol(cfg, split_seed=args.split_seed)
    cfg.setdefault("runtime", {})["cli_config_path"] = args.config
    from kd_sensing.engine.trainer import train

    result = train(cfg)
    print_result(result)
    return result


def _apply_runtime_args(cfg: dict, args: argparse.Namespace) -> None:
    training = cfg.setdefault("training", {})
    loader = cfg.setdefault("data", {}).setdefault("dataloader", {})
    cfg.setdefault("runtime", {})["evaluate_test_requested"] = bool(args.evaluate_test)
    if args.train_seed is not None:
        if int(args.train_seed) < 0:
            raise ValueError("--train-seed must be non-negative.")
        cfg.setdefault("experiment", {}).update(seed=int(args.train_seed), train_seed=int(args.train_seed))
    if args.resume and args.auto_resume:
        raise ValueError("Use either --resume PATH or --auto-resume, not both.")
    if args.resume:
        training["resume"] = args.resume
    elif args.auto_resume:
        training["resume"] = True
    if args.num_workers is not None:
        loader["num_workers"] = int(args.num_workers)
    if args.prefetch_factor is not None:
        loader["prefetch_factor"] = int(args.prefetch_factor)
    if args.persistent_workers and args.no_persistent_workers:
        raise ValueError("Use either --persistent-workers or --no-persistent-workers, not both.")
    if args.persistent_workers or args.no_persistent_workers:
        loader["persistent_workers"] = bool(args.persistent_workers)
    if args.pin_memory and args.no_pin_memory:
        raise ValueError("Use either --pin-memory or --no-pin-memory, not both.")
    if args.pin_memory or args.no_pin_memory:
        loader["pin_memory"] = bool(args.pin_memory)


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
