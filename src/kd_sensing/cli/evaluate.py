import argparse

from kd_sensing.cli.common import bind_cli_mmw_protocol, load_cli_config, parse_cli_args, print_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate U0, DeepSense6G, or a retained baseline from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
    parser.add_argument("--weights", help="Model weights or checkpoint path to evaluate.")
    parser.add_argument("--output-dir", help="Directory for metrics and report outputs.")
    parser.add_argument("--split-seed", type=int, help="MMW trajectory split seed; defaults to the config value (0).")
    parser.add_argument("--evaluate-test", action="store_true", help="Explicitly evaluate the sealed MMW test split.")
    parser.add_argument("--override", "-o", action="append", default=[], help="Override dotted config key=value. Can be repeated.")
    return parser


def run(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parse_cli_args(parser, argv)

    cfg = load_cli_config(args, unknown, parser=parser)
    cfg.setdefault("runtime", {})["evaluate_test_requested"] = bool(args.evaluate_test)
    bind_cli_mmw_protocol(cfg, split_seed=args.split_seed)
    cfg.setdefault("runtime", {})["cli_config_path"] = args.config
    from kd_sensing.engine.evaluator import evaluate

    result = evaluate(cfg, weights=args.weights, output_dir=args.output_dir)
    print_result(result)
    return result


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
