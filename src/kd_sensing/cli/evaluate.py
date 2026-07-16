import argparse

from kd_sensing.cli.common import load_cli_config, parse_cli_args, print_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a T2 or baseline model from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to a YAML config file.")
    parser.add_argument("--weights", help="Model weights or checkpoint path to evaluate.")
    parser.add_argument("--output-dir", help="Directory for metrics and report outputs.")
    parser.add_argument("--override", "-o", action="append", default=[], help="Override dotted config key=value. Can be repeated.")
    return parser


def run(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, unknown = parse_cli_args(parser, argv)

    cfg = load_cli_config(args, unknown, parser=parser)
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
