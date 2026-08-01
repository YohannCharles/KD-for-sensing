import argparse

from kd_sensing.cli.common import load_cli_config, parse_cli_args, print_result
from kd_sensing.registries import PREPROCESSORS, import_default_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMW preprocessing from a YAML config.")
    parser.add_argument("--config", "-c", help="Path to a preprocessing YAML config.")
    parser.add_argument(
        "--action",
        choices=["mmw_radar_maps"],
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, unknown = parse_cli_args(parser, argv)
    if not args.config:
        parser.error("--config is required.")
    cfg = load_cli_config(args, unknown, parser=parser)
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

if __name__ == "__main__":
    main()
