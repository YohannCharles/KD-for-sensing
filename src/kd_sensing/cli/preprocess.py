import argparse
from pathlib import Path

from kd_sensing.cli.common import load_cli_config, parse_cli_args, print_result
from kd_sensing.registries import PREPROCESSORS, import_default_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MMW preprocessing from a YAML config.")
    parser.add_argument("--config", "-c", help="Path to a preprocessing YAML config.")
    parser.add_argument(
        "--action",
        choices=[
            "mmw_radar_maps",
            "mmw_sequence_splits_from_manifest",
        ],
        help="Preprocessor name. Defaults to preprocessing.type from the config.",
    )
    parser.add_argument("--data-root", default="dataset/MMW/sunny", help="For MMW split materialization: condition root.")
    parser.add_argument("--scene", action="append", default=[], help="For MMW split materialization. Repeat for multiple scenes.")
    parser.add_argument("--seq-len", type=int, help="For MMW split materialization.")
    parser.add_argument("--pred-len", type=int, help="For MMW split materialization.")
    parser.add_argument("--split-tag", help="For MMW split materialization.")
    parser.add_argument("--split-seed", type=int, default=42, help="For MMW split materialization.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="For MMW split materialization.")
    parser.add_argument(
        "--split-strategy",
        choices=("group_safe_time_block",),
        default="group_safe_time_block",
        help="For MMW split materialization.",
    )
    parser.add_argument("--block-size-frames", type=int, help="For MMW split materialization.")
    parser.add_argument("--guard-band-frames", type=int, help="For MMW split materialization.")
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
    action = args.action
    if action == "mmw_sequence_splits_from_manifest":
        if args.override or unknown:
            parser.error("--override and bare key=value overrides require a config-backed preprocessing action.")
        print_result(_run_mmw_sequence_splits(args, parser))
        return
    if not args.config:
        parser.error("--config is required unless --action mmw_sequence_splits_from_manifest is used.")
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


def _run_mmw_sequence_splits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    from kd_sensing.data.mmw.preparation_splits import build_sequence_splits_from_manifest

    missing = [
        name
        for name in ("scene", "seq_len", "pred_len", "split_tag")
        if not getattr(args, name)
    ]
    if missing:
        parser.error("--action mmw_sequence_splits_from_manifest requires --" + ", --".join(missing).replace("_", "-"))
    reports = []
    for scene in args.scene:
        reports.append(
            build_sequence_splits_from_manifest(
                data_root=Path(args.data_root),
                scene=str(scene),
                seq_len=int(args.seq_len),
                pred_len=int(args.pred_len),
                split_tag=str(args.split_tag),
                split_seed=int(args.split_seed),
                train_ratio=float(args.train_ratio),
                split_strategy=str(args.split_strategy),
                block_size_frames=args.block_size_frames,
                guard_band_frames=args.guard_band_frames,
            )
        )
    return {"scenes": reports}


if __name__ == "__main__":
    main()
