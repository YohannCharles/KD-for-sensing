import argparse
import json
from pathlib import Path

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.data.mmw.preparation import (
    build_sequence_splits_from_manifest,
    load_preparation_config,
    prepare_town10_skybridge,
)
from kd_sensing.data.layouts import deepsense6g_scene_layout
from kd_sensing.data.scenes import resolve_deepsense_scene
from kd_sensing.registries import PREPROCESSORS, import_default_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KD sensing preprocessing from a YAML config.")
    parser.add_argument("--config", "-c", help="Path to a preprocessing YAML config.")
    parser.add_argument(
        "--action",
        choices=[
            "radar_fft_csv",
            "sequence_csv",
            "lidar_bev_cache",
            "image_derived_cache",
            "deepsense6g_sample_lmdb_cache",
            "mmw_radar_maps",
            "mmw_town10_skybridge",
            "mmw_sequence_splits_from_manifest",
        ],
        help="Preprocessor name. Defaults to preprocessing.type from the config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="For MMW Town10 preparation: resolve config without writes.")
    parser.add_argument("--force", action="store_true", help="For MMW Town10 preparation: rebuild derived artifacts.")
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
    parser.add_argument("--beam-label-calibration-json", help="For MMW split materialization.")
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
    args, unknown = parser.parse_known_args(argv)
    action = args.action
    if action == "mmw_town10_skybridge":
        if not args.config:
            parser.error("--action mmw_town10_skybridge requires --config.")
        overrides = list(args.override or []) + [item for item in unknown if "=" in item]
        config = load_preparation_config(args.config, overrides)
        print_result(prepare_town10_skybridge(config, dry_run=args.dry_run, force=args.force))
        return
    if action == "mmw_sequence_splits_from_manifest":
        print_result(_run_mmw_sequence_splits(args, parser))
        return
    if not args.config:
        parser.error("--config is required unless --action mmw_sequence_splits_from_manifest is used.")
    cfg = load_cli_config(args, unknown)
    import_default_components()
    pre_cfg = dict(cfg.get("preprocessing", {}))
    if args.action:
        pre_cfg["type"] = args.action
    _apply_scene_override_to_sequence_preprocess(pre_cfg, cfg)
    if "type" not in pre_cfg:
        parser.error("Preprocessing config must provide preprocessing.type or --action.")
    runner = PREPROCESSORS.build(pre_cfg)
    result = runner.run()
    payload = {"result": str(result)}
    print_result(payload)


def _run_mmw_sequence_splits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    missing = [
        name
        for name in ("scene", "seq_len", "pred_len", "split_tag")
        if not getattr(args, name)
    ]
    if missing:
        parser.error("--action mmw_sequence_splits_from_manifest requires --" + ", --".join(missing).replace("_", "-"))
    beam_label_calibration = (
        json.loads(args.beam_label_calibration_json) if args.beam_label_calibration_json else None
    )
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
                beam_label_calibration=beam_label_calibration,
            )
        )
    return {"scenes": reports}


def _apply_scene_override_to_sequence_preprocess(pre_cfg: dict, cfg: dict) -> None:
    if pre_cfg.get("type") != "sequence_csv":
        return
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return
    scene_value = dataset_cfg.get("scene", dataset_cfg.get("scene_id", dataset_cfg.get("scene_slug")))
    if scene_value is None:
        return
    scene = resolve_deepsense_scene(scene_value, dataset_type=dataset_cfg.get("type"))
    layout = deepsense6g_scene_layout(scene.scene_id)
    current_root = Path(str(pre_cfg.get("data_root", "")))
    current_csv = Path(str(pre_cfg.get("csv_path", "")))
    resolved_root = current_root if current_root.is_absolute() else Path(layout.canonical_root)
    if not current_root.is_absolute():
        pre_cfg["data_root"] = str(resolved_root)
    if current_csv.name.startswith("scenario") and current_csv.name.endswith("_RA.csv"):
        if not current_csv.is_absolute():
            pre_cfg["csv_path"] = str(resolved_root / layout.radar_csv_name)


if __name__ == "__main__":
    main()
