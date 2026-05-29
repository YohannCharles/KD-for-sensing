from __future__ import annotations

import argparse
from pathlib import Path

from kd_sensing.cli.common import load_cli_config, print_result
from kd_sensing.data.layouts import deepsense6g_scene_layout
from kd_sensing.data.scenes import resolve_deepsense_scene
from kd_sensing.registries import PREPROCESSORS, import_default_components


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run KD sensing preprocessing from a YAML config.")
    parser.add_argument("--config", "-c", required=True, help="Path to a preprocessing YAML config.")
    parser.add_argument(
        "--action",
        choices=[
            "radar_fft_csv",
            "sequence_csv",
            "lidar_bev_cache",
            "image_derived_cache",
            "mmw_radar_maps",
            "raymobtime_s008_audit",
            "raymobtime_s008_index",
            "raymobtime_s008_ray_features",
            "raymobtime_s008_cache",
            "multimodal_nf_audit",
            "multimodal_nf_index",
            "multimodal_nf_derived_cache",
        ],
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
    args, unknown = parser.parse_known_args(argv)
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
