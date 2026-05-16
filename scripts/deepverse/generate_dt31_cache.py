#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.data.deepverse import DeepVerseDT31Generator, DeepVerseDependencyError, DeepVerseLabelBuilder


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml(args.config)
    merged = merge_config(cfg, args)

    output_root = Path(merged["output_root"])
    generator = DeepVerseDT31Generator(
        scenario_root=merged["scenario_root"],
        scenario=merged["scenario"],
        config_m=merged["config_m"],
        scenes=merged["scenes"],
        enable_camera=merged["enable_camera"],
        enable_lidar=merged["enable_lidar"],
        enable_radar=merged["enable_radar"],
        enable_comm=merged["enable_comm"],
        enable_position=merged["enable_position"],
    )

    try:
        if args.dry_run:
            generator.validate_environment()
            print(json.dumps({"status": "ok", "message": "DeepVerse DT31 generation inputs are available."}))
            return 0

        dataset = generator.load_dataset(output_root=output_root)
        builder = DeepVerseLabelBuilder(
            dataset=dataset,
            scenario=merged["scenario"],
            scenario_root=merged["scenario_root"],
            scenes=merged["scenes"],
            seq_len=merged["seq_len"],
            pred_horizon=merged["pred_horizon"],
            num_beams=merged["num_beams"],
            beam_topk=merged["beam_topk"],
            position_noise_std=merged["position_noise_std"],
            seed=merged["seed"],
            ue_ids=merged["ue_ids"],
            bs_ids=merged["bs_ids"],
            camera_ids=merged["camera_ids"],
            lidar_ids=merged["lidar_ids"],
            enable_camera=merged["enable_camera"],
            enable_lidar=merged["enable_lidar"],
            enable_radar=merged["enable_radar"],
            blockage_min_class_count=merged["blockage_min_class_count"],
            blockage_min_class_ratio=merged["blockage_min_class_ratio"],
        )
        result = builder.write_cache(
            output_root,
            split_by=merged["split_by"],
            train_ratio=merged["train_ratio"],
            val_ratio=merged["val_ratio"],
        )
    except DeepVerseDependencyError as exc:
        print(f"DeepVerse DT31 generation cannot start: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"status": "ok", "output_root": str(output_root), "metadata": result["metadata"]}, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DeepVerse6G-DT31 Phase 1 cache artifacts.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deepverse/dt31_generation.yaml")
    parser.add_argument("--scenario-root", type=str)
    parser.add_argument("--scenario", type=str)
    parser.add_argument("--config-m", type=str)
    parser.add_argument("--output-root", type=str)
    parser.add_argument("--scenes", type=str, help="'all' or comma-separated scene ids.")
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--pred-horizon", type=int)
    parser.add_argument("--beam-codebook-size", type=int)
    parser.add_argument("--beam-topk", type=int)
    parser.add_argument("--position-noise-std", type=float)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument(
        "--split-by",
        choices=("sequence", "ue", "time_contiguous", "sample_random"),
        help="Default is leakage-safe sequence grouping; sample_random is debug-only and high leakage risk.",
    )
    parser.add_argument("--blockage-min-class-count", type=int)
    parser.add_argument("--blockage-min-class-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ue-ids", type=str, help="Comma-separated UE ids. Omit to let the builder infer.")
    parser.add_argument("--bs-ids", type=str, help="Comma-separated BS ids.")
    parser.add_argument("--camera-ids", type=str, help="Comma-separated camera device ids.")
    parser.add_argument("--lidar-ids", type=str, help="Comma-separated LiDAR device ids.")
    parser.add_argument("--enable-camera", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-lidar", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-radar", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-comm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-position", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate external dependency and scenario paths only.")
    return parser.parse_args(argv)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain a mapping: {path}")
    return payload


def merge_config(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    scenario_cfg = _mapping(cfg.get("scenario"))
    generation_cfg = _mapping(cfg.get("generation"))
    sequence_cfg = _mapping(cfg.get("sequence"))
    beam_cfg = _mapping(cfg.get("beam"))
    position_cfg = _mapping(cfg.get("position"))
    split_cfg = _mapping(cfg.get("split"))
    blockage_cfg = _mapping(cfg.get("blockage"))
    modality_cfg = _mapping(cfg.get("modalities"))
    output_cfg = _mapping(cfg.get("output"))
    scenario_root = args.scenario_root or scenario_cfg.get("root", "/root/datasets/DeepVerse/scenarios")
    scenario = args.scenario or scenario_cfg.get("name", "DT31")
    scene_value = args.scenes if args.scenes is not None else scenario_cfg.get("scenes", "all")

    return {
        "scenario_root": scenario_root,
        "scenario": scenario,
        "config_m": args.config_m or scenario_cfg.get("config_m"),
        "scenes": parse_scenes(scene_value, scenario_root=scenario_root, scenario=scenario),
        "output_root": args.output_root or output_cfg.get("root", "dataset/deepverse_dt31/cache"),
        "enable_camera": _override_bool(args.enable_camera, generation_cfg.get("enable_camera", True)),
        "enable_lidar": _override_bool(args.enable_lidar, generation_cfg.get("enable_lidar", True)),
        "enable_radar": _override_bool(args.enable_radar, generation_cfg.get("enable_radar", True)),
        "enable_comm": _override_bool(args.enable_comm, generation_cfg.get("enable_comm", True)),
        "enable_position": _override_bool(args.enable_position, generation_cfg.get("enable_position", True)),
        "seq_len": args.seq_len or int(sequence_cfg.get("seq_len", 8)),
        "pred_horizon": args.pred_horizon or int(sequence_cfg.get("pred_horizon", 3)),
        "num_beams": args.beam_codebook_size or int(beam_cfg.get("num_beams", 64)),
        "beam_topk": args.beam_topk or int(beam_cfg.get("topk", 5)),
        "position_noise_std": args.position_noise_std
        if args.position_noise_std is not None
        else float(position_cfg.get("noise_std", 1.0)),
        "split_by": args.split_by or split_cfg.get("split_by", "sequence"),
        "train_ratio": args.train_ratio if args.train_ratio is not None else float(split_cfg.get("train_ratio", 0.8)),
        "val_ratio": args.val_ratio if args.val_ratio is not None else float(split_cfg.get("val_ratio", 0.2)),
        "seed": args.seed if args.seed is not None else int(split_cfg.get("seed", 42)),
        "blockage_min_class_count": args.blockage_min_class_count
        if args.blockage_min_class_count is not None
        else int(blockage_cfg.get("min_class_count", 1)),
        "blockage_min_class_ratio": args.blockage_min_class_ratio
        if args.blockage_min_class_ratio is not None
        else float(blockage_cfg.get("min_class_ratio", 0.0)),
        "ue_ids": parse_int_list(args.ue_ids),
        "bs_ids": parse_int_list(args.bs_ids) or [0],
        "camera_ids": parse_int_list(args.camera_ids) or [int(v) for v in modality_cfg.get("camera_ids", [1])],
        "lidar_ids": parse_int_list(args.lidar_ids) or [int(v) for v in modality_cfg.get("lidar_ids", [1])],
    }


def parse_scenes(value: Any, *, scenario_root: str | Path | None = None, scenario: str = "DT31") -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "all":
        return _discover_scene_ids(scenario_root=scenario_root, scenario=scenario)
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return parse_int_list(str(value))


def _discover_scene_ids(*, scenario_root: str | Path | None, scenario: str) -> list[int] | None:
    if scenario_root is None:
        return None
    wireless_dir = Path(scenario_root).expanduser() / scenario / "wireless"
    scene_ids: list[int] = []
    for path in wireless_dir.glob("scene_*"):
        if not path.is_dir():
            continue
        try:
            scene_ids.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(scene_ids) or None


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _override_bool(value: bool | None, default: Any) -> bool:
    return bool(default) if value is None else bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
