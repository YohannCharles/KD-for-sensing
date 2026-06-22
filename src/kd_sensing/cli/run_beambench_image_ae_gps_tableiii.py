import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from kd_sensing.baselines.beambench.image_ae_gps_paper_split import (
    run_image_ae_gps_paper_split_evaluation,
    run_image_ae_gps_paper_split_training,
)
from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Camera AE + GPS Direct on scenes 32-34 and evaluate scenes 31-34 "
            "for local Arnold22 BeamBench Table III reproduction."
        )
    )
    parser.add_argument("--config", "-c", default="configs/fusion/beambench_image_ae_gps_direct.yaml")
    parser.add_argument("--train-scenes", type=int, nargs="+", default=[32, 33, 34])
    parser.add_argument("--eval-scenes", "--scenes", type=int, nargs="+", default=[31, 32, 33, 34])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/scenegroup_s32_s34/beambench_image_ae_gps_direct_tableiii/paper_split"),
    )
    parser.add_argument("--fusion-checkpoint", type=Path, default=None, help="Evaluate an existing paper-split checkpoint without retraining.")
    parser.add_argument("--selection-split", choices=("test_as_validation", "validation"), default="validation")
    parser.add_argument("--fusion-val-fraction", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ae-batch-size", type=int, default=None)
    parser.add_argument("--fusion-batch-size", type=int, default=None)
    parser.add_argument("--feature-cache-batch-size", type=int, default=None)
    parser.add_argument(
        "--gps-feature-mode",
        choices=("paper_distance_angle", "paper_calibrated_relative_polar", "relative_polar"),
        default=None,
        help="GPS Direct feature mode; paper_distance_angle matches the official challenge.py GPS input.",
    )
    parser.add_argument("--gps-angle-offset-rad", type=float, default=None)
    parser.add_argument(
        "--target-beam-source",
        choices=("current", "future"),
        default=None,
        help="Use current beamN labels for paper Table III, or future_beam1 for sequence-prediction ablations.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override", "-o", action="append", default=[])
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    cfg = _load_config(args.config, _overrides(args, unknown))
    if args.fusion_checkpoint is not None:
        return run_image_ae_gps_paper_split_evaluation(
            args.fusion_checkpoint,
            config=cfg,
            train_scenes=tuple(int(scene) for scene in args.train_scenes),
            eval_scenes=tuple(int(scene) for scene in args.eval_scenes),
            output_root=args.output_root,
        )
    return run_image_ae_gps_paper_split_training(
        cfg,
        train_scenes=tuple(int(scene) for scene in args.train_scenes),
        eval_scenes=tuple(int(scene) for scene in args.eval_scenes),
        output_root=args.output_root,
    )


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run_main(argv), indent=2, sort_keys=True))
    return 0


def _load_config(path: str | Path, overrides: list[str]) -> dict[str, Any]:
    payload = safe_load_yaml(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"BeamBench Image AE + GPS config must be a mapping: {path}")
    if overrides:
        payload = deep_merge(payload, parse_overrides(overrides))
    return payload


def _overrides(args: argparse.Namespace, unknown: Sequence[str]) -> list[str]:
    overrides = [
        f"beambench_paper.output_dir={Path(args.output_root).as_posix()}",
        f"beambench_paper.selection_split={args.selection_split}",
    ] + list(args.override or []) + [item for item in unknown if "=" in item]
    if args.fusion_val_fraction is not None:
        overrides.append(f"beambench_paper.fusion_val_fraction={float(args.fusion_val_fraction)}")
    if args.num_workers is not None:
        overrides.append(f"data.dataloader.num_workers={int(args.num_workers)}")
    if args.ae_batch_size is not None:
        overrides.append(f"beambench_paper.ae_batch_size={int(args.ae_batch_size)}")
    if args.fusion_batch_size is not None:
        overrides.append(f"beambench_paper.fusion_batch_size={int(args.fusion_batch_size)}")
    if args.feature_cache_batch_size is not None:
        overrides.append(f"beambench_paper.feature_cache_batch_size={int(args.feature_cache_batch_size)}")
    if args.gps_feature_mode is not None:
        overrides.append(f"beambench_paper.gps_feature_mode={args.gps_feature_mode}")
    if args.gps_angle_offset_rad is not None:
        overrides.append(f"beambench_paper.gps_angle_offset_rad={float(args.gps_angle_offset_rad)}")
    if args.target_beam_source is not None:
        overrides.append(f"beambench_paper.target_beam_source={args.target_beam_source}")
    if args.dry_run:
        overrides.append("beambench_paper.dry_run=true")
    return overrides


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
