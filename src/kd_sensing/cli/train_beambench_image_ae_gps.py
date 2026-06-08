from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from kd_sensing.baselines.beambench.image_ae_gps import run_image_ae_gps_training
from kd_sensing.config.io import deep_merge, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Arnold22 BeamBench Table III Camera AE + GPS Direct fusion baseline locally."
    )
    parser.add_argument("--config", "-c", default="configs/fusion/beambench_image_ae_gps_direct.yaml")
    parser.add_argument("--scene", type=int, default=None, help="DeepSense6G scene id, e.g. 31, 32, 33, or 34.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ae-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-ae-training", action="store_true", help="Require an existing AE checkpoint.")
    parser.add_argument(
        "--selection-split",
        choices=("test_as_validation", "validation"),
        default=None,
        help="Split used to select the best checkpoint.",
    )
    parser.add_argument("--fusion-val-fraction", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers for AE/cache/fusion stages.")
    parser.add_argument("--ae-batch-size", type=int, default=None)
    parser.add_argument("--fusion-batch-size", type=int, default=None)
    parser.add_argument("--feature-cache-batch-size", type=int, default=None)
    parser.add_argument(
        "--gps-feature-mode",
        choices=("paper_distance_angle", "paper_calibrated_relative_polar", "relative_polar"),
        default=None,
        help="GPS Direct feature mode for the BeamBench paper runner.",
    )
    parser.add_argument("--gps-angle-offset-rad", type=float, default=None)
    parser.add_argument("--target-beam-source", choices=("current", "future"), default=None)
    parser.add_argument("--no-feature-cache", action="store_true", help="Disable frozen Camera AE latent cache.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP for debugging or exact fp32 runs.")
    parser.add_argument("--no-tf32", action="store_true", help="Disable CUDA TF32 matmul/cuDNN acceleration.")
    parser.add_argument("--dry-run", action="store_true", help="Use tiny sample counts and one epoch for smoke testing.")
    parser.add_argument("--override", "-o", action="append", default=[])
    return parser


def run_main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    if args.scene is not None:
        overrides.append(f"data.dataset.scene={int(args.scene)}")
        if args.output_dir is None:
            overrides.append(f"output.run_name=beambench_image_ae_gps_direct_scene{int(args.scene)}")
    if args.output_dir is not None:
        overrides.append(f"beambench_paper.output_dir={args.output_dir.as_posix()}")
    if args.ae_checkpoint is not None:
        overrides.append(f"beambench_paper.ae_checkpoint_path={args.ae_checkpoint.as_posix()}")
    if args.skip_ae_training:
        overrides.append("beambench_paper.auto_train_ae=false")
    if args.selection_split is not None:
        overrides.append(f"beambench_paper.selection_split={args.selection_split}")
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
    if args.no_feature_cache:
        overrides.append("beambench_paper.cache_frozen_ae_features=false")
    if args.no_amp:
        overrides.append("beambench_paper.amp=false")
    if args.no_tf32:
        overrides.append("beambench_paper.allow_tf32=false")
    if args.dry_run:
        overrides.append("beambench_paper.dry_run=true")
    cfg = _load_config(args.config, overrides)
    return run_image_ae_gps_training(cfg)


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
