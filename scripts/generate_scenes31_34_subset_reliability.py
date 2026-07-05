#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scene31_generator_common import DEFAULT_BASE_CONFIG, truthy, write_scene31_manifest_configs


EXPECTED_EPOCHS = 40
DEFAULT_OUT_DIR = "configs/scene31/scenes31_34_subset_reliability"
DEFAULT_OUTPUT_DIR = "outputs/scenes31_34_subset_reliability_lmdb"
MODALITIES = ["image", "radar", "gps", "lidar"]
DEFAULT_SCENES = [31, 32, 33, 34]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Scene31-34 subset reliability configs and manifest.")
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenes", default="31,32,33,34")
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args(argv)

    scenes = _parse_scenes(args.scenes)
    rows = write_scene31_manifest_configs(
        specs=_specs(scenes),
        base_config=Path(args.base_config),
        out_dir=Path(args.out_dir),
        output_dir=str(args.output_dir),
        overwrite=truthy(args.overwrite),
        expected_epochs=EXPECTED_EPOCHS,
        fieldnames=[
            "run_name",
            "group",
            "config_path",
            "seed",
            "method_tags",
            "expected_epochs",
            "priority",
            "execution_mode",
        ],
    )
    print(f"Wrote {len(rows)} Scene31-34 subset reliability manifest rows to {args.out_dir}.")
    return 0


def _specs(scenes: list[int]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    es40 = {"epochs": EXPECTED_EPOCHS, "max_epochs": EXPECTED_EPOCHS}
    scene_extra = {"data": {"dataset": _dataset_cfg(scenes)}}

    def add(
        group: str,
        name: str,
        seed: int,
        tags: list[str],
        *,
        training: dict[str, Any] | None = None,
        model: dict[str, Any] | None = None,
        priority: str = "high",
    ) -> None:
        specs.append(
            {
                "group": group,
                "name": name,
                "seed": seed,
                "tags": tags,
                "training": {**es40, **(training or {})},
                "model": {
                    "fusion_type": "weighted_sum",
                    "use_jepa_loss": False,
                    "use_beam_prototype_alignment": True,
                    "use_full_to_partial_kd": False,
                    "kd_teacher_mode": "disabled",
                    "use_mask_adapter": False,
                    **(model or {}),
                },
                "loss": {"u_mask_beam_jepa": {"enabled": False}},
                "extra": scene_extra,
                "priority": priority,
                "execution_mode": "train",
            }
        )

    randomdrop_subset = {"random_modality_dropout": _randomdrop_subset()}
    reliability_cfg = {
        "reliability_fusion": {
            "enabled": True,
            "mode": "mask_weighted",
            "hidden_dim": 16,
            "normalize_available_only": True,
            "missing_weight_zero": True,
            "log_weights": True,
        }
    }
    prefix = "scenes31_34"
    add("quick_seed1", f"{prefix}_proto_natural_es40", 1, ["scenes31_34", "proto", "natural", "es40"])
    add(
        "quick_seed1",
        f"{prefix}_proto_sampler_uniform_es40",
        1,
        ["scenes31_34", "proto", "sampler", "uniform", "es40"],
        training={"missing_pattern_sampler": "uniform"},
    )
    for seed in (1, 2, 3):
        group = "quick_seed1" if seed == 1 else "subset_vs_reliability_seed123"
        add(
            group,
            f"{prefix}_proto_randomdrop_subset_es40",
            seed,
            ["scenes31_34", "proto", "randomdrop_subset", "es40"],
            training=randomdrop_subset,
        )
        add(
            group,
            f"{prefix}_proto_randomdrop_subset_reliability_fusion_es40",
            seed,
            ["scenes31_34", "proto", "randomdrop_subset", "reliability_fusion", "mask_weighted", "es40"],
            training=randomdrop_subset,
            model=reliability_cfg,
        )
    return specs


def _dataset_cfg(scenes: list[int]) -> dict[str, Any]:
    return {
        "type": "deepsense6g",
        "scene": int(scenes[0]),
        "scenes": scenes,
        "train_scenes": scenes,
        "validation_scenes": scenes,
        "test_scenes": scenes,
        "split_protocol": "stratified_80_10_10",
        "split_strategy": "stratified_by_target_beam_per_scene",
        "split_seed": 42,
        "split_source_splits": ["train", "test"],
        "split_fractions": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "train_csv_name": "train_seqs_RA_GPS_LIDAR.csv",
        "test_csv_name": "test_seqs_RA_GPS_LIDAR.csv",
        "sample_cache": {"enabled": False},
    }


def _randomdrop_subset() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "random_nonempty_subset",
        "modalities": MODALITIES,
        "ensure_at_least_one_modality": True,
    }


def _parse_scenes(value: str) -> list[int]:
    scenes = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    return scenes or list(DEFAULT_SCENES)


if __name__ == "__main__":
    raise SystemExit(main())
