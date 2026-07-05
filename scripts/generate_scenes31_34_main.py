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
DEFAULT_OUT_DIR = "outputs/scenes31_34_main_lmdb/generated_configs"
DEFAULT_OUTPUT_DIR = "outputs/scenes31_34_main_lmdb"
MODALITIES = ["image", "radar", "gps", "lidar"]
DEFAULT_SCENES = [31, 32, 33, 34]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Scene31-34 main experiment configs and manifest.")
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenes", default="31,32,33,34")
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args(argv)

    rows = write_scene31_manifest_configs(
        specs=_specs(_parse_scenes(args.scenes)),
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
    print(f"Wrote {len(rows)} Scene31-34 main manifest rows to {args.out_dir}.")
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
        loss: dict[str, Any] | None = None,
        priority: str = "high",
        execution_mode: str = "train",
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
                "loss": loss or {"u_mask_beam_jepa": {"enabled": False}},
                "extra": scene_extra,
                "priority": priority,
                "execution_mode": execution_mode,
            }
        )

    prefix = "scenes31_34"
    for seed in (1, 2, 3, 4, 5):
        add("proto_all", f"{prefix}_proto_natural_es40", seed, ["scenes31_34", "proto", "natural", "es40"])
        add(
            "proto_all",
            f"{prefix}_proto_sampler_uniform_es40",
            seed,
            ["scenes31_34", "proto", "sampler", "uniform", "es40"],
            training={"missing_pattern_sampler": "uniform"},
        )
        add(
            "proto_all",
            f"{prefix}_proto_randomdrop_bernoulli_k075_es40",
            seed,
            ["scenes31_34", "proto", "randomdrop_bernoulli", "keep_prob_0.75", "es40"],
            training={"random_modality_dropout": _randomdrop("bernoulli", keep_prob=0.75)},
        )
        add(
            "proto_all",
            f"{prefix}_proto_randomdrop_subset_es40",
            seed,
            ["scenes31_34", "proto", "randomdrop_subset", "es40"],
            training={"random_modality_dropout": _randomdrop("random_nonempty_subset")},
        )

    classifier_training = {
        "use_beam_prototype_alignment": False,
        "lambda_proto": 0.0,
        "lambda_modality_proto": 0.0,
        "use_beam_topology_proto": False,
        "use_full_aux_loss": False,
        "use_full_to_partial_kd": False,
        "kd_teacher_mode": "disabled",
        "training_mode": "pooled_multi_scene",
        "head_type": "classifier",
    }
    classifier_model = {
        "use_beam_prototype_alignment": False,
        "use_full_to_partial_kd": False,
        "kd_teacher_mode": "disabled",
        "head_type": "classifier",
        "paper_metadata": {
            "baseline_scope": "classifier_baseline",
            "baseline_family": "ordinary_classifier",
            "prototype_head_enabled": False,
        },
    }
    classifier_loss = {
        "type": "cross_entropy",
        "u_mask_beam_jepa": {
            "enabled": False,
            "use_beam_prototype_alignment": False,
        },
    }
    for seed in (1, 2, 3):
        add(
            "classifier_seed123",
            f"{prefix}_classifier_natural_es40",
            seed,
            ["scenes31_34", "classifier", "natural", "cross_entropy", "es40"],
            training=classifier_training,
            model=classifier_model,
            loss=classifier_loss,
            priority="medium",
        )
        add(
            "classifier_seed123",
            f"{prefix}_classifier_randomdrop_subset_es40",
            seed,
            ["scenes31_34", "classifier", "randomdrop_subset", "cross_entropy", "es40"],
            training={**classifier_training, "random_modality_dropout": _randomdrop("random_nonempty_subset")},
            model=classifier_model,
            loss=classifier_loss,
            priority="medium",
        )
        add(
            "classifier_optional",
            f"{prefix}_classifier_sampler_uniform_es40",
            seed,
            ["scenes31_34", "classifier", "sampler", "uniform", "cross_entropy", "es40"],
            training={**classifier_training, "missing_pattern_sampler": "uniform"},
            model=classifier_model,
            loss=classifier_loss,
            priority="low",
            execution_mode="optional_config",
        )
        add(
            "classifier_optional",
            f"{prefix}_classifier_randomdrop_bernoulli_k075_es40",
            seed,
            ["scenes31_34", "classifier", "randomdrop_bernoulli", "keep_prob_0.75", "cross_entropy", "es40"],
            training={**classifier_training, "random_modality_dropout": _randomdrop("bernoulli", keep_prob=0.75)},
            model=classifier_model,
            loss=classifier_loss,
            priority="low",
            execution_mode="optional_config",
        )

    for seed in (1, 2, 3):
        for family, core in (("amr_lite", _amr_core()), ("amber_lite", _amber_core())):
            for sampler, training in (("natural", {}), ("uniform", {"random_modality_dropout": _pattern_balanced_dropout()})):
                add(
                    "external_optional",
                    f"{prefix}_{family}_{sampler}_es40",
                    seed,
                    ["scenes31_34", family, sampler, "maskfix_optional", "es40"],
                    training=training,
                    model=_modular_model(core, family),
                    loss=_modular_loss(),
                    priority="low",
                    execution_mode="optional_config",
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


def _randomdrop(mode: str, *, keep_prob: float | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "modalities": MODALITIES,
        "ensure_at_least_one_modality": True,
    }
    if keep_prob is not None:
        cfg["keep_prob"] = keep_prob
    return cfg


def _pattern_balanced_dropout() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "pattern_balanced",
        "modalities": MODALITIES,
        "patterns": ["full", "missing_gps", "missing_radar", "radar_only", "lidar_only"],
        "ensure_at_least_one_modality": True,
    }


def _modular_model(core: dict[str, Any], family: str) -> dict[str, Any]:
    return {
        "type": "modular_sequence",
        "modalities": MODALITIES,
        "image_profile": "rgb_imagenet",
        "image_channels": 3,
        "radar_channels": 2,
        "gps_input_size": 3,
        "lidar_channels": 3,
        "feature_size": 64,
        "d_model": 64,
        "num_classes": 64,
        "num_pred": 1,
        "consume_missing_modality_metadata": True,
        "paper_metadata": {
            "baseline_scope": "local_experimental_baseline",
            "baseline_family": family,
            "maskfix_required": True,
        },
        "encoders": {
            "image": {"type": "resnet18_imagenet_rgb", "pretrained": False, "weights": None, "freeze_backbone": True},
            "radar": {"type": "radar_cnn"},
            "gps": {"type": "gps_mlp"},
            "lidar": {"type": "lidar_cnn"},
        },
        "representation_core": core,
        "heads": {"beam": {"type": "beam_head"}},
    }


def _amr_core() -> dict[str, Any]:
    return {
        "type": "amr_lite",
        "d_model": 64,
        "hidden_dim": 64,
        "dropout": 0.1,
        "imputation_type": "learnable_token",
    }


def _amber_core() -> dict[str, Any]:
    return {
        "type": "amber_lite_missing_modality_transformer",
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.1,
        "max_seq_len": 8,
        "mask_token_strategy": "learned_per_modality",
    }


def _modular_loss() -> dict[str, Any]:
    return {
        "type": "focal_loss",
        "alpha": 1,
        "gamma": 2,
        "u_mask_beam_jepa": {"enabled": False},
    }


def _parse_scenes(value: str) -> list[int]:
    scenes = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    return scenes or list(DEFAULT_SCENES)


if __name__ == "__main__":
    raise SystemExit(main())
