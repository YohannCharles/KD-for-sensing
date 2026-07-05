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
DEFAULT_OUT_DIR = "configs/scene31/baseline_pack"
DEFAULT_OUTPUT_DIR = "outputs/scene31_baseline_pack_lmdb"
MODALITIES = ["image", "radar", "gps", "lidar"]
BALANCED_PATTERNS = ["full", "missing_gps", "missing_radar", "radar_only", "lidar_only"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Scene31 baseline pack configs and manifest.")
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args(argv)

    rows = write_scene31_manifest_configs(
        specs=_baseline_specs(),
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
    print(f"Wrote {len(rows)} baseline pack manifest rows to {args.out_dir}.")
    return 0


def _baseline_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    es40 = {"epochs": EXPECTED_EPOCHS, "max_epochs": EXPECTED_EPOCHS}

    def add(
        group: str,
        name: str,
        seed: int,
        tags: list[str],
        training: dict[str, Any] | None = None,
        *,
        model: dict[str, Any] | None = None,
        loss: dict[str, Any] | None = None,
        priority: str = "medium",
    ) -> None:
        specs.append(
            {
                "group": group,
                "name": name,
                "seed": seed,
                "tags": tags,
                "training": {**es40, **(training or {})},
                "model": model or {},
                "loss": loss or {},
                "priority": priority,
                "execution_mode": "train",
            }
        )

    for seed in (1, 2, 3):
        add("proto", "proto_natural_es40", seed, ["proto", "natural", "es40"], priority="high")
        add(
            "randomdrop",
            "proto_randomdrop_bernoulli_k075_es40",
            seed,
            ["proto", "randomdrop_bernoulli", "keep_prob_0.75", "es40"],
            {"random_modality_dropout": _randomdrop("bernoulli", keep_prob=0.75)},
            priority="high",
        )
        add(
            "randomdrop",
            "proto_randomdrop_subset_es40",
            seed,
            ["proto", "randomdrop_subset", "es40"],
            {"random_modality_dropout": _randomdrop("random_nonempty_subset")},
            priority="high",
        )

    for family, core in (("amr_lite", _amr_core()), ("amber_lite", _amber_core())):
        for seed in (1, 2, 3):
            add(
                family,
                f"{family}_natural_es40",
                seed,
                [family, "natural", "local_experimental_baseline", "es40"],
                {},
                model=_modular_model(core, family),
                loss=_modular_loss(),
                priority="high",
            )
            add(
                family,
                f"{family}_uniform_es40",
                seed,
                [family, "pattern_balanced_uniform", "local_experimental_baseline", "es40"],
                {"random_modality_dropout": _pattern_balanced_dropout()},
                model=_modular_model(core, family),
                loss=_modular_loss(),
                priority="high",
            )

    for seed in (1, 2, 3):
        add(
            "featuremod",
            "featuremod_lite_uniform_es40",
            seed,
            ["featuremod_lite", "pattern_balanced_uniform", "local_experimental_baseline", "es40"],
            {"random_modality_dropout": _pattern_balanced_dropout()},
            model=_modular_model(_featuremod_core(), "featuremod_lite"),
            loss=_modular_loss(),
        )
    return specs


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
        "patterns": BALANCED_PATTERNS,
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


def _featuremod_core() -> dict[str, Any]:
    return {
        "type": "featuremod_lite",
        "d_model": 64,
        "adapter_dim": 16,
        "condition": "missing_modalities",
    }


def _modular_loss() -> dict[str, Any]:
    return {
        "type": "focal_loss",
        "alpha": 1,
        "gamma": 2,
        "u_mask_beam_jepa": {"enabled": False},
    }


if __name__ == "__main__":
    raise SystemExit(main())
