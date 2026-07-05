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
DEFAULT_OUT_DIR = "configs/scene31/subset_reliability"
DEFAULT_OUTPUT_DIR = "outputs/scene31_subset_reliability_lmdb"
MODALITIES = ["image", "radar", "gps", "lidar"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Scene31 subset reliability/PatternFiLM configs and manifest.")
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args(argv)

    rows = write_scene31_manifest_configs(
        specs=_specs(),
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
    print(f"Wrote {len(rows)} subset reliability manifest rows to {args.out_dir}.")
    return 0


def _specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    es40 = {"epochs": EXPECTED_EPOCHS, "max_epochs": EXPECTED_EPOCHS}
    subset = {"random_modality_dropout": _randomdrop_subset()}

    def add(
        group: str,
        name: str,
        seed: int,
        tags: list[str],
        *,
        model: dict[str, Any] | None = None,
        priority: str = "high",
    ) -> None:
        specs.append(
            {
                "group": group,
                "name": name,
                "seed": seed,
                "tags": tags,
                "training": {**es40, **subset},
                "model": {
                    "fusion_type": "weighted_sum",
                    "use_jepa_loss": False,
                    "use_beam_prototype_alignment": True,
                    "use_full_to_partial_kd": False,
                    "kd_teacher_mode": "disabled",
                    "use_mask_adapter": False,
                    **(model or {}),
                },
                "loss": {
                    "u_mask_beam_jepa": {
                        "enabled": False,
                    }
                },
                "priority": priority,
                "execution_mode": "train",
            }
        )

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
    film_cfg = {
        "pattern_film": {
            "enabled": True,
            "dim": 8,
            "init_identity": True,
            "apply_at": "pre_head",
        }
    }
    for seed in (1, 2, 3):
        add(
            "reliability",
            "proto_randomdrop_subset_reliability_fusion_es40",
            seed,
            ["proto", "randomdrop_subset", "reliability_fusion", "mask_weighted", "es40"],
            model=reliability_cfg,
        )
    for seed in (4, 5):
        add(
            "reliability_seed45",
            "proto_randomdrop_subset_reliability_fusion_es40",
            seed,
            ["proto", "randomdrop_subset", "reliability_fusion", "mask_weighted", "es40", "explicit_seed45"],
            model=reliability_cfg,
            priority="medium",
        )
    for seed in (1, 2, 3):
        add(
            "subset_film",
            "proto_randomdrop_subset_pattern_film_d8_es40",
            seed,
            ["proto", "randomdrop_subset", "pattern_film", "d8", "es40"],
            model=film_cfg,
        )
    specs.append(
        {
            "group": "optional_combo",
            "name": "proto_randomdrop_subset_reliability_fusion_pattern_film_d8_es40",
            "seed": 1,
            "tags": ["optional", "proto", "randomdrop_subset", "reliability_fusion", "pattern_film", "d8", "es40"],
            "training": {**es40, **subset},
            "model": {
                "fusion_type": "weighted_sum",
                "use_jepa_loss": False,
                "use_beam_prototype_alignment": True,
                "use_full_to_partial_kd": False,
                "kd_teacher_mode": "disabled",
                "use_mask_adapter": False,
                **reliability_cfg,
                **film_cfg,
            },
            "loss": {"u_mask_beam_jepa": {"enabled": False}},
            "priority": "low",
            "execution_mode": "generate_only",
        }
    )
    return specs


def _randomdrop_subset() -> dict[str, Any]:
    return {
        "enabled": True,
        "mode": "random_nonempty_subset",
        "modalities": MODALITIES,
        "ensure_at_least_one_modality": True,
    }


if __name__ == "__main__":
    raise SystemExit(main())
