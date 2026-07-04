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
DEFAULT_OUT_DIR = "configs/scene31/next_round"
DEFAULT_OUTPUT_DIR = "outputs/scene31_next_round"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Scene31 next-round es40 configs and manifest.")
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args(argv)

    base_config = Path(args.base_config)
    out_dir = Path(args.out_dir)
    rows = write_scene31_manifest_configs(
        specs=_next_round_specs(),
        base_config=base_config,
        out_dir=out_dir,
        output_dir=str(args.output_dir),
        overwrite=truthy(args.overwrite),
        expected_epochs=EXPECTED_EPOCHS,
    )
    print(f"Wrote {len(rows)} next-round manifest rows to {out_dir}.")
    return 0


def _next_round_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    es40 = {"epochs": EXPECTED_EPOCHS, "max_epochs": EXPECTED_EPOCHS}
    cond_base = {
        "use_btapa": True,
        "use_beam_prototype_alignment": True,
        "use_pattern_conditional_btapa": True,
        "btapa_tau_beam": 1.0,
        "btapa_lambda": 0.2,
        "btapa_modality_weight": 0.5,
        "btapa_disable_on_patterns": [],
        "btapa_fallback_to_ordinary_proto": True,
        "proto_target_type": "gaussian",
        "btapa_apply_patterns": ["radar_only", "lidar_only"],
    }
    uniform = {"missing_pattern_sampler": "uniform"}

    def add(
        group: str,
        name: str,
        seed: int,
        tags: list[str],
        training: dict[str, Any] | None = None,
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
            }
        )

    for seed in (3, 4, 5):
        add("p0", "proto_sampler_uniform_es40", seed, ["sampler", "uniform", "es40"], uniform, priority="high")
        add(
            "p0",
            "proto_condbtapa_weaksingle_lam005_es40",
            seed,
            ["condbtapa", "weak_single", "lambda_0.05", "es40"],
            {**cond_base, "btapa_lambda": 0.05},
            None,
            None,
            priority="high",
        )

    for lam_name, lam_value, group, priority in (
        ("lam005", 0.05, "p0", "high"),
        ("lam0025", 0.025, "p0", "high"),
        ("lam001", 0.01, "p0_optional", "optional"),
    ):
        for seed in (1, 2, 3):
            add(
                group,
                f"proto_sampler_uniform_condbtapa_weaksingle_{lam_name}_es40",
                seed,
                ["sampler", "uniform", "condbtapa", "weak_single", f"lambda_{lam_value:g}", "es40"],
                {**uniform, **cond_base, "btapa_lambda": lam_value},
                None,
                None,
                priority=priority,
            )

    adaptive_gap = {
        "missing_pattern_sampler": "adaptive_pattern",
        "adaptive_score_mode": "gap_to_full",
        "adaptive_alpha": 0.5,
        "adaptive_temperature": 1.0,
        "adaptive_ema_beta": 0.9,
        "adaptive_warmup_epochs": 3,
        "adaptive_min_prob": 0.05,
        "adaptive_max_prob": 0.40,
        "adaptive_update_freq": "step",
        "use_pattern_conditional_btapa": False,
        "use_weak_pattern_kd": False,
        "lambda_kd": 0.0,
    }
    beamsoft_s15 = {"type": "beam_neighborhood_ce", "sigma": 1.5, "mix_ce": 0.5, "circular": True}
    beamsoft_weak = {
        "s10": {"type": "beam_neighborhood_ce", "sigma": 1.0, "mix_ce": 0.25, "circular": True},
        "s15": {"type": "beam_neighborhood_ce", "sigma": 1.5, "mix_ce": 0.25, "circular": True},
    }
    labelsmooth = {"type": "label_smoothing_ce", "smoothing": 0.05}
    for seed in (1, 2, 3):
        add(
            "b_p0",
            "proto_sampler_adaptive_gap_a05_t1_es40",
            seed,
            ["sampler", "adaptive", "gap_to_full", "a0.5", "t1.0", "es40"],
            adaptive_gap,
            priority="high",
        )
        add(
            "c_p0",
            "proto_sampler_uniform_beamsoft_s15_mix05_es40",
            seed,
            ["sampler", "uniform", "beamsoft", "sigma_1.5", "mix_0.5", "es40"],
            uniform,
            loss=beamsoft_s15,
            priority="high",
        )
        add(
            "c_p0",
            "proto_sampler_uniform_labelsmooth005_es40",
            seed,
            ["sampler", "uniform", "label_smoothing", "smoothing_0.05", "es40"],
            uniform,
            loss=labelsmooth,
            priority="high",
        )
        add(
            "bc_p0",
            "proto_sampler_adaptive_gap_a05_t1_beamsoft_s15_mix05_es40",
            seed,
            ["sampler", "adaptive", "gap_to_full", "a0.5", "t1.0", "beamsoft", "sigma_1.5", "mix_0.5", "es40"],
            adaptive_gap,
            loss=beamsoft_s15,
            priority="high",
        )
        add(
            "b_p1",
            "proto_sampler_adaptive_loss_a05_t1_es40",
            seed,
            ["sampler", "adaptive", "loss", "a0.5", "t1.0", "es40"],
            {**adaptive_gap, "adaptive_score_mode": "loss"},
            priority="medium",
        )
        add(
            "b_p1",
            "proto_sampler_adaptive_gap_a03_t1_es40",
            seed,
            ["sampler", "adaptive", "gap_to_full", "a0.3", "t1.0", "es40"],
            {**adaptive_gap, "adaptive_alpha": 0.3},
            priority="optional",
        )
        for sigma_name, sigma in (("s10", 1.0), ("s20", 2.0)):
            add(
                "c_p1",
                f"proto_sampler_uniform_beamsoft_{sigma_name}_mix05_es40",
                seed,
                ["sampler", "uniform", "beamsoft", f"sigma_{sigma:g}", "mix_0.5", "es40"],
                uniform,
                loss={"type": "beam_neighborhood_ce", "sigma": sigma, "mix_ce": 0.5, "circular": True},
                priority="medium",
            )
        for sigma_name, loss_cfg in beamsoft_weak.items():
            add(
                "beamsoft_weak_p0",
                f"proto_sampler_uniform_beamsoft_{sigma_name}_mix025_es40",
                seed,
                ["sampler", "uniform", "beamsoft", f"sigma_{loss_cfg['sigma']:g}", "mix_0.25", "es40"],
                uniform,
                loss=loss_cfg,
                priority="high",
            )

    add(
        "p1",
        "proto_curriculum_easy2hard_es40",
        3,
        ["sampler", "curriculum_easy_to_hard", "es40"],
        {
            "missing_pattern_sampler": "curriculum_easy_to_hard",
            "curriculum_schedule": {
                "epochs_1_5": ["full", "missing_image", "missing_radar", "missing_lidar"],
                "epochs_6_10": ["missing_gps", "non_gps_only"],
                "epochs_11_40": ["gps_only", "image_only", "radar_only", "lidar_only"],
            },
        },
        priority="medium",
    )
    add(
        "p1",
        "proto_maskadapter_d16_condbtapa_weaksingle_es40",
        3,
        ["mask_adapter", "d16", "condbtapa", "weak_single", "es40"],
        cond_base,
        {"use_mask_adapter": True, "mask_adapter_dim": 16},
        None,
        priority="medium",
    )

    for seed in (1, 2):
        add(
            "p1_optional",
            "proto_sampler_uniform_curriculum_easy2hard_es40",
            seed,
            ["sampler", "uniform_schedule", "curriculum_easy_to_hard", "es40"],
            {
                "missing_pattern_sampler": "curriculum_easy_to_hard",
                "curriculum_schedule": {
                    "epochs_1_5": ["full", "missing_image", "missing_radar", "missing_lidar"],
                    "epochs_6_10": ["missing_gps", "non_gps_only"],
                    "epochs_11_40": ["gps_only", "image_only", "radar_only", "lidar_only"],
                },
            },
            None,
            None,
            priority="optional",
        )
        add(
            "p1_optional",
            "proto_sampler_uniform_maskadapter_d16_es40",
            seed,
            ["sampler", "uniform", "mask_adapter", "d16", "es40"],
            uniform,
            {"use_mask_adapter": True, "mask_adapter_dim": 16},
            None,
            priority="optional",
        )

    return specs


if __name__ == "__main__":
    raise SystemExit(main())
