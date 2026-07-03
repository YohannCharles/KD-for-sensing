#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from generate_experiment_grid import ROOT, _config_payload, _rel, _truthy


EXPECTED_EPOCHS = 40
DEFAULT_OUT_DIR = "configs/scene31/next_round"
DEFAULT_BASE_CONFIG = "configs/scene31/templates/main_v3_proto_es20_base.yaml"
DEFAULT_OUTPUT_DIR = "outputs/scene31_next_round"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Scene31 next-round es40 configs and manifest.")
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args()

    base_config = Path(args.base_config)
    out_dir = Path(args.out_dir)
    overwrite = _truthy(args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in _next_round_specs():
        run_name = f"{spec['name']}_seed{spec['seed']}"
        config_path = out_dir / f"{run_name}.yaml"
        if overwrite or not config_path.exists():
            payload = _config_payload(base_config, config_path, run_name, int(spec["seed"]), spec)
            payload.setdefault("output", {})["dir"] = str(args.output_dir)
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append(
            {
                "run_name": run_name,
                "group": spec["group"],
                "config_path": _rel(config_path),
                "seed": spec["seed"],
                "method_tags": ",".join(spec["tags"]),
                "expected_epochs": EXPECTED_EPOCHS,
                "priority": spec.get("priority", "medium"),
            }
        )

    fieldnames = ["run_name", "group", "config_path", "seed", "method_tags", "expected_epochs", "priority"]
    with (out_dir / "experiment_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "experiment_manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} next-round manifest rows to {out_dir}.")


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
                priority=priority,
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
            priority="optional",
        )
        add(
            "p1_optional",
            "proto_sampler_uniform_maskadapter_d16_es40",
            seed,
            ["sampler", "uniform", "mask_adapter", "d16", "es40"],
            uniform,
            {"use_mask_adapter": True, "mask_adapter_dim": 16},
            priority="optional",
        )

    return specs


if __name__ == "__main__":
    main()
