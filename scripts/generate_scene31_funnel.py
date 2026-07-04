#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_experiment_grid import _config_payload, _rel, _truthy

EXPECTED_EPOCHS = 40
DEFAULT_OUT_DIR = "configs/scene31/funnel"
DEFAULT_BASE_CONFIG = "configs/scene31/templates/main_v3_proto_es20_base.yaml"
DEFAULT_OUTPUT_DIR = "outputs/scene31_funnel_lmdb"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Scene31 funnel configs and manifest.")
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args(argv)

    base_config = Path(args.base_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overwrite = _truthy(args.overwrite)

    rows: list[dict[str, Any]] = []
    for spec in _funnel_specs():
        run_name = spec["name"] if spec.get("seed") is None else f"{spec['name']}_seed{spec['seed']}"
        config_path = out_dir / f"{run_name}.yaml"
        if spec.get("execution_mode") != "selection" and (overwrite or not config_path.exists()):
            payload = _config_payload(base_config, config_path, run_name, int(spec.get("seed") or 1), spec)
            payload.setdefault("output", {})["dir"] = str(args.output_dir)
            for key, value in spec.get("extra", {}).items():
                payload[key] = value
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        rows.append(
            {
                "run_name": run_name,
                "group": spec["group"],
                "config_path": "" if spec.get("execution_mode") == "selection" else _rel(config_path),
                "seed": "" if spec.get("seed") is None else spec["seed"],
                "method_tags": ",".join(spec["tags"]),
                "expected_epochs": spec.get("expected_epochs", EXPECTED_EPOCHS),
                "priority": spec.get("priority", "medium"),
                "execution_mode": spec.get("execution_mode", "train"),
            }
        )

    fieldnames = [
        "run_name",
        "group",
        "config_path",
        "seed",
        "method_tags",
        "expected_epochs",
        "priority",
        "execution_mode",
    ]
    with (out_dir / "experiment_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "experiment_manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} funnel manifest rows to {out_dir}.")
    return 0


def _funnel_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    es40 = {"epochs": EXPECTED_EPOCHS, "max_epochs": EXPECTED_EPOCHS}
    uniform = {"missing_pattern_sampler": "uniform"}
    five_pattern_probs = {
        "full": 1.0,
        "missing_gps": 1.0,
        "missing_radar": 1.0,
        "radar_only": 1.0,
        "lidar_only": 1.0,
    }

    def add(
        group: str,
        name: str,
        seed: int | None,
        tags: list[str],
        training: dict[str, Any] | None = None,
        *,
        model: dict[str, Any] | None = None,
        loss: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
        priority: str = "medium",
        execution_mode: str = "train",
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
                "extra": extra or {},
                "priority": priority,
                "execution_mode": execution_mode,
            }
        )

    for name in (
        "checkpoint_selection_uniform_all_available",
        "checkpoint_selection_jtt_all_available",
        "checkpoint_selection_mpdro_all_available",
    ):
        add("selection", name, None, ["checkpoint_selection"], {}, priority="high", execution_mode="selection")

    for seed in (3, 4, 5):
        add(
            "jtt",
            "proto_sampler_uniform_jtt_sample_replay_es40",
            seed,
            ["jtt_sample_replay", "sample_level_baseline", "es40"],
            {**uniform, "failure_replay": {"enabled": True, "mode": "sample_level_proxy"}},
            priority="high",
        )

    mvfr_training = {
        **uniform,
        "mvfr": {
            "enabled": True,
            "stage1_source": "uniform_checkpoint_or_stage1",
            "score_patterns": "missing_only",
            "replay_ratio": 0.5,
            "score_threshold": 0.0,
            "score_power": 1.0,
            "stage2_epochs": 20,
            "normalize_scores": True,
            "freeze_encoder": False,
        },
    }
    for seed in (1, 2, 3):
        add(
            "mvfr",
            "proto_sampler_uniform_mvfr_score_es40",
            seed,
            ["mvfr", "missing_view_failure_replay", "quick_strict_pending", "es40"],
            mvfr_training,
            priority="high",
        )

    for tau, lam, group, priority in ((2.0, 0.25, "mild_mpdro_p0", "high"), (4.0, 0.25, "mild_mpdro_p0", "high"), (2.0, 0.5, "mild_mpdro_p1", "medium")):
        suffix = f"tau{int(tau)}_lam{str(lam).replace('0.', '0').replace('.', '')}"
        for seed in (1, 2, 3):
            add(
                group,
                f"proto_uniform_mpdro_{suffix}_es40",
                seed,
                ["mpdro", f"tau_{tau:g}", f"lambda_{lam:g}", "mild", "es40"],
                {
                    "missing_pattern_sampler": "pattern_balanced",
                    "pattern_probs": five_pattern_probs,
                    "mpdro": {
                        "enabled": True,
                        "tau": tau,
                        "lambda_dro": lam,
                        "patterns": list(five_pattern_probs),
                        "warmup_epochs": 3,
                        "ema_beta": 0.9,
                        "detach_weights": True,
                        "full_protection": True,
                        "min_full_weight": 0.10,
                    },
                },
                priority=priority,
            )

    add(
        "quick",
        "proto_uniform_pattern_logit_bias",
        1,
        ["quick_screen", "pattern_logit_bias", "posthoc"],
        uniform,
        extra={"logit_calibration": {"enabled": True, "mode": "pattern_class_bias", "l2": 0.01, "max_iter": 200}},
        execution_mode="posthoc",
    )
    add(
        "quick",
        "proto_sampler_uniform_modbias_entropy_lam001_es40",
        1,
        ["quick_screen", "modality_bias", "entropy_balance", "lambda_0.01"],
        {
            **uniform,
            "modality_bias": {
                "enabled": True,
                "type": "entropy_balance",
                "lambda_bias": 0.01,
                "patterns": ["full", "missing_gps", "missing_radar", "radar_only", "lidar_only"],
            },
        },
    )
    for dim in (8, 16):
        add(
            "quick",
            f"proto_sampler_uniform_pattern_film_d{dim}_es40",
            1,
            ["quick_screen", "pattern_film", f"d{dim}"],
            uniform,
            model={"pattern_film": {"enabled": True, "dim": dim, "init_identity": True, "apply_at": "pre_head"}},
        )
    add(
        "quick",
        "proto_uniform_tta_entropy_bn",
        1,
        ["quick_screen", "tta", "entropy_bn"],
        uniform,
        extra={"tta": {"enabled": True, "mode": "entropy_bn", "steps": 1, "lr": 1e-4, "update_bn_only": True}},
        execution_mode="eval",
    )
    add(
        "quick",
        "proto_uniform_pbpr_fixed",
        1,
        ["quick_screen", "pbpr_fixed", "sanity_fixed"],
        {**uniform, "prototype_recenter": {"enabled": True, "mode": "sanity_fixed", "fallback_empty_class": True}},
        execution_mode="posthoc",
    )
    return specs


if __name__ == "__main__":
    raise SystemExit(main())
