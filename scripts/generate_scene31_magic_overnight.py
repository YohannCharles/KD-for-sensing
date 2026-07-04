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
DEFAULT_OUT_DIR = "configs/scene31/magic_overnight"
DEFAULT_BASE_CONFIG = "configs/scene31/templates/main_v3_proto_es20_base.yaml"
DEFAULT_OUTPUT_DIR = "outputs/scene31_magic_overnight_lmdb"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Scene31 magic overnight es40 configs and manifest.")
    parser.add_argument("--base_config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args(argv)

    base_config = Path(args.base_config)
    out_dir = Path(args.out_dir)
    overwrite = _truthy(args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for spec in _magic_specs():
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
    print(f"Wrote {len(rows)} magic overnight manifest rows to {out_dir}.")
    return 0


def _magic_specs() -> list[dict[str, Any]]:
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
    mpfr_proxy = {
        "missing_pattern_sampler": "pattern_balanced",
        "pattern_probs": {
            "full": 0.10,
            "missing_gps": 0.25,
            "missing_radar": 0.25,
            "radar_only": 0.20,
            "lidar_only": 0.20,
        },
        "use_pattern_loss_weight": True,
        "apply_pattern_weight_to_ce": True,
        "apply_pattern_weight_to_proto": False,
        "pattern_loss_weights": {
            "missing_gps": 1.25,
            "missing_radar": 1.25,
            "radar_only": 1.5,
            "lidar_only": 1.5,
        },
        "failure_replay": {
            "enabled": True,
            "mode": "missing_pattern_proxy",
            "replay_ratio": 0.5,
            "stage2_epochs": 20,
            "note": "overnight proxy: pattern-balanced weighted replay, not strict sample-pair cache",
        },
    }
    pbpr_proxy = {
        "missing_pattern_sampler": "pattern_balanced",
        "pattern_probs": five_pattern_probs,
        "lambda_proto": 0.35,
        "lambda_modality_proto": 0.2,
        "apply_pattern_weight_to_proto": True,
        "prototype_recenter": {
            "enabled": True,
            "mode": "shared_balanced_training_proxy",
            "patterns": list(five_pattern_probs),
            "note": "overnight proxy: pattern-balanced prototype training, not post-hoc checkpoint recenter",
        },
    }
    lastlayer_proxy = {
        **uniform,
        "lambda_proto": 0.05,
        "lambda_modality_proto": 0.0,
        "prototype_recenter": {
            "enabled": True,
            "mode": "lastlayer_retrain_proxy",
            "note": "overnight proxy: low-prototype-weight retrain baseline",
        },
    }
    mpdro = {
        "missing_pattern_sampler": "pattern_balanced",
        "pattern_probs": five_pattern_probs,
        "mpdro": {
            "enabled": True,
            "tau": 1.0,
            "patterns": list(five_pattern_probs),
            "detach_weights": True,
            "ema_beta": 0.9,
            "warmup_epochs": 3,
        },
    }
    groupdro_proxy = {
        "missing_pattern_sampler": "pattern_balanced",
        "pattern_probs": five_pattern_probs,
        "use_pattern_loss_weight": True,
        "apply_pattern_weight_to_ce": True,
        "pattern_loss_weights": {name: 1.0 for name in five_pattern_probs},
        "groupdro": {
            "enabled": True,
            "mode": "vanilla_uniform_proxy",
        },
    }

    def add(group: str, name: str, seed: int, tags: list[str], training: dict[str, Any], priority: str) -> None:
        specs.append(
            {
                "group": group,
                "name": name,
                "seed": seed,
                "tags": tags,
                "training": {**es40, **training},
                "model": {},
                "loss": {},
                "priority": priority,
            }
        )

    for seed in (1, 2):
        add("uniform", "proto_sampler_uniform_es40", seed, ["sampler", "uniform", "es40"], uniform, "high")
        add(
            "mpfr_baseline",
            "proto_sampler_uniform_jtt_sample_replay_es40",
            seed,
            ["jtt_sample_replay", "overnight_proxy", "sample_level_baseline", "es40"],
            {**uniform, "failure_replay": {"enabled": True, "mode": "sample_level_proxy"}},
            "medium",
        )
        add(
            "pbpr_baseline",
            "proto_uniform_lastlayer_retrain_es40",
            seed,
            ["lastlayer_retrain", "overnight_proxy", "pbpr_baseline", "es40"],
            lastlayer_proxy,
            "medium",
        )
        add(
            "mpdro_baseline",
            "proto_uniform_groupdro_vanilla_es40",
            seed,
            ["groupdro", "vanilla", "overnight_proxy", "es40"],
            groupdro_proxy,
            "medium",
        )

    for seed in (1, 2, 3):
        add(
            "mpfr",
            "proto_sampler_uniform_mpfr_es40",
            seed,
            ["mpfr", "missing_pattern_failure_replay", "overnight_proxy", "es40"],
            mpfr_proxy,
            "high",
        )
        add(
            "pbpr",
            "proto_uniform_pattern_proto_recenter_es40",
            seed,
            ["pbpr", "pattern_balanced_proto", "overnight_proxy", "es40"],
            pbpr_proxy,
            "high",
        )
        add(
            "mpdro",
            "proto_uniform_mpdro_tau1_es40",
            seed,
            ["mpdro", "tau_1.0", "missing_pattern_dro", "es40"],
            mpdro,
            "high",
        )

    return specs


if __name__ == "__main__":
    raise SystemExit(main())
