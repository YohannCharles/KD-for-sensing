#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Scene31 night-grid experiment configs and manifest.")
    parser.add_argument("--base_config", default="configs/scene31/templates/main_v3_proto_es20_base.yaml")
    parser.add_argument("--out_dir", default="configs/scene31/night_grid")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--overwrite", default="false")
    args = parser.parse_args(argv)

    base_config = Path(args.base_config)
    out_dir = Path(args.out_dir)
    overwrite = _truthy(args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for row in _baseline_rows():
        rows.append(row)
    for spec in _grid_specs():
        for seed in args.seeds:
            run_name = f"{spec['name']}_seed{seed}"
            config_path = out_dir / f"{run_name}.yaml"
            if overwrite or not config_path.exists():
                payload = _config_payload(base_config, config_path, run_name, seed, spec)
                config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
            rows.append(
                {
                    "run_name": run_name,
                    "group": spec["group"],
                    "config_path": _rel(config_path),
                    "seed": seed,
                    "method_tags": ",".join(spec["tags"]),
                    "expected_epochs": 20,
                    "priority": spec.get("priority", "medium"),
                }
            )

    csv_path = out_dir / "experiment_manifest.csv"
    json_path = out_dir / "experiment_manifest.json"
    fieldnames = ["run_name", "group", "config_path", "seed", "method_tags", "expected_epochs", "priority"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} manifest rows to {csv_path} and {json_path}.")
    return 0


def _grid_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add(group: str, name: str, tags: list[str], training: dict[str, Any] | None = None, model: dict[str, Any] | None = None, priority: str = "medium") -> None:
        specs.append({"group": group, "name": name, "tags": tags, "training": training or {}, "model": model or {}, "priority": priority})

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
    }
    add("A", "proto_condbtapa_radaronly_es20", ["condbtapa", "radar_only"], {**cond_base, "btapa_apply_patterns": ["radar_only"]}, priority="high")
    add("A", "proto_condbtapa_lidaronly_es20", ["condbtapa", "lidar_only"], {**cond_base, "btapa_apply_patterns": ["lidar_only"]}, priority="high")
    add("A", "proto_condbtapa_weaksingle_es20", ["condbtapa", "weak_single"], {**cond_base, "btapa_apply_patterns": ["radar_only", "lidar_only"]}, priority="high")
    add("A", "proto_condbtapa_singleall_es20", ["condbtapa", "single_all"], {**cond_base, "btapa_apply_patterns": ["gps_only", "image_only", "radar_only", "lidar_only"]})
    add("A", "proto_condbtapa_sensingonly_es20", ["condbtapa", "sensing_only"], {**cond_base, "btapa_apply_patterns": ["missing_gps", "non_gps_only", "image_only", "radar_only", "lidar_only"]})
    add("A", "proto_condbtapa_weaksingle_lam005_es20", ["condbtapa", "weak_single", "lambda_0.05"], {**cond_base, "btapa_apply_patterns": ["radar_only", "lidar_only"], "btapa_lambda": 0.05})
    add("A", "proto_condbtapa_weaksingle_lam01_es20", ["condbtapa", "weak_single", "lambda_0.1"], {**cond_base, "btapa_apply_patterns": ["radar_only", "lidar_only"], "btapa_lambda": 0.1})

    add("B", "proto_sampler_uniform_es20", ["sampler", "uniform"], {"missing_pattern_sampler": "uniform"}, priority="high")
    add("B", "proto_sampler_weaksingle_over_es20", ["sampler", "weak_single_over"], {"missing_pattern_sampler": "weak_single_oversample", "pattern_sampling_weights": {"radar_only": 2.0, "lidar_only": 2.0}}, priority="high")
    add("B", "proto_sampler_sensingonly_over_es20", ["sampler", "sensing_only_over"], {"missing_pattern_sampler": "sensing_only_oversample", "pattern_sampling_weights": {"missing_gps": 1.5, "non_gps_only": 1.5, "image_only": 1.5, "radar_only": 2.0, "lidar_only": 2.0}})
    add("B", "proto_sampler_missinggps_over_es20", ["sampler", "missing_gps_over"], {"missing_pattern_sampler": "missing_gps_oversample", "pattern_sampling_weights": {"missing_gps": 2.0, "non_gps_only": 2.0}})
    add("B", "proto_curriculum_easy2hard_es20", ["sampler", "curriculum_easy_to_hard"], {"missing_pattern_sampler": "curriculum_easy_to_hard", "curriculum_schedule": {"epochs_1_5": ["full", "missing_image", "missing_radar", "missing_lidar"], "epochs_6_10": ["missing_gps", "non_gps_only"], "epochs_11_20": ["gps_only", "image_only", "radar_only", "lidar_only"]}})
    add("B", "proto_curriculum_hard2easy_es20", ["sampler", "curriculum_hard_to_easy"], {"missing_pattern_sampler": "curriculum_hard_to_easy", "curriculum_schedule": {"epochs_1_5": ["gps_only", "image_only", "radar_only", "lidar_only"], "epochs_6_10": ["missing_gps", "non_gps_only"], "epochs_11_20": ["full", "missing_image", "missing_radar", "missing_lidar"]}})

    reweight_base = {"use_pattern_loss_weight": True, "apply_pattern_weight_to_ce": True, "apply_pattern_weight_to_proto": False}
    add("C", "proto_reweight_weaksingle_w125_es20", ["reweight", "weak_single", "w1.25"], {**reweight_base, "pattern_loss_weights": {"radar_only": 1.25, "lidar_only": 1.25}}, priority="high")
    add("C", "proto_reweight_weaksingle_w15_es20", ["reweight", "weak_single", "w1.5"], {**reweight_base, "pattern_loss_weights": {"radar_only": 1.5, "lidar_only": 1.5}}, priority="high")
    add("C", "proto_reweight_weaksingle_w20_es20", ["reweight", "weak_single", "w2.0"], {**reweight_base, "pattern_loss_weights": {"radar_only": 2.0, "lidar_only": 2.0}})
    add("C", "proto_reweight_missinggps_w15_es20", ["reweight", "missing_gps", "w1.5"], {**reweight_base, "pattern_loss_weights": {"missing_gps": 1.5, "non_gps_only": 1.5}})
    add("C", "proto_reweight_hardall_w15_es20", ["reweight", "hard_all", "w1.5"], {**reweight_base, "pattern_loss_weights": {"missing_gps": 1.5, "non_gps_only": 1.5, "radar_only": 1.5, "lidar_only": 1.5}})

    add("D", "proto_maskadapter_d16_es20", ["mask_adapter", "d16"], model={"use_mask_adapter": True, "mask_adapter_dim": 16}, priority="high")
    add("D", "proto_maskadapter_d32_es20", ["mask_adapter", "d32"], model={"use_mask_adapter": True, "mask_adapter_dim": 32})
    add("D", "proto_maskadapter_d16_condbtapa_weaksingle_es20", ["mask_adapter", "d16", "condbtapa", "weak_single"], {**cond_base, "btapa_apply_patterns": ["radar_only", "lidar_only"]}, {"use_mask_adapter": True, "mask_adapter_dim": 16}, priority="high")
    add("D", "proto_maskadapter_d32_condbtapa_weaksingle_es20", ["mask_adapter", "d32", "condbtapa", "weak_single"], {**cond_base, "btapa_apply_patterns": ["radar_only", "lidar_only"]}, {"use_mask_adapter": True, "mask_adapter_dim": 32})

    kd_base = {"use_weak_pattern_kd": True, "kd_teacher": "full_modality_same_model_stopgrad", "kd_loss_type": "kl_logits"}
    add("E", "proto_weakkd_weaksingle_l005_t2_es20", ["weak_kd", "weak_single", "l0.05", "t2"], {**kd_base, "kd_apply_patterns": ["radar_only", "lidar_only"], "lambda_kd": 0.05, "kd_temperature": 2.0}, priority="low")
    add("E", "proto_weakkd_weaksingle_l01_t2_es20", ["weak_kd", "weak_single", "l0.1", "t2"], {**kd_base, "kd_apply_patterns": ["radar_only", "lidar_only"], "lambda_kd": 0.1, "kd_temperature": 2.0}, priority="low")
    add("E", "proto_weakkd_sensingonly_l01_t2_es20", ["weak_kd", "sensing_only", "l0.1", "t2"], {**kd_base, "kd_apply_patterns": ["missing_gps", "non_gps_only", "image_only", "radar_only", "lidar_only"], "lambda_kd": 0.1, "kd_temperature": 2.0}, priority="low")
    add("E", "proto_weakkd_hardall_l01_t15_es20", ["weak_kd", "hard_all", "l0.1", "t1.5"], {**kd_base, "kd_apply_patterns": ["missing_gps", "non_gps_only", "radar_only", "lidar_only"], "lambda_kd": 0.1, "kd_temperature": 1.5}, priority="low")

    lat_model = {"use_light_latent_pred": True, "latent_pred_hidden_dim": 256}
    lat_base = {"use_light_latent_pred": True, "latent_pred_apply_patterns": ["radar_only", "lidar_only"]}
    add("F", "proto_latpred_fullh_weaksingle_l001_es20", ["latent_pred", "full_fused", "weak_single", "l0.01"], {**lat_base, "latent_pred_target": "full_fused", "lambda_latent_pred": 0.01, "latent_pred_loss": "cosine"}, {**lat_model, "latent_pred_target": "full_fused"}, priority="low")
    add("F", "proto_latpred_fullh_weaksingle_l005_es20", ["latent_pred", "full_fused", "weak_single", "l0.05"], {**lat_base, "latent_pred_target": "full_fused", "lambda_latent_pred": 0.05, "latent_pred_loss": "cosine"}, {**lat_model, "latent_pred_target": "full_fused"}, priority="low")
    add("F", "proto_latpred_proto_weaksingle_l001_es20", ["latent_pred", "prototype_distribution", "weak_single", "l0.01"], {**lat_base, "latent_pred_target": "prototype_distribution", "lambda_latent_pred": 0.01, "latent_pred_loss": "kl"}, {**lat_model, "latent_pred_target": "prototype_distribution"}, priority="low")
    return specs


def _baseline_rows() -> list[dict[str, Any]]:
    return [
        _baseline("main_v3_strong_reliability_proto", "configs/scene31/main_v3_strong_reliability_proto.yaml", 1, "proto"),
        _baseline("main_v3_strong_reliability_proto_seed2", "configs/scene31/main_v3_strong_reliability_proto_seed2.yaml", 2, "proto"),
        _baseline("main_v3_strong_reliability_proto_seed3", "configs/scene31/main_v3_strong_reliability_proto_seed3.yaml", 3, "proto"),
        _baseline("main_v3_strong_reliability_btapa_tau1", "configs/scene31/main_v3_strong_reliability_btapa_tau1.yaml", 1, "btapa_tau1"),
        _baseline("main_v3_strong_reliability_btapa_tau1_seed2", "configs/scene31/main_v3_strong_reliability_btapa_tau1_seed2.yaml", 2, "btapa_tau1"),
        _baseline("main_v3_strong_reliability_btapa_tau1_seed3", "configs/scene31/main_v3_strong_reliability_btapa_tau1_seed3.yaml", 3, "btapa_tau1"),
    ]


def _baseline(run_name: str, config_path: str, seed: int, tag: str) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "group": "baseline",
        "config_path": config_path,
        "seed": seed,
        "method_tags": tag,
        "expected_epochs": 40,
        "priority": "reference",
    }


def _config_payload(base_config: Path, config_path: Path, run_name: str, seed: int, spec: dict[str, Any]) -> dict[str, Any]:
    base_text = os.path.relpath((ROOT / base_config).resolve(), (ROOT / config_path.parent).resolve())
    payload: dict[str, Any] = {
        "_base_": base_text,
        "experiment": {"name": run_name, "seed": int(seed)},
        "model": {"primary": {"ablation_id": run_name}},
        "training": dict(spec.get("training", {})),
        "loss": {"u_mask_beam_jepa": {}},
        "evaluation": {"beam_distance_circular": True},
        "output": {"run_name": run_name},
    }
    if spec.get("model"):
        payload["model"]["primary"].update(spec["model"])
    if spec.get("loss"):
        payload["loss"].update(spec["loss"])
    payload["loss"].setdefault("u_mask_beam_jepa", {})
    payload["loss"]["u_mask_beam_jepa"].update(payload["training"])
    return payload


def _rel(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
