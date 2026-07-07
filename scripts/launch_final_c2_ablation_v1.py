#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "outputs/final_c2_ablation_v1"
DEFAULT_BASELINE_ROOTS = (
    "outputs/overnight_branch_router_v2,"
    "outputs/pcpg_radar_balance_v1,"
    "outputs/bprr_reliability_router_v1_retry_gpus0_6_20260706_193654"
)
DEFAULT_BASE_CONFIG = "outputs/scenes31_34_tinyvit_lmdb/generated_configs/scenes31_34_proto_randomdrop_subset_tinyvit_es40_seed{seed}.yaml"
EVAL_PATTERNS = [
    "full",
    "missing_image",
    "missing_lidar",
    "missing_radar",
    "missing_gps",
    "image_only",
    "lidar_only",
    "radar_only",
    "gps_only",
    "missing_image_lidar",
    "missing_image_radar",
    "missing_image_gps",
    "missing_lidar_radar",
    "missing_lidar_gps",
    "missing_radar_gps",
]
EVAL_SUBSETS = ["all", "drop_image", "drop_radar", "drop_lidar", "drop_gps", "image", "radar", "lidar", "gps"]
ALIASES: dict[str, str] = {}


def experiment_specs() -> dict[str, dict[str, Any]]:
    low_lr = {
        "training": {
            "optimizer": {
                "parameter_groups": [
                    {"name": "slow_image_lidar_encoders", "module_patterns": ["encoders.image.*", "encoders.lidar.*"], "lr": 1.0e-5}
                ]
            }
        }
    }
    eval_cfg = {
        "checkpoint": {"selection_metric": "avg_missing_top1"},
        "evaluation": {
            "modality_subsets": {"enabled": True, "subsets": EVAL_SUBSETS},
            "missing_patterns": {"enabled": True, "patterns": EVAL_PATTERNS, "prediction_index": "last"},
        },
    }
    proto_on = {
        "model": {"primary": {"head_type": "prototype", "use_beam_prototype_alignment": True}},
        "training": {
            "use_beam_prototype_alignment": True,
            "beam_proto_align_weight": 0.2,
            "lambda_proto": 0.2,
            "use_modality_prototype_loss": True,
            "modality_proto_weight": 0.1,
            "lambda_modality_proto": 0.1,
            "use_circular_soft_targets": True,
            "use_gaussian_beam_targets": True,
            "beam_label_circular": True,
            "proto_target_type": "gaussian",
        },
    }
    c2_router = {
        "model": {
            "primary": {
                "fusion_type": "supervised_router",
                "router_supervision": "oracle",
                "router_distill_weight": 0.1,
                "router_focus_patterns": "missing_image,miss2,drop2",
                "router_fuse_level": "logits",
                "router_use_pattern_features": True,
                "router_use_reliability_features": True,
                "router_use_prototype_margin": True,
                "router_use_entropy": True,
                "router_use_confidence": True,
                "router_use_logit_norm": True,
            }
        },
        "loss": {
            "pcpg_radar_balance": {"enabled": True},
            "router_supervision": "oracle",
            "router_distill_weight": 0.1,
            "router_focus_patterns": "missing_image,miss2,drop2",
            "router_fuse_level": "logits",
        },
    }
    hard_soft = {
        "loss": {
            "pcpg_radar_balance": {"enabled": True},
            "hard_subset_weighting": {"enabled": True, "mode": "soft_static", "unknown_weight": 1.0},
            "hard_subset_focus": ["radar_only", "missing_image", "miss2", "miss3"],
        }
    }
    no_hard = {"loss": {"hard_subset_weighting": {"enabled": False, "mode": "none"}}}
    hard_static = {
        "loss": {
            "pcpg_radar_balance": {"enabled": True},
            "hard_subset_weighting": {"enabled": True, "mode": "static", "full_weight": 0.5, "unknown_weight": 1.0},
            "hard_subset_alpha": 1.5,
            "hard_subset_focus": ["radar_only", "missing_image", "miss3"],
        }
    }
    no_jepa = {"model": {"primary": {"use_jepa_loss": False}}, "loss": {"use_jepa": False, "jepa_weight": 0.0}}
    jepa = {"model": {"primary": {"use_jepa_loss": True}}, "loss": {"use_jepa": True, "jepa_weight": 0.1}}
    no_branch = {"loss": {"branch_aux_loss": False, "radar_protect_loss": False, "unimodal_aux_weight": 0.0, "radar_aux_weight": 0.0}}
    branch = {"loss": {"pcpg_radar_balance": {"enabled": True}, "branch_aux_loss": True, "unimodal_aux_weight": 0.2, "radar_protect_loss": True, "radar_aux_weight": 0.5}}
    pcpg = {"model": {"primary": {"fusion_type": "pcpg", "pcpg_fuse_level": "logits"}}}
    common_c2 = _merge(low_lr, eval_cfg, proto_on, c2_router, hard_soft, no_jepa, no_branch)
    specs = {
        "a0_c2_full_main": {"seed_group": "main", "section": "A", "overrides": common_c2},
        "a1_b4_nonrouter_soft_jepa": {"seed_group": "main", "section": "A", "overrides": _merge(low_lr, eval_cfg, proto_on, pcpg, hard_soft, jepa, no_branch)},
        "a2_c2_plus_jepa_negative": {"seed_group": "negative", "section": "A", "overrides": _merge(common_c2, jepa)},
        "a3_c2_plus_branch_aux_negative": {"seed_group": "negative", "section": "A", "overrides": _merge(common_c2, branch, no_jepa)},
        "a4_c2_plus_branch_aux_jepa_negative": {"seed_group": "negative", "section": "A", "overrides": _merge(common_c2, branch, jepa)},
        "b0_no_router_supervision": {"seed_group": "ablation", "section": "B", "overrides": _merge(common_c2, {"model": {"primary": {"router_supervision": "none", "router_distill_weight": 0.0}}, "loss": {"router_supervision": "none", "router_distill_weight": 0.0}})},
        "b1_no_pattern_features": {"seed_group": "ablation", "section": "B", "overrides": _merge(common_c2, {"model": {"primary": {"router_use_pattern_features": False}}, "loss": {"router_use_pattern_features": False}})},
        "b2_no_prototype_margin_feature": {"seed_group": "ablation", "section": "B", "overrides": _merge(common_c2, {"model": {"primary": {"router_use_prototype_margin": False}}, "loss": {"router_use_prototype_margin": False}})},
        "b3_no_reliability_features_pattern_only": {"seed_group": "ablation", "section": "B", "overrides": _merge(common_c2, {"model": {"primary": {"router_use_reliability_features": False, "router_use_prototype_margin": False}}, "loss": {"router_use_reliability_features": False, "router_use_prototype_margin": False}})},
        "b4_no_router_focus_all_patterns": {"seed_group": "ablation", "section": "B", "overrides": _merge(common_c2, {"model": {"primary": {"router_focus_patterns": "all_multimodal"}}, "loss": {"router_focus_patterns": "all_multimodal"}})},
        "c0_no_beam_prototype_alignment_loss": {"seed_group": "ablation", "section": "C", "overrides": _merge(common_c2, {"model": {"primary": {"use_beam_prototype_alignment": False}}, "training": {"use_beam_prototype_alignment": False, "beam_proto_align_weight": 0.0, "lambda_proto": 0.0}})},
        "c1_no_modality_prototype_loss": {"seed_group": "ablation", "section": "C", "overrides": _merge(common_c2, {"training": {"use_modality_prototype_loss": False, "modality_proto_weight": 0.0, "lambda_modality_proto": 0.0}})},
        "c2_no_circular_soft_targets": {"seed_group": "ablation", "section": "C", "overrides": _merge(common_c2, {"training": {"use_circular_soft_targets": False, "use_gaussian_beam_targets": False, "beam_label_circular": False, "proto_target_type": "onehot"}})},
        "c3_classifier_head_no_prototype": {"seed_group": "ablation", "section": "C", "overrides": _merge(common_c2, {"model": {"primary": {"head_type": "classifier", "use_beam_prototype_alignment": False, "router_use_prototype_margin": False}}, "training": {"use_beam_prototype_alignment": False, "use_modality_prototype_loss": False, "use_circular_soft_targets": False, "use_gaussian_beam_targets": False, "beam_proto_align_weight": 0.0, "lambda_proto": 0.0, "modality_proto_weight": 0.0, "lambda_modality_proto": 0.0}, "loss": {"router_use_prototype_margin": False}})},
        "d0_weighted_sum_fusion": {"seed_group": "ablation", "section": "D", "overrides": _merge(common_c2, {"model": {"primary": {"fusion_type": "weighted_sum"}}})},
        "d1_average_fusion": {"seed_group": "ablation", "section": "D", "overrides": _merge(common_c2, {"model": {"primary": {"fusion_type": "average"}}})},
        "d2_raw_confidence_gate": {"seed_group": "ablation", "section": "D", "overrides": _merge(common_c2, {"model": {"primary": {"fusion_type": "raw_conf_gate", "raw_conf_temperature": 1.0}}})},
        "d3_bprr_unsupervised_router": {"seed_group": "ablation", "section": "D", "overrides": _merge(common_c2, {"model": {"primary": {"fusion_type": "bprr", "bprr_fuse_level": "logits", "bprr_calibration": "temperature"}}})},
        "e0_no_soft_hard_subset": {"seed_group": "ablation", "section": "E", "overrides": _merge(common_c2, no_hard)},
        "e1_static_hard_subset": {"seed_group": "ablation", "section": "E", "overrides": _merge(common_c2, hard_static)},
        "e2_soft_hard_without_router": {"seed_group": "ablation", "section": "E", "overrides": _merge(low_lr, eval_cfg, proto_on, pcpg, hard_soft, no_jepa, no_branch)},
    }
    ALIASES.update({key.split("_", 1)[0]: key for key in specs})
    return specs


def plan_jobs(
    *,
    experiments: list[str],
    main_seeds: list[int],
    ablation_seeds: list[int],
    negative_seeds: list[int],
    gpus: list[str],
    per_gpu: int,
    output_root: str,
    baseline_roots: str,
    base_config: str = DEFAULT_BASE_CONFIG,
    max_epochs: int | None = None,
) -> list[dict[str, Any]]:
    specs = experiment_specs()
    jobs: list[dict[str, Any]] = []
    for requested in experiments:
        experiment = canonical_experiment(requested)
        group = specs[experiment]["seed_group"]
        seeds = main_seeds if group == "main" else negative_seeds if group == "negative" else ablation_seeds
        for seed in seeds:
            index = len(jobs)
            gpu = gpus[(index // max(int(per_gpu), 1)) % len(gpus)]
            config_path = Path(output_root) / "generated_configs" / f"{experiment}_seed{int(seed)}.yaml"
            log_path = Path(output_root) / "logs" / f"{experiment}_seed{int(seed)}.log"
            output_dir = Path(output_root) / experiment / f"seed{int(seed)}"
            job = {
                "experiment": experiment,
                "seed": int(seed),
                "gpu": str(gpu),
                "cmd": "",
                "status": "planned",
                "start_time": "",
                "end_time": "",
                "return_code": "",
                "log_path": str(log_path),
                "output_dir": str(output_dir),
                "run_name": f"{experiment}/seed{int(seed)}",
                "base_config": base_config.format(seed=int(seed)),
                "baseline_roots": baseline_roots,
                "output_root": output_root,
                "config_path": str(config_path),
                "max_epochs": max_epochs,
            }
            job["command"] = ["conda", "run", "-n", "kd_mm_beam", "kd-sensing-train", "--config", str(config_path)]
            job["cmd"] = " ".join(job["command"])
            jobs.append(job)
    return jobs


def write_generated_configs(jobs: list[dict[str, Any]], *, dry_run: bool) -> None:
    specs = experiment_specs()
    for job in jobs:
        payload = _load_payload_for_job(job, dry_run=dry_run)
        overrides = _merge(
            specs[str(job["experiment"])]["overrides"],
            {
                "experiment": {"name": str(job["experiment"]), "seed": int(job["seed"])},
                "output": {"dir": str(job["output_root"]), "run_name": str(job["run_name"]), "group_by_scene": False},
                "final_c2_ablation_v1": {"experiment": str(job["experiment"]), "seed": int(job["seed"])},
            },
        )
        if job.get("max_epochs") is not None:
            overrides = _merge(overrides, {"training": {"epochs": int(job["max_epochs"]), "max_epochs": int(job["max_epochs"])}})
        payload = _merge(payload, overrides)
        path = Path(str(job["config_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_manifest(jobs: list[dict[str, Any]], output_root: str) -> Path:
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "job_manifest.csv"
    fields = ["experiment", "seed", "gpu", "cmd", "status", "start_time", "end_time", "return_code", "log_path", "output_dir"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({key: job.get(key, "") for key in fields})
    (out_dir / "job_manifest.json").write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def run_jobs(jobs: list[dict[str, Any]], *, max_jobs: int, per_gpu: int, force: bool, skip_completed: bool) -> int:
    running: list[tuple[subprocess.Popen, dict[str, Any], Any]] = []
    failed: list[dict[str, Any]] = []
    output_root = str(jobs[0]["output_root"]) if jobs else DEFAULT_OUTPUT_ROOT
    for job in jobs:
        if skip_completed and not force and completed(job):
            job["status"] = "skipped"
            job["end_time"] = now()
            write_manifest(jobs, output_root)
            continue
        while len(running) >= int(max_jobs) or gpu_count(running, str(job["gpu"])) >= int(per_gpu):
            failed.extend(poll_finished(running))
            write_manifest(jobs, output_root)
            time.sleep(1.0)
        log_path = Path(str(job["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        handle = log_path.open("a", encoding="utf-8")
        job["status"] = "running"
        job["start_time"] = now()
        write_manifest(jobs, output_root)
        running.append((subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT), job, handle))
    while running:
        failed.extend(poll_finished(running, wait=True))
        write_manifest(jobs, output_root)
    if failed:
        write_failed_jobs(failed, output_root)
        return 1
    failed_path = Path(output_root) / "failed_jobs.csv"
    if failed_path.exists():
        failed_path.unlink()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch final c2 ablation v1 local experiments.")
    parser.add_argument("--base_config", "--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--baseline_roots", "--baseline-roots", default=DEFAULT_BASELINE_ROOTS)
    parser.add_argument("--output_root", "--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiments", default="all")
    parser.add_argument("--main_seeds", "--main-seeds", default="1,2,3,4,5")
    parser.add_argument("--ablation_seeds", "--ablation-seeds", default="1,2,3")
    parser.add_argument("--negative_seeds", "--negative-seeds", default="1,2,3")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--per_gpu", "--per-gpu", type=int, default=1)
    parser.add_argument("--max_jobs", "--max-jobs", type=int, default=8)
    parser.add_argument("--max_epochs", "--max-epochs", type=int, default=None)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--skip_completed", "--skip-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    gpus = split_csv(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id.")
    if int(args.per_gpu) <= 0 or int(args.max_jobs) <= 0:
        raise ValueError("--per_gpu and --max_jobs must be positive.")
    if int(args.max_jobs) > len(gpus) * int(args.per_gpu):
        raise ValueError("--max_jobs cannot exceed len(gpus) * per_gpu.")
    specs = experiment_specs()
    experiments = list(specs) if args.experiments == "all" else [canonical_experiment(item) for item in split_csv(args.experiments)]
    jobs = plan_jobs(
        experiments=experiments,
        main_seeds=[int(item) for item in split_csv(args.main_seeds)],
        ablation_seeds=[int(item) for item in split_csv(args.ablation_seeds)],
        negative_seeds=[int(item) for item in split_csv(args.negative_seeds)],
        gpus=gpus,
        per_gpu=int(args.per_gpu),
        output_root=str(args.output_root),
        baseline_roots=str(args.baseline_roots),
        base_config=str(args.base_config),
        max_epochs=args.max_epochs,
    )
    write_generated_configs(jobs, dry_run=bool(args.dry_run))
    if args.dry_run:
        for job in jobs:
            job["status"] = "dry_run"
            job["end_time"] = now()
    manifest = write_manifest(jobs, str(args.output_root))
    print(f"Wrote manifest: {manifest}")
    for job in jobs:
        print(f"[gpu {job['gpu']}] {job['cmd']}")
    if args.dry_run:
        return 0
    return run_jobs(jobs, max_jobs=int(args.max_jobs), per_gpu=int(args.per_gpu), force=bool(args.force), skip_completed=bool(args.skip_completed))


def completed(job: dict[str, Any]) -> bool:
    run_dir = Path(str(job["output_dir"]))
    marker = f"{job['experiment']}_seed{int(job['seed'])}_missing_patterns.csv"
    return (run_dir / "metrics.json").exists() and (run_dir.parent / "eval" / marker).exists()


def poll_finished(running: list[tuple[subprocess.Popen, dict[str, Any], Any]], *, wait: bool = False) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    if wait and running:
        running[0][0].wait()
    for index in range(len(running) - 1, -1, -1):
        process, job, handle = running[index]
        code = process.poll()
        if code is None:
            continue
        handle.close()
        job["return_code"] = int(code)
        job["end_time"] = now()
        job["status"] = "completed" if code == 0 else "failed"
        if code != 0:
            failed.append(job)
        running.pop(index)
    return failed


def write_failed_jobs(failed: list[dict[str, Any]], output_root: str) -> Path:
    path = Path(output_root) / "failed_jobs.csv"
    fields = ["experiment", "seed", "gpu", "cmd", "status", "start_time", "end_time", "return_code", "log_path", "output_dir"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in failed:
            writer.writerow({key: job.get(key, "") for key in fields})
    return path


def gpu_count(running: list[tuple[subprocess.Popen, dict[str, Any], Any]], gpu: str) -> int:
    return sum(1 for _process, job, _handle in running if str(job.get("gpu")) == gpu)


def canonical_experiment(value: str) -> str:
    key = str(value).strip()
    experiment = ALIASES.get(key, key)
    specs = experiment_specs()
    if experiment not in specs:
        raise ValueError(f"Unknown experiment {value!r}. Available: {sorted(ALIASES) + sorted(specs)}")
    return experiment


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_payload_for_job(job: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    base_rel = str(job["base_config"])
    base_path = ROOT / base_rel
    if not base_path.exists():
        fallback_rel = _seed1_base_config(base_rel, int(job["seed"]))
        if fallback_rel is not None and (ROOT / fallback_rel).exists():
            job["base_config_resolved"] = fallback_rel
            base_path = ROOT / fallback_rel
        elif dry_run:
            return {"_base_": str(job["base_config"])}
        else:
            raise FileNotFoundError(f"Base config not found: {base_path}")
    return yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}


def _seed1_base_config(base_config: str, seed: int) -> str | None:
    if int(seed) == 1:
        return None
    fallback = str(base_config).replace(f"seed{int(seed)}", "seed1")
    return fallback if fallback != str(base_config) else None


def _merge(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge(merged[key], value)
            else:
                merged[key] = value
    return merged


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
