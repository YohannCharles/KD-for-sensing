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
DEFAULT_OUTPUT_ROOT = "outputs/overnight_branch_router_v2"
DEFAULT_BASELINE_ROOT = "outputs/pcpg_radar_balance_v1,outputs/bprr_reliability_router_v1_retry_gpus0_6_20260706_193654"
DEFAULT_BASE_CONFIG = "outputs/scenes31_34_tinyvit_lmdb/generated_configs/scenes31_34_proto_randomdrop_subset_tinyvit_es40_seed{seed}.yaml"
DEFAULT_GPUS = "1,2"
ANCHOR_EXPERIMENTS = ("a1_e5_low_encoder_lr_anchor", "a2_e6_hard_subset_jepa_anchor")
EXPLORE_EXPERIMENTS = (
    "b1_hard_static_no_jepa",
    "b2_jepa_no_hard",
    "b3_hard_soft_no_jepa",
    "b4_hard_soft_jepa",
    "b5_branch_aux_hard_soft_no_jepa",
    "b6_branch_aux_hard_soft_jepa",
    "c1_supervised_router_e5",
    "c2_supervised_router_hard_soft",
    "c3_supervised_router_hard_soft_jepa",
    "c4_supervised_router_branch_hard_soft_jepa",
)
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
ALIASES = {
    "a1": "a1_e5_low_encoder_lr_anchor",
    "a2": "a2_e6_hard_subset_jepa_anchor",
    "b1": "b1_hard_static_no_jepa",
    "b2": "b2_jepa_no_hard",
    "b3": "b3_hard_soft_no_jepa",
    "b4": "b4_hard_soft_jepa",
    "b5": "b5_branch_aux_hard_soft_no_jepa",
    "b6": "b6_branch_aux_hard_soft_jepa",
    "c1": "c1_supervised_router_e5",
    "c2": "c2_supervised_router_hard_soft",
    "c3": "c3_supervised_router_hard_soft_jepa",
    "c4": "c4_supervised_router_branch_hard_soft_jepa",
}


def experiment_specs() -> dict[str, dict[str, Any]]:
    pcpg = {"model": {"primary": {"fusion_type": "pcpg", "pcpg_fuse_level": "logits"}}}
    supervised_router = {
        "model": {
            "primary": {
                "fusion_type": "supervised_router",
                "router_supervision": "oracle",
                "router_distill_weight": 0.1,
                "router_distill_temperature": 1.0,
                "router_focus_patterns": "missing_image,miss2,drop2",
                "router_fuse_level": "logits",
            }
        },
        "loss": {
            "pcpg_radar_balance": {"enabled": True},
            "router_supervision": "oracle",
            "router_distill_weight": 0.1,
            "router_distill_temperature": 1.0,
            "router_focus_patterns": "missing_image,miss2,drop2",
            "router_fuse_level": "logits",
        },
    }
    low_encoder_lr = {
        "training": {
            "optimizer": {
                "parameter_groups": [
                    {
                        "name": "slow_image_lidar_encoders",
                        "module_patterns": ["encoders.image.*", "encoders.lidar.*"],
                        "lr": 1.0e-5,
                    }
                ]
            }
        }
    }
    eval_cfg = {
        "checkpoint": {"selection_metric": "avg_missing_top1"},
        "evaluation": {
            "modality_subsets": {"enabled": True, "subsets": EVAL_SUBSETS},
            "missing_patterns": {"enabled": True, "patterns": EVAL_PATTERNS},
        },
    }
    no_jepa = {"model": {"primary": {"use_jepa_loss": False}}, "loss": {"use_jepa": False, "jepa_weight": 0.0}}
    jepa = {"model": {"primary": {"use_jepa_loss": True}}, "loss": {"use_jepa": True, "jepa_weight": 0.1}}
    hard_static = {
        "loss": {
            "pcpg_radar_balance": {"enabled": True},
            "hard_subset_weighting": {"enabled": True, "mode": "static", "full_weight": 0.5, "unknown_weight": 1.0},
            "hard_subset_alpha": 1.5,
            "hard_subset_focus": ["radar_only", "missing_image", "miss3"],
        }
    }
    hard_soft = {
        "loss": {
            "pcpg_radar_balance": {"enabled": True},
            "hard_subset_weighting": {"enabled": True, "mode": "soft_static", "unknown_weight": 1.0},
            "hard_subset_focus": ["radar_only", "missing_image", "miss2", "miss3"],
        }
    }
    branch_aux = {
        "loss": {
            "pcpg_radar_balance": {"enabled": True},
            "branch_aux_loss": True,
            "radar_protect_loss": True,
            "unimodal_aux_weight": 0.2,
            "radar_aux_weight": 0.5,
        }
    }
    return {
        "a1_e5_low_encoder_lr_anchor": {"group": "A", "overrides": _merge(pcpg, low_encoder_lr, eval_cfg, no_jepa)},
        "a2_e6_hard_subset_jepa_anchor": {
            "group": "A",
            "overrides": _merge(pcpg, low_encoder_lr, eval_cfg, hard_static, jepa),
        },
        "b1_hard_static_no_jepa": {"group": "B", "overrides": _merge(pcpg, low_encoder_lr, eval_cfg, hard_static, no_jepa)},
        "b2_jepa_no_hard": {"group": "B", "overrides": _merge(pcpg, low_encoder_lr, eval_cfg, jepa)},
        "b3_hard_soft_no_jepa": {"group": "B", "overrides": _merge(pcpg, low_encoder_lr, eval_cfg, hard_soft, no_jepa)},
        "b4_hard_soft_jepa": {"group": "B", "overrides": _merge(pcpg, low_encoder_lr, eval_cfg, hard_soft, jepa)},
        "b5_branch_aux_hard_soft_no_jepa": {"group": "B", "overrides": _merge(pcpg, low_encoder_lr, eval_cfg, branch_aux, hard_soft, no_jepa)},
        "b6_branch_aux_hard_soft_jepa": {"group": "B", "overrides": _merge(pcpg, low_encoder_lr, eval_cfg, branch_aux, hard_soft, jepa)},
        "c1_supervised_router_e5": {"group": "C", "overrides": _merge(supervised_router, low_encoder_lr, eval_cfg, no_jepa)},
        "c2_supervised_router_hard_soft": {"group": "C", "overrides": _merge(supervised_router, low_encoder_lr, eval_cfg, hard_soft, no_jepa)},
        "c3_supervised_router_hard_soft_jepa": {"group": "C", "overrides": _merge(supervised_router, low_encoder_lr, eval_cfg, hard_soft, jepa)},
        "c4_supervised_router_branch_hard_soft_jepa": {
            "group": "C",
            "overrides": _merge(supervised_router, low_encoder_lr, eval_cfg, branch_aux, hard_soft, jepa),
        },
    }


def plan_jobs(
    *,
    experiments: list[str],
    anchor_seeds: list[int],
    explore_seeds: list[int],
    gpus: list[str],
    per_gpu: int,
    output_root: str,
    baseline_root: str,
    base_config: str,
    max_epochs: int | None = None,
) -> list[dict[str, Any]]:
    specs = experiment_specs()
    jobs: list[dict[str, Any]] = []
    for requested in experiments:
        experiment = canonical_experiment(requested)
        seeds = anchor_seeds if specs[experiment]["group"] == "A" else explore_seeds
        for seed in seeds:
            index = len(jobs)
            gpu = gpus[(index // max(int(per_gpu), 1)) % len(gpus)]
            run_name = f"{experiment}/seed{int(seed)}"
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
                "run_name": run_name,
                "base_config": base_config.format(seed=int(seed)),
                "baseline_root": baseline_root,
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
        process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((process, job, handle))
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
    parser = argparse.ArgumentParser(description="Launch overnight branch/router v2 local experiments.")
    parser.add_argument("--base_config", "--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--baseline_root", "--baseline-root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output_root", "--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiments", default="all")
    parser.add_argument("--anchor_seeds", "--anchor-seeds", default="1,2,3,4,5")
    parser.add_argument("--explore_seeds", "--explore-seeds", default="1,2,3")
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--per_gpu", "--per-gpu", type=int, default=2)
    parser.add_argument("--max_jobs", "--max-jobs", type=int, default=4)
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
    experiments = list(ANCHOR_EXPERIMENTS + EXPLORE_EXPERIMENTS) if args.experiments == "all" else [canonical_experiment(item) for item in split_csv(args.experiments)]
    jobs = plan_jobs(
        experiments=experiments,
        anchor_seeds=[int(item) for item in split_csv(args.anchor_seeds)],
        explore_seeds=[int(item) for item in split_csv(args.explore_seeds)],
        gpus=gpus,
        per_gpu=int(args.per_gpu),
        output_root=str(args.output_root),
        baseline_root=str(args.baseline_root),
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
    eval_dir = run_dir.parent / "eval"
    marker = f"{Path(str(job['output_dir'])).parent.name}_seed{int(job['seed'])}_missing_patterns.csv"
    return (
        (run_dir / "metrics.json").exists()
        and ((run_dir / "checkpoints" / "best_avg_missing_top1.pth").exists() or (run_dir / "checkpoints" / "best.pth").exists())
        and (eval_dir / marker).exists()
    )


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
    if experiment not in experiment_specs():
        raise ValueError(f"Unknown experiment {value!r}. Available: {sorted(ALIASES) + sorted(experiment_specs())}")
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
    token = f"seed{int(seed)}"
    fallback = str(base_config).replace(token, "seed1")
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
