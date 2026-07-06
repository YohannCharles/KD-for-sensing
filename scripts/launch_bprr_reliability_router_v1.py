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
DEFAULT_OUTPUT_ROOT = "outputs/bprr_reliability_router_v1"
DEFAULT_BASELINE_ROOT = "outputs/pcpg_radar_balance_v1"
DEFAULT_BASE_CONFIG = "outputs/scenes31_34_tinyvit_lmdb/generated_configs/scenes31_34_proto_randomdrop_subset_tinyvit_es40_seed{seed}.yaml"
DEFAULT_EXPERIMENTS = "e3,e7,e8,e9,e10,e11,e12"
DEFAULT_GPUS = "0,1,2,3,4,5,6,7"
EVAL_SUBSETS = ["all", "drop_image", "drop_radar", "drop_lidar", "drop_gps", "image", "radar", "lidar", "gps"]

EXPERIMENT_ALIASES = {
    "e3": "e3_oracle_gate_eval",
    "e7": "e7_raw_confidence_gate",
    "e8": "e8_bprr_calibrated_router",
    "e9": "e9_bprr_radar_gate_reg",
    "e10": "e10_bprr_hard_subset_no_jepa",
    "e11": "e11_bprr_jepa_no_hard_subset",
    "e12": "e12_bprr_full_combo",
}


def experiment_specs() -> dict[str, dict[str, Any]]:
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
    avg_missing = {
        "checkpoint": {"selection_metric": "avg_missing_top1"},
        "evaluation": {"modality_subsets": {"enabled": True, "subsets": EVAL_SUBSETS}},
    }
    no_jepa = {"model": {"primary": {"use_jepa_loss": False}}, "loss": {"use_jepa": False, "jepa_weight": 0.0}}
    bprr = {
        "model": {
            "primary": {
                "fusion_type": "bprr",
                "bprr_fuse_level": "logits",
                "bprr_calibration": "temperature",
                "bprr_init_temperature": 1.0,
            }
        }
    }
    gate_reg = {
        "loss": {
            "bprr_gate_balance_weight": 0.02,
            "bprr_radar_gate_reg_weight": 0.05,
            "bprr_radar_gate_floor": 0.10,
            "bprr_radar_gate_reg_patterns": ["radar_only", "missing_image", "miss3"],
        }
    }
    hard_subset = {
        "loss": {
            "hard_subset_weighting": {"enabled": True, "mode": "static", "full_weight": 0.5, "unknown_weight": 1.0},
            "hard_subset_alpha": 1.5,
            "hard_subset_focus": ["radar_only", "missing_image", "miss3"],
        }
    }
    jepa = {"loss": {"use_jepa": True, "jepa_weight": 0.1}}
    return {
        "e3_oracle_gate_eval": {
            "mode": "eval",
            "source_experiment": "e5_pcpg_low_encoder_lr",
            "overrides": {"model": {"primary": {"fusion_type": "pcpg", "pcpg_fuse_level": "logits"}}},
        },
        "e7_raw_confidence_gate": {
            "mode": "train",
            "overrides": _merge(
                low_encoder_lr,
                avg_missing,
                no_jepa,
                {"model": {"primary": {"fusion_type": "raw_conf_gate", "raw_conf_temperature": 1.0}}},
            ),
        },
        "e8_bprr_calibrated_router": {
            "mode": "train",
            "overrides": _merge(low_encoder_lr, avg_missing, no_jepa, bprr),
        },
        "e9_bprr_radar_gate_reg": {
            "mode": "train",
            "overrides": _merge(low_encoder_lr, avg_missing, no_jepa, bprr, gate_reg),
        },
        "e10_bprr_hard_subset_no_jepa": {
            "mode": "train",
            "overrides": _merge(low_encoder_lr, avg_missing, no_jepa, bprr, gate_reg, hard_subset),
        },
        "e11_bprr_jepa_no_hard_subset": {
            "mode": "train",
            "overrides": _merge(low_encoder_lr, avg_missing, bprr, gate_reg, jepa),
        },
        "e12_bprr_full_combo": {
            "mode": "train",
            "overrides": _merge(low_encoder_lr, avg_missing, bprr, gate_reg, hard_subset, jepa),
        },
    }


def plan_jobs(
    *,
    experiments: list[str],
    seeds: list[int],
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
        experiment = _canonical_experiment(requested)
        spec = specs[experiment]
        for seed in seeds:
            index = len(jobs)
            gpu = gpus[(index // max(int(per_gpu), 1)) % len(gpus)]
            run_name = f"{experiment}/seed{int(seed)}"
            config_path = Path(output_root) / "generated_configs" / f"{experiment}_seed{int(seed)}.yaml"
            log_path = Path(output_root) / "logs" / f"{experiment}_seed{int(seed)}.log"
            job = {
                "experiment": experiment,
                "seed": int(seed),
                "run_name": run_name,
                "mode": str(spec["mode"]),
                "gpu": str(gpu),
                "cmd": "",
                "status": "planned",
                "start_time": "",
                "end_time": "",
                "return_code": "",
                "log_path": str(log_path),
                "base_config": base_config.format(seed=int(seed)),
                "baseline_root": baseline_root,
                "output_root": output_root,
                "config_path": str(config_path),
                "max_epochs": max_epochs,
            }
            job["command"] = _command_for_job(job)
            job["cmd"] = " ".join(job["command"])
            jobs.append(job)
    return jobs


def write_generated_configs(jobs: list[dict[str, Any]], *, dry_run: bool) -> None:
    specs = experiment_specs()
    for job in jobs:
        spec = specs[str(job["experiment"])]
        payload = _load_payload_for_job(job, dry_run=dry_run)
        overrides = _merge(
            spec["overrides"],
            {
                "experiment": {"name": str(job["experiment"]), "seed": int(job["seed"])},
                "output": {"dir": str(job["output_root"]), "run_name": str(job["run_name"]), "group_by_scene": False},
            },
        )
        if job.get("max_epochs") is not None:
            overrides = _merge(overrides, {"training": {"epochs": int(job["max_epochs"]), "max_epochs": int(job["max_epochs"])}})
        payload = _merge(payload, overrides)
        if job["mode"] == "eval":
            checkpoint = resolve_e3_checkpoint(int(job["seed"]), Path(str(job["baseline_root"])), allow_missing=dry_run)
            job["checkpoint"] = str(checkpoint)
            job["command"] = _command_for_job(job)
            job["cmd"] = " ".join(job["command"])
        path = Path(str(job["config_path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_manifest(jobs: list[dict[str, Any]], output_root: str) -> Path:
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "job_manifest.csv"
    fields = ["experiment", "seed", "gpu", "cmd", "status", "start_time", "end_time", "return_code", "log_path"]
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
        if skip_completed and not force and _completed(job):
            job["status"] = "skipped"
            job["end_time"] = _now()
            write_manifest(jobs, output_root)
            continue
        while len(running) >= int(max_jobs) or _gpu_count(running, str(job["gpu"])) >= int(per_gpu):
            failed.extend(_poll_finished(running))
            write_manifest(jobs, output_root)
            time.sleep(1.0)
        log_path = Path(str(job["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        handle = log_path.open("a", encoding="utf-8")
        job["status"] = "running"
        job["start_time"] = _now()
        write_manifest(jobs, output_root)
        process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((process, job, handle))
    while running:
        failed.extend(_poll_finished(running, wait=True))
        write_manifest(jobs, output_root)
    if failed:
        print("Failed jobs:")
        for job in failed:
            print(f"- {job['experiment']} seed{job['seed']} gpu{job['gpu']} rc={job['return_code']} log={job['log_path']}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch BPRR reliability-router v1 local experiments.")
    parser.add_argument("--base_config", "--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--baseline_root", "--baseline-root", default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output_root", "--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiments", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--per_gpu", "--per-gpu", type=int, default=1)
    parser.add_argument("--max_jobs", "--max-jobs", type=int, default=8)
    parser.add_argument("--max_epochs", "--max-epochs", type=int, default=None)
    parser.add_argument("--dry_run", "--dry-run", action="store_true")
    parser.add_argument("--skip_completed", "--skip-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    gpus = _split(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id.")
    if int(args.per_gpu) <= 0 or int(args.max_jobs) <= 0:
        raise ValueError("--per_gpu and --max_jobs must be positive.")
    experiments = [_canonical_experiment(item) for item in _split(args.experiments)]
    jobs = plan_jobs(
        experiments=experiments,
        seeds=[int(item) for item in _split(args.seeds)],
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
            job["end_time"] = _now()
    manifest = write_manifest(jobs, str(args.output_root))
    print(f"Wrote manifest: {manifest}")
    for job in jobs:
        print(f"[gpu {job['gpu']}] {job['cmd']}")
    if args.dry_run:
        return 0
    return run_jobs(
        jobs,
        max_jobs=int(args.max_jobs),
        per_gpu=int(args.per_gpu),
        force=bool(args.force),
        skip_completed=bool(args.skip_completed),
    )


def resolve_e3_checkpoint(seed: int, baseline_root: Path, *, allow_missing: bool = False) -> Path:
    candidates = [
        baseline_root / f"e5_pcpg_low_encoder_lr_seed{seed}" / "checkpoints" / "best_avg_missing_top1.pth",
        baseline_root / f"e5_low_encoder_lr" / f"seed{seed}" / "checkpoints" / "best_avg_missing_top1.pth",
        baseline_root / f"e5_pcpg_low_encoder_lr_seed{seed}" / "checkpoints" / "best_top1.pth",
        baseline_root / f"e5_low_encoder_lr" / f"seed{seed}" / "checkpoints" / "best_top1.pth",
    ]
    candidates.extend(sorted((baseline_root / "best_checkpoints").glob(f"e5*seed{seed}*.pth")))
    for path in candidates:
        if path.exists():
            return path
    if allow_missing:
        return candidates[0]
    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find e5 checkpoint for e3 oracle eval. Checked:\n{checked}")


def _load_payload_for_job(job: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    if job["mode"] == "eval":
        source_config = _source_config_for_checkpoint(int(job["seed"]), Path(str(job["baseline_root"])))
        if source_config is not None:
            return yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
    base_path = ROOT / str(job["base_config"])
    if not base_path.exists():
        if dry_run:
            return {"_base_": str(job["base_config"])}
        raise FileNotFoundError(f"Base config not found: {base_path}")
    return yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}


def _source_config_for_checkpoint(seed: int, baseline_root: Path) -> Path | None:
    candidates = [
        baseline_root / f"e5_pcpg_low_encoder_lr_seed{seed}" / "final_config.yaml",
        baseline_root / "e5_low_encoder_lr" / f"seed{seed}" / "final_config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checkpoint = resolve_e3_checkpoint(seed, baseline_root, allow_missing=True)
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    if sidecar.exists():
        try:
            run_dir = Path(json.loads(sidecar.read_text(encoding="utf-8")).get("run_dir", ""))
        except json.JSONDecodeError:
            run_dir = Path()
        config = run_dir / "final_config.yaml"
        if config.exists():
            return config
    return None


def _command_for_job(job: dict[str, Any]) -> list[str]:
    if job["mode"] == "eval":
        checkpoint = str(job.get("checkpoint") or resolve_e3_checkpoint(int(job["seed"]), Path(str(job["baseline_root"])), allow_missing=True))
        return [
            "conda",
            "run",
            "-n",
            "kd_mm_beam",
            "kd-sensing-eval-u-mask-matrix",
            "--config",
            str(job["config_path"]),
            "--checkpoint",
            checkpoint,
            "--output-dir",
            str(Path(str(job["output_root"])) / str(job["experiment"]) / f"seed{int(job['seed'])}"),
            "--eval-oracle-gate",
        ]
    return ["conda", "run", "-n", "kd_mm_beam", "kd-sensing-train", "--config", str(job["config_path"])]


def _completed(job: dict[str, Any]) -> bool:
    if job.get("mode") == "eval":
        return (Path(str(job["output_root"])) / str(job["experiment"]) / f"seed{int(job['seed'])}" / "oracle_eval_matrix.csv").exists()
    run_dir = Path(str(job["output_root"])) / str(job["run_name"])
    return (run_dir / "metrics.json").exists() or (run_dir / "checkpoints" / "best_avg_missing_top1.pth").exists()


def _poll_finished(
    running: list[tuple[subprocess.Popen, dict[str, Any], Any]],
    *,
    wait: bool = False,
) -> list[dict[str, Any]]:
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
        job["end_time"] = _now()
        job["status"] = "completed" if code == 0 else "failed"
        if code != 0:
            failed.append(job)
        running.pop(index)
    return failed


def _gpu_count(running: list[tuple[subprocess.Popen, dict[str, Any], Any]], gpu: str) -> int:
    return sum(1 for _process, job, _handle in running if str(job.get("gpu")) == gpu)


def _canonical_experiment(value: str) -> str:
    key = str(value).strip()
    experiment = EXPERIMENT_ALIASES.get(key, key)
    if experiment not in experiment_specs():
        raise ValueError(f"Unknown experiment {value!r}. Available: {sorted(EXPERIMENT_ALIASES) + sorted(experiment_specs())}")
    return experiment


def _split(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _merge(*items: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge(merged[key], value)
            else:
                merged[key] = value
    return merged


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
