#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "outputs/pcpg_radar_balance_v1"
DEFAULT_BASE_CONFIG = "outputs/scenes31_34_tinyvit_lmdb/generated_configs/scenes31_34_proto_randomdrop_subset_tinyvit_es40_seed{seed}.yaml"
DEFAULT_SEEDS = "1,2,3"
DEFAULT_GPUS = "1,2,3,4"
EVAL_SUBSETS = ["all", "drop_image", "drop_radar", "drop_lidar", "drop_gps", "image", "radar", "lidar", "gps"]


def experiment_specs() -> dict[str, dict[str, Any]]:
    pcpg = {"model": {"primary": {"fusion_type": "pcpg", "pcpg_fuse_level": "logits"}}}
    avg_missing = {"checkpoint": {"selection_metric": "avg_missing_top1"}, "evaluation": {"modality_subsets": {"enabled": True, "subsets": EVAL_SUBSETS}}}
    return {
        "e1_tinyvit_valacc_ckpt": {
            "mode": "train",
            "overrides": {"checkpoint": {"selection_metric": "val_acc"}},
        },
        "e2_tinyvit_avgmissing_ckpt": {
            "mode": "train",
            "overrides": avg_missing,
        },
        "e3_pcpg_oracle_eval": {
            "mode": "eval",
            "source_experiment": "e4_pcpg_branch_aux",
            "overrides": {**pcpg, "evaluation": {"eval_oracle_gate": True}},
        },
        "e4_pcpg_branch_aux": {
            "mode": "train",
            "overrides": _merge(
                pcpg,
                avg_missing,
                {
                    "loss": {
                        "pcpg_radar_balance": {"enabled": True},
                        "branch_aux_loss": True,
                        "radar_protect_loss": True,
                        "unimodal_aux_weight": 0.2,
                        "radar_aux_weight": 0.3,
                    }
                },
            ),
        },
        "e5_pcpg_low_encoder_lr": {
            "mode": "train",
            "overrides": _merge(
                pcpg,
                avg_missing,
                {
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
                },
            ),
        },
        "e6_pcpg_hard_subset_jepa": {
            "mode": "train",
            "overrides": _merge(
                pcpg,
                avg_missing,
                {
                    "loss": {
                        "pcpg_radar_balance": {"enabled": True},
                        "branch_aux_loss": True,
                        "radar_protect_loss": True,
                        "unimodal_aux_weight": 0.2,
                        "radar_aux_weight": 0.3,
                        "hard_subset_weighting": {"enabled": True, "full_weight": 0.5, "unknown_weight": 1.0},
                        "hard_subset_alpha": 1.5,
                        "hard_subset_focus": ["image_only", "lidar_only", "radar_only", "missing_image", "miss3"],
                        "use_jepa": True,
                        "jepa_weight": 0.05,
                    }
                },
            ),
        },
    }


def plan_jobs(
    *,
    experiments: list[str],
    seeds: list[int],
    gpus: list[str],
    slots_per_gpu: int,
    output_root: str,
    base_config: str,
) -> list[dict[str, Any]]:
    specs = experiment_specs()
    jobs: list[dict[str, Any]] = []
    for experiment in experiments:
        spec = specs[experiment]
        for seed in seeds:
            index = len(jobs)
            gpu = gpus[(index // max(int(slots_per_gpu), 1)) % len(gpus)]
            run_name = f"{experiment}_seed{seed}"
            config_path = Path(output_root) / "generated_configs" / f"{run_name}.yaml"
            mode = str(spec["mode"])
            command = _command_for_job(
                mode=mode,
                config_path=config_path,
                output_root=output_root,
                run_name=run_name,
                seed=seed,
                source_experiment=str(spec.get("source_experiment", "")),
            )
            jobs.append(
                {
                    "experiment": experiment,
                    "seed": int(seed),
                    "run_name": run_name,
                    "mode": mode,
                    "gpu": str(gpu),
                    "slot": index % max(int(slots_per_gpu), 1),
                    "wave": index // max(len(gpus) * max(int(slots_per_gpu), 1), 1),
                    "base_config": base_config.format(seed=seed),
                    "config_path": str(config_path),
                    "output_root": output_root,
                    "command": command,
                }
            )
    return jobs


def write_generated_configs(jobs: list[dict[str, Any]], *, dry_run: bool) -> None:
    specs = experiment_specs()
    for job in jobs:
        base_path = ROOT / str(job["base_config"])
        if not base_path.exists():
            if dry_run:
                payload = {"_base_": str(job["base_config"])}
            else:
                raise FileNotFoundError(f"Base config not found: {base_path}")
        else:
            payload = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        overrides = _merge(
            {
                "experiment": {"name": job["run_name"], "seed": int(job["seed"])},
                "output": {"dir": str(job["output_root"]), "run_name": job["run_name"], "group_by_scene": False},
            },
            specs[str(job["experiment"])]["overrides"],
        )
        payload = _merge(payload, overrides)
        path = Path(job["config_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_manifest(jobs: list[dict[str, Any]], output_root: str) -> Path:
    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "pcpg_radar_balance_v1_manifest.csv"
    fields = ["experiment", "seed", "run_name", "mode", "gpu", "slot", "wave", "base_config", "config_path", "command"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({key: job.get(key, "") if key != "command" else " ".join(job["command"]) for key in fields})
    (out_dir / "pcpg_radar_balance_v1_manifest.json").write_text(
        json.dumps(jobs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_jobs(jobs: list[dict[str, Any]], *, max_parallel: int, slots_per_gpu: int, force: bool, skip_completed: bool) -> int:
    running: list[tuple[subprocess.Popen, dict[str, Any]]] = []
    failures = 0
    for job in jobs:
        if skip_completed and not force and _completed(job):
            continue
        while len(running) >= int(max_parallel) or _gpu_count(running, str(job["gpu"])) >= int(slots_per_gpu):
            failures += _poll_finished(running)
            time.sleep(1.0)
        log_path = Path(job["output_root"]) / "logs" / f"{job['run_name']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((process, job))
    while running:
        failures += _poll_finished(running, wait=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch PCPG radar-balance v1 local experiments.")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiments", default="all")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--slots-per-gpu", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    all_specs = experiment_specs()
    experiments = list(all_specs) if args.experiments == "all" else _split(args.experiments)
    unknown = [item for item in experiments if item not in all_specs]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}. Available: {sorted(all_specs)}")
    jobs = plan_jobs(
        experiments=experiments,
        seeds=[int(item) for item in _split(args.seeds)],
        gpus=_split(args.gpus),
        slots_per_gpu=int(args.slots_per_gpu),
        output_root=str(args.output_root),
        base_config=str(args.base_config),
    )
    write_generated_configs(jobs, dry_run=bool(args.dry_run))
    manifest = write_manifest(jobs, str(args.output_root))
    print(f"Wrote manifest: {manifest}")
    for job in jobs:
        print(f"[gpu {job['gpu']}] {' '.join(job['command'])}")
    if args.dry_run:
        return 0
    return run_jobs(
        jobs,
        max_parallel=int(args.max_parallel),
        slots_per_gpu=int(args.slots_per_gpu),
        force=bool(args.force),
        skip_completed=bool(args.skip_completed),
    )


def _command_for_job(
    *,
    mode: str,
    config_path: Path,
    output_root: str,
    run_name: str,
    seed: int,
    source_experiment: str,
) -> list[str]:
    if mode == "eval":
        source = f"{source_experiment}_seed{seed}" if source_experiment else run_name
        checkpoint = Path(output_root) / source / "checkpoints" / "best_avg_missing_top1.pth"
        return [
            "conda",
            "run",
            "-n",
            "kd_mm_beam",
            "kd-sensing-eval-u-mask-matrix",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(Path(output_root) / "oracle_eval" / run_name),
            "--eval-oracle-gate",
        ]
    return ["conda", "run", "-n", "kd_mm_beam", "kd-sensing-train", "--config", str(config_path)]


def _completed(job: dict[str, Any]) -> bool:
    if job.get("mode") == "eval":
        return (Path(job["output_root"]) / "oracle_eval" / str(job["run_name"]) / "oracle_eval_matrix.csv").exists()
    run_dir = Path(job["output_root"]) / str(job["run_name"])
    return (run_dir / "metrics.json").exists() or (run_dir / "checkpoints" / "best_top1.pth").exists()


def _poll_finished(running: list[tuple[subprocess.Popen, dict[str, Any]]], *, wait: bool = False) -> int:
    failures = 0
    if wait and running:
        running[0][0].wait()
    for index in range(len(running) - 1, -1, -1):
        code = running[index][0].poll()
        if code is None:
            continue
        running.pop(index)
        failures += 0 if code == 0 else 1
    return failures


def _gpu_count(running: list[tuple[subprocess.Popen, dict[str, Any]]], gpu: str) -> int:
    return sum(1 for _process, job in running if str(job.get("gpu")) == gpu)


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


if __name__ == "__main__":
    raise SystemExit(main())
