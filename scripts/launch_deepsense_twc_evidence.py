#!/usr/bin/env python3
"""Plan and run the independent DeepSense6G secondary-evidence matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from kd_sensing.config.io import load_config
from kd_sensing.data.deepsense_twc import PROTOCOL_ID, SCENES, load_protocol, sha256_file, sha256_payload, write_json


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("T2", "masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m")
SEEDS = (1, 2, 3)
CONFIGS = {
    "T2": "configs/deepsense6g/t2.yaml",
    "masktrain_cls": "configs/deepsense6g/masktrain_cls.yaml",
    "amber_full": "configs/deepsense6g/amber_full.yaml",
    "rmbp_mm": "configs/deepsense6g/rmbp_mm.yaml",
    "amr_net_4m": "configs/deepsense6g/amr_net_4m.yaml",
}
FIDELITY = {
    "T2": {"scope": "project_mainline", "paper_equivalent": False},
    "masktrain_cls": {"scope": "plain_mask_training_control", "paper_equivalent": False},
    "amber_full": {"scope": "four_modality_local_adaptation", "paper_equivalent": False, "omitted_inputs": ["historical_beam_index"]},
    "rmbp_mm": {"scope": "channel_attention_local_adaptation", "paper_equivalent": False, "omitted_inputs": ["partial_beam_measurement"], "omitted_stages": ["unimodal_pretraining", "label_guided_similarity_imputation"]},
    "amr_net_4m": {"scope": "four_modality_window5_local_adaptation", "paper_equivalent": False},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or launch DeepSense6G TWC secondary training.")
    parser.add_argument("--output-root", default="outputs/deepsense6g_twc_secondary_v1")
    parser.add_argument("--protocol-manifest", default="outputs/cache/deepsense6g_twc_secondary_v1/protocol_manifest.json")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default=",".join(str(value) for value in SEEDS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    methods = _csv(args.methods, str)
    seeds = _csv(args.seeds, int)
    gpus = _csv(args.gpus, int)
    if any(method not in METHODS for method in methods) or len(set(methods)) != len(methods):
        parser.error(f"--methods must be a unique subset of {METHODS}")
    if any(seed <= 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain unique positive integers")
    if any(gpu not in range(8) for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain unique ids from 0 to 7")
    if args.batch_size <= 0 or args.batch_size % 16:
        parser.error("--batch-size must be a positive multiple of 16")
    if args.epochs <= 0 or args.min_free_mib <= 0 or args.poll_seconds <= 0:
        parser.error("epoch, memory, and poll values must be positive")
    manifest = prepare_plan(
        _path(args.output_root),
        _path(args.protocol_manifest),
        methods=methods,
        seeds=seeds,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    if not args.launch:
        print(json.dumps({"status": "planned", "manifest": str(manifest), "jobs": len(methods) * len(seeds)}, indent=2))
        return 0
    return run_queue(manifest, gpus=gpus, min_free_mib=args.min_free_mib, poll_seconds=args.poll_seconds)


def prepare_plan(
    output_root: Path,
    protocol_path: Path,
    *,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    batch_size: int,
    epochs: int,
) -> Path:
    protocol = load_protocol(protocol_path)
    request = {
        "protocol_manifest_sha256": protocol["manifest_sha256"],
        "methods": list(methods),
        "seeds": list(seeds),
        "dataset_scope": "deepsense6g_scene31_34_pooled_v1",
        "scenes": list(SCENES),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
    }
    request_sha256 = sha256_payload(request)
    plan_path = output_root / f"training_manifest_{request_sha256[:12]}.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing DeepSense6G plan differs from request: {plan_path}")
        return plan_path
    scene_records = sorted(protocol["scenes"], key=lambda item: int(item["scene"]))
    if tuple(int(item["scene"]) for item in scene_records) != SCENES:
        raise ValueError(f"DeepSense6G pooled protocol must contain exactly scenes {SCENES}.")
    config_dir = output_root / "generated_configs"
    log_dir = output_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in seeds:
        for method in methods:
            config = build_config(
                method,
                scene_records,
                protocol,
                output_root=output_root,
                seed=seed,
                batch_size=batch_size,
                epochs=epochs,
            )
            config_path = config_dir / f"{method}_seed{seed}.yaml"
            serialized = yaml.safe_dump(config, sort_keys=False)
            if config_path.exists() and config_path.read_text(encoding="utf-8") != serialized:
                raise FileExistsError(f"Existing DeepSense6G generated config differs: {config_path}")
            if not config_path.exists():
                config_path.write_text(serialized, encoding="utf-8")
            run_dir = output_root / method / f"seed{seed}"
            jobs.append(
                {
                    "method": method,
                    "variant": method,
                    "scope": "DeepSense6G-Scene31-34合并",
                    "scene_ids": list(SCENES),
                    "seed": seed,
                    "config_path": str(config_path.resolve()),
                    "config_sha256": sha256_file(config_path),
                    "run_dir": str(run_dir.resolve()),
                    "log_path": str((log_dir / f"{method}_seed{seed}.log").resolve()),
                    "evaluation_log_path": str((log_dir / f"{method}_seed{seed}.eval.log").resolve()),
                    "status": "planned",
                    "gpu": None,
                    "pid": None,
                    "return_code": None,
                    "evaluation_status": "planned",
                }
            )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "protocol_manifest": str(protocol_path.resolve()),
        "request": request,
        "request_sha256": request_sha256,
        "jobs": jobs,
    }
    payload["plan_sha256"] = sha256_payload(payload)
    write_json(plan_path, payload)
    return plan_path


def build_config(
    method: str,
    scene_records: list[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    output_root: Path,
    seed: int,
    batch_size: int,
    epochs: int,
) -> dict[str, Any]:
    cfg = load_config(ROOT / CONFIGS[method])
    dataset = cfg["data"]["dataset"]
    for key in ("scene", "data_root", "train_csv_name", "test_csv_name", "val_csv_name"):
        dataset.pop(key, None)
    dataset.update(
        domains=[
            {
                "id": f"scenario{int(item['scene'])}",
                "scene": int(item["scene"]),
                "data_root": str(item["data_root"]),
                "train_csv_name": str(item["train"]["path"]),
                "test_csv_name": str(item["test"]["path"]),
            }
            for item in scene_records
        ],
        portion_seed=int(seed),
    )
    loader = cfg["data"]["dataloader"]
    loader.update(train_batch_size=int(batch_size), test_batch_size=int(batch_size), validation_batch_size=int(batch_size))
    cfg["experiment"].update(name=method, seed=int(seed))
    cfg["temporal_missing"]["seed"] = int(seed)
    cfg["training"].update(
        epochs=int(epochs),
        max_epochs=int(epochs),
        resume=False,
        allow_tf32=False,
        cudnn_benchmark=False,
        final_test={"enabled": False, "reason": "fixed_secondary_evaluator_only"},
    )
    cfg["output"] = {
        "dir": str(output_root / method),
        "run_name": f"seed{seed}",
        "group_by_scene": False,
        "overwrite": False,
        "progress": {"enabled": False},
        "tensorboard": {"enabled": False},
    }
    cfg["deepsense6g_twc_evidence"] = {
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest_sha256": protocol["manifest_sha256"],
        "fixed_mask_cache_sha256": protocol["fixed_mask_cache"]["sha256"],
        "fixed_mask_cache_checksum": protocol["fixed_mask_cache"]["checksum"],
        "dataset_scope": "deepsense6g_scene31_34_pooled_v1",
        "scene_ids": list(SCENES),
        "pooled_dataset": dict(protocol["pooled_dataset"]),
        "scene_split_sha256": {
            f"scenario{int(item['scene'])}": {
                "train": str(item["train"]["sha256"]),
                "test": str(item["test"]["sha256"]),
            }
            for item in scene_records
        },
        "training_mask_seed": int(seed),
        "training_mask_schedule_id": "deepsense6g_fair_pattern_v1",
        "baseline_fidelity": FIDELITY[method],
    }
    return cfg


def run_queue(manifest_path: Path, *, gpus: tuple[int, ...], min_free_mib: int, poll_seconds: float) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest["jobs"]
    running: dict[int, tuple[subprocess.Popen, Any, dict[str, Any], str]] = {}
    while True:
        for job in jobs:
            if job.get("status") == "running" and not _pid_alive(job.get("pid")):
                job["status"] = "done" if _completed(job) else "planned"
                job["pid"] = None
                job["gpu"] = None
            if job.get("evaluation_status") == "running" and not _pid_alive(job.get("evaluation_pid")):
                job["evaluation_status"] = "done" if _completed_evaluation(job) else "planned"
                job["evaluation_pid"] = None
                job["evaluation_gpu"] = None
        for gpu, (process, handle, job, phase) in list(running.items()):
            result = process.poll()
            if result is None:
                continue
            handle.close()
            running.pop(gpu)
            if phase == "training":
                job.update(
                    status="done" if result == 0 and _completed(job) else "failed",
                    return_code=int(result),
                    end_time=_now(),
                )
            else:
                job.update(
                    evaluation_status="done" if result == 0 and _completed_evaluation(job) else "failed",
                    evaluation_return_code=int(result),
                    evaluation_end_time=_now(),
                )
            write_json(manifest_path, manifest)
        if any(job["status"] == "failed" or job.get("evaluation_status") == "failed" for job in jobs):
            return 1
        if all(job["status"] == "done" and job.get("evaluation_status") == "done" for job in jobs):
            if manifest.get("summary_status") != "done":
                subprocess.run(
                    [sys.executable, str(ROOT / "scripts/summarize_deepsense_twc_evidence.py"), "--root", str(manifest_path.parent)],
                    cwd=ROOT,
                    check=True,
                )
                manifest["summary_status"] = "done"
                write_json(manifest_path, manifest)
            return 0
        free = _gpu_free_memory()
        occupied = {int(job["gpu"]) for job in jobs if job.get("status") == "running" and job.get("gpu") is not None}
        occupied.update(
            int(job["evaluation_gpu"])
            for job in jobs
            if job.get("evaluation_status") == "running" and job.get("evaluation_gpu") is not None
        )
        planned = [job for job in jobs if job.get("status") == "planned"]
        training_active = bool(planned) or any(phase == "training" for _, _, _, phase in running.values())
        evaluations = [] if training_active else [job for job in jobs if job.get("evaluation_status") == "planned"]
        for gpu in gpus:
            if (not planned and not evaluations) or gpu in running or gpu in occupied or free.get(gpu, 0) < min_free_mib:
                continue
            phase = "training" if planned else "evaluation"
            job = (planned if planned else evaluations).pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            log_path = job["log_path"] if phase == "training" else job["evaluation_log_path"]
            handle = Path(log_path).open("a", encoding="utf-8")
            if phase == "training":
                command = [sys.executable, "-m", "kd_sensing.cli.train", "--config", job["config_path"]]
            else:
                command = [
                    sys.executable,
                    str(ROOT / "scripts/eval_deepsense_twc_evidence.py"),
                    "--root", str(manifest_path.parent),
                    "--protocol-manifest", str(manifest["protocol_manifest"]),
                    "--method", str(job["method"]),
                    "--seed", str(job["seed"]),
                    "--batch-size", str(manifest["request"]["batch_size"]),
                ]
            handle.write(f"\n[{_now()}] GPU{gpu}: {' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if phase == "training":
                job.update(status="running", gpu=gpu, pid=process.pid, start_time=_now())
            else:
                job.update(evaluation_status="running", evaluation_gpu=gpu, evaluation_pid=process.pid, evaluation_start_time=_now())
            running[gpu] = (process, handle, job, phase)
            write_json(manifest_path, manifest)
        time.sleep(poll_seconds)


def _completed(job: Mapping[str, Any]) -> bool:
    run_dir = Path(str(job["run_dir"]))
    status_path = run_dir / "run_status.json"
    if not (run_dir / "checkpoints/last.pth").is_file() or not status_path.is_file():
        return False
    try:
        return json.loads(status_path.read_text(encoding="utf-8")).get("state") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _completed_evaluation(job: Mapping[str, Any]) -> bool:
    path = (
        Path(str(job["run_dir"])).parents[1]
        / "eval_fixed"
        / str(job["method"])
        / f"seed{job['seed']}"
        / "provenance.json"
    )
    try:
        return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _gpu_free_memory() -> dict[int, int]:
    command = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return {int(index): int(memory) for index, memory in (line.split(",") for line in output.splitlines())}


def _pid_alive(value: Any) -> bool:
    try:
        return Path(f"/proc/{int(value)}").exists()
    except (TypeError, ValueError):
        return False


def _csv(value: str, kind):
    return tuple(kind(item.strip()) for item in value.split(",") if item.strip())


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
