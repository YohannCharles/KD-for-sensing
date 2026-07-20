#!/usr/bin/env python3
"""Prepare, preflight, and launch the eight-run MMW PGCD quick search."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml

import run_quick_pcer_validation as pcer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/pgcd_quick_search"
PROTOCOL_ID = "mmw_pgcd_quick_search_v1"
PROTOCOL_MANIFEST = ROOT / "outputs/cache/mmw_twc_outer_v1/protocol_manifest.json"
SEED = 1
EVAL_SEED = 20260720
EPOCHS = 16
BATCH_SIZE = 32
NUM_WORKERS = 4
EXPERIMENTS = (
    ("qv_c0_corrupt_global_prior", "C0", 4),
    ("qv_c1_severity_quality", "C1", 5),
    ("qv_c2_entropy_quality", "C2", 4),
    ("qv_c3_proto_drift_reg", "C3", 5),
    ("qv_c4_proto_drift_rank", "C4", 4),
    ("qv_c5_task_degradation", "C5", 5),
    ("qv_c6_combined_quality", "C6", 4),
    ("qv_c7_full_pgcd", "C7", 5),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight-all", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    output = Path(args.output_root).resolve()
    if not any((args.prepare, args.preflight_all, args.launch, args.status)):
        parser.error("select --prepare, --preflight-all, --launch, or --status")
    if args.prepare:
        prepare(output)
    if args.preflight_all:
        preflight(output)
    if args.launch:
        return launch(output, min_free_mib=int(args.min_free_mib), poll_seconds=float(args.poll_seconds))
    if args.status:
        print(json.dumps(status(output), ensure_ascii=False, indent=2))
    return 0


def prepare(output: Path) -> Path:
    protocol = _read_json(PROTOCOL_MANIFEST)
    domains = pcer._quick_domains(protocol)  # noqa: SLF001
    request = {
        "protocol": PROTOCOL_ID,
        "source_protocol_manifest": str(PROTOCOL_MANIFEST),
        "source_protocol_sha256": _sha256(PROTOCOL_MANIFEST),
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "effective_batch_size": BATCH_SIZE,
        "execution": "two_gpu_serial_queues_one_job_per_gpu",
        "num_workers": NUM_WORKERS,
        "history_window": 5,
        "modalities": ["image", "radar", "gps", "lidar"],
        "num_blocks": 20,
        "selection_split": "frozen_inner_validation",
        "test_split": "historical_h5p1_strict_v2_claim_ineligible",
        "initialization": "same_seed_scratch_no_common_checkpoint_available",
        "claim_eligible": False,
        "experiments": [list(item) for item in EXPERIMENTS],
    }
    request_sha = _payload_sha(request)
    manifest_path = output / "training_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("request_sha256") != request_sha:
            raise ValueError(f"Existing PGCD request differs from the frozen request: {manifest_path}")
        return manifest_path
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, ablation, gpu in EXPERIMENTS:
        config = build_experiment_config(output, domains, name=name, ablation=ablation)
        run_dir = output / name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "resolved_config.yaml"
        _write_yaml(config_path, config)
        jobs.append(
            {
                "experiment": name,
                "ablation": ablation,
                "variant": ablation.lower(),
                "gpu": gpu,
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
                "run_dir": str(run_dir),
                "train_log": str(run_dir / "train.log"),
                "eval_log": str(run_dir / "eval.log"),
                "status": "planned",
                "return_code": None,
                "claim_eligible": False,
            }
        )
    _write_yaml(
        output / "common_resolved_config.yaml",
        {
            **request,
            "training_sampling": {
                "clean_only": 0.20,
                "single_sensor_corruption": 0.40,
                "two_sensor_corruption": 0.20,
                "temporal_block_corruption": 0.10,
                "single_sensor_missing": 0.10,
            },
            "severity_sampling": {"mild": 0.30, "medium": 0.30, "severe": 0.30, "missing": 0.10},
            "loss": {
                "lambda_proto": 0.2,
                "lambda_modality_proto": 0.1,
                "lambda_quality": 0.2,
                "lambda_rank": 0.1,
                "lambda_consistency": 0.2,
                "rank_margin": 0.1,
                "alpha_drift": 0.5,
                "alpha_task": 0.5,
            },
            "forbidden": {
                "use_channel": False,
                "use_csi": False,
                "use_path_features": False,
                "use_channel_gain_target": False,
            },
        },
    )
    _write_shell_copy(output)
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol": PROTOCOL_ID,
            "request": request,
            "request_sha256": request_sha,
            "jobs": jobs,
            "preflight_status": "pending",
            "status": "planned",
            "created_at": _now(),
        },
    )
    return manifest_path


def build_experiment_config(
    output: Path,
    domains: list[dict[str, str]],
    *,
    name: str,
    ablation: str,
) -> dict[str, Any]:
    config = pcer.build_experiment_config(
        output,
        domains,
        name=name,
        ablation=ablation,
        pcer_mode="disabled",
        fusion_type="uniform_mean",
        oracle_weight=0.0,
    )
    variant = ablation.lower()
    dataset = config["data"]["dataset"]
    dataset["include_router_utility_targets"] = False
    dataset["include_router_corruption_metadata"] = True
    config["data"]["dataloader"].update(
        {
            "train_batch_size": BATCH_SIZE,
            "validation_batch_size": BATCH_SIZE,
            "test_batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "persistent_workers": True,
            "prefetch_factor": 1,
        }
    )
    primary = config["model"]["primary"]
    primary.pop("pcer", None)
    primary["pgcd"] = {
        "variant": variant,
        "hidden_dim": 64,
        "embedding_dim": 8,
        "dropout": 0.0,
        "beta_reliability_init": 1.0,
    }
    loss = config["loss"]["u_mask_beam_jepa"]
    loss.pop("pcer", None)
    loss["pgcd"] = {
        "variant": variant,
        "global_seed": EVAL_SEED,
        "lambda_quality": 0.2,
        "lambda_rank": 0.1,
        "lambda_consistency": 0.2,
        "rank_margin": 0.1,
        "rank_target_epsilon": 0.02,
        "task_clip": 4.0,
        "alpha_drift": 0.5,
        "alpha_task": 0.5,
    }
    config["temporal_missing"] = {
        "enabled": False,
        "history_window": 5,
        "prediction_window": 1,
        "mode": "none",
        "preserve_unmasked_for_superset": False,
    }
    config["training"].update(
        {
            "epochs": EPOCHS,
            "max_epochs": EPOCHS,
            "checkpoint_selection": "best_validation_loss",
            "amp": {"enabled": True, "dtype": "bfloat16", "grad_scaler": False},
            "validation": {"interval_epochs": 1},
            "final_test": {"enabled": False, "reason": "pgcd_fixed_evaluator_after_training"},
        }
    )
    config["output"].update({"dir": str(output), "run_name": name, "overwrite": True})
    config["mmw_all_weather_protocol"].update(
        {"split_tag": PROTOCOL_ID, "screening_role": "pgcd_inner_claim_ineligible", "checkpoint_policy": "best_validation_loss"}
    )
    config.pop("mmw_quick_pcer_protocol", None)
    config["mmw_pgcd_protocol"] = {
        "protocol": PROTOCOL_ID,
        "experiment": name,
        "ablation": ablation,
        "variant": variant,
        "seed": SEED,
        "eval_seed": EVAL_SEED,
        "claim_eligible": False,
        "use_channel": False,
        "use_csi": False,
        "use_path_features": False,
        "use_channel_gain_target": False,
        "weather_label_used_as_input": False,
    }
    return config


def preflight(output: Path) -> None:
    from kd_sensing.config import load_config
    from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config

    manifest_path = prepare(output)
    manifest = _read_json(manifest_path)
    rows = []
    for job in manifest["jobs"]:
        config = load_config(job["config_path"])
        resolved = u_mask_beam_jepa_config(config)
        rows.append(
            {
                "experiment": job["experiment"],
                "variant": resolved["pgcd"]["variant"],
                "channel_free": True,
                "config_sha256": _sha256(Path(job["config_path"])),
            }
        )
    command = [
        "conda",
        "run",
        "-n",
        "kd_mm_beam",
        "pytest",
        "tests/test_pgcd.py",
        "tests/test_pgcd_quick_search.py",
        "-q",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / "preflight_tests.txt").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        manifest["preflight_status"] = "failed"
        _write_json(manifest_path, manifest)
        raise RuntimeError("PGCD preflight tests failed; see preflight_tests.txt.")
    _write_json(output / "preflight_config_audit.json", rows)
    manifest.update(preflight_status="passed", preflight_at=_now())
    _write_json(manifest_path, manifest)


def launch(output: Path, *, min_free_mib: int, poll_seconds: float) -> int:
    manifest_path = output / "training_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("preflight_status") != "passed":
        raise RuntimeError("PGCD preflight must pass before launch.")
    free = _gpu_memory("memory.free")
    blocked = {
        int(job["gpu"]): free.get(int(job["gpu"]), 0)
        for job in manifest["jobs"]
        if not _completed(job) and free.get(int(job["gpu"]), 0) < int(min_free_mib)
    }
    if blocked:
        raise RuntimeError(f"PGCD GPUs below free-memory threshold: {blocked}")
    queues: dict[int, list[dict[str, Any]]] = {}
    for job in manifest["jobs"]:
        if _completed(job):
            job["status"] = "done"
            continue
        job["status"] = "queued"
        queues.setdefault(int(job["gpu"]), []).append(job)
    running: dict[int, tuple[subprocess.Popen, Any, dict[str, Any]]] = {}
    manifest.update(status="running" if any(queues.values()) else "complete", launched_at=_now())
    _write_json(manifest_path, manifest)
    _write_pids(output, manifest)
    while running or any(queues.values()):
        for gpu in sorted(queues):
            if gpu in running or not queues[gpu]:
                continue
            job = queues[gpu].pop(0)
            process, handle = _start_training_job(job, gpu=gpu)
            running[gpu] = (process, handle, job)
            _write_json(manifest_path, manifest)
            _write_pids(output, manifest)
            time.sleep(1)
        used = _gpu_memory("memory.used")
        for gpu, (process, handle, job) in list(running.items()):
            job["peak_gpu_memory_mib"] = max(
                int(job.get("peak_gpu_memory_mib", 0)),
                max(0, used.get(gpu, 0) - int(job.get("baseline_gpu_memory_mib", 0))),
            )
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job.update(
                status="done" if code == 0 and _completed(job) else "failed",
                return_code=int(code),
                end_time=_now(),
            )
            del running[gpu]
        _write_json(manifest_path, manifest)
        _write_pids(output, manifest)
        if running or any(queues.values()):
            time.sleep(float(poll_seconds))
    manifest.update(
        status="complete" if all(job["status"] == "done" for job in manifest["jobs"]) else "failed",
        completed_at=_now(),
    )
    _write_json(manifest_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


def _start_training_job(job: dict[str, Any], *, gpu: int):
    handle = Path(job["train_log"]).open("a", encoding="utf-8")
    command = [
        "conda",
        "run",
        "-n",
        "kd_mm_beam",
        "--no-capture-output",
        "kd-sensing-train",
        "--config",
        job["config_path"],
    ]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    baseline = _gpu_memory("memory.used").get(gpu, 0)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    job.update(
        status="running",
        pid=process.pid,
        start_time=_now(),
        baseline_gpu_memory_mib=baseline,
        peak_gpu_memory_mib=0,
    )
    return process, handle


def status(output: Path) -> dict[str, Any]:
    manifest = _read_json(output / "training_manifest.json")
    return {
        "status": manifest.get("status"),
        "preflight_status": manifest.get("preflight_status"),
        "jobs": [
            {
                "experiment": job["experiment"],
                "gpu": job["gpu"],
                "status": "done" if _completed(job) else job.get("status"),
                "pid": job.get("pid"),
                "return_code": job.get("return_code"),
            }
            for job in manifest["jobs"]
        ],
    }


def _completed(job: dict[str, Any]) -> bool:
    run = Path(job["run_dir"])
    status_path = run / "run_status.json"
    return (
        (run / "checkpoints/best.pth").is_file()
        and status_path.is_file()
        and _read_json(status_path).get("state") == "complete"
    )


def _write_shell_copy(output: Path) -> None:
    source = ROOT / "scripts/run_pgcd_quick_search_gpu4_5.sh"
    if source.is_file():
        target = output / source.name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        target.chmod(0o755)


def _write_pids(output: Path, manifest: dict[str, Any]) -> None:
    _write_json(
        output / "pids.json",
        {job["experiment"]: {key: job.get(key) for key in ("gpu", "pid", "status")} for job in manifest["jobs"]},
    )


def _gpu_memory(field: str) -> dict[int, int]:
    text = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu=index,{field}", "--format=csv,noheader,nounits"], text=True
    )
    return {int(index): int(value) for index, value in (line.split(",", 1) for line in text.splitlines())}


def _payload_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
