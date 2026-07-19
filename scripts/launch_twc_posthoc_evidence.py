#!/usr/bin/env python3
"""Run manifest-driven MMW post-hoc evidence.

Reliability/corruption stress is opt-in.  The default queue only profiles
complexity (and finalizes any already-produced mechanism traces); callers must
pass ``--run-reliability-stress`` to schedule corruption shards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kd_sensing.data.deepsense_twc import sha256_file, sha256_payload, write_json
from kd_sensing.evaluation.corruptions import CORRUPTION_GRID


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("T2", "S1", "masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m")
SEEDS = (1, 2, 3, 4, 5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan or run TWC post-hoc evidence.")
    parser.add_argument("--root", default="outputs/mmw_twc_fair_pattern_v1")
    parser.add_argument("--protocol-manifest", default="outputs/cache/mmw_twc_outer_v1/protocol_manifest.json")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--min-free-mib", type=int, default=12000)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument(
        "--run-reliability-stress",
        action="store_true",
        help="explicitly schedule GPS/image/radar/LiDAR corruption stress shards",
    )
    args = parser.parse_args()
    root = _path(args.root)
    manifest = prepare_plan(
        root,
        _path(args.protocol_manifest),
        reliability_stress=args.run_reliability_stress,
    )
    if not args.launch:
        print(json.dumps({"status": "planned", "manifest": str(manifest)}, indent=2))
        return 0
    return run_queue(
        manifest,
        gpus=tuple(int(value) for value in args.gpus.split(",")),
        min_free_mib=args.min_free_mib,
        poll_seconds=args.poll_seconds,
    )


def prepare_plan(root: Path, protocol_path: Path, *, reliability_stress: bool = False) -> Path:
    corruption_manifest = _prepare_corruption_manifest(protocol_path)
    request = {
        "protocol": "twc_posthoc_evidence_v1",
        "protocol_manifest": str(protocol_path.resolve()),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "corruptions": [{"name": item.name, "severity": item.severity} for item in CORRUPTION_GRID],
        "corruption_manifest": str(corruption_manifest),
        "corruption_manifest_sha256": sha256_file(corruption_manifest),
        "reliability_stress_enabled": bool(reliability_stress),
        "reliability_stress_policy": "explicit_flag_only",
        "complexity_batches": [1, 64],
    }
    request_sha256 = sha256_payload(request)
    path = root / f"posthoc_manifest_{request_sha256[:12]}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing post-hoc manifest differs: {path}")
        return path
    log_dir = root / "posthoc_logs"
    jobs = []
    if reliability_stress:
        for method in METHODS:
            for seed in SEEDS:
                for spec in CORRUPTION_GRID:
                    jobs.append(
                        {
                            "kind": "corruption",
                            "method": method,
                            "seed": seed,
                            "corruption": spec.name,
                            "severity": spec.severity,
                            "status": "planned",
                            "log_path": str((log_dir / f"{method}_seed{seed}_{spec.name}_s{spec.severity}.log").resolve()),
                        }
                    )
    for method in METHODS:
        jobs.append(
            {
                "kind": "complexity",
                "method": method,
                "seed": 1,
                "status": "planned",
                "log_path": str((log_dir / f"{method}_complexity.log").resolve()),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol": "twc_posthoc_evidence_v1",
        "protocol_manifest": str(protocol_path.resolve()),
        "request": request,
        "request_sha256": request_sha256,
        "reliability_stress_enabled": bool(reliability_stress),
        "jobs": jobs,
        "finalization_status": "planned",
    }
    payload["plan_sha256"] = sha256_payload(payload)
    write_json(path, payload)
    return path


def _prepare_corruption_manifest(parent_protocol: Path) -> Path:
    parent = json.loads(parent_protocol.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "protocol_id": "mmw_twc_corruption_v1",
        "parent_protocol_manifest": str(parent_protocol.resolve()),
        "parent_protocol_manifest_sha256": parent.get("manifest_sha256"),
        "seed_rule": "20260718 + training_seed*101 + severity; batch_seed += batch_index*97 + severity",
        "conditions": "clean_and_first_fixed_block80",
        "training_recipe_changed": False,
        "checkpoint_changed": False,
        "specification": {
            "gps_noise_std_normalized": [0.1, 0.25, 0.5],
            "image_center_occlusion_fraction": [0.15, 0.3, 0.45],
            "image_average_blur_kernel": [3, 7, 11],
            "radar_sample_std_noise_scale": [0.05, 0.15, 0.3],
            "lidar_bev_keep_probability": [0.75, 0.5, 0.25],
        },
    }
    payload["checksum"] = sha256_payload(payload)
    path = ROOT / "outputs/cache/mmw_twc_corruption_v1/spec_manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"Existing corruption manifest differs from the frozen specification: {path}")
    else:
        write_json(path, payload)
    return path.resolve()


def run_queue(manifest_path: Path, *, gpus: tuple[int, ...], min_free_mib: int, poll_seconds: float) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = payload["jobs"]
    running: dict[int, tuple[subprocess.Popen, Any, dict[str, Any]]] = {}
    while True:
        for job in jobs:
            if job["status"] == "running" and not _pid_alive(job.get("pid")):
                job["status"] = "done" if _complete(job, manifest_path.parent) else "planned"
        for gpu, (process, handle, job) in list(running.items()):
            result = process.poll()
            if result is None:
                continue
            handle.close()
            running.pop(gpu)
            job.update(status="done" if result == 0 and _complete(job, manifest_path.parent) else "failed", return_code=result, end_time=_now())
            write_json(manifest_path, payload)
        if any(job["status"] == "failed" for job in jobs):
            return 1
        if all(job["status"] == "done" for job in jobs):
            if payload.get("finalization_status") != "done":
                _finalize(
                    manifest_path.parent,
                    Path(payload["protocol_manifest"]),
                    reliability_stress=bool(payload.get("reliability_stress_enabled", False)),
                )
                payload["finalization_status"] = "done"
                write_json(manifest_path, payload)
            return 0
        free = _gpu_free_memory()
        planned = [job for job in jobs if job["status"] == "planned"]
        for gpu in gpus:
            if not planned or gpu in running or free.get(gpu, 0) < min_free_mib:
                continue
            job = planned.pop(0)
            command = _command(job, manifest_path.parent, Path(payload["protocol_manifest"]))
            log_path = Path(job["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            job.update(status="running", gpu=gpu, pid=process.pid, start_time=_now())
            running[gpu] = (process, handle, job)
            write_json(manifest_path, payload)
        time.sleep(poll_seconds)


def _command(job: dict[str, Any], root: Path, protocol_path: Path) -> list[str]:
    if job["kind"] == "corruption":
        return [
            sys.executable, str(ROOT / "scripts/eval_mmw_twc_corruption.py"),
            "--root", str(root), "--protocol-manifest", str(protocol_path),
            "--method", job["method"], "--seed", str(job["seed"]),
            "--corruption", job["corruption"], "--severity", str(job["severity"]),
        ]
    config = root / "generated_configs" / f"{job['method']}_seed1.yaml"
    checkpoint = root / job["method"] / "seed1" / "checkpoints" / "last.pth"
    output = root / "complexity" / f"{job['method']}.json"
    return [
        sys.executable, str(ROOT / "scripts/profile_twc_complexity.py"),
        "--config", str(config), "--checkpoint", str(checkpoint), "--output", str(output),
    ]


def _complete(job: dict[str, Any], root: Path) -> bool:
    if job["kind"] == "complexity":
        return (root / "complexity" / f"{job['method']}.json").is_file()
    return (
        root / "eval_corruption" / job["method"] / f"seed{job['seed']}"
        / f"{job['corruption']}_s{job['severity']}" / "provenance.json"
    ).is_file()


def _finalize(root: Path, protocol_path: Path, *, reliability_stress: bool = False) -> None:
    # Complexity-only post-hoc runs may finish before the outer evaluator has
    # emitted mechanism traces.  In that case leave mechanism finalization for
    # the normal evaluator and do not make the queue fail spuriously.
    trace_files = list((root / "eval_outer").glob("*/seed*/mechanism_trace.jsonl"))
    if trace_files:
        subprocess.run(
            [
                sys.executable, str(ROOT / "scripts/summarize_twc_mechanism.py"),
                "--eval-dir", str(root / "eval_outer"), "--protocol-manifest", str(protocol_path),
                "--output-dir", str(root / "mechanism_summary"),
            ],
            cwd=ROOT,
            check=True,
        )
    _summarize_complexity(root)
    if reliability_stress:
        _summarize_corruptions(root)


def _summarize_complexity(root: Path) -> None:
    rows = []
    for method in METHODS:
        payload = json.loads((root / "complexity" / f"{method}.json").read_text(encoding="utf-8"))
        row = {
            "method": method,
            "parameters_total": payload["parameters_total"],
            "parameters_trainable": payload["parameters_trainable"],
            "macs_batch1": payload.get("macs_batch1"),
            "gpu": payload["hardware"]["gpu"],
            "amp": payload["policy"]["amp"],
        }
        for item in payload["measurements"]:
            batch = int(item["batch_size"])
            row[f"latency_ms_batch{batch}"] = item["latency_ms_mean"]
            row[f"throughput_batch{batch}"] = item["throughput_samples_per_second"]
            row[f"peak_memory_mib_batch{batch}"] = item.get("peak_memory_mib")
        rows.append(row)
    _write_csv(root / "complexity" / "paper_complexity_table.csv", rows)


def _summarize_corruptions(root: Path) -> None:
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            for spec in CORRUPTION_GRID:
                path = root / "eval_corruption" / method / f"seed{seed}" / f"{spec.name}_s{spec.severity}" / "metrics.csv"
                selected = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "corruption": spec.name,
                        "severity": spec.severity,
                        "row_count": len(selected),
                        "top1": statistics.fmean(float(row["top1"]) for row in selected),
                        "normalized_gain": statistics.fmean(float(row["normalized_gain"]) for row in selected),
                        "gain_loss_db": statistics.fmean(float(row["gain_loss_db"]) for row in selected),
                    }
                )
    _write_csv(root / "eval_corruption" / "corruption_summary.csv", rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gpu_free_memory() -> dict[int, int]:
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return {int(index): int(memory) for index, memory in (line.split(",") for line in output.splitlines())}


def _pid_alive(value: Any) -> bool:
    try:
        return Path(f"/proc/{int(value)}").exists()
    except (TypeError, ValueError):
        return False


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
