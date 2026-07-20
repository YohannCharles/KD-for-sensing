#!/usr/bin/env python3
"""Evaluate each H2R simplification candidate as soon as its training finishes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml

from kd_sensing.data.mmw.twc_router_joint_stress import prepare_router_joint_stress_cache
from launch_mmw_h2r_simplification_screen import CANDIDATES, PROTOCOL_ID as TRAINING_PROTOCOL


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "mmw_h2r_simplification_joint_evaluation_v1"
DEFAULT_TRAINING_MANIFEST = ROOT / "outputs/mmw_h2r_simplification_screen_v1/training_manifest_seed1.json"
DEFAULT_OUTPUT = ROOT / "outputs/mmw_h2r_simplification_screen_v1/joint_stress_seed1"
DEFAULT_CACHE = ROOT / "outputs/cache/mmw_router_joint_stress_v1/fixed_state_cache.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", default=str(DEFAULT_TRAINING_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.bootstrap_iterations <= 0:
        parser.error("poll interval and bootstrap iterations must be positive")
    return run(
        training_manifest=Path(args.training_manifest).resolve(),
        output_root=Path(args.output_root).resolve(),
        cache=Path(args.cache).resolve(),
        poll_seconds=float(args.poll_seconds),
        bootstrap_iterations=int(args.bootstrap_iterations),
        plan_only=bool(args.plan_only),
    )


def run(
    *,
    training_manifest: Path,
    output_root: Path,
    cache: Path,
    poll_seconds: float,
    bootstrap_iterations: int,
    plan_only: bool = False,
) -> int:
    training = _read_json(training_manifest)
    _validate_training_manifest(training, training_manifest)
    prepared_cache = prepare_router_joint_stress_cache(cache)
    request = {
        "protocol": PROTOCOL_ID,
        "training_protocol": TRAINING_PROTOCOL,
        "training_manifest": str(training_manifest),
        "training_request_sha256": str(training["request_sha256"]),
        "cache": str(cache),
        "cache_sha256": _sha256(cache),
        "cache_checksum": str(prepared_cache["checksum"]),
        "output_root": str(output_root),
        "bootstrap_iterations": int(bootstrap_iterations),
        "candidates": [list(item) for item in CANDIDATES],
        "gpus": list(range(8)),
        "joint_launcher_sha256": _sha256(ROOT / "scripts/launch_mmw_router_joint_stress.py"),
        "joint_evaluator_sha256": _sha256(ROOT / "scripts/eval_mmw_router_joint_stress.py"),
        "watcher_sha256": _sha256(Path(__file__).resolve()),
        "claim_eligible": False,
    }
    request_sha256 = _payload_sha256(request)
    manifest_path = output_root / "evaluation_orchestration_manifest.json"
    manifest = _prepare_manifest(manifest_path, request, request_sha256, training)
    if plan_only:
        print(json.dumps({"status": "planned", "manifest": str(manifest_path), "jobs": 8}, indent=2))
        return 0

    children: dict[int, tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = {}
    while True:
        training = _read_json(training_manifest)
        _validate_training_manifest(training, training_manifest)
        training_by_candidate = {str(job["candidate"]): job for job in training["jobs"]}

        for gpu, (process, handle, job) in list(children.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job.update(
                status="complete" if code == 0 and _evaluation_complete(job) else "failed",
                returncode=int(code),
                ended_at=_now(),
            )
            children.pop(gpu)

        for job in manifest["jobs"]:
            source = training_by_candidate[str(job["candidate"])]
            if source.get("status") == "failed":
                job.update(status="failed_training", ended_at=_now())
                continue
            if job.get("status") == "running" and not _pid_alive(int(job.get("pid") or 0)):
                job["status"] = "complete" if _evaluation_complete(job) else "failed"
            if job.get("status") in {"planned", "failed"} and _evaluation_complete(job):
                job.update(status="complete", returncode=0)
            if source.get("status") == "done" and job.get("checkpoint_sha256") is None:
                _freeze_checkpoint(job, source)

        for job in manifest["jobs"]:
            gpu = int(job["gpu"])
            source = training_by_candidate[str(job["candidate"])]
            if (
                source.get("status") != "done"
                or job.get("checkpoint_sha256") is None
                or job.get("status") not in {"planned", "failed"}
                or gpu in children
                or int(job.get("attempts", 0)) >= 2
            ):
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts/launch_mmw_router_joint_stress.py"),
                "--output-root",
                str(job["output_root"]),
                "--cache",
                str(cache),
                "--config",
                str(job["config"]),
                "--checkpoint",
                str(job["checkpoint"]),
                "--gpus",
                str(gpu),
                "--allow-gpu0-3",
                "--batch-size",
                "64",
                "--bootstrap-iterations",
                str(bootstrap_iterations),
                "--retry-failed",
                "--launch",
            ]
            log_path = Path(job["log"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handle.write(f"\n[{_now()}] launch: {' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            job.update(
                status="running",
                pid=process.pid,
                attempts=int(job.get("attempts", 0)) + 1,
                started_at=_now(),
            )
            children[gpu] = (process, handle, job)

        statuses = {str(job.get("status")) for job in manifest["jobs"]}
        manifest.update(status="evaluating" if children else "waiting_for_training", updated_at=_now())
        _write_json(manifest_path, manifest)
        if statuses == {"complete"}:
            manifest.update(status="complete", completed_at=_now())
            _write_json(manifest_path, manifest)
            return 0
        if "failed_training" in statuses:
            manifest.update(status="failed_training", completed_at=_now())
            _write_json(manifest_path, manifest)
            return 1
        exhausted = [
            job
            for job in manifest["jobs"]
            if job.get("status") == "failed" and int(job.get("attempts", 0)) >= 2
        ]
        training_finished = all(
            job.get("status") in {"done", "failed"} for job in training["jobs"]
        )
        if exhausted and training_finished and not children:
            manifest.update(status="failed_evaluation", completed_at=_now())
            _write_json(manifest_path, manifest)
            return 1
        time.sleep(poll_seconds)


def _prepare_manifest(
    path: Path,
    request: Mapping[str, Any],
    request_sha256: str,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    if path.is_file():
        manifest = _read_json(path)
        if manifest.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing evaluation watcher differs from frozen request: {path}")
        return manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "logs").mkdir(exist_ok=True)
    jobs = []
    for training_job in training["jobs"]:
        candidate = str(training_job["candidate"])
        jobs.append(
            {
                "candidate": candidate,
                "gpu": int(training_job["gpu"]),
                "config": str(Path(training_job["config_path"]).resolve()),
                "config_sha256": str(training_job["config_sha256"]),
                "checkpoint": str((Path(training_job["run_dir"]) / "checkpoints/last.pth").resolve()),
                "checkpoint_sha256": None,
                "output_root": str((path.parent / candidate).resolve()),
                "log": str((path.parent / "logs" / f"{candidate}.log").resolve()),
                "status": "planned",
                "attempts": 0,
                "claim_eligible": False,
            }
        )
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "request": dict(request),
        "request_sha256": request_sha256,
        "status": "waiting_for_training",
        "created_at": _now(),
        "jobs": jobs,
    }
    _write_json(path, manifest)
    return manifest


def _validate_training_manifest(payload: Mapping[str, Any], path: Path) -> None:
    request = payload.get("request", {})
    jobs = payload.get("jobs", ())
    expected = {item[0]: (*item[1:], gpu) for gpu, item in enumerate(CANDIDATES)}
    if (
        payload.get("protocol") != TRAINING_PROTOCOL
        or payload.get("request_sha256") != _payload_sha256(request)
        or request.get("protocol") != TRAINING_PROTOCOL
        or request.get("candidates") != [list(item) for item in CANDIDATES]
        or request.get("gpus") != list(range(8))
        or int(request.get("seed", -1)) != 1
        or int(request.get("batch_size", -1)) != 64
        or len(jobs) != 8
    ):
        raise ValueError(f"Invalid H2R simplification training manifest: {path}")
    for job in jobs:
        candidate = str(job.get("candidate", ""))
        config = Path(str(job.get("config_path", "")))
        if candidate not in expected:
            raise ValueError(f"Unknown H2R simplification candidate: {candidate}")
        profile, supervision, epochs, loss_profile, gpu = expected[candidate]
        if (
            str(job.get("evidence_profile")) != profile
            or str(job.get("supervision")) != supervision
            or int(job.get("epochs", -1)) != epochs
            or str(job.get("loss_profile")) != loss_profile
            or int(job.get("gpu", -1)) != gpu
            or not config.is_file()
            or _sha256(config) != job.get("config_sha256")
        ):
            raise ValueError(f"H2R simplification job identity mismatch: {candidate}")
        config_payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        screen = config_payload.get("mmw_h2r_simplification_screen", {})
        if screen.get("candidate") != candidate or screen.get("evidence_profile") != profile:
            raise ValueError(f"H2R simplification config provenance mismatch: {candidate}")


def _freeze_checkpoint(job: dict[str, Any], source: Mapping[str, Any]) -> None:
    checkpoint = Path(job["checkpoint"])
    status = Path(str(source["run_dir"])) / "run_status.json"
    if not checkpoint.is_file() or not status.is_file() or _read_json(status).get("state") != "complete":
        raise ValueError(f"Completed training artifact is missing for {job['candidate']}")
    digest = _sha256(checkpoint)
    frozen = job.get("checkpoint_sha256")
    if frozen is not None and frozen != digest:
        raise ValueError(f"Checkpoint identity changed for {job['candidate']}")
    job["checkpoint_sha256"] = digest


def _evaluation_complete(job: Mapping[str, Any]) -> bool:
    root = Path(str(job["output_root"]))
    manifest = root / "evaluation_manifest.json"
    summary = root / "joint_summary.json"
    if not manifest.is_file() or not summary.is_file():
        return False
    try:
        return _read_json(manifest).get("status") == "complete"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
