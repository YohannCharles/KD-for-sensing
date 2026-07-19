#!/usr/bin/env python3
"""Wait for the seed1 dynamic Router screen, then evaluate all candidates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
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

from kd_sensing.data.mmw.twc_router_joint_stress import (
    BALANCE_POLICY as JOINT_BALANCE_POLICY,
    CORRUPTION_SEVERITY,
    GENERATOR as JOINT_MASK_GENERATOR,
    JOINT_RATES,
    MASKS_PER_RATE,
    MASK_SEED,
    prepare_router_joint_stress_cache,
)
from kd_sensing.data.mmw.twc_router_joint_training import (
    PANEL_SEED as ROUTER_PANEL_SEED,
    PROTOCOL_ID as ROUTER_PANEL_PROTOCOL,
    load_router_joint_training_panel,
)
from kd_sensing.evaluation.corruptions import CORRUPTION_PARAMETERS

from launch_mmw_dynamic_router_screen import (
    CANDIDATES,
    PROTOCOL_ID as TRAINING_PROTOCOL_ID,
    ROUTER_RELIABILITY_SOURCE,
    UTILITY_NUMERIC_POLICY,
)
from launch_mmw_router_joint_stress import (
    BRANCH_ALGORITHM,
    CORRUPTION_SEED,
    DYNAMIC_FUSIONS,
    EVALUATOR_ALGORITHM,
    EVALUATOR_PROTOCOL,
    _config_router_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "mmw_dynamic_router_joint_evaluation_v1"
DEFAULT_SCREEN_ROOT = ROOT / "outputs/mmw_dynamic_router_screen_v1"
DEFAULT_TRAINING_MANIFEST = DEFAULT_SCREEN_ROOT / "training_manifest_seed1.json"
DEFAULT_REPAIR_TRAINING_MANIFEST = (
    ROOT / "outputs/mmw_dynamic_router_power_ampfix_v1/training_manifest_seed1.json"
)
DEFAULT_OUTPUT = DEFAULT_SCREEN_ROOT / "joint_stress_seed1"
DEFAULT_CACHE = ROOT / "outputs/cache/mmw_router_joint_stress_v1/fixed_state_cache.json"


def _joint_stress_request_identity() -> dict[str, Any]:
    return {
        "mask_seed": MASK_SEED,
        "mask_generator": JOINT_MASK_GENERATOR,
        "mask_balance_policy": JOINT_BALANCE_POLICY,
        "corruption_seed": CORRUPTION_SEED,
        "corruption_severity": CORRUPTION_SEVERITY,
        "corruption_parameters": {
            name: {
                "unit": CORRUPTION_PARAMETERS[name]["unit"],
                "value": CORRUPTION_PARAMETERS[name]["values"][CORRUPTION_SEVERITY - 1],
            }
            for name in ("image_occlusion", "radar_noise", "lidar_sparsify", "gps_noise")
        },
        "condition_count": 1 + len(JOINT_RATES) * MASKS_PER_RATE,
        "joint_rates": [float(value) for value in JOINT_RATES],
        "masks_per_rate": MASKS_PER_RATE,
        "batch_size": 64,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", default=str(DEFAULT_TRAINING_MANIFEST))
    parser.add_argument(
        "--repair-training-manifest",
        default="",
        help="Optional four-Power AMP-fix manifest; may replace only same-name Power jobs.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--child-poll-seconds", type=float, default=2.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--wait-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.child_poll_seconds <= 0:
        parser.error("poll intervals must be positive")
    if args.bootstrap_iterations <= 0 or args.wait_timeout_seconds < 0:
        parser.error("bootstrap iterations must be positive and timeout must be non-negative")
    path = prepare_plan(
        training_manifest=Path(args.training_manifest).resolve(),
        repair_training_manifest=(
            Path(args.repair_training_manifest).resolve()
            if str(args.repair_training_manifest).strip()
            else None
        ),
        output_root=Path(args.output_root).resolve(),
        cache=Path(args.cache).resolve(),
        bootstrap_iterations=int(args.bootstrap_iterations),
        child_poll_seconds=float(args.child_poll_seconds),
    )
    if not args.launch:
        print(json.dumps({"status": "planned", "manifest": str(path), "jobs": 8}, indent=2))
        return 0
    return run_plan(
        path,
        poll_seconds=float(args.poll_seconds),
        wait_timeout_seconds=float(args.wait_timeout_seconds),
    )


def prepare_plan(
    *,
    training_manifest: Path,
    repair_training_manifest: Path | None = None,
    output_root: Path,
    cache: Path,
    bootstrap_iterations: int,
    child_poll_seconds: float,
) -> Path:
    training = _read_json(training_manifest)
    primary_jobs = _validate_training_manifest(training, training_manifest)
    repair = _read_json(repair_training_manifest) if repair_training_manifest is not None else None
    training_jobs = _merge_training_jobs(
        primary=training,
        primary_path=training_manifest,
        primary_jobs=primary_jobs,
        repair=repair,
        repair_path=repair_training_manifest,
    )
    manifest_requests = {
        "primary": {
            "path": str(training_manifest),
            "request_sha256": str(training["request_sha256"]),
        }
    }
    if repair is not None and repair_training_manifest is not None:
        manifest_requests["repair"] = {
            "path": str(repair_training_manifest),
            "request_sha256": str(repair["request_sha256"]),
        }
    request = {
        "protocol": PROTOCOL_ID,
        "training_protocol": TRAINING_PROTOCOL_ID,
        "training_manifests": manifest_requests,
        "candidate_sources": {
            str(job["candidate"]): {
                "role": str(job["_source_role"]),
                "manifest": str(job["_source_manifest"]),
                "request_sha256": str(job["_source_request_sha256"]),
                "config_sha256": str(job["config_sha256"]),
                "checkpoint_sha256": None,
            }
            for job in training_jobs
        },
        "cache": str(cache),
        "output_root": str(output_root),
        "bootstrap_iterations": int(bootstrap_iterations),
        "child_poll_seconds": float(child_poll_seconds),
        "fusion_branches": list(DYNAMIC_FUSIONS),
        "joint_stress_protocol": EVALUATOR_PROTOCOL,
        "branch_algorithm": BRANCH_ALGORITHM,
        "evaluator_algorithm": EVALUATOR_ALGORITHM,
        "identity_state": "pending_training",
        "cache_sha256": None,
        "cache_checksum": None,
        **_joint_stress_request_identity(),
        "joint_launcher_sha256": _sha256(ROOT / "scripts/launch_mmw_router_joint_stress.py"),
        "evaluator_sha256": _sha256(ROOT / "scripts/eval_mmw_router_joint_stress.py"),
        "oracle_helper_sha256": _sha256(ROOT / "scripts/eval_mmw_router_oracle_gap.py"),
        "corruption_runtime_sha256": _sha256(ROOT / "src/kd_sensing/evaluation/corruptions.py"),
        "joint_cache_runtime_sha256": _sha256(ROOT / "src/kd_sensing/data/mmw/twc_router_joint_stress.py"),
        "summary_sha256": _sha256(ROOT / "scripts/summarize_mmw_router_joint_stress.py"),
        "split": "frozen_inner_validation_only",
        "claim_eligible": False,
    }
    request_sha256 = _payload_sha256(request)
    path = output_root / "evaluation_orchestration_manifest.json"
    if path.is_file():
        manifest = _read_json(path)
        if manifest.get("request", {}).get("identity_state") == "frozen_training":
            _validate_frozen_request(manifest, request, training_jobs, output_root)
            return path
        if manifest.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing dynamic evaluation plan differs from the frozen request: {path}")
        _validate_orchestration_jobs(manifest, training_jobs, output_root)
        return path

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    jobs = []
    for training_job in training_jobs:
        candidate = str(training_job["candidate"])
        jobs.append(
            {
                "candidate": candidate,
                "router_variant": str(training_job["router_variant"]),
                "supervision": str(training_job["supervision"]),
                "gpu": int(training_job["gpu"]),
                "config": str(Path(training_job["config_path"]).resolve()),
                "checkpoint": str((Path(training_job["run_dir"]) / "checkpoints/last.pth").resolve()),
                "output_root": str((output_root / candidate).resolve()),
                "log": str((output_root / "logs" / f"{candidate}.log").resolve()),
                "status": "planned",
                "pid": None,
                "returncode": None,
                "attempts": 0,
                "attempt_id": None,
                "config_sha256": str(training_job["config_sha256"]),
                "checkpoint_sha256": None,
                "claim_eligible": False,
                "training_source": dict(request["candidate_sources"][candidate]),
            }
        )
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "request": request,
        "request_sha256": request_sha256,
        "status": "waiting_for_training",
        "created_at": _now(),
        "jobs": jobs,
    }
    _validate_orchestration_jobs(manifest, training_jobs, output_root)
    _write_json(path, manifest)
    return path


def run_plan(path: Path, *, poll_seconds: float, wait_timeout_seconds: float) -> int:
    manifest = _read_json(path)
    request = manifest["request"]
    started = time.monotonic()
    while True:
        state, detail = _training_state(manifest)
        manifest.update(status=f"training_{state}", training_state=detail, updated_at=_now())
        if state == "ready":
            _freeze_evaluation_identity(manifest, detail)
        _write_json(path, manifest)
        if state == "ready":
            break
        if state == "failed":
            manifest.update(status="failed_training", completed_at=_now())
            _write_json(path, manifest)
            return 1
        if wait_timeout_seconds and time.monotonic() - started >= wait_timeout_seconds:
            manifest.update(status="timed_out_waiting_for_training", completed_at=_now())
            _write_json(path, manifest)
            return 1
        time.sleep(poll_seconds)

    children: dict[int, tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = {}
    while True:
        for gpu, (process, handle, job) in list(children.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            complete = returncode == 0 and _candidate_complete(job, request)
            job.update(
                status="complete" if complete else "failed",
                returncode=int(returncode),
                ended_at=_now(),
            )
            if complete:
                job.pop("failed_attempt_id", None)
            else:
                job["failed_attempt_id"] = _job_attempt_id(job)
            children.pop(gpu)

        for job in manifest["jobs"]:
            _reconcile_candidate_job(job, request)

        occupied = {
            int(job["gpu"])
            for job in manifest["jobs"]
            if job.get("status") == "running" and _pid_alive(int(job.get("pid") or 0))
        }
        for job in manifest["jobs"]:
            gpu = int(job["gpu"])
            if (
                job.get("status") not in {"planned", "failed"}
                or int(job.get("attempts", 0)) >= 2
                or gpu in children
                or gpu in occupied
            ):
                continue
            checkpoint = Path(job["checkpoint"])
            job["checkpoint_sha256"] = _sha256(checkpoint)
            # A retry is the only point at which evidence from a failed child
            # attempt may be reconsidered.  Until this launch starts, an old
            # complete-looking output cannot overwrite the known non-zero exit.
            job.pop("failed_attempt_id", None)
            command = _candidate_command(job, request)
            handle = Path(job["log"]).open("a", encoding="utf-8")
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
                started_at=_now(),
                attempts=int(job.get("attempts", 0)) + 1,
            )
            children[gpu] = (process, handle, job)
            occupied.add(gpu)
        manifest.update(status="evaluating", updated_at=_now())
        _write_json(path, manifest)

        statuses = {str(job["status"]) for job in manifest["jobs"]}
        if statuses == {"complete"} and not children:
            break
        if "failed" in statuses and not children and not any(
            job.get("status") == "running" and _pid_alive(int(job.get("pid") or 0))
            for job in manifest["jobs"]
        ):
            manifest.update(status="failed_evaluation", completed_at=_now())
            _write_json(path, manifest)
            return 1
        time.sleep(poll_seconds)

    summary = summarize_candidates(path.parent, manifest)
    manifest.update(status="complete", completed_at=_now(), summary=summary)
    _write_json(path, manifest)
    return 0


def _job_attempt_id(job: Mapping[str, Any]) -> str:
    value = str(job.get("attempt_id") or "").strip()
    return "" if value.lower() == "none" else value


def _reconcile_candidate_job(job: dict[str, Any], request: Mapping[str, Any]) -> None:
    # Never consume pre-existing output while the launcher responsible for the
    # current attempt is still alive; that output may belong to an earlier
    # failed invocation with the same frozen orchestration identity.
    if job.get("status") == "running" and _pid_alive(int(job.get("pid") or 0)):
        return
    attempt = _job_attempt_id(job)
    failed_attempt = str(job.get("failed_attempt_id") or "").strip()
    failed_current_attempt = (
        job.get("status") == "failed"
        and bool(attempt)
        and failed_attempt == attempt
    )
    if not failed_current_attempt and _candidate_complete(job, request):
        job.update(status="complete", returncode=0)
        return
    if job.get("status") == "complete":
        exhausted = int(job.get("attempts", 0)) >= 2
        job.update(
            status="failed" if exhausted else "planned",
            pid=None,
            returncode=job.get("returncode") if exhausted else None,
            failure_reason="Recorded complete candidate fails the frozen evidence identity.",
        )
        return
    if job.get("status") == "running":
        job.update(
            status="planned",
            pid=None,
            returncode=None,
            failure_reason="Detached candidate launcher exited before frozen evidence completed.",
        )


def _freeze_evaluation_identity(manifest: dict[str, Any], detail: Mapping[str, Any]) -> None:
    request = manifest["request"]
    previous_state = request.get("identity_state")
    previous_request_sha = str(manifest.get("request_sha256", ""))
    cache_path = Path(str(request["cache"])).resolve()
    cache = prepare_router_joint_stress_cache(cache_path)
    checkpoint_sha256 = {
        str(item["candidate"]): str(item["checkpoint_sha256"])
        for item in detail["artifacts"]
    }
    existing = request.get("candidate_sources", {})
    for job in manifest["jobs"]:
        candidate = str(job["candidate"])
        checkpoint = checkpoint_sha256[candidate]
        config_sha = _sha256(Path(job["config"]))
        attempt_id = _payload_sha256(
            {
                "candidate": candidate,
                "config_sha256": config_sha,
                "checkpoint_sha256": checkpoint,
                "training_sources": job["training_source"],
                "cache_checksum": str(cache["checksum"]),
                "evaluator_algorithm": EVALUATOR_ALGORITHM,
            }
        )
        source = existing[candidate]
        expected_source = {
            **job["training_source"],
            "config_sha256": config_sha,
            "checkpoint_sha256": checkpoint,
        }
        if request.get("identity_state") == "frozen_training" and source != expected_source:
            raise ValueError(f"Frozen candidate identity changed: {candidate}")
        request["candidate_sources"][candidate] = expected_source
        job.update(config_sha256=config_sha, checkpoint_sha256=checkpoint, attempt_id=attempt_id)
    if previous_state not in {"pending_training", "frozen_training"}:
        raise ValueError("Unknown dynamic evaluation identity state.")
    request.update(
        identity_state="frozen_training",
        cache_sha256=_sha256(cache_path),
        cache_checksum=str(cache["checksum"]),
        identity_frozen_at=request.get("identity_frozen_at") or _now(),
    )
    new_request_sha = _payload_sha256(request)
    if previous_state == "frozen_training":
        # A pending plan may be promoted once; subsequent resumes must be immutable.
        if previous_request_sha != new_request_sha:
            raise ValueError("Frozen dynamic evaluation request identity drifted.")
    manifest["request_sha256"] = new_request_sha


def _training_state(orchestration: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    request = orchestration["request"]
    sources: dict[str, dict[str, Any]] = {}
    for role, identity in request["training_manifests"].items():
        payload = _read_json(Path(identity["path"]))
        if payload.get("request_sha256") != identity["request_sha256"]:
            return "failed", {"reason": "training_request_identity_drift", "role": role}
        sources[str(role)] = payload

    primary = sources["primary"]
    repair = sources.get("repair")
    try:
        primary_validated = _validate_manifest_jobs(
            primary,
            Path(request["training_manifests"]["primary"]["path"]),
            candidates=CANDIDATES,
            require_numeric_policy=False,
        )
        repair_validated = (
            _validate_manifest_jobs(
                repair,
                Path(request["training_manifests"]["repair"]["path"]),
                candidates=tuple(item for item in CANDIDATES if item[2] == "beam_power"),
                require_numeric_policy=True,
            )
            if repair is not None
            else []
        )
    except (KeyError, ValueError) as exc:
        return "failed", {"reason": "training_manifest_validation_failed", "detail": str(exc)}
    primary_jobs = {str(job["candidate"]): job for job in primary_validated}
    repair_jobs = {str(job["candidate"]): job for job in repair_validated}
    if repair is not None:
        superseded = [primary_jobs[name] for name in _power_candidate_names()]
        if any(job.get("status") not in {"running", "failed"} for job in superseded):
            return "failed", {"reason": "repair_did_not_replace_failed_primary_power"}
        if all(job.get("status") == "failed" for job in superseded) and any(
            int(job.get("return_code", 0)) != -15 for job in superseded
        ):
            return "failed", {"reason": "primary_power_failure_identity_mismatch"}

    artifacts = []
    waiting = []
    for planned in orchestration["jobs"]:
        source = planned["training_source"]
        source_jobs = primary_jobs if source["role"] == "primary" else repair_jobs
        job = source_jobs.get(str(planned["candidate"]))
        if job is None:
            return "failed", {"reason": "candidate_missing_from_training_source", "candidate": planned["candidate"]}
        status = str(job.get("status", ""))
        if status == "failed":
            return "failed", {"reason": "selected_training_job_failed", "candidate": job["candidate"]}
        if status != "done":
            waiting.append({"candidate": job["candidate"], "status": status, "role": source["role"]})
            continue
        try:
            checkpoint_digest = _validate_completed_training_artifact(job)
        except ValueError as exc:
            return "failed", {"reason": "completed_training_artifact_invalid", "candidate": job["candidate"], "detail": str(exc)}
        artifacts.append(
            {
                "candidate": job["candidate"],
                "source_role": source["role"],
                "checkpoint_sha256": checkpoint_digest,
            }
        )
    if waiting:
        return "waiting", {"manifests": {role: item.get("status") for role, item in sources.items()}, "jobs": waiting}
    if repair is None:
        if primary.get("status") != "complete":
            return "waiting", {"manifests": {"primary": primary.get("status")}}
    elif (
        repair.get("status") != "complete"
        or primary.get("status") != "failed"
        or any(primary_jobs[name].get("status") != "failed" for name in _power_candidate_names())
    ):
        return "waiting", {
            "manifests": {"primary": primary.get("status"), "repair": repair.get("status")},
            "reason": "waiting_for_terminal_primary_repair_pair",
        }
    return "ready", {
        "manifests": {role: item.get("status") for role, item in sources.items()},
        "artifacts": artifacts,
    }


def _validate_completed_training_artifact(job: Mapping[str, Any]) -> str:
    candidate = str(job["candidate"])
    config = Path(str(job["config_path"])).resolve()
    run_dir = Path(str(job["run_dir"])).resolve()
    checkpoint = (run_dir / "checkpoints/last.pth").resolve()
    if job.get("status") != "done" or int(job.get("return_code", -1)) != 0:
        raise ValueError("training manifest job is not done with return_code=0")
    if not config.is_file() or not run_dir.is_dir() or not checkpoint.is_file():
        raise ValueError("config, run_dir, or last.pth is missing")
    if _sha256(config) != str(job.get("config_sha256")):
        raise ValueError("config SHA256 differs from training manifest")
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        raise ValueError("run_status.json is missing")
    status = _read_json(status_path)
    if (
        status.get("state") != "complete"
        or Path(str(status.get("config_path", ""))).resolve() != config
        or Path(str(status.get("run_dir", ""))).resolve() != run_dir
        or status.get("experiment_name") != candidate
        or int(status.get("seed", -1)) != 1
    ):
        raise ValueError("run_status identity does not match candidate/config/run")
    config_payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    experiment = config_payload.get("experiment", {})
    output = config_payload.get("output", {})
    screen = config_payload.get("mmw_dynamic_router_screen", {})
    if (
        experiment.get("name") != candidate
        or int(experiment.get("seed", -1)) != 1
        or experiment.get("ablation_id") != candidate
        or Path(str(output.get("dir", ""))).resolve() != run_dir.parent
        or screen.get("candidate") != candidate
    ):
        raise ValueError("resolved config experiment/output identity does not match candidate/run")
    sidecar_path = Path(str(checkpoint) + ".json")
    if not sidecar_path.is_file():
        raise ValueError("checkpoint publication sidecar is missing")
    sidecar = _read_json(sidecar_path)
    digest = _sha256(checkpoint)
    if (
        sidecar.get("publish_complete") is not True
        or sidecar.get("checkpoint_role") != "last"
        or int(sidecar.get("checkpoint_schema_version", -1)) != 1
        or sidecar.get("checkpoint_sha256") != digest
        or int(sidecar.get("checkpoint_size_bytes", -1)) != checkpoint.stat().st_size
        or Path(str(sidecar.get("path", ""))).resolve() != checkpoint
        or Path(str(sidecar.get("run_dir", ""))).resolve() != run_dir
    ):
        raise ValueError("checkpoint publication sidecar is incomplete or mismatched")
    return digest


def _candidate_command(job: Mapping[str, Any], request: Mapping[str, Any]) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/launch_mmw_router_joint_stress.py"),
        "--output-root",
        str(job["output_root"]),
        "--cache",
        str(request["cache"]),
        "--config",
        str(job["config"]),
        "--checkpoint",
        str(job["checkpoint"]),
        "--gpus",
        str(job["gpu"]),
        "--allow-gpu0-3",
        "--batch-size",
        "64",
        "--poll-seconds",
        str(request["child_poll_seconds"]),
        "--bootstrap-iterations",
        str(request["bootstrap_iterations"]),
        "--retry-failed",
        "--launch",
    ]
    attempt_id = str(job.get("attempt_id") or "").strip()
    if attempt_id and attempt_id.lower() != "none":
        command.extend(["--orchestration-attempt", attempt_id])
    return command


def _candidate_complete(job: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    root = Path(job["output_root"])
    manifest_path = root / "evaluation_manifest.json"
    summary_path = root / "joint_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return False
    try:
        manifest = _read_json(manifest_path)
        summary = _read_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    eval_request = manifest.get("request", {})
    candidate_source = request.get("candidate_sources", {}).get(str(job.get("candidate")), {})
    # Frozen orchestration jobs normally carry both digests.  Keep the
    # post-hoc/manual-completion path compatible with older job records by
    # deriving missing values from the immutable candidate source; the
    # frozen-request validator still requires the digests before launch.
    expected_config_sha = str(job.get("config_sha256") or candidate_source.get("config_sha256") or "")
    expected_checkpoint_sha = str(
        job.get("checkpoint_sha256") or candidate_source.get("checkpoint_sha256") or ""
    )
    if not expected_config_sha or not expected_checkpoint_sha:
        return False
    cache_path = Path(str(request.get("cache", ""))).resolve()
    if not cache_path.is_file():
        return False
    expected_provenance = _config_router_provenance(Path(job["config"]))
    required_request = {
        "protocol": request.get("joint_stress_protocol"),
        "evaluator_protocol": request.get("joint_stress_protocol"),
        "evaluator_algorithm": request.get("evaluator_algorithm"),
        "branch_algorithm": request.get("branch_algorithm"),
        "fusion_branches": list(request.get("fusion_branches", ())),
        "config": str(Path(job["config"]).resolve()),
        "config_sha256": expected_config_sha,
        "checkpoint": str(Path(job["checkpoint"]).resolve()),
        "checkpoint_sha256": expected_checkpoint_sha,
        "cache": str(cache_path),
        "cache_sha256": request.get("cache_sha256"),
        "cache_checksum": request.get("cache_checksum"),
        "evaluator_sha256": request.get("evaluator_sha256"),
        "oracle_helper_sha256": request.get("oracle_helper_sha256"),
        "corruption_runtime_sha256": request.get("corruption_runtime_sha256"),
        "joint_cache_runtime_sha256": request.get("joint_cache_runtime_sha256"),
        "summary_sha256": request.get("summary_sha256"),
        "claim_eligible": False,
        "split": "frozen_inner_validation_only",
        "router_candidate_provenance": expected_provenance,
        "gpus": [int(job["gpu"])],
        **_joint_stress_request_identity(),
    }
    if any(eval_request.get(key) != value for key, value in required_request.items()):
        return False
    if manifest.get("request_sha256") != _payload_sha256(eval_request):
        return False
    if candidate_source.get("config_sha256") != expected_config_sha or candidate_source.get("checkpoint_sha256") != expected_checkpoint_sha:
        return False
    attempt = _job_attempt_id(job)
    recorded_attempt = str(eval_request.get("orchestration_attempt", ""))
    if attempt and recorded_attempt != attempt:
        return False
    if _sha256(cache_path) != str(request.get("cache_sha256")):
        return False
    try:
        cache_payload = _read_json(cache_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if str(cache_payload.get("checksum")) != str(request.get("cache_checksum")):
        return False
    if (
        manifest.get("status") != "complete"
        or manifest.get("summary", {}).get("status") != "complete"
        or len(manifest.get("jobs", ())) != 8
        or any(job_item.get("status") != "complete" or int(job_item.get("returncode", -1)) != 0 for job_item in manifest["jobs"])
    ):
        return False
    for shard in manifest["jobs"]:
        marker_path = Path(str(shard.get("completion_marker", "")))
        if not marker_path.is_file():
            return False
        try:
            marker = _read_json(marker_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        marker_expected = {
            "status": "complete",
            "protocol": eval_request.get("evaluator_protocol"),
            "shard": shard.get("shard"),
            "config_sha256": eval_request.get("config_sha256"),
            "checkpoint_sha256": eval_request.get("checkpoint_sha256"),
            "cache_sha256": eval_request.get("cache_sha256"),
            "cache_checksum": eval_request.get("cache_checksum"),
            "evaluator_algorithm": eval_request.get("evaluator_algorithm"),
            "request_sha256": manifest.get("request_sha256"),
        }
        if recorded_attempt:
            marker_expected["orchestration_attempt"] = recorded_attempt
        if any(marker.get(key) != value for key, value in marker_expected.items()):
            return False
    provenance = summary.get("provenance", {})
    source_hashes = provenance.get("source_sha256", {})
    return bool(
        provenance.get("evaluation_request_sha256") == manifest.get("request_sha256")
        and provenance.get("checkpoint_sha256") == expected_checkpoint_sha
        and provenance.get("cache_sha256") == request.get("cache_sha256")
        and provenance.get("cache_checksum") == request.get("cache_checksum")
        and provenance.get("branch_algorithm") == request.get("branch_algorithm")
        and provenance.get("evaluator_algorithm") == request.get("evaluator_algorithm")
        and tuple(provenance.get("fusion_branches", ())) == DYNAMIC_FUSIONS
        and provenance.get("claim_eligible") is False
        and source_hashes.get("evaluator_sha256") == request.get("evaluator_sha256")
        and source_hashes.get("oracle_helper_sha256") == request.get("oracle_helper_sha256")
        and source_hashes.get("corruption_runtime_sha256") == request.get("corruption_runtime_sha256")
        and source_hashes.get("joint_cache_runtime_sha256") == request.get("joint_cache_runtime_sha256")
        and source_hashes.get("summary_sha256") == request.get("summary_sha256")
    )


def summarize_candidates(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    gate_thresholds = None
    for job in manifest["jobs"]:
        summary = _read_json(Path(job["output_root"]) / "joint_summary.json")
        provenance = summary["provenance"]
        if tuple(provenance.get("fusion_branches", ())) != DYNAMIC_FUSIONS or provenance.get("claim_eligible") is not False:
            raise ValueError(f"Candidate summary evidence boundary mismatch: {job['candidate']}")
        if gate_thresholds is None:
            gate_thresholds = provenance.get("gate_thresholds")
        elif provenance.get("gate_thresholds") != gate_thresholds:
            raise ValueError("Dynamic Router candidate Gate thresholds differ across summaries.")
        rate_rows = {float(row["requested_stress_rate"]): row for row in summary["rate_summary"]}
        intervals = {
            (str(row["control"]), str(row["metric"])): row
            for row in summary["paired_domain_bootstrap"]
            if row["scope"] == "Joint40_60_80Combined"
        }
        row = {
            "candidate": job["candidate"],
            "router_variant": job["router_variant"],
            "supervision": job["supervision"],
            "gpu": int(job["gpu"]),
            "mechanism_gate_passed": bool(summary["gate"].get("mechanism_gate_passed", False)),
            "clean_adba_minus_static_prior": float(rate_rows[0.0]["learned_minus_train_fit_static_prior_adba"]),
            "clean_adba_minus_frozen_current": float(rate_rows[0.0]["learned_minus_frozen_current_router_adba"]),
            "joint_corrupt_weight_vs_static_ratio": _mean(
                float(rate_rows[rate]["corrupted_cell_weight_vs_static_ratio"]) for rate in (0.4, 0.6, 0.8)
            ),
            "joint_corrupt_downweight_vs_static_rate": _mean(
                float(rate_rows[rate]["corrupted_cell_downweight_vs_static_rate"])
                for rate in (0.4, 0.6, 0.8)
            ),
            "router_residual_abs_mean": _mean(
                float(rate_rows[rate]["router_residual_abs_mean"]) for rate in (0.4, 0.6, 0.8)
            ),
            "joint_adba_minus_post_health_static_prior": _mean(
                float(rate_rows[rate]["learned_minus_post_health_static_prior_adba"])
                for rate in (0.4, 0.6, 0.8)
            ),
            "joint_normalized_gain_minus_post_health_static_prior": _mean(
                float(rate_rows[rate]["learned_minus_post_health_static_prior_normalized_gain"])
                for rate in (0.4, 0.6, 0.8)
            ),
            "claim_eligible": False,
        }
        for control, label in (
            ("train_fit_static_prior", "static_prior"),
            ("frozen_current_router", "frozen_current"),
        ):
            for metric in ("adba", "normalized_gain"):
                interval = intervals[(control, metric)]
                row[f"combined_{metric}_minus_{label}"] = float(interval["mean_delta"])
                row[f"combined_{metric}_minus_{label}_ci_low"] = float(interval["ci_low"])
                row[f"combined_{metric}_minus_{label}_ci_high"] = float(interval["ci_high"])
        rows.append(row)
    _write_csv(root / "dynamic_router_candidate_summary.csv", rows)
    shortlist = [str(row["candidate"]) for row in rows if row["mechanism_gate_passed"]]
    payload = {
        "protocol": PROTOCOL_ID,
        "claim_eligible": False,
        "candidate_count": len(rows),
        "gate_thresholds": gate_thresholds,
        "shortlist_pending_pure_drop_and_multiseed": shortlist,
        "decision": "mechanism_candidates_require_pure_drop_and_multiseed" if shortlist else "no_seed1_candidate_passed",
        "rows": rows,
    }
    _write_json(root / "dynamic_router_screen_summary.json", payload)
    (root / "README.md").write_text(_summary_markdown(payload), encoding="utf-8")
    return {
        "path": str((root / "dynamic_router_screen_summary.json").resolve()),
        "status": "complete",
        "decision": payload["decision"],
        "claim_eligible": False,
    }


def _validate_training_manifest(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    return _validate_manifest_jobs(payload, path, candidates=CANDIDATES, require_numeric_policy=False)


def _merge_training_jobs(
    *,
    primary: Mapping[str, Any],
    primary_path: Path,
    primary_jobs: list[dict[str, Any]],
    repair: Mapping[str, Any] | None,
    repair_path: Path | None,
) -> list[dict[str, Any]]:
    if repair is None:
        result = [dict(job) for job in primary_jobs]
        for job in result:
            job.update(
                _source_role="primary",
                _source_manifest=str(primary_path),
                _source_request_sha256=str(primary["request_sha256"]),
            )
        return result
    if repair_path is None:
        raise ValueError("Repair manifest payload requires an explicit path.")
    repair_candidates = tuple(item for item in CANDIDATES if item[2] == "beam_power")
    repair_jobs = _validate_manifest_jobs(
        repair,
        repair_path,
        candidates=repair_candidates,
        require_numeric_policy=True,
    )
    _validate_repair_request_pair(
        primary,
        repair,
        primary_jobs=primary_jobs,
        repair_jobs=repair_jobs,
    )
    primary_by_name = {str(job["candidate"]): dict(job) for job in primary_jobs}
    repair_by_name = {str(job["candidate"]): dict(job) for job in repair_jobs}
    if set(repair_by_name) != set(_power_candidate_names()):
        raise ValueError("Repair manifest may contain only the four same-name Power candidates.")
    result = []
    for candidate, _, supervision in CANDIDATES:
        source_role = "repair" if supervision == "beam_power" else "primary"
        source = repair_by_name[candidate] if source_role == "repair" else primary_by_name[candidate]
        source.update(
            _source_role=source_role,
            _source_manifest=str(repair_path if source_role == "repair" else primary_path),
            _source_request_sha256=str(
                repair["request_sha256"] if source_role == "repair" else primary["request_sha256"]
            ),
        )
        result.append(source)
    return result


def _validate_manifest_jobs(
    payload: Mapping[str, Any],
    path: Path,
    *,
    candidates: tuple[tuple[str, str, str], ...],
    require_numeric_policy: bool,
) -> list[dict[str, Any]]:
    request = payload.get("request", {})
    jobs = payload.get("jobs", ())
    expected = {
        name: (variant, supervision, next(index for index, item in enumerate(CANDIDATES) if item[0] == name))
        for name, variant, supervision in candidates
    }
    _validate_training_request(request, path, candidates=candidates, require_numeric_policy=require_numeric_policy)
    panel_path = Path(str(payload.get("panel_path", ""))).resolve()
    try:
        panel = load_router_joint_training_panel(panel_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid seed1 dynamic Router training panel: {panel_path}") from exc
    if (
        payload.get("protocol") != TRAINING_PROTOCOL_ID
        or payload.get("request_sha256") != _payload_sha256(request)
        or len(jobs) != len(expected)
        or panel.get("protocol_id") != request.get("panel_protocol")
        or int(panel.get("seed", -1)) != int(request.get("panel_seed", -2))
        or panel.get("checksum") != request.get("panel_checksum")
    ):
        raise ValueError(f"Invalid seed1 dynamic Router training manifest: {path}")
    result = []
    for job in jobs:
        candidate = str(job.get("candidate", ""))
        if candidate not in expected:
            raise ValueError(f"Unknown candidate in dynamic Router training manifest: {candidate}")
        variant, supervision, gpu = expected[candidate]
        config_path = Path(str(job.get("config_path", "")))
        if (
            str(job.get("router_variant")) != variant
            or str(job.get("supervision")) != supervision
            or int(job.get("gpu", -1)) != gpu
            or int(job.get("seed", -1)) != 1
            or job.get("claim_eligible") is not False
            or not config_path.is_file()
            or _sha256(config_path) != job.get("config_sha256")
        ):
            raise ValueError(f"Candidate identity mismatch in dynamic Router training manifest: {candidate}")
        _validate_candidate_config(
            config_path,
            candidate=candidate,
            variant=variant,
            supervision=supervision,
            request=request,
            panel_path=panel_path,
            run_dir=Path(str(job.get("run_dir", ""))).resolve(),
            require_numeric_policy=require_numeric_policy,
        )
        result.append(dict(job))
    result.sort(key=lambda item: int(item["gpu"]))
    return result


def _validate_training_request(
    request: Mapping[str, Any],
    path: Path,
    *,
    candidates: tuple[tuple[str, str, str], ...],
    require_numeric_policy: bool,
) -> None:
    source_config = Path(str(request.get("source_config", "")))
    source_checkpoint = Path(str(request.get("source_checkpoint", "")))
    expected_gpus = [next(index for index, item in enumerate(CANDIDATES) if item[0] == name) for name, _, _ in candidates]
    if (
        request.get("protocol") != TRAINING_PROTOCOL_ID
        or request.get("panel_protocol") != ROUTER_PANEL_PROTOCOL
        or int(request.get("panel_seed", -1)) != ROUTER_PANEL_SEED
        or int(request.get("seed", -1)) != 1
        or int(request.get("batch_size", -1)) != 64
        or int(request.get("epochs", -1)) != 10
        or request.get("selection_split") != "frozen_inner_validation_only"
        or request.get("claim_eligible") is not False
        or request.get("gpus") != expected_gpus
        or request.get("candidates") != [list(item) for item in candidates]
        or not source_config.is_file()
        or _sha256(source_config) != request.get("source_config_sha256")
        or not source_checkpoint.is_file()
        or _sha256(source_checkpoint) != request.get("source_checkpoint_sha256")
    ):
        raise ValueError(f"Invalid frozen training request identity: {path}")
    policy = request.get("utility_numeric_policy")
    source_sha = request.get("router_reliability_source_sha256")
    if require_numeric_policy:
        if policy != UTILITY_NUMERIC_POLICY or source_sha != _sha256(ROUTER_RELIABILITY_SOURCE):
            raise ValueError(f"Repair manifest numeric policy/source identity mismatch: {path}")
    elif (policy is None) != (source_sha is None):
        raise ValueError(f"Primary manifest has partial numeric policy provenance: {path}")
    elif policy is not None and (
        policy != UTILITY_NUMERIC_POLICY or source_sha != _sha256(ROUTER_RELIABILITY_SOURCE)
    ):
        raise ValueError(f"Primary manifest numeric policy/source identity mismatch: {path}")


def _validate_candidate_config(
    path: Path,
    *,
    candidate: str,
    variant: str,
    supervision: str,
    request: Mapping[str, Any],
    panel_path: Path,
    run_dir: Path,
    require_numeric_policy: bool,
) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    screen = config.get("mmw_dynamic_router_screen", {})
    primary = config.get("model", {}).get("primary", {})
    dynamic = config.get("loss", {}).get("u_mask_beam_jepa", {}).get("dynamic_router", {})
    paired_joint = dynamic.get("paired_joint", {})
    initialization = config.get("training", {}).get("initialization_checkpoint", {})
    experiment = config.get("experiment", {})
    output = config.get("output", {})
    if (
        screen.get("protocol") != TRAINING_PROTOCOL_ID
        or screen.get("candidate") != candidate
        or screen.get("router_variant") != variant
        or screen.get("supervision") != supervision
        or int(screen.get("seed", -1)) != 1
        or screen.get("selection_split") != "frozen_inner_validation_only"
        or screen.get("claim_eligible") is not False
        or screen.get("source_checkpoint_sha256") != request.get("source_checkpoint_sha256")
        or screen.get("joint_panel_checksum") != request.get("panel_checksum")
        or primary.get("router_variant") != variant
        or dynamic.get("supervision") != supervision
        or paired_joint.get("panel_sha256") != request.get("panel_checksum")
        or Path(str(paired_joint.get("panel_path", ""))).resolve() != panel_path
        or initialization.get("sha256") != request.get("source_checkpoint_sha256")
        or Path(str(initialization.get("path", ""))).resolve()
        != Path(str(request.get("source_checkpoint", ""))).resolve()
        or experiment.get("name") != candidate
        or int(experiment.get("seed", -1)) != 1
        or experiment.get("ablation_id") != candidate
        or Path(str(output.get("dir", ""))).resolve() != run_dir.parent
        or output.get("run_name") != "seed1"
    ):
        raise ValueError(f"Dynamic Router candidate config identity mismatch: {path}")
    if require_numeric_policy and (
        screen.get("utility_numeric_policy") != UTILITY_NUMERIC_POLICY
        or screen.get("router_reliability_source_sha256") != request.get("router_reliability_source_sha256")
    ):
        raise ValueError(f"Power repair config numeric policy/source identity mismatch: {path}")


def _validate_repair_request_pair(
    primary: Mapping[str, Any],
    repair: Mapping[str, Any],
    *,
    primary_jobs: list[dict[str, Any]],
    repair_jobs: list[dict[str, Any]],
) -> None:
    left = primary["request"]
    right = repair["request"]
    expected_candidates = [
        item for item in left.get("candidates", ()) if len(item) == 3 and item[2] == "beam_power"
    ]
    expected_gpus = [
        gpu
        for gpu, item in zip(left.get("gpus", ()), left.get("candidates", ()), strict=True)
        if len(item) == 3 and item[2] == "beam_power"
    ]
    if right.get("candidates") != expected_candidates or right.get("gpus") != expected_gpus:
        raise ValueError("Power repair request does not preserve the primary Power candidate/GPU mapping.")

    # Lock every request field except the two deliberate repair-provenance
    # additions and the Power-only inventory.  This fails closed if a future
    # launcher silently changes any recipe field.
    normalized_left = dict(left)
    normalized_right = dict(right)
    for payload in (normalized_left, normalized_right):
        for key in (
            "candidates",
            "gpus",
            "router_reliability_source_sha256",
            "utility_numeric_policy",
        ):
            payload.pop(key, None)
    if normalized_left != normalized_right:
        raise ValueError("Primary and repair training requests do not share the frozen experiment identity.")

    primary_panel = Path(str(primary.get("panel_path", ""))).resolve()
    repair_panel = Path(str(repair.get("panel_path", ""))).resolve()
    if _sha256(primary_panel) != _sha256(repair_panel):
        raise ValueError("Primary and repair training panels are not byte-identical.")

    primary_by_name = {str(job["candidate"]): job for job in primary_jobs}
    for repair_job in repair_jobs:
        candidate = str(repair_job["candidate"])
        primary_job = primary_by_name.get(candidate)
        if primary_job is None:
            raise ValueError(f"Repair candidate has no same-name primary recipe: {candidate}")
        primary_recipe = _normalized_repair_recipe(Path(str(primary_job["config_path"])))
        repair_recipe = _normalized_repair_recipe(Path(str(repair_job["config_path"])))
        if primary_recipe != repair_recipe:
            raise ValueError(f"Power repair recipe differs outside the narrow numeric-fix allowlist: {candidate}")


def _normalized_repair_recipe(path: Path) -> dict[str, Any]:
    payload = deepcopy(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    paired = (
        payload.get("loss", {})
        .get("u_mask_beam_jepa", {})
        .get("dynamic_router", {})
        .get("paired_joint", {})
    )
    if isinstance(paired, dict):
        paired.pop("panel_path", None)
    output = payload.get("output", {})
    if isinstance(output, dict):
        output.pop("dir", None)
    screen = payload.get("mmw_dynamic_router_screen", {})
    if isinstance(screen, dict):
        screen.pop("router_reliability_source_sha256", None)
        screen.pop("utility_numeric_policy", None)
    return payload


def _power_candidate_names() -> tuple[str, ...]:
    return tuple(name for name, _, supervision in CANDIDATES if supervision == "beam_power")


def _validate_orchestration_jobs(
    manifest: Mapping[str, Any],
    training_jobs: list[dict[str, Any]],
    output_root: Path,
) -> None:
    jobs = manifest.get("jobs", ())
    if len(jobs) != 8:
        raise ValueError("Dynamic Router evaluation orchestration requires exactly eight candidates.")
    expected = {str(job["candidate"]): job for job in training_jobs}
    if {str(job.get("candidate")) for job in jobs} != set(expected):
        raise ValueError("Dynamic Router evaluation candidate inventory mismatch.")
    if {int(job["gpu"]) for job in jobs} != set(range(8)):
        raise ValueError("Dynamic Router evaluation requires a one-to-one GPU0--7 mapping.")
    for job in jobs:
        training = expected[str(job["candidate"])]
        if (
            int(job["gpu"]) != int(training["gpu"])
            or Path(job["config"]).resolve() != Path(training["config_path"]).resolve()
            or Path(job["checkpoint"]).resolve()
            != (Path(training["run_dir"]) / "checkpoints/last.pth").resolve()
            or Path(job["output_root"]).parent.resolve() != output_root.resolve()
            or job.get("training_source", {}).get("role") != training.get("_source_role")
            or job.get("training_source", {}).get("manifest") != training.get("_source_manifest")
            or job.get("training_source", {}).get("request_sha256")
            != training.get("_source_request_sha256")
            or job.get("config_sha256") != training.get("config_sha256")
        ):
            raise ValueError(f"Dynamic Router evaluation job mapping mismatch: {job['candidate']}")


def _validate_frozen_request(
    manifest: Mapping[str, Any],
    provisional: Mapping[str, Any],
    training_jobs: list[dict[str, Any]],
    output_root: Path,
) -> None:
    request = manifest.get("request", {})
    if (
        manifest.get("request_sha256") != _payload_sha256(request)
        or request.get("identity_state") != "frozen_training"
        or request.get("joint_launcher_sha256") != provisional.get("joint_launcher_sha256")
        or request.get("evaluator_sha256") != provisional.get("evaluator_sha256")
        or request.get("summary_sha256") != provisional.get("summary_sha256")
        or request.get("branch_algorithm") != provisional.get("branch_algorithm")
        or request.get("evaluator_algorithm") != provisional.get("evaluator_algorithm")
        or request.get("fusion_branches") != provisional.get("fusion_branches")
        or request.get("training_manifests") != provisional.get("training_manifests")
        or request.get("candidate_sources", {})
        != {
            str(job["candidate"]): {
                "role": str(job["_source_role"]),
                "manifest": str(job["_source_manifest"]),
                "request_sha256": str(job["_source_request_sha256"]),
                "config_sha256": str(job["config_sha256"]),
                "checkpoint_sha256": request.get("candidate_sources", {})
                .get(str(job["candidate"]), {})
                .get("checkpoint_sha256"),
            }
            for job in training_jobs
        }
    ):
        raise ValueError("Frozen dynamic evaluation request identity differs from current training sources.")
    _validate_orchestration_jobs(manifest, training_jobs, output_root)
    for job in manifest["jobs"]:
        if not job.get("checkpoint_sha256") or job.get("config_sha256") != _sha256(Path(job["config"])):
            raise ValueError(f"Frozen candidate identity is incomplete: {job['candidate']}")


def _summary_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Dynamic Router seed1 inner-only 汇总",
        "",
        "该表仅用于候选筛选，`claim_eligible=false`；Joint Gate 通过后仍需纯 Drop 保护检查和多 seed 确认。",
        "",
        "| Candidate | Gate | Δ Static ADBA | CI low | Δ Static gain | CI low | Δ Post-health static ADBA | Δ Post-health static gain | Weight/static | Downweight rate | abs(residual) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['candidate']} | {'PASS' if row['mechanism_gate_passed'] else 'FAIL'} | "
            f"{row['combined_adba_minus_static_prior']:+.4f} | "
            f"{row['combined_adba_minus_static_prior_ci_low']:+.4f} | "
            f"{row['combined_normalized_gain_minus_static_prior']:+.4f} | "
            f"{row['combined_normalized_gain_minus_static_prior_ci_low']:+.4f} | "
            f"{row['joint_adba_minus_post_health_static_prior']:+.4f} | "
            f"{row['joint_normalized_gain_minus_post_health_static_prior']:+.4f} | "
            f"{row['joint_corrupt_weight_vs_static_ratio']:.4f} | "
            f"{row['joint_corrupt_downweight_vs_static_rate']:.4f} | "
            f"{row['router_residual_abs_mean']:.4f} |"
        )
    lines.extend(["", f"Decision: `{payload['decision']}`", ""])
    lines.extend(
        [
            f"Gate thresholds: `{json.dumps(payload['gate_thresholds'], ensure_ascii=False, sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(lines)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _mean(values) -> float:
    items = list(values)
    if not items:
        raise ValueError("Cannot average an empty collection.")
    return float(sum(items) / len(items))


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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
