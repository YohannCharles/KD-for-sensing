#!/usr/bin/env python3
"""Plan and run the fixed MMW Router joint Drop+Corrupt screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from kd_sensing.data.mmw.twc_router_joint_stress import (
    BALANCE_POLICY,
    CORRUPTION_SEVERITY,
    GENERATOR,
    JOINT_RATES,
    MASKS_PER_RATE,
    MASK_SEED,
    PROTOCOL_ID,
    prepare_router_joint_stress_cache,
)
from kd_sensing.evaluation.corruptions import CORRUPTION_PARAMETERS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "outputs/cache/mmw_router_joint_stress_v1/fixed_state_cache.json"
DEFAULT_OUTPUT = ROOT / "outputs/mmw_router_joint_stress_v1"
DEFAULT_CONFIG = ROOT / "outputs/mmw_router_expected_utility_screen_v3/generated_configs/CurrentControl_seed1.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs/mmw_router_expected_utility_screen_v3/CurrentControl/seed1/checkpoints/last.pth"
DEFAULT_GPUS = (4, 5, 6, 7)
SHARD_WIDTH = MASKS_PER_RATE // 2
CORRUPTION_SEED = 20260718
EVALUATOR_PROTOCOL = PROTOCOL_ID
EVALUATOR_ALGORITHM = "paired_cell_selective_s2_then_temporal_drop_reference_controls_v2"
LEGACY_FUSIONS = ("uniform", "learned", "oracle")
DYNAMIC_FUSIONS = (
    *LEGACY_FUSIONS,
    "train_fit_static_prior",
    "frozen_current_router",
    "post_health_uniform",
    "post_health_static_prior",
)
BRANCH_ALGORITHM = "availability_normalized_reference_controls_candidate_health_oracle_v3"
DYNAMIC_SCREEN_KEYS = (
    "mmw_dynamic_router_screen",
    "mmw_dynamic_router_decision_screen",
    "mmw_h2r_simplification_screen",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--gpus", default=",".join(str(item) for item in DEFAULT_GPUS))
    parser.add_argument(
        "--allow-gpu0-3",
        action="store_true",
        help="Allow GPU0--3 only after explicit user authorization.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument(
        "--orchestration-attempt",
        default="",
        help="Optional immutable outer-watcher attempt identity.",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    try:
        gpus = _parse_gpus(args.gpus, allow_gpu0_3=bool(args.allow_gpu0_3))
    except ValueError as exc:
        parser.error(str(exc))
    if int(args.batch_size) != 64:
        parser.error("The fixed joint-stress screen requires --batch-size 64.")
    if float(args.poll_seconds) <= 0:
        parser.error("--poll-seconds must be positive.")
    if int(args.bootstrap_iterations) <= 0:
        parser.error("--bootstrap-iterations must be positive.")
    if args.retry_failed and not args.launch:
        parser.error("--retry-failed requires --launch.")

    output = Path(args.output_root).resolve()
    cache_path = Path(args.cache).resolve()
    config = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    _require_file(config, "config")
    _require_file(checkpoint, "checkpoint")
    cache = prepare_router_joint_stress_cache(cache_path)
    manifest_path = prepare_manifest(
        output,
        cache_path=cache_path,
        cache=cache,
        config=config,
        checkpoint=checkpoint,
        gpus=gpus,
        batch_size=int(args.batch_size),
        orchestration_attempt=str(args.orchestration_attempt).strip(),
    )
    if not args.launch:
        manifest = _read_json(manifest_path)
        print(
            json.dumps(
                {
                    "status": "planned",
                    "manifest": str(manifest_path),
                    "jobs": len(manifest["jobs"]),
                    "conditions": len(cache["conditions"]),
                    "gpus": list(gpus),
                },
                indent=2,
            )
        )
        return 0
    return run_manifest(
        manifest_path,
        poll_seconds=float(args.poll_seconds),
        retry_failed=bool(args.retry_failed),
        bootstrap_iterations=int(args.bootstrap_iterations),
    )


def prepare_manifest(
    output: Path,
    *,
    cache_path: Path,
    cache: Mapping[str, Any],
    config: Path,
    checkpoint: Path,
    gpus: tuple[int, ...],
    batch_size: int,
    orchestration_attempt: str = "",
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    fusion_branches = _config_fusion_branches(config)
    router_provenance = _config_router_provenance(config)
    request = {
        "protocol": PROTOCOL_ID,
        "evaluator_protocol": EVALUATOR_PROTOCOL,
        "evaluator_algorithm": EVALUATOR_ALGORITHM,
        "branch_algorithm": BRANCH_ALGORITHM,
        "fusion_branches": list(fusion_branches),
        "router_candidate_provenance": router_provenance,
        "evaluator_sha256": _sha256(ROOT / "scripts/eval_mmw_router_joint_stress.py"),
        "oracle_helper_sha256": _sha256(ROOT / "scripts/eval_mmw_router_oracle_gap.py"),
        "corruption_runtime_sha256": _sha256(ROOT / "src/kd_sensing/evaluation/corruptions.py"),
        "joint_cache_runtime_sha256": _sha256(
            ROOT / "src/kd_sensing/data/mmw/twc_router_joint_stress.py"
        ),
        "summary_sha256": _sha256(ROOT / "scripts/summarize_mmw_router_joint_stress.py"),
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "cache": str(cache_path),
        "cache_sha256": _sha256(cache_path),
        "cache_checksum": str(cache["checksum"]),
        "mask_seed": MASK_SEED,
        "mask_generator": GENERATOR,
        "mask_balance_policy": BALANCE_POLICY,
        "corruption_seed": CORRUPTION_SEED,
        "corruption_severity": CORRUPTION_SEVERITY,
        "corruption_parameters": {
            name: {
                "unit": CORRUPTION_PARAMETERS[name]["unit"],
                "value": CORRUPTION_PARAMETERS[name]["values"][CORRUPTION_SEVERITY - 1],
            }
            for name in ("image_occlusion", "radar_noise", "lidar_sparsify", "gps_noise")
        },
        "condition_count": len(cache["conditions"]),
        "joint_rates": [float(value) for value in JOINT_RATES],
        "masks_per_rate": MASKS_PER_RATE,
        "gpus": list(gpus),
        "batch_size": int(batch_size),
        "orchestration_attempt": str(orchestration_attempt),
        "split": "frozen_inner_validation_only",
        "claim_eligible": False,
    }
    request_sha256 = _payload_sha256(request)
    path = output / "evaluation_manifest.json"
    if path.is_file():
        manifest = _read_json(path)
        if manifest.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing joint-stress manifest differs from the frozen request: {path}")
        _validate_jobs(manifest, output)
        return path

    jobs: list[dict[str, Any]] = []
    job_index = 0
    gpu_assignment = ((0,) + tuple(item for item in gpus if item != 0)) if 0 in gpus else gpus
    for rate in JOINT_RATES:
        rate_percent = int(round(100 * float(rate)))
        for mask_start in range(0, MASKS_PER_RATE, SHARD_WIDTH):
            mask_end = min(mask_start + SHARD_WIDTH, MASKS_PER_RATE)
            include_clean = job_index == 0
            shard_name = _shard_name(rate_percent, mask_start, mask_end, include_clean)
            jobs.append(
                {
                    "job_index": job_index,
                    "rate": rate_percent,
                    "mask_start": mask_start,
                    "mask_end": mask_end,
                    "include_clean": include_clean,
                    "gpu": int(gpu_assignment[job_index % len(gpu_assignment)]),
                    "shard": shard_name,
                    "completion_marker": str((output / "shards" / shard_name / "complete.json").resolve()),
                    "log": str((output / "logs" / f"{shard_name}.log").resolve()),
                    "status": "planned",
                    "pid": None,
                    "returncode": None,
                    "attempts": 0,
                }
            )
            job_index += 1
    manifest = {
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha256,
        "created_at": _now(),
        "status": "planned",
        "jobs": jobs,
    }
    _validate_jobs(manifest, output)
    _write_json(path, manifest)
    return path


def run_manifest(
    path: Path,
    *,
    poll_seconds: float,
    retry_failed: bool,
    bootstrap_iterations: int,
) -> int:
    manifest = _read_json(path)
    output = path.parent
    _validate_jobs(manifest, output)
    request = dict(manifest["request"])
    request["_request_sha256"] = str(manifest["request_sha256"])
    if retry_failed:
        for job in manifest["jobs"]:
            if job.get("status") == "failed" and not _job_complete(job, request):
                job.update(status="planned", pid=None, returncode=None, retry_requested_at=_now())
    _reconcile_manifest(manifest, request)
    failed = [job for job in manifest["jobs"] if job.get("status") == "failed"]
    if failed:
        _write_json(path, manifest)
        print(f"{len(failed)} failed shard(s); rerun with --retry-failed after inspecting logs.", file=sys.stderr)
        return 1

    children: dict[int, tuple[subprocess.Popen[Any], Any, int]] = {}
    while True:
        changed = False
        for gpu, (process, handle, job_index) in list(children.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            job = manifest["jobs"][job_index]
            complete = returncode == 0 and _job_complete(job, request)
            job.update(
                status="complete" if complete else "failed",
                returncode=int(returncode),
                ended_at=_now(),
            )
            if returncode == 0 and not complete:
                job["failure_reason"] = "Evaluator exited successfully without the immutable shard completion marker."
            children.pop(gpu)
            changed = True

        # A previous launcher may have died while its evaluator stayed alive.
        # Preserve that process and GPU; only reconcile it after the PID exits.
        for job in manifest["jobs"]:
            if job.get("status") != "running" or int(job["gpu"]) in children:
                continue
            pid = int(job.get("pid") or 0)
            if _pid_alive(pid):
                continue
            if _job_complete(job, request):
                job.update(status="complete", returncode=0, ended_at=_now())
            else:
                job.update(
                    status="failed",
                    returncode=job.get("returncode"),
                    ended_at=_now(),
                    failure_reason="Detached evaluator PID exited without a completion marker.",
                )
            changed = True

        occupied_gpus = {
            int(job["gpu"])
            for job in manifest["jobs"]
            if job.get("status") == "running" and _pid_alive(int(job.get("pid") or 0))
        }
        for job_index, job in enumerate(manifest["jobs"]):
            gpu = int(job["gpu"])
            if job.get("status") != "planned" or gpu in children or gpu in occupied_gpus:
                continue
            if _job_complete(job, request):
                job.update(status="complete", returncode=0, ended_at=_now())
                changed = True
                continue
            command = _eval_command(job, request, output)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            handle = Path(job["log"]).open("a", encoding="utf-8")
            handle.write(f"\n[{_now()}] launch: {' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            job.update(
                status="running",
                pid=process.pid,
                returncode=None,
                started_at=_now(),
                command=command,
                attempts=int(job.get("attempts", 0)) + 1,
            )
            children[gpu] = (process, handle, job_index)
            occupied_gpus.add(gpu)
            changed = True
        if changed:
            manifest["status"] = "running"
            _write_json(path, manifest)

        statuses = {str(job.get("status")) for job in manifest["jobs"]}
        if "failed" in statuses:
            # Let already-running evaluators finish; never terminate unrelated or
            # successfully progressing work because one shard failed.
            if not children and not any(
                job.get("status") == "running" and _pid_alive(int(job.get("pid") or 0))
                for job in manifest["jobs"]
            ):
                manifest.update(status="failed", completed_at=_now())
                _write_json(path, manifest)
                return 1
        elif statuses == {"complete"}:
            break
        time.sleep(poll_seconds)

    manifest.update(status="complete", completed_at=_now())
    _write_json(path, manifest)
    command = [
        sys.executable,
        str(ROOT / "scripts/summarize_mmw_router_joint_stress.py"),
        "--root",
        str(output),
        "--cache",
        request["cache"],
        "--bootstrap-iterations",
        str(int(bootstrap_iterations)),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    manifest = _read_json(path)
    manifest["summary"] = {
        "status": "complete" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "command": command,
        "ended_at": _now(),
    }
    if result.returncode != 0:
        manifest["status"] = "failed_summary"
    _write_json(path, manifest)
    return int(result.returncode)


def _eval_command(job: Mapping[str, Any], request: Mapping[str, Any], output: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/eval_mmw_router_joint_stress.py"),
        "--rate",
        str(job["rate"]),
        "--mask-start",
        str(job["mask_start"]),
        "--mask-end",
        str(job["mask_end"]),
        "--cache",
        request["cache"],
        "--config",
        request["config"],
        "--checkpoint",
        request["checkpoint"],
        "--output-root",
        str(output),
        "--batch-size",
        str(request["batch_size"]),
        "--request-sha256",
        str(request["_request_sha256"]),
    ]
    if bool(job["include_clean"]):
        command.append("--include-clean")
    attempt = str(request.get("orchestration_attempt") or "").strip()
    if attempt and attempt.lower() != "none":
        command.extend(["--orchestration-attempt", attempt])
    return command


def _reconcile_manifest(manifest: dict[str, Any], request: Mapping[str, Any]) -> None:
    for job in manifest["jobs"]:
        if _job_complete(job, request):
            job.update(status="complete", returncode=0)
        elif job.get("status") == "complete":
            job.update(
                status="failed",
                failure_reason="Recorded complete shard is missing its completion marker.",
            )
        elif job.get("status") == "running" and not _pid_alive(int(job.get("pid") or 0)):
            job.update(
                status="failed",
                failure_reason="Recorded evaluator PID is not alive and shard is incomplete.",
            )


def _validate_jobs(manifest: Mapping[str, Any], output: Path) -> None:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 8:
        raise ValueError("Joint-stress manifest must contain exactly eight shards.")
    seen: set[tuple[int, int, int]] = set()
    clean_count = 0
    for job in jobs:
        key = (int(job["rate"]), int(job["mask_start"]), int(job["mask_end"]))
        if key in seen:
            raise ValueError(f"Duplicate joint-stress shard: {key}")
        seen.add(key)
        clean_count += int(bool(job["include_clean"]))
        expected_marker = output / "shards" / str(job["shard"]) / "complete.json"
        if Path(str(job["completion_marker"])).resolve() != expected_marker.resolve():
            raise ValueError(f"Joint-stress shard completion path mismatch: {key}")
    expected = {
        (int(round(rate * 100)), start, min(start + SHARD_WIDTH, MASKS_PER_RATE))
        for rate in JOINT_RATES
        for start in range(0, MASKS_PER_RATE, SHARD_WIDTH)
    }
    if seen != expected or clean_count != 1 or not bool(jobs[0]["include_clean"]):
        raise ValueError("Joint-stress shard inventory or clean ownership mismatch.")


def _job_complete(job: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    marker = Path(str(job["completion_marker"]))
    if not marker.is_file():
        return False
    try:
        payload = _read_json(marker)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = {
        "status": "complete",
        "protocol": request["evaluator_protocol"],
        "shard": job["shard"],
        "config_sha256": request["config_sha256"],
        "checkpoint_sha256": request["checkpoint_sha256"],
        "cache_sha256": request["cache_sha256"],
        "cache_checksum": request["cache_checksum"],
        "corruption_seed": request["corruption_seed"],
        "evaluator_algorithm": request["evaluator_algorithm"],
        "request_sha256": request["_request_sha256"],
    }
    attempt = str(request.get("orchestration_attempt") or "").strip()
    if attempt and attempt.lower() != "none":
        expected["orchestration_attempt"] = attempt
    return all(payload.get(key) == value for key, value in expected.items())


def _parse_gpus(value: str, *, allow_gpu0_3: bool) -> tuple[int, ...]:
    try:
        gpus = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--gpus must be a comma-separated list of GPU ids.") from exc
    if not gpus or len(set(gpus)) != len(gpus) or any(item < 0 or item > 7 for item in gpus):
        raise ValueError("--gpus must contain unique ids in [0, 7].")
    if any(item < 4 for item in gpus) and not allow_gpu0_3:
        raise ValueError("GPU0--3 require explicit --allow-gpu0-3 authorization.")
    return gpus


def _shard_name(rate: int, start: int, end: int, include_clean: bool) -> str:
    suffix = "_clean" if include_clean else ""
    return f"rate{rate:02d}_masks{start:02d}_{end:02d}{suffix}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _config_fusion_branches(config: Path) -> tuple[str, ...]:
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    dynamic = next((payload.get(key) for key in DYNAMIC_SCREEN_KEYS if payload.get(key)), None)
    if dynamic:
        if not isinstance(dynamic, Mapping) or dynamic.get("claim_eligible") is not False:
            raise ValueError("Dynamic Router joint stress requires an inner-only candidate config.")
        return DYNAMIC_FUSIONS
    return LEGACY_FUSIONS


def _config_router_provenance(config: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    dynamic = next((payload.get(key) for key in DYNAMIC_SCREEN_KEYS if payload.get(key)), None)
    if not dynamic:
        return {"router_family": "current_control"}
    if not isinstance(dynamic, Mapping):
        raise ValueError("Dynamic Router screen provenance must be a mapping.")
    return {
        "router_family": "dynamic_candidate",
        "candidate": dynamic.get("candidate"),
        "supervision": dynamic.get("supervision"),
        "fused_decision_objective": dynamic.get("fused_decision_objective"),
        "utility_numeric_policy": dynamic.get("utility_numeric_policy"),
        "router_reliability_source_sha256": dynamic.get("router_reliability_source_sha256"),
        "claim_eligible": dynamic.get("claim_eligible"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
