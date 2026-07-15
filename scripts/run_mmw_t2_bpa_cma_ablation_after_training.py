#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("T2-NoBPA", "T2-BPA2CMA", "T2-Linear", "T2-CLS", "T2-CLS-CMA")
SEEDS = (1, 2, 3)
REFERENCE_METHOD = "T2"
EXPECTED_DOMAINS = 15
FAILED_STATES = {"failed", "error", "aborted", "cancelled", "canceled"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Wait for the MMW T2 ablation training matrix, then extract paired task outputs and summarize."
    )
    parser.add_argument("--training-root", default="outputs/mmw_t2_bpa_cma_ablation_v1")
    parser.add_argument("--raw-root", default="outputs/mmw_all_weather_h5p1_seed1_v2/task_output_raw")
    parser.add_argument("--mask-cache", default="outputs/mmw_all_weather_h5p1_eval_masks_v2")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--gpu-grace-seconds", type=float, default=30.0)
    parser.add_argument("--gpus", default=",".join(str(index) for index in range(8)))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.gpu_grace_seconds < 0:
        parser.error("--gpu-grace-seconds must be non-negative")
    try:
        gpus = tuple(int(item.strip()) for item in args.gpus.split(",") if item.strip())
    except ValueError:
        parser.error("--gpus must be a comma-separated list of non-negative integers")
    if not gpus or len(gpus) != len(set(gpus)) or any(gpu < 0 for gpu in gpus):
        parser.error("--gpus must contain unique non-negative integers")

    training_root = _repo_path(args.training_root)
    raw_root = _repo_path(args.raw_root)
    mask_cache = _repo_path(args.mask_cache)
    status_path = training_root / "task_output_orchestrator_status.json"
    try:
        return run_orchestrator(
            training_root,
            raw_root,
            mask_cache,
            gpus=gpus,
            poll_seconds=float(args.poll_seconds),
            gpu_grace_seconds=float(args.gpu_grace_seconds),
            status_path=status_path,
        )
    except KeyboardInterrupt:
        _write_status(status_path, "interrupted", message="Rerun the same command to resume completed task outputs.")
        return 130
    except Exception as exc:  # noqa: BLE001 - preserve unexpected failure evidence for unattended runs.
        _write_status(status_path, "blocked_orchestrator_error", error=f"{type(exc).__name__}: {exc}")
        return 5


def run_orchestrator(
    training_root: Path,
    raw_root: Path,
    mask_cache: Path,
    *,
    gpus: tuple[int, ...],
    poll_seconds: float,
    gpu_grace_seconds: float,
    status_path: Path,
) -> int:
    jobs = tuple((method, seed) for seed in SEEDS for method in VARIANTS)
    while True:
        training = {job_label(*job): _training_state(training_root, *job) for job in jobs}
        failures = {
            label: item
            for label, item in training.items()
            if item["state"] in FAILED_STATES or item["state"] == "invalid"
        }
        if failures:
            _write_status(status_path, "blocked_training_failed", failures=failures, training=training)
            return 2
        missing_checkpoints = {
            label: item
            for label, item in training.items()
            if item["state"] == "complete" and not item["checkpoint_exists"]
        }
        if missing_checkpoints:
            _write_status(
                status_path,
                "blocked_training_checkpoint_missing",
                failures=missing_checkpoints,
                training=training,
            )
            return 2
        ready = [label for label, item in training.items() if item["ready"]]
        if len(ready) == len(jobs):
            break
        _write_status(
            status_path,
            "waiting_for_training",
            ready_count=len(ready),
            total_count=len(jobs),
            training=training,
        )
        time.sleep(poll_seconds)

    missing_configs = [
        str(training_root / "generated_configs" / f"{method}_seed{seed}.yaml")
        for method, seed in jobs
        if not (training_root / "generated_configs" / f"{method}_seed{seed}.yaml").is_file()
    ]
    if missing_configs:
        _write_status(status_path, "blocked_training_config_missing", missing=missing_configs)
        return 2

    reference_states = {
        job_label(REFERENCE_METHOD, seed): _task_output_state(raw_root, REFERENCE_METHOD, seed)
        for seed in SEEDS
    }
    if not all(item["complete"] for item in reference_states.values()):
        _write_status(
            status_path,
            "blocked_reference_task_output_missing",
            reference=reference_states,
            message="The existing T2 seed1-3 task outputs must each contain 15 domains and valid worker provenance.",
        )
        return 3

    output_states = {job_label(*job): _task_output_state(raw_root, *job) for job in jobs}
    corrupt = {label: item for label, item in output_states.items() if item["state"] == "corrupt"}
    if corrupt:
        _write_status(status_path, "blocked_task_output_corrupt", failures=corrupt)
        return 3
    pending = tuple(job for job in jobs if not output_states[job_label(*job)]["complete"])
    completed = [label for label, item in output_states.items() if item["complete"]]
    _write_status(
        status_path,
        "extracting_task_outputs" if pending else "task_outputs_complete",
        completed=completed,
        pending=[job_label(*job) for job in pending],
        gpu_ids=list(gpus),
        raw_root=str(raw_root),
    )
    if pending and gpu_grace_seconds > 0:
        _write_status(
            status_path,
            "waiting_for_gpu_release",
            grace_seconds=gpu_grace_seconds,
            pending=[job_label(*job) for job in pending],
        )
        time.sleep(gpu_grace_seconds)
    failure = _run_extract_jobs(
        pending,
        training_root,
        raw_root,
        mask_cache,
        gpus=gpus,
        log_dir=training_root / "task_output_logs",
        status_path=status_path,
        completed=completed,
    )
    if failure is not None:
        _write_status(status_path, "blocked_task_output_failed", **failure)
        return 3

    final_states = {job_label(*job): _task_output_state(raw_root, *job) for job in jobs}
    incomplete = {label: item for label, item in final_states.items() if not item["complete"]}
    if incomplete:
        _write_status(status_path, "blocked_task_output_incomplete", failures=incomplete)
        return 3

    output_dir = training_root / "task_output_summary"
    _write_status(status_path, "summarizing", raw_root=str(raw_root), output_dir=str(output_dir))
    summary_log = training_root / "task_output_logs" / "summary.log"
    summary_code = _run_summary(raw_root, output_dir, summary_log)
    if summary_code != 0:
        _write_status(
            status_path,
            "blocked_summary_failed",
            return_code=summary_code,
            log=str(summary_log),
        )
        return 4
    _write_status(
        status_path,
        "done",
        task_count=len(jobs),
        reused_reference=[job_label(REFERENCE_METHOD, seed) for seed in SEEDS],
        raw_root=str(raw_root),
        output_dir=str(output_dir),
    )
    return 0


def _run_extract_jobs(
    jobs: tuple[tuple[str, int], ...],
    training_root: Path,
    raw_root: Path,
    mask_cache: Path,
    *,
    gpus: tuple[int, ...],
    log_dir: Path,
    status_path: Path,
    completed: list[str],
) -> dict | None:
    pending = list(jobs)
    running: list[dict] = []
    log_dir.mkdir(parents=True, exist_ok=True)
    while pending or running:
        used_gpus = {item["gpu"] for item in running}
        for gpu in (item for item in gpus if item not in used_gpus):
            if not pending:
                break
            method, seed = pending.pop(0)
            label = job_label(method, seed)
            log_path = log_dir / f"{method}_seed{seed}.log"
            command = _extract_command(training_root, raw_root, mask_cache, method, seed)
            handle = log_path.open("a", encoding="utf-8")
            handle.write(f"\n[{_utc_now()}] GPU{gpu}: {shlex.join(command)}\n")
            handle.flush()
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "2",
                }
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            except OSError as exc:
                handle.close()
                _stop_running(running)
                return {"job": label, "gpu": gpu, "error": f"{type(exc).__name__}: {exc}", "log": str(log_path)}
            running.append(
                {
                    "method": method,
                    "seed": seed,
                    "label": label,
                    "gpu": gpu,
                    "process": process,
                    "handle": handle,
                    "log": log_path,
                }
            )
        _write_status(
            status_path,
            "extracting_task_outputs",
            completed=completed,
            running=[{"job": item["label"], "gpu": item["gpu"], "log": str(item["log"])} for item in running],
            pending=[job_label(*job) for job in pending],
        )
        finished = [item for item in running if item["process"].poll() is not None]
        if not finished:
            time.sleep(1.0)
            continue
        for item in finished:
            running.remove(item)
            return_code = int(item["process"].returncode)
            item["handle"].close()
            if return_code != 0:
                _stop_running(running)
                return {
                    "job": item["label"],
                    "gpu": item["gpu"],
                    "return_code": return_code,
                    "log": str(item["log"]),
                }
            state = _task_output_state(raw_root, item["method"], item["seed"])
            if not state["complete"]:
                _stop_running(running)
                return {
                    "job": item["label"],
                    "gpu": item["gpu"],
                    "error": "extract subprocess exited successfully but output completeness check failed",
                    "output_state": state,
                    "log": str(item["log"]),
                }
            completed.append(item["label"])
    return None


def _extract_command(
    training_root: Path,
    raw_root: Path,
    mask_cache: Path,
    method: str,
    seed: int,
) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        "kd_mm_beam",
        "--no-capture-output",
        "python",
        "scripts/analyze_mmw_fused_feature_geometry.py",
        "extract",
        "--root",
        str(training_root),
        "--output-dir",
        str(raw_root / f"seed{seed}"),
        "--mask-cache",
        str(mask_cache),
        "--method",
        method,
        "--seed",
        str(seed),
        "--domain-shard-index",
        "0",
        "--domain-shard-count",
        "1",
    ]


def _run_summary(raw_root: Path, output_dir: Path, log_path: Path) -> int:
    command = [
        "conda",
        "run",
        "-n",
        "kd_mm_beam",
        "--no-capture-output",
        "python",
        "scripts/summarize_mmw_t2_bpa_cma_ablation.py",
        "--raw-root",
        str(raw_root),
        "--output-dir",
        str(output_dir),
        "--seeds",
        "1,2,3",
        "--expected-domains",
        str(EXPECTED_DOMAINS),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{_utc_now()}] {shlex.join(command)}\n")
        handle.flush()
        try:
            return int(subprocess.call(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT))
        except OSError as exc:
            handle.write(f"{type(exc).__name__}: {exc}\n")
            return 127


def _training_state(training_root: Path, method: str, seed: int) -> dict:
    run_dir = training_root / method / f"seed{seed}"
    status_path = run_dir / "run_status.json"
    checkpoint = run_dir / "checkpoints" / "last.pth"
    if not status_path.exists():
        state = "pending"
        error = None
    else:
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("status payload is not an object")
            state = str(payload.get("state", "unknown")).strip().lower()
            error = None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            state = "invalid"
            error = f"{type(exc).__name__}: {exc}"
    result = {
        "state": state,
        "status_path": str(status_path),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "ready": state == "complete" and checkpoint.is_file(),
    }
    if error:
        result["error"] = error
    return result


def _task_output_state(raw_root: Path, method: str, seed: int) -> dict:
    method_dir = raw_root / f"seed{seed}" / method
    domain_dir = method_dir / "domains"
    domain_paths = sorted(domain_dir.glob("*.npz")) if domain_dir.is_dir() else []
    worker_path = method_dir / "worker_0_of_1.json"
    result = {
        "complete": False,
        "state": "incomplete",
        "domain_count": len(domain_paths),
        "worker": str(worker_path),
    }
    if len(domain_paths) > EXPECTED_DOMAINS:
        return {**result, "state": "corrupt", "error": f"expected exactly 15 domain NPZ files, got {len(domain_paths)}"}
    if len(domain_paths) != EXPECTED_DOMAINS or not worker_path.is_file():
        return result
    try:
        payload = json.loads(worker_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("worker provenance is not an object")
        completed_domains = tuple(str(item) for item in payload.get("completed_domains", ()))
        valid = (
            payload.get("method") == method
            and int(payload.get("seed", -1)) == seed
            and int(payload.get("domain_shard_index", -1)) == 0
            and int(payload.get("domain_shard_count", -1)) == 1
            and len(completed_domains) == EXPECTED_DOMAINS
            and len(set(completed_domains)) == EXPECTED_DOMAINS
            and {path.stem for path in domain_paths} == {_safe_name(item) for item in completed_domains}
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {**result, "state": "incomplete", "error": f"{type(exc).__name__}: {exc}"}
    if not valid:
        return {**result, "state": "incomplete", "error": "worker provenance does not match method/seed/15-domain output"}
    return {**result, "complete": True, "state": "complete"}


def _stop_running(running: list[dict]) -> None:
    for item in running:
        if item["process"].poll() is None:
            item["process"].terminate()
    for item in running:
        try:
            item["process"].wait(timeout=10)
        except subprocess.TimeoutExpired:
            item["process"].kill()
            item["process"].wait()
        item["handle"].close()
    running.clear()


def _write_status(path: Path, state: str, **extra) -> None:
    payload = {"state": state, "updated_at": _utc_now(), **extra}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _safe_name(value: str) -> str:
    return str(value).replace("/", "__").replace(" ", "_")


def job_label(method: str, seed: int) -> str:
    return f"{method}/seed{seed}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
