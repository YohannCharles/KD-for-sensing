#!/usr/bin/env python3
"""Prepare or run the protocol-bound prototype decision Adapter Stage A."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from kd_sensing.baselines.prototype_decision_adapter import EXPERIMENTS, prepare_stage, run_experiment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "outputs/clean_recovery_stage_a/gpu0_u0_seed1/final_config.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs/clean_recovery_stage_a/gpu0_u0_seed1/checkpoints/last.pth"
DEFAULT_ROOT = ROOT / "outputs/prototype_decision_adapter"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _capture(command: list[str]) -> str:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout


def launch_all(config: Path, checkpoint: Path, output_root: Path, epochs: int) -> int:
    if (output_root / "stage_a_manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {output_root / 'stage_a_manifest.json'}")
    protocol = prepare_stage(config, checkpoint, output_root / "protocol")
    config = Path(protocol["resolved_config_path"])
    preflight = {
        "time": _now(),
        "nvidia_smi": _capture(["nvidia-smi"]),
        "git_commit": _capture(["git", "rev-parse", "HEAD"]).strip(),
        "git_diff": _capture(["git", "diff", "--", "."]),
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "data_protocol_fingerprint": protocol["protocol"].get("protocol_fingerprint"),
        "mask_schedule_sha256": protocol["schedule_sha256"],
        "experiment_seed": 1,
        "prototype_cache": {"batch_size": 128, "num_workers": 8},
        "experiments": [value.__dict__ for value in EXPERIMENTS.values()],
    }
    _write(output_root / "preflight_environment.json", preflight)
    schedule = output_root / "protocol/mask_schedule_seed1.json"
    jobs = []
    running = []
    for experiment in EXPERIMENTS.values():
        run_dir = output_root / "stage_a" / experiment.run_name
        log_path = output_root / "logs" / f"{experiment.run_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        status_path = run_dir / "status.json"
        if status_path.exists() and (run_dir / "metrics.json").exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "completed" and status.get("return_code") == 0:
                jobs.append({
                    "experiment": experiment.key, "gpu_id": experiment.gpu, "pid": None,
                    "command": None, "config_path": str(config), "log_path": str(log_path),
                    "output_path": str(run_dir), "status": "completed", "return_code": 0,
                    "reused_completed_run": True,
                })
                continue
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite incomplete run: {run_dir}")
        command = [
            "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
            "tools/run_prototype_decision_adapter.py", "--experiment", experiment.key,
            "--config", str(config), "--checkpoint", str(checkpoint), "--schedule", str(schedule),
            "--output-root", str(output_root), "--epochs", str(epochs),
        ]
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(experiment.gpu), PYTHONUNBUFFERED="1")
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        job = {
            "experiment": experiment.key, "gpu_id": experiment.gpu, "pid": process.pid,
            "start_time": _now(), "command": command, "config_path": str(config),
            "log_path": str(log_path), "output_path": str(run_dir), "status": "running",
        }
        jobs.append(job)
        running.append((process, handle, job))
    manifest_path = output_root / "stage_a_manifest.json"
    _write(manifest_path, {"outer_test_accessed": False, "jobs": jobs})
    failed = False
    for process, handle, job in running:
        code = process.wait()
        handle.close()
        job.update(status="completed" if code == 0 else "failed", return_code=code, end_time=_now())
        failed |= code != 0
        _write(manifest_path, {"outer_test_accessed": False, "jobs": jobs})
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-stage", action="store_true")
    parser.add_argument("--launch-all", action="store_true")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    if sum((args.prepare_stage, args.launch_all, args.experiment is not None)) != 1:
        parser.error("choose exactly one of --prepare-stage, --launch-all, or --experiment")
    if args.prepare_stage:
        print(json.dumps(prepare_stage(args.config, args.checkpoint, args.output_root / "protocol"), indent=2))
        return 0
    if args.launch_all:
        return launch_all(args.config.resolve(), args.checkpoint.resolve(), args.output_root, args.epochs)
    schedule = args.schedule or args.output_root / "protocol/mask_schedule_seed1.json"
    experiment = EXPERIMENTS[args.experiment]
    run_dir = args.output_root / "stage_a" / experiment.run_name
    try:
        run_experiment(args.experiment, args.config, args.checkpoint, schedule, run_dir, epochs=args.epochs)
    except Exception as exc:
        if run_dir.exists():
            _write(run_dir / "status.json", {"status": "failed", "return_code": 1, "error": f"{type(exc).__name__}: {exc}"})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
