#!/usr/bin/env python3
"""Plan and run the 13-condition MMW Router Oracle Gap Test on GPU0--7."""

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
from typing import Any

from eval_mmw_router_oracle_gap import CONDITIONS, DEFAULT_CHECKPOINT, DEFAULT_CONFIG, DEFAULT_OUTPUT, PROTOCOL_ID
from kd_sensing.evaluation.corruptions import CORRUPTION_PARAMETERS


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    "image_occlusion": "https://arxiv.org/abs/1708.04896",
    "radar_noise": "https://ieeexplore.ieee.org/document/9274416",
    "lidar_sparsify": "https://arxiv.org/abs/2201.12296",
    "multi_sensor_corruption": "https://arxiv.org/abs/2501.01037",
    "gps_noise": "https://doi.org/10.3390/s19245402",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if gpus != tuple(range(8)):
        parser.error("This user-authorized protocol requires exactly --gpus 0,1,2,3,4,5,6,7.")
    output = Path(args.output_root).resolve()
    manifest_path = prepare_manifest(
        output,
        Path(args.config).resolve(),
        Path(args.checkpoint).resolve(),
        gpus,
        batch_size=int(args.batch_size),
    )
    if not args.launch:
        print(json.dumps({"status": "planned", "manifest": str(manifest_path), "jobs": len(CONDITIONS)}, indent=2))
        return 0
    return run_manifest(manifest_path)


def prepare_manifest(
    output: Path, config: Path, checkpoint: Path, gpus: tuple[int, ...], *, batch_size: int
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    request = {
        "protocol": PROTOCOL_ID,
        "config": str(config),
        "config_sha256": sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "conditions": list(CONDITIONS),
        "corruption_parameters": CORRUPTION_PARAMETERS,
        "corruption_seed": 20260718,
        "gpus": list(gpus),
        "batch_size": int(batch_size),
        "split": "frozen_inner_validation_only",
        "claim_eligible": False,
        "references": REFERENCES,
    }
    request_sha256 = payload_sha256(request)
    path = output / "evaluation_manifest.json"
    if path.is_file():
        manifest = read_json(path)
        if manifest.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing Oracle Gap manifest differs from the frozen request: {path}")
        return path
    jobs = []
    for index, condition in enumerate(CONDITIONS):
        jobs.append(
            {
                "condition": condition,
                "gpu": int(gpus[index % len(gpus)]),
                "status": "planned",
                "pid": None,
                "returncode": None,
                "log": str((output / "logs" / f"{condition}.log").resolve()),
                "output": str((output / condition).resolve()),
            }
        )
    write_json(
        path,
        {
            "schema_version": 1,
            "request": request,
            "request_sha256": request_sha256,
            "created_at": now(),
            "jobs": jobs,
        },
    )
    return path


def run_manifest(path: Path) -> int:
    manifest = read_json(path)
    request = manifest["request"]
    running: dict[int, tuple[subprocess.Popen, Any, int]] = {}
    while True:
        changed = False
        for gpu, (process, handle, job_index) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            job = manifest["jobs"][job_index]
            job.update({"status": "complete" if returncode == 0 else "failed", "returncode": returncode, "ended_at": now()})
            running.pop(gpu)
            changed = True
        for job_index, job in enumerate(manifest["jobs"]):
            if job["status"] != "planned" or int(job["gpu"]) in running:
                continue
            output = Path(job["output"])
            if (output / "complete.json").is_file():
                job.update({"status": "complete", "returncode": 0, "ended_at": now()})
                changed = True
                continue
            command = [
                sys.executable,
                str(ROOT / "scripts/eval_mmw_router_oracle_gap.py"),
                "--condition",
                job["condition"],
                "--config",
                request["config"],
                "--checkpoint",
                request["checkpoint"],
                "--output-root",
                str(path.parent),
                "--batch-size",
                str(request["batch_size"]),
            ]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
            handle = Path(job["log"]).open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            job.update({"status": "running", "pid": process.pid, "started_at": now(), "command": command})
            running[int(job["gpu"])] = (process, handle, job_index)
            changed = True
        if changed:
            write_json(path, manifest)
        if not running and all(job["status"] in {"complete", "failed"} for job in manifest["jobs"]):
            break
        time.sleep(2)
    failures = [job for job in manifest["jobs"] if job["status"] == "failed"]
    manifest["completed_at"] = now()
    manifest["status"] = "failed" if failures else "complete"
    write_json(path, manifest)
    return 1 if failures else 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
