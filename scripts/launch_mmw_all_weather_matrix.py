#!/usr/bin/env python3
"""Generate and launch the three Clean MMW training routes."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.config import dump_config, load_config
from kd_sensing.config.io import deep_merge, load_config_source
from kd_sensing.data.mmw.clean_protocol import (
    audit_clean_inner_protocol,
    load_clean_inner_protocol,
    protocol_dataset_domains,
    validate_clean_config_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("U0", "amber_full", "rmbp_mm")
METHOD_BASES = {
    "U0": "configs/mmw/u0.yaml",
    "amber_full": "configs/mmw/amber_full.yaml",
    "rmbp_mm": "configs/mmw/rmbp_mm.yaml",
}
DEFAULT_OUTPUT_ROOT = "outputs/mmw_clean_u0"


def build_config(
    method: str,
    output_root: Path,
    *,
    protocol_path: Path,
    audit_report: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    smoke: bool = False,
) -> dict[str, Any]:
    """Build one train-only config from a retained route and audited protocol."""
    if method not in METHOD_BASES:
        raise ValueError(f"Unsupported Clean MMW method: {method}")
    protocol = load_clean_inner_protocol(protocol_path)
    _validate_audit_report(protocol_path, audit_report, protocol)

    cfg = _load_base_config(ROOT / METHOD_BASES[method])
    dataset = cfg.setdefault("data", {}).setdefault("dataset", {})
    dataset["type"] = "mmw"
    dataset["domains"] = protocol_dataset_domains(protocol)
    for domain in dataset["domains"]:
        domain.pop("test_csv_name", None)
    cfg["data_protocol"] = {
        "mode": "clean_inner_development",
        "path": str(protocol_path.resolve()),
        "audit_report": str(audit_report.resolve()),
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "train_role": "inner_train",
        "validation_role": "inner_validation",
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
    }

    loader = cfg.setdefault("data", {}).setdefault("dataloader", {})
    loader.update(
        {
            "train_batch_size": int(batch_size),
            "validation_batch_size": int(batch_size),
            "test_batch_size": int(batch_size),
        }
    )
    training = cfg.setdefault("training", {})
    training.update({"epochs": 1 if smoke else int(epochs), "max_epochs": 1 if smoke else int(epochs)})
    training["final_test"] = {"enabled": False}
    cfg.setdefault("experiment", {}).update({"name": method, "seed": int(seed)})
    cfg["output"] = {
        "dir": str(output_root / method),
        "run_name": f"seed{seed}",
        "group_by_scene": False,
        "overwrite": False,
        "progress": {"enabled": False},
        "tensorboard": {"enabled": False},
    }
    return cfg


def build_job_matrix(
    methods: tuple[str, ...], seeds: tuple[int, ...], gpus: tuple[int, ...] | None, output_root: Path
) -> list[dict[str, Any]]:
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f"methods must be unique members of: {', '.join(METHODS)}")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed <= 0 for seed in seeds):
        raise ValueError("seeds must be unique positive integers")
    pairs = [(method, seed) for method in methods for seed in seeds]
    selected_gpus = tuple(range(len(pairs))) if gpus is None else gpus
    if len(selected_gpus) != len(pairs) or len(set(selected_gpus)) != len(selected_gpus) or any(gpu < 0 for gpu in selected_gpus):
        raise ValueError(f"provide {len(pairs)} unique non-negative GPU ids")
    return [
        {
            "method": method,
            "seed": seed,
            "gpu": gpu,
            "config_path": output_root / "generated_configs" / f"{method}_seed{seed}.yaml",
            "log_path": output_root / "logs" / f"{method}_seed{seed}.log",
            "run_dir": output_root / method / f"seed{seed}",
            "status": "planned",
        }
        for (method, seed), gpu in zip(pairs, selected_gpus)
    ]


def validate_job_targets(jobs: list[dict[str, Any]], manifest_path: Path) -> None:
    targets = [manifest_path]
    for job in jobs:
        targets.extend((Path(job["config_path"]), Path(job["log_path"]), Path(job["run_dir"])))
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing targets:\n" + "\n".join(str(path) for path in existing))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the Clean MMW U0, AMBER, and RMBP routes.")
    parser.add_argument("--protocol", required=True, help="Clean inner-development protocol YAML.")
    parser.add_argument("--audit-report", required=True, help="Matching passed clean split audit JSON.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--gpus")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        methods = tuple(_csv(args.methods))
        seeds = tuple(int(item) for item in _csv(args.seeds))
        gpus = None if args.gpus is None else tuple(int(item) for item in _csv(args.gpus))
        protocol_path = Path(args.protocol).resolve()
        audit_report = Path(args.audit_report).resolve()
        protocol = load_clean_inner_protocol(protocol_path)
        _validate_audit_report(protocol_path, audit_report, protocol)
        output_root = (ROOT / args.output_root).resolve()
        jobs = build_job_matrix(methods, seeds, gpus, output_root)
        manifest_path = output_root / "jobs.json"
        validate_job_targets(jobs, manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    output_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        config_path = Path(job["config_path"])
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = build_config(
            job["method"],
            output_root,
            protocol_path=protocol_path,
            audit_report=audit_report,
            seed=int(job["seed"]),
            epochs=args.epochs,
            batch_size=args.batch_size,
            smoke=args.smoke,
        )
        dump_config(config, config_path)
        resolved = load_config(config_path)
        validate_clean_config_protocol(resolved)
        dump_config(resolved, config_path)
        for key in ("config_path", "log_path", "run_dir"):
            job[key] = str(Path(job[key]).relative_to(ROOT))
    manifest_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    return _run_jobs(jobs, manifest_path)


def _load_base_config(path: Path) -> dict[str, Any]:
    source = load_config_source(path)
    payload = copy.deepcopy(source.data)
    bases = payload.pop("_base_", [])
    if isinstance(bases, (str, Path)):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(str(base))
        if not base_path.is_absolute():
            base_path = source.path.parent / base_path
        merged = deep_merge(merged, _load_base_config(base_path))
    return deep_merge(merged, payload)


def _validate_audit_report(protocol_path: Path, audit_report: Path, protocol: Mapping[str, Any]) -> None:
    if not audit_report.is_file():
        raise FileNotFoundError(f"Clean split audit report is missing: {audit_report}")
    actual = audit_clean_inner_protocol(protocol_path, fail_closed=True)
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Clean split audit report must be a JSON object.")
    required = (
        "audit_id",
        "protocol_file_sha256",
        "protocol_fingerprint",
        "train_sample_id_hash",
        "validation_sample_id_hash",
        "pair_count",
        "overlap_counts",
        "failed_pairs",
    )
    if report.get("status") != "passed" or report.get("outer_test_accessed") is not False:
        raise ValueError("Clean split audit report must be passed and must not access outer test.")
    if any(report.get(key) != actual[key] for key in required):
        raise ValueError("Clean split audit report does not match the supplied protocol.")
    if protocol["protocol_fingerprint"] != actual["protocol_fingerprint"]:
        raise ValueError("Clean protocol fingerprint changed during audit validation.")


def _run_jobs(jobs: list[dict[str, Any]], manifest_path: Path) -> int:
    running = []
    for job in jobs:
        log_path = ROOT / job["log_path"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "kd-sensing-train", "--config", job["config_path"]]
        environment = os.environ.copy()
        environment.update({"CUDA_VISIBLE_DEVICES": str(job["gpu"]), "PYTHONUNBUFFERED": "1"})
        handle = log_path.open("w", encoding="utf-8")
        job.update({"status": "running", "command": command, "start_time": _now()})
        running.append((subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT), job, handle))
    manifest_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")

    failed = False
    for process, job, handle in running:
        return_code = process.wait()
        handle.close()
        job.update({"status": "done" if return_code == 0 else "failed", "return_code": return_code, "end_time": _now()})
        failed = failed or return_code != 0
        manifest_path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")
    return int(failed)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
