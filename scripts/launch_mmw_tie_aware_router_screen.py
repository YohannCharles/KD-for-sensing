#!/usr/bin/env python3
"""Run the inner-only MMW tie-aware Router development screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import yaml

from kd_sensing.data.mmw.twc_evidence import load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "mmw_tie_aware_router_screen_v1"
DEFAULT_PROTOCOL = ROOT / "outputs/cache/mmw_twc_outer_v1/protocol_manifest.json"
DEFAULT_OUTPUT = ROOT / "outputs/mmw_tie_aware_router_screen_v1"
SEED = 1
BATCH_SIZE = 64
EPOCHS = 40
CANDIDATES: dict[str, dict[str, Any]] = {
    "HardFirstControl": {"mode": "hard_first", "temperature": 1.0},
    "HardConfidenceTie": {"mode": "hard_confidence_tie", "temperature": 1.0},
    "SoftUniformTie": {"mode": "soft_uniform_tie", "temperature": 1.0},
    "SoftConfidenceTie": {"mode": "soft_confidence_tie", "temperature": 1.0},
    "DistanceSoftT05": {"mode": "distance_soft", "temperature": 0.5},
    "DistanceSoftT10": {"mode": "distance_soft", "temperature": 1.0},
    "DistanceConfidenceT10": {"mode": "distance_confidence_soft", "temperature": 1.0},
    "UniformFusion": {"mode": "uniform_fusion", "temperature": 1.0},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--protocol-manifest", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    gpus = tuple(int(item.strip()) for item in args.gpus.split(",") if item.strip())
    if len(gpus) != len(CANDIDATES) or len(set(gpus)) != len(CANDIDATES) or any(not 0 <= gpu <= 7 for gpu in gpus):
        parser.error("--gpus must contain eight unique physical GPU ids in [0, 7].")
    if args.min_free_mib <= 0 or args.poll_seconds <= 0:
        parser.error("memory and poll settings must be positive.")

    output_root = _path(args.output_root)
    manifest_path = prepare_plan(output_root, _path(args.protocol_manifest), gpus)
    if not args.launch:
        print(json.dumps({"status": "planned", "manifest": str(manifest_path), "jobs": len(CANDIDATES)}, indent=2))
        return 0
    return run_plan(
        manifest_path,
        min_free_mib=int(args.min_free_mib),
        poll_seconds=float(args.poll_seconds),
    )


def prepare_plan(output_root: Path, protocol_path: Path, gpus: tuple[int, ...]) -> Path:
    protocol = load_protocol(protocol_path)
    domains, split_sha256 = inner_domains(protocol)
    launcher = _script_module("launch_mmw_all_weather_matrix")
    strict = _script_module("launch_mmw_twc_evidence")
    evaluator = _script_module("eval_mmw_all_weather_matrix")
    topology = strict._load_topology(strict._resolve_topology_path(None))
    mask_cache = output_root / "cache/inner_fixed_masks"
    evaluator._load_or_create_temporal_cache(
        mask_cache,
        modality_frame_masks=16,
        rates=evaluator.RATES,
        mask_types=evaluator.MASK_TYPES,
    )

    request = {
        "protocol_id": PROTOCOL_ID,
        "source_protocol_manifest_sha256": str(protocol["manifest_sha256"]),
        "split_sha256": split_sha256,
        "candidates": list(CANDIDATES),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "gpus": list(gpus),
        "selection_split": "frozen_inner_validation_only",
        "claim_eligible": False,
    }
    request_sha256 = _payload_sha256(request)
    manifest_path = output_root / "training_manifest_tie_aware_seed1.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing tie-aware screen plan differs from request: {manifest_path}")
        if any(Path(str(job.get("run_dir", ""))).exists() for job in existing.get("jobs", [])):
            return manifest_path

    config_dir = output_root / "generated_configs"
    log_dir = output_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for (candidate, settings), gpu in zip(CANDIDATES.items(), gpus):
        config = build_candidate_config(
            launcher,
            strict,
            topology,
            candidate,
            settings,
            output_root,
            domains=domains,
            source_protocol_sha256=str(protocol["manifest_sha256"]),
            split_sha256=split_sha256,
        )
        config_path = config_dir / f"{candidate}_seed{SEED}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        jobs.append(
            {
                "variant": candidate,
                "method": candidate,
                "scope": "MMW-15-domain-inner-validation",
                "seed": SEED,
                "gpu": int(gpu),
                "config_path": str(config_path.resolve()),
                "config_sha256": _file_sha256(config_path),
                "run_dir": str((output_root / candidate / f"seed{SEED}").resolve()),
                "log_path": str((log_dir / f"{candidate}_seed{SEED}.log").resolve()),
                "status": "planned",
                "evaluation_status": "planned",
                "evaluation_log_path": str((log_dir / f"{candidate}_seed{SEED}.eval.log").resolve()),
                "evaluation_output_path": str((output_root / "eval_inner" / candidate / "metrics.csv").resolve()),
            }
        )
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "request": request,
        "request_sha256": request_sha256,
        "protocol_manifest": str(protocol_path.resolve()),
        "mask_cache": str(mask_cache.resolve()),
        "jobs": jobs,
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def inner_domains(protocol: Mapping[str, Any]) -> tuple[list[dict[str, str]], str]:
    records = protocol.get("domains")
    if not isinstance(records, list) or len(records) != 15:
        raise ValueError("Tie-aware Router screen requires exactly 15 frozen MMW domains.")
    domains = []
    identity = []
    for record in records:
        split = record["split"]
        train = split["inner_train"]
        validation = split["inner_validation"]
        train_path = Path(str(train["csv"])).resolve()
        validation_path = Path(str(validation["csv"])).resolve()
        if _file_sha256(train_path) != train["sha256"] or _file_sha256(validation_path) != validation["sha256"]:
            raise ValueError(f"Frozen inner split changed for {record['id']}.")
        domains.append(
            {
                "id": str(record["id"]),
                "condition": str(record["condition"]),
                "scene": str(record["scene"]),
                "data_root": str(record["data_root"]),
                "train_csv_name": str(train_path),
                "val_csv_name": str(validation_path),
                "test_csv_name": str(validation_path),
            }
        )
        identity.append(
            {
                "id": str(record["id"]),
                "inner_train_sha256": str(train["sha256"]),
                "inner_validation_sha256": str(validation["sha256"]),
            }
        )
    return domains, _payload_sha256(identity)


def build_candidate_config(
    launcher: ModuleType,
    strict: ModuleType,
    topology: Mapping[str, Any],
    candidate: str,
    settings: Mapping[str, Any],
    output_root: Path,
    *,
    domains: list[dict[str, str]],
    source_protocol_sha256: str,
    split_sha256: str,
) -> dict[str, Any]:
    config = launcher.build_config(
        "T2",
        output_root,
        seed=SEED,
        smoke=False,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        umask_training_profile="umask_h4_v1",
        umask_router_architecture_profile="umask_router_nopattern_v1",
    )
    dataset = config["data"]["dataset"]
    dataset["domains"] = deepcopy(domains)
    for key in ("train_csv_name", "val_csv_name", "test_csv_name"):
        dataset.pop(key, None)
    config["training"].update(
        {
            "epochs": EPOCHS,
            "max_epochs": EPOCHS,
            "validation": {"interval_epochs": 5},
            "final_test": {"enabled": False, "reason": "tie_aware_router_inner_validation_only"},
            "allow_tf32": False,
            "cudnn_benchmark": False,
        }
    )
    config["data"]["domain_balanced_sampling"]["seed"] = SEED
    config["temporal_missing"]["seed"] = SEED
    config["experiment"].update({"name": candidate, "seed": SEED, "ablation_id": candidate})
    config["output"] = {
        "dir": str((output_root / candidate).resolve()),
        "run_name": f"seed{SEED}",
        "group_by_scene": False,
        "overwrite": False,
        "progress": {"enabled": False},
        "tensorboard": {"enabled": False},
    }
    loss = config["loss"]["u_mask_beam_jepa"]
    strict._set_physical_bpa(loss, topology)
    apply_candidate(config, candidate, settings)
    config.pop("mmw_twc_evidence", None)
    config["mmw_tie_aware_router_screen"] = {
        "protocol": PROTOCOL_ID,
        "candidate_id": candidate,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "selection_split": "frozen_inner_validation_only",
        "source_protocol_manifest_sha256": source_protocol_sha256,
        "inner_split_sha256": split_sha256,
        "router_oracle_target_mode": str(settings["mode"]),
        "router_oracle_temperature": float(settings["temperature"]),
        "candidate_recipe_sha256": _payload_sha256(
            {
                "candidate": candidate,
                "settings": dict(settings),
                "source_protocol_manifest_sha256": source_protocol_sha256,
                "inner_split_sha256": split_sha256,
                "training_profile": "umask_h4_v1",
                "router_profile": "umask_router_nopattern_v1",
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            }
        ),
        "claim_eligible": False,
    }
    config["mmw_all_weather_protocol"].update(
        {
            "split_tag": PROTOCOL_ID,
            "screening_role": "development_inner_validation",
            "checkpoint_policy": "fixed_epoch_last_pth",
        }
    )
    config.setdefault("evaluation", {}).update(
        {
            "beam_distance_circular": True,
            "dba_distance_mode": "circular",
            "metric_profile": "64_beam_ula_dft_phase_cycle_topk_progressive_top3_dba_v1",
        }
    )
    return config


def apply_candidate(config: dict[str, Any], candidate: str, settings: Mapping[str, Any]) -> None:
    if candidate not in CANDIDATES or dict(settings) != CANDIDATES[candidate]:
        raise ValueError(f"Unknown or mismatched tie-aware Router candidate {candidate!r}.")
    loss = config["loss"]["u_mask_beam_jepa"]
    primary = config["model"]["primary"]
    if candidate == "UniformFusion":
        primary["fusion_type"] = "uniform_mean"
        loss["router_oracle_weight"] = 0.0
        loss["router_oracle_target_mode"] = "hard_first"
        loss["router_oracle_temperature"] = 1.0
        return
    primary["fusion_type"] = "supervised_router"
    loss["router_oracle_weight"] = 0.1
    loss["router_oracle_target_mode"] = str(settings["mode"])
    loss["router_oracle_temperature"] = float(settings["temperature"])


def run_plan(manifest_path: Path, *, min_free_mib: int, poll_seconds: float) -> int:
    manifest = _read_json(manifest_path)
    free = _gpu_free_memory()
    required = {int(job["gpu"]) for job in manifest["jobs"]}
    blocked = {gpu: free.get(gpu, 0) for gpu in required if free.get(gpu, 0) < min_free_mib}
    if blocked:
        raise RuntimeError(f"Tie-aware screen GPUs do not meet free-memory threshold: {blocked}")

    running: dict[int, tuple[subprocess.Popen, Any, dict[str, Any], str]] = {}
    for job in manifest["jobs"]:
        if _completed_run(job):
            job["status"] = "done"
            phase = "evaluation"
            process, handle = _start_evaluation(job, manifest)
            job["evaluation_status"] = "running"
        else:
            phase = "training"
            process, handle = _start_training(job)
            job["status"] = "running"
        if phase == "training":
            job["pid"] = process.pid
            job["start_time"] = _now()
        else:
            job["evaluation_pid"] = process.pid
            job["evaluation_start_time"] = _now()
        running[int(job["gpu"])] = (process, handle, job, phase)
    _write_json(manifest_path, manifest)

    while running:
        for gpu, (process, handle, job, phase) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job[f"{phase}_return_code"] = int(code)
            job[f"{phase}_end_time"] = _now()
            if phase == "training":
                if code == 0 and _completed_run(job):
                    job["status"] = "done"
                    next_process, next_handle = _start_evaluation(job, manifest)
                    job.update(
                        {
                            "evaluation_status": "running",
                            "evaluation_pid": next_process.pid,
                            "evaluation_start_time": _now(),
                        }
                    )
                    running[gpu] = (next_process, next_handle, job, "evaluation")
                else:
                    job["status"] = "failed"
                    job["evaluation_status"] = "blocked"
                    running.pop(gpu)
            else:
                job["evaluation_status"] = "done" if code == 0 and Path(job["evaluation_output_path"]).is_file() else "failed"
                running.pop(gpu)
            _write_json(manifest_path, manifest)
        time.sleep(poll_seconds)
    return 0 if all(job["status"] == "done" and job["evaluation_status"] == "done" for job in manifest["jobs"]) else 1


def _start_training(job: Mapping[str, Any]) -> tuple[subprocess.Popen, Any]:
    return _start(
        job,
        "log_path",
        ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "kd-sensing-train", "--config", str(job["config_path"])],
    )


def _start_evaluation(job: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[subprocess.Popen, Any]:
    root = Path(str(job["run_dir"])).parents[1]
    return _start(
        job,
        "evaluation_log_path",
        [
            "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
            "scripts/eval_mmw_all_weather_matrix.py",
            "--root", str(root),
            "--methods", str(job["method"]),
            "--seeds", str(job["seed"]),
            "--output-dir", str(root / "eval_inner"),
            "--mask-cache", str(manifest["mask_cache"]),
            "--modality-frame-masks", "16",
            "--batch-size", str(BATCH_SIZE),
        ],
    )


def _start(job: Mapping[str, Any], log_key: str, command: list[str]) -> tuple[subprocess.Popen, Any]:
    log_path = Path(str(job[log_key]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
        }
    )
    handle.write(f"[{_now()}] GPU{job['gpu']}: {' '.join(command)}\n")
    handle.flush()
    return subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True), handle


def _completed_run(job: Mapping[str, Any]) -> bool:
    run_dir = Path(str(job["run_dir"]))
    status = run_dir / "run_status.json"
    checkpoint = run_dir / "checkpoints/last.pth"
    return status.is_file() and checkpoint.is_file() and _read_json(status).get("state") == "complete"


def _gpu_free_memory() -> dict[int, int]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True
    )
    return {int(index.strip()): int(memory.strip()) for index, memory in (line.split(",", 1) for line in output.splitlines())}


def _script_module(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tie_aware_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load project script {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
