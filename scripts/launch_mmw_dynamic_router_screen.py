#!/usr/bin/env python3
"""Prepare or launch seed1 dynamic reliability Router candidates."""

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
from typing import Any, Mapping

import yaml

from kd_sensing.data.mmw.twc_router_joint_training import (
    PANEL_SEED,
    PROTOCOL_ID as PANEL_PROTOCOL_ID,
    prepare_router_joint_training_panel,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "mmw_dynamic_router_screen_v1"
DEFAULT_OUTPUT = ROOT / "outputs/mmw_dynamic_router_screen_v1"
DEFAULT_SOURCE_CONFIG = (
    ROOT / "outputs/mmw_router_expected_utility_screen_v3/generated_configs/CurrentControl_seed1.yaml"
)
DEFAULT_SOURCE_CHECKPOINT = (
    ROOT / "outputs/mmw_router_expected_utility_screen_v3/CurrentControl/seed1/checkpoints/last.pth"
)
DEFAULT_SOURCE_SHA256 = "82b8d07fc4df3b38750a8d9f6d60064e4a6a35c740c2efefb0a58e7fef99f037"
ROUTER_RELIABILITY_SOURCE = ROOT / "src/kd_sensing/losses/router_reliability.py"
UTILITY_NUMERIC_POLICY = "beam_power_float32_before_linear_normalization_v1"
SEED = 1
CANDIDATES = (
    ("PATR-Label", "patr", "label_topology"),
    ("PATR-Power", "patr", "beam_power"),
    ("H2R-Label", "h2r", "label_topology"),
    ("H2R-Power", "h2r", "beam_power"),
    ("CoRe-Label", "core", "label_topology"),
    ("CoRe-Power", "core", "beam_power"),
    ("Unified-HPR-Label", "unified_hpr", "label_topology"),
    ("Unified-HPR-Power", "unified_hpr", "beam_power"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--source-config", default=str(DEFAULT_SOURCE_CONFIG))
    parser.add_argument("--source-checkpoint", default=str(DEFAULT_SOURCE_CHECKPOINT))
    parser.add_argument("--expected-source-sha256", default=DEFAULT_SOURCE_SHA256)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--candidates",
        default="all",
        help="Comma-separated candidate names, or 'all' for the frozen eight-job screen.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    gpus = tuple(int(value.strip()) for value in args.gpus.split(",") if value.strip())
    try:
        candidates = select_candidates(args.candidates)
    except ValueError as exc:
        parser.error(str(exc))
    if len(gpus) != len(candidates) or len(set(gpus)) != len(gpus):
        parser.error("--gpus must provide one unique GPU for each selected candidate.")
    if candidates == CANDIDATES and gpus != tuple(range(8)):
        parser.error("The full frozen screen requires exactly --gpus 0,1,2,3,4,5,6,7.")
    if args.batch_size <= 0 or args.batch_size % 16:
        parser.error("--batch-size must be a positive multiple of 16.")
    if args.epochs <= 0 or args.min_free_mib <= 0 or args.poll_seconds <= 0:
        parser.error("--epochs, --min-free-mib, and --poll-seconds must be positive.")
    manifest = prepare_plan(
        output_root=_path(args.output_root),
        source_config=_path(args.source_config),
        source_checkpoint=_path(args.source_checkpoint),
        expected_source_sha256=str(args.expected_source_sha256).strip().lower(),
        gpus=gpus,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        candidates=candidates,
    )
    if not args.launch:
        print(json.dumps({"status": "planned", "manifest": str(manifest), "jobs": len(candidates)}, indent=2))
        return 0
    return run_plan(
        manifest,
        min_free_mib=int(args.min_free_mib),
        poll_seconds=float(args.poll_seconds),
    )


def prepare_plan(
    *,
    output_root: Path,
    source_config: Path,
    source_checkpoint: Path,
    expected_source_sha256: str,
    gpus: tuple[int, ...],
    batch_size: int,
    epochs: int,
    candidates: tuple[tuple[str, str, str], ...] = CANDIDATES,
) -> Path:
    if not source_config.is_file() or not source_checkpoint.is_file():
        raise FileNotFoundError("Dynamic Router source config/checkpoint is missing.")
    if len(expected_source_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_source_sha256):
        raise ValueError("expected source checkpoint SHA256 must be lowercase hexadecimal.")
    actual_source_sha256 = _sha256(source_checkpoint)
    if actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            f"Source checkpoint SHA256 mismatch: expected={expected_source_sha256}, actual={actual_source_sha256}."
        )
    base = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
    _validate_source_config(base)
    panel_path = output_root / "cache/joint_training_panel_v1.json"
    panel = prepare_router_joint_training_panel(panel_path, seed=PANEL_SEED)
    request = {
        "protocol": PROTOCOL_ID,
        "panel_protocol": PANEL_PROTOCOL_ID,
        "panel_checksum": str(panel["checksum"]),
        "panel_seed": PANEL_SEED,
        "source_config": str(source_config),
        "source_config_sha256": _sha256(source_config),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": actual_source_sha256,
        "router_reliability_source_sha256": _sha256(ROUTER_RELIABILITY_SOURCE),
        "utility_numeric_policy": UTILITY_NUMERIC_POLICY,
        "seed": SEED,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "gpus": list(gpus),
        "candidates": [list(item) for item in candidates],
        "selection_split": "frozen_inner_validation_only",
        "claim_eligible": False,
    }
    request_sha256 = _payload_sha256(request)
    manifest_path = output_root / "training_manifest_seed1.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        if existing.get("request_sha256") != request_sha256:
            raise ValueError(f"Existing dynamic Router plan differs from the frozen request: {manifest_path}")
        return manifest_path
    config_dir = output_root / "generated_configs"
    log_dir = output_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for gpu, (name, variant, supervision) in zip(gpus, candidates, strict=True):
        config = build_candidate_config(
            base,
            name=name,
            variant=variant,
            supervision=supervision,
            output_root=output_root,
            panel_path=panel_path,
            panel_checksum=str(panel["checksum"]),
            source_checkpoint=source_checkpoint,
            source_sha256=actual_source_sha256,
            batch_size=batch_size,
            epochs=epochs,
        )
        config_path = config_dir / f"{name}_seed1.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        jobs.append(
            {
                "candidate": name,
                "router_variant": variant,
                "supervision": supervision,
                "seed": SEED,
                "gpu": int(gpu),
                "config_path": str(config_path.resolve()),
                "config_sha256": _sha256(config_path),
                "run_dir": str((output_root / name / "seed1").resolve()),
                "log_path": str((log_dir / f"{name}_seed1.log").resolve()),
                "status": "planned",
                "claim_eligible": False,
            }
        )
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol": PROTOCOL_ID,
            "request": request,
            "request_sha256": request_sha256,
            "panel_path": str(panel_path.resolve()),
            "jobs": jobs,
            "status": "planned",
            "created_at": _now(),
        },
    )
    return manifest_path


def select_candidates(value: str) -> tuple[tuple[str, str, str], ...]:
    requested = str(value).strip()
    if requested.lower() == "all":
        return CANDIDATES
    names = tuple(item.strip() for item in requested.split(",") if item.strip())
    if not names or len(set(names)) != len(names):
        raise ValueError("--candidates must contain unique candidate names or 'all'.")
    by_name = {item[0]: item for item in CANDIDATES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown dynamic Router candidates: {unknown}.")
    return tuple(by_name[name] for name in names)


def build_candidate_config(
    base: Mapping[str, Any],
    *,
    name: str,
    variant: str,
    supervision: str,
    output_root: Path,
    panel_path: Path,
    panel_checksum: str,
    source_checkpoint: Path,
    source_sha256: str,
    batch_size: int,
    epochs: int,
) -> dict[str, Any]:
    if (name, variant, supervision) not in CANDIDATES:
        raise ValueError(f"Unknown dynamic Router candidate {(name, variant, supervision)!r}.")
    config = deepcopy(dict(base))
    config["experiment"].update({"name": name, "seed": SEED, "ablation_id": name})
    dataset = config["data"]["dataset"]
    dataset["include_router_utility_targets"] = supervision == "beam_power"
    dataset["include_router_corruption_metadata"] = True
    dataloader = config["data"]["dataloader"]
    dataloader.update(
        {
            "train_batch_size": int(batch_size),
            "validation_batch_size": int(batch_size),
            "test_batch_size": int(batch_size),
        }
    )
    config.setdefault("temporal_missing", {})["preserve_unmasked_for_superset"] = True
    primary = config["model"]["primary"]
    topology = config["loss"]["u_mask_beam_jepa"]["prototype_topology"]
    primary.update(
        {
            "router_variant": variant,
            "router_calibration_only": True,
            "router_variant_config": {
                "topology_id": str(topology["id"]),
                "topology_permutation": topology.get("permutation"),
                "circular": bool(config["loss"]["u_mask_beam_jepa"].get("prototype_target_circular", True)),
                "residual_hidden_dim": 64,
                "health_hidden_dim": 16,
                "residual_scale": 1.0,
                "top_k": 3,
                "dropout": 0.0,
            },
        }
    )
    loss = config["loss"]["u_mask_beam_jepa"]
    loss.update(
        {
            "router_oracle_weight": 0.0,
            "lambda_proto": 0.0,
            "lambda_modality_proto": 0.0,
            "superset_consistency": {
                "enabled": False,
                "confidence_gated_kl": False,
                "kl_weight": 0.0,
                "temperature": 2.0,
            },
            "router_quality_pairing": {
                "enabled": False,
                "utility_weight": 0.0,
                "monotonic_weight": 0.0,
            },
            "dynamic_router": {
                "supervision": supervision,
                "utility_temperature": 0.5,
                "quality_regression_weight": 0.1,
                "fused_utility_weight": 0.2,
                "frame_rank_weight": 0.1 if variant in {"h2r", "unified_hpr"} else 0.0,
                "residual_anchor_weight": 0.01,
                "paired_joint": {
                    "enabled": True,
                    "panel_path": str(panel_path.resolve()),
                    "panel_sha256": panel_checksum,
                    "corruption_seed": 20260719,
                    "monotonic_weight": 0.1,
                    "monotonic_margin_scale": 0.25,
                    "quality_drop_epsilon": 0.01,
                },
            },
        }
    )
    training = config["training"]
    router_parameter_groups = [
        {
            "name": "train_fit_global_prior",
            "module_patterns": ["prototype_reliability_router.prior_logits"],
            "lr": 1.0e-2,
            "weight_decay": 0.0,
        },
        {
            "name": "dynamic_residual",
            "module_patterns": ["prototype_reliability_router.modality_residual_head"],
            "lr": 1.0e-3,
            "weight_decay": 3.0e-4,
        },
    ]
    if variant in {"h2r", "unified_hpr"}:
        router_parameter_groups.append(
            {
                "name": "temporal_health",
                "module_patterns": ["prototype_reliability_router.frame_health_head"],
                "lr": 1.0e-3,
                "weight_decay": 3.0e-4,
            }
        )
    training.update(
        {
            "epochs": int(epochs),
            "max_epochs": int(epochs),
            "start_epoch": 0,
            "resume": False,
            "lr": 1.0e-3,
            "weight_decay": 3.0e-4,
            "validation": {"interval_epochs": int(epochs)},
            "final_test": {"enabled": False, "reason": "dynamic_router_seed1_inner_only"},
            "initialization_checkpoint": {
                "path": str(source_checkpoint.resolve()),
                "sha256": source_sha256,
                "role": "last",
                "checkpoint_schema_version": 1,
                "required_prefixes": [
                    "encoders",
                    "reliability_heads",
                    "classifier",
                    "prototype_bank",
                    "supervised_router",
                ],
                "allowed_missing_prefixes": ["prototype_reliability_router"],
                "freeze_prefixes": [
                    "encoders",
                    "encoder_projections",
                    "reliability_heads",
                    "classifier",
                    "prototype_bank",
                    "supervised_router",
                ],
            },
            "optimizer": {
                "type": "adamw",
                "require_all_matched": True,
                "parameter_groups": router_parameter_groups,
            },
        }
    )
    config["scheduler"] = {"type": "cosine_warm_restarts", "T_0": int(epochs), "T_mult": 1, "eta_min": 1.0e-6}
    config["output"].update(
        {
            "dir": str((output_root / name).resolve()),
            "run_name": "seed1",
            "overwrite": False,
            "progress": {"enabled": False},
        }
    )
    for stale in ("mmw_tie_aware_router_screen", "mmw_router_utility_screen"):
        config.pop(stale, None)
    config["mmw_dynamic_router_screen"] = {
        "protocol": PROTOCOL_ID,
        "candidate": name,
        "router_variant": variant,
        "supervision": supervision,
        "seed": SEED,
        "source_checkpoint_sha256": source_sha256,
        "joint_panel_checksum": panel_checksum,
        "router_reliability_source_sha256": _sha256(ROUTER_RELIABILITY_SOURCE),
        "utility_numeric_policy": UTILITY_NUMERIC_POLICY,
        "selection_split": "frozen_inner_validation_only",
        "claim_eligible": False,
    }
    if isinstance(config.get("mmw_all_weather_protocol"), dict):
        config["mmw_all_weather_protocol"]["split_tag"] = PROTOCOL_ID
        config["mmw_all_weather_protocol"].pop("training_profile", None)
        config["mmw_all_weather_protocol"].pop("router_architecture_profile", None)
    return config


def run_plan(path: Path, *, min_free_mib: int, poll_seconds: float) -> int:
    manifest = _read_json(path)
    free = _gpu_free_memory()
    blocked = {
        int(job["gpu"]): free.get(int(job["gpu"]), 0)
        for job in manifest["jobs"]
        if free.get(int(job["gpu"]), 0) < int(min_free_mib)
    }
    if blocked:
        raise RuntimeError(f"Dynamic Router GPUs do not meet the free-memory threshold: {blocked}")
    running: list[tuple[subprocess.Popen[Any], Any, dict[str, Any]]] = []
    for job in manifest["jobs"]:
        if _completed_run(job):
            job["status"] = "done"
            continue
        log_path = Path(job["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        command = [
            "conda",
            "run",
            "-n",
            "kd_mm_beam",
            "--no-capture-output",
            "kd-sensing-train",
            "--config",
            str(job["config_path"]),
        ]
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(job["gpu"]),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "OMP_NUM_THREADS": "4",
                "PYTHONUNBUFFERED": "1",
            }
        )
        handle.write(f"[{_now()}] GPU{job['gpu']}: {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job.update({"status": "running", "pid": process.pid, "start_time": _now()})
        running.append((process, handle, job))
    manifest.update(status="running" if running else "complete", launched_at=_now())
    _write_json(path, manifest)
    while running:
        for process, handle, job in list(running):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job.update(
                {
                    "status": "done" if code == 0 and _completed_run(job) else "failed",
                    "return_code": int(code),
                    "end_time": _now(),
                }
            )
            running.remove((process, handle, job))
            _write_json(path, manifest)
        time.sleep(float(poll_seconds))
    manifest.update(
        status="complete" if all(job["status"] == "done" for job in manifest["jobs"]) else "failed",
        completed_at=_now(),
    )
    _write_json(path, manifest)
    return 0 if manifest["status"] == "complete" else 1


def _validate_source_config(config: Mapping[str, Any]) -> None:
    primary = config.get("model", {}).get("primary", {})
    loss = config.get("loss", {}).get("u_mask_beam_jepa", {})
    screen = config.get("mmw_router_utility_screen", {})
    if (
        primary.get("type") != "u_mask_beam_jepa"
        or primary.get("fusion_type") != "supervised_router"
        or primary.get("head_type") != "prototype"
        or bool(primary.get("router_use_pattern_features", True))
        or screen.get("candidate_id") != "CurrentControl"
        or screen.get("protocol") != "mmw_router_expected_utility_screen_v3"
        or screen.get("claim_eligible") is not False
        or float(loss.get("router_oracle_weight", -1.0)) != 0.1
    ):
        raise ValueError("Dynamic Router screen requires the frozen CurrentControl inner-only source config.")


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


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


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
