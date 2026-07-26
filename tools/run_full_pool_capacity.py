#!/usr/bin/env python3
"""Run the protocol-bound Full-pool U0 and A0--A7 two-stage pipeline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from kd_sensing.baselines.prototype_decision_adapter import (
    ADAPTER_LOSS_PROFILES,
    EXPERIMENTS,
    MASKS,
    NON_FULL_MASKS,
    _adapter,
    _amp,
    _inputs,
    checkpoint_normalization_overrides,
    dataset_sample_ids,
    generate_mask_schedule,
    load_frozen_u0,
    preflight,
    run_experiment,
    sha256_json,
    stratified_mask_folds,
    write_json,
)
from kd_sensing.baselines.full_pool_common import now
from kd_sensing.config import load_config
from kd_sensing.config.io import dump_config
from kd_sensing.data.mmw.full_pool_protocol import (
    build_full_pool_protocol,
    protocol_dataset_domains,
)
from kd_sensing.data.transform_ops.gps import read_gps_latlon
from kd_sensing.data.transform_ops.image import (
    build_rgb_imagenet_transform,
    image_derived_cache_path,
    load_rgb_imagenet_cache_frame,
    load_rgb_imagenet_frames,
)
from kd_sensing.data.transform_ops.lidar import (
    lidar_cache_path,
    load_lidar_bev_sequence,
    parameterized_lidar_cache_dir,
    validate_lidar_cache_metadata,
)
from kd_sensing.engine.batch import prepare_labels
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.models.missing_decision_adapter import FrozenU0DecisionAdapter
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.checkpoint import checkpoint_file_digest, load_torch_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/full_pool_capacity"
POLL_SECONDS = 600
CONVERGENCE_EPOCHS = 20
LMDB_THROUGHPUT_MARGIN = 1.5
TRAINING_LOSS_EARLY_STOPPING = {
    "enabled": True,
    "min_epochs": 8,
    "patience": 3,
    "relative_min_delta": 0.005,
}
STAGE2_GPUS = {
    "a0": 0,
    "a1": 0,
    "a3": 0,
    "a2": 4,
    "a4": 4,
    "a6": 6,
    "a5": 6,
    "a7": 7,
}
ALLOWED_PHYSICAL_GPUS = frozenset({0, 4, 6, 7})
ADBA_SURROGATE_JOBS = {
    "b1": ("a1", 0),
    "b4": ("a4", 4),
    "b6": ("a6", 6),
    "b7": ("a7", 7),
}
MASK_BIAS_ABLATION_EPOCHS = 8
MASK_BIAS_ALL_SEEN_JOBS = {
    "global_bias": ("global_bias", 0),
    "mask_lookup": ("mask_lookup", 4),
}
MASK_BIAS_UNSEEN_JOBS = {
    "mask_mlp": ("a1", 6),
    "factorized_bias": ("factorized_bias", 7),
}
CIRCULAR_TRANSPORT_EPOCHS = 8
CIRCULAR_TRANSPORT_JOBS = {
    "circular_transport": ("circular_transport", 0),
    "factorized_all_seen": ("factorized_all_seen", 4),
}
ERROR_PATTERN = re.compile(r"(?:Traceback|\bNaN\b|\bOOM\b|out of memory|\bError\b)", re.IGNORECASE)
PROTOTYPE_COLLAPSE_THRESHOLD = 0.95
GPU_LAUNCH_MAX_USED_MIB = 1024


def _running_adapter_process(
    adapter: str,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any] | None:
    """Find the unique live conda wrapper for an incomplete Adapter run."""
    candidates = []
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = [part.decode(errors="replace") for part in (process / "cmdline").read_bytes().split(b"\0") if part]
        except (OSError, PermissionError):
            continue
        if not any(Path(arg).name == "conda" for arg in args[:3]):
            continue
        try:
            adapter_index = args.index("--adapter")
        except ValueError:
            continue
        if adapter_index + 1 < len(args) and args[adapter_index + 1] == adapter:
            candidates.append(int(process.name))
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous live Stage 2 processes for {adapter}: {candidates}")
    if not candidates:
        return None
    return {"name": adapter, "status": "running", "pid": candidates[0]}


def apply_reference_u0_profile(cfg: dict[str, Any], *, epochs: int) -> None:
    """Keep the Full-pool capacity run comparable to the clean U0 reference."""
    cfg["model"]["primary"]["router_use_pattern_features"] = False
    cfg["temporal_missing"]["seed"] = 1
    cfg["training"].update(
        epochs=int(epochs),
        max_epochs=int(epochs),
        lr=5e-4,
        weight_decay=3e-4,
        resume=False,
        allow_tf32=False,
        cudnn_benchmark=False,
        amp={"enabled": True, "dtype": "bfloat16", "grad_scaler": False},
        optimizer={"type": "adamw"},
        checkpoint_selection="last",
        early_stopping={
            **TRAINING_LOSS_EARLY_STOPPING,
            "enabled": int(epochs) > 1,
            "monitor": "train_task_loss",
        },
    )
    cfg["training"].setdefault("validation", {})["interval_epochs"] = int(epochs)
    cfg["scheduler"] = {"type": "cosine_warm_restarts", "T_0": 40, "T_mult": 1, "eta_min": 1e-6}


def prepare_config(
    output_root: Path,
    *,
    epochs: int,
    run_name: str,
    normalization_artifacts: dict[str, Any] | None = None,
) -> Path:
    protocol = build_full_pool_protocol(output_root)
    cfg = load_config(ROOT / "configs/mmw/u0.yaml")
    cfg["experiment"].update(name="FullPool_U0_seed1", seed=1, device="auto")
    cfg["data"]["dataset"].update(
        domains=protocol_dataset_domains(protocol),
        portion=1.0,
        frame_cache_root=str((ROOT / "outputs/cache/MMW").resolve()),
        frame_cache_strict=True,
        gps_coordinate_cache_root=str((output_root / "cache/gps_coordinates").resolve()),
    )
    if normalization_artifacts:
        cfg["data"]["normalization_artifacts"] = dict(normalization_artifacts)
    cfg["data"]["dataloader"].update(
        train_batch_size=64,
        validation_batch_size=64,
        test_batch_size=64,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    cfg["data_protocol"] = {
        "mode": protocol["mode"],
        "path": str(output_root / "protocol/split_manifest.json"),
        "audit_report": str(output_root / "protocol/split_audit.json"),
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "train_role": protocol["train_role"],
        "validation_role": protocol["validation_role"],
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
    }
    apply_reference_u0_profile(cfg, epochs=epochs)
    cfg["training"]["timing"] = {
        "enabled": True,
        "profile": "host",
        "log_interval": 10,
        "slow_batch_seconds": 20.0,
    }
    cfg["training"]["final_test"] = {"enabled": False}
    cfg["output"].update(
        dir=str(output_root),
        run_name=run_name,
        overwrite=False,
        progress={"enabled": False},
        tensorboard={"enabled": False},
    )
    target = output_root / "protocol" / f"{run_name}_config.yaml"
    dump_config(cfg, target)
    audit_u0_config(cfg, output_root / "protocol" / f"{run_name}_structure_preflight.json")
    return target


def audit_u0_config(cfg: dict[str, Any], output_path: Path) -> None:
    primary = cfg["model"]["primary"]
    loss = cfg["loss"]["u_mask_beam_jepa"]
    checks = {
        "model_type_is_current_u0": primary.get("type") == "u_mask_beam_jepa",
        "modalities_are_current_four": primary.get("modalities") == ["image", "radar", "gps", "lidar"],
        "fusion_is_current": primary.get("fusion_type") == "supervised_router",
        "prototype_bank_enabled": primary.get("head_type") == "prototype",
        "prototype_used_at_inference": primary.get("head_type") == "prototype",
        "prototype_alignment_enabled": loss.get("use_beam_prototype_alignment") is True,
        "mixed_mask_training_enabled": cfg["temporal_missing"].get("mode") == "balanced_pattern_schedule",
        "router_pattern_bias_disabled": primary.get("router_use_pattern_features") is False,
        "temporal_missing_seed_is_reference": cfg["temporal_missing"].get("seed") == 1,
        "formal_training_never_resumes_probe": cfg["training"].get("resume") is False,
        "max_epochs_matches_epochs": cfg["training"].get("max_epochs") == cfg["training"].get("epochs"),
        "amp_is_bfloat16_without_scaler": cfg["training"].get("amp")
        == {"enabled": True, "dtype": "bfloat16", "grad_scaler": False},
        "optimizer_is_reference_adamw": cfg["training"].get("optimizer", {}).get("type") == "adamw",
        "weight_decay_is_reference": cfg["training"].get("weight_decay") == 3e-4,
        "scheduler_is_reference": cfg.get("scheduler")
        == {"type": "cosine_warm_restarts", "T_0": 40, "T_mult": 1, "eta_min": 1e-6},
        "outer_test_disabled": cfg["data_protocol"].get("outer_test_enabled") is False,
        "strict_frame_cache_enabled": cfg["data"]["dataset"].get("frame_cache_strict") is True,
        "gps_coordinate_cache_bound": bool(cfg["data"]["dataset"].get("gps_coordinate_cache_root")),
        "training_loss_early_stopping_pre_registered": int(cfg["training"].get("epochs", 0)) == 1
        or cfg["training"].get("early_stopping")
        == {**TRAINING_LOSS_EARLY_STOPPING, "monitor": "train_task_loss"},
        "checkpoint_selection_is_last": cfg["training"].get("checkpoint_selection") == "last",
    }
    forbidden = [name for name in ("bcacl", "cmsbl", "teacher", "moe", "private", "shared") if name in cfg]
    payload = {
        "status": "passed" if all(checks.values()) and not forbidden else "failed",
        "checks": checks,
        "forbidden_top_level_sections": forbidden,
        "prototype_restoration_enabled": False,
        "prototype_residual_diagnostic_enabled": True,
        "separate_restoration_module_note": "Current U0 uses its 64-Beam prototype bank directly as the inference head; it has no separate restoration module.",
        "outer_test_accessed": False,
    }
    write_json(output_path, payload)
    if payload["status"] != "passed":
        raise ValueError(f"Full-data U0 config audit failed: {payload}")


def prototype_health(state_dict: dict[str, Any], *, threshold: float = PROTOTYPE_COLLAPSE_THRESHOLD) -> dict[str, Any]:
    prototypes = state_dict.get("prototype_bank.prototypes")
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise ValueError("U0 checkpoint is missing the 2D prototype_bank.prototypes tensor.")
    vectors = prototypes.detach().float().cpu()
    normalized = F.normalize(vectors, dim=1)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~torch.eye(len(vectors), dtype=torch.bool)]
    finite = bool(torch.isfinite(vectors).all() and torch.isfinite(off_diagonal).all())
    mean_cosine = float(off_diagonal.mean())
    passed = finite and len(vectors) == 64 and mean_cosine < float(threshold)
    return {
        "status": "passed" if passed else "failed",
        "prototype_count": int(len(vectors)),
        "prototype_dimension": int(vectors.shape[1]),
        "matrix_rank": int(torch.linalg.matrix_rank(vectors).item()),
        "all_finite": finite,
        "off_diagonal_cosine": {
            "mean": mean_cosine,
            "std": float(off_diagonal.std(unbiased=False)),
            "min": float(off_diagonal.min()),
            "max": float(off_diagonal.max()),
        },
        "collapse_threshold": float(threshold),
    }


def audit_prototype_checkpoint(checkpoint: Path, output_path: Path) -> dict[str, Any]:
    payload = load_torch_payload(checkpoint, map_location="cpu")
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError(f"U0 checkpoint has no state_dict: {checkpoint}")
    result = {
        **prototype_health(state_dict),
        "checkpoint": str(checkpoint),
        "outer_test_accessed": False,
    }
    write_json(output_path, result)
    if result["status"] != "passed":
        raise ValueError(f"Prototype-collapse audit failed closed: {result}")
    return result


def benchmark(config: Path, checkpoint: Path, expected_sha256: str, output: Path, u0_epoch_wall: float) -> dict[str, Any]:
    started = time.monotonic()
    cfg, _ = preflight(config, checkpoint, expected_sha256=expected_sha256)
    load_started = time.monotonic()
    loaders = build_dataloaders(
        cfg,
        normalization_overrides=checkpoint_normalization_overrides(checkpoint),
    )
    dataloader_seconds = time.monotonic() - load_started
    loader_batches = min(64, max(0, len(loaders["train"]) - 4))
    loader_samples = 0
    loader_started = None
    for batch_index, batch in enumerate(loaders["train"]):
        if batch_index == 4:
            loader_started = time.monotonic()
        if batch_index >= 4:
            loader_samples += len(batch["target_beam"])
        if batch_index + 1 >= loader_batches + 4:
            break
    loader_wall = time.monotonic() - loader_started if loader_started is not None else 0.0
    loader_samples_per_second = loader_samples / max(loader_wall, 1e-9)
    timing_csv = Path(cfg["output"]["dir"]) / str(cfg["output"]["run_name"]) / "timing.csv"
    step_times: list[float] = []
    if timing_csv.is_file():
        with timing_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = float(row["step_time"])
                if np.isfinite(value) and value > 0:
                    step_times.append(value)
    model_step_seconds = float(np.median(step_times)) if step_times else None
    model_samples_per_second = (
        float(cfg["data"]["dataloader"]["train_batch_size"]) / model_step_seconds
        if model_step_seconds is not None
        else None
    )
    required_loader_rate = (
        model_samples_per_second * LMDB_THROUGHPUT_MARGIN
        if model_samples_per_second is not None
        else None
    )
    lmdb_required = bool(
        required_loader_rate is not None and loader_samples_per_second < required_loader_rate
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_started = time.monotonic()
    model, _ = load_frozen_u0(cfg, checkpoint, device)
    checkpoint_load_seconds = time.monotonic() - checkpoint_started
    wrapper = FrozenU0DecisionAdapter(model, None).to(device).eval()
    validation_started = time.monotonic()
    validation_samples = 0
    with torch.no_grad():
        for batch in loaders["validation"]:
            size = len(batch["target_beam"])
            mask = torch.ones(size, 4, dtype=torch.bool, device=device)
            with _amp(device):
                wrapper(**_inputs(batch, cfg, device), missing_mask=mask)
            validation_samples += size
    if device.type == "cuda":
        torch.cuda.synchronize()
    one_mask_validation_seconds = time.monotonic() - validation_started

    adapter = _adapter(EXPERIMENTS["a6"], model, device)
    adapter_wrapper = FrozenU0DecisionAdapter(model, adapter).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    short_batches = min(32, len(loaders["train"]))
    short_samples = 0
    short_started = time.monotonic()
    for batch_index, batch in enumerate(loaders["train"]):
        if batch_index >= short_batches:
            break
        size = len(batch["target_beam"])
        raw = tuple(int(bit) for bit in f"{1 + batch_index % 14:04b}")
        mask = torch.tensor([raw] * size, dtype=torch.bool, device=device)
        target = prepare_labels(dict(batch), num_pred=1, device=device)[:, 0]
        optimizer.zero_grad(set_to_none=True)
        with _amp(device):
            result = adapter_wrapper(**_inputs(batch, cfg, device), missing_mask=mask)
            logits = result["logits"][:, 0, :]
            delta = result["delta_logits"]
            loss = F.cross_entropy(logits.float(), target) + 1e-4 * delta.float().pow(2).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
        short_samples += size
    if device.type == "cuda":
        torch.cuda.synchronize()
    short_seconds = time.monotonic() - short_started
    adapter_epoch_seconds = short_seconds / max(1, short_batches) * len(loaders["train"])
    result = {
        "schema_version": 1,
        "generated_at": now(),
        "outer_test_accessed": False,
        "u0_epoch_wall_seconds": float(u0_epoch_wall),
        "dataloader_and_train_scaler_seconds": dataloader_seconds,
        "cached_loader_benchmark": {
            "warmup_batches": 4,
            "timed_batches": loader_batches,
            "samples": loader_samples,
            "seconds": loader_wall,
            "samples_per_second": loader_samples_per_second,
        },
        "lmdb_decision": {
            "required": lmdb_required,
            "margin": LMDB_THROUGHPUT_MARGIN,
            "model_step_seconds": model_step_seconds,
            "model_samples_per_second": model_samples_per_second,
            "required_loader_samples_per_second": required_loader_rate,
            "policy": "build unique-frame sharded LMDB only when cached loader rate is below 1.5x model consumption",
        },
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "one_mask_validation_seconds": one_mask_validation_seconds,
        "estimated_15_mask_validation_seconds": one_mask_validation_seconds * 15,
        "validation_samples": validation_samples,
        "validation_samples_per_second": validation_samples / max(one_mask_validation_seconds, 1e-9),
        "adapter_short_segment": {
            "batches": short_batches,
            "samples": short_samples,
            "seconds": short_seconds,
            "estimated_epoch_seconds": adapter_epoch_seconds,
        },
        "benchmark_total_seconds": time.monotonic() - started,
    }
    write_json(output, result)
    for loader in loaders.values():
        shutdown_dataloader_workers(loader)
    return result


def choose_epochs(timing: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "wall_budget_seconds": None,
        "budget_policy": "pre_registered_training_loss_early_stopping",
        "elapsed_before_epoch_selection_seconds": elapsed,
        "remaining_before_epoch_selection_seconds": None,
        "u0_epochs": CONVERGENCE_EPOCHS,
        "u0_additional_epochs": 0,
        "adapter_epochs": CONVERGENCE_EPOCHS,
        "adapter_max_optimizer_steps": CONVERGENCE_EPOCHS * 579,
        "early_stopping": dict(TRAINING_LOSS_EARLY_STOPPING),
        "basis": {
            "reason": "User requested early stopping after limiting this run to three physical GPUs.",
            "validation_used_for_epoch_selection": False,
            "shared_adapter_max_epochs": True,
            "shared_adapter_early_stopping_rule": True,
            "training_curve_audit_required": True,
        },
    }


def write_cache_manifest(output_root: Path, protocol: dict[str, Any]) -> Path:
    protocol_hash = json.loads((output_root / "protocol/protocol_hash.json").read_text(encoding="utf-8"))
    domains = [
        {
            "domain": item["id"],
            "source_csv": item["source_csv"],
            "source_csv_sha256": item["source_csv_sha256"],
            "train_csv": item["train_split"],
            "train_csv_sha256": item["train_csv_sha256"],
            "validation_csv": item["validation_split"],
            "validation_csv_sha256": item["validation_csv_sha256"],
        }
        for item in protocol["domains"]
    ]
    path = output_root / "cache/cache_manifest.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "generated_at": now(),
            "source_csv_hashes": {item["domain"]: item["source_csv_sha256"] for item in domains},
            "split_manifest_sha256": protocol_hash["files"]["split_manifest.json"],
            "split_manifest_fingerprint": protocol["protocol_fingerprint"],
            "augmentation_code_sha256": protocol_hash["augmentation_code_sha256"],
            "domains": domains,
            "outer_test_accessed": False,
        },
    )
    return path


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_paths(path: str | Path, prefixes: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [
            name
            for name in (reader.fieldnames or [])
            if any(re.fullmatch(rf"{re.escape(prefix)}\d+", name) for prefix in prefixes)
        ]
        for row in reader:
            values.update(str(row[name]).strip() for name in columns if str(row.get(name, "")).strip())
    return values


def _completed_u0_epochs(run_dir: Path) -> int:
    path = run_dir / "metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"U0 metrics contain no completed epochs: {path}")
    return max(int(float(row["epoch"])) for row in rows)


def _read_gps_coordinate_task(task: tuple[int, str, str]) -> tuple[int, str, tuple[float, float]]:
    domain_index, data_root, rel_path = task
    coordinates = read_gps_latlon(data_root, rel_path)
    return domain_index, rel_path, (float(coordinates[0]), float(coordinates[1]))


def build_gps_coordinate_caches(output_root: Path, protocol: dict[str, Any]) -> Path:
    cache_root = output_root / "cache/gps_coordinates"
    cache_root.mkdir(parents=True, exist_ok=True)
    code_sha256 = _sha256_file(ROOT / "src/kd_sensing/data/transform_ops/gps.py")
    records: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for domain in protocol["domains"]:
        target = cache_root / f"{domain['condition']}__{domain['scene']}.npz"
        sidecar = target.with_suffix(target.suffix + ".json")
        expected = {
            "schema_version": 1,
            "domain_id": domain["id"],
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "train_csv_sha256": domain["train_csv_sha256"],
            "validation_csv_sha256": domain["validation_csv_sha256"],
            "transform_code_sha256": code_sha256,
        }
        if target.is_file() and sidecar.is_file():
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))
            if all(recorded.get(key) == value for key, value in expected.items()) and recorded.get(
                "artifact_sha256"
            ) == _sha256_file(target):
                records[domain["id"]] = recorded | {"reused": True}
                continue

        paths = sorted(
            _csv_paths(domain["train_split"], ("gps", "bs_gps"))
            | _csv_paths(domain["validation_split"], ("gps", "bs_gps"))
        )
        pending.append(
            {
                "domain": domain,
                "target": target,
                "sidecar": sidecar,
                "expected": expected,
                "paths": paths,
                "coordinates": {},
            }
        )

    task_count = sum(len(item["paths"]) for item in pending)
    workers = min(90, max(1, int(os.cpu_count() or 1)), max(1, task_count))
    tasks = (
        (domain_index, str(item["domain"]["data_root"]), rel_path)
        for domain_index, item in enumerate(pending)
        for rel_path in item["paths"]
    )
    if task_count:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for domain_index, rel_path, coordinates in executor.map(
                _read_gps_coordinate_task, tasks, chunksize=32
            ):
                pending[domain_index]["coordinates"][rel_path] = coordinates

    for item in pending:
        domain = item["domain"]
        target = item["target"]
        sidecar = item["sidecar"]
        paths = item["paths"]
        array = np.asarray([item["coordinates"][path] for path in paths], dtype=np.float64)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, paths=np.asarray(paths), coordinates=array)
        temporary.replace(target)
        payload = {
            **item["expected"],
            "generated_at": now(),
            "artifact": str(target),
            "artifact_sha256": _sha256_file(target),
            "coordinate_count": len(paths),
            "parallel_workers": workers,
            "outer_test_accessed": False,
            "reused": False,
        }
        write_json(sidecar, payload)
        records[domain["id"]] = payload

    domains = [records[domain["id"]] for domain in protocol["domains"]]
    write_json(
        cache_root / "manifest.json",
        {
            "schema_version": 1,
            "generated_at": now(),
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "transform_code_sha256": code_sha256,
            "domain_count": len(domains),
            "coordinate_count": sum(int(item["coordinate_count"]) for item in domains),
            "domains": domains,
            "outer_test_accessed": False,
        },
    )
    return cache_root


def audit_frame_cache_reuse(output_root: Path, protocol: dict[str, Any]) -> Path:
    frame_cache_root = (ROOT / "outputs/cache/MMW").resolve()
    by_condition: dict[str, dict[str, set[str]]] = {}
    for domain in protocol["domains"]:
        resources = by_condition.setdefault(domain["condition"], {"image": set(), "lidar": set()})
        for split in (domain["train_split"], domain["validation_split"]):
            resources["image"].update(_csv_paths(split, ("camera",)))
            resources["lidar"].update(_csv_paths(split, ("lidar",)))

    conditions: dict[str, Any] = {}
    for condition, resources in sorted(by_condition.items()):
        data_root = ROOT / "dataset/MMW" / condition
        image_root = frame_cache_root / condition / "image_derived"
        lidar_root = parameterized_lidar_cache_dir(frame_cache_root / condition / "lidar_bev")
        validate_lidar_cache_metadata(lidar_root)

        def image_entry(rel_path: str) -> str | None:
            path, expected = image_derived_cache_path(data_root, rel_path, cache_dir=image_root)
            sidecar = path.with_suffix(path.suffix + ".json")
            if not path.is_file() or not sidecar.is_file():
                return rel_path
            try:
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return rel_path
            return (
                rel_path
                if metadata.get("version") != "image_derived_cache_metadata_v1"
                or any(metadata.get(key) != value for key, value in expected.items())
                else None
            )

        with ThreadPoolExecutor(max_workers=30, thread_name_prefix=f"image-cache-{condition}") as executor:
            image_failures = [item for item in executor.map(image_entry, sorted(resources["image"])) if item]
        lidar_failures = [
            rel_path
            for rel_path in sorted(resources["lidar"])
            if not lidar_cache_path(lidar_root, rel_path).is_file()
        ]

        sample_domain = next(item for item in protocol["domains"] if item["condition"] == condition)
        with Path(sample_domain["train_split"]).open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        transform = build_rgb_imagenet_transform((224, 224))
        image_rel = row["camera1"]
        cached_image = load_rgb_imagenet_cache_frame(data_root, image_rel, cache_dir=image_root).numpy()
        raw_image = load_rgb_imagenet_frames(
            data_root, [image_rel], 1, transform, image_size=(224, 224)
        )[0].numpy()
        lidar_rel = row["lidar1"]
        cached_lidar = load_lidar_bev_sequence(
            data_root,
            [lidar_rel],
            seq_len=1,
            cache_dir=lidar_root,
            strict_cache=True,
        )[0]
        raw_lidar = load_lidar_bev_sequence(data_root, [lidar_rel], seq_len=1)[0]
        conditions[condition] = {
            "unique_image_count": len(resources["image"]),
            "image_failure_count": len(image_failures),
            "image_failure_examples": image_failures[:10],
            "unique_lidar_count": len(resources["lidar"]),
            "lidar_failure_count": len(lidar_failures),
            "lidar_failure_examples": lidar_failures[:10],
            "image_max_abs_diff": float(np.max(np.abs(cached_image - raw_image))),
            "image_array_equal": bool(np.array_equal(cached_image, raw_image)),
            "lidar_max_abs_diff": float(np.max(np.abs(cached_lidar - raw_lidar))),
            "lidar_array_equal": bool(np.array_equal(cached_lidar, raw_lidar)),
        }

    passed = all(
        item["image_failure_count"] == 0
        and item["lidar_failure_count"] == 0
        and item["image_array_equal"]
        and item["lidar_array_equal"]
        for item in conditions.values()
    )
    path = output_root / "cache/frame_cache_reuse_audit.json"
    protocol_hash = json.loads((output_root / "protocol/protocol_hash.json").read_text(encoding="utf-8"))
    write_json(
        path,
        {
            "schema_version": 1,
            "status": "passed" if passed else "failed",
            "generated_at": now(),
            "frame_cache_root": str(frame_cache_root),
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "split_manifest_sha256": protocol_hash["files"]["split_manifest.json"],
            "source_csv_hashes": {item["id"]: item["source_csv_sha256"] for item in protocol["domains"]},
            "transform_code_sha256": {
                "image": _sha256_file(ROOT / "src/kd_sensing/data/transform_ops/image.py"),
                "lidar": _sha256_file(ROOT / "src/kd_sensing/data/transform_ops/lidar.py"),
            },
            "conditions": conditions,
            "outer_test_accessed": False,
        },
    )
    if not passed:
        raise ValueError(f"Full-pool frame cache reuse audit failed: {path}")
    return path


def prepare_adapter_schedule(config: Path, checkpoint: Path, expected_sha256: str, epochs: int, output_root: Path) -> Path:
    cfg, _ = preflight(config, checkpoint, expected_sha256=expected_sha256)
    loaders = build_dataloaders(
        cfg,
        normalization_overrides=checkpoint_normalization_overrides(checkpoint),
    )
    schedule = generate_mask_schedule(dataset_sample_ids(loaders["train"].dataset), epochs=epochs, seed=1)
    path = output_root / "protocol/adapter_mask_schedule_seed1.json"
    write_json(path, schedule)
    for loader in loaders.values():
        shutdown_dataloader_workers(loader)
    return path


def run_adapter_child(args: argparse.Namespace) -> int:
    run_experiment(
        args.adapter,
        args.config.resolve(),
        args.checkpoint.resolve(),
        args.schedule.resolve(),
        args.run_dir.resolve(),
        epochs=0 if args.adapter == "a0" else args.epochs,
        expected_u0_sha256=args.expected_sha256,
        early_stopping=TRAINING_LOSS_EARLY_STOPPING,
        loss_profile=args.loss_profile,
    )
    return 0


def run_jobs(stage: str, jobs: list[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    runtime = output_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    active: list[tuple[dict[str, Any], subprocess.Popen, Any]] = []
    for job in pending:
        job.update(status="pending", return_code=None)

    def launch_available() -> list[tuple[dict[str, Any], subprocess.Popen, Any]]:
        busy = {int(job["gpu"]) for job, process, _ in active if process.poll() is None}
        launched = []
        for job in list(pending):
            gpu = int(job["gpu"])
            if gpu in busy:
                continue
            gpu_status = _gpu_status(gpu)
            memory_used = gpu_status.get("memory_used_mib")
            if memory_used is None or int(memory_used) > GPU_LAUNCH_MAX_USED_MIB:
                job.update(status="waiting_for_gpu", gpu_status=gpu_status)
                continue
            gpu_uuid = physical_gpu_uuid(gpu)
            busy.add(gpu)
            pending.remove(job)
            job["log_path"].parent.mkdir(parents=True, exist_ok=True)
            handle = job["log_path"].open("w", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                CUDA_DEVICE_ORDER="PCI_BUS_ID",
                CUDA_VISIBLE_DEVICES=gpu_uuid,
                PYTHONUNBUFFERED="1",
            )
            try:
                process = subprocess.Popen(
                    job["command"],
                    cwd=ROOT,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                handle.close()
                raise
            job.update(gpu_uuid=gpu_uuid, pid=process.pid, start_time=now(), status="running")
            item = (job, process, handle)
            active.append(item)
            launched.append(item)
        return launched

    launched = launch_available()
    _write_runtime_manifest(output_root, stage, jobs)
    poll_jobs(stage, launched, output_root, event="started")
    while active or pending:
        time.sleep(POLL_SECONDS)
        poll_jobs(stage, active, output_root, event="scheduled_poll")
        completed = [item for item in active if item[1].poll() is not None]
        for job, process, handle in completed:
            poll_jobs(stage, [(job, process, handle)], output_root, event="completed")
            job.update(
                status="completed" if process.returncode == 0 else "failed",
                return_code=process.returncode,
                end_time=now(),
            )
            handle.close()
            active.remove((job, process, handle))
        launched = launch_available()
        _write_runtime_manifest(output_root, stage, jobs)
        if launched:
            poll_jobs(stage, launched, output_root, event="started")
    return jobs


def physical_gpu_uuid(gpu: int) -> str:
    if int(gpu) not in ALLOWED_PHYSICAL_GPUS:
        allowed = ", ".join(str(item) for item in sorted(ALLOWED_PHYSICAL_GPUS))
        raise ValueError(f"Full-pool jobs only permit physical GPUs {allowed}; refusing GPU {gpu}.")
    result = subprocess.run(
        ["nvidia-smi", f"--id={int(gpu)}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    uuid = result.stdout.strip()
    if result.returncode != 0 or not uuid.startswith("GPU-") or "\n" in uuid:
        raise RuntimeError(f"Cannot resolve physical GPU {gpu} UUID: {result.stdout.strip()}")
    return uuid


def poll_jobs(stage: str, active: list[tuple[dict[str, Any], subprocess.Popen, Any]], output_root: Path, *, event: str) -> None:
    for job, process, _ in active:
        tail = _tail(job["log_path"], 30)
        gpu = _gpu_status(job["gpu"])
        checkpoint = Path(job.get("checkpoint_watch", ""))
        record = {
            "timestamp": now(),
            "stage": stage,
            "event": event,
            "task": job["name"],
            "gpu": job["gpu"],
            "pid": process.pid,
            "pid_exists": process.poll() is None,
            "return_code": process.poll(),
            "gpu_status": gpu,
            "progress": _progress_summary(tail, job),
            "checkpoint_last_modified": (
                datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat() if checkpoint.is_file() else None
            ),
            "error_keyword_detected": bool(ERROR_PATTERN.search(tail)),
            "log_tail": tail.splitlines()[-30:],
            "elapsed_seconds": time.time() - job.get("started_epoch", time.time()),
        }
        with (output_root / "runtime/poll_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        with (output_root / f"runtime/gpu{job['gpu']}_status.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _gpu_status(gpu: int) -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    values = [item.strip() for item in result.stdout.strip().split(",")]
    return {
        "utilization_percent": int(values[0]) if len(values) == 3 and values[0].isdigit() else None,
        "memory_used_mib": int(values[1]) if len(values) == 3 and values[1].isdigit() else None,
        "memory_total_mib": int(values[2]) if len(values) == 3 and values[2].isdigit() else None,
        "raw": result.stdout.strip(),
    }


def _tail(path: Path, count: int) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-count:])


def _progress_summary(tail: str, job: dict[str, Any]) -> dict[str, Any]:
    epoch = loss = None
    for line in reversed(tail.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("epoch") is not None:
            epoch = value.get("epoch")
            loss = value.get("loss", value.get("train_loss"))
            break
    return {"epoch": epoch, "loss": loss, "expected_epochs": job.get("expected_epochs")}


def _write_runtime_manifest(output_root: Path, stage: str, jobs: list[dict[str, Any]]) -> None:
    serializable = []
    for job in jobs:
        serializable.append(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in job.items()
                if key not in {"started_epoch"}
            }
        )
    write_json(output_root / "runtime/orchestration_manifest.json", {"stage": stage, "outer_test_accessed": False, "jobs": serializable})


def orchestrate(output_root: Path) -> int:
    started_epoch = time.time()
    started_monotonic = time.monotonic()
    protocol = build_full_pool_protocol(output_root)
    write_cache_manifest(output_root, protocol)
    build_gps_coordinate_caches(output_root, protocol)
    audit_frame_cache_reuse(output_root, protocol)
    timing_config = prepare_config(output_root, epochs=1, run_name="u0_timing_probe")
    u0_checkpoint = output_root / "u0_timing_probe/checkpoints/last.pth"
    u0_log = output_root / "runtime/gpu4_u0_timing.log"
    timing_job = {
        "name": "u0_timing_epoch",
        "gpu": 4,
        "command": ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "kd-sensing-train", "--config", str(timing_config)],
        "log_path": u0_log,
        "checkpoint_watch": str(u0_checkpoint),
        "expected_epochs": 1,
        "started_epoch": time.time(),
    }
    timing_started = time.monotonic()
    results = run_jobs("stage1_timing_epoch", [timing_job], output_root)
    u0_epoch_wall = time.monotonic() - timing_started
    if results[0]["return_code"] != 0 or not u0_checkpoint.is_file():
        return 1
    audit_prototype_checkpoint(u0_checkpoint, output_root / "protocol/u0_timing_probe_health.json")
    checkpoint_sha, _ = checkpoint_file_digest(u0_checkpoint)

    benchmark_path = output_root / "timing_estimate.json"
    benchmark_command = [
        "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python", str(Path(__file__).resolve()),
        "--benchmark", "--config", str(timing_config), "--checkpoint", str(u0_checkpoint),
        "--expected-sha256", checkpoint_sha, "--timing-output", str(benchmark_path),
        "--u0-epoch-wall", str(u0_epoch_wall),
    ]
    benchmark_job = {
        "name": "post_u0_timing",
        "gpu": 4,
        "command": benchmark_command,
        "log_path": output_root / "runtime/gpu4_post_u0_timing.log",
        "checkpoint_watch": str(u0_checkpoint),
        "expected_epochs": None,
        "started_epoch": time.time(),
    }
    benchmark_started = time.monotonic()
    results = run_jobs("timing_benchmark", [benchmark_job], output_root)
    benchmark_wall = time.monotonic() - benchmark_started
    if results[0]["return_code"] != 0:
        return 1
    timing = json.loads(benchmark_path.read_text(encoding="utf-8"))
    selection = choose_epochs(timing, time.monotonic() - started_monotonic)
    timing.update(epoch_selection=selection)
    write_json(benchmark_path, timing)
    if timing["lmdb_decision"]["required"]:
        write_json(
            output_root / "runtime/lmdb_required.json",
            {
                **timing["lmdb_decision"],
                "status": "formal_training_blocked_until_unique_frame_lmdb_is_built",
                "outer_test_accessed": False,
            },
        )
        return 2

    return run_formal_stages(
        output_root,
        timing,
        selection,
        started_epoch=started_epoch,
        started_monotonic=started_monotonic,
        prior_wall_seconds=0.0,
        u0_epoch_wall=u0_epoch_wall,
        benchmark_wall=benchmark_wall,
    )


def continue_after_benchmark(output_root: Path) -> int:
    timing = json.loads((output_root / "timing_estimate.json").read_text(encoding="utf-8"))
    if timing.get("lmdb_decision", {}).get("required"):
        raise ValueError("Cached-loader benchmark requires the unique-frame LMDB before formal training.")
    selection = choose_epochs(timing, 0.0)
    timing["epoch_selection"] = selection
    write_json(output_root / "timing_estimate.json", timing)
    checkpoint = output_root / "u0_timing_probe/checkpoints/last.pth"
    if not checkpoint.is_file() or (output_root / "stage2").exists():
        raise ValueError("Continuation requires the timing checkpoint and no existing Stage 2 runs.")
    return run_formal_stages(
        output_root,
        timing,
        selection,
        started_epoch=time.time(),
        started_monotonic=time.monotonic(),
        prior_wall_seconds=0.0,
        u0_epoch_wall=float(timing["u0_epoch_wall_seconds"]),
        benchmark_wall=float(timing.get("benchmark_total_seconds", 0.0)),
    )


def run_formal_stages(
    output_root: Path,
    timing: dict[str, Any],
    selection: dict[str, Any],
    *,
    started_epoch: float,
    started_monotonic: float,
    prior_wall_seconds: float,
    u0_epoch_wall: float,
    benchmark_wall: float,
) -> int:
    benchmark_path = output_root / "timing_estimate.json"
    timing_checkpoint = output_root / "u0_timing_probe/checkpoints/last.pth"
    u0_checkpoint = output_root / "u0_seed1/checkpoints/last.pth"

    timing_metadata = load_checkpoint_metadata(timing_checkpoint)
    normalization_artifacts = dict((timing_metadata or {}).get("normalization_artifacts") or {})
    if not normalization_artifacts:
        raise ValueError("Timing probe checkpoint lacks a protocol-bound train-only GPS scaler artifact.")
    final_config = prepare_config(
        output_root,
        epochs=selection["u0_epochs"],
        run_name="u0_seed1",
        normalization_artifacts=normalization_artifacts,
    )
    stage1_job = {
        "name": "u0_full_training",
        "gpu": 4,
        "command": ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "kd-sensing-train", "--config", str(final_config)],
        "log_path": output_root / "runtime/gpu4_u0_full.log",
        "checkpoint_watch": str(u0_checkpoint),
        "expected_epochs": selection["u0_epochs"],
        "started_epoch": time.time(),
    }
    stage1_started = time.monotonic()
    results = run_jobs("stage1_u0", [stage1_job], output_root)
    stage1_formal_wall = time.monotonic() - stage1_started
    if results[0]["return_code"] != 0 or not u0_checkpoint.is_file():
        return 1

    return run_stage2(
        output_root,
        timing,
        selection,
        started_epoch=started_epoch,
        started_monotonic=started_monotonic,
        prior_wall_seconds=prior_wall_seconds,
        u0_epoch_wall=u0_epoch_wall,
        benchmark_wall=benchmark_wall,
        stage1_formal_wall=stage1_formal_wall,
    )


def continue_stage2(output_root: Path, *, resume: bool = False) -> int:
    timing = json.loads((output_root / "timing_estimate.json").read_text(encoding="utf-8"))
    selection = timing.get("epoch_selection") or choose_epochs(timing, 0.0)
    checkpoint = output_root / "u0_seed1/checkpoints/last.pth"
    early_stopping_path = output_root / "u0_seed1/early_stopping.json"
    if not checkpoint.is_file() or not early_stopping_path.is_file():
        raise ValueError("Stage 2 requires a completed formal U0 checkpoint and early-stopping record.")
    early_stopping = json.loads(early_stopping_path.read_text(encoding="utf-8"))
    if int(early_stopping.get("actual_epochs", 0)) < int(selection["early_stopping"]["min_epochs"]):
        raise ValueError("Formal U0 did not reach the pre-registered minimum epoch count.")
    stage2_exists = (output_root / "stage2").exists()
    if stage2_exists and not resume:
        raise ValueError("Stage 2 output already exists; refusing to overwrite or resume implicitly.")
    if resume and not stage2_exists:
        raise ValueError("Stage 2 resume requires existing run artifacts.")
    manifest = json.loads((output_root / "runtime/orchestration_manifest.json").read_text(encoding="utf-8"))
    external_jobs = {
        str(job["name"]): job
        for job in manifest.get("jobs", [])
        if job.get("status") == "running"
    }
    final_config = output_root / "u0_seed1/final_config.yaml"
    stage1_formal_wall = max(0.0, checkpoint.stat().st_mtime - final_config.stat().st_mtime)
    return run_stage2(
        output_root,
        timing,
        selection,
        started_epoch=time.time(),
        started_monotonic=time.monotonic(),
        prior_wall_seconds=stage1_formal_wall,
        u0_epoch_wall=float(timing["u0_epoch_wall_seconds"]),
        benchmark_wall=float(timing.get("benchmark_total_seconds", 0.0)),
        stage1_formal_wall=stage1_formal_wall,
        resume=resume,
        external_jobs=external_jobs,
    )


def run_stage2(
    output_root: Path,
    timing: dict[str, Any],
    selection: dict[str, Any],
    *,
    started_epoch: float,
    started_monotonic: float,
    prior_wall_seconds: float,
    u0_epoch_wall: float,
    benchmark_wall: float,
    stage1_formal_wall: float,
    resume: bool = False,
    external_jobs: dict[str, dict[str, Any]] | None = None,
) -> int:
    benchmark_path = output_root / "timing_estimate.json"
    u0_checkpoint = output_root / "u0_seed1/checkpoints/last.pth"
    u0_actual_epochs = _completed_u0_epochs(output_root / "u0_seed1")
    audit_prototype_checkpoint(u0_checkpoint, output_root / "protocol/u0_checkpoint_health.json")

    checkpoint_sha, checkpoint_size = checkpoint_file_digest(u0_checkpoint)
    write_json(
        output_root / "u0_checkpoint_sha256.json",
        {"path": str(u0_checkpoint), "sha256": checkpoint_sha, "size_bytes": checkpoint_size, "outer_test_accessed": False},
    )
    artifact_config = output_root / "u0_seed1/final_config.yaml"
    schedule = output_root / "protocol/adapter_mask_schedule_seed1.json"
    if resume:
        if not schedule.is_file():
            raise ValueError("Stage 2 resume requires the existing shared mask schedule.")
    else:
        schedule = prepare_adapter_schedule(
            artifact_config, u0_checkpoint, checkpoint_sha, selection["adapter_epochs"], output_root
        )
    jobs = []
    return_codes: dict[str, int] = {}
    adopted = {}
    external_jobs = external_jobs or {}
    for key, gpu in STAGE2_GPUS.items():
        run_dir = output_root / "stage2" / key
        if (run_dir / "metrics.json").is_file():
            return_codes[key] = 0
            continue
        if run_dir.exists():
            external = external_jobs.get(key) or _running_adapter_process(key)
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            pid = int((external or {}).get("pid", 0))
            if not resume or status.get("status") != "running" or pid <= 0:
                raise ValueError(f"Cannot safely adopt incomplete Stage 2 run: {key}")
            adopted[key] = {**external, "gpu": gpu, "pid": pid}
            continue
        command = [
            "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python", str(Path(__file__).resolve()),
            "--adapter", key, "--config", str(artifact_config), "--checkpoint", str(u0_checkpoint),
            "--schedule", str(schedule), "--run-dir", str(run_dir), "--epochs", str(selection["adapter_epochs"]),
            "--expected-sha256", checkpoint_sha,
        ]
        jobs.append(
            {
                "name": key,
                "gpu": gpu,
                "command": command,
                "log_path": output_root / f"runtime/gpu{gpu}_{key}.log",
                "checkpoint_watch": str(run_dir / "checkpoints/last.pth" if key != "a0" else run_dir / "metrics.json"),
                "expected_epochs": 0 if key == "a0" else selection["adapter_epochs"],
                "started_epoch": time.time(),
            }
        )
    stage2_started = time.monotonic()
    stage2_results = run_jobs("stage2_resume" if resume else "stage2", jobs, output_root)
    return_codes.update({job["name"]: job["return_code"] for job in stage2_results})
    while adopted:
        for key, job in list(adopted.items()):
            metrics_path = output_root / "stage2" / key / "metrics.json"
            if metrics_path.is_file():
                return_codes[key] = 0
                adopted.pop(key)
                continue
            state_path = Path(f"/proc/{job['pid']}/stat")
            if not state_path.is_file() or state_path.read_text(encoding="utf-8").split()[2] == "Z":
                raise RuntimeError(f"Adopted Stage 2 task exited without metrics: {key}")
        if adopted:
            time.sleep(POLL_SECONDS)
    stage2_wall = time.monotonic() - stage2_started
    adapter_actual_epochs = {}
    adapter_actual_steps = {}
    for method in STAGE2_GPUS:
        metrics_path = output_root / "stage2" / method / "metrics.json"
        if not metrics_path.is_file():
            continue
        training = json.loads(metrics_path.read_text(encoding="utf-8")).get("training", {})
        adapter_actual_epochs[method] = int(training.get("actual_epochs", 0))
        adapter_actual_steps[method] = int(training.get("actual_optimizer_steps", 0))
    timing["actual"] = {
        "pipeline_started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
        "pipeline_finished_at": now(),
        "total_wall_seconds": prior_wall_seconds + time.monotonic() - started_monotonic,
        "u0_max_epochs": selection["u0_epochs"],
        "u0_actual_epochs": u0_actual_epochs,
        "adapter_max_epochs": selection["adapter_epochs"],
        "adapter_actual_epochs": adapter_actual_epochs,
        "adapter_actual_optimizer_steps": adapter_actual_steps,
        "stage1_timing_epoch_wall_seconds": u0_epoch_wall,
        "timing_benchmark_wall_seconds": benchmark_wall,
        "stage1_formal_wall_seconds": stage1_formal_wall,
        "stage1_total_wall_seconds": u0_epoch_wall + benchmark_wall + stage1_formal_wall,
        "stage2_parallel_wall_seconds": stage2_wall,
        "stage2_return_codes": return_codes,
    }
    write_json(benchmark_path, timing)
    write_json(
        output_root / "runtime/final_gpu_status.json",
        {
            "outer_test_accessed": False,
            "return_codes": return_codes,
            "gpus": {str(gpu): _gpu_status(gpu) for gpu in sorted(ALLOWED_PHYSICAL_GPUS)},
        },
    )
    if any(code != 0 for code in return_codes.values()):
        return 1
    analysis_started = time.monotonic()
    analysis = subprocess.run(
        ["conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python", "tools/analyze_full_pool_capacity.py", "--root", str(output_root)],
        cwd=ROOT,
        check=False,
    )
    analysis_wall = time.monotonic() - analysis_started
    timing = json.loads(benchmark_path.read_text(encoding="utf-8"))
    timing["actual"].update(
        analysis_wall_seconds=analysis_wall,
        analysis_return_code=analysis.returncode,
        pipeline_finished_at=now(),
        total_wall_seconds=prior_wall_seconds + time.monotonic() - started_monotonic,
    )
    write_json(benchmark_path, timing)
    return analysis.returncode


def run_adba_surrogate(output_root: Path) -> int:
    run_root = output_root / "adba_surrogate"
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing ADBA-surrogate run: {run_root}")
    config = output_root / "u0_seed1/final_config.yaml"
    checkpoint = output_root / "u0_seed1/checkpoints/last.pth"
    schedule = output_root / "protocol/adapter_mask_schedule_seed1.json"
    checkpoint_record = json.loads((output_root / "u0_checkpoint_sha256.json").read_text(encoding="utf-8"))
    expected_sha256 = str(checkpoint_record["sha256"])
    actual_sha256, checkpoint_size = checkpoint_file_digest(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"U0 checkpoint SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}")
    _, protocol_audit = preflight(config, checkpoint, expected_sha256=expected_sha256)
    schedule_payload = json.loads(schedule.read_text(encoding="utf-8"))
    schedule_sha256 = str(schedule_payload.pop("schedule_sha256"))
    if sha256_json(schedule_payload) != schedule_sha256:
        raise ValueError("Mask schedule hash mismatch before ADBA-surrogate launch.")
    run_root.mkdir(parents=True)
    prototype_audit = audit_prototype_checkpoint(checkpoint, run_root / "preflight/u0_checkpoint_health.json")
    write_json(
        run_root / "preflight/launch_audit.json",
        {
            "status": "passed",
            "outer_test_accessed": False,
            "protocol_status": protocol_audit.get("status"),
            "protocol_fingerprint": protocol_audit.get("protocol_fingerprint"),
            "u0_checkpoint_sha256": expected_sha256,
            "u0_checkpoint_size_bytes": checkpoint_size,
            "prototype_status": prototype_audit["status"],
            "mask_schedule_sha256": schedule_sha256,
            "loss_profile": {"name": "adba_surrogate", **ADAPTER_LOSS_PROFILES["adba_surrogate"]},
            "jobs": {key: {"experiment": experiment, "gpu": gpu} for key, (experiment, gpu) in ADBA_SURROGATE_JOBS.items()},
        },
    )
    jobs = []
    for key, (experiment, gpu) in ADBA_SURROGATE_JOBS.items():
        run_dir = run_root / key
        command = [
            "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python", str(Path(__file__).resolve()),
            "--adapter", experiment, "--loss-profile", "adba_surrogate", "--config", str(config),
            "--checkpoint", str(checkpoint), "--schedule", str(schedule), "--run-dir", str(run_dir),
            "--epochs", str(CONVERGENCE_EPOCHS), "--expected-sha256", expected_sha256,
        ]
        jobs.append(
            {
                "name": key,
                "gpu": gpu,
                "command": command,
                "log_path": run_root / f"runtime/gpu{gpu}_{key}.log",
                "checkpoint_watch": str(run_dir / "checkpoints/last.pth"),
                "expected_epochs": CONVERGENCE_EPOCHS,
                "started_epoch": time.time(),
            }
        )
    results = run_jobs("adba_surrogate", jobs, run_root)
    return_codes = {job["name"]: int(job["return_code"]) for job in results}
    write_json(
        run_root / "runtime/final_status.json",
        {"outer_test_accessed": False, "return_codes": return_codes, "finished_at": now()},
    )
    if any(code != 0 for code in return_codes.values()):
        return 1
    analysis = subprocess.run(
        [
            "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
            "tools/analyze_full_pool_adba_surrogate.py", "--root", str(output_root),
        ],
        cwd=ROOT,
        check=False,
    )
    return analysis.returncode


def run_mask_bias_ablation(output_root: Path) -> int:
    run_root = output_root / "mask_bias_ablation"
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing mask-bias ablation: {run_root}")
    config = output_root / "u0_seed1/final_config.yaml"
    checkpoint = output_root / "u0_seed1/checkpoints/last.pth"
    full_schedule = output_root / "protocol/adapter_mask_schedule_seed1.json"
    checkpoint_record = json.loads((output_root / "u0_checkpoint_sha256.json").read_text(encoding="utf-8"))
    expected_sha256 = str(checkpoint_record["sha256"])
    actual_sha256, checkpoint_size = checkpoint_file_digest(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"U0 checkpoint SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}")
    _, protocol_audit = preflight(config, checkpoint, expected_sha256=expected_sha256)

    schedule_payload = json.loads(full_schedule.read_text(encoding="utf-8"))
    schedule_sha256 = str(schedule_payload.pop("schedule_sha256"))
    if sha256_json(schedule_payload) != schedule_sha256:
        raise ValueError("Mask schedule hash mismatch before mask-bias ablation.")
    b1_metrics = json.loads((output_root / "adba_surrogate/b1/metrics.json").read_text(encoding="utf-8"))
    if (
        b1_metrics.get("loss_profile", {}).get("name") != "adba_surrogate"
        or int(b1_metrics.get("training", {}).get("actual_epochs", -1)) != MASK_BIAS_ABLATION_EPOCHS
        or float(b1_metrics.get("full_equivalence", {}).get("max_abs_logit_diff", 1.0)) > 1e-7
    ):
        raise ValueError("Reusable B1 does not match the preregistered 8-epoch ADBA-surrogate contract.")

    folds = stratified_mask_folds(seed=1, fold_count=4)
    held_out = folds[0]
    allowed = tuple(mask for mask in NON_FULL_MASKS if mask not in held_out)
    unseen_schedule = generate_mask_schedule(
        list(schedule_payload["sample_ids"]),
        epochs=MASK_BIAS_ABLATION_EPOCHS,
        seed=1,
        allowed_masks=allowed,
    )
    by_mask = {tuple(mask): key for key, _, mask in MASKS}
    run_root.mkdir(parents=True)
    write_json(run_root / "protocol/unseen_mask_schedule_seed1.json", unseen_schedule)
    write_json(
        run_root / "protocol/unseen_fold0.json",
        {
            "schema_version": 1,
            "seed": 1,
            "fold_count": 4,
            "selected_fold": 0,
            "folds": [[list(mask) for mask in fold] for fold in folds],
            "held_out_masks": [list(mask) for mask in held_out],
            "held_out_keys": [by_mask[mask] for mask in held_out],
            "allowed_masks": [list(mask) for mask in allowed],
            "train_sample_ids_sha256": sha256_json(schedule_payload["sample_ids"]),
            "train_sample_count": len(schedule_payload["sample_ids"]),
            "outer_test_accessed": False,
        },
    )
    prototype_audit = audit_prototype_checkpoint(checkpoint, run_root / "preflight/u0_checkpoint_health.json")
    write_json(
        run_root / "preflight/launch_audit.json",
        {
            "status": "passed",
            "outer_test_accessed": False,
            "protocol_status": protocol_audit.get("status"),
            "u0_checkpoint_sha256": expected_sha256,
            "u0_checkpoint_size_bytes": checkpoint_size,
            "prototype_status": prototype_audit["status"],
            "full_mask_schedule_sha256": schedule_sha256,
            "unseen_mask_schedule_sha256": unseen_schedule["schedule_sha256"],
            "loss_profile": {"name": "adba_surrogate", **ADAPTER_LOSS_PROFILES["adba_surrogate"]},
            "epochs": MASK_BIAS_ABLATION_EPOCHS,
            "reused_b1": str((output_root / "adba_surrogate/b1").resolve()),
        },
    )

    def jobs_for(mapping: dict[str, tuple[str, int]], stage_root: Path, schedule: Path) -> list[dict[str, Any]]:
        jobs = []
        for name, (experiment, gpu) in mapping.items():
            run_dir = stage_root / name
            jobs.append(
                {
                    "name": name,
                    "gpu": gpu,
                    "command": [
                        "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
                        str(Path(__file__).resolve()), "--adapter", experiment,
                        "--loss-profile", "adba_surrogate", "--config", str(config),
                        "--checkpoint", str(checkpoint), "--schedule", str(schedule),
                        "--run-dir", str(run_dir), "--epochs", str(MASK_BIAS_ABLATION_EPOCHS),
                        "--expected-sha256", expected_sha256,
                    ],
                    "log_path": stage_root / f"runtime/gpu{gpu}_{name}.log",
                    "checkpoint_watch": str(run_dir / "checkpoints/last.pth"),
                    "expected_epochs": MASK_BIAS_ABLATION_EPOCHS,
                    "started_epoch": time.time(),
                }
            )
        return jobs

    all_seen_root = run_root / "all_seen"
    all_seen = run_jobs(
        "mask_bias_all_seen",
        jobs_for(MASK_BIAS_ALL_SEEN_JOBS, all_seen_root, full_schedule),
        all_seen_root,
    )
    if any(int(job["return_code"]) != 0 for job in all_seen):
        write_json(run_root / "runtime/final_status.json", {"status": "failed", "stage": "all_seen"})
        return 1

    analysis_command = [
        "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
        "tools/analyze_mask_bias_ablation.py", "--root", str(output_root),
    ]
    analysis = subprocess.run(analysis_command, cwd=ROOT, check=False)
    if analysis.returncode != 0:
        return analysis.returncode
    decision = json.loads((run_root / "stage_decision.json").read_text(encoding="utf-8"))
    if not bool(decision["unseen_pilot_authorized"]):
        write_json(
            run_root / "runtime/final_status.json",
            {"status": "completed", "unseen_stage": "skipped_by_preregistered_gate", "stage_decision": decision},
        )
        return 0

    unseen_root = run_root / "unseen_fold0"
    unseen = run_jobs(
        "mask_bias_unseen_fold0",
        jobs_for(
            MASK_BIAS_UNSEEN_JOBS,
            unseen_root,
            run_root / "protocol/unseen_mask_schedule_seed1.json",
        ),
        unseen_root,
    )
    return_codes = {job["name"]: int(job["return_code"]) for job in unseen}
    if any(code != 0 for code in return_codes.values()):
        write_json(
            run_root / "runtime/final_status.json",
            {"status": "failed", "stage": "unseen_fold0", "return_codes": return_codes},
        )
        return 1
    analysis = subprocess.run(analysis_command, cwd=ROOT, check=False)
    write_json(
        run_root / "runtime/final_status.json",
        {
            "status": "completed" if analysis.returncode == 0 else "failed",
            "return_codes": {
                **{job["name"]: int(job["return_code"]) for job in all_seen},
                **return_codes,
            },
            "analysis_return_code": analysis.returncode,
            "stage_decision": decision,
            "outer_test_accessed": False,
            "finished_at": now(),
        },
    )
    return analysis.returncode


def run_circular_transport(output_root: Path) -> int:
    """Run the fixed-budget local circular transport versus factorized-bias comparison."""
    run_root = output_root / "circular_transport"
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing circular transport run: {run_root}")
    config = output_root / "u0_seed1/final_config.yaml"
    checkpoint = output_root / "u0_seed1/checkpoints/last.pth"
    schedule = output_root / "protocol/adapter_mask_schedule_seed1.json"
    checkpoint_record = json.loads((output_root / "u0_checkpoint_sha256.json").read_text(encoding="utf-8"))
    expected_sha256 = str(checkpoint_record["sha256"])
    actual_sha256, checkpoint_size = checkpoint_file_digest(checkpoint)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"U0 checkpoint SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}")
    _, protocol_audit = preflight(config, checkpoint, expected_sha256=expected_sha256)
    schedule_payload = json.loads(schedule.read_text(encoding="utf-8"))
    schedule_sha256 = str(schedule_payload.pop("schedule_sha256"))
    if sha256_json(schedule_payload) != schedule_sha256:
        raise ValueError("Mask schedule hash mismatch before circular transport launch.")
    a0_metrics = output_root / "stage2/a0/metrics.json"
    b1_metrics_path = output_root / "adba_surrogate/b1/metrics.json"
    if not a0_metrics.is_file() or not b1_metrics_path.is_file():
        raise FileNotFoundError("Circular transport comparison requires completed A0 and B1 artifacts.")
    b1_metrics = json.loads(b1_metrics_path.read_text(encoding="utf-8"))
    if (
        b1_metrics.get("loss_profile", {}).get("name") != "adba_surrogate"
        or int(b1_metrics.get("training", {}).get("actual_epochs", -1)) != CIRCULAR_TRANSPORT_EPOCHS
        or float(b1_metrics.get("full_equivalence", {}).get("max_abs_logit_diff", 1.0)) > 1e-7
    ):
        raise ValueError("Reusable B1 does not match the fixed 8-epoch ADBA-surrogate contract.")

    run_root.mkdir(parents=True)
    prototype_audit = audit_prototype_checkpoint(checkpoint, run_root / "preflight/u0_checkpoint_health.json")
    write_json(
        run_root / "preflight/launch_audit.json",
        {
            "status": "passed",
            "outer_test_accessed": False,
            "protocol_status": protocol_audit.get("status"),
            "u0_checkpoint_sha256": expected_sha256,
            "u0_checkpoint_size_bytes": checkpoint_size,
            "prototype_status": prototype_audit["status"],
            "mask_schedule_sha256": schedule_sha256,
            "loss_profile": {"name": "adba_surrogate", **ADAPTER_LOSS_PROFILES["adba_surrogate"]},
            "epochs": CIRCULAR_TRANSPORT_EPOCHS,
            "reused_a0": str(a0_metrics.parent.resolve()),
            "reused_b1": str(b1_metrics_path.parent.resolve()),
            "jobs": {key: {"experiment": experiment, "gpu": gpu} for key, (experiment, gpu) in CIRCULAR_TRANSPORT_JOBS.items()},
        },
    )
    jobs = []
    for name, (experiment, gpu) in CIRCULAR_TRANSPORT_JOBS.items():
        run_dir = run_root / name
        jobs.append(
            {
                "name": name,
                "gpu": gpu,
                "command": [
                    "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
                    str(Path(__file__).resolve()), "--adapter", experiment,
                    "--loss-profile", "adba_surrogate", "--config", str(config),
                    "--checkpoint", str(checkpoint), "--schedule", str(schedule),
                    "--run-dir", str(run_dir), "--epochs", str(CIRCULAR_TRANSPORT_EPOCHS),
                    "--expected-sha256", expected_sha256,
                ],
                "log_path": run_root / f"runtime/gpu{gpu}_{name}.log",
                "checkpoint_watch": str(run_dir / "checkpoints/last.pth"),
                "expected_epochs": CIRCULAR_TRANSPORT_EPOCHS,
                "started_epoch": time.time(),
            }
        )
    results = run_jobs("circular_transport", jobs, run_root)
    return_codes = {job["name"]: int(job["return_code"]) for job in results}
    if any(code != 0 for code in return_codes.values()):
        write_json(run_root / "runtime/final_status.json", {"status": "failed", "return_codes": return_codes})
        return 1
    analysis = subprocess.run(
        [
            "conda", "run", "-n", "kd_mm_beam", "--no-capture-output", "python",
            "tools/analyze_circular_transport.py", "--root", str(output_root),
        ],
        cwd=ROOT,
        check=False,
    )
    write_json(
        run_root / "runtime/final_status.json",
        {
            "status": "completed" if analysis.returncode == 0 else "failed",
            "return_codes": return_codes,
            "analysis_return_code": analysis.returncode,
            "outer_test_accessed": False,
            "finished_at": now(),
        },
    )
    return analysis.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--continue-after-benchmark", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--resume-stage2", action="store_true")
    parser.add_argument("--adba-surrogate", action="store_true")
    parser.add_argument("--mask-bias-ablation", action="store_true")
    parser.add_argument("--circular-transport", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--adapter", choices=tuple(EXPERIMENTS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--loss-profile", choices=tuple(ADAPTER_LOSS_PROFILES), default="cross_entropy")
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument("--u0-epoch-wall", type=float, default=0.0)
    args = parser.parse_args(argv)
    selected = sum(
        (
            args.orchestrate,
            args.continue_after_benchmark,
            args.stage2_only,
            args.resume_stage2,
            args.adba_surrogate,
            args.mask_bias_ablation,
            args.circular_transport,
            args.benchmark,
            args.adapter is not None,
        )
    )
    if selected != 1:
        parser.error(
            "choose exactly one of --orchestrate, --continue-after-benchmark, --stage2-only, --resume-stage2, --adba-surrogate, --mask-bias-ablation, --circular-transport, --benchmark, or --adapter"
        )
    if args.orchestrate:
        return orchestrate(args.output_root.resolve())
    if args.continue_after_benchmark:
        return continue_after_benchmark(args.output_root.resolve())
    if args.stage2_only:
        return continue_stage2(args.output_root.resolve())
    if args.resume_stage2:
        return continue_stage2(args.output_root.resolve(), resume=True)
    if args.adba_surrogate:
        return run_adba_surrogate(args.output_root.resolve())
    if args.mask_bias_ablation:
        return run_mask_bias_ablation(args.output_root.resolve())
    if args.circular_transport:
        return run_circular_transport(args.output_root.resolve())
    required = (args.config, args.checkpoint, args.expected_sha256)
    if any(value is None for value in required):
        parser.error("child modes require --config, --checkpoint, and --expected-sha256")
    if args.benchmark:
        if args.timing_output is None:
            parser.error("--benchmark requires --timing-output")
        print(json.dumps(benchmark(args.config, args.checkpoint, args.expected_sha256, args.timing_output, args.u0_epoch_wall), indent=2))
        return 0
    if args.schedule is None or args.run_dir is None:
        parser.error("--adapter requires --schedule and --run-dir")
    return run_adapter_child(args)


if __name__ == "__main__":
    raise SystemExit(main())
