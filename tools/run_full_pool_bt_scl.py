#!/usr/bin/env python3
"""Run the local, protocol-bound Full-pool BT-SCL experiment.

This is intentionally a local research tool rather than a package CLI.  It
does not load U0, old clean-inner splits, channel/path tensors, or outer data.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from kd_sensing.baselines.full_pool_bt_scl import (
    BTSCLModel,
    METHODS,
    MODALITIES,
    PATTERNS,
    btscl_losses,
    check_missing_token_invariance,
    coarse_to_fine_loss,
    generate_nested_schedule,
    load_audited_topology,
    monotonicity_loss,
    parameter_rows,
    schedule_masks,
    sha256_file,
    topology_risk,
    write_json,
)
from kd_sensing.baselines.full_pool_common import atomic_csv as _atomic_csv
from kd_sensing.baselines.full_pool_common import now
from kd_sensing.baselines.full_pool_common import set_seed as _set_seed
from kd_sensing.config import load_config
from kd_sensing.data.mmw.full_pool_protocol import (
    FULL_POOL_DEVELOPMENT_WINDOWS,
    FULL_POOL_HISTORICAL_VALIDATION_RETAINED,
    FULL_POOL_PROTOCOL_ID,
    FULL_POOL_RESOURCE_INTERSECTION_NAMES,
    FULL_POOL_SPLIT_EXPECTATIONS,
    load_full_pool_protocol,
    protocol_dataset_domains,
)
from kd_sensing.data.transform_ops.gps import load_gps_scaler
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices
from kd_sensing.engine.normalization_artifacts import save_normalization_artifacts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/full_pool_bt_scl"
DEFAULT_PROTOCOL = ROOT / "outputs/full_pool_capacity/protocol/split_manifest.json"
DEFAULT_TOPOLOGY = ROOT / "outputs/cache/mmw_codebook_topology/v1/a692c2b43365b483/topology_manifest.json"
# Bound to the canonical protocol rather than restated, so a protocol change can
# never leave this workflow asserting stale window counts.
TRAIN_WINDOWS = FULL_POOL_SPLIT_EXPECTATIONS["train_sample_count"]
VALIDATION_WINDOWS = FULL_POOL_SPLIT_EXPECTATIONS["validation_sample_count"]
BOUNDARY_PURGE = FULL_POOL_SPLIT_EXPECTATIONS["boundary_crossing_excluded_count"]
HISTORICAL_TRAIN_REMOVED = FULL_POOL_SPLIT_EXPECTATIONS["historical_removed_from_train_count"]
SEED = 2026
EPOCHS = 30
STABLE_EPOCHS = 20
STABLE_METHODS = {
    "r0_subset_task_only",
    "r3_coarse_to_fine",
    "r6_topological_stochastic_dominance",
}
MODEL_CONFIG = {"d_model": 256, "seq_len": 5, "gps_input_size": 3, "dropout": 0.1, "num_classes": 64}
BASE_WEIGHTS = {
    "task_topology": 0.05,
    "uni": 1.0,
    "mono": 1.0,
    "c2f": 1.0,
    "local": 0.2,
    "hierarchy": 0.25,
    "dominance": 1.0,
}


def set_seed(seed: int = SEED) -> None:
    _set_seed(seed)


def _protocol_audit(protocol: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    checks = {
        key: int(protocol.get(key, -1)) == value
        for key, value in FULL_POOL_SPLIT_EXPECTATIONS.items()
    }
    checks.update(
        validation_historical_retained=int(protocol.get("historical_protected_count", -1)) - int(protocol.get("historical_removed_from_train_count", -1)) == FULL_POOL_HISTORICAL_VALIDATION_RETAINED,
        protocol_id=protocol.get("protocol_id") == FULL_POOL_PROTOCOL_ID,
        protocol_outer_disabled=protocol.get("outer_test_enabled") is False,
        protocol_outer_unaccessed=protocol.get("outer_test_accessed") is False,
        confirmation_disabled=protocol.get("allow_confirmation_train") is False,
        audit_passed=audit.get("status") == "passed",
        audit_outer_unaccessed=audit.get("outer_test_accessed") is False,
        protocol_fingerprint=audit.get("protocol_fingerprint") == protocol.get("protocol_fingerprint"),
        all_required_resource_intersections_zero=all(
            int(audit.get("overlap_counts", {}).get(name, -1)) == 0
            for name in FULL_POOL_RESOURCE_INTERSECTION_NAMES
        ),
        legacy_clean_inner_unused=True,
        channel_or_path_model_input_disabled=True,
    )
    result = {
        "audit_id": "full_pool_bt_scl_protocol_audit_v1",
        "created_at": now(),
        "canonical_protocol": "full_pool_contiguous_time_v1",
        "checks": checks,
        "manifest": str(DEFAULT_PROTOCOL),
        "audit": str(audit_path),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "train_windows": int(protocol["train_sample_count"]),
        "validation_windows": int(protocol["validation_sample_count"]),
        "development_windows": int(protocol["train_sample_count"] + protocol["validation_sample_count"]),
        "outer_test_accessed": False,
        "status": "passed" if all(checks.values()) else "failed",
    }
    if result["status"] != "passed":
        raise ValueError(f"BT-SCL protocol audit failed: {checks}")
    return result


def build_config(protocol: dict[str, Any], root: Path) -> dict[str, Any]:
    cfg = load_config(ROOT / "configs/mmw/u0.yaml")
    cfg["experiment"].update(name="FullPool_BT_SCL", seed=SEED, device="auto")
    cfg["data"]["dataset"].update(
        domains=protocol_dataset_domains(protocol),
        portion=1.0,
        frame_cache_root=str((ROOT / "outputs/cache/MMW").resolve()),
        frame_cache_strict=True,
        gps_coordinate_cache_root=str((ROOT / "outputs/full_pool_capacity/cache/gps_coordinates").resolve()),
        include_router_utility_targets=False,
        include_router_corruption_metadata=False,
        lidar_augment=False,
    )
    cfg["data"]["dataloader"].update(
        train_batch_size=64,
        validation_batch_size=64,
        test_batch_size=64,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        train_drop_last=False,
        test_drop_last=False,
    )
    cfg["data_protocol"] = {
        "mode": protocol["mode"],
        "path": str((ROOT / "outputs/full_pool_capacity/protocol/split_manifest.json").resolve()),
        "audit_report": str((ROOT / "outputs/full_pool_capacity/protocol/split_audit.json").resolve()),
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "train_role": protocol["train_role"],
        "validation_role": protocol["validation_role"],
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
    }
    cfg["training"].update(epochs=EPOCHS, max_epochs=EPOCHS, lr=3e-4, weight_decay=1e-4, resume=False)
    cfg["training"]["final_test"] = {"enabled": False}
    cfg["output"].update(dir=str(root), run_name="bt_scl", overwrite=False)
    return cfg


def _stable_sample_ids(dataset: Any) -> list[str]:
    values = []
    for leaf, indices in leaf_datasets_with_indices(dataset):
        rows = getattr(getattr(leaf, "samples", None), "rows", None)
        if not isinstance(rows, list):
            raise ValueError("BT-SCL schedule requires MMW sample rows.")
        for index in indices:
            row = rows[int(index)]
            sample_id = str(row.get("sample_id") or row.get("target_sample_id") or "").strip()
            if not sample_id:
                raise ValueError("BT-SCL schedule requires a stable source sample id.")
            values.append(f"mmw:{leaf.condition}:{leaf.scene_slug}:{leaf.split}:{sample_id}")
    return values


def _batch_ids(batch: dict[str, Any]) -> list[str]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, dict) or "stable_sample_id" not in metadata:
        raise ValueError("BT-SCL batch is missing stable_sample_id metadata for the schedule.")
    return [str(value) for value in metadata["stable_sample_id"]]


def _inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    image = torch.as_tensor(batch["image"]).to(device=device, non_blocking=True)
    lidar = torch.as_tensor(batch["lidar"]).to(device=device, non_blocking=True)
    radar_ra = torch.as_tensor(batch["radar_ra"]).to(device=device, non_blocking=True)
    radar_da = torch.as_tensor(batch["radar_da"]).to(device=device, non_blocking=True)
    if radar_ra.ndim == 4:
        radar_ra = radar_ra.unsqueeze(2)
        radar_da = radar_da.unsqueeze(2)
    radar = torch.cat((radar_ra, radar_da), dim=2)
    gps = torch.as_tensor(batch["gps"]).to(device=device, non_blocking=True)
    expected = {"image": (5, 3, 224, 224), "lidar": (5, 3, 224, 224), "radar": (5, 2, 128, 64), "gps": (5, 3)}
    actual = {
        "image": tuple(image.shape[1:]),
        "lidar": tuple(lidar.shape[1:]),
        "radar": tuple(radar.shape[1:]),
        "gps": tuple(gps.shape[1:]),
    }
    if actual != expected:
        raise ValueError(f"BT-SCL input contract mismatch: expected {expected}, got {actual}.")
    return {"image": image, "lidar": lidar, "radar": radar, "gps": gps}


def _labels(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    labels = torch.as_tensor(batch["target_beam"], device=device, dtype=torch.long).reshape(-1)
    if not bool(((labels >= 0) & (labels < 64)).all()):
        raise ValueError("BT-SCL labels must be in [0, 63].")
    return labels


def _take_batch(batch: Any, count: int) -> Any:
    """Use a small fixed train-only slice for calibration and preflight only."""
    if torch.is_tensor(batch):
        return batch[:count]
    if isinstance(batch, dict):
        return {key: _take_batch(value, count) for key, value in batch.items()}
    if isinstance(batch, list):
        return batch[:count]
    if isinstance(batch, tuple):
        return tuple(_take_batch(value, count) for value in batch)
    return batch


def _small_train_only_loader(loader: Any, *, batch_size: int = 1) -> DataLoader:
    """Avoid materializing a formal 64-sample batch for calibration/preflight."""
    return DataLoader(loader.dataset, batch_size=int(batch_size), shuffle=False, num_workers=0, pin_memory=False)


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _loaders(root: Path, protocol: dict[str, Any], *, create_normalization: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = build_config(protocol, root)
    artifact = root / "artifacts/gps_scaler.npz"
    if artifact.is_file():
        loaders = build_dataloaders(cfg, normalization_overrides={"gps_scaler": load_gps_scaler(artifact)})
        manifest = json.loads((root / "normalization_manifest.json").read_text(encoding="utf-8"))
    elif create_normalization:
        loaders = build_dataloaders(cfg)
        manifest = save_normalization_artifacts(loaders, root)
        write_json(root / "normalization_manifest.json", manifest)
    else:
        raise FileNotFoundError("BT-SCL normalization manifest is absent; run --prepare first.")
    if not manifest or not (root / "artifacts/gps_scaler.npz").is_file():
        raise ValueError("BT-SCL did not create the shared train-only GPS scaler.")
    metadata = manifest.get("metadata", {})
    if metadata.get("source_split") != "full_pool_train" or int(metadata.get("effective_sample_count", -1)) != TRAIN_WINDOWS:
        raise ValueError("BT-SCL GPS scaler is not bound to the Full-pool train split.")
    return loaders, cfg


def _save_initialization(root: Path) -> dict[str, Any]:
    directory = root / "initialization"
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "initial_model.pt"
    if model_path.exists():
        return json.loads((directory / "parameter_report.json").read_text(encoding="utf-8"))
    set_seed(SEED)
    model = BTSCLModel(**{key: value for key, value in MODEL_CONFIG.items() if key != "num_classes"})
    torch.save({"state_dict": model.state_dict(), "model_config": MODEL_CONFIG, "seed": SEED}, model_path)
    model_config_path = directory / "model_config.yaml"
    model_config_path.write_text(yaml.safe_dump(MODEL_CONFIG, sort_keys=True), encoding="utf-8")
    (directory / "initialization_sha256.txt").write_text(sha256_file(model_path) + "\n", encoding="utf-8")
    report = {
        "seed": SEED,
        "initialization_sha256": sha256_file(model_path),
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "model_config": MODEL_CONFIG,
    }
    write_json(directory / "parameter_report.json", report)
    return report


def _load_initial_model(root: Path, device: torch.device) -> BTSCLModel:
    payload = torch.load(root / "initialization/initial_model.pt", map_location="cpu", weights_only=False)
    if payload.get("model_config") != MODEL_CONFIG:
        raise ValueError("BT-SCL initialization model config mismatch.")
    model = BTSCLModel(**{key: value for key, value in MODEL_CONFIG.items() if key != "num_classes"})
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device)


def _loss_calibration(root: Path, loaders: dict[str, Any], topology: Any) -> dict[str, float]:
    target = root / "loss_scale_calibration.json"
    if target.is_file():
        weights = dict(BASE_WEIGHTS)
        weights.update(json.loads(target.read_text(encoding="utf-8"))["weights"])
        # R6 weights are structurally bounded and pre-registered, never inverse-scaled.
        weights.update(hierarchy=BASE_WEIGHTS["hierarchy"], dominance=BASE_WEIGHTS["dominance"])
        return weights
    write_json(root / "prepare_progress.json", {"stage": "loss_calibration_started", "at": now()})
    # Calibration is a fixed train-only numeric measurement; CPU avoids sharing
    # a production GPU before the explicit smoke stage.
    device = torch.device("cpu")
    model = _load_initial_model(root, device).train()
    schedule = json.loads((root / "train_nested_subset_schedule.json").read_text(encoding="utf-8"))
    values: dict[str, list[float]] = defaultdict(list)
    for batch_index, batch in enumerate(_small_train_only_loader(loaders["train"])):
        with torch.no_grad(), _autocast(device):
            inputs = _inputs(batch, device)
            labels = _labels(batch, device)
            masks = schedule_masks(_batch_ids(batch), schedule, device)
            views = model.forward_views(inputs, masks)
            _, report = btscl_losses(model, views, labels, topology, "r5_full_bt_scl", BASE_WEIGHTS)
        for name in ("base", "uni", "mono", "c2f", "c2f_local"):
            if name in report:
                values[name].append(float(report[name]))
        if batch_index >= 3:
            break
    means = {name: max(float(np.mean(items)), 1e-8) for name, items in values.items()}
    weights = dict(BASE_WEIGHTS)
    for loss_name, weight_name, fraction in (("uni", "uni", 0.20), ("mono", "mono", 0.15), ("c2f", "c2f", 0.20)):
        weights[weight_name] = float(fraction * means["base"] / means[loss_name])
    if "c2f_local" in means:
        weights["local"] = float(0.20 * max(means["c2f"] - BASE_WEIGHTS["local"] * means["c2f_local"], 1e-8) / means["c2f_local"])
    payload = {"created_at": now(), "batches": 4, "train_only": True, "raw_means": means, "weights": weights}
    write_json(target, payload)
    write_json(root / "prepare_progress.json", {"stage": "loss_calibration_completed", "at": now()})
    return weights


def _write_preflight(root: Path, loaders: dict[str, Any], topology: Any) -> None:
    target = root / "preflight_tests.txt"
    if target.is_file():
        return
    write_json(root / "prepare_progress.json", {"stage": "preflight_started", "at": now()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_initial_model(root, device).eval()
    train_schedule = json.loads((root / "train_nested_subset_schedule.json").read_text(encoding="utf-8"))
    batch = next(iter(_small_train_only_loader(loaders["train"])))
    inputs, labels = _inputs(batch, device), _labels(batch, device)
    masks = schedule_masks(_batch_ids(batch), train_schedule, device)
    lines = []
    with torch.no_grad(), _autocast(device):
        views = model.forward_views(inputs, masks)
        lines.append(f"four_view_logits_shape={tuple(views['logits'].shape)}")
        for pattern, values in PATTERNS.items():
            availability = torch.tensor(values, device=device, dtype=torch.bool).expand(labels.shape[0], -1)
            output = model.logits_from_tokens(views["tokens"], availability)[0]
            if tuple(output.shape) != (labels.shape[0], 64) or not bool(torch.isfinite(output).all()):
                raise AssertionError(f"BT-SCL forward failed for {pattern}.")
        lines.append("all_15_patterns_forward=passed")
        check_missing_token_invariance(model, inputs, torch.tensor(PATTERNS["missing_image"], device=device, dtype=torch.bool).expand(labels.shape[0], -1))
        lines.append("missing_token_zero_and_input_invariance=passed")
        try:
            model.logits_from_tokens(views["tokens"], torch.zeros((labels.shape[0], 4), dtype=torch.bool, device=device))
        except ValueError:
            lines.append("all_missing_rejected=passed")
        else:
            raise AssertionError("BT-SCL accepted all-missing input.")
        mono = monotonicity_loss(views["logits"], labels, topology)
        c2f, detail = coarse_to_fine_loss(views["logits"], labels, topology)
        lines.extend((f"topology_risk_and_mono_finite={bool(torch.isfinite(mono))}", f"c2f_finite={bool(torch.isfinite(c2f))}", f"c2f_detail={sorted(detail)}"))
    lines.extend(
        (
            f"protocol_counts={TRAIN_WINDOWS},{VALIDATION_WINDOWS},{FULL_POOL_DEVELOPMENT_WINDOWS}",
            f"boundary_purge={BOUNDARY_PURGE}",
            f"historical_train_removed={HISTORICAL_TRAIN_REMOVED}",
            f"historical_validation_retained={FULL_POOL_HISTORICAL_VALIDATION_RETAINED}",
            "outer_test_accessed=false",
            "legacy_clean_inner_used=false",
            "channel_path_model_input=false",
            f"topology_descriptor_sha256={topology.descriptor_sha256}",
            "status=passed",
        )
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "prepare_progress.json", {"stage": "preflight_completed", "at": now()})


def prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_full_pool_protocol(DEFAULT_PROTOCOL)
    audit = _protocol_audit(protocol, ROOT / "outputs/full_pool_capacity/protocol/split_audit.json")
    shutil.copy2(DEFAULT_PROTOCOL, root / "protocol_manifest_copy.json")
    shutil.copy2(ROOT / "outputs/full_pool_capacity/protocol/split_audit.json", root / "protocol_audit_copy.json")
    write_json(root / "protocol_audit.json", audit)
    (root / "protocol_audit.md").write_text(
        "# Full-pool BT-SCL Protocol Audit\n\n"
        "CANONICAL_PROTOCOL=full_pool_contiguous_time_v1\n\n"
        f"TRAIN_WINDOWS={TRAIN_WINDOWS}\n\nVALIDATION_WINDOWS={VALIDATION_WINDOWS}\n\n"
        f"DEVELOPMENT_WINDOWS={FULL_POOL_DEVELOPMENT_WINDOWS}\n\n"
        "LEGACY_CLEAN_INNER_USED=false\n\nOUTER_TEST_ACCESSED=false\n",
        encoding="utf-8",
    )
    topology = load_audited_topology(DEFAULT_TOPOLOGY)
    write_json(root / "topology_audit.json", {"manifest": topology.manifest_path, "manifest_sha256": topology.manifest_sha256, "descriptor_sha256": topology.descriptor_sha256, "labels_by_phase": list(topology.labels_by_position), "status": "passed"})
    write_json(root / "prepare_progress.json", {"stage": "loader_build_started", "at": now()})
    loaders, cfg = _loaders(root, protocol, create_normalization=True)
    write_json(root / "prepare_progress.json", {"stage": "loader_build_completed", "at": now()})
    try:
        train_ids, validation_ids = _stable_sample_ids(loaders["train"].dataset), _stable_sample_ids(loaders["validation"].dataset)
        if len(train_ids) != TRAIN_WINDOWS or len(validation_ids) != VALIDATION_WINDOWS:
            raise ValueError("BT-SCL loader counts differ from the canonical Full-pool protocol.")
        train_schedule = generate_nested_schedule(train_ids, seed=SEED, split="train")
        validation_schedule = generate_nested_schedule(validation_ids, seed=SEED, split="validation")
        write_json(root / "train_nested_subset_schedule.json", train_schedule)
        write_json(root / "validation_nested_subset_manifest.json", validation_schedule)
        write_json(root / "prepare_progress.json", {"stage": "schedule_completed", "at": now()})
        write_json(root / "input_contract.json", {"modalities": list(MODALITIES), "token_order": list(MODALITIES), "seq_len": 5, "image": [5, 3, 224, 224], "lidar": [5, 3, 224, 224], "radar": [5, 2, 128, 64], "gps": [5, 3], "label": "target_beam [B,1] in [0,63]", "forbidden_model_inputs": ["metadata", "weather", "scene", "channel", "path", "beam_power", "historical_beam"]})
        (root / "implementation_notes.md").write_text("BT-SCL is a local single-seed development workflow. It uses only RGB, LiDAR BEV, Radar RA/DA, train-normalized GPS, availability masks and beam labels. Metadata is used only to bind the train/validation nested schedules.\n", encoding="utf-8")
        _save_initialization(root)
        rows = []
        for method in METHODS:
            rows.extend(parameter_rows(_load_initial_model(root, torch.device("cpu")), method))
        _atomic_csv(root / "trainable_scope.csv", rows, ["method", "module", "total_params", "trainable_params", "requires_grad", "used_at_inference"])
        write_json(root / "prepare_progress.json", {"stage": "initialization_completed", "at": now()})
        _loss_calibration(root, loaders, topology)
        _write_preflight(root, loaders, topology)
    except Exception as exc:
        write_json(root / "prepare_failure.json", {"at": now(), "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def _pattern_name_groups(pattern: str) -> tuple[str, list[str]]:
    available = sum(PATTERNS[pattern])
    if pattern == "full":
        return "full", []
    missing = [MODALITIES[index] for index, value in enumerate(PATTERNS[pattern]) if not value]
    return {3: "single", 2: "double", 1: "triple"}[available], missing


def _metric_values(logits: torch.Tensor, labels: torch.Tensor, topology: Any) -> torch.Tensor:
    prediction = logits.argmax(dim=-1)
    distance = topology.distance.to(device=logits.device)[labels, prediction].to(torch.float32)
    top = logits.topk(5, dim=-1).indices
    return torch.stack(
        (
            prediction.eq(labels),
            top[:, :3].eq(labels[:, None]).any(dim=-1),
            top.eq(labels[:, None]).any(dim=-1),
            distance.le(3),
            distance,
            F.cross_entropy(logits.float(), labels, reduction="none"),
            topology_risk(logits.float(), labels, topology),
            distance.gt(5),
        ),
        dim=1,
    ).to(torch.float64).cpu()


def _metric_update_values(bucket: dict[str, float], values: torch.Tensor) -> None:
    totals = values.sum(dim=0).tolist()
    bucket["count"] += float(values.shape[0])
    for key, value in zip(
        ("top1_sum", "top3_sum", "top5_sum", "within3_sum", "mae_sum", "ce_sum", "risk_sum", "distance_gt5_sum"),
        totals,
    ):
        bucket[key] += float(value)


def _metric_update(bucket: dict[str, float], logits: torch.Tensor, labels: torch.Tensor, topology: Any) -> None:
    _metric_update_values(bucket, _metric_values(logits, labels, topology))


def _metric_finalize(bucket: dict[str, float]) -> dict[str, float]:
    count = max(bucket["count"], 1.0)
    return {"sample_count": int(bucket["count"]), "top1": bucket["top1_sum"] / count, "top3": bucket["top3_sum"] / count, "top5": bucket["top5_sum"] / count, "within3": bucket["within3_sum"] / count, "mae": bucket["mae_sum"] / count, "ce_loss": bucket["ce_sum"] / count, "topology_risk": bucket["risk_sum"] / count, "distance_gt5_rate": bucket["distance_gt5_sum"] / count}


def _new_bucket() -> dict[str, float]:
    return defaultdict(float)


def _group_indices(values: Iterable[Any]) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        groups[value].append(index)
    return groups


def evaluate_selection(
    model: BTSCLModel,
    loader: Any,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    sums = {pattern: torch.zeros((), dtype=torch.float64, device=device) for pattern in PATTERNS}
    count = 0
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            with _autocast(device):
                inputs, labels = _inputs(batch, device), _labels(batch, device)
                tokens = model.encode(inputs)
                for pattern, availability_values in PATTERNS.items():
                    availability = torch.tensor(availability_values, dtype=torch.bool, device=device).expand(labels.shape[0], -1)
                    logits = model.logits_from_tokens(tokens, availability)[0]
                    sums[pattern] += F.cross_entropy(logits.float(), labels, reduction="sum").to(torch.float64)
            count += int(labels.numel())
            if max_batches is not None and batch_index + 1 >= int(max_batches):
                break
    if count == 0:
        raise ValueError("BT-SCL selection validation produced no samples.")
    return {
        "patterns": {pattern: {"ce_loss": float(value / count)} for pattern, value in sums.items()},
        "sample_count": count,
        "evaluation_seconds": time.monotonic() - started,
    }


def _selection_loss(patterns: dict[str, dict[str, float]]) -> float:
    return 0.25 * (
        patterns["full"]["ce_loss"]
        + np.mean([patterns[name]["ce_loss"] for name in ("missing_image", "missing_lidar", "missing_radar", "missing_gps")])
        + np.mean([patterns[name]["ce_loss"] for name in ("missing_image_lidar", "missing_image_radar", "missing_image_gps", "missing_lidar_radar", "missing_lidar_gps", "missing_radar_gps")])
        + np.mean([patterns[name]["ce_loss"] for name in ("only_image", "only_lidar", "only_radar", "only_gps")])
    )


def evaluate(model: BTSCLModel, loader: Any, schedule: dict[str, Any], topology: Any, device: torch.device, *, include_aux: bool, include_diagnostics: bool, max_batches: int | None = None) -> dict[str, Any]:
    model.eval()
    metrics: dict[str, dict[str, float]] = {pattern: _new_bucket() for pattern in PATTERNS}
    per_domain: dict[tuple[str, str], dict[str, float]] = defaultdict(_new_bucket)
    per_weather: dict[tuple[str, str], dict[str, float]] = defaultdict(_new_bucket)
    per_sector: dict[tuple[str, int], dict[str, float]] = defaultdict(_new_bucket)
    aux: dict[str, dict[str, float]] = {name: _new_bucket() for name in MODALITIES}
    mono = {"pairs": [_new_bucket() for _ in range(3)], "addition": defaultdict(_new_bucket)}
    radius_dominance = {(radius, index): _new_bucket() for radius in (0, 3, 5) for index in range(3)}
    c2f_values: dict[str, list[float]] = defaultdict(list)
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            with _autocast(device):
                inputs, labels = _inputs(batch, device), _labels(batch, device)
                tokens = model.encode(inputs)
                if include_aux:
                    for name, logits in model.auxiliary_logits(tokens).items():
                        _metric_update(aux[name], logits.float(), labels, topology)
                pattern_values: dict[str, torch.Tensor] = {}
                for pattern, values in PATTERNS.items():
                    availability = torch.tensor(values, dtype=torch.bool, device=device).expand(labels.shape[0], -1)
                    logits = model.logits_from_tokens(tokens, availability)[0].float()
                    pattern_values[pattern] = _metric_values(logits, labels, topology)
                    _metric_update_values(metrics[pattern], pattern_values[pattern])
            metadata = batch["metadata"]
            conditions = [str(value) for value in metadata["condition"]]
            scenarios = [str(value) for value in metadata["scenario"]]
            domains = [f"{condition}/{scenario}" for condition, scenario in zip(conditions, scenarios)]
            sectors = ((topology.distance[0, labels.detach().cpu()] // 4) % 8).tolist()
            domain_groups = _group_indices(domains)
            weather_groups = _group_indices(conditions)
            sector_groups = _group_indices(sectors)
            for pattern, values in pattern_values.items():
                for domain, indices in domain_groups.items():
                    _metric_update_values(per_domain[(pattern, domain)], values[indices])
                for weather, indices in weather_groups.items():
                    _metric_update_values(per_weather[(pattern, weather)], values[indices])
                for sector, indices in sector_groups.items():
                    _metric_update_values(per_sector[(pattern, int(sector))], values[indices])
            if include_diagnostics:
                masks = schedule_masks(_batch_ids(batch), schedule, device)
                chain_logits = torch.stack([model.logits_from_tokens(tokens, masks[:, index])[0].float() for index in range(4)], dim=1)
                risks = torch.stack([topology_risk(chain_logits[:, index], labels, topology) for index in range(4)], dim=1)
                probabilities = torch.softmax(chain_logits, dim=-1)
                label_distance = topology.distance.to(device=device)[labels]
                for index in range(3):
                    excess = (risks[:, index + 1] - risks[:, index]).clamp_min(0)
                    bucket = mono["pairs"][index]
                    bucket["count"] += float(labels.numel())
                    bucket["violation_sum"] += float(excess.gt(0).sum())
                    bucket["excess_sum"] += float(excess.sum())
                    added = masks[:, index + 1].to(torch.int64) - masks[:, index].to(torch.int64)
                    for modality_index, name in enumerate(MODALITIES):
                        selected = added[:, modality_index].bool()
                        if bool(selected.any()):
                            item = mono["addition"][(index, name)]
                            item["count"] += float(selected.sum())
                            item["top1_before_sum"] += float(chain_logits[selected, index].argmax(-1).eq(labels[selected]).sum())
                            item["top1_after_sum"] += float(chain_logits[selected, index + 1].argmax(-1).eq(labels[selected]).sum())
                            item["risk_before_sum"] += float(risks[selected, index].sum())
                            item["risk_after_sum"] += float(risks[selected, index + 1].sum())
                            item["violation_sum"] += float(excess[selected].gt(0).sum())
                for radius in (0, 3, 5):
                    mass = (probabilities * label_distance.le(radius)[:, None, :]).sum(dim=-1)
                    for index in range(3):
                        delta = mass[:, index + 1] - mass[:, index]
                        bucket = radius_dominance[(radius, index)]
                        bucket["count"] += float(labels.numel())
                        bucket["violation_sum"] += float(delta.lt(0).sum())
                        bucket["delta_sum"] += float(delta.sum())
                _, detail = coarse_to_fine_loss(chain_logits, labels, topology)
                for name, value in detail.items():
                    c2f_values[name].append(float(value))
            if max_batches is not None and batch_index + 1 >= int(max_batches):
                break
    results = {pattern: _metric_finalize(bucket) for pattern, bucket in metrics.items()}
    payload: dict[str, Any] = {"patterns": results, "evaluation_seconds": time.monotonic() - started}
    payload["per_domain"] = [{"pattern": pattern, "domain": domain, **_metric_finalize(bucket)} for (pattern, domain), bucket in sorted(per_domain.items())]
    payload["per_weather"] = [{"pattern": pattern, "weather": weather, **_metric_finalize(bucket)} for (pattern, weather), bucket in sorted(per_weather.items())]
    payload["per_sector"] = [{"pattern": pattern, "sector": sector, **_metric_finalize(bucket)} for (pattern, sector), bucket in sorted(per_sector.items())]
    if include_aux:
        payload["auxiliary"] = {name: _metric_finalize(bucket) for name, bucket in aux.items()}
    if include_diagnostics:
        pairs = []
        for index, bucket in enumerate(mono["pairs"]):
            count = max(bucket["count"], 1.0)
            pairs.append({"transition": ("S1_to_S2", "S2_to_S3", "S3_to_Full")[index], "sample_count": int(bucket["count"]), "violation_rate": bucket["violation_sum"] / count, "mean_excess_risk": bucket["excess_sum"] / count})
        payload["monotonicity"] = pairs
        payload["modality_addition"] = [
            {"source_subset_size": index + 1, "added_modality": name, "target_subset_size": index + 2, "sample_count": int(bucket["count"]), "top1_before": bucket["top1_before_sum"] / max(bucket["count"], 1.0), "top1_after": bucket["top1_after_sum"] / max(bucket["count"], 1.0), "risk_before": bucket["risk_before_sum"] / max(bucket["count"], 1.0), "risk_after": bucket["risk_after_sum"] / max(bucket["count"], 1.0), "violation_rate": bucket["violation_sum"] / max(bucket["count"], 1.0)}
            for (index, name), bucket in sorted(mono["addition"].items())
        ]
        payload["consistency"] = {name: float(np.mean(values)) for name, values in c2f_values.items()}
        payload["radius_dominance"] = [
            {
                "radius": radius,
                "transition": ("S1_to_S2", "S2_to_S3", "S3_to_Full")[index],
                "sample_count": int(bucket["count"]),
                "violation_rate": bucket["violation_sum"] / max(bucket["count"], 1.0),
                "mean_mass_delta": bucket["delta_sum"] / max(bucket["count"], 1.0),
            }
            for (radius, index), bucket in sorted(radius_dominance.items())
        ]
    return payload


def _summary(patterns: dict[str, dict[str, float]]) -> dict[str, float]:
    groups = {"single": [], "double": [], "triple": [], "all14": []}
    absent = {name: [] for name in MODALITIES}
    for pattern, metric in patterns.items():
        group, missing = _pattern_name_groups(pattern)
        if group != "full":
            groups[group].append(metric)
            groups["all14"].append(metric)
            for name in missing:
                absent[name].append(metric)
    def mean(items: list[dict[str, float]], key: str) -> float:
        return float(np.mean([item[key] for item in items])) if items else float("nan")
    def worst(items: list[dict[str, float]], key: str) -> float:
        return float(np.min([item[key] for item in items])) if items else float("nan")
    result = {"full_top1": patterns["full"]["top1"], "full_within3": patterns["full"]["within3"], "full_mae": patterns["full"]["mae"]}
    for group, items in groups.items():
        result[f"{group}_macro_top1"] = mean(items, "top1")
        result[f"{group}_worst_top1"] = worst(items, "top1")
        result[f"{group}_macro_mae"] = mean(items, "mae")
    for name, items in absent.items():
        result[f"{name}_absent_macro_top1"] = mean(items, "top1")
    result["missing_lidar_top1"] = patterns["missing_lidar"]["top1"]
    result["radar_gps_top1"] = patterns["missing_image_lidar"]["top1"]
    result["overall_within3"] = mean(groups["all14"], "within3")
    result["overall_mae"] = mean(groups["all14"], "mae")
    return result


def _write_evaluation(run_dir: Path, result: dict[str, Any]) -> None:
    patterns = [{"pattern": name, **metric} for name, metric in result["patterns"].items()]
    _atomic_csv(run_dir / "per_pattern_metrics.csv", patterns, ["pattern", "sample_count", "top1", "top3", "top5", "within3", "mae", "ce_loss", "topology_risk", "distance_gt5_rate"])
    counts = []
    for group in ("full", "single", "double", "triple"):
        values = [metric for pattern, metric in result["patterns"].items() if _pattern_name_groups(pattern)[0] == group]
        if values:
            counts.append({"missing_group": group, "patterns": len(values), "top1": float(np.mean([value["top1"] for value in values])), "within3": float(np.mean([value["within3"] for value in values])), "mae": float(np.mean([value["mae"] for value in values]))})
    _atomic_csv(run_dir / "per_missing_count_metrics.csv", counts, ["missing_group", "patterns", "top1", "within3", "mae"])
    for name, key in (("per_domain_metrics.csv", "per_domain"), ("per_weather_metrics.csv", "per_weather"), ("per_sector_metrics.csv", "per_sector")):
        rows = result.get(key, [])
        if rows:
            _atomic_csv(run_dir / name, rows, list(rows[0]))
    error_rows = [{"pattern": name, "mae": value["mae"], "distance_gt5_rate": value["distance_gt5_rate"]} for name, value in result["patterns"].items()]
    _atomic_csv(run_dir / "error_distance_metrics.csv", error_rows, ["pattern", "mae", "distance_gt5_rate"])
    if result.get("auxiliary"):
        rows = [{"modality": name, **metric} for name, metric in result["auxiliary"].items()]
        _atomic_csv(run_dir / "unimodal_probe_metrics.csv", rows, ["modality", "sample_count", "top1", "top3", "top5", "within3", "mae", "ce_loss", "topology_risk", "distance_gt5_rate"])
    if result.get("monotonicity"):
        _atomic_csv(run_dir / "monotonicity_metrics.csv", result["monotonicity"], list(result["monotonicity"][0]))
        _atomic_csv(run_dir / "modality_addition_matrix.csv", result["modality_addition"], list(result["modality_addition"][0]))
    if result.get("consistency"):
        _atomic_csv(run_dir / "consistency_diagnostics.csv", [{"metric": key, "value": value} for key, value in result["consistency"].items()], ["metric", "value"])
    if result.get("radius_dominance"):
        _atomic_csv(run_dir / "radius_dominance_metrics.csv", result["radius_dominance"], list(result["radius_dominance"][0]))


def _r6_stable_assessment(root: Path) -> None:
    r0_path = root / "r0_subset_task_only/metrics.json"
    r6_path = root / "r6_topological_stochastic_dominance/metrics.json"
    reference_path = root / "mechanism_references/r0_subset_task_only/metrics.json"
    if not all(path.is_file() for path in (r0_path, r6_path, reference_path)):
        return

    r0 = json.loads(r0_path.read_text(encoding="utf-8"))
    r6 = json.loads(r6_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    s0, s6 = r0["summary"], r6["summary"]
    missing = [name for name in PATTERNS if name != "full"]

    def macro(payload: dict[str, Any], key: str) -> float:
        return float(np.mean([payload["patterns"][name][key] for name in missing]))

    def grouped_top1(payload: dict[str, Any], group_key: str) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for row in payload[f"per_{group_key}"]:
            if row["pattern"] != "full":
                values[str(row[group_key])].append(float(row["top1"]))
        return {key: float(np.mean(items)) for key, items in values.items()}

    pattern_deltas = {
        name: r6["patterns"][name]["top1"] - r0["patterns"][name]["top1"]
        for name in missing
    }
    weather0, weather6 = grouped_top1(r0, "weather"), grouped_top1(r6, "weather")
    sector0, sector6 = grouped_top1(r0, "sector"), grouped_top1(r6, "sector")
    weather_nonworse = sum(weather6[key] >= weather0[key] for key in weather0)
    sector_nonworse = sum(sector6[key] >= sector0[key] for key in sector0)

    gate4_checks = {
        "missing_lidar": s6["missing_lidar_top1"] - s0["missing_lidar_top1"] >= 0.0075,
        "radar_gps": s6["radar_gps_top1"] - s0["radar_gps_top1"] >= 0.01,
        "lidar_absent": s6["lidar_absent_macro_top1"] - s0["lidar_absent_macro_top1"] >= 0.0075,
        "all14_worst": s6["all14_worst_top1"] - s0["all14_worst_top1"] >= 0.005,
        "patterns_nonworse": sum(delta >= 0 for delta in pattern_deltas.values()) >= 10,
        "max_pattern_drop": min(pattern_deltas.values()) >= -0.0075,
    }
    radius0 = {
        (int(row["radius"]), row["transition"]): float(row["violation_rate"])
        for row in reference["radius_dominance"]
    }
    radius6 = {
        (int(row["radius"]), row["transition"]): float(row["violation_rate"])
        for row in r6["radius_dominance"]
    }
    mean_violation0 = float(np.mean(list(radius0.values())))
    mean_violation6 = float(np.mean(list(radius6.values())))
    violation_relative_delta = (mean_violation6 - mean_violation0) / mean_violation0
    radius_nonworse = sum(radius6[key] <= radius0[key] for key in radius0)

    gate_rows = [
        {
            "gate": "1_full_preservation",
            "passed": s6["full_top1"] >= s0["full_top1"] - 0.003
            and s6["full_within3"] >= s0["full_within3"] - 0.002
            and s6["full_mae"] <= s0["full_mae"] + 0.03,
            "evidence": f"Top1 {100*(s6['full_top1']-s0['full_top1']):+.3f} pp; Within3 {100*(s6['full_within3']-s0['full_within3']):+.3f} pp; MAE {s6['full_mae']-s0['full_mae']:+.3f}",
        },
        {
            "gate": "2_all14_gain",
            "passed": s6["all14_macro_top1"] >= s0["all14_macro_top1"] + 0.005
            and s6["all14_macro_mae"] <= s0["all14_macro_mae"],
            "evidence": f"Top1 {100*(s6['all14_macro_top1']-s0['all14_macro_top1']):+.3f} pp; MAE {s6['all14_macro_mae']-s0['all14_macro_mae']:+.3f}",
        },
        {
            "gate": "3_missing_severity",
            "passed": sum(
                (
                    s6["single_macro_top1"] - s0["single_macro_top1"] >= 0.003,
                    s6["double_macro_top1"] - s0["double_macro_top1"] >= 0.005,
                    s6["triple_macro_top1"] - s0["triple_macro_top1"] >= 0.005,
                )
            ) >= 2
            and min(
                s6["single_macro_top1"] - s0["single_macro_top1"],
                s6["double_macro_top1"] - s0["double_macro_top1"],
                s6["triple_macro_top1"] - s0["triple_macro_top1"],
            ) >= -0.002,
            "evidence": f"Single {100*(s6['single_macro_top1']-s0['single_macro_top1']):+.3f} pp; Double {100*(s6['double_macro_top1']-s0['double_macro_top1']):+.3f} pp; Triple {100*(s6['triple_macro_top1']-s0['triple_macro_top1']):+.3f} pp",
        },
        {
            "gate": "4_difficult_patterns",
            "passed": sum(gate4_checks.values()) >= 3,
            "evidence": f"{sum(gate4_checks.values())}/6 subcriteria; {sum(delta >= 0 for delta in pattern_deltas.values())}/14 patterns nonworse; worst delta {100*min(pattern_deltas.values()):+.3f} pp",
        },
        {
            "gate": "5_topology_stability",
            "passed": s6["overall_within3"] >= s0["overall_within3"]
            and s6["overall_mae"] <= s0["overall_mae"]
            and macro(r6, "topology_risk") < macro(r0, "topology_risk")
            and macro(r6, "distance_gt5_rate") <= macro(r0, "distance_gt5_rate")
            and weather_nonworse >= 2
            and sector_nonworse >= 5,
            "evidence": f"Within3 {100*(s6['overall_within3']-s0['overall_within3']):+.3f} pp; MAE {s6['overall_mae']-s0['overall_mae']:+.3f}; topology risk {macro(r6, 'topology_risk')-macro(r0, 'topology_risk'):+.3f}; distance>5 {macro(r6, 'distance_gt5_rate')-macro(r0, 'distance_gt5_rate'):+.3f}; weather {weather_nonworse}/3; sectors {sector_nonworse}/8",
        },
        {
            "gate": "6_r6_mechanism",
            "passed": violation_relative_delta <= -0.1
            and radius_nonworse >= 6
            and sum(
                (
                    s6["all14_macro_top1"] > s0["all14_macro_top1"],
                    s6["overall_within3"] > s0["overall_within3"],
                    s6["overall_mae"] < s0["overall_mae"],
                )
            ) >= 2,
            "evidence": f"mean radius violation {100*violation_relative_delta:+.2f}%; nonworse transitions {radius_nonworse}/9; main Top1/Within3/MAE all improve",
        },
    ]
    _atomic_csv(root / "success_gates.csv", gate_rows, ["gate", "passed", "evidence"])

    recomputed = [_summary(payload["patterns"]) for payload in (r0, r6)]
    max_recompute_diff = max(
        abs(float(payload["summary"][key]) - float(summary[key]))
        for payload, summary in zip((r0, r6), recomputed)
        for key in summary
    )
    only_radar_delta = pattern_deltas["only_radar"]
    passed = sum(bool(row["passed"]) for row in gate_rows)
    lines = [
        "# Stable R6 assessment",
        "",
        "Post-hoc development-only, single seed, claim-ineligible. Outer test was not accessed.",
        "",
        f"R6 passes {passed}/6 gates against the same-profile stable R0. Independent summary reaggregation max absolute difference: {max_recompute_diff:.3g}.",
        "",
        "| Gate | Passed | Evidence |",
        "|---|---:|---|",
    ]
    lines.extend(f"| {row['gate']} | {str(row['passed']).lower()} | {row['evidence']} |" for row in gate_rows)
    lines.extend(
        (
            "",
            f"Important local failure: Radar Only changes by {100*only_radar_delta:+.3f} pp, so the per-pattern maximum-drop subcriterion fails even though Gate 4 passes 5/6 subcriteria.",
            "",
            "The stable R0 is a controlled optimizer-profile baseline, not the historical canonical anchor. Claims must also compare R6 descriptively with the original R0 and must not promote this post-hoc result without a preregistered multi-seed rerun.",
        )
    )
    (root / "stable_r6_assessment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(
    root: Path,
    method: str,
    *,
    smoke: bool = False,
    prepared_root: Path | None = None,
    stable_profile: bool = False,
) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown BT-SCL method {method!r}")
    if stable_profile and method not in STABLE_METHODS:
        raise ValueError(f"stable BT-SCL follow-up only supports {sorted(STABLE_METHODS)}")
    artifacts = prepared_root or root
    if not (artifacts / "preflight_tests.txt").is_file():
        raise ValueError("BT-SCL preflight has not completed; run --prepare first.")
    run_dir = root / "smoke_tests" / method if smoke else root / method
    if run_dir.exists() and not smoke:
        existing = {path.name for path in run_dir.iterdir()}
        if existing - {"train.log"}:
            raise FileExistsError(f"BT-SCL run directory already contains artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_full_pool_protocol(DEFAULT_PROTOCOL)
    topology = load_audited_topology(DEFAULT_TOPOLOGY)
    loaders, cfg = _loaders(artifacts, protocol, create_normalization=False)
    if smoke:
        loaders = {
            "train": _small_train_only_loader(loaders["train"]),
            "validation": _small_train_only_loader(loaders["validation"]),
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    model = _load_initial_model(artifacts, device)
    for parameter in model.auxiliary.parameters():
        parameter.requires_grad_(method in {"r1_available_evidence", "r5_full_bt_scl"})
    weights = _loss_calibration(artifacts, loaders, topology)
    if stable_profile:
        weights.update(uni=0.0, mono=0.0, c2f=0.25, local=0.2, hierarchy=0.25, dominance=1.0)
    encoder_lr = 1e-5 if stable_profile else 1e-4
    main_lr = 3e-5 if stable_profile else 3e-4
    weight_decay = 1e-3 if stable_profile else 1e-4
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoders.parameters()), "lr": encoder_lr},
            {"params": list(model.projections.parameters()) + list(model.fusion.parameters()) + list(model.prototype_bank.parameters()) + [model.modality_embedding, model.time_embedding], "lr": main_lr},
            {"params": list(model.auxiliary.parameters()), "lr": main_lr},
        ],
        weight_decay=weight_decay,
    )
    total_epochs = 1 if smoke else (STABLE_EPOCHS if stable_profile else EPOCHS)
    total_steps = max(1, total_epochs * len(loaders["train"]))
    warmup_fraction = 0.10 if stable_profile else 0.05
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, int(total_steps * warmup_fraction))) * 0.5 * (1 + math.cos(math.pi * min(step + 1, total_steps) / total_steps)))
    train_schedule = json.loads((artifacts / "train_nested_subset_schedule.json").read_text(encoding="utf-8"))
    validation_schedule = json.loads((artifacts / "validation_nested_subset_manifest.json").read_text(encoding="utf-8"))
    resolved = {"method": method, "profile": "post_hoc_stable_v1" if stable_profile else "original_v1", "seed": SEED, "epochs": total_epochs, "model": MODEL_CONFIG, "weights": weights, "optimizer": "AdamW", "encoder_lr": encoder_lr, "main_lr": main_lr, "weight_decay": weight_decay, "warmup_fraction": warmup_fraction, "scheduler": "cosine_warmup", "batch_size": cfg["data"]["dataloader"]["train_batch_size"], "prepared_root": str(artifacts), "protocol_fingerprint": protocol["protocol_fingerprint"], "topology_descriptor_sha256": topology.descriptor_sha256, "schedule_sha256": train_schedule["schedule_sha256"], "outer_test_accessed": False, "legacy_clean_inner_used": False, "claim_eligible": False, "early_stopping": {"enabled": stable_profile, "metric": "validation_selection_loss", "min_epochs": 6, "patience": 4, "relative_min_delta": 0.001}}
    if method == "r6_topological_stochastic_dominance":
        write_json(run_dir / "r6_loss_definition.json", {"hierarchy_sectors": [4, 8, 16], "dominance_radii": [0, 3, 5], "hierarchy_weight": weights["hierarchy"], "dominance_weight": weights["dominance"], "teacher_kl": False, "dynamic_weighting": False})
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    (run_dir / "config_sha256.txt").write_text(sha256_file(run_dir / "resolved_config.yaml") + "\n", encoding="utf-8")
    log_rows, best_selection, best_epoch, best_state = [], float("inf"), 0, None
    early_best, stale_epochs, stop_reason = float("inf"), 0, "max_epochs"
    started = time.monotonic()
    try:
        for epoch in range(1, total_epochs + 1):
            epoch_started = time.monotonic()
            model.train()
            totals, sample_count = defaultdict(float), 0
            for batch_index, batch in enumerate(loaders["train"]):
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    inputs, labels = _inputs(batch, device), _labels(batch, device)
                    masks = schedule_masks(_batch_ids(batch), train_schedule, device)
                    views = model.forward_views(inputs, masks)
                    loss, report = btscl_losses(model, views, labels, topology, method, weights)
                loss.backward()
                phase = "joint"
                if method == "r5_full_bt_scl":
                    phase = ("image", "lidar", "radar", "gps", "joint", "joint", "joint", "joint")[batch_index % 8]
                    if phase != "joint":
                        allowed = (f"projections.{phase}.", f"auxiliary.{phase}.")
                        if phase in {"image", "lidar", "radar"}:
                            allowed = allowed + (f"encoders.{phase}.net.6.", f"encoders.{phase}.net.7.", f"encoders.{phase}.net.8.", f"encoders.{phase}.output.")
                        else:
                            allowed = allowed + ("encoders.gps.",)
                        for name, parameter in model.named_parameters():
                            if parameter.grad is not None and not name.startswith(allowed):
                                parameter.grad = None
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                count = int(labels.numel())
                sample_count += count
                for name, value in report.items():
                    totals[name] += value * count
                if smoke and batch_index >= 1:
                    break
            train_seconds = time.monotonic() - epoch_started
            evaluation = evaluate_selection(model, loaders["validation"], device, max_batches=1 if smoke else None)
            selection_patterns = evaluation["patterns"]
            selection = _selection_loss(selection_patterns)
            log_rows.append({"epoch": epoch, "train_samples": sample_count, "train_total_loss": totals["total"] / max(sample_count, 1), "validation_selection_loss": selection, "lr": optimizer.param_groups[0]["lr"], "train_seconds": train_seconds, "selection_seconds": evaluation["evaluation_seconds"], "epoch_seconds": time.monotonic() - epoch_started, "r5_phase_cycle": "I:L:R:G:Joint=1:1:1:1:4" if method == "r5_full_bt_scl" else "joint"})
            curve_fields = ["epoch", "train_samples", "train_total_loss", "validation_selection_loss", "lr", "train_seconds", "selection_seconds", "epoch_seconds", "r5_phase_cycle"]
            _atomic_csv(run_dir / "training_curve.csv", log_rows, curve_fields)
            print(json.dumps({"event": "bt_scl_epoch", "method": method, **log_rows[-1]}), flush=True)
            if selection < best_selection:
                best_selection, best_epoch = selection, epoch
                best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            if smoke:
                break
            if stable_profile:
                if selection < early_best * (1.0 - 0.001):
                    early_best, stale_epochs = selection, 0
                else:
                    stale_epochs += 1
                if epoch >= 6 and stale_epochs >= 4:
                    stop_reason = "validation_patience"
                    break
        if best_state is None:
            raise RuntimeError("BT-SCL never produced a checkpoint selection state.")
        checkpoint = run_dir / "best.pt"
        torch.save({"state_dict": best_state, "epoch": best_epoch, "selection_loss": best_selection, "method": method, "initialization_sha256": sha256_file(artifacts / "initialization/initial_model.pt"), "protocol_fingerprint": protocol["protocol_fingerprint"]}, checkpoint)
        (run_dir / "checkpoint_sha256.txt").write_text(sha256_file(checkpoint) + "\n", encoding="utf-8")
        model.load_state_dict(best_state)
        final = evaluate(model, loaders["validation"], validation_schedule, topology, device, include_aux=method in {"r1_available_evidence", "r5_full_bt_scl"}, include_diagnostics=method in {"r2_topology_monotonicity", "r3_coarse_to_fine", "r4_mono_c2f", "r5_full_bt_scl", "r6_topological_stochastic_dominance"}, max_batches=1 if smoke else None)
        final["summary"] = _summary(final["patterns"])
        final["best_epoch"] = best_epoch
        final["selection_loss"] = best_selection
        final["method"] = method
        final["wall_seconds"] = time.monotonic() - started
        final["peak_memory_mib"] = float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
        write_json(run_dir / "metrics.json", final)
        _write_evaluation(run_dir, final)
        _atomic_csv(run_dir / "training_curve.csv", log_rows, curve_fields)
        completed_epochs = len(log_rows)
        write_json(run_dir / "efficiency.json", {"total_params": sum(parameter.numel() for parameter in model.parameters()), "trainable_params": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "peak_gpu_memory_mib": final["peak_memory_mib"], "wall_seconds": final["wall_seconds"], "epochs": completed_epochs, "samples_per_second": (completed_epochs * TRAIN_WINDOWS) / max(final["wall_seconds"], 1e-8)})
        write_json(run_dir / "status.json", {"status": "passed", "method": method, "profile": resolved["profile"], "epochs_completed": completed_epochs, "best_epoch": best_epoch, "stop_reason": "smoke" if smoke else stop_reason, "outer_test_accessed": False, "legacy_clean_inner_used": False, "claim_eligible": False, "smoke": smoke})
    except Exception as exc:
        write_json(run_dir / "status.json", {"status": "failed", "method": method, "error": repr(exc), "outer_test_accessed": False, "smoke": smoke})
        raise
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def diagnose_checkpoint(root: Path, method: str, *, prepared_root: Path | None = None) -> None:
    """Read-only mechanism evaluation for an existing checkpoint."""
    checkpoint_path = root / method / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BT-SCL checkpoint is absent: {checkpoint_path}")
    output = root / "mechanism_references" / method
    if output.exists():
        raise FileExistsError(f"BT-SCL mechanism reference already exists: {output}")
    artifacts = prepared_root or root
    protocol = load_full_pool_protocol(DEFAULT_PROTOCOL)
    topology = load_audited_topology(DEFAULT_TOPOLOGY)
    loaders, _ = _loaders(artifacts, protocol, create_normalization=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_initial_model(artifacts, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    schedule = json.loads((artifacts / "validation_nested_subset_manifest.json").read_text(encoding="utf-8"))
    try:
        result = evaluate(model, loaders["validation"], schedule, topology, device, include_aux=False, include_diagnostics=True)
        result.update(method=method, checkpoint_sha256=sha256_file(checkpoint_path), outer_test_accessed=False)
        write_json(output / "metrics.json", result)
        _write_evaluation(output, result)
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def aggregate(root: Path) -> None:
    rows = []
    for method in METHODS:
        path = root / method / "metrics.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        mono = payload.get("monotonicity", [])
        rows.append({"method": method, **payload["summary"], "mono_violation": float(np.mean([item["violation_rate"] for item in mono])) if mono else float("nan")})
    if not rows:
        raise ValueError("BT-SCL has no completed metrics to aggregate.")
    fields = list(rows[0])
    _atomic_csv(root / "combined_metrics.csv", rows, fields)
    ranking = sorted(rows, key=lambda row: row["all14_macro_top1"], reverse=True)
    _atomic_csv(root / "direction_ranking.csv", [{"rank": index + 1, **row} for index, row in enumerate(ranking)], ["rank", *fields])
    lines = ["# Full-Pool BT-SCL Comparison", "", "Development-only, single seed. Outer test was not accessed.", "", "| Method | Full Top1 | Single Macro | Double Macro | Triple Macro | All-14 Macro | Overall Within-3 | Overall MAE | Mono Violation |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['method']} | {row['full_top1']:.4f} | {row['single_macro_top1']:.4f} | {row['double_macro_top1']:.4f} | {row['triple_macro_top1']:.4f} | {row['all14_macro_top1']:.4f} | {row['overall_within3']:.4f} | {row['overall_mae']:.4f} | {row['mono_violation']:.4f} |")
    lines.extend(("", "Protocol: 37,038 train, 9,180 validation; legacy clean-inner=false; outer test=false; channel/path model input=false.", "", "No multi-seed, outer-test, or follow-up experiment was launched automatically."))
    (root / "full_pool_bt_scl_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _r6_stable_assessment(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--diagnose-checkpoint", choices=METHODS)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--stable-profile", action="store_true")
    args = parser.parse_args()
    if not any((args.prepare, args.method, args.aggregate, args.diagnose_checkpoint)):
        parser.error("select --prepare, --method, --diagnose-checkpoint, or --aggregate")
    if args.prepare:
        prepare(args.output_root.resolve())
    if args.method:
        train(
            args.output_root.resolve(),
            args.method,
            smoke=args.smoke,
            prepared_root=args.prepared_root.resolve() if args.prepared_root else None,
            stable_profile=args.stable_profile,
        )
    if args.aggregate:
        aggregate(args.output_root.resolve())
    if args.diagnose_checkpoint:
        diagnose_checkpoint(
            args.output_root.resolve(),
            args.diagnose_checkpoint,
            prepared_root=args.prepared_root.resolve() if args.prepared_root else None,
        )


if __name__ == "__main__":
    main()
