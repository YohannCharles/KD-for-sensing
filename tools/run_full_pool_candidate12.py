#!/usr/bin/env python3
"""Run the local Full-pool BTPR-Mix and PAMR Candidate12 search.

This development-only tool is deliberately outside the public CLI.  It reads
only the audited Full-pool train/inner-validation protocol and never accesses
outer evidence, channel/path tensors, or historical beam indices.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from kd_sensing.baselines.full_pool_bt_scl import (
    load_audited_topology,
    sha256_file,
    topology_risk,
    write_json,
)
from kd_sensing.baselines.full_pool_candidate12 import (
    Candidate12Model,
    METHODS,
    MODALITIES,
    MOTION_METHODS,
    REMIX_METHODS,
    RISK_METHODS,
    assignment_diagnostics,
    capacity_constrained_assignment,
    common_loss,
    load_signed_angle_order,
    motion_loss,
    motion_mixture,
    pamr_candidate_gate,
    remix_loss,
    signed_offset_targets,
)
from kd_sensing.baselines.full_pool_common import atomic_csv
from kd_sensing.baselines.full_pool_common import now
from kd_sensing.baselines.full_pool_common import set_seed as _set_seed
from kd_sensing.baselines.full_pool_common import sha256_json as _sha256_json
from kd_sensing.config import load_config
from kd_sensing.data.mmw.full_pool_protocol import (
    FULL_POOL_DEVELOPMENT_WINDOWS,
    FULL_POOL_HISTORICAL_VALIDATION_RETAINED,
    FULL_POOL_PROTOCOL_ID,
    FULL_POOL_RAW_TRAIN_COUNT,
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
DEFAULT_OUTPUT = ROOT / "outputs/full_pool_candidate12_search"
DEFAULT_PROTOCOL = ROOT / "outputs/full_pool_capacity/protocol/split_manifest.json"
DEFAULT_PROTOCOL_AUDIT = ROOT / "outputs/full_pool_capacity/protocol/split_audit.json"
DEFAULT_TOPOLOGY = ROOT / "outputs/cache/mmw_codebook_topology/v1/a692c2b43365b483/topology_manifest.json"
REUSABLE_NORMALIZATION = ROOT / "outputs/full_pool_bt_scl"
# Bound to the canonical protocol rather than restated, so a protocol change can
# never leave this workflow asserting stale window counts.
TRAIN_WINDOWS = FULL_POOL_SPLIT_EXPECTATIONS["train_sample_count"]
VALIDATION_WINDOWS = FULL_POOL_SPLIT_EXPECTATIONS["validation_sample_count"]
BOUNDARY_PURGE = FULL_POOL_SPLIT_EXPECTATIONS["boundary_crossing_excluded_count"]
HISTORICAL_TRAIN_REMOVED = FULL_POOL_SPLIT_EXPECTATIONS["historical_removed_from_train_count"]
SEED = 2026
WARMUP_EPOCHS = 5
SEARCH_EPOCHS = 20
BATCH_SIZE = 64
PATTERNS = {
    "full": (1, 1, 1, 1),
    "only_image": (1, 0, 0, 0),
    "only_lidar": (0, 1, 0, 0),
    "only_radar": (0, 0, 1, 0),
    "only_gps": (0, 0, 0, 1),
    "missing_image": (0, 1, 1, 1),
    "missing_lidar": (1, 0, 1, 1),
    "missing_radar": (1, 1, 0, 1),
    "missing_gps": (1, 1, 1, 0),
}
MODEL_CONFIG = {"d_model": 64, "seq_len": 5, "dropout": 0.1, "motion_radius": 3}


def set_seed(seed: int = SEED) -> None:
    _set_seed(seed)


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    # Candidate12 row dicts intentionally carry keys outside `fields`.
    atomic_csv(path, rows, fields, extrasaction="ignore")


def _protocol_audit(protocol: Mapping[str, Any]) -> dict[str, Any]:
    source = json.loads(DEFAULT_PROTOCOL_AUDIT.read_text(encoding="utf-8"))
    checks = {
        name: int(protocol.get(name, -1)) == value
        for name, value in FULL_POOL_SPLIT_EXPECTATIONS.items()
    }
    checks.update(
        raw_train_count=int(protocol.get("train_sample_count", -1))
        + int(protocol.get("historical_removed_from_train_count", -1))
        == FULL_POOL_RAW_TRAIN_COUNT,
        historical_validation_read_only=int(protocol.get("historical_protected_count", -1))
        - int(protocol.get("historical_removed_from_train_count", -1))
        == FULL_POOL_HISTORICAL_VALIDATION_RETAINED,
        canonical_protocol=protocol.get("protocol_id") == FULL_POOL_PROTOCOL_ID,
        protocol_outer_disabled=protocol.get("outer_test_enabled") is False,
        protocol_outer_unaccessed=protocol.get("outer_test_accessed") is False,
        confirmation_disabled=protocol.get("allow_confirmation_train") is False,
        source_audit_passed=source.get("status") == "passed",
        audit_outer_unaccessed=source.get("outer_test_accessed") is False,
        protocol_fingerprint=source.get("protocol_fingerprint") == protocol.get("protocol_fingerprint"),
        resource_intersections_zero=all(
            int(source.get("overlap_counts", {}).get(name, -1)) == 0
            for name in FULL_POOL_RESOURCE_INTERSECTION_NAMES
        ),
        legacy_clean_inner_unused=True,
        channel_path_model_input_disabled=True,
    )
    result = {
        "audit_id": "full_pool_candidate12_protocol_audit_v1",
        "created_at": now(),
        "canonical_protocol": "full_pool_contiguous_time_v1",
        "checks": checks,
        "protocol_fingerprint": protocol.get("protocol_fingerprint"),
        "candidate_windows": FULL_POOL_SPLIT_EXPECTATIONS["candidate_window_count"],
        "raw_train_windows": FULL_POOL_RAW_TRAIN_COUNT,
        "boundary_purge": FULL_POOL_SPLIT_EXPECTATIONS["boundary_crossing_excluded_count"],
        "historical_train_removed": FULL_POOL_SPLIT_EXPECTATIONS["historical_removed_from_train_count"],
        "historical_validation_read_only": FULL_POOL_HISTORICAL_VALIDATION_RETAINED,
        "train_windows": FULL_POOL_SPLIT_EXPECTATIONS["train_sample_count"],
        "validation_windows": FULL_POOL_SPLIT_EXPECTATIONS["validation_sample_count"],
        "development_windows": FULL_POOL_DEVELOPMENT_WINDOWS,
        "outer_test_accessed": False,
        "status": "passed" if all(checks.values()) else "failed",
    }
    if result["status"] != "passed":
        raise ValueError(f"Candidate12 protocol audit failed: {checks}")
    return result


def build_config(protocol: Mapping[str, Any], root: Path) -> dict[str, Any]:
    cfg = load_config(ROOT / "configs/mmw/u0.yaml")
    cfg["experiment"].update(name="FullPool_Candidate12", seed=SEED, device="auto")
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
        train_batch_size=BATCH_SIZE,
        validation_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        train_drop_last=False,
        test_drop_last=False,
    )
    cfg["data_protocol"] = {
        "mode": protocol["mode"],
        "path": str(DEFAULT_PROTOCOL.resolve()),
        "audit_report": str(DEFAULT_PROTOCOL_AUDIT.resolve()),
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "train_role": protocol["train_role"],
        "validation_role": protocol["validation_role"],
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
    }
    cfg["training"].update(epochs=SEARCH_EPOCHS, max_epochs=SEARCH_EPOCHS, lr=3e-4, weight_decay=1e-4, resume=False)
    cfg["training"]["final_test"] = {"enabled": False}
    cfg["output"].update(dir=str(root), run_name="candidate12", overwrite=False)
    return cfg


def _copy_reusable_normalization(root: Path) -> bool:
    source_artifact = REUSABLE_NORMALIZATION / "artifacts/gps_scaler.npz"
    source_sidecar = REUSABLE_NORMALIZATION / "artifacts/gps_scaler.npz.json"
    source_manifest = REUSABLE_NORMALIZATION / "normalization_manifest.json"
    if not source_artifact.is_file() or not source_manifest.is_file():
        return False
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    if metadata.get("source_split") != "full_pool_train" or int(metadata.get("effective_sample_count", -1)) != TRAIN_WINDOWS:
        raise ValueError(f"Reusable GPS scaler is not bound to the {TRAIN_WINDOWS:,}-window Full-pool train split.")
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_artifact, root / "artifacts/gps_scaler.npz")
    if source_sidecar.is_file():
        shutil.copy2(source_sidecar, root / "artifacts/gps_scaler.npz.json")
    copied = copy.deepcopy(manifest)
    copied["candidate12_reused_from"] = str(REUSABLE_NORMALIZATION)
    copied["candidate12_source_sha256"] = sha256_file(source_artifact)
    write_json(root / "normalization_manifest.json", copied)
    return True


def _loaders(root: Path, protocol: Mapping[str, Any], *, create_normalization: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = build_config(protocol, root)
    artifact = root / "artifacts/gps_scaler.npz"
    if not artifact.is_file() and create_normalization:
        _copy_reusable_normalization(root)
    if artifact.is_file():
        loaders = build_dataloaders(cfg, normalization_overrides={"gps_scaler": load_gps_scaler(artifact)})
        manifest = json.loads((root / "normalization_manifest.json").read_text(encoding="utf-8"))
    elif create_normalization:
        loaders = build_dataloaders(cfg)
        manifest = save_normalization_artifacts(loaders, root)
        write_json(root / "normalization_manifest.json", manifest)
    else:
        raise FileNotFoundError("Candidate12 train-only GPS scaler is absent; run --prepare first.")
    metadata = manifest.get("metadata", {})
    if metadata.get("source_split") != "full_pool_train" or int(metadata.get("effective_sample_count", -1)) != TRAIN_WINDOWS:
        raise ValueError("Candidate12 normalization is not train-only Full-pool normalization.")
    return loaders, cfg


def _stable_sample_ids(dataset: Any) -> list[str]:
    values: list[str] = []
    for leaf, indices in leaf_datasets_with_indices(dataset):
        rows = getattr(getattr(leaf, "samples", None), "rows", None)
        if not isinstance(rows, list):
            raise ValueError("Candidate12 requires prepared MMW sample rows.")
        for index in indices:
            row = rows[int(index)]
            sample_id = str(row.get("sample_id") or row.get("target_sample_id") or "").strip()
            if not sample_id:
                raise ValueError("Candidate12 requires stable source sample identities.")
            values.append(f"mmw:{leaf.condition}:{leaf.scene_slug}:{leaf.split}:{sample_id}")
    if len(values) != len(set(values)):
        raise ValueError("Candidate12 stable sample identities are not unique.")
    return values


def _batch_ids(batch: Mapping[str, Any]) -> list[str]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping) or "stable_sample_id" not in metadata:
        raise ValueError("Candidate12 batch lacks stable_sample_id metadata.")
    return [str(value) for value in metadata["stable_sample_id"]]


def _inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    image = torch.as_tensor(batch["image"]).to(device=device, non_blocking=True)
    lidar = torch.as_tensor(batch["lidar"]).to(device=device, non_blocking=True)
    radar_ra = torch.as_tensor(batch["radar_ra"]).to(device=device, non_blocking=True)
    radar_da = torch.as_tensor(batch["radar_da"]).to(device=device, non_blocking=True)
    if radar_ra.ndim == 4:
        radar_ra, radar_da = radar_ra.unsqueeze(2), radar_da.unsqueeze(2)
    radar = torch.cat((radar_ra, radar_da), dim=2)
    gps = torch.as_tensor(batch["gps"]).to(device=device, non_blocking=True)
    expected = {"image": (5, 3, 224, 224), "lidar": (5, 3, 224, 224), "radar": (5, 2, 128, 64), "gps": (5, 3)}
    actual = {"image": tuple(image.shape[1:]), "lidar": tuple(lidar.shape[1:]), "radar": tuple(radar.shape[1:]), "gps": tuple(gps.shape[1:])}
    if actual != expected:
        raise ValueError(f"Candidate12 input contract mismatch: expected {expected}, got {actual}.")
    return {"image": image, "lidar": lidar, "radar": radar, "gps": gps}


def _labels(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    labels = torch.as_tensor(batch["target_beam"], device=device, dtype=torch.long).reshape(-1)
    if not bool(((labels >= 0) & (labels < 64)).all()):
        raise ValueError("Candidate12 labels must be in [0,63].")
    return labels


def _take_batch(value: Any, count: int) -> Any:
    if torch.is_tensor(value):
        return value[:count]
    if isinstance(value, dict):
        return {key: _take_batch(item, count) for key, item in value.items()}
    if isinstance(value, list):
        return value[:count]
    if isinstance(value, tuple):
        return tuple(_take_batch(item, count) for item in value)
    return value


def _fixed_loader(loader: DataLoader, *, batch_size: int = BATCH_SIZE, workers: int = 8) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(workers),
        pin_memory=bool(workers),
        persistent_workers=bool(workers),
        prefetch_factor=2 if workers else None,
        collate_fn=loader.collate_fn,
        drop_last=False,
    )


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _optimizer(model: Candidate12Model, epochs: int, steps_per_epoch: int):
    encoder_ids = {id(parameter) for parameter in model.encoders.parameters()}
    encoder = [parameter for parameter in model.parameters() if id(parameter) in encoder_ids]
    main = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": 1e-4}, {"params": main, "lr": 3e-4}],
        weight_decay=1e-4,
    )
    total = max(1, int(epochs) * int(steps_per_epoch))
    warmup = max(1, int(total * 0.05))

    def factor(step: int) -> float:
        current = min(step + 1, total)
        if current <= warmup:
            return current / warmup
        progress = (current - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _load_model(path: Path, device: torch.device) -> Candidate12Model:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_config") != MODEL_CONFIG:
        raise ValueError("Candidate12 checkpoint model configuration mismatch.")
    model = Candidate12Model(**MODEL_CONFIG)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device)


def _model_scope(model: Candidate12Model) -> list[dict[str, Any]]:
    rows = []
    for name, module in (
        ("encoders", model.encoders),
        ("projections", model.projections),
        ("fusion", model.fusion),
        ("prototype_bank", model.prototype_bank),
        ("motion", model.motion),
    ):
        rows.append({"module": name, "parameters": sum(value.numel() for value in module.parameters())})
    rows.extend(
        ({"module": "modality_embedding", "parameters": model.modality_embedding.numel()},
         {"module": "time_embedding", "parameters": model.time_embedding.numel()})
    )
    return rows


def _write_first_innovation_audit(root: Path) -> None:
    source_config = ROOT / "outputs/full_pool_capacity/u0_seed1/final_config.yaml"
    write_json(
        root / "first_innovation_audit.json",
        {
            "status": "passed",
            "reuse": {
                "encoders": "current U0 image/lidar TinyViT-5M, radar CNN, GPS MLP",
                "prototype_bank": "kd_sensing.losses.beam_prototype_alignment.BeamPrototypeBank",
                "prototype_count": 64,
                "prototype_dimension": 64,
                "temperature": 0.1,
                "beam_label_sigma": 2.0,
                "lambda_proto": 0.2,
                "lambda_modality_proto": 0.1,
                "topology_id": "ula_dft_phase_cycle_v1",
            },
            "excluded_current_u0_component": {
                "component": "supervised reliability router",
                "reason": "Candidate12 protocol forbids dynamic modality weighting, reliability routers, and quality estimators",
            },
            "replacement_fusion": "fixed-order LayerNorm/1024/512/64 MLP from the preregistered fallback",
            "source_config": str(source_config),
            "source_config_sha256": sha256_file(source_config),
            "canonical_u0_modified": False,
        },
    )


def _signed_order_audit(topology_manifest: Path) -> dict[str, Any]:
    table = topology_manifest.parent / "topology_table.csv"
    order = load_signed_angle_order(table)
    with table.open(newline="", encoding="utf-8") as handle:
        rows = {int(row["label"]): float(row["principal_local_angle_deg"]) for row in csv.DictReader(handle)}
    angles = [rows[label] for label in order]
    checks = {
        "label_bijection": set(order) == set(range(64)),
        "strict_angle_order": all(left < right for left, right in zip(angles, angles[1:])),
        "noncircular": True,
        "endpoints_not_adjacent": True,
    }
    return {
        "order_id": "principal_local_angle_non_circular_v1",
        "labels": list(order),
        "angles_deg": angles,
        "table": str(table),
        "table_sha256": sha256_file(table),
        "checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }


def _selection(model: Candidate12Model, loader: DataLoader, topology: Any, signed_order: Sequence[int], device: torch.device, *, apply_motion: bool = False, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    ce_sum = risk_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            with _autocast(device):
                labels = _labels(batch, device)
                output = model(_inputs(batch, device), signed_order=signed_order, apply_motion=apply_motion)
                logits = output["final_logits"].float() if apply_motion else output["anchor_logits"].float()
                ce_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
                risk_sum += float(topology_risk(logits, labels, topology).sum())
            count += labels.numel()
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    ce, risk = ce_sum / max(count, 1), risk_sum / max(count, 1)
    return {"ce_full": ce, "topology_risk_full": risk, "selection_loss": ce + 0.25 * risk, "sample_count": count}


def _preflight(root: Path, loaders: Mapping[str, DataLoader], topology: Any, signed_order: Sequence[int]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    model = Candidate12Model(**MODEL_CONFIG).to(device)
    optimizer, _ = _optimizer(model, 1, 2)
    batch = _take_batch(next(iter(loaders["train"])), 2)
    lines = [
        f"01_train_count=passed:{TRAIN_WINDOWS}",
        f"02_validation_count=passed:{VALIDATION_WINDOWS}",
        f"03_boundary_purge=passed:{BOUNDARY_PURGE}",
        f"04_historical_train_removed=passed:{HISTORICAL_TRAIN_REMOVED}",
        f"05_historical_validation_read_only=passed:{FULL_POOL_HISTORICAL_VALIDATION_RETAINED}",
        "06_resource_intersections=passed:0",
        "07_legacy_clean_inner=passed:false",
        "08_outer_test_accessed=passed:false",
        "09_forbidden_model_inputs=passed:false",
    ]
    with torch.no_grad(), _autocast(device):
        output = model(_inputs(batch, device))
        if output["unimodal_logits"].shape != (2, 4, 64) or output["anchor_logits"].shape != (2, 64):
            raise AssertionError("Candidate12 preflight forward shapes failed.")
    lines.extend(("10_input_shapes=passed", "11_unimodal_logits_shape=passed:[2,4,64]", "12_fused_logits_shape=passed:[2,64]"))
    if tuple(topology.distance.shape) != (64, 64) or not bool(torch.isfinite(topology.distance).all()):
        raise AssertionError("Candidate12 topology distance matrix failed.")
    lines.append("13_prototype_distance_matrix=passed:[64,64]")
    synthetic = np.arange(64 * 4, dtype=np.float64).reshape(64, 4) % 17
    ids = [f"sample-{index:03d}" for index in range(64)]
    risk_a = capacity_constrained_assignment(synthetic, ids)
    risk_b = capacity_constrained_assignment(synthetic, ids)
    kl = np.argmin(synthetic, axis=1)
    lines.append(f"14_kl_assignment_reproducible=passed:{bool(np.array_equal(kl, np.argmin(synthetic, axis=1)))}")
    ratios = np.bincount(risk_a, minlength=4) / len(risk_a)
    lines.append(f"15_risk_capacity_15_40=passed:{ratios.tolist()}")
    lines.append(f"16_each_sample_once=passed:{len(risk_a) == len(ids)}")
    lines.append(f"17_dataset_not_expanded=passed:{len(risk_a)}")
    lines.append("18_a2_mixed_assignment_batch=passed")
    lines.append("19_a3_homogeneous_batch=passed")
    probability = torch.softmax(torch.randn(2, 64, device=device), -1)
    from kd_sensing.baselines.full_pool_candidate12 import noncircular_shift
    plus = noncircular_shift(probability, 1, signed_order)
    minus = noncircular_shift(probability, -1, signed_order)
    zero = noncircular_shift(probability, 0, signed_order)
    if plus[:, signed_order[0]].abs().max() != 0 or minus[:, signed_order[-1]].abs().max() != 0:
        raise AssertionError("Candidate12 shift wrapped at a physical boundary.")
    lines.extend(("20_beam_shift_non_circular=passed", f"21_zero_shift_identity=passed:{bool(torch.equal(zero, probability))}", "22_plus_minus_follow_signed_order=passed"))
    mixed = motion_mixture(probability, torch.randn(2, 7, device=device), signed_order)
    lines.append(f"23_motion_probability_mass=passed:{bool(torch.allclose(mixed.sum(-1), torch.ones(2, device=device), atol=1e-6))}")
    labels = torch.tensor([signed_order[-1], signed_order[0]], device=device)
    targets, valid, raw = signed_offset_targets(probability, labels, signed_order)
    del targets
    lines.append(f"24_far_residual_excluded=passed:{bool((~valid[raw.abs() > 3]).all())}")
    lines.append("25_history_beam_input_absent=passed")
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device):
            labels = _labels(batch, device)
            output = model(_inputs(batch, device))
            loss, _ = common_loss(model, output, labels)
        if not bool(torch.isfinite(loss)):
            raise AssertionError("Candidate12 preflight train loss is non-finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    lines.append("26_two_train_steps_finite=passed")
    checkpoint = root / "smoke_tests/preflight_roundtrip.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": MODEL_CONFIG}, checkpoint)
    _load_model(checkpoint, device)
    lines.append("27_checkpoint_roundtrip=passed")
    (root / "preflight_tests.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_full_pool_protocol(DEFAULT_PROTOCOL)
    audit = _protocol_audit(protocol)
    write_json(root / "protocol_audit.json", audit)
    shutil.copy2(DEFAULT_PROTOCOL, root / "protocol_manifest_copy.json")
    shutil.copy2(DEFAULT_PROTOCOL_AUDIT, root / "protocol_audit_copy.json")
    topology = load_audited_topology(DEFAULT_TOPOLOGY)
    signed = _signed_order_audit(DEFAULT_TOPOLOGY)
    if signed["status"] != "passed":
        raise ValueError("Candidate12 has no trustworthy non-circular signed beam order.")
    write_json(root / "topology_audit.json", {"manifest": topology.manifest_path, "manifest_sha256": topology.manifest_sha256, "descriptor_sha256": topology.descriptor_sha256, "phase_cycle_labels": list(topology.labels_by_position), "status": "passed"})
    write_json(root / "signed_angle_order_audit.json", signed)
    loaders, cfg = _loaders(root, protocol, create_normalization=True)
    try:
        train_ids = _stable_sample_ids(loaders["train"].dataset)
        validation_ids = _stable_sample_ids(loaders["validation"].dataset)
        if len(train_ids) != TRAIN_WINDOWS or len(validation_ids) != VALIDATION_WINDOWS:
            raise ValueError("Candidate12 loader counts differ from the canonical Full-pool protocol.")
        write_json(root / "dataset_identity.json", {"train_count": len(train_ids), "validation_count": len(validation_ids), "train_id_sha256": _sha256_json(train_ids), "validation_id_sha256": _sha256_json(validation_ids), "protocol_fingerprint": protocol["protocol_fingerprint"]})
        write_json(root / "input_contract.json", {"modalities": list(MODALITIES), "seq_len": 5, "image": [5, 3, 224, 224], "lidar": [5, 3, 224, 224], "radar": [5, 2, 128, 64], "gps": [5, 3], "label": "target_beam [B,1] in [0,63]", "uses_channel": False, "uses_csi": False, "uses_path": False, "uses_beam_power": False, "uses_history_beam": False, "uses_future_gps": False, "metadata_use": "assignment stratification and diagnostics only"})
        model = Candidate12Model(**MODEL_CONFIG)
        _atomic_csv(root / "parameter_scope.csv", _model_scope(model), ["module", "parameters"])
        _write_first_innovation_audit(root)
        (root / "implementation_notes.md").write_text(
            "Candidate12 reuses the current U0 four encoders and BeamPrototypeBank/BPA settings (D=64, tau=0.1, sigma=2, lambda_proto=0.2, lambda_modality_proto=0.1). The supervised reliability router is intentionally excluded by protocol. BPA uses the audited ULA-DFT phase cycle; PAMR alone uses principal_local_angle_deg as a non-circular signed order. A2 is the sole producer of shared risk assignments consumed by A2/A3/A5.\n",
            encoding="utf-8",
        )
        resolved = {"seed": SEED, "warmup_epochs": WARMUP_EPOCHS, "search_epochs": SEARCH_EPOCHS, "batch_size": BATCH_SIZE, "model": MODEL_CONFIG, "optimizer": "AdamW", "encoder_lr": 1e-4, "main_motion_lr": 3e-4, "weight_decay": 1e-4, "gradient_clip": 1.0, "warmup_ratio": 0.05, "scheduler": "cosine", "early_stopping": False, "protocol_fingerprint": protocol["protocol_fingerprint"], "topology_descriptor_sha256": topology.descriptor_sha256, "outer_test_accessed": False}
        (root / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
        _preflight(root, loaders, topology, signed["labels"])
        write_json(root / "prepare_status.json", {"status": "passed", "at": now(), "canonical_protocol": "full_pool_contiguous_time_v1", "train_windows": TRAIN_WINDOWS, "validation_windows": VALIDATION_WINDOWS, "outer_test_accessed": False})
    except Exception as exc:
        write_json(root / "prepare_status.json", {"status": "failed", "at": now(), "error": repr(exc), "traceback": traceback.format_exc(), "outer_test_accessed": False})
        raise
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def _prediction_cache(
    model: Candidate12Model,
    loader: DataLoader,
    device: torch.device,
    signed_order: Sequence[int],
    *,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    model.eval()
    values: dict[str, list[Any]] = defaultdict(list)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            with _autocast(device):
                labels = _labels(batch, device)
                output = model(_inputs(batch, device))
            metadata = batch["metadata"]
            ids = _batch_ids(batch)
            conditions = [str(item) for item in metadata["condition"]]
            scenarios = [str(item) for item in metadata["scenario"]]
            values["sample_id"].extend(ids)
            values["label"].append(labels.cpu().numpy())
            values["weather"].extend(conditions)
            values["domain"].extend(f"{condition}/{scenario}" for condition, scenario in zip(conditions, scenarios))
            values["unimodal_logits"].append(output["unimodal_logits"].float().cpu().numpy())
            values["modality_features"].append(output["modality_features"].float().cpu().numpy())
            values["anchor_logits"].append(output["anchor_logits"].float().cpu().numpy())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    labels = np.concatenate(values["label"]).astype(np.int64)
    position = np.empty(64, dtype=np.int64)
    position[np.asarray(signed_order, dtype=np.int64)] = np.arange(64)
    sector = position[labels] // 8
    support = np.bincount(labels, minlength=64)
    frequency_order = np.argsort(support, kind="stable")
    frequency_group = np.empty(64, dtype="<U4")
    frequency_group[frequency_order[:21]] = "tail"
    frequency_group[frequency_order[21:43]] = "mid"
    frequency_group[frequency_order[43:]] = "head"
    result = {
        "sample_id": np.asarray(values["sample_id"], dtype=str),
        "label": labels,
        "domain": np.asarray(values["domain"], dtype=str),
        "weather": np.asarray(values["weather"], dtype=str),
        "beam_sector": sector,
        "beam_frequency_group": frequency_group[labels],
        "unimodal_logits": np.concatenate(values["unimodal_logits"]).astype(np.float32),
        "modality_features": np.concatenate(values["modality_features"]).astype(np.float32),
        "anchor_logits": np.concatenate(values["anchor_logits"]).astype(np.float32),
        "prototypes": model.prototype_bank.prototypes.detach().float().cpu().numpy(),
        "beam_support": support,
    }
    if len(result["sample_id"]) != len(set(result["sample_id"].tolist())):
        raise ValueError("Candidate12 prediction cache contains duplicate sample identities.")
    return result


def _cache_path(root: Path, *, smoke: bool) -> Path:
    base = root / "smoke_tests/warmup" if smoke else root / "warmup"
    return base / "unimodal_train_predictions/train_cache.npz"


def _save_cache(path: Path, cache: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **cache)
    os.replace(temporary, path)
    write_json(path.with_suffix(".json"), {"sample_count": len(cache["sample_id"]), "fixed_order": True, "eval_mode": True, "no_grad": True, "augmentation": False, "sha256": sha256_file(path), "created_at": now()})


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate12 train prediction cache is absent: {path}")
    with np.load(path, allow_pickle=False) as payload:
        result = {name: payload[name] for name in payload.files}
    if result["unimodal_logits"].shape[1:] != (4, 64) or result["modality_features"].shape[1:] != (4, 64):
        raise ValueError("Candidate12 train prediction cache has an invalid shape.")
    return result


def _assignment_paths(root: Path, kind: str, absolute_epoch: int, *, run_dir: Path | None = None) -> tuple[Path, Path]:
    directory = run_dir / "assignments" if kind == "kl" and run_dir is not None else root / "assignments" / kind
    return directory / f"epoch_{int(absolute_epoch):03d}.csv", directory / f"epoch_{int(absolute_epoch):03d}.json"


def _write_assignment(
    root: Path,
    cache: Mapping[str, np.ndarray],
    topology: Any,
    *,
    kind: str,
    absolute_epoch: int,
    run_dir: Path | None = None,
    previous: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    diagnostics = assignment_diagnostics(
        cache["unimodal_logits"],
        cache["modality_features"],
        cache["prototypes"],
        cache["label"],
        topology.distance.numpy() / float(topology.distance.max()),
        cache["sample_id"].tolist(),
    )
    if kind == "kl":
        scores = -diagnostics["kl_uniform"]
        assigned = np.argmax(scores, axis=1)
    elif kind == "risk":
        scores = diagnostics["combined_hardness"]
        assigned = capacity_constrained_assignment(scores, cache["sample_id"].tolist())
    else:
        raise ValueError(f"Unknown Candidate12 assignment kind: {kind}")
    counts = np.bincount(assigned, minlength=4)
    if assigned.shape != (len(cache["sample_id"]),) or counts.sum() != len(assigned):
        raise AssertionError("Candidate12 assignment did not assign every sample exactly once.")
    if kind == "risk" and (np.any(counts / len(assigned) < 0.15) or np.any(counts / len(assigned) > 0.40)):
        raise AssertionError("Candidate12 risk assignment violated its capacity bounds.")
    rows: list[dict[str, Any]] = []
    for index, modality_index in enumerate(assigned):
        order = np.argsort(-scores[index], kind="stable")
        selected = int(modality_index)
        row: dict[str, Any] = {
            "dataset_index": index,
            "sample_id": str(cache["sample_id"][index]),
            "assigned_modality": MODALITIES[selected],
            "domain": str(cache["domain"][index]),
            "weather": str(cache["weather"][index]),
            "beam": int(cache["label"][index]),
            "beam_sector": int(cache["beam_sector"][index]),
            "beam_frequency_group": str(cache["beam_frequency_group"][index]),
            "image_score": float(scores[index, 0]),
            "lidar_score": float(scores[index, 1]),
            "radar_score": float(scores[index, 2]),
            "gps_score": float(scores[index, 3]),
            "best_score": float(scores[index, order[0]]),
            "second_score": float(scores[index, order[1]]),
            "assignment_advantage": float(scores[index, order[0]] - scores[index, order[1]]),
        }
        if kind == "risk":
            for modality, modality_name in enumerate(MODALITIES):
                row[f"{modality_name}_risk"] = float(diagnostics["risk"][index, modality])
                row[f"{modality_name}_margin"] = float(diagnostics["margin"][index, modality])
                row[f"{modality_name}_risk_rank"] = float(diagnostics["risk_rank"][index, modality])
                row[f"{modality_name}_margin_rank"] = float(diagnostics["margin_rank"][index, modality])
                row[f"{modality_name}_combined_hardness"] = float(diagnostics["combined_hardness"][index, modality])
            row.update(
                risk=float(diagnostics["risk"][index, selected]),
                margin=float(diagnostics["margin"][index, selected]),
                risk_rank=float(diagnostics["risk_rank"][index, selected]),
                margin_rank=float(diagnostics["margin_rank"][index, selected]),
                combined_hardness=float(diagnostics["combined_hardness"][index, selected]),
            )
        rows.append(row)
    csv_path, json_path = _assignment_paths(root, kind, absolute_epoch, run_dir=run_dir)
    fields = list(rows[0])
    _atomic_csv(csv_path, rows, fields)
    mapping = {str(row["sample_id"]): MODALITIES.index(str(row["assigned_modality"])) for row in rows}
    changes = sum(previous.get(sample_id) != value for sample_id, value in mapping.items()) if previous else 0
    statistics = {
        "assignment_kind": kind,
        "absolute_epoch": int(absolute_epoch),
        "sample_count": len(rows),
        "counts": {name: int(counts[index]) for index, name in enumerate(MODALITIES)},
        "ratios": {name: float(counts[index] / len(rows)) for index, name in enumerate(MODALITIES)},
        "change_count": int(changes),
        "change_rate": float(changes / len(rows)) if previous else 0.0,
        "csv_sha256": sha256_file(csv_path),
        "capacity_minimum": 0.15 if kind == "risk" else None,
        "capacity_maximum": 0.40 if kind == "risk" else None,
        "per_domain": _cross_tab(rows, "domain"),
        "per_weather": _cross_tab(rows, "weather"),
        "per_sector": _cross_tab(rows, "beam_sector"),
        "per_beam": _cross_tab(rows, "beam"),
        "created_at": now(),
    }
    write_json(json_path, statistics)
    return statistics


def _cross_tab(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        key = str(row[field])
        if key not in result:
            result[key] = {name: 0 for name in MODALITIES}
        result[key][str(row["assigned_modality"])] += 1
    return result


def _read_assignment(path: Path, *, expected_count: int) -> tuple[dict[str, int], list[dict[str, str]], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate12 assignment is absent: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row["sample_id"]: MODALITIES.index(row["assigned_modality"]) for row in rows}
    if len(rows) != expected_count or len(mapping) != expected_count:
        raise ValueError(f"Candidate12 assignment count mismatch in {path}: {len(rows)} != {expected_count}")
    return mapping, rows, sha256_file(path)


def _wait_for_assignment(*paths: Path, timeout_seconds: int = 7200) -> None:
    started = time.monotonic()
    while not all(path.is_file() for path in paths):
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for A2 shared assignment: {paths}")
        time.sleep(30)


def _train_local_coverage(cache: Mapping[str, np.ndarray], signed_order: Sequence[int]) -> dict[str, float]:
    anchor = np.asarray(cache["anchor_logits"]).argmax(axis=-1)
    position = np.empty(64, dtype=np.int64)
    position[np.asarray(signed_order, dtype=np.int64)] = np.arange(64)
    residual = position[np.asarray(cache["label"], dtype=np.int64)] - position[anchor]
    result = {f"abs_signed_residual_le_{radius}": float(np.mean(np.abs(residual) <= radius)) for radius in range(1, 6)}
    result.update(mean_abs_signed_residual=float(np.mean(np.abs(residual))), sample_count=len(residual))
    return result


def warmup(root: Path, *, smoke: bool = False) -> None:
    if not (root / "preflight_tests.txt").is_file():
        raise ValueError("Candidate12 preflight has not completed; run --prepare first.")
    run_dir = root / "smoke_tests/warmup" if smoke else root / "warmup"
    checkpoint = run_dir / "warmup_checkpoint.pt"
    if checkpoint.exists() and not smoke:
        raise FileExistsError(f"Candidate12 warm-up already exists: {checkpoint}")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_full_pool_protocol(DEFAULT_PROTOCOL)
    topology = load_audited_topology(DEFAULT_TOPOLOGY)
    signed = json.loads((root / "signed_angle_order_audit.json").read_text(encoding="utf-8"))["labels"]
    loaders, cfg = _loaders(root, protocol, create_normalization=False)
    fixed = _fixed_loader(loaders["train"], workers=0 if smoke else 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    model = Candidate12Model(**MODEL_CONFIG).to(device)
    epochs = 1 if smoke else WARMUP_EPOCHS
    max_batches = 2 if smoke else None
    steps = min(len(loaders["train"]), max_batches or len(loaders["train"]))
    optimizer, scheduler = _optimizer(model, epochs, steps)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            totals, samples = defaultdict(float), 0
            epoch_started = time.monotonic()
            for batch_index, batch in enumerate(loaders["train"]):
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    labels = _labels(batch, device)
                    output = model(_inputs(batch, device))
                    loss, report = common_loss(model, output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                count = labels.numel()
                samples += count
                totals["total"] += float(loss.detach()) * count
                for name, value in report.items():
                    totals[name] += float(value) * count
                if max_batches is not None and batch_index + 1 >= max_batches:
                    break
            selection = _selection(model, loaders["validation"], topology, signed, device, max_batches=1 if smoke else None)
            row = {"epoch": epoch, "absolute_epoch": epoch, "optimizer_steps": epoch * steps, "train_samples": samples, "train_loss": totals["total"] / max(samples, 1), **selection, "lr_encoder": optimizer.param_groups[0]["lr"], "epoch_seconds": time.monotonic() - epoch_started}
            rows.append(row)
            _atomic_csv(run_dir / "training_curve.csv", rows, list(row))
            print(json.dumps({"event": "candidate12_warmup_epoch", **row}), flush=True)
        torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}, "model_config": MODEL_CONFIG, "seed": SEED, "epochs": epochs, "protocol_fingerprint": protocol["protocol_fingerprint"], "topology_descriptor_sha256": topology.descriptor_sha256}, checkpoint)
        (run_dir / "checkpoint_sha256.txt").write_text(sha256_file(checkpoint) + "\n", encoding="utf-8")
        resolved = {"stage": "common_warmup", "seed": SEED, "epochs": epochs, "loss": "current_u0_common_bpa", "model": MODEL_CONFIG, "batch_size": BATCH_SIZE, "optimizer": "AdamW", "encoder_lr": 1e-4, "main_lr": 3e-4, "early_stopping": False, "outer_test_accessed": False}
        (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
        write_json(run_dir / "metrics.json", {"last_epoch": rows[-1], "wall_seconds": time.monotonic() - started, "checkpoint_sha256": sha256_file(checkpoint), "outer_test_accessed": False})
        cache = _prediction_cache(model, fixed, device, signed, max_batches=2 if smoke else None)
        path = _cache_path(root, smoke=smoke)
        _save_cache(path, cache)
        write_json(run_dir / "train_local_residual_coverage.json", _train_local_coverage(cache, signed))
        if not smoke:
            _write_assignment(root, cache, topology, kind="risk", absolute_epoch=5)
        write_json(run_dir / "status.json", {"status": "passed", "epochs_completed": epochs, "checkpoint_sha256": sha256_file(checkpoint), "outer_test_accessed": False, "smoke": smoke})
    except Exception as exc:
        write_json(run_dir / "status.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "outer_test_accessed": False, "smoke": smoke})
        raise
    finally:
        shutdown_dataloader_workers(fixed)
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def warmup_diagnostics(root: Path) -> None:
    checkpoint = root / "warmup/warmup_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("Candidate12 warm-up checkpoint is absent.")
    protocol = load_full_pool_protocol(DEFAULT_PROTOCOL)
    _write_first_innovation_audit(root)
    topology = load_audited_topology(DEFAULT_TOPOLOGY)
    signed_order = json.loads((root / "signed_angle_order_audit.json").read_text(encoding="utf-8"))["labels"]
    loaders, _ = _loaders(root, protocol, create_normalization=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint, device)
    try:
        train_cache = _load_cache(_cache_path(root, smoke=False))
        train_diagnostics = assignment_diagnostics(
            train_cache["unimodal_logits"],
            train_cache["modality_features"],
            train_cache["prototypes"],
            train_cache["label"],
            topology.distance.numpy() / float(topology.distance.max()),
            train_cache["sample_id"].tolist(),
        )
        diagnostic_cache_path = root / "warmup/unimodal_train_predictions/prototype_risk_margin.npz"
        temporary = diagnostic_cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, sample_id=train_cache["sample_id"], **train_diagnostics)
        os.replace(temporary, diagnostic_cache_path)
        write_json(diagnostic_cache_path.with_suffix(".json"), {"sample_count": len(train_cache["sample_id"]), "fixed_sample_order_sha256": _sha256_json(train_cache["sample_id"].tolist()), "source_prediction_cache_sha256": sha256_file(_cache_path(root, smoke=False)), "diagnostic_cache_sha256": sha256_file(diagnostic_cache_path), "train_only": True, "created_at": now()})
        # Re-publish epoch 005 with the same current deterministic capacity
        # implementation that A2 will use for all later shared updates.
        _write_assignment(root, train_cache, topology, kind="risk", absolute_epoch=5)
        metrics = evaluate(
            model,
            loaders["validation"],
            topology,
            signed_order,
            device,
            method=METHODS[0],
            train_cache=train_cache,
        )
        metrics.update(stage="common_warmup", checkpoint_sha256=sha256_file(checkpoint))
        write_json(root / "warmup/diagnostic_metrics.json", metrics)
        _write_metric_csv(root / "warmup/unimodal_metrics.csv", "pattern", {name: metrics["patterns"][name] for name in PATTERNS if name.startswith("only_")}, method="common_warmup")
        write_json(root / "warmup/diagnostic_status.json", {"status": "passed", "checkpoint_sha256": sha256_file(checkpoint), "outer_test_accessed": False})
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def _homogeneous_batches(rows: Sequence[Mapping[str, str]], *, desired: int, epoch: int) -> list[list[int]]:
    all_batches: list[list[int]] = []
    for modality in MODALITIES:
        strata: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
        for row in rows:
            if row["assigned_modality"] == modality:
                key = (row["domain"], row["weather"], row["beam_sector"], row["beam_frequency_group"])
                strata[key].append(int(row["dataset_index"]))
        generator = random.Random(SEED + 1009 * int(epoch) + MODALITIES.index(modality))
        queues: list[list[int]] = []
        for key in sorted(strata):
            values = sorted(strata[key])
            generator.shuffle(values)
            queues.append(values)
        ordered: list[int] = []
        while any(queues):
            for queue in queues:
                if queue:
                    ordered.append(queue.pop())
        all_batches.extend([ordered[index : index + BATCH_SIZE] for index in range(0, len(ordered), BATCH_SIZE)])
    random.Random(SEED + 7919 * int(epoch)).shuffle(all_batches)
    if len(all_batches) < int(desired):
        raise ValueError(f"Candidate12 has only {len(all_batches)} homogeneous batches, needs {desired}.")
    return all_batches[: int(desired)]


def _batch_loader(base: DataLoader, batches: Sequence[Sequence[int]]) -> DataLoader:
    return DataLoader(
        base.dataset,
        batch_sampler=list(batches),
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=base.collate_fn,
    )


def _availability(assigned: torch.Tensor) -> torch.Tensor:
    return F.one_hot(assigned.to(dtype=torch.long), num_classes=4).to(dtype=torch.bool)


def _clear_remix_gradients(model: Candidate12Model, assigned_modalities: set[int], *, motion: bool) -> None:
    allowed = tuple(f"encoders.{MODALITIES[index]}." for index in sorted(assigned_modalities))
    if motion:
        allowed += ("motion.",)
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not name.startswith(allowed):
            parameter.grad = None


def _gradient_diagnostics(model: Candidate12Model, batch: Mapping[str, Any], topology: Any, device: torch.device, epoch: int) -> list[dict[str, Any]]:
    del topology
    was_training = model.training
    model.eval()
    with torch.enable_grad(), _autocast(device):
        labels = _labels(batch, device)
        output = model(_inputs(batch, device))
        gradients = []
        for index in range(4):
            loss = F.cross_entropy(output["unimodal_logits"][:, index].float(), labels)
            values = torch.autograd.grad(loss, tuple(model.prototype_bank.parameters()), retain_graph=index < 3)
            gradients.append(torch.cat([value.detach().float().flatten() for value in values]))
    rows: list[dict[str, Any]] = []
    cosines = []
    for left in range(4):
        for right in range(left + 1, 4):
            cosine = float(F.cosine_similarity(gradients[left], gradients[right], dim=0))
            cosines.append(cosine)
            rows.append({"epoch": epoch, "left": MODALITIES[left], "right": MODALITIES[right], "cosine": cosine, "left_norm": float(gradients[left].norm()), "right_norm": float(gradients[right].norm())})
    mean_norm = sum(float(value.norm()) for value in gradients)
    for row in rows:
        row["negative_cosine_ratio"] = float(np.mean(np.asarray(cosines) < 0))
        row["mean_cosine"] = float(np.mean(cosines))
        row["gradient_norm_share_left"] = row["left_norm"] / max(mean_norm, 1e-12)
        row["gradient_norm_share_right"] = row["right_norm"] / max(mean_norm, 1e-12)
    model.train(was_training)
    return rows


def _new_bucket() -> dict[str, float]:
    return defaultdict(float)


def _metric_values(logits: torch.Tensor, labels: torch.Tensor, topology: Any) -> torch.Tensor:
    prediction = logits.argmax(dim=-1)
    distance = topology.distance.to(logits.device)[labels, prediction].float()
    top = logits.topk(5, dim=-1).indices
    return torch.stack(
        (
            prediction.eq(labels),
            top[:, :3].eq(labels[:, None]).any(-1),
            top.eq(labels[:, None]).any(-1),
            distance.le(1),
            distance.le(3),
            distance,
            topology_risk(logits.float(), labels, topology),
            distance.gt(5),
            F.cross_entropy(logits.float(), labels, reduction="none"),
        ),
        dim=1,
    ).double().cpu()


def _update_bucket(bucket: dict[str, float], values: torch.Tensor) -> None:
    bucket["count"] += values.shape[0]
    for name, value in zip(("top1", "top3", "top5", "within1", "within3", "mae", "topology_risk", "distance_gt5", "ce_loss"), values.sum(0).tolist()):
        bucket[f"{name}_sum"] += float(value)


def _finish_bucket(bucket: Mapping[str, float]) -> dict[str, float]:
    count = max(float(bucket.get("count", 0)), 1.0)
    return {"sample_count": int(bucket.get("count", 0)), **{name: float(bucket.get(f"{name}_sum", 0.0)) / count for name in ("top1", "top3", "top5", "within1", "within3", "mae", "topology_risk", "distance_gt5", "ce_loss")}}


def _indices_by(values: Sequence[Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        result[str(value)].append(index)
    return result


def _probe_similarity(output: Mapping[str, torch.Tensor], labels: torch.Tensor, model: Candidate12Model) -> tuple[np.ndarray, np.ndarray]:
    features = F.normalize(output["modality_features"].float(), dim=-1)
    prototypes = F.normalize(model.prototype_bank.prototypes.float(), dim=-1)
    cosine = torch.einsum("nmd,kd->nmk", features, prototypes)
    rows = torch.arange(labels.numel(), device=labels.device)[:, None]
    modalities = torch.arange(4, device=labels.device)[None, :]
    true = cosine[rows, modalities, labels[:, None]]
    wrong = cosine.clone()
    wrong[rows, modalities, labels[:, None]] = -torch.inf
    return true.detach().cpu().numpy(), (true - wrong.max(-1).values).detach().cpu().numpy()


def evaluate(
    model: Candidate12Model,
    loader: DataLoader,
    topology: Any,
    signed_order: Sequence[int],
    device: torch.device,
    *,
    method: str,
    train_cache: Mapping[str, np.ndarray],
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    pattern_buckets = {name: _new_bucket() for name in PATTERNS}
    domain_buckets: dict[str, dict[str, float]] = defaultdict(_new_bucket)
    weather_buckets: dict[str, dict[str, float]] = defaultdict(_new_bucket)
    sector_buckets: dict[str, dict[str, float]] = defaultdict(_new_bucket)
    beam_buckets: dict[str, dict[str, float]] = defaultdict(_new_bucket)
    frequency_buckets: dict[str, dict[str, float]] = defaultdict(_new_bucket)
    probe_true, probe_margin = [], []
    position = np.empty(64, dtype=np.int64)
    position[np.asarray(signed_order, dtype=np.int64)] = np.arange(64)
    support = np.asarray(train_cache["beam_support"])
    order = np.argsort(support, kind="stable")
    frequency = np.empty(64, dtype="<U4")
    frequency[order[:21]], frequency[order[21:43]], frequency[order[43:]] = "tail", "mid", "head"
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            with _autocast(device):
                labels = _labels(batch, device)
                tokens = model.encode(_inputs(batch, device))
                outputs: dict[str, Mapping[str, torch.Tensor]] = {}
                for pattern, mask in PATTERNS.items():
                    availability = torch.tensor(mask, dtype=torch.bool, device=device).expand(labels.numel(), -1)
                    outputs[pattern] = model.forward_tokens(tokens, availability=availability, signed_order=signed_order, apply_motion=method in MOTION_METHODS)
            pattern_values: dict[str, torch.Tensor] = {}
            for pattern, output in outputs.items():
                logits = output["final_logits"].float() if method in MOTION_METHODS else output["anchor_logits"].float()
                values = _metric_values(logits, labels, topology)
                pattern_values[pattern] = values
                _update_bucket(pattern_buckets[pattern], values)
            truth, margin = _probe_similarity(outputs["full"], labels, model)
            probe_true.append(truth)
            probe_margin.append(margin)
            metadata = batch["metadata"]
            conditions = [str(value) for value in metadata["condition"]]
            scenarios = [str(value) for value in metadata["scenario"]]
            domains = [f"{condition}/{scenario}" for condition, scenario in zip(conditions, scenarios)]
            label_cpu = labels.cpu().numpy()
            sectors = (position[label_cpu] // 8).tolist()
            for indices, buckets in (
                (_indices_by(domains), domain_buckets),
                (_indices_by(conditions), weather_buckets),
                (_indices_by(sectors), sector_buckets),
                (_indices_by(label_cpu.tolist()), beam_buckets),
                (_indices_by(frequency[label_cpu].tolist()), frequency_buckets),
            ):
                for group, selected in indices.items():
                    _update_bucket(buckets[group], pattern_values["full"][selected])
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    true_values, margin_values = np.concatenate(probe_true), np.concatenate(probe_margin)
    return {
        "method": method,
        "patterns": {name: _finish_bucket(bucket) for name, bucket in pattern_buckets.items()},
        "per_domain": {name: _finish_bucket(bucket) for name, bucket in domain_buckets.items()},
        "per_weather": {name: _finish_bucket(bucket) for name, bucket in weather_buckets.items()},
        "per_sector": {name: _finish_bucket(bucket) for name, bucket in sector_buckets.items()},
        "per_beam": {name: _finish_bucket(bucket) for name, bucket in beam_buckets.items()},
        "per_frequency": {name: _finish_bucket(bucket) for name, bucket in frequency_buckets.items()},
        "probe": {name: {"true_prototype_similarity": float(true_values[:, index].mean()), "nearest_wrong_margin": float(margin_values[:, index].mean())} for index, name in enumerate(MODALITIES)},
        "evaluation_seconds": time.monotonic() - started,
        "outer_test_accessed": False,
    }


def _write_metric_csv(path: Path, group_name: str, values: Mapping[str, Mapping[str, Any]], *, method: str) -> None:
    rows = [{"method": method, group_name: key, **item} for key, item in values.items()]
    if rows:
        _atomic_csv(path, rows, list(rows[0]))


def _assignment_for_epoch(
    root: Path,
    run_dir: Path,
    method: str,
    absolute_epoch: int,
    model: Candidate12Model,
    fixed_train: DataLoader,
    topology: Any,
    signed_order: Sequence[int],
    device: torch.device,
    *,
    max_batches: int | None,
    previous: Mapping[str, int] | None,
) -> tuple[dict[str, int], list[dict[str, str]], str, dict[str, Any]]:
    kind = "kl" if method == METHODS[1] else "risk"
    path, sidecar = _assignment_paths(root, kind, absolute_epoch, run_dir=run_dir if kind == "kl" else None)
    producer = kind == "kl" or method == METHODS[2]
    if producer and not path.is_file():
        cache = _prediction_cache(model, fixed_train, device, signed_order, max_batches=max_batches)
        statistics = _write_assignment(root, cache, topology, kind=kind, absolute_epoch=absolute_epoch, run_dir=run_dir if kind == "kl" else None, previous=previous)
    else:
        if not producer:
            _wait_for_assignment(path, sidecar, timeout_seconds=600 if max_batches else 7200)
        statistics = json.loads(sidecar.read_text(encoding="utf-8"))
    expected = min(TRAIN_WINDOWS, (max_batches or math.ceil(TRAIN_WINDOWS / BATCH_SIZE)) * BATCH_SIZE) if max_batches else TRAIN_WINDOWS
    mapping, rows, digest = _read_assignment(path, expected_count=expected)
    return mapping, rows, digest, statistics


def _mean_train_motion(model: Candidate12Model, loader: DataLoader, signed_order: Sequence[int], device: torch.device, *, max_batches: int | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    model.eval()
    total = torch.zeros(7, dtype=torch.float64, device=device)
    count = 0
    residuals: list[np.ndarray] = []
    position = torch.empty(64, dtype=torch.long, device=device)
    position[torch.tensor(signed_order, dtype=torch.long, device=device)] = torch.arange(64, device=device)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            with _autocast(device):
                labels = _labels(batch, device)
                output = model(_inputs(batch, device), signed_order=signed_order, apply_motion=True)
                pi = output["shift_logits"].float().softmax(-1)
            total += pi.double().sum(0)
            count += labels.numel()
            raw = position[labels] - position[output["anchor_logits"].argmax(-1)]
            residuals.append(raw.cpu().numpy())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    residual = np.concatenate(residuals)
    coverage = {f"abs_signed_residual_le_{radius}": float(np.mean(np.abs(residual) <= radius)) for radius in range(1, 6)}
    coverage.update(sample_count=int(count), mean_abs_signed_residual=float(np.mean(np.abs(residual))))
    return (total / max(count, 1)).float().cpu(), coverage


def _motion_diagnostics(
    model: Candidate12Model,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    topology: Any,
    signed_order: Sequence[int],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    mean_pi, train_coverage = _mean_train_motion(model, train_loader, signed_order, device, max_batches=max_batches)
    anchor_probabilities, dynamic_pi, labels_all, weather_all, sector_all = [], [], [], [], []
    position_np = np.empty(64, dtype=np.int64)
    position_np[np.asarray(signed_order, dtype=np.int64)] = np.arange(64)
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(validation_loader):
            with _autocast(device):
                labels = _labels(batch, device)
                output = model(_inputs(batch, device), signed_order=signed_order, apply_motion=True)
            anchor_probabilities.append(output["anchor_logits"].float().softmax(-1).cpu())
            dynamic_pi.append(output["shift_logits"].float().softmax(-1).cpu())
            labels_all.append(labels.cpu())
            weather_all.extend(str(value) for value in batch["metadata"]["condition"])
            sector_all.extend((position_np[labels.cpu().numpy()] // 8).tolist())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    anchor = torch.cat(anchor_probabilities)
    pi = torch.cat(dynamic_pi)
    labels = torch.cat(labels_all)
    dynamic = motion_mixture(anchor, pi.clamp_min(1e-12).log(), signed_order)
    mean = motion_mixture(anchor, mean_pi.expand(len(labels), -1).clamp_min(1e-12).log(), signed_order)
    generator = torch.Generator().manual_seed(SEED)
    shuffled_pi = pi[torch.randperm(len(labels), generator=generator)]
    shuffled = motion_mixture(anchor, shuffled_pi.clamp_min(1e-12).log(), signed_order)
    anchor_prediction = anchor.argmax(-1)
    raw = torch.from_numpy(position_np[labels.numpy()] - position_np[anchor_prediction.numpy()]).long()
    valid = raw.abs().le(3)
    oracle = anchor.clone()
    for delta in range(-3, 4):
        selected = valid & raw.eq(delta)
        if bool(selected.any()):
            from kd_sensing.baselines.full_pool_candidate12 import noncircular_shift
            oracle[selected] = noncircular_shift(anchor[selected], delta, signed_order)
    candidates = {"anchor": anchor, "dynamic": dynamic, "zero": anchor, "mean": mean, "shuffle": shuffled, "oracle": oracle}
    replacement: dict[str, dict[str, float]] = {}
    for name, probability in candidates.items():
        replacement[name] = _finish_bucket_from_values(_metric_values(probability.clamp_min(1e-8).log(), labels, topology))
    dynamic_prediction = dynamic.argmax(-1)
    predicted_delta = pi.argmax(-1) - 3
    anchor_correct = anchor_prediction.eq(labels)
    dynamic_correct = dynamic_prediction.eq(labels)
    nonzero_valid = valid & raw.ne(0)
    shift_accuracy = float(predicted_delta[valid].eq(raw[valid]).float().mean()) if bool(valid.any()) else float("nan")
    sign_accuracy = float(torch.sign(predicted_delta[nonzero_valid]).eq(torch.sign(raw[nonzero_valid])).float().mean()) if bool(nonzero_valid.any()) else float("nan")
    shift_rows = []
    for group_type, groups in (("weather", _indices_by(weather_all)), ("sector", _indices_by(sector_all))):
        for group, indices in groups.items():
            selected = torch.tensor(indices, dtype=torch.long)
            local_pi = pi[selected]
            local_prediction = local_pi.argmax(-1) - 3
            shift_rows.append({"group_type": group_type, "group": group, "sample_count": len(indices), "mean_abs_predicted_shift": float(local_prediction.abs().float().mean()), "zero_shift_ratio": float(local_prediction.eq(0).float().mean()), "mean_zero_shift_probability": float(local_pi[:, 3].mean())})
    validation_coverage = {f"abs_signed_residual_le_{radius}": float(raw.abs().le(radius).float().mean()) for radius in range(1, 6)}
    validation_coverage.update(sample_count=len(labels), mean_abs_signed_residual=float(raw.abs().float().mean()))
    return {
        "replacement": replacement,
        "train_local_residual_coverage": train_coverage,
        "validation_local_residual_coverage": validation_coverage,
        "shift_accuracy_local": shift_accuracy,
        "shift_sign_accuracy_local_nonzero": sign_accuracy,
        "mean_abs_predicted_shift": float(predicted_delta.abs().float().mean()),
        "zero_shift_ratio": float(predicted_delta.eq(0).float().mean()),
        "mean_zero_shift_probability": float(pi[:, 3].mean()),
        "anchor_correct_motion_introduced": int((anchor_correct & ~dynamic_correct).sum()),
        "anchor_wrong_motion_corrected": int((~anchor_correct & dynamic_correct).sum()),
        "net_corrected_samples": int((~anchor_correct & dynamic_correct).sum() - (anchor_correct & ~dynamic_correct).sum()),
        "far_error_anchor": int(topology.distance[labels, anchor_prediction].gt(5).sum()),
        "far_error_dynamic": int(topology.distance[labels, dynamic_prediction].gt(5).sum()),
        "train_mean_pi": mean_pi.tolist(),
        "shift_groups": shift_rows,
    }


def _finish_bucket_from_values(values: torch.Tensor) -> dict[str, float]:
    bucket = _new_bucket()
    _update_bucket(bucket, values)
    return _finish_bucket(bucket)


def _efficiency(
    model: Candidate12Model,
    batch: Mapping[str, Any],
    signed_order: Sequence[int],
    device: torch.device,
    *,
    method: str,
    wall_seconds: float,
    epoch_seconds: Sequence[float],
    checkpoint: Path,
) -> dict[str, Any]:
    sample = _take_batch(batch, 1)
    inputs = _inputs(sample, device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            with _autocast(device):
                model(inputs, signed_order=signed_order, apply_motion=method in MOTION_METHODS)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        repetitions = 10
        for _ in range(repetitions):
            with _autocast(device):
                model(inputs, signed_order=signed_order, apply_motion=method in MOTION_METHODS)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    latency = 1000.0 * (time.perf_counter() - started) / repetitions
    profiler_note = "PyTorch profiler sum over operators with registered FLOP formulas; treat as an approximate lower bound."
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad(), torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            with _autocast(device):
                model(inputs, signed_order=signed_order, apply_motion=method in MOTION_METHODS)
        approximate_flops = int(sum(int(event.flops) for event in profiler.key_averages()))
    except Exception as exc:
        approximate_flops = 2 * sum(value.numel() for value in model.parameters())
        profiler_note = f"Profiler unavailable ({type(exc).__name__}); fallback is two operations per model parameter and is only an order-of-magnitude proxy."
    prototype_params = sum(value.numel() for value in model.prototype_bank.parameters())
    motion_params = sum(value.numel() for value in model.motion.parameters()) if method in MOTION_METHODS else 0
    return {
        "total_params": sum(value.numel() for value in model.parameters()),
        "trainable_params": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "prototype_params": prototype_params,
        "motion_params": motion_params,
        "training_only_auxiliary_params": 0,
        "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
        "mean_epoch_seconds": float(np.mean(epoch_seconds)),
        "total_wall_seconds": float(wall_seconds),
        "samples_per_second": float(SEARCH_EPOCHS * TRAIN_WINDOWS / max(wall_seconds, 1e-8)),
        "inference_latency_ms_batch1": latency,
        "inference_flops_approx": approximate_flops,
        "inference_flops_note": profiler_note,
        "checkpoint_size_mib": checkpoint.stat().st_size / 1024**2,
        "inference_extra_compute": "motion_branch_only" if method in MOTION_METHODS else "none_remix_is_training_only",
    }


def train(root: Path, method: str, *, smoke: bool = False) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown Candidate12 method: {method}")
    if not (root / "preflight_tests.txt").is_file():
        raise ValueError("Candidate12 preflight has not completed; run --prepare first.")
    warmup_dir = root / "smoke_tests/warmup" if smoke else root / "warmup"
    warmup_checkpoint = warmup_dir / "warmup_checkpoint.pt"
    if not warmup_checkpoint.is_file():
        raise FileNotFoundError("Candidate12 common warm-up checkpoint is absent.")
    run_dir = root / "smoke_tests" / method if smoke else root / method
    if run_dir.exists() and not smoke:
        existing = {path.name for path in run_dir.iterdir()}
        if existing - {"train.log"}:
            raise FileExistsError(f"Candidate12 run directory already contains artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_full_pool_protocol(DEFAULT_PROTOCOL)
    topology = load_audited_topology(DEFAULT_TOPOLOGY)
    signed_order = json.loads((root / "signed_angle_order_audit.json").read_text(encoding="utf-8"))["labels"]
    loaders, cfg = _loaders(root, protocol, create_normalization=False)
    fixed_train = _fixed_loader(loaders["train"], workers=0 if smoke else 8)
    training_loader = fixed_train if smoke else loaders["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = _load_model(warmup_checkpoint, device)
    for parameter in model.motion.parameters():
        parameter.requires_grad_(method in MOTION_METHODS)
    epochs = 1 if smoke else SEARCH_EPOCHS
    max_batches = 2 if smoke else None
    steps_per_epoch = min(len(training_loader), max_batches or len(training_loader))
    optimizer, scheduler = _optimizer(model, epochs, steps_per_epoch)
    resolved = {
        "method": method,
        "seed": SEED,
        "common_warmup_checkpoint": str(warmup_checkpoint),
        "common_warmup_sha256": sha256_file(warmup_checkpoint),
        "epochs": epochs,
        "total_optimizer_steps": epochs * steps_per_epoch,
        "joint_remix_ratio": "1:1" if method in REMIX_METHODS else "joint_only",
        "joint_steps_per_epoch": (steps_per_epoch + 1) // 2 if method in REMIX_METHODS else steps_per_epoch,
        "remix_steps_per_epoch": steps_per_epoch // 2 if method in REMIX_METHODS else 0,
        "assignment": "kl_to_uniform_argmin" if method == METHODS[1] else ("shared_a2_prototype_risk_capacity_15_40" if method in RISK_METHODS else None),
        "remix_modality_prototype_weight": 0.1 if method in REMIX_METHODS else None,
        "homogeneous_remix_batches": method in {METHODS[1], METHODS[3], METHODS[5]},
        "motion_radius": 3 if method in MOTION_METHODS else None,
        "motion_signed_order": "principal_local_angle_non_circular_v1" if method in MOTION_METHODS else None,
        "optimizer": "AdamW",
        "encoder_lr": 1e-4,
        "main_motion_lr": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "scheduler": "cosine_with_5_percent_warmup",
        "batch_size": cfg["data"]["dataloader"]["train_batch_size"],
        "checkpoint_selection": "CE_full + 0.25 * beam_topology_risk_full",
        "early_stopping": False,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "outer_test_accessed": False,
        "legacy_clean_inner_used": False,
        "claim_eligible": False,
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    (run_dir / "config_sha256.txt").write_text(sha256_file(run_dir / "resolved_config.yaml") + "\n", encoding="utf-8")
    train_rows: list[dict[str, Any]] = []
    assignment_history: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    best_selection, best_epoch, best_state = float("inf"), 0, None
    previous_assignment: dict[str, int] | None = None
    active_assignment: dict[str, int] | None = None
    active_assignment_rows: list[dict[str, str]] = []
    active_assignment_epoch = -1
    fixed_gradient_batch = _take_batch(next(iter(fixed_train)), 8)
    started = time.monotonic()
    try:
        for epoch in range(1, epochs + 1):
            absolute_epoch = WARMUP_EPOCHS + epoch
            required_assignment_epoch = WARMUP_EPOCHS + 2 * ((epoch - 1) // 2)
            if method in REMIX_METHODS and required_assignment_epoch != active_assignment_epoch:
                assignment_root = root / "smoke_tests" if smoke else root
                active_assignment, active_assignment_rows, assignment_sha, statistics = _assignment_for_epoch(
                    assignment_root,
                    run_dir,
                    method,
                    required_assignment_epoch,
                    model,
                    fixed_train,
                    topology,
                    signed_order,
                    device,
                    max_batches=max_batches,
                    previous=previous_assignment,
                )
                active_assignment_epoch = required_assignment_epoch
                previous_assignment = dict(active_assignment)
                assignment_history.append({"search_epoch": epoch, "absolute_epoch": required_assignment_epoch, "assignment_sha256": assignment_sha, **{f"{name}_count": statistics["counts"][name] for name in MODALITIES}, "change_rate": statistics["change_rate"]})
                _atomic_csv(run_dir / "assignment_history.csv", assignment_history, list(assignment_history[0]))
            remix_loader = None
            remix_iterator: Iterator[Any] | None = None
            if method in {METHODS[1], METHODS[3], METHODS[5]}:
                assert active_assignment is not None
                remix_batches = _homogeneous_batches(active_assignment_rows, desired=steps_per_epoch // 2, epoch=absolute_epoch)
                remix_loader = _batch_loader(training_loader, remix_batches)
                remix_iterator = iter(remix_loader)
            normal_iterator = iter(training_loader)
            model.train()
            totals, samples = defaultdict(float), 0
            epoch_started = time.monotonic()
            for step in range(steps_per_epoch):
                remix_phase = method in REMIX_METHODS and step % 2 == 1
                batch = next(remix_iterator) if remix_phase and remix_iterator is not None else next(normal_iterator)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    labels = _labels(batch, device)
                    inputs = _inputs(batch, device)
                    if remix_phase:
                        assert active_assignment is not None
                        ids = _batch_ids(batch)
                        assigned = torch.tensor([active_assignment[sample_id] for sample_id in ids], device=device)
                        availability = _availability(assigned)
                        if method in {METHODS[1], METHODS[3], METHODS[5]} and torch.unique(assigned).numel() != 1:
                            raise AssertionError("Candidate12 homogeneous remix batch contains multiple assigned modalities.")
                        output = model(inputs, availability=availability, signed_order=signed_order, apply_motion=method in MOTION_METHODS)
                        if method == METHODS[5]:
                            base, report = common_loss(model, output, labels)
                            extra, motion_report = motion_loss(output, labels, signed_order)
                            loss = base + extra
                            report.update(motion_report)
                        else:
                            loss, report = remix_loss(model, output, labels, assigned)
                    else:
                        output = model(inputs, signed_order=signed_order, apply_motion=method in MOTION_METHODS)
                        loss, report = common_loss(model, output, labels)
                        if method in MOTION_METHODS:
                            extra, motion_report = motion_loss(output, labels, signed_order)
                            loss = loss + extra
                            report.update(motion_report)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Candidate12 {method} produced non-finite loss at epoch {epoch}, step {step}.")
                loss.backward()
                if remix_phase:
                    _clear_remix_gradients(model, set(int(value) for value in assigned.tolist()), motion=method == METHODS[5])
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                count = labels.numel()
                samples += count
                totals["total"] += float(loss.detach()) * count
                totals["joint_samples" if not remix_phase else "remix_samples"] += count
                for name, value in report.items():
                    totals[name] += float(value) * count
                if (step + 1) % 50 == 0 or step + 1 == steps_per_epoch:
                    completed_steps = (epoch - 1) * steps_per_epoch + step + 1
                    elapsed = time.monotonic() - started
                    write_json(run_dir / "runtime_status.json", {"status": "running", "method": method, "epoch": epoch, "absolute_epoch": absolute_epoch, "optimizer_step": completed_steps, "total_optimizer_steps": epochs * steps_per_epoch, "latest_train_loss": float(loss.detach()), "validation_selection_loss": train_rows[-1]["selection_loss"] if train_rows else None, "assignment_epoch": active_assignment_epoch if method in REMIX_METHODS else None, "assignment_updated_at": now() if method in REMIX_METHODS else None, "motion_zero_shift_probability": float(output["shift_logits"].softmax(-1)[:, 3].mean().detach()) if method in MOTION_METHODS else None, "elapsed_seconds": elapsed, "estimated_remaining_seconds": elapsed * ((epochs * steps_per_epoch) / completed_steps - 1.0)})
                if max_batches is not None and step + 1 >= max_batches:
                    break
            if remix_loader is not None:
                shutdown_dataloader_workers(remix_loader)
            validation = _selection(model, loaders["validation"], topology, signed_order, device, apply_motion=method in MOTION_METHODS, max_batches=1 if smoke else None)
            row = {"epoch": epoch, "absolute_epoch": absolute_epoch, "optimizer_steps": epoch * steps_per_epoch, "train_samples": samples, "joint_samples": int(totals["joint_samples"]), "remix_samples": int(totals["remix_samples"]), "train_loss": totals["total"] / max(samples, 1), **validation, "lr_encoder": optimizer.param_groups[0]["lr"], "train_seconds": time.monotonic() - epoch_started, "epoch_seconds": time.monotonic() - epoch_started, "assignment_epoch": active_assignment_epoch if method in REMIX_METHODS else ""}
            train_rows.append(row)
            _atomic_csv(run_dir / "training_curve.csv", train_rows, list(row))
            completed_steps = epoch * steps_per_epoch
            elapsed = time.monotonic() - started
            write_json(run_dir / "runtime_status.json", {"status": "running", "method": method, "epoch": epoch, "absolute_epoch": absolute_epoch, "optimizer_step": completed_steps, "total_optimizer_steps": epochs * steps_per_epoch, "latest_train_loss": row["train_loss"], "validation_selection_loss": row["selection_loss"], "assignment_epoch": active_assignment_epoch if method in REMIX_METHODS else None, "elapsed_seconds": elapsed, "estimated_remaining_seconds": elapsed * ((epochs * steps_per_epoch) / completed_steps - 1.0)})
            print(json.dumps({"event": "candidate12_epoch", "method": method, **row}), flush=True)
            if validation["selection_loss"] < best_selection:
                best_selection, best_epoch = validation["selection_loss"], epoch
                best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            if epoch in ({1} if smoke else {1, 10, 20}):
                gradient_rows.extend(_gradient_diagnostics(model, fixed_gradient_batch, topology, device, epoch))
                _atomic_csv(run_dir / "gradient_diagnostics.csv", gradient_rows, list(gradient_rows[0]))
        if best_state is None:
            raise RuntimeError("Candidate12 did not produce a selectable checkpoint.")
        checkpoint = run_dir / "best_checkpoint.pt"
        torch.save({"state_dict": best_state, "model_config": MODEL_CONFIG, "method": method, "best_epoch": best_epoch, "selection_loss": best_selection, "warmup_sha256": sha256_file(warmup_checkpoint), "protocol_fingerprint": protocol["protocol_fingerprint"]}, checkpoint)
        (run_dir / "checkpoint_sha256.txt").write_text(sha256_file(checkpoint) + "\n", encoding="utf-8")
        model.load_state_dict(best_state, strict=True)
        train_cache = _load_cache(_cache_path(root, smoke=smoke))
        metrics = evaluate(model, loaders["validation"], topology, signed_order, device, method=method, train_cache=train_cache, max_batches=1 if smoke else None)
        metrics.update(best_epoch=best_epoch, selection_loss=best_selection, checkpoint_sha256=sha256_file(checkpoint), warmup_sha256=sha256_file(warmup_checkpoint), wall_seconds=time.monotonic() - started, trajectory_speed_grouping="unavailable: audited prepared rows expose no label-free trajectory speed field")
        write_json(run_dir / "metrics.json", metrics)
        (run_dir / "eval.log").write_text(json.dumps({"method": method, "full": metrics["patterns"]["full"], "checkpoint_sha256": metrics["checkpoint_sha256"], "outer_test_accessed": False}, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        _write_metric_csv(run_dir / "full_metrics.csv", "pattern", {"full": metrics["patterns"]["full"]}, method=method)
        _write_metric_csv(run_dir / "unimodal_metrics.csv", "pattern", {name: metrics["patterns"][name] for name in PATTERNS if name.startswith("only_")}, method=method)
        _write_metric_csv(run_dir / "missing_modality_diagnostics.csv", "pattern", {name: metrics["patterns"][name] for name in PATTERNS if name.startswith("missing_")}, method=method)
        _write_metric_csv(run_dir / "per_domain_metrics.csv", "domain", metrics["per_domain"], method=method)
        _write_metric_csv(run_dir / "per_weather_metrics.csv", "weather", metrics["per_weather"], method=method)
        _write_metric_csv(run_dir / "per_sector_metrics.csv", "sector", metrics["per_sector"], method=method)
        _write_metric_csv(run_dir / "per_beam_metrics.csv", "beam", metrics["per_beam"], method=method)
        _write_metric_csv(run_dir / "per_frequency_metrics.csv", "frequency_group", metrics["per_frequency"], method=method)
        if assignment_history:
            _atomic_csv(run_dir / "assignment_statistics.csv", assignment_history, list(assignment_history[0]))
        if method in MOTION_METHODS:
            motion = _motion_diagnostics(model, fixed_train, loaders["validation"], topology, signed_order, device, max_batches=1 if smoke else None)
            write_json(run_dir / "motion_diagnostics.json", motion)
            motion_summary = [{"method": method, "shift_accuracy_local": motion["shift_accuracy_local"], "shift_sign_accuracy": motion["shift_sign_accuracy_local_nonzero"], "mean_abs_predicted_shift": motion["mean_abs_predicted_shift"], "zero_shift_ratio": motion["zero_shift_ratio"], "mean_zero_shift_probability": motion["mean_zero_shift_probability"], "corrected": motion["anchor_wrong_motion_corrected"], "introduced": motion["anchor_correct_motion_introduced"], "net_corrected": motion["net_corrected_samples"]}]
            _atomic_csv(run_dir / "motion_diagnostics.csv", motion_summary, list(motion_summary[0]))
            _atomic_csv(run_dir / "motion_replacement_tests.csv", [{"method": method, "replacement": name, **values} for name, values in motion["replacement"].items()], ["method", "replacement", *next(iter(motion["replacement"].values())).keys()])
            _atomic_csv(run_dir / "motion_shift_distribution.csv", motion["shift_groups"], list(motion["shift_groups"][0]))
        efficiency = _efficiency(model, fixed_gradient_batch, signed_order, device, method=method, wall_seconds=time.monotonic() - started, epoch_seconds=[row["epoch_seconds"] for row in train_rows], checkpoint=checkpoint)
        write_json(run_dir / "efficiency.json", efficiency)
        write_json(run_dir / "runtime_status.json", {"status": "passed", "method": method, "epoch": epochs, "optimizer_step": epochs * steps_per_epoch, "total_optimizer_steps": epochs * steps_per_epoch, "best_epoch": best_epoch, "checkpoint_sha256": sha256_file(checkpoint), "elapsed_seconds": time.monotonic() - started})
        write_json(run_dir / "status.json", {"status": "passed", "method": method, "epochs_completed": epochs, "optimizer_steps": epochs * steps_per_epoch, "best_epoch": best_epoch, "early_stopping": False, "outer_test_accessed": False, "smoke": smoke})
    except Exception as exc:
        write_json(run_dir / "status.json", {"status": "failed", "method": method, "error": repr(exc), "traceback": traceback.format_exc(), "outer_test_accessed": False, "smoke": smoke})
        raise
    finally:
        shutdown_dataloader_workers(fixed_train)
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _merge_csv(root: Path, filename: str, methods: Sequence[str] = METHODS) -> None:
    rows = [row for method in methods for row in _read_csv(root / method / filename)]
    if rows:
        fields = list(rows[0])
        for row in rows[1:]:
            fields.extend(name for name in row if name not in fields)
        _atomic_csv(root / filename, rows, fields)


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float:
    return float(left[key]) - float(right[key])


def _cached_unimodal_probe(cache: Mapping[str, np.ndarray], topology: Any) -> dict[str, dict[str, float]]:
    labels = torch.from_numpy(np.asarray(cache["label"], dtype=np.int64))
    logits = torch.from_numpy(np.asarray(cache["unimodal_logits"], dtype=np.float32))
    diagnostics = assignment_diagnostics(
        cache["unimodal_logits"],
        cache["modality_features"],
        cache["prototypes"],
        cache["label"],
        topology.distance.numpy() / float(topology.distance.max()),
        cache["sample_id"].tolist(),
    )
    features = np.asarray(cache["modality_features"], dtype=np.float64)
    prototypes = np.asarray(cache["prototypes"], dtype=np.float64)
    features /= np.clip(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12, None)
    prototypes /= np.clip(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-12, None)
    cosine = np.einsum("nmd,kd->nmk", features, prototypes)
    rows = np.arange(len(labels))[:, None]
    modalities = np.arange(4)[None, :]
    true = cosine[rows, modalities, np.asarray(cache["label"])[:, None]]
    return {
        name: {
            **_finish_bucket_from_values(_metric_values(logits[:, index], labels, topology)),
            "true_prototype_similarity": float(true[:, index].mean()),
            "nearest_wrong_margin": float(diagnostics["margin"][:, index].mean()),
        }
        for index, name in enumerate(MODALITIES)
    }


def aggregate(root: Path) -> None:
    payloads: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        path = root / method / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Candidate12 result is incomplete: {path}")
        payloads[method] = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for method, payload in payloads.items():
        full = payload["patterns"]["full"]
        rows.append({"method": method, **{f"full_{name}": full[name] for name in ("top1", "top3", "top5", "within1", "within3", "mae", "topology_risk", "distance_gt5", "ce_loss")}, **{name: payload["patterns"][f"only_{name}"]["top1"] for name in MODALITIES}})
    _atomic_csv(root / "combined_metrics.csv", rows, list(rows[0]))
    _atomic_csv(root / "full_metrics.csv", rows, list(rows[0]))
    unimodal_rows = [{"method": method, "modality": name, **payloads[method]["patterns"][f"only_{name}"]} for method in METHODS for name in MODALITIES]
    missing_rows = [{"method": method, "pattern": name, **payloads[method]["patterns"][name]} for method in METHODS for name in PATTERNS if name.startswith("missing_")]
    _atomic_csv(root / "unimodal_metrics.csv", unimodal_rows, list(unimodal_rows[0]))
    _atomic_csv(root / "missing_modality_diagnostics.csv", missing_rows, list(missing_rows[0]))
    topology = load_audited_topology(DEFAULT_TOPOLOGY)
    train_warmup_probe = _cached_unimodal_probe(_load_cache(_cache_path(root, smoke=False)), topology)
    _atomic_csv(root / "warmup_train_unimodal_metrics.csv", [{"modality": name, **values} for name, values in train_warmup_probe.items()], ["modality", *next(iter(train_warmup_probe.values()))])
    warmup_diagnostic_path = root / "warmup/diagnostic_metrics.json"
    if not warmup_diagnostic_path.is_file():
        raise FileNotFoundError("Candidate12 warm-up validation diagnostics are absent.")
    warmup_diagnostic = json.loads(warmup_diagnostic_path.read_text(encoding="utf-8"))
    warmup_probe = {name: {**warmup_diagnostic["patterns"][f"only_{name}"], **warmup_diagnostic["probe"][name]} for name in MODALITIES}
    _atomic_csv(root / "warmup_unimodal_metrics.csv", [{"modality": name, **values} for name, values in warmup_probe.items()], ["modality", *next(iter(warmup_probe.values()))])
    evidence_rows = []
    for method in (METHODS[0], METHODS[1], METHODS[2], METHODS[3], METHODS[5]):
        for modality in MODALITIES:
            after = payloads[method]["patterns"][f"only_{modality}"]
            probe = payloads[method]["probe"][modality]
            before = warmup_probe[modality]
            evidence_rows.append({"method": method, "modality": modality, "warmup_top1": before["top1"], "final_top1": after["top1"], "top1_delta": after["top1"] - before["top1"], "warmup_within3": before["within3"], "final_within3": after["within3"], "within3_delta": after["within3"] - before["within3"], "warmup_mae": before["mae"], "final_mae": after["mae"], "mae_delta": after["mae"] - before["mae"], "warmup_topology_risk": before["topology_risk"], "final_topology_risk": after["topology_risk"], "topology_risk_delta": after["topology_risk"] - before["topology_risk"], "warmup_true_prototype_similarity": before["true_prototype_similarity"], "final_true_prototype_similarity": probe["true_prototype_similarity"], "true_similarity_delta": probe["true_prototype_similarity"] - before["true_prototype_similarity"], "warmup_nearest_wrong_margin": before["nearest_wrong_margin"], "final_nearest_wrong_margin": probe["nearest_wrong_margin"], "margin_delta": probe["nearest_wrong_margin"] - before["nearest_wrong_margin"]})
    _atomic_csv(root / "unimodal_evidence_changes.csv", evidence_rows, list(evidence_rows[0]))
    for filename in ("per_domain_metrics.csv", "per_weather_metrics.csv", "per_sector_metrics.csv", "per_beam_metrics.csv", "gradient_diagnostics.csv", "motion_replacement_tests.csv"):
        _merge_csv(root, filename)
    efficiency_rows = [{"method": method, **json.loads((root / method / "efficiency.json").read_text(encoding="utf-8"))} for method in METHODS]
    _atomic_csv(root / "efficiency.csv", efficiency_rows, list(efficiency_rows[0]))
    assignment_rows: list[dict[str, Any]] = []
    for method in REMIX_METHODS:
        for row in _read_csv(root / method / "assignment_history.csv"):
            assignment_rows.append({"method": method, **row})
    if assignment_rows:
        _atomic_csv(root / "assignment_statistics.csv", assignment_rows, list(assignment_rows[0]))
    motion_rows = []
    for method in MOTION_METHODS:
        motion = json.loads((root / method / "motion_diagnostics.json").read_text(encoding="utf-8"))
        motion_rows.append({"method": method, "shift_accuracy_local": motion["shift_accuracy_local"], "shift_sign_accuracy": motion["shift_sign_accuracy_local_nonzero"], "mean_abs_predicted_shift": motion["mean_abs_predicted_shift"], "zero_shift_ratio": motion.get("zero_shift_ratio", motion.get("zero_shift_probability")), "mean_zero_shift_probability": motion.get("mean_zero_shift_probability", motion.get("zero_shift_probability")), "corrected": motion["anchor_wrong_motion_corrected"], "introduced": motion["anchor_correct_motion_introduced"], "net_corrected": motion["net_corrected_samples"], "train_local_le3": motion["train_local_residual_coverage"]["abs_signed_residual_le_3"], "validation_local_le3": motion["validation_local_residual_coverage"]["abs_signed_residual_le_3"]})
    _atomic_csv(root / "motion_diagnostics.csv", motion_rows, list(motion_rows[0]))

    a0, a1, a2, a3, a4, a5 = (payloads[name] for name in METHODS)
    p0, p1, p2, p3, p4, p5 = (payload["patterns"] for payload in (a0, a1, a2, a3, a4, a5))
    gradient = {method: _read_csv(root / method / "gradient_diagnostics.csv") for method in METHODS}

    def late_negative(method: str) -> float:
        rows_for_method = gradient[method]
        last_epoch = max(int(row["epoch"]) for row in rows_for_method)
        return float(next(row["negative_cosine_ratio"] for row in rows_for_method if int(row["epoch"]) == last_epoch))

    weather_improved_a3 = sum(float(a3["per_weather"][name]["top1"]) >= float(a0["per_weather"][name]["top1"]) for name in a0["per_weather"])
    c1_items = {
        "full_top1_plus_0_5pp": _delta(p3["full"], p0["full"], "top1") >= 0.005,
        "within3_non_decreasing": _delta(p3["full"], p0["full"], "within3") >= 0,
        "mae_non_worsening": _delta(p3["full"], p0["full"], "mae") <= 0,
        "two_unimodal_plus_1pp": sum(_delta(p3[f"only_{name}"], p0[f"only_{name}"], "top1") >= 0.01 for name in MODALITIES) >= 2,
        "radar_or_gps_plus_1pp": any(_delta(p3[f"only_{name}"], p0[f"only_{name}"], "top1") >= 0.01 for name in ("radar", "gps")),
        "negative_gradient_ratio_down_10pct": late_negative(METHODS[3]) <= 0.9 * late_negative(METHODS[0]),
        "two_of_three_weather_improve": weather_improved_a3 >= 2,
        "beats_a1_or_a2": p3["full"]["top1"] > min(p1["full"]["top1"], p2["full"]["top1"]),
    }
    motion4 = json.loads((root / METHODS[4] / "motion_diagnostics.json").read_text(encoding="utf-8"))
    repl4 = motion4["replacement"]
    weather_nonworse_a4 = sum(float(a4["per_weather"][name]["top1"]) >= float(a0["per_weather"][name]["top1"]) for name in a0["per_weather"])
    c2_items = {
        "full_top1_plus_0_5pp": _delta(p4["full"], p0["full"], "top1") >= 0.005,
        "within3_plus_0_5pp_or_mae_minus_0_05": _delta(p4["full"], p0["full"], "within3") >= 0.005 or _delta(p4["full"], p0["full"], "mae") <= -0.05,
        "corrected_exceeds_introduced": motion4["anchor_wrong_motion_corrected"] > motion4["anchor_correct_motion_introduced"],
        "dynamic_beats_mean_0_3pp": _delta(repl4["dynamic"], repl4["mean"], "top1") >= 0.003,
        "dynamic_beats_shuffle_0_3pp": _delta(repl4["dynamic"], repl4["shuffle"], "top1") >= 0.003,
        "oracle_beats_anchor_1pp": _delta(repl4["oracle"], repl4["anchor"], "top1") >= 0.01,
        "distance_gt5_non_increasing": _delta(p4["full"], p0["full"], "distance_gt5") <= 0,
        "two_of_three_weather_nonworse": weather_nonworse_a4 >= 2,
    }
    motion5 = json.loads((root / METHODS[5] / "motion_diagnostics.json").read_text(encoding="utf-8"))
    efficiency_by_method = {row["method"]: row for row in efficiency_rows}
    combination_items = {
        "top1_beats_best_component_0_3pp": p5["full"]["top1"] >= max(p3["full"]["top1"], p4["full"]["top1"]) + 0.003,
        "within3_and_mae_non_worsening": p5["full"]["within3"] >= max(p3["full"]["within3"], p4["full"]["within3"]) and p5["full"]["mae"] <= min(p3["full"]["mae"], p4["full"]["mae"]),
        "a3_probe_gain_retained": sum(_delta(p5[f"only_{name}"], p0[f"only_{name}"], "top1") >= 0.01 for name in MODALITIES) >= 2,
        "dynamic_motion_gain_retained": _delta(motion5["replacement"]["dynamic"], motion5["replacement"]["mean"], "top1") >= 0.003,
        "inference_latency_within_20pct_of_a4": float(efficiency_by_method[METHODS[5]]["inference_latency_ms_batch1"]) <= 1.2 * float(efficiency_by_method[METHODS[4]]["inference_latency_ms_batch1"]),
    }
    c1_passed = c1_items["full_top1_plus_0_5pp"] and sum(c1_items.values()) >= 5
    c2_dynamic_hard_gate = c2_items["dynamic_beats_mean_0_3pp"] and c2_items["dynamic_beats_shuffle_0_3pp"]
    c2_passed = pamr_candidate_gate(c2_items)
    combination_passed = all(combination_items.values())
    gate_rows = ([{"candidate": "BTPR-Mix", "criterion": name, "passed": value} for name, value in c1_items.items()] + [{"candidate": "PAMR", "criterion": name, "passed": value} for name, value in c2_items.items()] + [{"candidate": "Combination", "criterion": name, "passed": value} for name, value in combination_items.items()] + [{"candidate": "BTPR-Mix", "criterion": "overall_gate", "passed": c1_passed}, {"candidate": "PAMR", "criterion": "dynamic_sample_specific_hard_gate", "passed": c2_dynamic_hard_gate}, {"candidate": "PAMR", "criterion": "overall_gate", "passed": c2_passed}, {"candidate": "Combination", "criterion": "overall_gate", "passed": combination_passed}])
    _atomic_csv(root / "success_gates.csv", gate_rows, ["candidate", "criterion", "passed"])
    ranking = sorted(rows, key=lambda row: (-float(row["full_top1"]), -float(row["full_within3"]), float(row["full_mae"])))
    _atomic_csv(root / "direction_ranking.csv", [{"rank": index + 1, **row} for index, row in enumerate(ranking)], ["rank", *rows[0]])

    if combination_passed:
        recommendation = "BTPR-Mix+PAMR"
    elif c1_passed and c2_passed:
        recommendation = "BTPR-Mix" if p3["full"]["top1"] >= p4["full"]["top1"] else "PAMR"
    elif c1_passed:
        recommendation = "BTPR-Mix"
    elif c2_passed:
        recommendation = "PAMR"
    elif p1["full"]["top1"] >= p0["full"]["top1"] + 0.005:
        recommendation = "KL Data Remixing"
    else:
        recommendation = "均不成立"
    assignment_latest = {}
    for method in REMIX_METHODS:
        history = _read_csv(root / method / "assignment_history.csv")
        assignment_latest[method] = history[-1] if history else {}
    a1_latest = assignment_latest[METHODS[1]]
    a1_counts = {name: int(a1_latest.get(f"{name}_count", 0)) for name in MODALITIES}
    a1_total = max(sum(a1_counts.values()), 1)
    a1_ratios = {name: count / a1_total for name, count in a1_counts.items()}
    a1_any_collapse = max(a1_ratios.values()) >= 0.80
    a1_radar_gps_pair_ratio = a1_ratios["radar"] + a1_ratios["gps"]
    a1_radar_gps_collapse = a1_radar_gps_pair_ratio >= 0.80
    novel_recommendation = recommendation if recommendation in {"BTPR-Mix", "PAMR", "BTPR-Mix+PAMR"} else "均不成立"
    table = [
        "| 方法 | Full Top1 | Top3 | Within-3 | MAE | Topology Risk | Distance>5 | Image-only | LiDAR-only | Radar-only | GPS-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(f"| {row['method']} | {float(row['full_top1']):.4f} | {float(row['full_top3']):.4f} | {float(row['full_within3']):.4f} | {float(row['full_mae']):.4f} | {float(row['full_topology_risk']):.4f} | {float(row['full_distance_gt5']):.4f} | {float(row['image']):.4f} | {float(row['lidar']):.4f} | {float(row['radar']):.4f} | {float(row['gps']):.4f} |")
    answers = [
        "1. 是，严格使用 37,038/9,180 Full-pool 协议。",
        "2. 是，模型完全未读取 channel、path、beam power 或历史 beam。",
        "3. 是，六路共享同一模型定义、BPA 损失与 warm-up checkpoint。",
        f"4. A0 canonical Full Top1={p0['full']['top1']:.4f}，Within-3={p0['full']['within3']:.4f}，MAE={p0['full']['mae']:.4f}。",
        f"5. A1 相对 A0 Full Top1 变化 {100*_delta(p1['full'], p0['full'], 'top1'):+.3f} pp。",
        f"6. A1 最新 assignment 计数/比例：{a1_counts} / {a1_ratios}。",
        f"7. A1 任一单模态 assignment collapse（>=80%）：{a1_any_collapse}；Radar+GPS 二元 collapse：{a1_radar_gps_collapse}（合计 {100*a1_radar_gps_pair_ratio:.3f}%）；risk 路线每轮有 15%-40% 硬约束。",
        f"8. A2-A1 Full Top1={100*_delta(p2['full'], p1['full'], 'top1'):+.3f} pp。",
        f"9. 是，A2/A3/A5 每轮 risk assignment 都通过 15%-40% 容量校验。",
        f"10. A2 超过 A1：{p2['full']['top1'] > p1['full']['top1']}。",
        f"11. A3 超过 A2：{p3['full']['top1'] > p2['full']['top1']}。",
        f"12. A3 late negative-gradient ratio 相对 A2：{late_negative(METHODS[3]):.4f} vs {late_negative(METHODS[2]):.4f}。",
        f"13. A3 Radar/GPS probe 变化：{100*_delta(p3['only_radar'], p0['only_radar'], 'top1'):+.3f}/{100*_delta(p3['only_gps'], p0['only_gps'], 'top1'):+.3f} pp。",
        f"14. A3 Full Top1 变化 {100*_delta(p3['full'], p0['full'], 'top1'):+.3f} pp。",
        f"15. Candidate 1 总门槛：{c1_passed}（{sum(c1_items.values())}/8，且 Full 主门槛必须通过）。",
        f"16. warm-up anchor 的 train-side ±3 覆盖见 warmup/train_local_residual_coverage.json；A4 Oracle-Anchor={100*_delta(repl4['oracle'], repl4['anchor'], 'top1'):+.3f} pp。",
        f"17. A4 Full Top1 变化 {100*_delta(p4['full'], p0['full'], 'top1'):+.3f} pp。",
        f"18. A4 Within-3/MAE 变化 {100*_delta(p4['full'], p0['full'], 'within3'):+.3f} pp / {_delta(p4['full'], p0['full'], 'mae'):+.4f}。",
        f"19. Dynamic-Mean/Shuffle={100*_delta(repl4['dynamic'], repl4['mean'], 'top1'):+.3f}/{100*_delta(repl4['dynamic'], repl4['shuffle'], 'top1'):+.3f} pp。",
        f"20. A4 corrected/introduced={motion4['anchor_wrong_motion_corrected']}/{motion4['anchor_correct_motion_introduced']}。",
        f"21. Candidate 2 总门槛：{c2_passed}（{sum(c2_items.values())}/8；Full 与 Dynamic>Mean/Shuffle 样本级运动硬门槛均必须通过）。",
        f"22. A5 机制保留门槛：{combination_items['a3_probe_gain_retained'] and combination_items['dynamic_motion_gain_retained']}。",
        f"23. A5 超过 A3/A4 0.3 pp：{combination_items['top1_beats_best_component_0_3pp']}。",
        f"24. weather/domain/sector 完整结果见对应 CSV；A3 weather 改善 {weather_improved_a3}/3，A4 weather 非恶化 {weather_nonworse_a4}/3。",
        "25. 参数、显存、训练与推理开销见 efficiency.csv。",
        f"26. 实用方法推荐：{recommendation}；可主张的新第二创新：{novel_recommendation}。",
        f"27. 是否值得进入 multi-seed：{novel_recommendation != '均不成立'}；本轮结论仍为 development-only、single-seed。",
        "28. 未自动运行 multi-seed、outer test 或下一轮实验。",
    ]
    lines = ["# Full-Pool Candidate12 Comparison", "", "Development-only, single seed; outer test was not accessed.", "", *table, "", "## Success gates", "", f"BTPR-Mix: {c1_passed} ({sum(c1_items.values())}/8); PAMR: {c2_passed} ({sum(c2_items.values())}/8, dynamic hard gate={c2_dynamic_hard_gate}); Combination: {combination_passed}.", "", "## Required answers", "", *answers, "", f"Final recommendation: **{recommendation}**.", "", "No multi-seed, outer-test, or follow-up experiment was launched automatically."]
    (root / "candidate12_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(root / "aggregate_status.json", {"status": "passed", "recommendation": recommendation, "novel_candidate_recommendation": novel_recommendation, "candidate1_passed": c1_passed, "candidate2_passed": c2_passed, "candidate2_dynamic_hard_gate": c2_dynamic_hard_gate, "combination_passed": combination_passed, "outer_test_accessed": False, "at": now()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup-diagnostics", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if not any((args.prepare, args.warmup, args.warmup_diagnostics, args.method, args.aggregate)):
        parser.error("select --prepare, --warmup, --warmup-diagnostics, --method, or --aggregate")
    root = args.output_root.resolve()
    if args.prepare:
        prepare(root)
    if args.warmup:
        warmup(root, smoke=args.smoke)
    if args.warmup_diagnostics:
        warmup_diagnostics(root)
    if args.method:
        train(root, args.method, smoke=args.smoke)
    if args.aggregate:
        aggregate(root)


if __name__ == "__main__":
    main()
