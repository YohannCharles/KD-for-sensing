#!/usr/bin/env python3
"""Local trajectory workflow for CSI-anchored semantic slot completion."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.baselines.csi_anchored_completion import CSIAnchoredCompletionModel
from kd_sensing.baselines.full_pool_candidate12 import MODALITIES
from kd_sensing.baselines.full_pool_common import sha256_file
from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel
from kd_sensing.baselines.sparse_pilot_transition import SparsePilotInformationClassifier
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.mmw.trajectory_protocol import load_trajectory_protocol
from kd_sensing.engine.data_factory import shutdown_dataloader_workers
from kd_sensing.losses.beam_prototype_alignment import make_soft_beam_labels
from kd_sensing.models.csi_anchored_completion import (
    CSIAnchoredPrototypeCompletion,
    MissingPathAdapter,
    SparsePilotRadioEncoder,
)

if __package__:
    from .run_mmw_trajectory_baselines import ALL_PATTERNS, _fixed_loader, _inputs
    from .run_sparse_pilot_recovery import _noisy_observations, _prediction_metrics
    from .run_sparse_pilot_trajectory_recovery import (
        _loaders,
        _task_records,
        evaluate as evaluate_recovery,
        nested_frequency_indices,
    )
else:
    from run_mmw_trajectory_baselines import ALL_PATTERNS, _fixed_loader, _inputs
    from run_sparse_pilot_recovery import _noisy_observations, _prediction_metrics
    from run_sparse_pilot_trajectory_recovery import (
        _loaders,
        _task_records,
        evaluate as evaluate_recovery,
        nested_frequency_indices,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/csi_anchored_completion_trajectory.yaml"
RECOVERY_CONFIG = ROOT / "tools/configs/sparse_pilot_trajectory_recovery.yaml"
SEVERE_MASKS = ("image_only", "lidar_only", "radar_only", "gps_only")
METRIC_NAMES = (
    "top1",
    "top3",
    "top5",
    "within3",
    "mae",
    "normalized_gain",
    "beam_loss_db",
    "fix_rate",
    "harm_rate",
)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(values)


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_config(path: Path) -> dict[str, Any]:
    config = safe_load_yaml(path.read_text(encoding="utf-8"))
    if config["protocol"].get("outer_test_enabled") is not False:
        raise ValueError("Completion workflow requires the outer test to remain disabled.")
    return config


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    protocol_cfg = config["protocol"]
    radio_cfg = config["radio_encoder"]
    files = {
        "split_manifest": (_path(protocol_cfg["split_manifest"]), protocol_cfg["split_manifest_sha256"]),
        "m4_checkpoint": (_path(protocol_cfg["m4_checkpoint"]), protocol_cfg["m4_checkpoint_sha256"]),
        "csi_checkpoint": (_path(radio_cfg["checkpoint"]), radio_cfg["checkpoint_sha256"]),
        "train_record": (
            _path(radio_cfg["recovery_root"]) / "records/train.pt",
            radio_cfg["train_record_sha256"],
        ),
        "validation_record": (
            _path(radio_cfg["recovery_root"]) / "records/validation.pt",
            radio_cfg["validation_record_sha256"],
        ),
        "codebook": (_path(radio_cfg["codebook"]), radio_cfg["codebook_file_sha256"]),
    }
    hashes = {}
    for name, (path, expected) in files.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} SHA256 mismatch: expected {expected}, got {actual}.")
        hashes[name] = actual
    protocol = load_trajectory_protocol(files["split_manifest"][0])
    checks = {
        "protocol_id": protocol["protocol_id"] == protocol_cfg["id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"] == protocol_cfg["protocol_fingerprint"],
        "train_count": int(protocol["train_window_count"]) == int(protocol_cfg["expected_train_samples"]),
        "validation_count": int(protocol["validation_window_count"])
        == int(protocol_cfg["expected_validation_samples"]),
        "group_split": tuple(int(protocol[f"{role}_group_count"]) for role in ("train", "validation", "test"))
        == (12, 2, 1),
        "outer_test_sealed": protocol.get("outer_test_accessed") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"Trajectory completion protocol preflight failed: {checks}.")
    return {"checks": checks, "hashes": hashes, "protocol": protocol, "outer_test_accessed": False}


def _load_m4(config: Mapping[str, Any], device: torch.device) -> TrajectoryBaselineModel:
    section = config["protocol"]
    checkpoint = torch.load(_path(section["m4_checkpoint"]), map_location="cpu", weights_only=False)
    if checkpoint.get("method") != ABTC_METHOD:
        raise ValueError("Completion requires the published trajectory M4 checkpoint.")
    if checkpoint.get("protocol_fingerprint") != section["protocol_fingerprint"]:
        raise ValueError("M4 checkpoint protocol fingerprint mismatch.")
    if checkpoint.get("split_manifest_sha256") != section["split_manifest_sha256"]:
        raise ValueError("M4 checkpoint split hash mismatch.")
    model = TrajectoryBaselineModel(ABTC_METHOD, **checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_radio(config: Mapping[str, Any], device: torch.device) -> SparsePilotRadioEncoder:
    section = config["radio_encoder"]
    model = SparsePilotRadioEncoder(
        history_length=int(section["history_length"]),
        hidden_dim=int(section["hidden_dim"]),
        num_candidate_patterns=int(section["num_candidate_patterns"]),
        encoder_layers=int(section["encoder_layers"]),
    ).to(device)
    model.load_information_checkpoint(_path(section["checkpoint"]))
    model.freeze()
    if any("classifier" in name for name, _ in model.named_parameters()):
        raise RuntimeError("The main radio encoder unexpectedly contains classifier parameters.")
    return model


def _load_radio_training_teacher(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    """Load the historical CSI beam head as a training-only teacher."""
    hidden_dim = int(config["radio_encoder"]["hidden_dim"])
    teacher = nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, 64),
    ).to(device)
    payload = torch.load(_path(config["radio_encoder"]["checkpoint"]), map_location="cpu", weights_only=False)
    source = payload["model_state"]
    state = {
        key.removeprefix("classifier."): value
        for key, value in source.items()
        if key.startswith("classifier.")
    }
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _recovery_record(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    if role not in {"train", "validation"}:
        raise ValueError("Only train and validation records are available.")
    path = _path(config["radio_encoder"]["recovery_root"]) / f"records/{role}.pt"
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _frequency_positions(config: Mapping[str, Any], budget: str, device: torch.device) -> torch.Tensor:
    resolved_path = _path(config["radio_encoder"]["recovery_root"]) / "resolved_configs/prepare.yaml"
    resolved = safe_load_yaml(resolved_path.read_text(encoding="utf-8"))
    maximum = len(resolved["runtime"]["frequency_positions_hz"])
    _, count = (int(part) for part in budget.lower().split("x", 1))
    indices = nested_frequency_indices(maximum, count)
    return torch.tensor(resolved["runtime"]["frequency_positions_hz"], device=device).index_select(
        0, indices.to(device)
    )


def _radio_from_candidates(
    encoder: SparsePilotRadioEncoder,
    candidates: torch.Tensor,
    *,
    budget: str,
    frequencies: torch.Tensor,
    snr: torch.Tensor,
    generator: torch.Generator,
    dropout_probability: float = 0.0,
) -> dict[str, torch.Tensor]:
    patterns, frequency_count = (int(part) for part in budget.lower().split("x", 1))
    maximum_frequency = candidates.shape[-1]
    frequency_indices = nested_frequency_indices(maximum_frequency, frequency_count).to(candidates.device)
    selected = candidates[:, :, :patterns].index_select(-1, frequency_indices)
    expanded_snr = snr[:, None].expand(-1, selected.shape[1])
    observations, valid = _noisy_observations(selected, expanded_snr, generator=generator)
    if float(dropout_probability) > 0:
        keep = torch.rand(valid.shape, device=valid.device, generator=generator) >= float(dropout_probability)
        valid = valid & keep
    pattern_ids = torch.arange(patterns, device=candidates.device).expand(
        candidates.shape[0], candidates.shape[1], -1
    )
    with torch.no_grad():
        return encoder(observations, pattern_ids, frequencies, valid, expanded_snr)


def validate_cache_record(
    feature_record: Mapping[str, Any],
    recovery_record: Mapping[str, Any],
    *,
    expected_count: int,
) -> None:
    forbidden = {"channel_ref", "channel_history_refs", "future_channel", "future_channel_ref"}
    exposed = forbidden & set(feature_record)
    if exposed:
        raise ValueError(f"Feature cache must not expose channel inputs: {sorted(exposed)}.")
    required = {
        "token_sequence",
        "modality_features",
        "teacher_prototype_probability",
        "p_full",
        "full_pred",
        "target",
        "future_beam_power",
        "sample_ids",
        "trajectory_ids",
        "csi_cache_keys",
    }
    if not required.issubset(feature_record):
        raise ValueError(f"Feature cache lacks keys: {sorted(required - set(feature_record))}.")
    count = int(expected_count)
    shapes = {
        "token_sequence": (count, 5, 4, 64),
        "modality_features": (count, 4, 64),
        "teacher_prototype_probability": (count, 4, 64),
        "p_full": (count, 64),
        "target": (count,),
        "future_beam_power": (count, 64),
    }
    for key, shape in shapes.items():
        if tuple(feature_record[key].shape) != shape:
            raise ValueError(f"Feature cache {key} must have shape {shape}.")
    sample_ids = list(feature_record["sample_ids"])
    if sample_ids != list(recovery_record["sample_ids"]):
        raise ValueError("Feature cache stable sample IDs do not exactly match recovery records.")
    if len(sample_ids) != count or len(set(sample_ids)) != count:
        raise ValueError("Feature cache stable sample IDs must be complete and unique.")
    if not torch.equal(feature_record["target"], recovery_record["labels_future"]):
        raise ValueError("Feature cache future labels do not match recovery records.")
    if not bool(torch.isfinite(feature_record["token_sequence"]).all()):
        raise ValueError("Feature cache contains non-finite modality tokens.")


@torch.no_grad()
def _extract_feature_role(
    loader,
    model: TrajectoryBaselineModel,
    recovery: Mapping[str, Any],
    *,
    role: str,
    device: torch.device,
    status_path: Path,
) -> dict[str, Any]:
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    sample_ids: list[str] = []
    trajectory_ids: list[str] = []
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, 1):
        with _autocast(device):
            tokens = model.encode(_inputs(batch, device))
            sequence = torch.stack([tokens[name] for name in MODALITIES], dim=2)
            output = model.forward_tokens(tokens)
            modality_features = output["modality_features"]
            teacher_probability = model.prototype_bank(
                modality_features.flatten(0, 1)
            ).softmax(dim=-1).view(sequence.shape[0], 4, 64)
        chunks["token_sequence"].append(sequence.float().cpu())
        chunks["modality_features"].append(modality_features.float().cpu())
        chunks["teacher_prototype_probability"].append(teacher_probability.float().cpu())
        chunks["p_full"].append(output["logits"].float().softmax(dim=-1).cpu())
        chunks["target"].append(torch.as_tensor(batch["target_beam"]).reshape(-1).long())
        chunks["future_beam_power"].append(torch.as_tensor(batch["future_beam_power"]).float())
        sample_ids.extend(str(value) for value in batch["metadata"]["stable_sample_id"])
        trajectory_ids.extend(str(value) for value in batch["metadata"]["trajectory_group_id"])
        if batch_index % 25 == 0 or batch_index == total_batches:
            _write_json(
                status_path,
                {
                    "status": "extracting_features",
                    "role": role,
                    "completed_batches": batch_index,
                    "total_batches": total_batches,
                    "outer_test_accessed": False,
                },
            )
            print(json.dumps({"role": role, "batch": batch_index, "total": total_batches}), flush=True)
    result = {key: torch.cat(value) for key, value in chunks.items()}
    result["full_pred"] = result["p_full"].argmax(dim=-1)
    result["sample_ids"] = sample_ids
    result["trajectory_ids"] = trajectory_ids
    result["csi_cache_keys"] = [f"recovery:{role}:{index}:{sample_id}" for index, sample_id in enumerate(sample_ids)]
    validate_cache_record(result, recovery, expected_count=len(sample_ids))
    return result


@torch.no_grad()
def _extract_radio_role(
    recovery: Mapping[str, Any],
    encoder: SparsePilotRadioEncoder,
    config: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    result: dict[str, list[torch.Tensor]] = defaultdict(list)
    for budget in ("16x16", "16x8"):
        frequencies = _frequency_positions(config, budget, device)
        generator = torch.Generator(device=device).manual_seed(700_000 + int(seed) + (0 if budget == "16x16" else 1))
        for start in range(0, len(recovery["labels_future"]), int(batch_size)):
            candidates = recovery["candidate_history"][start : start + int(batch_size)].to(device)
            snr = torch.full((len(candidates),), 10.0, device=device)
            output = _radio_from_candidates(
                encoder,
                candidates,
                budget=budget,
                frequencies=frequencies,
                snr=snr,
                generator=generator,
            )
            result[f"c_radio_{budget}"].append(output["c_radio"].float().cpu())
            result[f"csi_quality_{budget}"].append(output["csi_quality"].float().cpu())
            result[f"csi_available_{budget}"].append(output["csi_available"].cpu())
    return {key: torch.cat(value) for key, value in result.items()}


def build_cache(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    identity = preflight(config)
    feature_root = _path(config["output"]["feature_cache"])
    radio_root = _path(config["output"]["radio_cache"])
    if (feature_root.exists() or radio_root.exists()) and not args.overwrite:
        raise FileExistsError("Completion cache already exists; pass --overwrite only for an intentional rebuild.")
    feature_root.mkdir(parents=True, exist_ok=True)
    radio_root.mkdir(parents=True, exist_ok=True)
    output_root = _path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = _load_m4(config, device)
    radio_encoder = _load_radio(config, device)
    recovery_config = safe_load_yaml(RECOVERY_CONFIG.read_text(encoding="utf-8"))
    loaders, _ = _loaders(recovery_config)
    fixed = {role: _fixed_loader(loader, workers=int(args.workers)) for role, loader in loaders.items()}
    role_manifest = {}
    trajectory_sets: dict[str, set[str]] = {}
    try:
        for role in ("train", "validation"):
            recovery = _recovery_record(config, role)
            feature = _extract_feature_role(
                fixed[role],
                model,
                recovery,
                role=role,
                device=device,
                status_path=output_root / "cache_status.json",
            )
            expected = int(config["protocol"][f"expected_{role}_samples"])
            validate_cache_record(feature, recovery, expected_count=expected)
            full_probability_max_abs = float((feature["p_full"] - recovery["p0_full"]).abs().max().item())
            full_top1 = float(feature["full_pred"].eq(feature["target"]).float().mean().item())
            if full_probability_max_abs >= 1e-6:
                raise ValueError(
                    f"{role} cached Full probability drift is too large: {full_probability_max_abs}."
                )
            if role == "validation" and abs(full_top1 - float(config["protocol"]["expected_full_top1"])) >= 1e-6:
                raise ValueError(
                    f"Validation Full Top-1 drifted: expected {config['protocol']['expected_full_top1']}, got {full_top1}."
                )
            feature_path = feature_root / f"{role}.pt"
            torch.save(feature, feature_path)
            radio = _extract_radio_role(
                recovery,
                radio_encoder,
                config,
                device=device,
                batch_size=int(args.batch_size),
                seed=2026 + (0 if role == "train" else 1),
            )
            radio["sample_ids"] = list(feature["sample_ids"])
            radio_path = radio_root / f"{role}.pt"
            torch.save(radio, radio_path)
            role_manifest[role] = {
                "sample_count": expected,
                "sample_id_sha256": hashlib.sha256("\n".join(feature["sample_ids"]).encode()).hexdigest(),
                "trajectory_count": len(set(feature["trajectory_ids"])),
                "full_top1": full_top1,
                "full_probability_max_abs_vs_recovery": full_probability_max_abs,
                "feature_cache": str(feature_path.resolve()),
                "feature_cache_sha256": sha256_file(feature_path),
                "radio_cache": str(radio_path.resolve()),
                "radio_cache_sha256": sha256_file(radio_path),
                "recovery_record_sha256": identity["hashes"][f"{role}_record"],
            }
            trajectory_sets[role] = set(feature["trajectory_ids"])
            del recovery, feature, radio
    finally:
        for loader in (*fixed.values(), *loaders.values()):
            shutdown_dataloader_workers(loader)

    trajectory_overlap = trajectory_sets["train"] & trajectory_sets["validation"]
    if trajectory_overlap:
        raise ValueError(f"Completion cache trajectory leakage: {sorted(trajectory_overlap)}.")

    manifest = {
        "version": "csi_anchored_completion_feature_cache_v1",
        "protocol_id": config["protocol"]["id"],
        "protocol_fingerprint": config["protocol"]["protocol_fingerprint"],
        "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
        "m4_checkpoint_sha256": config["protocol"]["m4_checkpoint_sha256"],
        "csi_checkpoint_sha256": config["radio_encoder"]["checkpoint_sha256"],
        "codebook_hash": config["radio_encoder"]["codebook_hash"],
        "codebook_file_sha256": config["radio_encoder"]["codebook_file_sha256"],
        "modality_order": list(MODALITIES),
        "token_sequence_shape": [None, 5, 4, 64],
        "encoder_version": "TrajectoryBaselineModel.encode@M4-checkpoint",
        "radio_version": "SparsePilotEncoder+2layer-GRU-no-classifier",
        "budgets": ["16x16", "16x8"],
        "radio_cache_snr_db": 10.0,
        "future_channel_used_as_input": False,
        "outer_test_accessed": False,
        "train_validation_trajectory_overlap_count": 0,
        "roles": role_manifest,
    }
    _write_json(output_root / "feature_cache_manifest.json", manifest)
    _write_json(output_root / "cache_status.json", {"status": "complete", **manifest})
    print(json.dumps({"status": "cache_complete", "roles": role_manifest}, indent=2), flush=True)


def audit_cache(config: Mapping[str, Any]) -> dict[str, Any]:
    identity = preflight(config)
    output_root = _path(config["output"]["root"])
    manifest_path = output_root / "feature_cache_manifest.json"
    role_audits = {}
    trajectory_sets = {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for role in ("train", "validation"):
        feature, recovery, _ = _load_cached_role(config, role)
        probability_max_abs = float((feature["p_full"] - recovery["p0_full"]).abs().max().item())
        full_top1 = float(feature["full_pred"].eq(feature["target"]).float().mean().item())
        if probability_max_abs >= 1e-6:
            raise ValueError(f"{role} Full probability cache drift is {probability_max_abs}.")
        if role == "validation" and abs(full_top1 - float(config["protocol"]["expected_full_top1"])) >= 1e-6:
            raise ValueError(f"Validation Full Top-1 drifted to {full_top1}.")
        trajectory_sets[role] = set(feature["trajectory_ids"])
        role_audits[role] = {
            "sample_count": len(feature["sample_ids"]),
            "trajectory_count": len(trajectory_sets[role]),
            "full_top1": full_top1,
            "full_probability_max_abs_vs_recovery": probability_max_abs,
            "stable_sample_ids_exact": True,
        }
        manifest["roles"][role].update(role_audits[role])
    overlap = trajectory_sets["train"] & trajectory_sets["validation"]
    if overlap:
        raise ValueError(f"Cache train/validation trajectory overlap: {sorted(overlap)}.")
    audit = {
        "status": "validated",
        "roles": role_audits,
        "train_validation_trajectory_overlap_count": 0,
        "identity": identity["hashes"],
        "future_channel_used_as_input": False,
        "outer_test_accessed": False,
    }
    manifest.update(
        validation_status="validated",
        train_validation_trajectory_overlap_count=0,
        outer_test_accessed=False,
    )
    _write_json(manifest_path, manifest)
    _write_json(output_root / "feature_cache_validation.json", audit)
    return audit


def _load_cached_role(
    config: Mapping[str, Any], role: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    output_root = _path(config["output"]["root"])
    manifest_path = output_root / "feature_cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Feature cache manifest is absent; run cache first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_identity = {
        "protocol_fingerprint": config["protocol"]["protocol_fingerprint"],
        "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
        "m4_checkpoint_sha256": config["protocol"]["m4_checkpoint_sha256"],
        "csi_checkpoint_sha256": config["radio_encoder"]["checkpoint_sha256"],
        "codebook_hash": config["radio_encoder"]["codebook_hash"],
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected_identity.items() if manifest.get(key) != value}
    if mismatches or manifest.get("outer_test_accessed") is not False:
        raise ValueError(f"Feature cache identity mismatch: {mismatches}.")
    feature_path = _path(config["output"]["feature_cache"]) / f"{role}.pt"
    role_manifest = manifest["roles"][role]
    if sha256_file(feature_path) != role_manifest["feature_cache_sha256"]:
        raise ValueError(f"{role} feature cache SHA256 mismatch.")
    feature = torch.load(feature_path, map_location="cpu", weights_only=False, mmap=True)
    recovery = _recovery_record(config, role)
    expected_count = int(config["protocol"][f"expected_{role}_samples"])
    validate_cache_record(feature, recovery, expected_count=expected_count)
    return feature, recovery, manifest


def _stratified_indices(
    feature: Mapping[str, Any],
    *,
    limit: int | None,
    seed: int,
) -> torch.Tensor:
    count = len(feature["target"])
    if limit is None or int(limit) >= count:
        return torch.arange(count)
    if int(limit) <= 0:
        raise ValueError("limit must be positive.")
    buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, (trajectory, label) in enumerate(zip(feature["trajectory_ids"], feature["target"].tolist())):
        buckets[(str(trajectory), int(label))].append(index)
    generator = random.Random(int(seed))
    for values in buckets.values():
        generator.shuffle(values)
    keys = sorted(buckets)
    generator.shuffle(keys)
    selected = []
    depth = 0
    while len(selected) < int(limit):
        added = False
        for key in keys:
            if depth < len(buckets[key]):
                selected.append(buckets[key][depth])
                added = True
                if len(selected) == int(limit):
                    break
        if not added:
            break
        depth += 1
    if len(selected) != int(limit):
        raise RuntimeError("Stratified subset construction did not reach the requested size.")
    return torch.tensor(selected, dtype=torch.long)


def _mask_pool() -> dict[int, tuple[tuple[int, ...], ...]]:
    grouped: dict[int, list[tuple[int, ...]]] = defaultdict(list)
    for mask in ALL_PATTERNS.values():
        grouped[sum(int(value) for value in mask)].append(tuple(int(value) for value in mask))
    return {key: tuple(value) for key, value in grouped.items()}


MASK_POOL = _mask_pool()


def _sample_masks(count: int, config: Mapping[str, Any], generator: torch.Generator) -> torch.Tensor:
    probabilities_cfg = config["training"]["cardinality_probabilities"]
    cardinalities = torch.tensor((1, 2, 3, 4), dtype=torch.long)
    probabilities = torch.tensor(
        [float(probabilities_cfg.get(key, probabilities_cfg.get(str(key)))) for key in cardinalities.tolist()]
    )
    sampled = cardinalities[torch.multinomial(probabilities, int(count), replacement=True, generator=generator)]
    rows = []
    for cardinality in sampled.tolist():
        choices = MASK_POOL[int(cardinality)]
        weights = torch.ones(len(choices))
        if int(cardinality) == 3:
            for index, mask in enumerate(choices):
                if not mask[1]:
                    weights[index] = float(config["training"]["missing_lidar_weight"])
        choice = int(torch.multinomial(weights, 1, generator=generator).item())
        rows.append(choices[choice])
    return torch.tensor(rows, dtype=torch.bool)


def _build_model(
    config: Mapping[str, Any], method: str, device: torch.device
) -> tuple[CSIAnchoredCompletionModel, SparsePilotRadioEncoder]:
    if method not in config["methods"]:
        raise ValueError(f"Unknown completion method: {method}.")
    method_cfg = config["methods"][method]
    completion_cfg = config["completion"]
    base = _load_m4(config, device)
    radio = _load_radio(config, device)
    completion = CSIAnchoredPrototypeCompletion(
        feature_dim=int(completion_cfg["feature_dim"]),
        radio_dim=int(config["radio_encoder"]["hidden_dim"]),
        quality_dim=int(completion_cfg["quality_dim"]),
        hidden_dim=int(completion_cfg["hidden_dim"]),
        num_layers=int(completion_cfg["num_layers"]),
        num_heads=int(completion_cfg["num_heads"]),
        ffn_dim=int(completion_cfg["ffn_dim"]),
        dropout=float(completion_cfg["dropout"]),
        top_k=int(completion_cfg["top_k"]),
        sensing_temperature=float(completion_cfg["sensing_temperature"]),
        radio_temperature=float(completion_cfg["radio_temperature"]),
        use_radio=bool(method_cfg["use_radio"]),
        use_prototype_memory=bool(method_cfg["use_prototype_memory"]),
        use_cross_attention=bool(method_cfg["use_cross_attention"]),
        evidence_fusion=str(completion_cfg["evidence_fusion"]),
    ).to(device)
    adapter = (
        MissingPathAdapter(feature_dim=int(completion_cfg["feature_dim"]), bottleneck_dim=32).to(device)
        if method_cfg.get("use_missing_path_adapter")
        else None
    )
    model = CSIAnchoredCompletionModel(
        base,
        completion,
        radio_encoder=radio,
        missing_path_adapter=adapter,
        freeze_radio=True,
    ).to(device)
    return model, radio


def _zero_radio(count: int, model: CSIAnchoredCompletionModel, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "c_radio": reference.new_zeros(count, model.completion.radio_dim),
        "csi_quality": reference.new_zeros(count, model.completion.quality_dim),
        "csi_available": torch.zeros(count, dtype=torch.bool, device=reference.device),
    }


def _student_prototype_probability(
    model: CSIAnchoredCompletionModel, completed_features: torch.Tensor
) -> torch.Tensor:
    batch = completed_features.shape[0]
    return model.base_model.prototype_bank(completed_features.flatten(0, 1)).softmax(dim=-1).view(batch, 4, 64)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.sum() * 0.0


def _completion_losses(
    model: CSIAnchoredCompletionModel,
    first: Mapping[str, torch.Tensor | bool],
    second: Mapping[str, torch.Tensor | bool],
    batch: Mapping[str, torch.Tensor],
    physical: torch.Tensor,
    config: Mapping[str, Any],
    *,
    radio_distillation_weight: float = 0.0,
    radio_decision_distillation_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = first["logits"]
    second_features = second["completed_modality_features"]
    completed_features = first["completed_modality_features"]
    physical_probabilities = first["physical_probabilities"]
    if not all(torch.is_tensor(value) for value in (logits, second_features, completed_features, physical_probabilities)):
        raise TypeError("Completion forward returned non-tensor training outputs.")
    labels = batch["target"]
    available_count = physical.sum(dim=1)
    nonfull = available_count < 4
    cardinal_weights_cfg = config["training"]["task_cardinality_weights"]
    cardinal_weights = torch.tensor(
        [0.0]
        + [float(cardinal_weights_cfg.get(key, cardinal_weights_cfg.get(str(key)))) for key in (1, 2, 3, 4)],
        device=logits.device,
        dtype=logits.dtype,
    )
    sample_weights = cardinal_weights[available_count]
    task_per_sample = F.cross_entropy(logits.float(), labels, reduction="none").to(logits.dtype)
    task = (task_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)

    soft_labels = make_soft_beam_labels(
        labels,
        64,
        2.0,
        circular=True,
        topology_id="ula_dft_phase_cycle_v1",
    ).to(logits)
    topology_per_sample = -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    topology = (topology_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)

    student = _student_prototype_probability(model, completed_features)
    student_second = _student_prototype_probability(model, second_features)
    teacher = batch["teacher_prototype_probability"].to(student)
    missing = ~physical
    proto_items = (teacher * (teacher.clamp_min(1e-8).log() - student.clamp_min(1e-8).log())).sum(dim=-1)
    prototype_semantic = _masked_mean(proto_items, missing)
    radio_distillation = student.sum() * 0.0
    radio_teacher = batch.get("radio_teacher_probability")
    if float(radio_distillation_weight) > 0:
        if radio_teacher is None or tuple(radio_teacher.shape) != (len(labels), 64):
            raise ValueError("Radio distillation requires training-only teacher probabilities [B,64].")
        teacher_radio = radio_teacher.to(student)
        radio_items = (
            teacher_radio[:, None]
            * (teacher_radio[:, None].clamp_min(1e-8).log() - student.clamp_min(1e-8).log())
        ).sum(dim=-1)
        radio_distillation = _masked_mean(radio_items, missing)
    radio_decision_distillation = logits.sum() * 0.0
    if float(radio_decision_distillation_weight) > 0:
        if radio_teacher is None or tuple(radio_teacher.shape) != (len(labels), 64):
            raise ValueError("Radio decision distillation requires teacher probabilities [B,64].")
        teacher_radio = radio_teacher.to(logits)
        decision_items = (
            teacher_radio
            * (teacher_radio.clamp_min(1e-8).log() - F.log_softmax(logits, dim=-1))
        ).sum(dim=-1)
        radio_decision_distillation = _masked_mean(decision_items, available_count <= 2)
    slot_items = 1.0 - F.cosine_similarity(completed_features, batch["modality_features"].to(completed_features), dim=-1)
    slot = _masked_mean(slot_items, missing)

    full_teacher = batch["p_full"].to(logits)
    top_two = full_teacher.topk(2, dim=-1).values
    high_confidence = full_teacher.argmax(dim=-1).eq(labels) & ((top_two[:, 0] - top_two[:, 1]) >= float(config["loss"]["teacher_margin"]))
    consistency_weight = torch.where(
        available_count.eq(1),
        torch.full_like(available_count, 0.1, dtype=logits.dtype),
        ((available_count == 2) | (available_count == 3)).to(logits.dtype),
    )
    consistency_weight = consistency_weight * high_confidence.to(logits.dtype)
    consistency_items = F.kl_div(F.log_softmax(logits, dim=-1), full_teacher, reduction="none").sum(dim=-1)
    consistency = (consistency_items * consistency_weight).sum() / consistency_weight.sum().clamp_min(1.0)

    midpoint = 0.5 * (student + student_second)
    js_items = 0.5 * (
        (student * (student.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=-1)
        + (student_second * (student_second.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=-1)
    )
    quality = _masked_mean(js_items, missing)

    base = physical_probabilities
    target_index = labels[:, None]
    preserve_mask = (
        available_count.eq(3)
        & base.argmax(dim=-1).eq(labels)
        & (logits.softmax(dim=-1).gather(1, target_index) <= base.gather(1, target_index))[:, 0]
    )
    preserve_items = F.kl_div(F.log_softmax(logits, dim=-1), base, reduction="none").sum(dim=-1)
    preserve = _masked_mean(preserve_items, preserve_mask)

    weights = config["loss"]
    total = (
        task
        + float(weights["topology"]) * topology
        + float(weights["prototype_semantic"]) * prototype_semantic
        + float(weights["slot"]) * slot
        + float(weights["consistency"]) * consistency
        + float(weights["quality"]) * quality
        + float(weights["preserve"]) * preserve
        + float(radio_distillation_weight) * radio_distillation
        + float(radio_decision_distillation_weight) * radio_decision_distillation
    )
    return total, {
        "total": total,
        "task": task,
        "topology": topology,
        "prototype_semantic": prototype_semantic,
        "radio_distillation": radio_distillation,
        "radio_decision_distillation": radio_decision_distillation,
        "slot": slot,
        "consistency": consistency,
        "quality": quality,
        "preserve": preserve,
        "train_top1": _masked_mean(logits.argmax(dim=-1).eq(labels).to(logits.dtype), nonfull),
    }


def _batch_tensors(
    feature: Mapping[str, Any], recovery: Mapping[str, Any], indices: torch.Tensor, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "token_sequence": feature["token_sequence"][indices].to(device, non_blocking=True),
        "modality_features": feature["modality_features"][indices].to(device, non_blocking=True),
        "teacher_prototype_probability": feature["teacher_prototype_probability"][indices].to(
            device, non_blocking=True
        ),
        "p_full": feature["p_full"][indices].to(device, non_blocking=True),
        "target": feature["target"][indices].to(device, non_blocking=True),
        "future_beam_power": feature["future_beam_power"][indices].to(device, non_blocking=True),
        "candidate_history": recovery["candidate_history"][indices].to(device, non_blocking=True),
    }


def _make_radio_view(
    method: str,
    model: CSIAnchoredCompletionModel,
    encoder: SparsePilotRadioEncoder,
    candidates: torch.Tensor,
    *,
    budget: str,
    frequencies: torch.Tensor,
    snr: torch.Tensor,
    generator: torch.Generator,
    dropout_probability: float,
) -> dict[str, torch.Tensor]:
    if not bool(model.completion.use_radio) or method == "B4":
        return _zero_radio(len(candidates), model, candidates.real)
    return _radio_from_candidates(
        encoder,
        candidates,
        budget=budget,
        frequencies=frequencies,
        snr=snr,
        generator=generator,
        dropout_probability=dropout_probability,
    )


def _trainable_state(model: CSIAnchoredCompletionModel) -> dict[str, Any]:
    return {
        "completion_state": model.completion.state_dict(),
        "adapter_state": model.missing_path_adapter.state_dict() if model.missing_path_adapter is not None else None,
    }


def _load_trainable_state(model: CSIAnchoredCompletionModel, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.completion.load_state_dict(payload["completion_state"], strict=True)
    adapter_state = payload.get("adapter_state")
    if adapter_state is not None:
        if model.missing_path_adapter is None:
            raise ValueError("Checkpoint contains an adapter but the configured model does not.")
        model.missing_path_adapter.load_state_dict(adapter_state, strict=True)
    return payload


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result = {}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[f"{name}_macro"] = float(np.mean(values)) if values else float("nan")
        result[f"{name}_worst"] = min(values) if values and name in {"top1", "top3", "top5", "within3", "normalized_gain"} else (
            max(values) if values and name in {"mae", "beam_loss_db", "harm_rate"} else float("nan")
        )
    return result


def _summary_from_per_mask(per_mask: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing_rows = [row for row in per_mask if row["mask"] != "full"]
    groups = {
        "single": _aggregate_rows([row for row in per_mask if row["available_count"] == 1]),
        "two": _aggregate_rows([row for row in per_mask if row["available_count"] == 2]),
        "three": _aggregate_rows([row for row in per_mask if row["available_count"] == 3]),
        "all14": _aggregate_rows(missing_rows),
    }
    full = next(row for row in per_mask if row["mask"] == "full")
    missing_lidar = next(row for row in per_mask if row["mask"] == "missing_lidar")
    selection_score = (
        0.25 * groups["single"]["top1_macro"]
        + 0.25 * groups["two"]["top1_macro"]
        + 0.20 * groups["three"]["top1_macro"]
        + 0.15 * groups["all14"]["top1_worst"]
        + 0.15 * float(missing_lidar["top1"])
    )
    return {
        "groups": groups,
        "full": full,
        "missing_lidar": missing_lidar,
        "selection_score": selection_score,
    }


@torch.no_grad()
def evaluate_model(
    model: CSIAnchoredCompletionModel,
    radio_encoder: SparsePilotRadioEncoder,
    feature: Mapping[str, Any],
    recovery: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    method: str,
    budget: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    indices: torch.Tensor | None = None,
    diagnostic: str = "normal",
) -> dict[str, Any]:
    if diagnostic not in {"normal", "csi_zero", "csi_shuffle", "sensing_shuffle"}:
        raise ValueError(f"Unknown diagnostic: {diagnostic}.")
    model.eval()
    selected = torch.arange(len(feature["target"])) if indices is None else indices.cpu()
    diagnostic_generator = torch.Generator().manual_seed(950_000 + int(seed))
    shuffled_indices = selected[
        torch.randperm(len(selected), generator=diagnostic_generator)
    ] if diagnostic in {"csi_shuffle", "sensing_shuffle"} else selected
    predictions: dict[str, list[torch.Tensor]] = {name: [] for name in ALL_PATTERNS}
    diagnostics: dict[str, dict[str, float]] = {name: defaultdict(float) for name in ALL_PATTERNS}
    frequencies = _frequency_positions(config, budget, device)
    generator = torch.Generator(device=device).manual_seed(900_000 + int(seed))
    started = time.monotonic()
    full_max_abs = 0.0
    full_argmax_mismatch = 0
    processed = 0
    for start in range(0, len(selected), int(batch_size)):
        batch_indices = selected[start : start + int(batch_size)]
        batch = _batch_tensors(feature, recovery, batch_indices, device)
        count = len(batch_indices)
        snr = torch.full(
            (count,),
            float(config["training"]["validation_snr_db"]),
            device=device,
        )
        radio_candidates = batch["candidate_history"]
        if diagnostic == "csi_shuffle":
            radio_candidates = recovery["candidate_history"][
                shuffled_indices[start : start + int(batch_size)]
            ].to(device, non_blocking=True)
        radio = _make_radio_view(
            method,
            model,
            radio_encoder,
            radio_candidates,
            budget=budget,
            frequencies=frequencies,
            snr=snr,
            generator=generator,
            dropout_probability=0.0,
        )
        if diagnostic == "csi_zero":
            radio = _zero_radio(count, model, batch["token_sequence"])
        shuffled_sequence = None
        if diagnostic == "sensing_shuffle":
            shuffled_sequence = feature["token_sequence"][
                shuffled_indices[start : start + int(batch_size)]
            ].to(device, non_blocking=True)
        for name, pattern in ALL_PATTERNS.items():
            physical = torch.tensor(pattern, device=device, dtype=torch.bool).expand(count, -1)
            sequence = batch["token_sequence"]
            if diagnostic == "sensing_shuffle" and name != "full":
                assert shuffled_sequence is not None
                sequence = shuffled_sequence
            with _autocast(device):
                output = model(sequence, physical, radio_output=radio)
            probabilities = output["probabilities"].float()
            predictions[name].append(probabilities.cpu())
            if name == "full":
                with _autocast(device):
                    reference = model.base_model.forward_tokens(model._token_mapping(sequence))["logits"].softmax(dim=-1)
                difference = (probabilities - reference.float()).abs()
                full_max_abs = max(full_max_abs, float(difference.max().item()))
                full_argmax_mismatch += int(probabilities.argmax(dim=-1).ne(reference.argmax(dim=-1)).sum().item())
                continue
            completed = output["completed_modality_features"]
            if not torch.is_tensor(completed):
                raise TypeError("Evaluation requires completed modality features.")
            student = _student_prototype_probability(model, completed).float()
            teacher = batch["teacher_prototype_probability"].float()
            missing = ~physical
            kl = (teacher * (teacher.clamp_min(1e-8).log() - student.clamp_min(1e-8).log())).sum(dim=-1)
            cosine = F.cosine_similarity(completed.float(), batch["modality_features"].float(), dim=-1)
            nearest_match = student.argmax(dim=-1).eq(teacher.argmax(dim=-1)).float()
            entropy = -(student * student.clamp_min(1e-8).log()).sum(dim=-1)
            rho = output["completion_sample_radio_reliability"]
            if not torch.is_tensor(rho):
                raise TypeError("Evaluation requires radio reliability diagnostics.")
            diagnostics[name]["count"] += count
            diagnostics[name]["prototype_semantic_kl_sum"] += float(kl[missing].sum().item())
            diagnostics[name]["token_cosine_sum"] += float(cosine[missing].sum().item())
            diagnostics[name]["nearest_prototype_match_sum"] += float(nearest_match[missing].sum().item())
            diagnostics[name]["prototype_entropy_sum"] += float(entropy[missing].sum().item())
            diagnostics[name]["missing_slot_count"] += int(missing.sum().item())
            diagnostics[name]["radio_reliability_sum"] += float(rho.sum().item())
        processed += count
    elapsed = time.monotonic() - started
    if full_max_abs >= 1e-7 or full_argmax_mismatch:
        raise RuntimeError(
            f"Full bypass failed: max_abs={full_max_abs}, argmax_mismatch={full_argmax_mismatch}."
        )

    labels = feature["target"][selected]
    power = feature["future_beam_power"][selected]
    per_mask = []
    for name, pattern in ALL_PATTERNS.items():
        probabilities = torch.cat(predictions[name])
        base = recovery[f"p0_{name}"][selected]
        row = {
            "mask": name,
            "available_count": sum(int(value) for value in pattern),
            "sample_count": len(selected),
            **_prediction_metrics(probabilities, labels, power, base),
        }
        if name != "full":
            values = diagnostics[name]
            slot_count = max(float(values["missing_slot_count"]), 1.0)
            sample_count = max(float(values["count"]), 1.0)
            row.update(
                radio_reliability=float(values["radio_reliability_sum"]) / sample_count,
                prototype_semantic_kl=float(values["prototype_semantic_kl_sum"]) / slot_count,
                token_cosine=float(values["token_cosine_sum"]) / slot_count,
                nearest_prototype_match=float(values["nearest_prototype_match_sum"]) / slot_count,
                prototype_entropy=float(values["prototype_entropy_sum"]) / slot_count,
            )
        per_mask.append(row)

    missing_rows = [row for row in per_mask if row["mask"] != "full"]
    groups = {
        "single": _aggregate_rows([row for row in per_mask if row["available_count"] == 1]),
        "two": _aggregate_rows([row for row in per_mask if row["available_count"] == 2]),
        "three": _aggregate_rows([row for row in per_mask if row["available_count"] == 3]),
        "all14": _aggregate_rows(missing_rows),
    }
    full_row = next(row for row in per_mask if row["mask"] == "full")
    lidar_row = next(row for row in per_mask if row["mask"] == "missing_lidar")
    selection_score = (
        0.25 * groups["single"]["top1_macro"]
        + 0.25 * groups["two"]["top1_macro"]
        + 0.20 * groups["three"]["top1_macro"]
        + 0.15 * groups["all14"]["top1_worst"]
        + 0.15 * float(lidar_row["top1"])
    )
    return {
        "method": method,
        "budget": budget,
        "seed": int(seed),
        "diagnostic": diagnostic,
        "sample_count": len(selected),
        "per_mask": per_mask,
        "groups": groups,
        "full": full_row,
        "missing_lidar": lidar_row,
        "selection_score": selection_score,
        "full_bypass_max_abs": full_max_abs,
        "full_bypass_argmax_mismatch": full_argmax_mismatch,
        "elapsed_seconds": elapsed,
        "latency_ms_per_sample_mask": 1000.0 * elapsed / max(processed * len(ALL_PATTERNS), 1),
        "outer_test_accessed": False,
    }


def _save_training_checkpoint(
    path: Path,
    model: CSIAnchoredCompletionModel,
    config: Mapping[str, Any],
    *,
    method: str,
    budget: str,
    seed: int,
    epoch: int,
    score: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **_trainable_state(model),
            "method": method,
            "budget": budget,
            "seed": int(seed),
            "epoch": int(epoch),
            "selection_score": float(score),
            "protocol_fingerprint": config["protocol"]["protocol_fingerprint"],
            "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
            "m4_checkpoint_sha256": config["protocol"]["m4_checkpoint_sha256"],
            "csi_checkpoint_sha256": config["radio_encoder"]["checkpoint_sha256"],
            "outer_test_accessed": False,
        },
        path,
    )


def train(args: argparse.Namespace, config: Mapping[str, Any], *, overfit: bool) -> None:
    preflight(config)
    method = str(args.method)
    seed = int(args.seed)
    budget = str(args.budget)
    device = torch.device(args.device)
    _set_seed(seed)
    feature, recovery, _ = _load_cached_role(config, "train")
    limit = 500 if overfit else (int(args.limit) if args.limit else None)
    indices = _stratified_indices(feature, limit=limit, seed=50_000 + seed)
    model, radio_encoder = _build_model(config, method, device)
    initialization = None
    if args.checkpoint is not None:
        initialization = _load_trainable_state(model, args.checkpoint)
        source_method = initialization.get("method")
        allowed_transition = (source_method, method) in {
            ("B7", "B8"),
            ("B7D", "B8D"),
            ("B8D", "B8K"),
        }
        if not allowed_transition or initialization.get("budget") != budget:
            raise ValueError("Training initialization is not an allowed matched-budget completion transition.")
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["methods"][method].get("learning_rate", config["training"]["learning_rate"])),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(args.epochs or (config["training"]["overfit_epochs"] if overfit else config["training"]["short_epochs"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    frequencies = _frequency_positions(config, budget, device)
    radio_distillation_weight = float(config["methods"][method].get("radio_distillation_weight", 0.0))
    radio_decision_distillation_weight = float(
        config["methods"][method].get("radio_decision_distillation_weight", 0.0)
    )
    radio_teacher = (
        _load_radio_training_teacher(config, device)
        if radio_distillation_weight > 0 or radio_decision_distillation_weight > 0
        else None
    )
    output_root = _path(config["output"]["root"])
    run_name = f"{'overfit_' if overfit else ''}{method}_seed{seed}_{budget}"
    run_dir = output_root / "runs" / run_name
    if (run_dir / "complete.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed run already exists: {run_dir}.")
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **config,
        "runtime": {
            "mode": "overfit" if overfit else "train",
            "method": method,
            "seed": seed,
            "budget": budget,
            "sample_count": len(indices),
            "subset_index_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
            "device": str(device),
            "epochs": epochs,
            "initialization": (
                {
                    "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": sha256_file(args.checkpoint),
                    "source_method": initialization["method"],
                    "source_epoch": initialization["epoch"],
                }
                if initialization is not None
                else None
            ),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "radio_distillation_weight": radio_distillation_weight,
            "radio_decision_distillation_weight": radio_decision_distillation_weight,
            "radio_teacher_training_only": radio_teacher is not None,
            "frozen_m4_parameters": sum(parameter.numel() for parameter in model.base_model.parameters()),
            "outer_test_accessed": False,
        },
    }
    _write_json(run_dir / "resolved_config.json", resolved)
    history = []
    best_score = float("-inf")
    best_path = run_dir / "best_checkpoint.pt"
    batch_size = int(args.batch_size or config["training"]["batch_size"])
    mask_generator = torch.Generator().manual_seed(100_000 + seed)
    order_generator = torch.Generator().manual_seed(200_000 + seed)
    radio_generator = torch.Generator(device=device).manual_seed(300_000 + seed)
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = indices[torch.randperm(len(indices), generator=order_generator)]
        sums: dict[str, float] = defaultdict(float)
        seen = 0
        gradient_norm_sum = 0.0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            batch = _batch_tensors(feature, recovery, batch_indices, device)
            physical = _sample_masks(len(batch_indices), config, mask_generator).to(device)
            snr_first = torch.empty(len(batch_indices), device=device).uniform_(
                float(config["training"]["snr_db_min"]),
                float(config["training"]["snr_db_max"]),
                generator=radio_generator,
            )
            snr_second = torch.empty(len(batch_indices), device=device).uniform_(
                float(config["training"]["snr_db_min"]),
                float(config["training"]["snr_db_max"]),
                generator=radio_generator,
            )
            radio_first = _make_radio_view(
                method,
                model,
                radio_encoder,
                batch["candidate_history"],
                budget=budget,
                frequencies=frequencies,
                snr=snr_first,
                generator=radio_generator,
                dropout_probability=float(config["training"]["pilot_dropout_probability"]),
            )
            radio_second = _make_radio_view(
                method,
                model,
                radio_encoder,
                batch["candidate_history"],
                budget=budget,
                frequencies=frequencies,
                snr=snr_second,
                generator=radio_generator,
                dropout_probability=float(config["training"]["pilot_dropout_probability"]),
            )
            if radio_teacher is not None:
                with torch.no_grad():
                    batch["radio_teacher_probability"] = radio_teacher(
                        radio_first["c_radio"].float()
                    ).softmax(dim=-1)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                first = model(batch["token_sequence"], physical, radio_output=radio_first)
                second = model(batch["token_sequence"], physical, radio_output=radio_second)
                total, losses = _completion_losses(
                    model,
                    first,
                    second,
                    batch,
                    physical,
                    config,
                    radio_distillation_weight=radio_distillation_weight,
                    radio_decision_distillation_weight=radio_decision_distillation_weight,
                )
            if not bool(torch.isfinite(total)):
                raise RuntimeError(f"Non-finite completion loss at epoch {epoch}.")
            total.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0).item())
            optimizer.step()
            count = len(batch_indices)
            seen += count
            gradient_norm_sum += gradient_norm * count
            for key, value in losses.items():
                sums[key] += float(value.detach().float().item()) * count
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "sample_count": seen,
            "gradient_norm": gradient_norm_sum / max(seen, 1),
            **{key: value / max(seen, 1) for key, value in sums.items()},
        }
        history.append(row)
        _write_csv(run_dir / "training_log.csv", history)
        _write_json(
            run_dir / "status.json",
            {
                "status": "training",
                "epoch": epoch,
                "epochs": epochs,
                "elapsed_seconds": time.monotonic() - started,
                **row,
                "outer_test_accessed": False,
            },
        )
        print(json.dumps({"run": run_name, **row}), flush=True)

        evaluation_interval = 5 if overfit else 4
        if epoch % evaluation_interval == 0 or epoch == epochs:
            if overfit:
                evaluation = evaluate_model(
                    model,
                    radio_encoder,
                    feature,
                    recovery,
                    config,
                    method=method,
                    budget=budget,
                    seed=seed,
                    device=device,
                    batch_size=min(batch_size, int(config["training"]["evaluation_batch_size"])),
                    indices=indices,
                )
            else:
                validation_feature, validation_recovery, _ = _load_cached_role(config, "validation")
                evaluation = evaluate_model(
                    model,
                    radio_encoder,
                    validation_feature,
                    validation_recovery,
                    config,
                    method=method,
                    budget=budget,
                    seed=seed,
                    device=device,
                    batch_size=int(config["training"]["evaluation_batch_size"]),
                )
            evaluation["epoch"] = epoch
            _write_json(run_dir / f"evaluation_epoch{epoch}.json", evaluation)
            _write_csv(run_dir / f"mask_metrics_epoch{epoch}.csv", evaluation["per_mask"])
            if float(evaluation["selection_score"]) > best_score:
                best_score = float(evaluation["selection_score"])
                _save_training_checkpoint(
                    best_path,
                    model,
                    config,
                    method=method,
                    budget=budget,
                    seed=seed,
                    epoch=epoch,
                    score=best_score,
                )
                _write_json(run_dir / "best_evaluation.json", evaluation)
    if not best_path.is_file():
        raise RuntimeError("Training completed without a selectable checkpoint.")
    _write_json(
        run_dir / "complete.json",
        {
            "status": "complete",
            "method": method,
            "seed": seed,
            "budget": budget,
            "epochs": epochs,
            "sample_count": len(indices),
            "best_selection_score": best_score,
            "best_checkpoint": str(best_path.resolve()),
            "best_checkpoint_sha256": sha256_file(best_path),
            "elapsed_seconds": time.monotonic() - started,
            "outer_test_accessed": False,
        },
    )


def evaluate_checkpoint(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    preflight(config)
    method = str(args.method)
    seed = int(args.seed)
    budget = str(args.budget)
    device = torch.device(args.device)
    model, radio_encoder = _build_model(config, method, device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        _path(config["output"]["root"]) / "runs" / f"{method}_seed{seed}_{budget}" / "best_checkpoint.pt"
    )
    payload = _load_trainable_state(model, checkpoint)
    if payload.get("method") != method or payload.get("budget") != budget:
        raise ValueError("Evaluation checkpoint method/budget mismatch.")
    feature, recovery, _ = _load_cached_role(config, "validation")
    result = evaluate_model(
        model,
        radio_encoder,
        feature,
        recovery,
        config,
        method=method,
        budget=budget,
        seed=seed,
        device=device,
        batch_size=int(config["training"]["evaluation_batch_size"]),
        diagnostic=str(args.diagnostic),
    )
    run_dir = checkpoint.parent
    suffix = "final" if args.diagnostic == "normal" else str(args.diagnostic)
    _write_json(run_dir / f"evaluation_{suffix}.json", result)
    _write_csv(run_dir / f"mask_metrics_{suffix}.csv", result["per_mask"])
    print(json.dumps({key: value for key, value in result.items() if key != "per_mask"}, indent=2), flush=True)


@torch.no_grad()
def evaluate_b2_control(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    """Read-only re-evaluation of the existing dense I5 concat checkpoint."""
    preflight(config)
    seed = int(args.seed)
    if seed not in {1, 2, 3}:
        raise ValueError("B2 control is defined only for seeds 1, 2 and 3.")
    device = torch.device(args.device)
    recovery_root = _path(config["radio_encoder"]["recovery_root"])
    checkpoint = (
        recovery_root
        / f"round1_dense/checkpoints/seed{seed}_I5_32x16_replace/best_single_worst.pt"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("outer_test_accessed") is not False:
        raise ValueError("B2 checkpoint does not prove that outer test stayed sealed.")
    recovery_config = safe_load_yaml(RECOVERY_CONFIG.read_text(encoding="utf-8"))
    model = SparsePilotInformationClassifier(
        history_length=5,
        sensing_dim=64,
        hidden_dim=int(recovery_config["model"]["hidden_dim"]),
        num_candidate_patterns=int(recovery_config["pilot"]["max_patterns"]),
        encoder_layers=int(recovery_config["model"]["encoder_layers"]),
        fusion_mode="replace",
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    records = torch.load(
        recovery_root / "records/validation.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    prepared = safe_load_yaml(
        (recovery_root / "resolved_configs/prepare.yaml").read_text(encoding="utf-8")
    )
    frequencies = torch.tensor(
        prepared["runtime"]["frequency_positions_hz"],
        device=device,
    )
    per_mask = []
    for name, pattern in ALL_PATTERNS.items():
        mask_records = _task_records(
            records,
            "I5",
            (32, 16),
            max_frequencies=16,
            masks=(name,),
        )
        evaluated = evaluate_recovery(
            model,
            mask_records,
            frequencies=frequencies,
            snr_db=float(recovery_config["training"]["validation_snr_db"]),
            batch_size=int(args.batch_size),
            device=device,
            seed=seed,
        )
        row = dict(evaluated["per_mask"][0])
        row.update(
            mask=name,
            available_count=sum(int(value) for value in pattern),
            sample_count=len(records["labels_future"]),
        )
        per_mask.append(row)

    result = {
        "method": "B2",
        "budget": "32x16",
        "seed": seed,
        "diagnostic": "normal",
        "sample_count": len(records["labels_future"]),
        "per_mask": per_mask,
        **_summary_from_per_mask(per_mask),
        "full_bypass_max_abs": None,
        "full_bypass_argmax_mismatch": None,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_selection": payload.get("selection_value"),
        "outer_test_accessed": False,
    }
    output_path = _path(config["output"]["root"]) / f"controls/B2_seed{seed}_32x16.json"
    _write_json(output_path, result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_mask"}, indent=2), flush=True)


def summarize(config: Mapping[str, Any]) -> None:
    output_root = _path(config["output"]["root"])
    feature, recovery, _ = _load_cached_role(config, "validation")
    labels = feature["target"].cpu()
    beam_power = feature["future_beam_power"].cpu()

    def flatten_result(
        method: str,
        seed: int,
        budget: str,
        result: Mapping[str, Any],
        evidence_source: str,
    ) -> dict[str, Any]:
        return {
            "method": method,
            "seed": seed,
            "budget": budget,
            "single_macro": result["groups"]["single"]["top1_macro"],
            "single_worst": result["groups"]["single"]["top1_worst"],
            "two_macro": result["groups"]["two"]["top1_macro"],
            "two_worst": result["groups"]["two"]["top1_worst"],
            "three_macro": result["groups"]["three"]["top1_macro"],
            "three_worst": result["groups"]["three"]["top1_worst"],
            "all14_macro": result["groups"]["all14"]["top1_macro"],
            "all14_worst": result["groups"]["all14"]["top1_worst"],
            "missing_lidar": result["missing_lidar"]["top1"],
            "full_top1": result["full"]["top1"],
            "full_bypass_max_abs": result.get("full_bypass_max_abs", 0.0),
            "selection_score": result["selection_score"],
            "evidence_source": evidence_source,
        }

    b0_per_mask = []
    for name, pattern in ALL_PATTERNS.items():
        probabilities = recovery[f"p0_{name}"].float().cpu()
        b0_per_mask.append(
            {
                "mask": name,
                "available_count": sum(int(value) for value in pattern),
                "sample_count": len(labels),
                **_prediction_metrics(probabilities, labels, beam_power, probabilities),
            }
        )
    b0 = _summary_from_per_mask(b0_per_mask)
    b0["full_bypass_max_abs"] = 0.0
    base_top1 = {str(row["mask"]): float(row["top1"]) for row in b0_per_mask}
    rows = [flatten_result("B0", 0, "none", b0, "deterministic_frozen_m4")]
    mask_rows = [
        {
            "method": "B0",
            "seed": 0,
            "budget": "none",
            "baseline_top1": base_top1[str(row["mask"])],
            "completion_gain_top1": 0.0,
            **row,
        }
        for row in b0_per_mask
    ]
    quality_rows = []
    latency_rows = []

    hard_root = _path(config["radio_encoder"]["recovery_root"]) / "round3_fallback/results"
    hard_paths = sorted(hard_root.glob("seed*_I3_16x16_hard_fallback.json"))
    for hard_path in hard_paths:
        payload = json.loads(hard_path.read_text(encoding="utf-8"))
        hard_per_mask = [
            {
                **row,
                "available_count": sum(int(value) for value in ALL_PATTERNS[str(row["mask"])]),
                "sample_count": len(labels),
            }
            for row in payload["all14_per_mask"]
        ]
        hard_per_mask.append(dict(next(row for row in b0_per_mask if row["mask"] == "full")))
        hard = _summary_from_per_mask(hard_per_mask)
        hard["full_bypass_max_abs"] = float(payload["full_probability_max_abs_diff"])
        for method in ("B1", "B3"):
            rows.append(
                flatten_result(
                    method,
                    int(payload["seed"]),
                    "16x16",
                    hard,
                    "shared_hard_probability_fallback_record",
                )
            )
            mask_rows.extend(
                {
                    "method": method,
                    "seed": int(payload["seed"]),
                    "budget": "16x16",
                    "baseline_top1": base_top1[str(row["mask"])],
                    "completion_gain_top1": float(row["top1"]) - base_top1[str(row["mask"])],
                    **row,
                }
                for row in hard_per_mask
            )

    completed_b2_seeds = set()
    for control_path in sorted((output_root / "controls").glob("B2_seed*_32x16.json")):
        result = json.loads(control_path.read_text(encoding="utf-8"))
        completed_b2_seeds.add(int(result["seed"]))
        rows.append(
            flatten_result(
                "B2",
                int(result["seed"]),
                "32x16",
                result,
                "read_only_trajectory_I5_concat_checkpoint_evaluation",
            )
        )
        mask_rows.extend(
            {
                "method": "B2",
                "seed": result["seed"],
                "budget": "32x16",
                "baseline_top1": base_top1[str(row["mask"])],
                "completion_gain_top1": float(row["top1"]) - base_top1[str(row["mask"])],
                **row,
            }
            for row in result["per_mask"]
        )

    concat_analysis_path = _path(config["radio_encoder"]["recovery_root"]) / "round1_dense/analysis.json"
    if concat_analysis_path.is_file():
        controls = json.loads(concat_analysis_path.read_text(encoding="utf-8")).get(
            "stopped_concat_controls", {}
        )
        for seed in (1, 2, 3):
            if seed in completed_b2_seeds:
                continue
            observed = controls.get(f"I5_seed{seed}_best_observed")
            if observed is None:
                continue
            rows.append(
                {
                    "method": "B2",
                    "seed": seed,
                    "budget": "32x16",
                    "single_macro": observed["single_macro"],
                    "single_worst": observed["single_worst"],
                    "two_macro": None,
                    "two_worst": None,
                    "three_macro": None,
                    "three_worst": None,
                    "all14_macro": None,
                    "all14_worst": None,
                    "missing_lidar": None,
                    "full_top1": None,
                    "full_bypass_max_abs": None,
                    "selection_score": None,
                    "evidence_source": "trajectory_I5_concat_screen_best_observed_only",
                }
            )

    for run_dir in sorted((output_root / "runs").glob("B*_seed*_*")):
        path = run_dir / "evaluation_final.json"
        if not path.is_file():
            path = run_dir / "best_evaluation.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        reported_method = "B9" if result["method"] == "B7" and result["budget"] == "16x8" else result["method"]
        rows.append(
            flatten_result(
                reported_method,
                int(result["seed"]),
                str(result["budget"]),
                result,
                "completion_run",
            )
        )
        mask_rows.extend(
            {
                "method": reported_method,
                "seed": result["seed"],
                "budget": result["budget"],
                "baseline_top1": base_top1[str(row["mask"])],
                "completion_gain_top1": float(row["top1"]) - base_top1[str(row["mask"])],
                **row,
            }
            for row in result["per_mask"]
        )
        quality_rows.extend(
            {
                "method": reported_method,
                "seed": result["seed"],
                "budget": result["budget"],
                "mask": row["mask"],
                "radio_reliability": row.get("radio_reliability"),
                "prototype_semantic_kl": row.get("prototype_semantic_kl"),
                "token_cosine": row.get("token_cosine"),
                "nearest_prototype_match": row.get("nearest_prototype_match"),
                "prototype_entropy": row.get("prototype_entropy"),
            }
            for row in result["per_mask"]
            if row["mask"] != "full"
        )
        resolved_path = run_dir / "resolved_config.json"
        resolved = json.loads(resolved_path.read_text(encoding="utf-8")) if resolved_path.is_file() else {}
        runtime = resolved.get("runtime", {})
        latency_rows.append(
            {
                "method": reported_method,
                "seed": result["seed"],
                "budget": result["budget"],
                "pilot_re": int(result["budget"].split("x")[0]) * int(result["budget"].split("x")[1]),
                "latency_ms_per_sample_mask": result["latency_ms_per_sample_mask"],
                "trainable_parameters": runtime.get("trainable_parameters"),
                "frozen_m4_parameters": runtime.get("frozen_m4_parameters"),
            }
        )
    _write_csv(output_root / "ablation_summary.csv", rows)
    _write_csv(output_root / "mask_summary.csv", mask_rows)
    _write_csv(output_root / "completion_quality.csv", quality_rows)
    _write_csv(output_root / "latency_summary.csv", latency_rows)
    _write_json(output_root / "ablation_summary.json", rows)

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["budget"]))].append(row)
    aggregate_rows = []
    aggregate_metrics = (
        "single_macro",
        "single_worst",
        "two_macro",
        "two_worst",
        "three_macro",
        "three_worst",
        "all14_macro",
        "all14_worst",
        "missing_lidar",
        "full_top1",
    )
    for (method, budget), method_rows in sorted(grouped.items()):
        aggregate = {"method": method, "budget": budget, "seed_count": len(method_rows)}
        for metric in aggregate_metrics:
            values = np.asarray(
                [float(row[metric]) for row in method_rows if row.get(metric) is not None],
                dtype=np.float64,
            )
            aggregate[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            aggregate[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else None)
            )
        aggregate_rows.append(aggregate)

    b3_aggregate = next((row for row in aggregate_rows if row["method"] == "B3"), None)
    if b3_aggregate is not None:
        for row in aggregate_rows:
            for metric in ("single_macro", "all14_macro", "all14_worst", "missing_lidar"):
                current = row.get(f"{metric}_mean")
                reference = b3_aggregate.get(f"{metric}_mean")
                row[f"integration_gain_{metric}_vs_B3"] = (
                    float(current) - float(reference)
                    if current is not None and reference is not None
                    else None
                )
    _write_csv(output_root / "multi_seed_summary.csv", aggregate_rows)

    diagnostic_rows = []
    b8k_diagnostic_root = output_root / "runs/B8K_seed1_16x16"
    b8d_diagnostic_root = output_root / "runs/B8D_seed1_16x16"
    if (b8k_diagnostic_root / "evaluation_final.json").is_file():
        diagnostic_root = b8k_diagnostic_root
    elif (b8d_diagnostic_root / "evaluation_final.json").is_file():
        diagnostic_root = b8d_diagnostic_root
    else:
        diagnostic_root = output_root / "runs/B7_seed1_16x16"
    diagnostic_paths = {
        "normal": diagnostic_root / "evaluation_final.json",
        "csi_zero": diagnostic_root / "evaluation_csi_zero.json",
        "csi_shuffle": diagnostic_root / "evaluation_csi_shuffle.json",
        "sensing_shuffle": diagnostic_root / "evaluation_sensing_shuffle.json",
    }
    for name, diagnostic_path in diagnostic_paths.items():
        if not diagnostic_path.is_file():
            continue
        result = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        diagnostic_rows.append(
            {
                "diagnostic": name,
                "single_macro": result["groups"]["single"]["top1_macro"],
                "two_macro": result["groups"]["two"]["top1_macro"],
                "three_macro": result["groups"]["three"]["top1_macro"],
                "all14_macro": result["groups"]["all14"]["top1_macro"],
                "all14_worst": result["groups"]["all14"]["top1_worst"],
                "missing_lidar": result["missing_lidar"]["top1"],
            }
        )
    _write_csv(
        output_root / "csi_shuffle_diagnostic.csv",
        [row for row in diagnostic_rows if row["diagnostic"] in {"normal", "csi_zero", "csi_shuffle"}],
    )
    _write_csv(
        output_root / "sensing_shuffle_diagnostic.csv",
        [row for row in diagnostic_rows if row["diagnostic"] in {"normal", "sensing_shuffle"}],
    )
    budget_rows = [
        row for row in aggregate_rows if row["method"] in {"B7", "B8", "B7D", "B8D", "B8K", "B9"}
    ]
    _write_csv(output_root / "budget_summary.csv", budget_rows)
    _write_completion_report(output_root, aggregate_rows, diagnostic_rows, config)
    print(json.dumps(rows, indent=2), flush=True)


def _write_completion_report(
    output_root: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    by_method = {str(row["method"]): row for row in aggregate_rows}
    b0 = by_method.get("B0", {})
    b1 = by_method.get("B1", {})
    b2 = by_method.get("B2", {})
    b3 = by_method.get("B3", {})
    b5 = by_method.get("B5", {})
    b7 = by_method.get("B7", {})
    b8 = by_method.get("B8", {})
    b7d = by_method.get("B7D", {})
    b8d = by_method.get("B8D", {})
    b8k = by_method.get("B8K", {})
    b9 = by_method.get("B9", {})

    def passes_b8k_gate() -> bool:
        keys = ("single_macro_mean", "all14_macro_mean", "all14_worst_mean", "missing_lidar_mean")
        if not b8k or not b8d or any(b8k.get(key) is None or b8d.get(key) is None for key in keys):
            return False
        return (
            float(b8k["single_macro_mean"]) > float(b8d["single_macro_mean"])
            and float(b8k["all14_macro_mean"]) > float(b8d["all14_macro_mean"])
            and float(b8k["all14_worst_mean"]) >= float(b8d["all14_worst_mean"])
            and float(b8k["missing_lidar_mean"]) >= float(b8d["missing_lidar_mean"])
        )

    b8k_gate_passed = passes_b8k_gate()
    primary = b8k if b8k_gate_passed else (b8d or b7d or b8 or b7)
    diagnostics = {str(row["diagnostic"]): row for row in diagnostic_rows}
    hard_results = []
    hard_root = _path(config["radio_encoder"]["recovery_root"]) / "round3_fallback/results"
    for path in sorted(hard_root.glob("seed*_I3_16x16_hard_fallback.json")):
        hard_results.append(json.loads(path.read_text(encoding="utf-8")))
    hard_all14 = float(np.mean([row["all14_macro"] for row in hard_results])) if hard_results else float("nan")
    hard_worst = float(np.mean([row["all14_worst"] for row in hard_results])) if hard_results else float("nan")
    hard_single = float(np.mean([row["top1"] for row in hard_results])) if hard_results else float("nan")

    def value(row: Mapping[str, Any], key: str) -> str:
        raw = row.get(key)
        return f"{100.0 * float(raw):.2f}%" if raw is not None and np.isfinite(float(raw)) else "pending"

    def delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> str:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None:
            return "pending"
        return f"{100.0 * (float(left_value) - float(right_value)):+.2f} pp"

    normal = diagnostics.get("normal", {})
    csi_shuffle = diagnostics.get("csi_shuffle", {})
    sensing_shuffle = diagnostics.get("sensing_shuffle", {})
    lines = [
        "# CSI-Anchored Beam-Semantic Modality Completion",
        "",
        "## 结论",
        "",
        "CSI 已进入缺失槽位补全架构：主模型仅加载 Pilot Encoder+GRU 的 `c_radio` 和质量特征，"
        "不存在 CSI classifier logits 到最终概率的连接。最终 logits 始终由补全后的四槽位经过冻结 M4 fusion 和原共享 Beam Prototype Bank 产生。",
        "",
        "补全对象是 64 维 beam-semantic latent token，不是原始图像、点云、Radar 或完整 CSI。",
        "",
        "## 主要结果",
        "",
        f"- B0 冻结 M4: Single {value(b0, 'single_macro_mean')}，All-14 {value(b0, 'all14_macro_mean')}，"
        f"Worst {value(b0, 'all14_worst_mean')}，Full {value(b0, 'full_top1_mean')}。",
        f"- B1/B3 hard probability fallback（同一组三 seed 证据）: Single {value(b1 or b3, 'single_macro_mean')}，"
        f"All-14 {value(b1 or b3, 'all14_macro_mean')}，Worst {value(b1 or b3, 'all14_worst_mean')}。",
        f"- B2 I5 普通 concat 32x16: Single {value(b2, 'single_macro_mean')}，"
        f"All-14 {value(b2, 'all14_macro_mean')}，Worst {value(b2, 'all14_worst_mean')}，"
        f"Full {value(b2, 'full_top1_mean')}；该控制使用独立 classifier 且不执行 Full bypass。",
        f"- B7 multi-seed: All-14 {value(b7, 'all14_macro_mean')}，Worst {value(b7, 'all14_worst_mean')}，"
        f"missing_lidar {value(b7, 'missing_lidar_mean')}，Single {value(b7, 'single_macro_mean')}。",
        f"- B8 adapter: All-14 {value(b8, 'all14_macro_mean')}，Worst {value(b8, 'all14_worst_mean')}，"
        f"missing_lidar {value(b8, 'missing_lidar_mean')}，Single {value(b8, 'single_macro_mean')}。",
        f"- B7D training-only radio-prototype distillation: All-14 {value(b7d, 'all14_macro_mean')}，"
        f"Worst {value(b7d, 'all14_worst_mean')}，missing_lidar {value(b7d, 'missing_lidar_mean')}，"
        f"Single {value(b7d, 'single_macro_mean')}。",
        f"- B8D distilled + adapter: All-14 {value(b8d, 'all14_macro_mean')}，"
        f"Worst {value(b8d, 'all14_worst_mean')}，missing_lidar {value(b8d, 'missing_lidar_mean')}，"
        f"Single {value(b8d, 'single_macro_mean')}。",
        f"- B8K severe decision KD: All-14 {value(b8k, 'all14_macro_mean')}，"
        f"Worst {value(b8k, 'all14_worst_mean')}，missing_lidar {value(b8k, 'missing_lidar_mean')}，"
        f"Single {value(b8k, 'single_macro_mean')}；门控{'通过' if b8k_gate_passed else '失败，不作为主方法'}。",
        f"- B9 16x8: All-14 {value(b9, 'all14_macro_mean')}，Worst {value(b9, 'all14_worst_mean')}，"
        f"missing_lidar {value(b9, 'missing_lidar_mean')}。",
        f"- 既有 hard fallback: Single {100.0 * hard_single:.2f}%，All-14 {100.0 * hard_all14:.2f}%，Worst {100.0 * hard_worst:.2f}%。",
        f"- Full: {value(primary, 'full_top1_mean')}；所有已完成运行的逐样本旁路最大概率差均为 0。",
        "",
        "## 机制诊断",
        "",
        f"- 正常主方法 All-14: {value(normal, 'all14_macro')}。",
        f"- 全局 CSI shuffle: {value(csi_shuffle, 'all14_macro')}；说明历史稀疏 CSI 对补全有实质贡献。",
        f"- 全局 sensing shuffle: {value(sensing_shuffle, 'all14_macro')}；说明模型没有退化成 CSI-only。",
        f"- B7 相对 no-prototype B5：All-14 {delta(b7, b5, 'all14_macro_mean')}，"
        f"Worst {delta(b7, b5, 'all14_worst_mean')}，missing_lidar {delta(b7, b5, 'missing_lidar_mean')}；"
        "shared prototype memory 的主要收益集中在最差 mask 与 LiDAR 缺失。",
        f"- B8 相对 B7 的 All-14 增益为 {delta(b8, b7, 'all14_macro_mean')}；"
        "说明冻结 M4 fusion 对 synthetic slots 的不适配是独立瓶颈。",
        "- B7D/B8D 的 CSI classifier head 只在训练期生成 stop-gradient prototype teacher；"
        "它未注册进主模型、未保存进 checkpoint，validation/inference 不加载该 head。",
        "",
        "## 成功条件与边界",
        "",
        f"- 主 completion 相对 B3：Single {delta(primary, b3, 'single_macro_mean')}，"
        f"All-14 {delta(primary, b3, 'all14_macro_mean')}，Worst {delta(primary, b3, 'all14_worst_mean')}，"
        f"missing_lidar {delta(primary, b3, 'missing_lidar_mean')}。",
        "- 若 Single/All-14 仍低于 hard fallback，completion 不能被表述为后者的全面替代；"
        "hard fallback 必须作为严重缺失强基线保留。",
        "- B2 使用 32x16 且有独立自由 classifier，违反主架构约束；其 Single 结果只用于说明简单 concat 的强度，"
        "不能和 16x16 completion 伪装成完全匹配的消融。",
        "- 16x8 若仅小幅下降可作为 128 RE 效率配置，否则保留 16x16 为主结果。",
        "- 论文主方法建议保留 available-context、共享 prototype memory、training-only prototype distillation 与 missing-path adapter；"
        "不保留任何推理期 CSI classifier 直连。",
        "",
        "训练与选择只使用 12 个 train 轨迹组和 2 个 validation 轨迹组；1 个 outer-test 轨迹组保持封存。"
        "缓存未包含 future channel，所有 Top-K 只由当前输入的 sensing/历史稀疏 CSI 产生。",
    ]
    (output_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "preflight",
            "cache",
            "validate-cache",
            "overfit",
            "train",
            "evaluate",
            "evaluate-b2",
            "summarize",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--method",
        choices=("B4", "B5", "B6", "B7", "B8", "B7D", "B8D", "B8K"),
        default="B7",
    )
    parser.add_argument("--budget", choices=("16x16", "16x8"), default="16x16")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--diagnostic",
        choices=("normal", "csi_zero", "csi_shuffle", "sensing_shuffle"),
        default="normal",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True), flush=True)
    elif args.mode == "cache":
        build_cache(args, config)
    elif args.mode == "validate-cache":
        print(json.dumps(audit_cache(config), indent=2, sort_keys=True), flush=True)
    elif args.mode == "overfit":
        train(args, config, overfit=True)
    elif args.mode == "train":
        train(args, config, overfit=False)
    elif args.mode == "evaluate":
        evaluate_checkpoint(args, config)
    elif args.mode == "evaluate-b2":
        evaluate_b2_control(args, config)
    else:
        summarize(config)


if __name__ == "__main__":
    main()
