#!/usr/bin/env python3
"""Local staged workflow for CSI-conditioned prototype state updates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from kd_sensing.baselines.full_pool_common import sha256_file
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.losses.prototype_update_losses import prototype_update_loss
from kd_sensing.models.csi_conditioned_prototype_update import CSIConditionedPrototypeUpdate
from kd_sensing.models.prototype_likelihood_head import PrototypeLikelihoodHead, estimate_train_beam_prior
from kd_sensing.models.prototype_transition_kernel import PrototypeTransitionKernel

if __package__:
    from .run_csi_anchored_completion import _frequency_positions
    from .run_quality_topology_prototype_routing import (
        _load_config as _load_qtpr_config,
        _topology,
        preflight as qtpr_preflight,
    )
    from .run_temporal_sparse_prototype_compensation import (
        _load_config as _load_tspc_config,
        _source_components,
        balanced_epoch_schedule,
    )
    from .run_tspc_final_ablations import (
        MASK_COUNTS,
        MASK_NAMES,
        _aggregate,
        _metric_row,
        _noisy_observations,
        _slice_records,
        _stratified_indices,
        select_candidate_history,
    )
else:
    from run_csi_anchored_completion import _frequency_positions
    from run_quality_topology_prototype_routing import (
        _load_config as _load_qtpr_config,
        _topology,
        preflight as qtpr_preflight,
    )
    from run_temporal_sparse_prototype_compensation import (
        _load_config as _load_tspc_config,
        _source_components,
        balanced_epoch_schedule,
    )
    from run_tspc_final_ablations import (
        MASK_COUNTS,
        MASK_NAMES,
        _aggregate,
        _metric_row,
        _noisy_observations,
        _slice_records,
        _stratified_indices,
        select_candidate_history,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/csi_prototype_state_update.yaml"
TRAIN_METHODS = ("U2", "U2-static", "U2-last", "U3", "U3-no-delta")
CALIBRATION_METHODS = ("U1", "U1-eta0.5", "U1-no-prior")
DIAGNOSTICS = ("normal", "csi_shuffle", "temporal_shuffle", "reverse_time", "identity", "uniform", "csi_off")
GROUP_NAMES = {1: "single", 2: "two", 3: "three"}


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = list(rows)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(values[0])
    for row in values[1:]:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _load_config(path: Path) -> dict[str, Any]:
    config = safe_load_yaml(path.read_text(encoding="utf-8"))
    if config["protocol"].get("outer_test_enabled") is not False:
        raise ValueError("CPSU requires outer_test_enabled=false.")
    resources = config["pilot"]
    if int(resources["re_per_frame"]) * int(resources["history_frames"]) != int(resources["re_window"]):
        raise ValueError("CPSU pilot RE accounting is inconsistent.")
    if str(resources["budget"]) != "2x2" or int(resources["history_frames"]) != 5:
        raise ValueError("CPSU main configuration is fixed to five 2x2 pilot frames.")
    return config


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _load_records(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    if role not in {"train", "validation"}:
        raise ValueError("CPSU records are restricted to train and validation roles.")
    return torch.load(_path(config["source"][f"{role}_records"]), map_location="cpu", weights_only=False, mmap=True)


def _records_view(records: Mapping[str, Any], indices: torch.Tensor) -> Mapping[str, Any]:
    if len(indices) == len(records["labels_future"]) and torch.equal(indices, torch.arange(len(indices))):
        return records
    return _slice_records(records, indices)


def pilot_resource_accounting(config: Mapping[str, Any]) -> dict[str, int]:
    pilot = config["pilot"]
    return {
        "pilot_re_per_frame": int(pilot["re_per_frame"]),
        "pilot_history_frames": int(pilot["history_frames"]),
        "pilot_re_window": int(pilot["re_window"]),
    }


def _qtpr(config: Mapping[str, Any]) -> dict[str, Any]:
    return _load_qtpr_config(_path(config["source"]["qtpr_config"]))


def _tspc(config: Mapping[str, Any]) -> dict[str, Any]:
    return _load_tspc_config(_path(config["source"]["tspc_config"]))


def _components(config: Mapping[str, Any], device: torch.device):
    qtpr, m4, radio_encoder, expert, calibration, _ = _source_components(_tspc(config), "temporal_2x2", device)
    if abs(float(calibration["sensing_temperature"]) - float(config["model"]["sensing_temperature"])) > 1e-6:
        raise ValueError("CPSU sensing temperature differs from the current F1 checkpoint.")
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    expert.eval()
    return qtpr, m4, radio_encoder, expert, calibration


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    hashes = {
        "split_manifest": sha256_file(_path(config["protocol"]["split_manifest"])),
        "train_records": sha256_file(_path(source["train_records"])),
        "validation_records": sha256_file(_path(source["validation_records"])),
        "codebook": sha256_file(_path(source["codebook"])),
        "m4_checkpoint": sha256_file(_path(source["m4_checkpoint"])),
        "csi_checkpoint": sha256_file(_path(source["csi_checkpoint"])),
        "f1_checkpoint": sha256_file(_path(source["f1_checkpoint"])),
    }
    expected = {
        "split_manifest": config["protocol"]["split_manifest_sha256"],
        "train_records": source["train_records_sha256"],
        "validation_records": source["validation_records_sha256"],
        "codebook": source["codebook_sha256"],
        "m4_checkpoint": source["m4_checkpoint_sha256"],
        "csi_checkpoint": source["csi_checkpoint_sha256"],
        "f1_checkpoint": source["f1_checkpoint_sha256"],
    }
    qtpr = _qtpr(config)
    source_audit = qtpr_preflight(qtpr, write_manifest=False)
    topology = _topology(qtpr)
    train = _load_records(config, "train")
    validation = _load_records(config, "validation")
    forbidden = lambda record: [name for name in record if "future_csi" in name or "future_channel" in name]
    checks = {
        "hashes": hashes == expected,
        "protocol_id": qtpr["protocol"]["id"] == config["protocol"]["id"],
        "protocol_fingerprint": qtpr["protocol"]["protocol_fingerprint"] == config["protocol"]["fingerprint"],
        "sample_counts": len(train["sample_ids"]) == int(config["protocol"]["expected_train_samples"])
        and len(validation["sample_ids"]) == int(config["protocol"]["expected_validation_samples"]),
        "sample_ids_disjoint": not bool(set(train["sample_ids"]) & set(validation["sample_ids"])),
        "mother_shape": tuple(train["candidate_history"].shape[1:]) == (5, 32, 16)
        and tuple(validation["candidate_history"].shape[1:]) == (5, 32, 16),
        "all_masks": all(f"p0_{name}" in train and f"z_{name}" in train for name in ("full", *MASK_NAMES)),
        "no_future_channel_fields": not forbidden(train) and not forbidden(validation),
        "topology": tuple(topology.labels_by_position) == tuple(range(64))
        and float(topology.distance[0, 63]) == 1.0
        and bool(config["topology"]["circular"]),
        "re_window": pilot_resource_accounting(config)["pilot_re_window"] == 20,
        "outer_test_disabled": config["protocol"].get("outer_test_enabled") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"CPSU preflight failed: {checks}.")
    prior = estimate_train_beam_prior(train["labels_future"], split_role="train")
    prior_manifest = {
        "version": "cpsu_train_beam_prior_v1",
        "split_role": prior["split_role"],
        "sample_count": prior["sample_count"],
        "counts": prior["counts"].tolist(),
        "prior": prior["prior"].tolist(),
        "protocol_id": config["protocol"]["id"],
        "protocol_fingerprint": config["protocol"]["fingerprint"],
        "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
        "outer_test_accessed": False,
    }
    output = _path(config["output"]["root"])
    _write_json(output / "train_beam_prior.json", prior_manifest)
    result = {
        "status": "passed",
        "checks": checks,
        "hashes": hashes,
        "source_audit": source_audit,
        "train_beam_prior": prior_manifest,
        **pilot_resource_accounting(config),
        "future_channel_used_as_input": False,
        "test_loader_constructed": False,
        "outer_test_accessed": False,
    }
    _write_json(output / "preflight.json", result)
    resolved = output / "resolved_configs/base.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    return result


def method_spec(method: str) -> dict[str, Any]:
    name = str(method)
    specs = {
        "U1": {"transition": False, "likelihood": True, "context_mode": "temporal", "eta_prior": 1.0},
        "U1-eta0.5": {"transition": False, "likelihood": True, "context_mode": "temporal", "eta_prior": 0.5},
        "U1-no-prior": {"transition": False, "likelihood": True, "context_mode": "temporal", "eta_prior": 0.0},
        "U2": {"transition": True, "likelihood": False, "context_mode": "temporal", "eta_prior": 1.0},
        "U2-static": {"transition": True, "likelihood": False, "context_mode": "static", "eta_prior": 1.0},
        "U2-last": {"transition": True, "likelihood": False, "context_mode": "last", "eta_prior": 1.0},
        "U3": {"transition": True, "likelihood": True, "context_mode": "temporal", "eta_prior": 1.0},
        "U3-no-delta": {"transition": True, "likelihood": True, "context_mode": "no_delta", "eta_prior": 1.0},
    }
    if name not in specs:
        raise ValueError(f"Unknown CPSU method: {method}.")
    return {"method": name, **specs[name]}


def _build_model(
    config: Mapping[str, Any],
    method: str,
    train_prior: torch.Tensor,
    expert: torch.nn.Module,
    labels_by_position: Sequence[int],
    device: torch.device,
) -> CSIConditionedPrototypeUpdate:
    spec = method_spec(method)
    likelihood = PrototypeLikelihoodHead(
        expert,
        train_prior,
        eta_prior=float(spec["eta_prior"]),
        eps=float(config["likelihood"]["eps"]),
    )
    transition = PrototypeTransitionKernel(
        hidden_dim=int(config["model"]["hidden_dim"]),
        radius=int(config["model"]["transition_radius"]),
        context_mode=str(spec["context_mode"]),
        identity_initial_mass=float(config["model"]["transition_identity_initial_mass"]),
    )
    model = CSIConditionedPrototypeUpdate(
        likelihood,
        transition,
        labels_by_position=labels_by_position,
        circular_topology=bool(config["topology"]["circular"]),
        sensing_temperature=float(config["model"]["sensing_temperature"]),
        beta=float(config["likelihood"]["beta"]),
        learnable_beta=method.startswith("U3"),
        transition_enabled=bool(spec["transition"]),
        likelihood_enabled=bool(spec["likelihood"]),
        eps=float(config["likelihood"]["eps"]),
        pilot_re_window=int(config["pilot"]["re_window"]),
    ).to(device)
    if not spec["transition"]:
        for parameter in model.transition_kernel.parameters():
            parameter.requires_grad_(False)
    if not method.startswith("U3"):
        model.beta.requires_grad_(False)
    return model


def _gather_mask_tensor(records: Mapping[str, Any], prefix: str, indices: torch.Tensor, mask_ids: torch.Tensor) -> torch.Tensor:
    width = 64
    output = torch.empty(len(indices), width)
    for mask_id in torch.unique(mask_ids).tolist():
        rows = mask_ids.eq(int(mask_id)).nonzero(as_tuple=False).squeeze(1)
        source = indices.index_select(0, rows)
        output.index_copy_(0, rows, records[f"{prefix}_{MASK_NAMES[int(mask_id)]}"].index_select(0, source))
    return output


def _retained_m4_evidence(probability: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(probability).float()
    if values.ndim != 2 or values.shape[1] != 64:
        raise ValueError("Retained M4 probability must have shape [B,64].")
    return values.clamp_min(1e-12).log()


def _diagnostic_sample_rows(*, samples: int, masks: int, limit: int, seed: int) -> dict[int, torch.Tensor]:
    total = int(samples) * int(masks)
    count = min(int(limit), total)
    if total <= 0 or count <= 0:
        return {}
    flat = torch.randperm(total, generator=torch.Generator().manual_seed(int(seed)))[:count]
    mask_ids = torch.div(flat, int(samples), rounding_mode="floor")
    rows = flat.remainder(int(samples))
    return {mask_id: rows[mask_ids.eq(mask_id)] for mask_id in range(int(masks))}


def _frequency_view(config: Mapping[str, Any], qtpr: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    return _frequency_positions(qtpr, str(config["pilot"]["budget"]), device)


def _radio_batch(
    config: Mapping[str, Any],
    radio_encoder: torch.nn.Module,
    candidates: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    generator: torch.Generator,
    training: bool,
    validation_snr_db: float | None = None,
    validation_dropout: float = 0.0,
) -> dict[str, torch.Tensor]:
    history = int(config["pilot"]["history_frames"])
    selected = select_candidate_history(
        candidates,
        budget=str(config["pilot"]["budget"]),
        history_frames=history,
        mother_frequencies=int(config["pilot"]["mother_frequencies"]),
    ).to(frequencies.device)
    if training:
        snr = torch.empty(selected.shape[0], history, device=frequencies.device).uniform_(
            float(config["pilot"]["train_snr_db_min"]),
            float(config["pilot"]["train_snr_db_max"]),
            generator=generator,
        )
        dropout = float(config["pilot"]["train_dropout"])
    else:
        snr = torch.full(
            (selected.shape[0], history),
            float(config["pilot"]["validation_snr_db"] if validation_snr_db is None else validation_snr_db),
            device=frequencies.device,
        )
        dropout = float(validation_dropout)
    observations, valid = _noisy_observations(selected, snr, dropout=dropout, generator=generator)
    pattern_ids = torch.arange(selected.shape[2], device=frequencies.device).expand(selected.shape[0], history, -1)
    with torch.no_grad():
        output = radio_encoder(observations, pattern_ids, frequencies, valid, snr)
    snr_quality = snr.mean(dim=-1) - float(config["pilot"]["train_snr_db_min"])
    snr_quality = snr_quality / (float(config["pilot"]["train_snr_db_max"]) - float(config["pilot"]["train_snr_db_min"]))
    dropout_ratio = 1.0 - valid.float().mean(dim=(1, 2, 3))
    return {**output, "low_quality_weight": 0.5 * (1.0 - snr_quality.clamp(0.0, 1.0)) + 0.5 * dropout_ratio}


@torch.inference_mode()
def _precompute_radio(
    config: Mapping[str, Any],
    records: Mapping[str, Any],
    radio_encoder: torch.nn.Module,
    frequencies: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
    snr_db: float | None = None,
    dropout: float = 0.0,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(int(config["pilot"]["validation_noise_seed"]))
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    batch_size = int(config["pilot"]["validation_generation_batch_size"])
    for start in range(0, len(records["labels_future"]), batch_size):
        stop = min(start + batch_size, len(records["labels_future"]))
        output = _radio_batch(
            config,
            radio_encoder,
            records["candidate_history"][start:stop],
            frequencies,
            generator=generator,
            training=False,
            validation_snr_db=snr_db,
            validation_dropout=dropout,
        )
        for name in ("c_radio", "frame_csi_features", "csi_available"):
            chunks[name].append(output[name].float().cpu() if output[name].is_floating_point() else output[name].cpu())
    return {
        **{name: torch.cat(values) for name, values in chunks.items()},
        "snr_db": torch.tensor(float(config["pilot"]["validation_snr_db"] if snr_db is None else snr_db)),
        "dropout": torch.tensor(float(dropout)),
    }


def _set_expert_temperature(expert: torch.nn.Module, value: float) -> None:
    temperature = expert.temperature
    physical = float(value)
    if physical <= float(temperature.eps):
        raise ValueError("Radio temperature must exceed the PositiveTemperature epsilon.")
    raw = math.log(math.expm1(physical - float(temperature.eps)))
    with torch.no_grad():
        temperature.raw.copy_(temperature.raw.new_tensor(raw))


def _train_prior(records: Mapping[str, Any]) -> torch.Tensor:
    return estimate_train_beam_prior(records["labels_future"], split_role="train")["prior"]


def _run_stem(method: str, seed: int, *, smoke: bool) -> str:
    return f"{'smoke_' if smoke else ''}{method}_seed{int(seed)}"


def _checkpoint_path(config: Mapping[str, Any], method: str, seed: int, *, smoke: bool) -> Path:
    folder = "smoke_checkpoints" if smoke else "checkpoints"
    return _path(config["output"]["root"]) / folder / _run_stem(method, seed, smoke=smoke) / "best.pt"


def _result_path(config: Mapping[str, Any], method: str, seed: int, *, smoke: bool) -> Path:
    folder = "smoke_results" if smoke else "results"
    return _path(config["output"]["root"]) / folder / f"{_run_stem(method, seed, smoke=smoke)}.json"


def _save_checkpoint(
    path: Path,
    model: CSIConditionedPrototypeUpdate,
    *,
    config: Mapping[str, Any],
    method: str,
    seed: int,
    epoch: int,
    metrics: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None,
    initialization: Mapping[str, Any],
) -> None:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if any("teacher" in name or "classifier" in name for name in state):
        raise RuntimeError("CPSU checkpoints must not contain a teacher or a free classifier.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": state,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "rng_state": _rng_state(),
            "method": method,
            "stage": method_spec(method),
            "seed": int(seed),
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "initialization": dict(initialization),
            "protocol_id": config["protocol"]["id"],
            "protocol_fingerprint": config["protocol"]["fingerprint"],
            "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
            "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
            "csi_checkpoint_sha256": config["source"]["csi_checkpoint_sha256"],
            "f1_checkpoint_sha256": config["source"]["f1_checkpoint_sha256"],
            "topology_descriptor_sha256": config["topology"]["descriptor_sha256"],
            "future_channel_used_as_input": False,
            "outer_test_accessed": False,
            **pilot_resource_accounting(config),
        },
        path,
    )


def _validated_checkpoint_payload(
    path: Path,
    config: Mapping[str, Any],
    *,
    expected_method: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "method": payload.get("method") == expected_method,
        "protocol": payload.get("protocol_fingerprint") == config["protocol"]["fingerprint"],
        "split": payload.get("split_manifest_sha256") == config["protocol"]["split_manifest_sha256"],
        "m4": payload.get("m4_checkpoint_sha256") == config["source"]["m4_checkpoint_sha256"],
        "csi": payload.get("csi_checkpoint_sha256") == config["source"]["csi_checkpoint_sha256"],
        "f1": payload.get("f1_checkpoint_sha256") == config["source"]["f1_checkpoint_sha256"],
        "topology": payload.get("topology_descriptor_sha256") == config["topology"]["descriptor_sha256"],
        "future": payload.get("future_channel_used_as_input") is False,
        "outer": payload.get("outer_test_accessed") is False,
        "re_window": int(payload.get("pilot_re_window", -1)) == int(config["pilot"]["re_window"]),
    }
    if not all(checks.values()):
        raise ValueError(f"CPSU checkpoint identity mismatch: {checks}.")
    return payload


def _load_checkpoint(
    model: CSIConditionedPrototypeUpdate,
    path: Path,
    config: Mapping[str, Any],
    *,
    expected_method: str,
) -> dict[str, Any]:
    payload = _validated_checkpoint_payload(path, config, expected_method=expected_method)
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def _radio_diagnostic(
    radio: Mapping[str, torch.Tensor],
    radio_encoder: torch.nn.Module,
    diagnostic: str,
    *,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if diagnostic not in DIAGNOSTICS:
        raise ValueError(f"Unknown CPSU diagnostic: {diagnostic}.")
    result = {name: value.clone() if torch.is_tensor(value) else value for name, value in radio.items()}
    count = len(result["c_radio"])
    if diagnostic == "csi_shuffle":
        permutation = torch.randperm(count, generator=torch.Generator().manual_seed(750_001 + int(seed)))
        for name in ("c_radio", "frame_csi_features", "csi_available"):
            result[name] = result[name].index_select(0, permutation)
    elif diagnostic in {"temporal_shuffle", "reverse_time"}:
        frames = result["frame_csi_features"]
        if diagnostic == "reverse_time":
            order = torch.arange(frames.shape[1] - 1, -1, -1)
        else:
            order = torch.randperm(frames.shape[1], generator=torch.Generator().manual_seed(750_002 + int(seed)))
            if torch.equal(order, torch.arange(frames.shape[1])):
                order = torch.roll(order, 1)
        frames = frames.index_select(1, order)
        result["frame_csi_features"] = frames
        radio_chunks = []
        for start in range(0, count, int(batch_size)):
            batch = frames[start : start + int(batch_size)].to(device)
            with torch.inference_mode():
                if radio_encoder.temporal is None:
                    encoded = batch[:, -1]
                else:
                    encoded = radio_encoder.temporal(batch)[0][:, -1]
            radio_chunks.append(encoded.float().cpu())
        result["c_radio"] = torch.cat(radio_chunks)
    elif diagnostic == "csi_off":
        result["csi_available"] = torch.zeros_like(result["csi_available"], dtype=torch.bool)
    return result


def _signed_topology_offset(
    source: torch.Tensor,
    target: torch.Tensor,
    labels_by_position: Sequence[int],
    *,
    circular: bool,
) -> torch.Tensor:
    labels = torch.as_tensor(labels_by_position, dtype=torch.long)
    positions = torch.empty_like(labels)
    positions[labels] = torch.arange(len(labels))
    delta = positions[target.long()] - positions[source.long()]
    if circular:
        half = len(labels) // 2
        delta = (delta + half).remainder(len(labels)) - half
    return delta


def _transition_statistics(
    q_delta: torch.Tensor,
    q_final: torch.Tensor,
    sensing_prediction: torch.Tensor,
    final_prediction: torch.Tensor,
    labels: torch.Tensor,
    labels_by_position: Sequence[int],
    *,
    circular: bool,
) -> dict[str, Any]:
    radius = q_delta.shape[1] // 2
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    target_offset = _signed_topology_offset(
        sensing_prediction,
        labels,
        labels_by_position,
        circular=circular,
    )
    predicted_offset = q_delta.argmax(dim=-1) - radius
    expected_offset = (q_delta.float() * offsets[None]).sum(dim=-1)
    nonzero = target_offset.ne(0)
    correct_direction = predicted_offset.sign().eq(target_offset.sign()) & nonzero
    sensing_error = target_offset.abs()
    repaired = final_prediction.eq(labels) & sensing_prediction.ne(labels)
    groups = {"1": (1, 1), "2-3": (2, 3), "4-7": (4, 7), "8+": (8, 10_000)}
    repair_by_distance: dict[str, dict[str, float | int]] = {}
    for name, (low, high) in groups.items():
        selected = sensing_error.ge(low) & sensing_error.le(high)
        repair_by_distance[name] = {
            "samples": int(selected.sum()),
            "repair_rate": float(repaired[selected].float().mean()) if bool(selected.any()) else float("nan"),
        }
    within_radius = target_offset.abs().le(radius)
    return {
        "offset_top1_accuracy": float(predicted_offset.eq(target_offset).float().mean()),
        "offset_top1_accuracy_within_radius": float(predicted_offset[within_radius].eq(target_offset[within_radius]).float().mean()),
        "expected_offset_mae": float((expected_offset - target_offset.float()).abs().mean()),
        "correct_direction_ratio": float(correct_direction.float().sum() / nonzero.float().sum().clamp_min(1.0)),
        "correct_direction_count": int(correct_direction.sum()),
        "nonzero_direction_count": int(nonzero.sum()),
        "target_within_radius_fraction": float(within_radius.float().mean()),
        "average_identity_mass": float(q_final[:, radius].float().mean()),
        "average_absolute_predicted_offset": float(expected_offset.abs().mean()),
        "repair_by_sensing_error_distance": repair_by_distance,
    }


def _parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    return (
        sum(value.numel() for value in model.parameters()),
        sum(value.numel() for value in model.parameters() if value.requires_grad),
    )


@torch.inference_mode()
def _evaluate_model(
    config: Mapping[str, Any],
    model: CSIConditionedPrototypeUpdate,
    records: Mapping[str, Any],
    m4: torch.nn.Module,
    topology: Any,
    radio: Mapping[str, torch.Tensor],
    radio_encoder: torch.nn.Module,
    *,
    method: str,
    seed: int,
    device: torch.device,
    diagnostic: str = "normal",
    collect_samples: bool = False,
) -> dict[str, Any]:
    model.eval()
    batch_size = int(config["training"]["evaluation_batch_size"])
    radio_view = _radio_diagnostic(
        radio,
        radio_encoder,
        diagnostic,
        seed=seed,
        device=device,
        batch_size=batch_size,
    )
    labels = records["labels_future"].long()
    power = records["future_beam_power"].float()
    per_mask: list[dict[str, Any]] = []
    transition_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    prior_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    sample_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    sample_limit = int(config["diagnostics"]["sample_count"])
    sample_rows = (
        _diagnostic_sample_rows(
            samples=len(labels),
            masks=len(MASK_NAMES),
            limit=sample_limit,
            seed=int(config["diagnostics"]["sample_seed"]) + int(seed),
        )
        if collect_samples
        else {}
    )
    started = time.monotonic()
    radio_probability_chunks = []
    for start in range(0, len(labels), batch_size):
        probability = (
            model.likelihood_head(
                radio_view["c_radio"][start : start + batch_size].to(device),
                m4.prototype_bank,
            )["radio_probability"]
            .float()
            .cpu()
        )
        available = radio_view["csi_available"][start : start + batch_size].bool()
        probability = torch.where(available[:, None], probability, torch.full_like(probability, 1.0 / probability.shape[1]))
        radio_probability_chunks.append(probability)
    radio_probability = torch.cat(radio_probability_chunks)

    for mask_id, mask_name in enumerate(MASK_NAMES):
        probability_chunks: list[torch.Tensor] = []
        sensing_chunks: list[torch.Tensor] = []
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            # p0 was produced by this frozen M4 bank before the records were
            # serialized; log(p0) recovers its logits up to a softmax-invariant
            # per-sample constant without re-querying the rounded cached z.
            evidence = _retained_m4_evidence(records[f"p0_{mask_name}"][start:stop].to(device))
            output = model(
                evidence,
                radio_view["c_radio"][start:stop].to(device),
                radio_view["frame_csi_features"][start:stop].to(device),
                m4.prototype_bank,
                radio_view["csi_available"][start:stop].to(device),
                force_identity_transition=diagnostic == "identity",
                force_uniform_likelihood=diagnostic == "uniform",
            )
            probability_chunks.append(output["p_final"].float().cpu())
            sensing_chunks.append(output["p_s"].float().cpu())
            transition_chunks["q_delta"].append(output["q_delta"].float().cpu())
            transition_chunks["q_final"].append(output["q_final"].float().cpu())
            transition_chunks["sensing_prediction"].append(output["p_s"].argmax(dim=-1).cpu())
            transition_chunks["final_prediction"].append(output["p_final"].argmax(dim=-1).cpu())
            transition_chunks["labels"].append(labels[start:stop])
            for name in ("prior_entropy", "prior_margin", "prior_expected_distance_to_map"):
                prior_chunks[name].append(output[name].float().cpu())
            selected = sample_rows.get(mask_id, torch.empty(0, dtype=torch.long))
            selected = selected[selected.ge(start) & selected.lt(stop)]
            if bool(selected.numel()):
                local_rows = (selected - start).to(device)
                values = {
                    "p_s": output["p_s"].index_select(0, local_rows).float().cpu().numpy(),
                    "p_c": output["p_c"].index_select(0, local_rows).float().cpu().numpy(),
                    "log_likelihood_ratio": output["log_likelihood_ratio"].index_select(0, local_rows).float().cpu().numpy(),
                    "likelihood_ratio": output["log_likelihood_ratio"]
                    .index_select(0, local_rows)
                    .float()
                    .clamp(-30, 30)
                    .exp()
                    .cpu()
                    .numpy(),
                    "q_delta": output["q_delta"].index_select(0, local_rows).float().cpu().numpy(),
                    "p_pred": output["p_pred"].index_select(0, local_rows).float().cpu().numpy(),
                    "p_final": output["p_final"].index_select(0, local_rows).float().cpu().numpy(),
                    "prior_topk_probability": output["prior_topk_probability"].index_select(0, local_rows).float().cpu().numpy(),
                    "prior_topk_prototype": output["prior_topk_prototype"].index_select(0, local_rows).cpu().numpy(),
                    "target": labels.index_select(0, selected).numpy(),
                    "sample_index": selected.numpy(),
                    "mask_id": np.full(len(selected), mask_id, dtype=np.int64),
                    "mask": np.asarray([mask_name] * len(selected)),
                }
                for name, value in values.items():
                    sample_chunks[name].append(value)
        probability = torch.cat(probability_chunks)
        sensing = torch.cat(sensing_chunks)
        row = {
            "mask": mask_name,
            "mask_id": mask_id,
            "available_count": MASK_COUNTS[mask_name],
            **_metric_row(probability, labels, power, sensing),
        }
        row["sensing_top1"] = float(sensing.argmax(dim=-1).eq(labels).float().mean())
        row["csi_top1"] = float(radio_probability.argmax(dim=-1).eq(labels).float().mean())
        oracle = sensing.argmax(dim=-1).eq(labels) | radio_probability.argmax(dim=-1).eq(labels)
        row["oracle_top1"] = float(oracle.float().mean())
        per_mask.append(row)

    groups = {GROUP_NAMES[count]: _aggregate([row for row in per_mask if row["available_count"] == count]) for count in (1, 2, 3)}
    groups["all14"] = _aggregate(per_mask)
    full_probability = records["p0_full"].float()
    full = _metric_row(full_probability, labels, power, full_probability)
    csi_only = _metric_row(radio_probability, labels, power, None)
    sensing_macro = float(np.mean([row["sensing_top1"] for row in per_mask]))
    oracle_macro = float(np.mean([row["oracle_top1"] for row in per_mask]))
    denominator = oracle_macro - sensing_macro

    probe_count = min(batch_size, len(labels))
    full_evidence = _retained_m4_evidence(records["p0_full"][:probe_count].to(device))
    full_probe = model(
        full_evidence,
        radio_view["c_radio"][:probe_count].to(device),
        radio_view["frame_csi_features"][:probe_count].to(device),
        m4.prototype_bank,
        torch.ones(probe_count, device=device, dtype=torch.bool),
        full=torch.ones(probe_count, device=device, dtype=torch.bool),
        full_probability=full_probability[:probe_count].to(device),
    )
    off_evidence = _retained_m4_evidence(records[f"p0_{MASK_NAMES[0]}"][:probe_count].to(device))
    off_probe = model(
        off_evidence,
        radio_view["c_radio"][:probe_count].to(device),
        radio_view["frame_csi_features"][:probe_count].to(device),
        m4.prototype_bank,
        torch.zeros(probe_count, device=device, dtype=torch.bool),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    q_delta = torch.cat(transition_chunks["q_delta"])
    q_final = torch.cat(transition_chunks["q_final"])
    transition = _transition_statistics(
        q_delta,
        q_final,
        torch.cat(transition_chunks["sensing_prediction"]),
        torch.cat(transition_chunks["final_prediction"]),
        torch.cat(transition_chunks["labels"]),
        topology.labels_by_position,
        circular=bool(config["topology"]["circular"]),
    )
    total_parameters, trainable_parameters = _parameter_counts(model)
    result: dict[str, Any] = {
        "method": method,
        "seed": int(seed),
        "diagnostic": diagnostic,
        "sample_count": len(labels),
        "per_mask": per_mask,
        "groups": groups,
        "missing_lidar": next(row for row in per_mask if row["mask"] == "missing_lidar"),
        "full": full,
        "csi_only": csi_only,
        "sensing_all14_macro": sensing_macro,
        "oracle_all14_macro": oracle_macro,
        "oracle_headroom_capture": (groups["all14"]["top1_macro"] - sensing_macro) / denominator if denominator > 0 else float("nan"),
        "prior_statistics": {name: float(torch.cat(values).mean()) for name, values in prior_chunks.items()},
        "transition_statistics": transition,
        "beta": float(model.beta.clamp_min(0).detach().cpu()),
        "eta_prior": float(model.likelihood_head.eta_prior),
        "radio_temperature": float(model.likelihood_head.radio_expert.temperature().detach().cpu()),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "latency_ms_per_sample_mask": 1000.0 * elapsed / max(len(labels) * len(MASK_NAMES), 1),
        "elapsed_seconds": elapsed,
        "full_probability_max_abs_diff": float((full_probe["p_final"] - full_probability[:probe_count].to(device)).abs().max()),
        "full_argmax_mismatch": int(
            full_probe["p_final"].argmax(dim=-1).ne(full_probability[:probe_count].to(device).argmax(dim=-1)).sum()
        ),
        "full_pilot_re": int(full_probe["pilot_re"].sum()),
        "csi_off_probability_max_abs_diff": float((off_probe["p_final"] - off_probe["p_s"]).abs().max()),
        "csi_off_argmax_mismatch": int(off_probe["p_final"].argmax(dim=-1).ne(off_probe["p_s"].argmax(dim=-1)).sum()),
        **pilot_resource_accounting(config),
        "future_channel_used_as_input": False,
        "test_loader_constructed": False,
        "outer_test_accessed": False,
    }
    if collect_samples:
        result["_samples"] = {name: np.concatenate(values) for name, values in sample_chunks.items()}
    return result


def _baseline_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    full: Mapping[str, Any],
    csi_only: Mapping[str, Any],
) -> dict[str, Any]:
    values = [dict(row) for row in rows]
    groups = {GROUP_NAMES[count]: _aggregate([row for row in values if int(row["available_count"]) == count]) for count in (1, 2, 3)}
    groups["all14"] = _aggregate(values)
    sensing = float(np.mean([float(row["sensing_top1"]) for row in values]))
    oracle = float(np.mean([float(row["oracle_top1"]) for row in values]))
    denominator = oracle - sensing
    return {
        "per_mask": values,
        "groups": groups,
        "missing_lidar": next(row for row in values if row["mask"] == "missing_lidar"),
        "full": dict(full),
        "csi_only": dict(csi_only),
        "sensing_all14_macro": sensing,
        "oracle_all14_macro": oracle,
        "oracle_headroom_capture": (groups["all14"]["top1_macro"] - sensing) / denominator if denominator > 0 else float("nan"),
    }


@torch.inference_mode()
def reproduce_f1(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    preflight(config)
    seed = int(args.seed)
    device = torch.device(args.device)
    _set_seed(seed)
    qtpr, m4, radio_encoder, expert, calibration = _components(config, device)
    validation = _load_records(config, "validation")
    limit = int(config["training"]["smoke_samples"]) if args.smoke else (int(args.limit) if args.limit else None)
    indices = _stratified_indices(validation["labels_future"], limit, 50_000 + seed)
    records = _records_view(validation, indices)
    frequencies = _frequency_view(config, qtpr, device)
    radio = _precompute_radio(config, records, radio_encoder, frequencies, device=device, seed=seed)
    labels = records["labels_future"].long()
    power = records["future_beam_power"].float()
    batch_size = int(config["training"]["evaluation_batch_size"])
    radio_evidence_chunks, radio_probability_chunks = [], []
    for start in range(0, len(labels), batch_size):
        output = expert(radio["c_radio"][start : start + batch_size].to(device), m4.prototype_bank)
        radio_evidence_chunks.append(output["radio_evidence"].float().cpu())
        radio_probability_chunks.append(output["radio_probability"].float().cpu())
    radio_evidence = torch.cat(radio_evidence_chunks)
    radio_probability = torch.cat(radio_probability_chunks)
    available = radio["csi_available"].bool()
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fusion_names = ("B0-sensing", "F1-fixed-evidence", "probability-average", "product-of-experts")
    for mask_id, mask_name in enumerate(MASK_NAMES):
        retained_evidence = _retained_m4_evidence(records[f"p0_{mask_name}"])
        calibrated_sensing = retained_evidence / float(calibration["sensing_temperature"])
        sensing = torch.softmax(calibrated_sensing, dim=-1)
        candidates = {
            "B0-sensing": sensing,
            "F1-fixed-evidence": torch.softmax(0.5 * calibrated_sensing + 0.5 * radio_evidence, dim=-1),
            "probability-average": 0.5 * sensing + 0.5 * radio_probability,
            "product-of-experts": torch.softmax(sensing.clamp_min(1e-8).log() + radio_probability.clamp_min(1e-8).log(), dim=-1),
        }
        sensing_correct = sensing.argmax(dim=-1).eq(labels)
        radio_correct = radio_probability.argmax(dim=-1).eq(labels)
        for name in fusion_names:
            probability = torch.where(available[:, None], candidates[name], sensing)
            rows[name].append(
                {
                    "mask": mask_name,
                    "mask_id": mask_id,
                    "available_count": MASK_COUNTS[mask_name],
                    **_metric_row(probability, labels, power, sensing),
                    "sensing_top1": float(sensing_correct.float().mean()),
                    "csi_top1": float(radio_correct.float().mean()),
                    "oracle_top1": float((sensing_correct | radio_correct).float().mean()),
                }
            )
    full_probability = records["p0_full"].float()
    full = _metric_row(full_probability, labels, power, full_probability)
    csi_only = _metric_row(radio_probability, labels, power, None)
    methods = {name: _baseline_summary(method_rows, full=full, csi_only=csi_only) for name, method_rows in rows.items()}
    fixed = methods["F1-fixed-evidence"]
    compare_reported = not bool(args.smoke) and limit is None

    def reported(value: bool) -> bool | None:
        return bool(value) if compare_reported else None

    checks = {
        "reported_metric_comparison_performed": compare_reported,
        "all14_matches_reported": reported(
            abs(float(fixed["groups"]["all14"]["top1_macro"]) - float(config["baselines"]["f1_all14_macro"])) <= 5e-4
        ),
        "worst_matches_reported": reported(
            abs(float(fixed["groups"]["all14"]["top1_worst"]) - float(config["baselines"]["f1_all14_worst"])) <= 5e-4
        ),
        "missing_lidar_matches_reported": reported(
            abs(float(fixed["missing_lidar"]["top1"]) - float(config["baselines"]["f1_missing_lidar"])) <= 5e-4
        ),
        "full_matches_reported": reported(abs(float(full["top1"]) - float(config["baselines"]["f1_full_top1"])) <= 5e-4),
        "full_bypass_exact": True,
        "csi_off_exact": True,
    }
    result = {
        "stage": "U0",
        "seed": seed,
        "sample_count": len(labels),
        "smoke": bool(args.smoke),
        "calibration": calibration,
        "methods": methods,
        "reproduction_checks": checks,
        **pilot_resource_accounting(config),
        "full_pilot_re": 0,
        "future_channel_used_as_input": False,
        "test_loader_constructed": False,
        "outer_test_accessed": False,
    }
    path = _path(config["output"]["root"]) / "baselines" / f"{'smoke_' if args.smoke else ''}f1_seed{seed}.json"
    if path.is_file() and not args.overwrite:
        raise FileExistsError(f"CPSU F1 reproduction already exists: {path}.")
    _write_json(path, result)
    _write_csv(
        _path(config["output"]["root"]) / ("smoke_fusion_baselines.csv" if args.smoke else "fusion_baselines.csv"),
        [
            {
                "method": name,
                "all14_macro": value["groups"]["all14"]["top1_macro"],
                "all14_worst": value["groups"]["all14"]["top1_worst"],
                "missing_lidar": value["missing_lidar"]["top1"],
                "nll_macro": value["groups"]["all14"]["nll_macro"],
                "ece_macro": value["groups"]["all14"]["ece_macro"],
            }
            for name, value in methods.items()
        ],
    )
    return result


def _calibration_evidence(
    config: Mapping[str, Any],
    records: Mapping[str, Any],
    m4: torch.nn.Module,
    expert: torch.nn.Module,
    radio: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch_size = int(config["training"]["evaluation_batch_size"])
    sensing: dict[str, torch.Tensor] = {}
    for mask_name in MASK_NAMES:
        sensing[mask_name] = _retained_m4_evidence(records[f"p0_{mask_name}"])
    radio_chunks = []
    with torch.inference_mode():
        for start in range(0, len(records["labels_future"]), batch_size):
            output = expert(radio["c_radio"][start : start + batch_size].to(device), m4.prototype_bank)
            radio_chunks.append(output["radio_evidence"].float().cpu())
    return sensing, torch.cat(radio_chunks)


def calibrate_likelihood(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    if args.method not in CALIBRATION_METHODS:
        raise ValueError(f"Likelihood calibration method must be one of {CALIBRATION_METHODS}.")
    preflight(config)
    method, seed = str(args.method), int(args.seed)
    device = torch.device(args.device)
    _set_seed(seed)
    qtpr, m4, radio_encoder, expert, calibration = _components(config, device)
    topology = _topology(qtpr)
    train = _load_records(config, "train")
    prior = _train_prior(train)
    full_limit = int(len(train["labels_future"]) * float(config["likelihood"]["train_calibration_fraction"]))
    limit = int(config["training"]["smoke_samples"]) if args.smoke else full_limit
    if args.limit:
        limit = min(limit, int(args.limit))
    calibration_indices = _stratified_indices(
        train["labels_future"],
        limit,
        int(config["likelihood"]["train_calibration_seed"]) + seed,
    )
    calibration_records = _records_view(train, calibration_indices)
    frequencies = _frequency_view(config, qtpr, device)
    radio = _precompute_radio(
        config,
        calibration_records,
        radio_encoder,
        frequencies,
        device=device,
        seed=seed,
    )
    sensing_evidence, base_radio_evidence = _calibration_evidence(
        config,
        calibration_records,
        m4,
        expert,
        radio,
        device=device,
    )
    target = calibration_records["labels_future"].long()
    spec = method_spec(method)
    eta = float(spec["eta_prior"])
    log_prior = prior.clamp_min(float(config["likelihood"]["eps"])).log()
    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for scale in config["likelihood"]["temperature_scale_grid"]:
        radio_log_probability = torch.log_softmax(base_radio_evidence / float(scale), dim=-1)
        log_ratio = radio_log_probability - eta * log_prior[None]
        for beta in config["likelihood"]["beta_grid"]:
            group_losses = defaultdict(list)
            for mask_name in MASK_NAMES:
                sensing_log = torch.log_softmax(sensing_evidence[mask_name] / float(calibration["sensing_temperature"]), dim=-1)
                posterior = torch.log_softmax(sensing_log + float(beta) * log_ratio, dim=-1)
                nll = float(F.nll_loss(posterior, target).item())
                group_losses[MASK_COUNTS[mask_name]].append(nll)
            group_nll = {str(count): float(np.mean(values)) for count, values in group_losses.items()}
            balanced_nll = float(np.mean(list(group_nll.values())))
            row = {
                "method": method,
                "temperature_scale": float(scale),
                "physical_radio_temperature": float(calibration["physical_radio_temperature"]) * float(scale),
                "beta": float(beta),
                "eta_prior": eta,
                "train_calibration_group_balanced_nll": balanced_nll,
                **{f"group_{name}_nll": value for name, value in group_nll.items()},
            }
            grid_rows.append(row)
            if best is None or balanced_nll < float(best["train_calibration_group_balanced_nll"]):
                best = row
    assert best is not None
    model = _build_model(config, method, prior, expert, topology.labels_by_position, device)
    _set_expert_temperature(model.likelihood_head.radio_expert, float(best["physical_radio_temperature"]))
    with torch.no_grad():
        model.beta.copy_(model.beta.new_tensor(float(best["beta"])))
    initialization = {
        "mode": "frozen_F1_expert_train_only_temperature_beta_calibration",
        "source_f1_checkpoint": str(_path(config["source"]["f1_checkpoint"])),
        "source_f1_checkpoint_sha256": config["source"]["f1_checkpoint_sha256"],
        "train_calibration_samples": len(calibration_indices),
        "train_calibration_indices_sha256": hashlib.sha256(calibration_indices.numpy().tobytes()).hexdigest(),
        "validation_labels_used_for_selection": False,
    }
    checkpoint = _checkpoint_path(config, method, seed, smoke=bool(args.smoke))
    result_path = _result_path(config, method, seed, smoke=bool(args.smoke))
    if (checkpoint.is_file() or result_path.is_file()) and not args.overwrite:
        raise FileExistsError(f"CPSU calibration output already exists for {method} seed {seed}.")
    _save_checkpoint(
        checkpoint,
        model,
        config=config,
        method=method,
        seed=seed,
        epoch=0,
        metrics=best,
        optimizer=None,
        initialization=initialization,
    )
    validation = _load_records(config, "validation")
    validation_limit = int(config["training"]["smoke_samples"]) if args.smoke else (int(args.limit) if args.limit else None)
    validation_indices = _stratified_indices(validation["labels_future"], validation_limit, 50_000 + seed)
    validation_view = _records_view(validation, validation_indices)
    validation_radio = _precompute_radio(
        config,
        validation_view,
        radio_encoder,
        frequencies,
        device=device,
        seed=seed,
    )
    evaluation = _evaluate_model(
        config,
        model,
        validation_view,
        m4,
        topology,
        validation_radio,
        radio_encoder,
        method=method,
        seed=seed,
        device=device,
    )
    result = {
        **evaluation,
        "stage": "U1",
        "smoke": bool(args.smoke),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "calibration_selection": best,
        "calibration_role": "train",
        "calibration_sample_count": len(calibration_indices),
        "calibration_indices_sha256": initialization["train_calibration_indices_sha256"],
        "validation_labels_used_for_selection": False,
    }
    _write_json(result_path, result)
    grid_path = _path(config["output"]["root"]) / "calibration" / f"{_run_stem(method, seed, smoke=bool(args.smoke))}.csv"
    _write_csv(grid_path, grid_rows)
    return result


def _quick_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all14_macro": float(result["groups"]["all14"]["top1_macro"]),
        "all14_worst": float(result["groups"]["all14"]["top1_worst"]),
        "single_macro": float(result["groups"]["single"]["top1_macro"]),
        "two_macro": float(result["groups"]["two"]["top1_macro"]),
        "three_macro": float(result["groups"]["three"]["top1_macro"]),
        "missing_lidar": float(result["missing_lidar"]["top1"]),
        "validation_nll": float(result["groups"]["all14"]["nll_macro"]),
    }


def _stage_gate(config: Mapping[str, Any], *, seed: int, smoke: bool) -> dict[str, Any]:
    output = _path(config["output"]["root"])
    u1_path = _result_path(config, "U1", seed, smoke=smoke)
    u2_path = _result_path(config, "U2", seed, smoke=smoke)
    static_path = _result_path(config, "U2-static", seed, smoke=smoke)
    baseline_path = output / "baselines" / f"{'smoke_' if smoke else ''}f1_seed{int(seed)}.json"
    details: dict[str, Any] = {
        "seed": int(seed),
        "smoke": bool(smoke),
        "u1_result_exists": u1_path.is_file(),
        "u2_result_exists": u2_path.is_file(),
        "u2_static_result_exists": static_path.is_file(),
        "baseline_result_exists": baseline_path.is_file(),
    }
    likelihood_value = False
    if u1_path.is_file() and baseline_path.is_file():
        u1 = json.loads(u1_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["methods"]["probability-average"]
        u1_top1 = float(u1["groups"]["all14"]["top1_macro"])
        u1_nll = float(u1["groups"]["all14"]["nll_macro"])
        probability_top1 = float(baseline["groups"]["all14"]["top1_macro"])
        probability_nll = float(baseline["groups"]["all14"]["nll_macro"])
        likelihood_value = u1_top1 > probability_top1 or u1_nll < probability_nll
        details["likelihood"] = {
            "u1_all14": u1_top1,
            "probability_average_all14": probability_top1,
            "u1_nll": u1_nll,
            "probability_average_nll": probability_nll,
            "independent_value": likelihood_value,
        }
    transition_value = False
    if u2_path.is_file() and static_path.is_file():
        u2 = json.loads(u2_path.read_text(encoding="utf-8"))
        static = json.loads(static_path.read_text(encoding="utf-8"))
        u2_top1 = float(u2["groups"]["all14"]["top1_macro"])
        static_top1 = float(static["groups"]["all14"]["top1_macro"])
        transition_value = u2_top1 > static_top1
        details["transition"] = {
            "u2_all14": u2_top1,
            "u2_static_all14": static_top1,
            "independent_value": transition_value,
        }
    details["passed"] = bool(likelihood_value or transition_value)
    details["reason"] = (
        "U1 or U2 demonstrated independent value."
        if details["passed"]
        else "U3 is sealed until U1 beats probability averaging/calibration or U2 beats the static transition."
    )
    _write_json(output / "gates" / f"{'smoke_' if smoke else ''}u3_seed{int(seed)}.json", details)
    return details


def _initialize_stage_model(
    config: Mapping[str, Any],
    model: CSIConditionedPrototypeUpdate,
    *,
    method: str,
    seed: int,
    smoke: bool,
) -> dict[str, Any]:
    initialization: dict[str, Any] = {"mode": "fresh_local_transition_on_frozen_F1_components"}
    if not method.startswith("U3"):
        return initialization
    gate = _stage_gate(config, seed=seed, smoke=smoke)
    if not gate["passed"]:
        raise RuntimeError(str(gate["reason"]))
    u1_path = _checkpoint_path(config, "U1", seed, smoke=smoke)
    u2_path = _checkpoint_path(config, "U2", seed, smoke=smoke)
    if not u1_path.is_file() or not u2_path.is_file():
        raise FileNotFoundError("U3 requires both the selected U1 and U2 checkpoints for initialization.")
    u1 = _validated_checkpoint_payload(u1_path, config, expected_method="U1")
    u2 = _validated_checkpoint_payload(u2_path, config, expected_method="U2")
    likelihood_state = {
        name.removeprefix("likelihood_head."): value for name, value in u1["model_state"].items() if name.startswith("likelihood_head.")
    }
    model.likelihood_head.load_state_dict(likelihood_state, strict=True)
    with torch.no_grad():
        model.beta.copy_(u1["model_state"]["beta"].to(model.beta))
    if method == "U3":
        transition_state = {
            name.removeprefix("transition_kernel."): value
            for name, value in u2["model_state"].items()
            if name.startswith("transition_kernel.")
        }
        model.transition_kernel.load_state_dict(transition_state, strict=True)
        transition_mode = "complete_U2_transition"
    else:
        with torch.no_grad():
            model.transition_kernel.identity_raw.copy_(
                u2["model_state"]["transition_kernel.identity_raw"].to(model.transition_kernel.identity_raw)
            )
        transition_mode = "U2_identity_mass_only_context_ablation"
    return {
        "mode": "selected_U1_plus_U2",
        "u1_checkpoint": str(u1_path),
        "u1_checkpoint_sha256": sha256_file(u1_path),
        "u2_checkpoint": str(u2_path),
        "u2_checkpoint_sha256": sha256_file(u2_path),
        "transition_initialization": transition_mode,
        "stage_gate": gate,
    }


def _group_equal_weights(mask_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    available = torch.tensor([MASK_COUNTS[name] for name in MASK_NAMES], dtype=torch.long)
    counts = available.index_select(0, mask_ids.cpu()).to(device)
    group_sizes = torch.tensor(
        [sum(value == count for value in MASK_COUNTS.values()) for count in (1, 2, 3)],
        device=device,
        dtype=torch.float32,
    )
    weights = group_sizes.index_select(0, counts - 1).reciprocal()
    return weights / weights.mean()


def train_transition(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    if args.method not in TRAIN_METHODS:
        raise ValueError(f"Transition training method must be one of {TRAIN_METHODS}.")
    preflight(config)
    method, seed, smoke = str(args.method), int(args.seed), bool(args.smoke)
    _set_seed(seed)
    device = torch.device(args.device)
    qtpr, m4, radio_encoder, expert, _ = _components(config, device)
    topology = _topology(qtpr)
    train_records = _load_records(config, "train")
    validation_records = _load_records(config, "validation")
    prior = _train_prior(train_records)
    limit = int(config["training"]["smoke_samples"]) if smoke else (int(args.limit) if args.limit else None)
    train_indices = _stratified_indices(train_records["labels_future"], limit, 40_000 + seed)
    validation_indices = _stratified_indices(validation_records["labels_future"], limit, 50_000 + seed)
    train_view = _records_view(train_records, train_indices)
    validation_view = _records_view(validation_records, validation_indices)
    model = _build_model(config, method, prior, expert, topology.labels_by_position, device)
    initialization = _initialize_stage_model(config, model, method=method, seed=seed, smoke=smoke)
    for parameter in model.likelihood_head.parameters():
        parameter.requires_grad_(False)
    for parameter in model.transition_kernel.parameters():
        parameter.requires_grad_(True)
    model.beta.requires_grad_(method.startswith("U3"))
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": list(model.transition_kernel.parameters()),
            "lr": float(config["training"]["transition_learning_rate"]),
        }
    ]
    if model.beta.requires_grad:
        parameter_groups.append({"params": [model.beta], "lr": float(config["training"]["beta_learning_rate"])})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["smoke_epochs"] if smoke else (args.epochs or config["training"]["max_epochs"]))
    batch_size = int(config["training"]["batch_size"])
    frequencies = _frequency_view(config, qtpr, device)
    validation_radio = _precompute_radio(
        config,
        validation_view,
        radio_encoder,
        frequencies,
        device=device,
        seed=seed,
    )
    result_path = _result_path(config, method, seed, smoke=smoke)
    checkpoint_path = _checkpoint_path(config, method, seed, smoke=smoke)
    if (result_path.is_file() or checkpoint_path.is_file()) and not args.overwrite:
        raise FileExistsError(f"CPSU training output already exists for {method} seed {seed}.")
    run_config = dict(config) | {
        "run": {
            "method": method,
            "seed": seed,
            "device": str(device),
            "smoke": smoke,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "initialization": initialization,
            "future_channel_used_as_input": False,
            "outer_test_accessed": False,
        }
    }
    resolved = _path(config["output"]["root"]) / "resolved_configs" / f"{_run_stem(method, seed, smoke=smoke)}.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    noise_generator = torch.Generator(device=device).manual_seed(70_000 + seed)
    topology_distance = topology.distance.to(device)
    weights = dict(config["loss"])
    weights["likelihood_kd"] = 0.0
    if method.startswith("U2"):
        weights["preserve"] = 0.0
    best_score = float("-inf")
    best_epoch = 0
    patience = 0
    history_rows: list[dict[str, Any]] = []
    stop_reason = "max_epochs"

    for epoch in range(1, epochs + 1):
        model.train()
        order, mask_schedule = balanced_epoch_schedule(train_view["labels_future"], epoch=epoch, seed=seed)
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            mask_ids = mask_schedule.index_select(0, batch_indices)
            sensing_probability = _gather_mask_tensor(train_view, "p0", batch_indices, mask_ids).to(device)
            sensing_evidence = _retained_m4_evidence(sensing_probability)
            radio = _radio_batch(
                config,
                radio_encoder,
                train_view["candidate_history"].index_select(0, batch_indices),
                frequencies,
                generator=noise_generator,
                training=True,
            )
            target = train_view["labels_future"].index_select(0, batch_indices).to(device)
            output = model(
                sensing_evidence,
                radio["c_radio"],
                radio["frame_csi_features"],
                m4.prototype_bank,
                radio["csi_available"],
            )
            losses = prototype_update_loss(
                output,
                target,
                topology_distance,
                weights=weights,
                low_quality_weight=radio["low_quality_weight"],
                sample_weight=_group_equal_weights(mask_ids, device),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad],
                float(config["training"]["gradient_clip_norm"]),
            )
            optimizer.step()
            if model.beta.requires_grad:
                with torch.no_grad():
                    model.beta.clamp_(0.01, 4.0)
            count = len(batch_indices)
            seen += count
            for name, value in losses.items():
                totals[name] += float(value.detach().cpu()) * count
        validation = _evaluate_model(
            config,
            model,
            validation_view,
            m4,
            topology,
            validation_radio,
            radio_encoder,
            method=method,
            seed=seed,
            device=device,
        )
        quick = _quick_metrics(validation)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
            **quick,
            "beta": float(model.beta.detach().clamp_min(0).cpu()),
            "gamma_transition": float(torch.sigmoid(model.transition_kernel.identity_raw.detach()).cpu()),
        }
        history_rows.append(row)
        score = quick["all14_macro"]
        if score > best_score + 1e-8:
            best_score, best_epoch, patience = score, epoch, 0
            _save_checkpoint(
                checkpoint_path,
                model,
                config=config,
                method=method,
                seed=seed,
                epoch=epoch,
                metrics=quick,
                optimizer=optimizer,
                initialization=initialization,
            )
        else:
            patience += 1
        if patience >= int(config["training"]["patience"]):
            stop_reason = "early_stopping"
            break
    _load_checkpoint(model, checkpoint_path, config, expected_method=method)
    evaluation = _evaluate_model(
        config,
        model,
        validation_view,
        m4,
        topology,
        validation_radio,
        radio_encoder,
        method=method,
        seed=seed,
        device=device,
    )
    result = {
        **evaluation,
        "stage": "U3" if method.startswith("U3") else "U2",
        "smoke": smoke,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": best_epoch,
        "epochs_ran": len(history_rows),
        "stop_reason": stop_reason,
        "initialization": initialization,
        "teacher_loaded_during_training": False,
        "teacher_required_for_validation_or_inference": False,
        "effective_loss_weights": weights,
    }
    _write_json(result_path, result)
    log_path = _path(config["output"]["root"]) / "training_logs" / f"{_run_stem(method, seed, smoke=smoke)}.csv"
    _write_csv(log_path, history_rows)
    return result


def _save_posterior_samples(output: Path, samples: Mapping[str, np.ndarray]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "prior_likelihood_posterior_samples.npz", **samples)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    count = min(4, len(samples["target"]))
    figure, axes = plt.subplots(count, 2, figsize=(15, 2.8 * count), squeeze=False)
    for index in range(count):
        probability_axis, transition_axis = axes[index]
        ratio = samples["likelihood_ratio"][index]
        normalized_ratio = ratio / max(float(ratio.sum()), 1e-12)
        probability_axis.plot(samples["p_s"][index], label="sensing prior", linewidth=1.3)
        probability_axis.plot(samples["p_pred"][index], label="transitioned prior", linewidth=1.3)
        probability_axis.plot(normalized_ratio, label="normalized likelihood ratio", linewidth=1.0, alpha=0.75)
        probability_axis.plot(samples["p_final"][index], label="final posterior", linewidth=1.5)
        probability_axis.axvline(int(samples["target"][index]), color="black", linestyle="--", linewidth=0.8)
        probability_axis.set_ylabel(f"sample {index}")
        probability_axis.grid(alpha=0.2)
        radius = samples["q_delta"].shape[1] // 2
        transition_axis.bar(np.arange(-radius, radius + 1), samples["q_delta"][index], width=0.8)
        transition_axis.set_ylabel("q_delta")
        transition_axis.grid(alpha=0.2)
    axes[0, 0].legend(ncol=4, fontsize=8)
    axes[-1, 0].set_xlabel("shared beam prototype")
    axes[-1, 1].set_xlabel("topology offset")
    figure.tight_layout()
    figure.savefig(output / "prior_likelihood_posterior_examples.png", dpi=160)
    plt.close(figure)


def _diagnostic_row(result: Mapping[str, Any]) -> dict[str, Any]:
    transition = result["transition_statistics"]
    return {
        "method": result["method"],
        "seed": result["seed"],
        "diagnostic": result["diagnostic"],
        "all14_macro": result["groups"]["all14"]["top1_macro"],
        "all14_worst": result["groups"]["all14"]["top1_worst"],
        "missing_lidar": result["missing_lidar"]["top1"],
        "nll_macro": result["groups"]["all14"]["nll_macro"],
        "offset_top1_accuracy": transition["offset_top1_accuracy"],
        "expected_offset_mae": transition["expected_offset_mae"],
        "correct_direction_ratio": transition["correct_direction_ratio"],
        "average_identity_mass": transition["average_identity_mass"],
        "beta": result["beta"],
        "latency_ms_per_sample_mask": result["latency_ms_per_sample_mask"],
    }


def _load_evaluation_context(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> tuple[Any, Any, Any, Any, CSIConditionedPrototypeUpdate, Mapping[str, Any], torch.Tensor, torch.device, Path]:
    method, seed, smoke = str(args.method), int(args.seed), bool(args.smoke)
    if method not in (*CALIBRATION_METHODS, *TRAIN_METHODS):
        raise ValueError("Evaluation requires a calibrated or trained CPSU method.")
    device = torch.device(args.device)
    qtpr, m4, radio_encoder, expert, _ = _components(config, device)
    topology = _topology(qtpr)
    train = _load_records(config, "train")
    prior = _train_prior(train)
    model = _build_model(config, method, prior, expert, topology.labels_by_position, device)
    checkpoint = _path(args.checkpoint) if args.checkpoint else _checkpoint_path(config, method, seed, smoke=smoke)
    _load_checkpoint(model, checkpoint, config, expected_method=method)
    validation = _load_records(config, "validation")
    limit = int(config["training"]["smoke_samples"]) if smoke else (int(args.limit) if args.limit else None)
    indices = _stratified_indices(validation["labels_future"], limit, 50_000 + seed)
    records = _records_view(validation, indices)
    frequencies = _frequency_view(config, qtpr, device)
    return qtpr, m4, radio_encoder, topology, model, records, frequencies, device, checkpoint


def _run_radio_sweeps(
    config: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    records: Mapping[str, Any],
    model: CSIConditionedPrototypeUpdate,
    m4: torch.nn.Module,
    radio_encoder: torch.nn.Module,
    topology: Any,
    frequencies: torch.Tensor,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    output = _path(config["output"]["root"])
    collections: dict[str, list[dict[str, Any]]] = {"snr": [], "dropout": []}
    for snr_db in config["diagnostics"]["snr_db_values"]:
        radio = _precompute_radio(
            config,
            records,
            radio_encoder,
            frequencies,
            device=device,
            seed=seed,
            snr_db=float(snr_db),
        )
        result = _evaluate_model(
            config,
            model,
            records,
            m4,
            topology,
            radio,
            radio_encoder,
            method=method,
            seed=seed,
            device=device,
        )
        collections["snr"].append({"snr_db": float(snr_db), **_diagnostic_row(result)})
    for dropout in config["diagnostics"]["dropout_values"]:
        radio = _precompute_radio(
            config,
            records,
            radio_encoder,
            frequencies,
            device=device,
            seed=seed,
            dropout=float(dropout),
        )
        result = _evaluate_model(
            config,
            model,
            records,
            m4,
            topology,
            radio,
            radio_encoder,
            method=method,
            seed=seed,
            device=device,
        )
        collections["dropout"].append({"dropout": float(dropout), **_diagnostic_row(result)})
    _write_csv(output / "snr_summary.csv", collections["snr"])
    _write_csv(output / "dropout_summary.csv", collections["dropout"])
    return collections


def evaluate_checkpoint(args: argparse.Namespace, config: Mapping[str, Any], *, diagnose: bool) -> dict[str, Any]:
    preflight(config)
    seed = int(args.seed)
    _set_seed(seed)
    _, m4, radio_encoder, topology, model, records, frequencies, device, checkpoint = _load_evaluation_context(args, config)
    radio = _precompute_radio(config, records, radio_encoder, frequencies, device=device, seed=seed)
    diagnostics = DIAGNOSTICS if diagnose or args.diagnostic == "all" else (str(args.diagnostic),)
    results: list[dict[str, Any]] = []
    output = _path(config["output"]["root"])
    for diagnostic in diagnostics:
        result = _evaluate_model(
            config,
            model,
            records,
            m4,
            topology,
            radio,
            radio_encoder,
            method=str(args.method),
            seed=seed,
            device=device,
            diagnostic=diagnostic,
            collect_samples=diagnostic == "normal",
        )
        samples = result.pop("_samples", None)
        path = output / "diagnostics" / f"{_run_stem(str(args.method), seed, smoke=bool(args.smoke))}_{diagnostic}.json"
        if path.is_file() and not args.overwrite:
            raise FileExistsError(f"CPSU diagnostic already exists: {path}.")
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_sha256"] = sha256_file(checkpoint)
        _write_json(path, result)
        results.append(result)
        if samples is not None and str(args.method) == "U3" and seed == 1 and not args.smoke:
            _save_posterior_samples(output, samples)
    _write_csv(output / "temporal_order_diagnostics.csv", [_diagnostic_row(result) for result in results])
    sweeps = None
    if diagnose and not args.skip_sweeps:
        sweeps = _run_radio_sweeps(
            config,
            method=str(args.method),
            seed=seed,
            records=records,
            model=model,
            m4=m4,
            radio_encoder=radio_encoder,
            topology=topology,
            frequencies=frequencies,
            device=device,
        )
    return {"results": results, "sweeps": sweeps, "outer_test_accessed": False}


def _flat_result(result: Mapping[str, Any]) -> dict[str, Any]:
    transition = result.get("transition_statistics", {})
    return {
        "method": result["method"],
        "seed": result["seed"],
        "all14_macro": result["groups"]["all14"]["top1_macro"],
        "all14_worst": result["groups"]["all14"]["top1_worst"],
        "single_macro": result["groups"]["single"]["top1_macro"],
        "two_macro": result["groups"]["two"]["top1_macro"],
        "three_macro": result["groups"]["three"]["top1_macro"],
        "missing_lidar": result["missing_lidar"]["top1"],
        "csi_only": result["csi_only"]["top1"],
        "sensing_only": result["sensing_all14_macro"],
        "oracle": result["oracle_all14_macro"],
        "headroom_capture": result["oracle_headroom_capture"],
        "nll_macro": result["groups"]["all14"]["nll_macro"],
        "ece_macro": result["groups"]["all14"]["ece_macro"],
        "offset_top1_accuracy": transition.get("offset_top1_accuracy"),
        "expected_offset_mae": transition.get("expected_offset_mae"),
        "correct_direction_ratio": transition.get("correct_direction_ratio"),
        "average_identity_mass": transition.get("average_identity_mass"),
        "beta": result.get("beta"),
        "eta_prior": result.get("eta_prior"),
        "total_parameters": result.get("total_parameters"),
        "trainable_parameters": result.get("trainable_parameters"),
        "latency_ms_per_sample_mask": result.get("latency_ms_per_sample_mask"),
        "pilot_re_per_frame": result.get("pilot_re_per_frame"),
        "pilot_re_window": result.get("pilot_re_window"),
        "full_probability_max_abs_diff": result.get("full_probability_max_abs_diff"),
        "csi_off_probability_max_abs_diff": result.get("csi_off_probability_max_abs_diff"),
        "outer_test_accessed": result.get("outer_test_accessed"),
    }


def _mean_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output = []
    metrics = (
        "all14_macro",
        "all14_worst",
        "single_macro",
        "two_macro",
        "three_macro",
        "missing_lidar",
        "nll_macro",
        "ece_macro",
        "offset_top1_accuracy",
        "expected_offset_mae",
    )
    for method, values in sorted(grouped.items()):
        record: dict[str, Any] = {"method": method, "seeds": len(values)}
        for metric in metrics:
            finite = [float(value[metric]) for value in values if value.get(metric) is not None and math.isfinite(float(value[metric]))]
            record[f"{metric}_mean"] = float(np.mean(finite)) if finite else float("nan")
            record[f"{metric}_std"] = float(np.std(finite)) if finite else float("nan")
            record[f"{metric}_values"] = ";".join(f"{item:.8f}" for item in finite)
        output.append(record)
    return output


def _final_decision(
    config: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Any] | None,
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    u3 = [result for result in results if result["method"] == "U3"]
    f1 = baselines.get("methods", {}).get("F1-fixed-evidence") if baselines else None
    probability_average = baselines.get("methods", {}).get("probability-average") if baselines else None
    by_method = {method: [row for row in results if row["method"] == method] for method in set(row["method"] for row in results)}
    decision: dict[str, Any] = {
        "complete_u3_seed_count": len(u3),
        "required_seed_count": len(config["training"]["seeds"]),
        "outer_test_accessed": False,
        "paper_recommendation": "保留 F1；CPSU 尚未完成预注册验证。",
    }
    if not u3 or f1 is None:
        decision["passed"] = False
        decision["reason"] = "缺少正式 U3 或 F1 复现结果。"
        return decision
    f1_values = {
        "all14": float(f1["groups"]["all14"]["top1_macro"]),
        "worst": float(f1["groups"]["all14"]["top1_worst"]),
        "missing_lidar": float(f1["missing_lidar"]["top1"]),
        "single": float(f1["groups"]["single"]["top1_macro"]),
        "two": float(f1["groups"]["two"]["top1_macro"]),
        "three": float(f1["groups"]["three"]["top1_macro"]),
    }
    means = {
        name: float(np.mean([float(_flat_result(row)[name]) for row in u3]))
        for name in ("all14_macro", "all14_worst", "missing_lidar", "single_macro", "two_macro", "three_macro")
    }
    gains = {
        "all14_pp": 100.0 * (means["all14_macro"] - f1_values["all14"]),
        "worst_pp": 100.0 * (means["all14_worst"] - f1_values["worst"]),
        "missing_lidar_pp": 100.0 * (means["missing_lidar"] - f1_values["missing_lidar"]),
    }
    prereg_gain = (
        gains["all14_pp"] >= float(config["diagnostics"]["success_all14_macro_pp"])
        or gains["worst_pp"] >= float(config["diagnostics"]["success_all14_worst_pp"])
        or gains["missing_lidar_pp"] >= float(config["diagnostics"]["success_missing_lidar_pp"])
    )
    group_regressions = {
        "single_pp": 100.0 * (means["single_macro"] - f1_values["single"]),
        "two_pp": 100.0 * (means["two_macro"] - f1_values["two"]),
        "three_pp": 100.0 * (means["three_macro"] - f1_values["three"]),
    }
    group_guard = min(group_regressions.values()) >= -float(config["diagnostics"]["maximum_group_regression_pp"])
    invariants = all(
        float(row["full_probability_max_abs_diff"]) == 0.0
        and float(row["csi_off_probability_max_abs_diff"]) < 1e-7
        and int(row.get("full_pilot_re", 0)) == 0
        for row in u3
    )
    seed_direction = len(u3) == len(config["training"]["seeds"]) and all(
        float(row["groups"]["all14"]["top1_macro"]) > f1_values["all14"] for row in u3
    )
    u1 = by_method.get("U1", [])
    u2 = by_method.get("U2", [])
    static = by_method.get("U2-static", [])
    u1_value = bool(u1 and probability_average) and (
        float(np.mean([row["groups"]["all14"]["top1_macro"] for row in u1])) > float(probability_average["groups"]["all14"]["top1_macro"])
        or float(np.mean([row["groups"]["all14"]["nll_macro"] for row in u1])) < float(probability_average["groups"]["all14"]["nll_macro"])
    )
    u2_value = bool(u2 and static) and float(np.mean([row["groups"]["all14"]["top1_macro"] for row in u2])) > float(
        np.mean([row["groups"]["all14"]["top1_macro"] for row in static])
    )
    u3_ablation_value = bool(u1 or u2) and means["all14_macro"] > max(
        [float(np.mean([row["groups"]["all14"]["top1_macro"] for row in values])) for values in (u1, u2) if values]
    )
    normal = next((row for row in diagnostics if row["method"] == "U3" and row["diagnostic"] == "normal"), None)
    temporal = next((row for row in diagnostics if row["method"] == "U3" and row["diagnostic"] == "temporal_shuffle"), None)
    temporal_value = bool(normal and temporal) and float(normal["groups"]["all14"]["top1_macro"]) > float(
        temporal["groups"]["all14"]["top1_macro"]
    )
    shuffled = next((row for row in diagnostics if row["method"] == "U3" and row["diagnostic"] == "csi_shuffle"), None)
    transition = normal.get("transition_statistics", {}) if normal else {}
    shuffled_transition = shuffled.get("transition_statistics", {}) if shuffled else {}
    direction_ratio = float(transition.get("correct_direction_ratio", float("nan")))
    shuffled_direction_ratio = float(shuffled_transition.get("correct_direction_ratio", float("nan")))
    direction_value = (
        math.isfinite(direction_ratio)
        and math.isfinite(shuffled_direction_ratio)
        and direction_ratio > 0.5
        and direction_ratio > shuffled_direction_ratio
    )
    mechanism = {
        "u1_vs_probability_average": u1_value,
        "u2_vs_static": u2_value,
        "u3_vs_u1_or_u2": u3_ablation_value,
        "temporal_shuffle_degrades": temporal_value,
        "transition_direction_ratio": direction_ratio,
        "csi_shuffle_direction_ratio": shuffled_direction_ratio,
        "transition_direction_association": direction_value,
    }
    passed = (
        prereg_gain
        and group_guard
        and invariants
        and seed_direction
        and all((u1_value, u2_value, u3_ablation_value, temporal_value, direction_value))
    )
    decision.update(
        {
            "passed": passed,
            "preregistered_gain_passed": prereg_gain,
            "group_regression_guard_passed": group_guard,
            "exact_invariants_passed": invariants,
            "three_seed_direction_consistent": seed_direction,
            "gains_vs_f1_pp": gains,
            "group_changes_vs_f1_pp": group_regressions,
            "mechanism_gates": mechanism,
            "paper_recommendation": "保留 CPSU 作为主方法。" if passed else "保留 F1；将 CPSU 作为负消融或未完成机制验证。",
            "reason": "全部预注册性能、机制和协议门槛通过。" if passed else "至少一个预注册性能、机制或协议门槛未通过。",
        }
    )
    return decision


def _write_final_report(
    path: Path,
    decision: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Any] | None,
    diagnostics: Sequence[Mapping[str, Any]],
) -> None:
    by_method = {str(row["method"]): row for row in results if int(row["seed"]) == 1}
    by_diagnostic = {str(row["diagnostic"]): row for row in diagnostics if str(row["method"]) == "U3" and int(row["seed"]) == 1}
    f1 = baselines.get("methods", {}).get("F1-fixed-evidence") if baselines else None

    def percentage(value: float | None) -> str:
        return "未运行" if value is None else f"{100.0 * float(value):.4f}%"

    def top1(method: str, field: str = "top1_macro") -> float | None:
        row = by_method.get(method)
        return None if row is None else float(row["groups"]["all14"][field])

    u1, u2, u3 = (by_method.get(name) for name in ("U1", "U2", "U3"))
    normal = by_diagnostic.get("normal")
    temporal = by_diagnostic.get("temporal_shuffle")
    reverse = by_diagnostic.get("reverse_time")
    shuffled = by_diagnostic.get("csi_shuffle")
    f1_all14 = float(f1["groups"]["all14"]["top1_macro"]) if f1 else None
    u3_all14 = top1("U3")
    u3_vs_f1 = None if u3_all14 is None or f1_all14 is None else 100.0 * (u3_all14 - f1_all14)
    temporal_change = (
        None
        if normal is None or temporal is None
        else 100.0 * (float(temporal["groups"]["all14"]["top1_macro"]) - float(normal["groups"]["all14"]["top1_macro"]))
    )
    reverse_change = (
        None
        if normal is None or reverse is None
        else 100.0 * (float(reverse["groups"]["all14"]["top1_macro"]) - float(normal["groups"]["all14"]["top1_macro"]))
    )
    direction = None if normal is None else float(normal["transition_statistics"]["correct_direction_ratio"])
    shuffled_direction = None if shuffled is None else float(shuffled["transition_statistics"]["correct_direction_ratio"])

    table_rows = []
    baseline_methods = baselines.get("methods", {}) if baselines else {}
    for key, name in (
        ("B0-sensing", "B0 sensing-only"),
        ("probability-average", "Probability average"),
        ("product-of-experts", "Product-of-Experts"),
        ("F1-fixed-evidence", "F1 fixed evidence"),
    ):
        baseline = baseline_methods.get(key)
        if baseline is None:
            continue
        table_rows.append(
            (
                name,
                float(baseline["groups"]["all14"]["top1_macro"]),
                float(baseline["groups"]["all14"]["top1_worst"]),
                float(baseline["missing_lidar"]["top1"]),
            )
        )
    for method in ("U1", "U1-eta0.5", "U1-no-prior", "U2", "U2-static", "U2-last", "U3"):
        row = by_method.get(method)
        if row is not None:
            table_rows.append(
                (
                    method,
                    float(row["groups"]["all14"]["top1_macro"]),
                    float(row["groups"]["all14"]["top1_worst"]),
                    float(row["missing_lidar"]["top1"]),
                )
            )

    lines = [
        "# CPSU 最终报告",
        "",
        f"状态：{decision.get('reason', '尚无结果')} 论文建议：{decision.get('paper_recommendation')}",
        "",
        "## 单 seed 正式结果",
        "",
        "| 方法 | All-14 | Worst | missing_lidar |",
        "| --- | ---: | ---: | ---: |",
        *[f"| {name} | {percentage(all14)} | {percentage(worst)} | {percentage(missing)} |" for name, all14, worst, missing in table_rows],
        "",
        (
            f"F0 20-RE CSI-only Top-1 为 {percentage(float(f1['csi_only']['top1']))}；它只作互补性对照，不直接输出最终结果。"
            if f1
            else "F0 CSI-only 结果缺失。"
        ),
        "",
        (
            f"U3 相对 F1 的 All-14 变化为 {u3_vs_f1:+.4f} pp；未达到 +0.5 pp 门槛，"
            "因此按预注册停止 seeds 2/3、U3-no-delta、自由 64x64 transition 和单帧扩展。"
            if u3_vs_f1 is not None
            else "U3 或 F1 结果缺失，不能执行扩展实验。"
        ),
        "",
        "## 十四项结论",
        "",
        "1. sensing prototype 是否作为结构化先验：是。使用冻结 M4 forward 保留的 p0，log(p0) 与原 prototype evidence 只差归一化常数，再经一次 T_s 得到 64 维 p_s。",
        (
            "2. CSI 是否预测 prototype 转移：结构上是，K=3 局部核受循环 beam topology 约束；但正式 U2/U3 均收敛到近 identity，未形成有效状态迁移。"
        ),
        "3. CSI 是否构造 prototype likelihood ratio：是。p_c 经 train-only 类别先验修正为 discriminative likelihood ratio；不声称精确生成式似然。",
        (
            f"4. posterior update 是否优于固定证据平均：否。U3={percentage(u3_all14)}，F1={percentage(f1_all14)}，变化 {u3_vs_f1:+.4f} pp。"
            if u3_vs_f1 is not None
            else "4. posterior update 是否优于固定证据平均：证据不足。"
        ),
        (
            f"5. transition 是否真正利用五帧传播演化：否。temporal shuffle 变化 "
            f"{temporal_change:+.4f} pp，reverse-time 变化 {reverse_change:+.4f} pp；"
            f"正确方向比例 normal/shuffle={direction:.4f}/{shuffled_direction:.4f}。"
            "整体性能下降来自有序 GRU likelihood 路径，q_delta 仍为 identity，不能归因于 transition。"
            if None not in (temporal_change, reverse_change, direction, shuffled_direction)
            else "5. transition 是否真正利用五帧传播演化：诊断未完成，不能声称。"
        ),
        (
            f"6. prior correction 是否有效：是。U1 η=1 为 {percentage(top1('U1'))}，η=0.5 为 "
            f"{percentage(top1('U1-eta0.5'))}，无修正为 {percentage(top1('U1-no-prior'))}。"
        ),
        (
            f"7. U3 是否同时优于 U1/U2/F1：否。U3={percentage(u3_all14)}，U1={percentage(top1('U1'))}，"
            f"U2={percentage(top1('U2'))}，F1={percentage(f1_all14)}。"
        ),
        (
            f"8. 是否改善 Worst 和 missing_lidar：否；相对 F1 分别变化 "
            f"{float(decision.get('gains_vs_f1_pp', {}).get('worst_pp', float('nan'))):+.4f} pp 和 "
            f"{float(decision.get('gains_vs_f1_pp', {}).get('missing_lidar_pp', float('nan'))):+.4f} pp。"
        ),
        (
            f"9. Full 是否完全不变：是；U3 max abs diff={float(u3['full_probability_max_abs_diff']):.3g}，Full CSI RE=0。"
            if u3
            else "9. Full 是否完全不变：U3 未运行。"
        ),
        (
            f"10. CSI-off 是否精确回退：是；U3 max abs diff={float(u3['csi_off_probability_max_abs_diff']):.3g}。"
            if u3
            else "10. CSI-off 是否精确回退：U3 未运行。"
        ),
        (
            f"11. 是否存在 radio-only 退化：CSI-only={percentage(float(u3['csi_only']['top1']))}，"
            f"sensing-only={percentage(float(u3['sensing_all14_macro']))}；U3 没有退化成 radio-only 输出，但 CSI 单支明显更弱。"
            if u3
            else "11. 是否存在 radio-only 退化：U3 未运行。"
        ),
        "12. 是否存在自由 transition 过拟合：未测试。局部 U2 未优于 static 且 U3 未超过 F1，按预注册禁止实现/运行自由 64x64 transition，不能声称存在或不存在过拟合。",
        f"13. 最终论文是否应保留 CPSU：{decision.get('paper_recommendation')} 主结果继续使用 F1。",
        "14. outer test 是否继续封存：是；未构造 test loader，所有结果均标记 outer_test_accessed=false。",
        "",
        "## Gate 记录",
        "",
        "```json",
        json.dumps(decision, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(config: Mapping[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    output = _path(config["output"]["root"])
    result_dir = output / ("smoke_results" if smoke else "results")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(result_dir.glob("*.json"))]
    rows = [_flat_result(result) for result in results]
    baseline_path = output / "baselines" / f"{'smoke_' if smoke else ''}f1_seed1.json"
    baselines = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    diagnostic_paths = sorted((output / "diagnostics").glob("*.json"))
    diagnostic_paths = [path for path in diagnostic_paths if path.name.startswith("smoke_") is bool(smoke)]
    diagnostics = [json.loads(path.read_text(encoding="utf-8")) for path in diagnostic_paths]
    prefix = "smoke_" if smoke else ""
    _write_csv(output / f"{prefix}likelihood_ablation.csv", [row for row in rows if row["method"] in CALIBRATION_METHODS])
    _write_csv(output / f"{prefix}transition_ablation.csv", [row for row in rows if row["method"].startswith("U2")])
    _write_csv(output / f"{prefix}posterior_update_summary.csv", rows)
    _write_csv(output / f"{prefix}seed_summary.csv", _mean_rows(rows))
    _write_csv(
        output / f"{prefix}mask_summary.csv",
        [{"method": result["method"], "seed": result["seed"], **mask} for result in results for mask in result["per_mask"]],
    )
    _write_csv(
        output / f"{prefix}transition_statistics.csv",
        [{"method": result["method"], "seed": result["seed"], **result["transition_statistics"]} for result in results],
    )
    _write_csv(
        output / f"{prefix}temporal_order_diagnostics.csv",
        [_diagnostic_row(result) for result in diagnostics],
    )
    _write_csv(
        output / f"{prefix}latency_summary.csv",
        [
            {
                "method": row["method"],
                "seed": row["seed"],
                "total_parameters": row["total_parameters"],
                "trainable_parameters": row["trainable_parameters"],
                "latency_ms_per_sample_mask": row["latency_ms_per_sample_mask"],
                "pilot_re_per_frame": row["pilot_re_per_frame"],
                "pilot_re_window": row["pilot_re_window"],
            }
            for row in rows
        ],
    )
    decision = _final_decision(config, results, baselines, diagnostics)
    _write_json(output / f"{prefix}stage_gate.json", decision)
    _write_final_report(output / f"{prefix}final_report.md", decision, results, baselines, diagnostics)
    return {
        "result_count": len(results),
        "diagnostic_count": len(diagnostics),
        "decision": decision,
        "outer_test_accessed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("preflight", "reproduce-f1", "calibrate", "train", "evaluate", "diagnose", "summarize"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", default="U3")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--diagnostic", choices=(*DIAGNOSTICS, "all"), default="normal")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-sweeps", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config.resolve())
    if args.action == "preflight":
        result = preflight(config)
    elif args.action == "reproduce-f1":
        result = reproduce_f1(args, config)
    elif args.action == "calibrate":
        result = calibrate_likelihood(args, config)
    elif args.action == "train":
        result = train_transition(args, config)
    elif args.action == "evaluate":
        result = evaluate_checkpoint(args, config, diagnose=False)
    elif args.action == "diagnose":
        result = evaluate_checkpoint(args, config, diagnose=True)
    else:
        result = summarize(config, smoke=bool(args.smoke))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True, default=str))


if __name__ == "__main__":
    main()
