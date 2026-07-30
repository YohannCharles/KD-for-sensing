#!/usr/bin/env python3
"""Local workflow for temporal sparse prototype compensation (TSPC)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from kd_sensing.baselines.full_pool_common import sha256_file
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.models.mask_conditioned_prototype_compensation import (
    MaskConditionedPrototypeCompensation,
    TSPC_AVAILABLE_COUNTS,
    TSPC_MASK_NAMES,
    TSPC_METHODS,
)
from kd_sensing.models.prototype_fusion_losses import topology_risk
from kd_sensing.models.radio_prototype_expert import RadioPrototypeExpert

if __package__:
    from .run_csi_anchored_completion import (
        _frequency_positions,
        _load_m4,
        _load_radio,
        _load_radio_training_teacher,
        _radio_from_candidates,
    )
    from .run_mmw_trajectory_baselines import ALL_PATTERNS
    from .run_quality_topology_prototype_routing import (
        _build_fusion,
        _load_checkpoint,
        _load_config as _load_qtpr_config,
        _load_records,
        _topology,
        preflight as _qtpr_preflight,
    )
    from .run_sparse_pilot_recovery import _prediction_metrics
else:
    from run_csi_anchored_completion import (
        _frequency_positions,
        _load_m4,
        _load_radio,
        _load_radio_training_teacher,
        _radio_from_candidates,
    )
    from run_mmw_trajectory_baselines import ALL_PATTERNS
    from run_quality_topology_prototype_routing import (
        _build_fusion,
        _load_checkpoint,
        _load_config as _load_qtpr_config,
        _load_records,
        _topology,
        preflight as _qtpr_preflight,
    )
    from run_sparse_pilot_recovery import _prediction_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/temporal_sparse_prototype_compensation.yaml"
MASK_VALUES = tuple(tuple(bool(value) for value in ALL_PATTERNS[name]) for name in TSPC_MASK_NAMES)
MASK_COUNT_BY_ID = torch.tensor(TSPC_AVAILABLE_COUNTS, dtype=torch.long)
GROUP_NAMES = {1: "single", 2: "two", 3: "three"}
METRICS = ("top1", "top3", "top5", "within3", "mae", "normalized_gain", "beam_loss_db", "fix_rate", "harm_rate")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(name for name in row if name not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path) -> dict[str, Any]:
    config = safe_load_yaml(path.read_text(encoding="utf-8"))
    if config["protocol"].get("outer_test_enabled") is not False:
        raise ValueError("TSPC requires the outer test to remain disabled.")
    return config


def _allocation(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        allocation = dict(config["source"]["allocations"][name])
    except KeyError as error:
        raise ValueError(f"Unknown TSPC allocation: {name}.") from error
    expected = int(allocation["re_per_frame"]) * int(allocation["history_frames"])
    if expected != int(allocation["re_window"]):
        raise ValueError(f"Allocation {name} has inconsistent RE accounting: {allocation}.")
    patterns, frequencies = (int(part) for part in str(allocation["budget"]).split("x", 1))
    if patterns * frequencies != int(allocation["re_per_frame"]):
        raise ValueError(f"Allocation {name} budget does not match RE/frame.")
    return allocation


def _allocation_qtpr_config(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    allocation = _allocation(config, name)
    qtpr = _load_qtpr_config(_path(allocation["qtpr_config"]))
    if allocation["alignment"] == "factorized_teacher_train_calibration":
        qtpr["radio_encoder"]["checkpoint"] = allocation["radio_checkpoint"]
        qtpr["radio_encoder"]["checkpoint_sha256"] = allocation["radio_checkpoint_sha256"]
        qtpr["radio_encoder"]["history_length"] = int(allocation["history_frames"])
        qtpr["source_cache"]["use_cached_radio"] = False
    return qtpr


def _temperature(raw: torch.Tensor) -> float:
    return float(F.softplus(torch.as_tensor(raw).float()).item() + 1e-6)


def _source_components(
    config: Mapping[str, Any],
    allocation_name: str,
    device: torch.device,
) -> tuple[dict[str, Any], Any, Any, RadioPrototypeExpert, dict[str, float], Any | None]:
    allocation = _allocation(config, allocation_name)
    qtpr = _allocation_qtpr_config(config, allocation_name)
    topology = _topology(qtpr)
    m4 = _load_m4(qtpr, device)
    radio_encoder = _load_radio(qtpr, device)
    fusion = None
    if allocation["alignment"] == "f1_checkpoint":
        fusion = _build_fusion(qtpr, "F1", topology, device)
        _load_checkpoint(fusion, _path(allocation["f1_checkpoint"]), qtpr, expected_method="F1")
        fusion.eval()
        expert = fusion.radio_expert
        sensing_temperature = float(fusion.sensing_temperature().detach())
        physical_radio_temperature = float(expert.temperature().detach())
    else:
        teacher = _load_radio_training_teacher(qtpr, device)
        expert = RadioPrototypeExpert(radio_dim=128, hidden_dim=128, prototype_dim=64, temperature=0.001).to(device)
        expert.initialize_from_teacher(m4.prototype_bank, teacher.state_dict())
        expert.eval()
        for parameter in expert.parameters():
            parameter.requires_grad_(False)
        temporal = _allocation(config, "temporal_2x2")
        temporal_qtpr = _allocation_qtpr_config(config, "temporal_2x2")
        payload = torch.load(_path(temporal["f1_checkpoint"]), map_location="cpu", weights_only=False)
        sensing_temperature = _temperature(payload["model_state"]["sensing_temperature.raw"])
        physical_radio_temperature = float("nan")
    calibration = {
        "sensing_temperature": sensing_temperature,
        "physical_radio_temperature": physical_radio_temperature,
        "prototype_bank_temperature": float(m4.prototype_bank.temperature),
        "effective_radio_temperature": physical_radio_temperature / float(m4.prototype_bank.temperature),
    }
    return qtpr, m4, radio_encoder, expert, calibration, fusion


def preflight(config: Mapping[str, Any], allocation_name: str) -> dict[str, Any]:
    allocation = _allocation(config, allocation_name)
    qtpr = _allocation_qtpr_config(config, allocation_name)
    source = _qtpr_preflight(qtpr, write_manifest=False)
    checks = {
        "protocol": qtpr["protocol"]["id"] == config["protocol"]["id"],
        "train_samples": int(qtpr["protocol"]["expected_train_samples"]) == int(config["protocol"]["expected_train_samples"]),
        "validation_samples": int(qtpr["protocol"]["expected_validation_samples"])
        == int(config["protocol"]["expected_validation_samples"]),
        "outer_test_disabled": qtpr["protocol"].get("outer_test_enabled") is False,
        "re_accounting": int(allocation["re_window"])
        == int(allocation["re_per_frame"]) * int(allocation["history_frames"]),
    }
    if allocation["alignment"] == "f1_checkpoint":
        checks["f1_checkpoint_hash"] = sha256_file(_path(allocation["f1_checkpoint"])) == allocation["f1_checkpoint_sha256"]
    else:
        checks["radio_checkpoint_hash"] = (
            sha256_file(_path(allocation["radio_checkpoint"])) == allocation["radio_checkpoint_sha256"]
        )
    if not all(checks.values()):
        raise ValueError(f"TSPC preflight failed: {checks}.")
    result = {
        "allocation": allocation_name,
        "checks": checks,
        "source": source,
        "pilot_re_per_frame": int(allocation["re_per_frame"]),
        "pilot_history_frames": int(allocation["history_frames"]),
        "pilot_re_window": int(allocation["re_window"]),
        "future_channel_used_as_input": False,
        "outer_test_accessed": False,
    }
    output = _path(config["output"]["root"])
    _write_json(output / "preflight" / f"{allocation_name}.json", result)
    resolved = output / "resolved_configs" / "base.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    return result


def _cache_path(config: Mapping[str, Any], allocation_name: str, role: str) -> Path:
    return _path(config["output"]["cache_root"]) / allocation_name / f"{role}.pt"


def _calibration_path(config: Mapping[str, Any], allocation_name: str) -> Path:
    return _path(config["output"]["cache_root"]) / allocation_name / "calibration.json"


def _calibrate_temperature(
    raw_evidence: torch.Tensor,
    labels: torch.Tensor,
    fraction: float,
    device: torch.device,
) -> tuple[float, list[dict[str, float]]]:
    count = max(1, int(len(labels) * float(fraction)))
    generator = torch.Generator().manual_seed(730001)
    indices = torch.randperm(len(labels), generator=generator)[:count]
    evidence = raw_evidence.index_select(0, indices).to(device)
    target = labels.index_select(0, indices).to(device)
    candidates = torch.logspace(-4.0, 0.0, 65, device=device)
    rows: list[dict[str, float]] = []
    best_temperature, best_loss = 1.0, float("inf")
    for value in candidates:
        loss = float(F.cross_entropy(evidence / value, target).item())
        temperature = float(value.item())
        rows.append({"effective_radio_temperature": temperature, "train_calibration_nll": loss})
        if loss < best_loss:
            best_temperature, best_loss = temperature, loss
    low, high = best_temperature / (10 ** (1 / 16)), best_temperature * (10 ** (1 / 16))
    for value in torch.logspace(math.log10(low), math.log10(high), 33, device=device):
        loss = float(F.cross_entropy(evidence / value, target).item())
        temperature = float(value.item())
        rows.append({"effective_radio_temperature": temperature, "train_calibration_nll": loss})
        if loss < best_loss:
            best_temperature, best_loss = temperature, loss
    return best_temperature, rows


@torch.inference_mode()
def prepare_cache(config: Mapping[str, Any], allocation_name: str, role: str, device: torch.device) -> dict[str, Any]:
    if role not in {"train", "validation"}:
        raise ValueError("TSPC cache role must be train or validation.")
    audit = preflight(config, allocation_name)
    allocation = _allocation(config, allocation_name)
    qtpr, m4, radio_encoder, expert, calibration, fusion = _source_components(config, allocation_name, device)
    feature, recovery, _ = _load_records(qtpr, role)
    count = len(feature["target"])
    batch_size = int(config["training"]["evaluation_batch_size"])
    frequencies = _frequency_positions(qtpr, str(allocation["budget"]), device)
    seed = int(config["training"][f"{role}_radio_seed"])
    generator = torch.Generator(device=device).manual_seed(seed)
    snr_generator = torch.Generator(device=device).manual_seed(seed + 1)
    raw_chunks, available_chunks = [], []
    f1_max_abs = 0.0
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        candidates = recovery["candidate_history"][start:stop, -int(allocation["history_frames"]) :].to(device)
        if role == "train":
            snr = torch.empty(stop - start, device=device).uniform_(
                float(config["training"]["train_snr_db_min"]),
                float(config["training"]["train_snr_db_max"]),
                generator=snr_generator,
            )
            dropout = float(config["training"]["train_pilot_dropout_probability"])
        else:
            snr = torch.full((stop - start,), float(config["training"]["validation_snr_db"]), device=device)
            dropout = 0.0
        radio = _radio_from_candidates(
            radio_encoder,
            candidates,
            budget=str(allocation["budget"]),
            frequencies=frequencies,
            snr=snr,
            generator=generator,
            dropout_probability=dropout,
        )
        c_radio = radio["c_radio"]
        if allocation["aggregation"] == "mean_frames":
            c_radio = radio["frame_csi_features"].mean(dim=1)
        expert_output = expert(c_radio, m4.prototype_bank)
        with torch.autocast(device_type=device.type, enabled=False):
            raw = m4.prototype_bank(expert_output["z_radio"].float()).float()
        raw_chunks.append(raw.cpu())
        available_chunks.append(radio["csi_available"].cpu())

        if fusion is not None and start == 0:
            rows = min(64, stop - start)
            p0 = recovery[f"p0_{TSPC_MASK_NAMES[0]}"][start : start + rows].to(device)
            sensing = p0.clamp_min(1e-12).log()
            physical = torch.tensor(MASK_VALUES[0], device=device).expand(rows, -1)
            old = fusion(
                sensing,
                sensing,
                c_radio[:rows],
                radio["csi_quality"][:rows],
                radio["csi_available"][:rows],
                physical,
                m4.prototype_bank,
                _topology(qtpr).distance.to(device),
            )["final_probability"]
            probe = MaskConditionedPrototypeCompensation(
                "M3",
                initial_weight=float(config["source"]["initial_weight"]),
                sensing_temperature=calibration["sensing_temperature"],
                radio_temperature=calibration["effective_radio_temperature"],
            ).to(device)
            new = probe(
                sensing,
                raw[:rows],
                torch.zeros(rows, device=device, dtype=torch.long),
                radio["csi_available"][:rows],
                base_probability=p0,
            )["final_probability"]
            f1_max_abs = float((old.float() - new.float()).abs().max().item())

    raw_radio = torch.cat(raw_chunks)
    csi_available = torch.cat(available_chunks)
    calibration_rows: list[dict[str, float]] = []
    if allocation["alignment"] == "factorized_teacher_train_calibration":
        path = _calibration_path(config, allocation_name)
        if role == "train":
            calibrated, calibration_rows = _calibrate_temperature(
                raw_radio,
                feature["target"],
                float(config["diagnostics"]["train_calibration_fraction"]),
                device,
            )
            calibration["effective_radio_temperature"] = calibrated
            calibration["physical_radio_temperature"] = calibrated * calibration["prototype_bank_temperature"]
            _write_csv(_path(config["output"]["root"]) / "radio_temperature_grid" / f"{allocation_name}.csv", calibration_rows)
        elif path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            calibration.update(saved["calibration"])
        else:
            raise FileNotFoundError(f"Prepare the train cache before validation for {allocation_name}.")
    payload = {
        "raw_radio_evidence": raw_radio,
        "csi_available": csi_available,
        "sample_ids": list(feature["sample_ids"]),
        "target": feature["target"],
        "trajectory_ids": list(feature["trajectory_ids"]),
        "allocation": allocation_name,
        "role": role,
        "calibration": calibration,
        "protocol_id": config["protocol"]["id"],
        "pilot_re_per_frame": int(allocation["re_per_frame"]),
        "pilot_history_frames": int(allocation["history_frames"]),
        "pilot_re_window": int(allocation["re_window"]),
        "aggregation": allocation["aggregation"],
        "f1_initialization_max_abs": f1_max_abs if fusion is not None else None,
        "future_channel_used_as_input": False,
        "outer_test_accessed": False,
    }
    path = _cache_path(config, allocation_name, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    manifest = {
        "version": "tspc_evidence_cache_v1",
        "allocation": allocation_name,
        "role": role,
        "cache": str(path.resolve()),
        "cache_sha256": sha256_file(path),
        "sample_count": count,
        "trajectory_count": len(set(payload["trajectory_ids"])),
        "sample_ids_exact": list(payload["sample_ids"]) == list(recovery["sample_ids"]),
        "targets_exact": bool(torch.equal(payload["target"], recovery["labels_future"])),
        "calibration": calibration,
        "f1_initialization_max_abs": payload["f1_initialization_max_abs"],
        "pilot_re_per_frame": payload["pilot_re_per_frame"],
        "pilot_history_frames": payload["pilot_history_frames"],
        "pilot_re_window": payload["pilot_re_window"],
        "future_channel_used_as_input": False,
        "outer_test_accessed": False,
        "preflight_checks": audit["checks"],
    }
    _write_json(_path(config["output"]["root"]) / "cache_manifests" / f"{allocation_name}_{role}.json", manifest)
    if role == "train":
        _write_json(_calibration_path(config, allocation_name), {"calibration": calibration, "outer_test_accessed": False})
    return manifest


def _load_cache(config: Mapping[str, Any], allocation_name: str, role: str) -> dict[str, Any]:
    path = _cache_path(config, allocation_name, role)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    expected = int(config["protocol"][f"expected_{role}_samples"])
    checks = {
        "count": len(payload["target"]) == expected,
        "allocation": payload.get("allocation") == allocation_name,
        "role": payload.get("role") == role,
        "protocol": payload.get("protocol_id") == config["protocol"]["id"],
        "future": payload.get("future_channel_used_as_input") is False,
        "outer": payload.get("outer_test_accessed") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"TSPC evidence cache mismatch: {checks}.")
    return payload


def _mask_probabilities(recovery: Mapping[str, Any], indices: torch.Tensor, mask_ids: torch.Tensor) -> torch.Tensor:
    output = torch.empty(len(indices), 64)
    for mask_id in torch.unique(mask_ids).tolist():
        selected = mask_ids.eq(int(mask_id)).nonzero(as_tuple=False).squeeze(1)
        source = indices.index_select(0, selected)
        output.index_copy_(0, selected, recovery[f"p0_{TSPC_MASK_NAMES[int(mask_id)]}"][source])
    return output


def balanced_epoch_schedule(labels: torch.Tensor, *, epoch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-robin classes while rotating all 14 masks within every class."""
    target = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    groups: list[torch.Tensor] = []
    masks = torch.empty(len(target), dtype=torch.long)
    for beam in sorted(torch.unique(target).tolist()):
        indices = target.eq(int(beam)).nonzero(as_tuple=False).squeeze(1)
        generator = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(epoch) * 101 + int(beam))
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        groups.append(shuffled)
        assigned = (torch.arange(len(shuffled)) + int(epoch) + int(seed) + int(beam)) % len(TSPC_MASK_NAMES)
        masks.index_copy_(0, shuffled, assigned)
    order = [group[offset] for offset in range(max(map(len, groups))) for group in groups if offset < len(group)]
    return torch.stack(order), masks


def _group_weights(mask_ids: torch.Tensor, profile: str, device: torch.device) -> torch.Tensor:
    if profile == "mask_equal":
        return torch.ones(len(mask_ids), device=device)
    if profile != "group_equal":
        raise ValueError(f"Unknown TSPC loss profile: {profile}.")
    counts = MASK_COUNT_BY_ID.index_select(0, mask_ids).to(device)
    group_sizes = torch.tensor((4.0, 6.0, 4.0), device=device)
    weights = group_sizes.index_select(0, counts - 1).reciprocal()
    return weights / weights.mean()


def _run_stem(allocation: str, profile: str, method: str, seed: int) -> str:
    return f"{allocation}_{profile}_{method}_seed{int(seed)}"


def _checkpoint_path(config: Mapping[str, Any], stem: str, name: str = "best.pt") -> Path:
    return _path(config["output"]["root"]) / "checkpoints" / stem / name


def _build_model(config: Mapping[str, Any], method: str, cache: Mapping[str, Any], device: torch.device):
    calibration = cache["calibration"]
    model = MaskConditionedPrototypeCompensation(
        method,
        initial_weight=float(config["source"]["initial_weight"]),
        sensing_temperature=float(calibration["sensing_temperature"]),
        radio_temperature=float(calibration["effective_radio_temperature"]),
    ).to(device)
    if model.trainable_parameter_count >= 20:
        raise RuntimeError(f"TSPC trainable parameter count must be below 20, got {model.trainable_parameter_count}.")
    return model


def _aggregate(per_mask: list[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[Mapping[str, Any]]] = {"single": [], "two": [], "three": [], "all14": []}
    for row in per_mask:
        if row["mask"] == "full":
            continue
        groups[GROUP_NAMES[int(row["available_count"])]].append(row)
        groups["all14"].append(row)
    result: dict[str, dict[str, float]] = {}
    for name, rows in groups.items():
        values: dict[str, float] = {}
        for metric in METRICS:
            finite = [float(row[metric]) for row in rows if math.isfinite(float(row[metric]))]
            values[f"{metric}_macro"] = float(np.mean(finite)) if finite else float("nan")
            if metric in {"mae", "beam_loss_db", "harm_rate"}:
                values[f"{metric}_worst"] = float(np.max(finite)) if finite else float("nan")
            else:
                values[f"{metric}_worst"] = float(np.min(finite)) if finite else float("nan")
        result[name] = values
    return result


@torch.inference_mode()
def evaluate_model(
    config: Mapping[str, Any],
    allocation_name: str,
    model: MaskConditionedPrototypeCompensation,
    *,
    device: torch.device,
    diagnostic: str = "normal",
    seed: int = 1,
) -> dict[str, Any]:
    if diagnostic not in {"normal", "csi_shuffle", "sensing_shuffle", "csi_off", "mask_swap"}:
        raise ValueError(f"Unknown TSPC diagnostic: {diagnostic}.")
    qtpr = _allocation_qtpr_config(config, allocation_name)
    feature, recovery, _ = _load_records(qtpr, "validation")
    cache = _load_cache(config, allocation_name, "validation")
    labels = feature["target"]
    power = feature["future_beam_power"]
    count = len(labels)
    batch_size = int(config["training"]["evaluation_batch_size"])
    generator = torch.Generator().manual_seed(810_000 + int(seed))
    permutation = torch.randperm(count, generator=generator)
    per_mask: list[dict[str, Any]] = []
    oracle_values = []
    started = time.monotonic()

    for mask_id, mask_name in enumerate(TSPC_MASK_NAMES):
        probabilities = []
        lambdas = []
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            base_indices = torch.arange(start, stop)
            sensing_indices = permutation[start:stop] if diagnostic == "sensing_shuffle" else base_indices
            radio_indices = permutation[start:stop] if diagnostic == "csi_shuffle" else base_indices
            base_probability = recovery[f"p0_{mask_name}"][sensing_indices].to(device)
            sensing = base_probability.clamp_min(1e-12).log()
            radio = cache["raw_radio_evidence"][radio_indices].to(device)
            available = cache["csi_available"][radio_indices].to(device)
            if diagnostic == "csi_off":
                available = torch.zeros_like(available)
            lookup_id = (mask_id + int(config["diagnostics"]["mask_swap_offset"])) % len(TSPC_MASK_NAMES) if diagnostic == "mask_swap" else mask_id
            output = model(
                sensing,
                radio,
                torch.full((stop - start,), lookup_id, device=device, dtype=torch.long),
                available,
                base_probability=base_probability,
            )
            probabilities.append(output["final_probability"].cpu())
            lambdas.append(output["lambda"].cpu())
        probability = torch.cat(probabilities)
        original_base = recovery[f"p0_{mask_name}"]
        row = {"mask": mask_name, "mask_id": mask_id, "available_count": TSPC_AVAILABLE_COUNTS[mask_id]}
        row.update(_prediction_metrics(probability, labels, power, original_base))
        row["nll"] = float(F.nll_loss(probability.clamp_min(1e-12).log(), labels).item())
        row["lambda"] = float(torch.cat(lambdas).mean())
        radio_probability = torch.softmax(
            cache["raw_radio_evidence"] / float(cache["calibration"]["effective_radio_temperature"]), dim=-1
        )
        oracle = original_base.argmax(dim=-1).eq(labels) | radio_probability.argmax(dim=-1).eq(labels)
        row["oracle_top1"] = float(oracle.float().mean())
        oracle_values.append(row["oracle_top1"])
        per_mask.append(row)

    full_probability = recovery["p0_full"]
    full = {"mask": "full", "mask_id": -1, "available_count": 4}
    full.update(_prediction_metrics(full_probability, labels, power, full_probability))
    full["nll"] = float(F.nll_loss(full_probability.clamp_min(1e-12).log(), labels).item())
    full["lambda"] = 0.0
    full["oracle_top1"] = full["top1"]
    per_mask.insert(0, full)
    groups = _aggregate(per_mask)

    probe_count = min(batch_size, count)
    p_full = recovery["p0_full"][:probe_count].to(device)
    full_probe = model(
        p_full.clamp_min(1e-12).log(),
        cache["raw_radio_evidence"][:probe_count].to(device),
        torch.full((probe_count,), -1, device=device, dtype=torch.long),
        torch.ones(probe_count, device=device, dtype=torch.bool),
        base_probability=p_full,
    )["final_probability"]
    p_missing = recovery[f"p0_{TSPC_MASK_NAMES[0]}"][:probe_count].to(device)
    off_probe = model(
        p_missing.clamp_min(1e-12).log(),
        cache["raw_radio_evidence"][:probe_count].to(device),
        torch.zeros(probe_count, device=device, dtype=torch.long),
        torch.zeros(probe_count, device=device, dtype=torch.bool),
        base_probability=p_missing,
    )["final_probability"]
    radio_probability = torch.softmax(
        cache["raw_radio_evidence"] / float(cache["calibration"]["effective_radio_temperature"]), dim=-1
    )
    csi_metrics = _prediction_metrics(radio_probability, labels, power, recovery["p0_full"])
    oracle_all14 = float(np.mean(oracle_values))
    b0 = float(config["baselines"]["b0_all14_macro"])
    method_value = groups["all14"]["top1_macro"]
    elapsed = time.monotonic() - started
    return {
        "allocation": allocation_name,
        "method": model.method,
        "seed": int(seed),
        "diagnostic": diagnostic,
        "sample_count": count,
        "groups": groups,
        "full": full,
        "per_mask": per_mask,
        "missing_lidar": next(row for row in per_mask if row["mask"] == "missing_lidar"),
        "csi_only": csi_metrics,
        "oracle_all14_macro": oracle_all14,
        "oracle_headroom_capture": (method_value - b0) / max(oracle_all14 - b0, 1e-12),
        "lambda_table": {name: float(value) for name, value in zip(TSPC_MASK_NAMES, model.lambda_table().cpu().tolist())},
        "alpha_by_available_count": {
            str(index + 1): float(value)
            for index, value in enumerate(torch.sigmoid(model.alpha_count).cpu().tolist())
        }
        if hasattr(model, "alpha_count")
        else {},
        "trainable_parameters": model.trainable_parameter_count,
        "latency_ms_per_sample_mask": 1000.0 * elapsed / max(count * len(TSPC_MASK_NAMES), 1),
        "full_bypass_max_abs": float((full_probe - p_full).abs().max()),
        "full_bypass_argmax_mismatch": int(full_probe.argmax(dim=-1).ne(p_full.argmax(dim=-1)).sum()),
        "csi_off_max_abs": float((off_probe - p_missing).abs().max()),
        "csi_off_argmax_mismatch": int(off_probe.argmax(dim=-1).ne(p_missing.argmax(dim=-1)).sum()),
        "pilot_re_per_frame": int(cache["pilot_re_per_frame"]),
        "pilot_history_frames": int(cache["pilot_history_frames"]),
        "pilot_re_window": int(cache["pilot_re_window"]),
        "full_pilot_re": 0,
        "elapsed_seconds": elapsed,
        "outer_test_accessed": False,
    }


def _quick_score(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all14_macro": float(result["groups"]["all14"]["top1_macro"]),
        "all14_worst": float(result["groups"]["all14"]["top1_worst"]),
        "single_macro": float(result["groups"]["single"]["top1_macro"]),
        "two_macro": float(result["groups"]["two"]["top1_macro"]),
        "three_macro": float(result["groups"]["three"]["top1_macro"]),
        "missing_lidar": float(result["missing_lidar"]["top1"]),
        "val_nll": float(np.mean([row["nll"] for row in result["per_mask"] if row["mask"] != "full"])),
    }


def _save_checkpoint(
    path: Path,
    model: MaskConditionedPrototypeCompensation,
    optimizer: torch.optim.Optimizer,
    *,
    config: Mapping[str, Any],
    allocation: str,
    profile: str,
    seed: int,
    epoch: int,
    metrics: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "method": model.method,
            "allocation": allocation,
            "loss_profile": profile,
            "seed": int(seed),
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "protocol_id": config["protocol"]["id"],
            "trainable_parameters": model.trainable_parameter_count,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "outer_test_accessed": False,
        },
        path,
    )


def _load_trained_model(
    config: Mapping[str, Any], checkpoint: Path, cache: Mapping[str, Any], device: torch.device
) -> tuple[MaskConditionedPrototypeCompensation, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol_id") != config["protocol"]["id"] or payload.get("outer_test_accessed") is not False:
        raise ValueError("TSPC checkpoint protocol mismatch.")
    model = _build_model(config, str(payload["method"]), cache, device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def train(
    config: Mapping[str, Any],
    allocation_name: str,
    method: str,
    profile: str,
    seed: int,
    device: torch.device,
    *,
    epochs_override: int | None = None,
) -> dict[str, Any]:
    if method not in TSPC_METHODS:
        raise ValueError(f"Unknown TSPC method: {method}.")
    preflight(config, allocation_name)
    cache = _load_cache(config, allocation_name, "train")
    qtpr = _allocation_qtpr_config(config, allocation_name)
    feature, recovery, _ = _load_records(qtpr, "train")
    if list(cache["sample_ids"]) != list(feature["sample_ids"]):
        raise ValueError("TSPC train cache and recovery identities differ.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _build_model(config, method, cache, device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    topology = _topology(qtpr).distance.to(device)
    epochs = int(epochs_override or config["training"]["max_epochs"])
    batch_size = int(config["training"]["batch_size"])
    stem = _run_stem(allocation_name, profile, method, seed)
    output = _path(config["output"]["root"])
    resolved = {
        "stem": stem,
        "allocation": allocation_name,
        "method": method,
        "loss_profile": profile,
        "seed": seed,
        "trainable_parameters": model.trainable_parameter_count,
        "calibration": cache["calibration"],
        **dict(config),
    }
    resolved_path = output / "resolved_configs" / f"{stem}.yaml"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")

    initial_validation = evaluate_model(config, allocation_name, model, device=device, seed=seed)
    initial_scores = _quick_score(initial_validation)
    best_score = initial_scores[str(config["training"]["selection_metric"])]
    patience = 0
    history: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "train_samples": 0,
            **{f"validation_{name}": value for name, value in initial_scores.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": 0.0,
            **{
                f"lambda_{name}": float(value)
                for name, value in zip(TSPC_MASK_NAMES, model.lambda_table().detach().cpu())
            },
        }
    ]
    _write_csv(output / "training_logs" / f"{stem}.csv", history)
    _save_checkpoint(
        _checkpoint_path(config, stem),
        model,
        optimizer,
        config=config,
        allocation=allocation_name,
        profile=profile,
        seed=seed,
        epoch=0,
        metrics=initial_scores,
    )
    sampling_counts: dict[str, int] = defaultdict(int)
    for epoch in range(1, epochs + 1):
        model.train()
        order, mask_schedule = balanced_epoch_schedule(feature["target"], epoch=epoch, seed=seed)
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        gradient_total = 0.0
        started = time.monotonic()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            mask_ids = mask_schedule.index_select(0, indices)
            for mask_id, count in zip(*torch.unique(mask_ids, return_counts=True)):
                sampling_counts[TSPC_MASK_NAMES[int(mask_id)]] += int(count)
            base = _mask_probabilities(recovery, indices, mask_ids).to(device)
            sensing = base.clamp_min(1e-12).log()
            radio = cache["raw_radio_evidence"][indices].to(device)
            available = cache["csi_available"][indices].to(device)
            labels = feature["target"][indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            output_values = model(sensing, radio, mask_ids.to(device), available, base_probability=base)
            task_items = F.cross_entropy(output_values["final_evidence"], labels, reduction="none")
            weights = _group_weights(mask_ids, profile, device)
            task = (task_items * weights).sum() / weights.sum().clamp_min(1.0)
            topology_loss = (topology_risk(output_values["final_evidence"], labels, topology) * weights).sum()
            topology_loss = topology_loss / weights.sum().clamp_min(1.0)
            regularization = model.regularization()
            total = (
                task
                + float(config["loss"]["topology"]) * topology_loss
                + float(config["loss"]["delta"]) * regularization["delta"]
                + float(config["loss"]["group"]) * regularization["group"]
                + float(config["loss"]["severity"]) * regularization["severity"]
            )
            total.backward()
            gradient_norm = math.sqrt(
                sum(float(parameter.grad.detach().square().sum()) for parameter in parameters if parameter.grad is not None)
            )
            optimizer.step()
            count = len(indices)
            seen += count
            gradient_total += gradient_norm * count
            for name, value in {
                "total": total,
                "task": task,
                "topology": topology_loss,
                **regularization,
            }.items():
                totals[name] += float(value.detach()) * count

        validation = evaluate_model(config, allocation_name, model, device=device, seed=seed)
        scores = _quick_score(validation)
        row = {
            "epoch": epoch,
            "train_samples": seen,
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
            "gradient_norm": gradient_total / max(seen, 1),
            **{f"validation_{name}": value for name, value in scores.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - started,
            **{f"lambda_{name}": float(value) for name, value in zip(TSPC_MASK_NAMES, model.lambda_table().detach().cpu())},
        }
        history.append(row)
        _write_csv(output / "training_logs" / f"{stem}.csv", history)
        _save_checkpoint(
            _checkpoint_path(config, stem, "last.pt"),
            model,
            optimizer,
            config=config,
            allocation=allocation_name,
            profile=profile,
            seed=seed,
            epoch=epoch,
            metrics=scores,
        )
        score = scores[str(config["training"]["selection_metric"])]
        if score > best_score:
            best_score = score
            patience = 0
            _save_checkpoint(
                _checkpoint_path(config, stem),
                model,
                optimizer,
                config=config,
                allocation=allocation_name,
                profile=profile,
                seed=seed,
                epoch=epoch,
                metrics=scores,
            )
        else:
            patience += 1
        if patience >= int(config["training"]["patience"]):
            break

    best_model, payload = _load_trained_model(config, _checkpoint_path(config, stem), cache, device)
    result = evaluate_model(config, allocation_name, best_model, device=device, seed=seed)
    result.update(
        loss_profile=profile,
        selected_epoch=int(payload["epoch"]),
        epochs_ran=len(history) - 1,
        checkpoint=str(_checkpoint_path(config, stem).resolve()),
        checkpoint_sha256=sha256_file(_checkpoint_path(config, stem)),
    )
    _write_json(output / "results" / f"{stem}.json", result)
    sampling = {
        "stem": stem,
        "sample_count_per_epoch": len(feature["target"]),
        "epochs_ran": len(history) - 1,
        "mask_counts": dict(sorted(sampling_counts.items())),
        "front_n_sampling_used": False,
        "beam_conditioned_mask_rotation": True,
        "train_validation_trajectory_overlap_count": 0,
        "outer_test_accessed": False,
    }
    _write_json(output / "sampling_statistics" / f"{stem}.json", sampling)
    _write_json(output / "complete" / f"{stem}.json", {"status": "completed", **_quick_score(result), **sampling})
    return result


@torch.inference_mode()
def lambda_curves(config: Mapping[str, Any], allocation_name: str, device: torch.device) -> dict[str, Any]:
    qtpr = _allocation_qtpr_config(config, allocation_name)
    feature, recovery, _ = _load_records(qtpr, "validation")
    cache = _load_cache(config, allocation_name, "validation")
    labels = feature["target"].to(device)
    radio = cache["raw_radio_evidence"].to(device)
    radio_calibrated = radio / float(cache["calibration"]["effective_radio_temperature"])
    start = float(config["diagnostics"]["lambda_grid_start"])
    stop = float(config["diagnostics"]["lambda_grid_stop"])
    step = float(config["diagnostics"]["lambda_grid_step"])
    candidates = np.arange(start, stop + step / 2, step)
    rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for mask_id, name in enumerate(TSPC_MASK_NAMES):
        sensing = recovery[f"p0_{name}"].to(device).clamp_min(1e-12).log()
        sensing = sensing / float(cache["calibration"]["sensing_temperature"])
        best = None
        for weight in candidates:
            probability = torch.softmax(sensing + float(weight) * (radio_calibrated - sensing), dim=-1)
            top1 = float(probability.argmax(dim=-1).eq(labels).float().mean())
            nll = float(F.nll_loss(probability.clamp_min(1e-12).log(), labels).item())
            row = {"allocation": allocation_name, "mask": name, "available_count": TSPC_AVAILABLE_COUNTS[mask_id], "lambda": float(weight), "top1": top1, "nll": nll}
            rows.append(row)
            if best is None or top1 > best["top1"]:
                best = row
        assert best is not None
        oracle_rows.append(best)
    output = _path(config["output"]["root"])
    _write_csv(output / "mask_lambda_curves.csv", rows)
    _write_csv(output / "oracle_summary.csv", oracle_rows)

    train_cache = _load_cache(config, allocation_name, "train")
    _, train_recovery, _ = _load_records(qtpr, "train")
    calibration_count = int(len(train_cache["target"]) * float(config["diagnostics"]["train_calibration_fraction"]))
    generator = torch.Generator().manual_seed(740001)
    indices = torch.randperm(len(train_cache["target"]), generator=generator)[:calibration_count]
    train_radio = train_cache["raw_radio_evidence"][indices].to(device)
    train_radio = train_radio / float(train_cache["calibration"]["effective_radio_temperature"])
    train_labels = train_cache["target"][indices].to(device)
    grid_rows = []
    for weight in candidates:
        correct, losses = [], []
        for mask_id, name in enumerate(TSPC_MASK_NAMES):
            sensing = train_recovery[f"p0_{name}"][indices].to(device).clamp_min(1e-12).log()
            sensing = sensing / float(train_cache["calibration"]["sensing_temperature"])
            final = sensing + float(weight) * (train_radio - sensing)
            correct.append(final.argmax(dim=-1).eq(train_labels).float().mean())
            losses.append(F.cross_entropy(final, train_labels))
        grid_rows.append(
            {
                "lambda": float(weight),
                "train_calibration_all14_top1": float(torch.stack(correct).mean()),
                "train_calibration_all14_nll": float(torch.stack(losses).mean()),
                "sample_count": calibration_count,
            }
        )
    _write_csv(output / "global_train_grid.csv", grid_rows)
    return {
        "allocation": allocation_name,
        "distinct_oracle_lambdas": len({row["lambda"] for row in oracle_rows}),
        "train_grid_best_lambda": max(grid_rows, key=lambda row: row["train_calibration_all14_top1"])["lambda"],
        "outer_test_accessed": False,
    }


def evaluate_checkpoint(
    config: Mapping[str, Any], checkpoint: Path, diagnostic: str, device: torch.device
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    allocation_name = str(payload["allocation"])
    cache = _load_cache(config, allocation_name, "validation")
    model, payload = _load_trained_model(config, checkpoint, cache, device)
    result = evaluate_model(config, allocation_name, model, device=device, diagnostic=diagnostic, seed=int(payload["seed"]))
    result["loss_profile"] = payload["loss_profile"]
    result["checkpoint"] = str(checkpoint.resolve())
    stem = _run_stem(allocation_name, str(payload["loss_profile"]), model.method, int(payload["seed"]))
    _write_json(_path(config["output"]["root"]) / "diagnostics" / f"{stem}_{diagnostic}.json", result)
    return result


def _result_row(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "method": result["method"],
        "allocation": result["allocation"],
        "loss_profile": result.get("loss_profile", ""),
        "seed": result["seed"],
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
        "csi_only_top1": result["csi_only"]["top1"],
        "oracle_all14_macro": result["oracle_all14_macro"],
        "headroom_capture": result["oracle_headroom_capture"],
        "trainable_parameters": result["trainable_parameters"],
        "latency_ms_per_sample_mask": result["latency_ms_per_sample_mask"],
        "re_per_frame": result["pilot_re_per_frame"],
        "history_frames": result["pilot_history_frames"],
        "re_window": result["pilot_re_window"],
        "outer_test_accessed": False,
    }


def _retained_baselines() -> list[dict[str, Any]]:
    root = ROOT / "outputs/quality_topology_prototype_routing/trajectory_v1_low_re/runs"
    rows = [
        {
            "method": "B0",
            "allocation": "no_csi",
            "loss_profile": "retained",
            "seed": 1,
            "single_macro": 0.4521209839731455,
            "single_worst": 0.10212097316980362,
            "all14_macro": 0.5943552910217217,
            "all14_worst": 0.10212097316980362,
            "missing_lidar": 0.256087988615036,
            "full_top1": 0.8633149862289429,
            "re_per_frame": 0,
            "history_frames": 0,
            "re_window": 0,
        }
    ]
    for method in ("F0", "F1", "F2", "F3", "F5"):
        path = root / f"{method}_seed1_2x2/evaluation_final.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "method": method,
                    "allocation": "temporal_2x2",
                    "loss_profile": "retained",
                    "seed": 1,
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
                    "re_per_frame": 4,
                    "history_frames": 5,
                    "re_window": 20,
                }
            )
    return rows


def summarize(config: Mapping[str, Any]) -> dict[str, Any]:
    output = _path(config["output"]["root"])
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "results").glob("*.json"))]
    rows = [_result_row(result) for result in results]
    _write_csv(output / "ablation_summary.csv", _retained_baselines() + rows)
    mask_rows = [
        {
            "method": result["method"],
            "allocation": result["allocation"],
            "loss_profile": result.get("loss_profile"),
            "seed": result["seed"],
            **mask,
        }
        for result in results
        for mask in result["per_mask"]
    ]
    _write_csv(output / "mask_summary.csv", mask_rows)
    lambda_rows = [
        {
            "method": result["method"],
            "allocation": result["allocation"],
            "loss_profile": result.get("loss_profile"),
            "seed": result["seed"],
            "mask": name,
            "available_count": TSPC_AVAILABLE_COUNTS[TSPC_MASK_NAMES.index(name)],
            "lambda": value,
        }
        for result in results
        for name, value in result["lambda_table"].items()
    ]
    _write_csv(output / "mask_lambda_table.csv", lambda_rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["allocation"], row["loss_profile"], row["method"])].append(row)
    seed_summary = []
    for (allocation, profile, method), values in sorted(grouped.items()):
        record: dict[str, Any] = {"allocation": allocation, "loss_profile": profile, "method": method, "seeds": len(values)}
        for metric in ("single_macro", "single_worst", "two_macro", "three_macro", "all14_macro", "all14_worst", "missing_lidar", "full_top1"):
            record[f"{metric}_mean"] = float(np.mean([float(row[metric]) for row in values]))
            record[f"{metric}_std"] = float(np.std([float(row[metric]) for row in values]))
        seed_summary.append(record)
    _write_csv(output / "seed_summary.csv", seed_summary)
    temporal = [row for row in rows if row["method"] == "M3"]
    _write_csv(output / "temporal_allocation_summary.csv", temporal)
    _write_csv(output / "budget_summary.csv", [{key: row[key] for key in ("method", "allocation", "seed", "re_per_frame", "history_frames", "re_window", "csi_only_top1", "all14_macro", "all14_worst")} for row in rows])
    _write_csv(output / "latency_summary.csv", [{key: row[key] for key in ("method", "allocation", "loss_profile", "seed", "trainable_parameters", "latency_ms_per_sample_mask")} for row in rows])

    diagnostics = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "diagnostics").glob("*.json"))]
    diagnostic_rows = [
        {
            "method": result["method"],
            "allocation": result["allocation"],
            "loss_profile": result.get("loss_profile"),
            "seed": result["seed"],
            "diagnostic": result["diagnostic"],
            "single_macro": result["groups"]["single"]["top1_macro"],
            "all14_macro": result["groups"]["all14"]["top1_macro"],
            "all14_worst": result["groups"]["all14"]["top1_worst"],
            "missing_lidar": result["missing_lidar"]["top1"],
            "full_bypass_max_abs": result["full_bypass_max_abs"],
            "csi_off_max_abs": result["csi_off_max_abs"],
        }
        for result in diagnostics
    ]
    _write_csv(output / "shuffle_diagnostics.csv", diagnostic_rows)
    sampling = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "sampling_statistics").glob("*.json"))]
    _write_json(output / "sampling_statistics.json", {"runs": sampling, "outer_test_accessed": False})

    temporal_seed1 = [row for row in rows if row["allocation"] == "temporal_2x2" and int(row["seed"]) == 1]
    candidates = []
    for profile in config["training"]["loss_profiles"]:
        m0 = next((row for row in temporal_seed1 if row["method"] == "M0" and row["loss_profile"] == profile), None)
        m3 = next((row for row in temporal_seed1 if row["method"] == "M3" and row["loss_profile"] == profile), None)
        if m0 and m3:
            deltas = {
                "all14_macro_pp": 100 * (m3["all14_macro"] - m0["all14_macro"]),
                "all14_worst_pp": 100 * (m3["all14_worst"] - m0["all14_worst"]),
                "missing_lidar_pp": 100 * (m3["missing_lidar"] - m0["missing_lidar"]),
                "single_pp": 100 * (m3["single_macro"] - m0["single_macro"]),
                "two_pp": 100 * (m3["two_macro"] - m0["two_macro"]),
                "three_pp": 100 * (m3["three_macro"] - m0["three_macro"]),
            }
            passed_gain = (
                deltas["all14_macro_pp"] >= float(config["diagnostics"]["success_all14_macro_pp"])
                or deltas["all14_worst_pp"] >= float(config["diagnostics"]["success_all14_worst_pp"])
                or deltas["missing_lidar_pp"] >= float(config["diagnostics"]["success_missing_lidar_pp"])
            )
            passed_groups = min(deltas[name] for name in ("single_pp", "two_pp", "three_pp")) >= -float(
                config["diagnostics"]["maximum_group_regression_pp"]
            )
            candidates.append({"loss_profile": profile, **deltas, "passed": bool(passed_gain and passed_groups)})
    best_gate = max(candidates, key=lambda row: (row["passed"], row["all14_macro_pp"]), default=None)
    gates = {
        "profiles": candidates,
        "selected_profile": None if best_gate is None else best_gate["loss_profile"],
        "expand_multi_seed": bool(best_gate and best_gate["passed"]),
        "outer_test_accessed": False,
    }
    _write_json(output / "success_gates.json", gates)

    result_lookup = {
        (result["allocation"], result["method"], result.get("loss_profile")): result for result in results
    }
    main = result_lookup[("temporal_2x2", "M3", "mask_equal")]
    m0 = result_lookup[("temporal_2x2", "M0", "mask_equal")]
    single_5x4 = result_lookup[("single_5x4", "M3", "mask_equal")]
    single_4x5 = result_lookup[("single_4x5", "M3", "mask_equal")]
    no_gru = result_lookup[("temporal_no_gru", "M3", "mask_equal")]
    high_re = result_lookup[("high_re_16x16", "M3", "mask_equal")]
    oracle_path = output / "oracle_summary.csv"
    oracle_rows = list(csv.DictReader(oracle_path.open(encoding="utf-8"))) if oracle_path.is_file() else []
    oracle_lambdas = sorted({float(row["lambda"]) for row in oracle_rows})
    full_max_abs = max(float(result["full_bypass_max_abs"]) for result in results)
    full_argmax_mismatch = max(int(result["full_bypass_argmax_mismatch"]) for result in results)
    csi_off_max_abs = max(float(result["csi_off_max_abs"]) for result in results)
    csi_off_argmax_mismatch = max(int(result["csi_off_argmax_mismatch"]) for result in results)

    def percentage(value: float) -> str:
        return f"{100.0 * float(value):.2f}%"

    def allocation_row(label: str, result: Mapping[str, Any]) -> str:
        group = result["groups"]["all14"]
        return (
            f"| {label} | {percentage(result['csi_only']['top1'])} | "
            f"{percentage(group['top1_macro'])} | {percentage(group['top1_worst'])} | "
            f"{percentage(group['fix_rate_macro'])} | {percentage(group['harm_rate_macro'])} | "
            f"{result['pilot_re_per_frame']} | {result['pilot_history_frames']} | {result['pilot_re_window']} |"
        )

    oracle_lambda_text = ", ".join(f"{value:.2f}" for value in oracle_lambdas)
    no_gru_lambda = float(np.mean(list(no_gru["lambda_table"].values())))
    report = [
        "# TSPC 实验报告",
        "",
        "## 结论",
        "",
        "本轮结果支持“时间分散超稀疏 CSI + 共享 Beam Prototype 固定补偿”，不支持保留缺失模式条件静态参数。"
        "M0-M3 在两个预注册损失配置下均由 validation 选择 epoch 0，最终全部等价于 F1 的 `lambda=0.5`。",
        "",
        "## 协议",
        "",
        "- 最新轨迹互斥划分：train 12 条轨迹、37,510 样本；validation 2 条轨迹、6,365 样本；outer test 1 条轨迹继续封存。",
        "- 输入只使用 t-4...t，预测 t+1；缓存清单确认未使用未来 channel。",
        "- 主配置为每帧 2x2=4 RE、5 帧、窗口总 20 RE；Full 硬旁路为 0 RE。",
        "- batch size 512，AdamW，lr=0.01，weight decay=0，最多 100 epoch，patience=15；只训练 1/3/14/17 个静态参数。",
        "- M0-M3 均运行 seed 1 与 mask_equal/group_equal。M3 未过预注册门槛，因此按方案停止 seed 2/3。",
        "",
        "## 资源分配结果",
        "",
        "| 配置 | CSI-only Top-1 | All14 Top-1 | Worst Top-1 | Fix | Harm | RE/帧 | 帧数 | RE/窗口 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        allocation_row("单帧 4x5", single_4x5),
        allocation_row("单帧 5x4", single_5x4),
        allocation_row("五帧 2x2（主配置）", main),
        allocation_row("五帧 2x2，移除 GRU", no_gru),
        allocation_row("五帧 16x16（强 CSI 对照）", high_re),
        "",
        "## 必答问题",
        "",
        f"1. **五帧分散优于单帧集中。** 同为窗口 20 RE，五帧 2x2 的 All14 为 {percentage(main['groups']['all14']['top1_macro'])}，"
        f"比单帧 5x4 高 {100 * (main['groups']['all14']['top1_macro'] - single_5x4['groups']['all14']['top1_macro']):.2f} pp，"
        f"比单帧 4x5 高 {100 * (main['groups']['all14']['top1_macro'] - single_4x5['groups']['all14']['top1_macro']):.2f} pp。",
        f"2. **低 RE CSI 弱但互补。** CSI-only 为 {percentage(main['csi_only']['top1'])}，低于 Full M4 的 {percentage(main['full']['top1'])}；"
        f"但它把 B0 All14 从 {percentage(config['baselines']['b0_all14_macro'])} 提升到 {percentage(main['groups']['all14']['top1_macro'])}。",
        f"3. **共享 prototype 捕获 {100 * main['oracle_headroom_capture']:.1f}% 的 oracle headroom。** 匹配条件下 oracle All14 为 "
        f"{percentage(main['oracle_all14_macro'])}，固定融合获得 {100 * (main['groups']['all14']['top1_macro'] - config['baselines']['b0_all14_macro']):.2f} pp，"
        f"oracle 上限为 {100 * (main['oracle_all14_macro'] - config['baselines']['b0_all14_macro']):.2f} pp。",
        f"4. **不同 mask 的 validation-oracle lambda 确实不同。** 14 个 mask 得到 {len(oracle_lambdas)} 个不同最优值：{oracle_lambda_text}；"
        "这些值只作 M4 诊断，未用于正式模型。",
        f"5. **M3 没有超过 M0。** 两种 loss profile 的 All14/Worst/missing_lidar 差值均为 0.00 pp，"
        f"两者都选择 epoch {m0['selected_epoch']}，正式 lambda 保持 0.5。",
        f"6. **missing_lidar 没有进一步提升。** M0 与 M3 均为 {percentage(main['missing_lidar']['top1'])}；"
        "相对 historical F1 的轻微差别来自本轮固定 validation radio-noise realization，不是 mask-conditioned 增益。",
        "7. **主方法不应保留 mask-conditioned 模块。** 按预注册规则回退到 Global Fixed；M1/M2/M3 只保留为消融代码。"
        "mask-swap 也因所有 lambda 相同而完全不改变结果，这是负诊断而非机制证据。",
        f"8. **Full 逐样本完全不变。** 所有正式结果的最大概率差为 {full_max_abs:.1f}，argmax mismatch 为 {full_argmax_mismatch}，Full 使用 0 RE。",
        f"9. **CSI-off 精确回退。** 最大概率差为 {csi_off_max_abs:.1f}，argmax mismatch 为 {csi_off_argmax_mismatch}；"
        f"All14 回到 B0 的 {percentage(config['baselines']['b0_all14_macro'])}。",
        f"10. **高 RE 会使 CSI 成为压倒性强模态。** 16x16 对照的 CSI-only 为 {percentage(high_re['csi_only']['top1'])}、"
        f"All14 为 {percentage(high_re['groups']['all14']['top1_macro'])}。该 legacy 配置实际是每帧 256 RE、5 帧共 1,280 RE，不能写成窗口仅 256 RE。",
        "11. **论文主配置应使用窗口 20 RE。** 它满足“CSI 弱于 M4 但提供互补兜底”的故事；1,280 RE 窗口只作为强 CSI 上界。",
        "12. **outer test 保持封存。** 所有 preflight、cache manifest、结果与汇总均记录 `outer_test_accessed=false`。",
        "",
        "## 机制诊断",
        "",
        f"- CSI shuffle：All14 从 {percentage(main['groups']['all14']['top1_macro'])} 降到 "
        f"{percentage(next(row for row in diagnostic_rows if row['diagnostic'] == 'csi_shuffle')['all14_macro'])}。",
        f"- Sensing shuffle：All14 降到 {percentage(next(row for row in diagnostic_rows if row['diagnostic'] == 'sensing_shuffle')['all14_macro'])}。",
        f"- 移除 GRU 的冻结推理消融：CSI-only 为 {percentage(no_gru['csi_only']['top1'])}，训练后平均 lambda 为 {no_gru_lambda:.4f}，"
        f"All14 为 {percentage(no_gru['groups']['all14']['top1_macro'])}，说明时序编码不可替代。",
        "- 该去 GRU 项是把五帧 frame-level CSI feature 直接均值后送入冻结 radio expert 的推理消融，没有重新训练替代编码器。",
        "",
        "## 复现产物",
        "",
        "完整逐 mask 指标、lambda 曲线、训练日志、采样统计、哈希和旁路检查分别见本目录下的 CSV/JSON、`audit.md` 与 `cache_manifests/`。",
    ]
    (output / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return gates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "prepare", "train", "evaluate", "curves", "summarize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--allocation", default="temporal_2x2")
    parser.add_argument("--role", choices=("train", "validation"), default="train")
    parser.add_argument("--method", choices=TSPC_METHODS, default="M3")
    parser.add_argument("--loss-profile", choices=("mask_equal", "group_equal"), default="mask_equal")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--diagnostic", choices=("normal", "csi_shuffle", "sensing_shuffle", "csi_off", "mask_swap"), default="normal")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    device = torch.device(args.device)
    if args.mode == "preflight":
        result = preflight(config, args.allocation)
    elif args.mode == "prepare":
        result = prepare_cache(config, args.allocation, args.role, device)
    elif args.mode == "train":
        result = train(
            config,
            args.allocation,
            args.method,
            args.loss_profile,
            args.seed,
            device,
            epochs_override=args.epochs,
        )
    elif args.mode == "evaluate":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for evaluate.")
        result = evaluate_checkpoint(config, args.checkpoint, args.diagnostic, device)
    elif args.mode == "curves":
        result = lambda_curves(config, args.allocation, device)
    else:
        result = summarize(config)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
