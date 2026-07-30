#!/usr/bin/env python3
"""Local train/evaluate workflow for TSPC-V2; never constructs an outer-test loader."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from kd_sensing.baselines.full_pool_common import sha256_file
from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.models.tspc_v2 import TSPCV2LossConfig, TSPCV2Model, TSPCV2ModelConfig, tspc_v2_losses
from kd_sensing.models.tspc_ablation_heads import expected_calibration_error

if __package__:
    from .run_csi_anchored_completion import validate_cache_record
    from .run_mmw_trajectory_baselines import ALL_PATTERNS
    from .run_sparse_pilot_recovery import _prediction_metrics
    from .run_sparse_pilot_trajectory_recovery import nested_frequency_indices
else:
    from run_csi_anchored_completion import validate_cache_record
    from run_mmw_trajectory_baselines import ALL_PATTERNS
    from run_sparse_pilot_recovery import _prediction_metrics
    from run_sparse_pilot_trajectory_recovery import nested_frequency_indices


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/tspc_v2/stage_c_joint.yaml"
MASK_NAMES = tuple(name for name in ALL_PATTERNS if name != "full")
MASK_VALUES = torch.tensor([ALL_PATTERNS[name] for name in MASK_NAMES], dtype=torch.bool)
MASK_COUNTS = {name: int(sum(ALL_PATTERNS[name])) for name in MASK_NAMES}
GROUP_NAMES = {1: "single", 2: "two", 3: "three"}
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
    "nll",
    "ece",
)


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
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_config(path: Path) -> dict[str, Any]:
    config = safe_load_yaml(path.read_text(encoding="utf-8"))
    if config["protocol"].get("outer_test_enabled") is not False:
        raise ValueError("TSPC-V2 requires outer_test_enabled=false.")
    return config


def _autocast(device: torch.device, enabled: bool):
    """Use the repository's BF16 CUDA convention without affecting CPU runs."""

    if device.type == "cuda" and bool(enabled):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_record(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _load_m4(config: Mapping[str, Any], device: torch.device) -> TrajectoryBaselineModel:
    source = config["source"]
    payload = torch.load(_path(source["m4_checkpoint"]), map_location="cpu", weights_only=False)
    if payload.get("method") != ABTC_METHOD or payload.get("protocol_fingerprint") != config["protocol"]["fingerprint"]:
        raise ValueError("TSPC-V2 requires the published trajectory M4 checkpoint.")
    if payload.get("split_manifest_sha256") != config["protocol"]["split_manifest_sha256"]:
        raise ValueError("TSPC-V2 requires the published trajectory split identity.")
    model = TrajectoryBaselineModel(ABTC_METHOD, **payload.get("model_config", {})).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.prototype_bank is None or tuple(model.prototype_bank.prototypes.shape) != (64, 64):
        raise ValueError("TSPC-V2 requires the frozen [64,64] M4 prototype bank.")
    return model


def _forbidden_record_keys(record: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Reject future-channel/test payloads even when nested in a cache mapping."""

    forbidden = []
    for key, value in record.items():
        name = str(key)
        location = f"{prefix}.{name}" if prefix else name
        lowered = name.lower()
        if "future_channel" in lowered or "future_csi" in lowered or "outer_test" in lowered:
            forbidden.append(location)
        if isinstance(value, Mapping):
            forbidden.extend(_forbidden_record_keys(value, location))
    return sorted(forbidden)


def _load_role(config: Mapping[str, Any], role: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if role not in {"train", "validation"}:
        raise ValueError("TSPC-V2 only permits train or validation roles.")
    source = config["source"]
    feature = _load_record(_path(source[f"{role}_feature_cache"]))
    recovery = _load_record(_path(source[f"{role}_records"]))
    forbidden = _forbidden_record_keys(feature) + _forbidden_record_keys(recovery)
    if forbidden:
        raise ValueError(f"V2 records expose prohibited future/test keys: {sorted(set(forbidden))}.")
    validate_cache_record(feature, recovery, expected_count=int(config["protocol"][f"expected_{role}_samples"]))
    return feature, recovery


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    checksums = {
        "split_manifest": (config["protocol"]["split_manifest"], config["protocol"]["split_manifest_sha256"]),
        "m4_checkpoint": (source["m4_checkpoint"], source["m4_checkpoint_sha256"]),
        "train_feature_cache": (source["train_feature_cache"], source["train_feature_cache_sha256"]),
        "validation_feature_cache": (source["validation_feature_cache"], source["validation_feature_cache_sha256"]),
        "train_records": (source["train_records"], source["train_records_sha256"]),
        "validation_records": (source["validation_records"], source["validation_records_sha256"]),
        "codebook": (source["codebook"], source["codebook_sha256"]),
    }
    if "prepared_config_sha256" in source:
        checksums["prepared_config"] = (source["prepared_config"], source["prepared_config_sha256"])
    hashes = {name: sha256_file(_path(value)) for name, (value, _) in checksums.items()}
    hash_ok = all(hashes[name] == expected for name, (_, expected) in checksums.items())
    if not hash_ok:
        mismatches = {
            name: {"expected": expected, "actual": hashes[name]}
            for name, (_, expected) in checksums.items()
            if hashes[name] != expected
        }
        raise ValueError(f"TSPC-V2 source SHA256 mismatch before cache load: {mismatches}.")
    manifest = json.loads(_path(config["protocol"]["split_manifest"]).read_text(encoding="utf-8"))
    train_feature, train_recovery = _load_role(config, "train")
    validation_feature, validation_recovery = _load_role(config, "validation")
    train_groups = set(train_feature["trajectory_ids"])
    validation_groups = set(validation_feature["trajectory_ids"])
    checks = {
        "hashes": hash_ok,
        "protocol_id": manifest.get("protocol_id") == config["protocol"]["id"],
        "fingerprint": manifest.get("protocol_fingerprint") == config["protocol"]["fingerprint"],
        "outer_test_disabled": manifest.get("outer_test_enabled") is False,
        "outer_test_unaccessed": manifest.get("outer_test_accessed") is False,
        "train_count": int(manifest.get("train_window_count", -1)) == int(config["protocol"]["expected_train_samples"]),
        "validation_count": int(manifest.get("validation_window_count", -1))
        == int(config["protocol"]["expected_validation_samples"]),
        "train_trajectory_count": int(manifest.get("train_group_count", -1))
        == int(config["protocol"]["expected_train_trajectories"]),
        "validation_trajectory_count": int(manifest.get("validation_group_count", -1))
        == int(config["protocol"]["expected_validation_trajectories"]),
        "trajectory_disjoint": not bool(train_groups & validation_groups),
        "feature_recovery_train_identity": train_feature["sample_ids"] == train_recovery["sample_ids"],
        "feature_recovery_validation_identity": validation_feature["sample_ids"] == validation_recovery["sample_ids"],
        "mother_shape": tuple(train_recovery["candidate_history"].shape[1:]) == (5, 32, 16)
        and tuple(validation_recovery["candidate_history"].shape[1:]) == (5, 32, 16),
    }
    if not all(checks.values()):
        raise ValueError(f"TSPC-V2 preflight failed: {checks}.")
    result = {
        "status": "passed",
        "checks": checks,
        "hashes": hashes,
        "train_samples": len(train_feature["sample_ids"]),
        "validation_samples": len(validation_feature["sample_ids"]),
        "train_trajectories": len(train_groups),
        "validation_trajectories": len(validation_groups),
        "future_channel_used_as_input": False,
        "test_loader_constructed": False,
        "outer_test_accessed": False,
    }
    output = _path(config["output"]["root"])
    _write_json(output / "preflight.json", result)
    for directory in ("configs", "resolved_configs", "checkpoints", "logs", "seed_results"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    resolved_yaml = yaml.safe_dump(dict(config), sort_keys=False)
    (output / "configs/base.yaml").write_text(resolved_yaml, encoding="utf-8")
    (output / "resolved_configs/base.yaml").write_text(resolved_yaml, encoding="utf-8")
    return result


def _parse_budget(value: str) -> tuple[int, int]:
    try:
        patterns, frequencies = (int(part) for part in str(value).lower().split("x", 1))
    except ValueError as error:
        raise ValueError(f"Pilot budget must be formatted as MxK, got {value!r}.") from error
    if patterns <= 0 or frequencies <= 0:
        raise ValueError("Pilot budget dimensions must be positive.")
    return patterns, frequencies


def _method_spec(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    methods = config.get("methods", {})
    if method not in methods:
        raise ValueError(f"Unknown TSPC-V2 method {method!r}.")
    override = methods[method]
    if bool(override.get("external_baseline", False)):
        raise ValueError(
            f"{method} is the immutable legacy B0 baseline; use tools/run_tspc_final_ablations.py rather than V2 training."
        )
    model_values = dict(config["model"])
    model_values.update(override.get("model", {}))
    accepted = set(TSPCV2ModelConfig.__dataclass_fields__)
    unknown = set(model_values) - accepted
    if unknown:
        raise ValueError(f"Unknown TSPCV2ModelConfig fields: {sorted(unknown)}.")
    pilot_values = dict(config["pilot"])
    pilot_values.update(override.get("pilot", {}))
    patterns, frequencies = _parse_budget(pilot_values["budget"])
    history_frames = int(pilot_values.get("history_frames", 5))
    if history_frames not in {1, int(model_values["history_length"])}:
        raise ValueError("V2 supports either one actual CSI frame or the configured full history.")
    if patterns > int(pilot_values["mother_patterns"]) or frequencies > int(pilot_values["mother_frequencies"]):
        raise ValueError("V2 pilot budget exceeds the mother observation.")
    return {
        "method": str(method),
        "stage": str(override.get("stage", config["training"]["stage"])),
        "model_config": TSPCV2ModelConfig(**model_values),
        "loss_config": TSPCV2LossConfig(**dict(config["loss"]) | dict(override.get("loss", {}))),
        "budget": str(pilot_values["budget"]),
        "patterns": patterns,
        "frequencies": frequencies,
        "history_frames": history_frames,
        "csi_enabled": bool(override.get("csi_enabled", True)),
        "prototype_label": str(model_values["prototype_mode"]),
        "residual_label": str(model_values["residual_mode"]),
    }


def _frequency_metadata(config: Mapping[str, Any], frequencies: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = safe_load_yaml(_path(config["source"]["prepared_config"]).read_text(encoding="utf-8"))
    positions = torch.as_tensor(prepared["runtime"]["frequency_positions_hz"], dtype=torch.float32, device=device)
    selected_ids = nested_frequency_indices(len(positions), int(frequencies)).to(device)
    return positions.index_select(0, selected_ids), selected_ids


def _candidate_history_for_missing_rows(
    recovery: Mapping[str, Any],
    indices: torch.Tensor,
    availability: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Materialize CSI only for non-Full rows, leaving Full rows inert.

    The model already excludes Full rows from its CSI encoder.  Keeping the
    cache access split here makes that boundary true for mixed batches too.
    """

    source = torch.as_tensor(recovery["candidate_history"])
    if not torch.is_complex(source) or tuple(source.shape[1:]) != (5, 32, 16):
        raise ValueError("Recovery candidate_history must be complex [N,5,32,16].")
    batch = len(indices)
    candidates = torch.zeros((batch, *source.shape[1:]), dtype=source.dtype, device=device)
    missing_rows = (~availability.all(dim=1)).nonzero(as_tuple=False).squeeze(1)
    if missing_rows.numel():
        source_rows = indices.index_select(0, missing_rows.detach().cpu()).to(device="cpu", dtype=torch.long)
        selected = source.index_select(0, source_rows).to(device, non_blocking=True)
        candidates.index_copy_(0, missing_rows, selected)
    return candidates


def _build_pilot_view(
    candidate_history: torch.Tensor,
    *,
    patterns: int,
    frequencies: int,
    history_frames: int,
    frequency_positions: torch.Tensor,
    frequency_ids: torch.Tensor,
    csi_enabled: bool,
) -> dict[str, torch.Tensor]:
    """Select only historical pilots and pad a one-frame control to the V2 history."""

    candidates = torch.as_tensor(candidate_history)
    if not torch.is_complex(candidates) or candidates.ndim != 4 or tuple(candidates.shape[1:]) != (5, 32, 16):
        raise ValueError("candidate_history must be complex [B,5,32,16].")
    batch, history, _, _ = candidates.shape
    if history != 5:
        raise ValueError("TSPC-V2 is fixed to a five-frame sensing history.")
    selected_ids = nested_frequency_indices(candidates.shape[-1], int(frequencies)).to(candidates.device)
    selected = candidates[:, :, :patterns].index_select(-1, selected_ids)
    valid = torch.ones_like(selected, dtype=torch.bool)
    if int(history_frames) == 1:
        padded = torch.zeros_like(selected)
        padded[:, -1] = selected[:, -1]
        selected = padded
        valid[:, :-1] = False
    if not csi_enabled:
        selected = torch.zeros_like(selected)
        valid = torch.zeros_like(valid)
    pattern_ids = torch.arange(patterns, device=candidates.device, dtype=torch.long).view(1, 1, patterns)
    pattern_ids = pattern_ids.expand(batch, history, -1)
    return {
        "clean_observations": selected,
        "pilot_mask": valid,
        "pattern_ids": pattern_ids,
        "frequency_positions": frequency_positions,
        "frequency_ids": frequency_ids,
    }


def _noisy_pilots(
    clean_observations: torch.Tensor,
    valid_mask: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    dropout: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(clean_observations)
    valid = torch.as_tensor(valid_mask, device=values.device, dtype=torch.bool).clone()
    snr = torch.as_tensor(snr_db, device=values.device, dtype=values.real.dtype)
    if snr.shape != values.shape[:2]:
        raise ValueError("snr_db must have shape [B,T].")
    power = values.abs().square().mean(dim=(-2, -1), keepdim=True)
    variance = power / torch.pow(10.0, snr[..., None, None] / 10.0)
    scale = (variance / 2.0).sqrt()
    noise = torch.complex(
        torch.randn(values.shape, device=values.device, generator=generator),
        torch.randn(values.shape, device=values.device, generator=generator),
    ) * scale
    if float(dropout):
        valid &= torch.rand(values.shape, device=values.device, generator=generator) >= float(dropout)
    return (values + noise) * valid, valid


def _stratified_indices(labels: torch.Tensor, limit: int | None, seed: int) -> torch.Tensor:
    total = len(labels)
    if limit is None or int(limit) >= total:
        return torch.arange(total)
    generator = torch.Generator().manual_seed(int(seed))
    classes = torch.unique(labels).tolist()
    quota = max(1, int(limit) // max(len(classes), 1))
    selected = []
    for label in classes:
        members = labels.eq(int(label)).nonzero(as_tuple=False).squeeze(1)
        selected.append(members[torch.randperm(len(members), generator=generator)[:quota]])
    result = torch.cat(selected)
    if len(result) < int(limit):
        remaining = torch.ones(total, dtype=torch.bool)
        remaining[result] = False
        values = remaining.nonzero(as_tuple=False).squeeze(1)
        result = torch.cat((result, values[torch.randperm(len(values), generator=generator)[: int(limit) - len(result)]]))
    return result[torch.randperm(len(result), generator=generator)[: int(limit)]]


def _metric_row(probability: torch.Tensor, labels: torch.Tensor, power: torch.Tensor, base: torch.Tensor | None) -> dict[str, float]:
    row = dict(_prediction_metrics(probability, labels, power, base))
    row["nll"] = float(F.nll_loss(probability.clamp_min(1e-12).log(), labels).item())
    row["ece"] = expected_calibration_error(probability, labels)
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result = {}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows if math.isfinite(float(row[name]))]
        result[f"{name}_macro"] = float(np.mean(values)) if values else float("nan")
        result[f"{name}_worst"] = (
            float(np.max(values)) if name in {"mae", "beam_loss_db", "harm_rate", "nll", "ece"} else float(np.min(values))
        ) if values else float("nan")
    return result


def _forward_batch(
    model: TSPCV2Model,
    m4: TrajectoryBaselineModel,
    feature: Mapping[str, Any],
    recovery: Mapping[str, Any],
    indices: torch.Tensor,
    availability: torch.Tensor,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    generator: torch.Generator,
    snr_db: float | torch.Tensor,
    dropout: float,
    force_csi_off: bool = False,
    frequency_metadata: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    tokens = feature["token_sequence"].index_select(0, indices).to(device, non_blocking=True)
    full_probability = feature["p_full"].index_select(0, indices).to(device, non_blocking=True)
    availability = torch.as_tensor(availability, device=device, dtype=torch.bool)
    if availability.ndim == 1:
        availability = availability[None].expand(len(indices), -1)
    if availability.shape != (len(indices), 4):
        raise ValueError("availability must resolve to [B,4].")
    if bool(availability.all()):
        with _autocast(device, bool(config["training"].get("amp", False))):
            return model(
                tokens,
                availability,
                shared_prototype_bank=m4.prototype_bank,
                full_probability=full_probability,
            )
    csi_active = bool(spec["csi_enabled"]) and not force_csi_off
    if csi_active:
        candidates = _candidate_history_for_missing_rows(recovery, indices, availability, device=device)
    else:
        # A/C0 and explicit CSI-off diagnostics do not even index the CSI cache.
        candidates = torch.zeros((len(indices), 5, 32, 16), dtype=torch.complex64, device=device)
    positions, frequency_ids = frequency_metadata or _frequency_metadata(config, int(spec["frequencies"]), device)
    pilot = _build_pilot_view(
        candidates,
        patterns=int(spec["patterns"]),
        frequencies=int(spec["frequencies"]),
        history_frames=int(spec["history_frames"]),
        frequency_positions=positions,
        frequency_ids=frequency_ids,
        csi_enabled=csi_active,
    )
    if isinstance(snr_db, torch.Tensor):
        snr = snr_db.to(device=device, dtype=tokens.dtype)
        if tuple(snr.shape) == (len(indices),):
            snr = snr[:, None].expand(-1, 5)
    else:
        snr = torch.full((len(indices), 5), float(snr_db), device=device, dtype=tokens.dtype)
    observations, valid = _noisy_pilots(
        pilot["clean_observations"],
        pilot["pilot_mask"],
        snr,
        dropout=float(dropout),
        generator=generator,
    )
    with _autocast(device, bool(config["training"].get("amp", False))):
        return model(
            tokens,
            availability,
            shared_prototype_bank=m4.prototype_bank,
            pilot_observations=observations,
            pattern_ids=pilot["pattern_ids"],
            frequency_positions=pilot["frequency_positions"],
            frequency_ids=pilot["frequency_ids"],
            pilot_mask=valid,
            snr_db=snr,
            full_probability=full_probability,
        )


@torch.inference_mode()
def evaluate_model(
    model: TSPCV2Model,
    m4: TrajectoryBaselineModel,
    feature: Mapping[str, Any],
    recovery: Mapping[str, Any],
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    device: torch.device,
    indices: torch.Tensor,
    seed: int,
    snr_db: float | None = None,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """Evaluate all 14 masks with no model state updates."""

    model.eval()
    evaluation_batch_size = int(config["training"]["evaluation_batch_size"])
    validation_snr = float(config["pilot"]["validation_snr_db"] if snr_db is None else snr_db)
    frequency_metadata = _frequency_metadata(config, int(spec["frequencies"]), device)
    labels_all = recovery["labels_future"].index_select(0, indices)
    power_all = recovery["future_beam_power"].index_select(0, indices)
    per_mask = []
    elapsed = 0.0
    pilot_total = 0
    pilot_samples = 0
    for mask_index, mask_name in enumerate(MASK_NAMES):
        availability = MASK_VALUES[mask_index]
        final_chunks = []
        sensing_chunks = []
        re_chunks = []
        generator = torch.Generator(device=device).manual_seed(
            int(config["pilot"]["validation_noise_seed"]) + 10_000 + int(mask_index)
        )
        started = time.monotonic()
        for start in range(0, len(indices), evaluation_batch_size):
            current = indices[start : start + evaluation_batch_size]
            output = _forward_batch(
                model,
                m4,
                feature,
                recovery,
                current,
                availability,
                spec,
                config,
                device=device,
                generator=generator,
                snr_db=validation_snr,
                dropout=float(dropout),
                frequency_metadata=frequency_metadata,
            )
            final_chunks.append(output["final_probability"].float().cpu())
            sensing_chunks.append(torch.softmax(output["sensing_evidence"].float(), dim=-1).cpu())
            re_chunks.append(output["pilot_re_window"].cpu())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.monotonic() - started
        probability = torch.cat(final_chunks)
        sensing_probability = torch.cat(sensing_chunks)
        re_window = torch.cat(re_chunks)
        final_correct = probability.argmax(dim=-1).eq(labels_all)
        sensing_correct = sensing_probability.argmax(dim=-1).eq(labels_all)
        pilot_total += int(re_window.sum())
        pilot_samples += len(re_window)
        row = {
            "mask": mask_name,
            "available_count": MASK_COUNTS[mask_name],
            **_metric_row(probability, labels_all, power_all, sensing_probability),
            "sensing_top1": float(sensing_probability.argmax(dim=-1).eq(labels_all).float().mean()),
            "csi_only_top1": float("nan"),
            "oracle_top1": float((final_correct | sensing_correct).float().mean()),
            "pilot_re_window_actual_mean": float(re_window.float().mean()),
        }
        per_mask.append(row)
    groups = {
        GROUP_NAMES[count]: _aggregate([row for row in per_mask if row["available_count"] == count])
        for count in (1, 2, 3)
    }
    groups["all14"] = _aggregate(per_mask)

    full_probability_chunks = []
    reference_chunks = []
    for start in range(0, len(indices), evaluation_batch_size):
        current = indices[start : start + evaluation_batch_size]
        output = _forward_batch(
            model,
            m4,
            feature,
            recovery,
            current,
            torch.ones(len(current), 4, dtype=torch.bool),
            spec,
            config,
            device=device,
            generator=torch.Generator(device=device).manual_seed(1),
            snr_db=validation_snr,
            dropout=0.0,
            frequency_metadata=frequency_metadata,
        )
        full_probability_chunks.append(output["final_probability"].cpu())
        reference_chunks.append(feature["p_full"].index_select(0, current).float())
    full_probability = torch.cat(full_probability_chunks)
    full_reference = torch.cat(reference_chunks)
    full_metric = _metric_row(full_probability, labels_all, power_all, full_reference)

    csi_off_evidence_max_abs = 0.0
    csi_off_probability_max_abs = 0.0
    csi_off_mismatch = 0
    for mask_index, _ in enumerate(MASK_NAMES):
        availability = MASK_VALUES[mask_index]
        for start in range(0, len(indices), evaluation_batch_size):
            current = indices[start : start + evaluation_batch_size]
            output = _forward_batch(
                model,
                m4,
                feature,
                recovery,
                current,
                availability,
                spec,
                config,
                device=device,
                generator=torch.Generator(device=device).manual_seed(50_000 + mask_index + start),
                snr_db=validation_snr,
                dropout=0.0,
                force_csi_off=True,
                frequency_metadata=frequency_metadata,
            )
            csi_off_evidence_max_abs = max(
                csi_off_evidence_max_abs,
                float((output["final_evidence"] - output["sensing_evidence"]).abs().max().item()),
            )
            csi_off_probability_max_abs = max(
                csi_off_probability_max_abs,
                float(
                    (
                        output["final_probability"]
                        - torch.softmax(output["sensing_evidence"].float(), dim=-1)
                    )
                    .abs()
                    .max()
                    .item()
                ),
            )
            csi_off_mismatch += int(
                output["final_evidence"].argmax(dim=-1).ne(output["sensing_evidence"].argmax(dim=-1)).sum().item()
            )
    sensing_macro = float(np.mean([row["sensing_top1"] for row in per_mask]))
    return {
        "per_mask": per_mask,
        "groups": groups,
        "missing_lidar": next(row for row in per_mask if row["mask"] == "missing_lidar"),
        "full": full_metric,
        "full_probability_max_abs_diff": float((full_probability - full_reference).abs().max().item()),
        "full_argmax_mismatch": int(full_probability.argmax(dim=-1).ne(full_reference.argmax(dim=-1)).sum().item()),
        "full_pilot_re": 0,
        "csi_off_evidence_max_abs_diff": csi_off_evidence_max_abs,
        "csi_off_probability_max_abs_diff": csi_off_probability_max_abs,
        "csi_off_argmax_mismatch": csi_off_mismatch,
        "sensing_all14_macro": sensing_macro,
        "csi_only_top1": float("nan"),
        "oracle_all14_macro": float(np.mean([row["oracle_top1"] for row in per_mask])),
        "oracle_definition": "per_sample_upper_bound_between_sensing_and_compensated_final",
        "pilot_re_window_actual_mean": float(pilot_total / max(pilot_samples, 1)),
        "latency_ms_per_sample_mask": 1000.0 * elapsed / max(len(indices) * len(MASK_NAMES), 1),
        "snr_db": validation_snr,
        "dropout": float(dropout),
        "seed": int(seed),
        "outer_test_accessed": False,
    }


def _checkpoint_path(config: Mapping[str, Any], method: str, seed: int, name: str) -> Path:
    return _path(config["output"]["root"]) / "checkpoints" / f"{method}_seed{int(seed)}" / name


def _save_checkpoint(
    path: Path,
    model: TSPCV2Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    seed: int,
    epoch: int,
    metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_config": dict(spec["model_config"].__dict__),
            "loss_config": dict(spec["loss_config"].__dict__),
            "method": spec["method"],
            "stage": spec["stage"],
            "budget": spec["budget"],
            "history_frames": spec["history_frames"],
            "csi_enabled": spec["csi_enabled"],
            "seed": int(seed),
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "protocol_fingerprint": config["protocol"]["fingerprint"],
            "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
            "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
            "train_feature_cache_sha256": config["source"]["train_feature_cache_sha256"],
            "train_records_sha256": config["source"]["train_records_sha256"],
            "codebook_sha256": config["source"]["codebook_sha256"],
            "validation_noise_seed": config["pilot"]["validation_noise_seed"],
            "training_config": dict(config["training"]),
            "saved_at_utc": _utc_now(),
            "pid": os.getpid(),
            "git_commit": _git_commit(),
            "future_channel_used_as_input": False,
            "outer_test_accessed": False,
        },
        path,
    )


def _initialize_from(model: TSPCV2Model, checkpoint: Path, stage: str) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload)
    if not isinstance(state, Mapping):
        raise ValueError("Initialization checkpoint lacks a model state.")
    if str(stage).lower() == "stage_b":
        state = {name: value for name, value in state.items() if name.startswith("sensing.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Initialization checkpoint has unexpected tensors: {unexpected}.")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "missing_tensors": list(missing),
        "source_method": payload.get("method"),
        "source_stage": payload.get("stage"),
    }


def _load_checkpoint(checkpoint: Path, config: Mapping[str, Any], device: torch.device) -> tuple[TSPCV2Model, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("protocol_fingerprint") != config["protocol"]["fingerprint"]:
        raise ValueError("TSPC-V2 checkpoint protocol fingerprint mismatch.")
    if payload.get("split_manifest_sha256") != config["protocol"]["split_manifest_sha256"]:
        raise ValueError("TSPC-V2 checkpoint split-manifest identity mismatch.")
    if payload.get("m4_checkpoint_sha256") != config["source"]["m4_checkpoint_sha256"]:
        raise ValueError("TSPC-V2 checkpoint M4 identity mismatch.")
    if payload.get("outer_test_accessed") is not False:
        raise ValueError("TSPC-V2 checkpoint has invalid outer-test lineage.")
    model = TSPCV2Model(TSPCV2ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.configure_stage(str(payload["stage"]))
    model.eval()
    return model, payload


def train(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    started_at = _utc_now()
    preflight(config)
    spec = _method_spec(config, args.method)
    seed = int(args.seed)
    _set_seed(seed)
    device = torch.device(args.device)
    m4 = _load_m4(config, device)
    train_feature, train_recovery = _load_role(config, "train")
    validation_feature, validation_recovery = _load_role(config, "validation")
    batch_size = int(config["training"]["batch_size"])
    train_limit = args.limit
    if args.smoke and train_limit is None:
        train_limit = int(config["training"]["smoke_samples"])
    unique_train_indices = _stratified_indices(train_recovery["labels_future"], train_limit, 10_000 + seed)
    train_indices = unique_train_indices
    optimizer_updates_target = None
    if args.smoke:
        optimizer_updates_target = int(config["training"]["smoke_updates"])
        required = optimizer_updates_target * batch_size
        repetitions = math.ceil(required / len(unique_train_indices))
        train_indices = unique_train_indices.repeat(repetitions)[:required]
    validation_limit = args.validation_limit
    if validation_limit is None:
        validation_limit = int(config["training"]["smoke_validation_samples"]) if args.smoke else args.limit
    validation_indices = _stratified_indices(validation_recovery["labels_future"], validation_limit, 20_000 + seed)
    model = TSPCV2Model(spec["model_config"]).to(device)
    initialization = {"mode": "random"}
    if args.initialize_from is not None:
        initialization = _initialize_from(model, _path(args.initialize_from), str(spec["stage"]))
    model.configure_stage(str(spec["stage"]))
    frequency_metadata = _frequency_metadata(config, int(spec["frequencies"]), device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Selected V2 stage has no trainable parameters.")
    learning_rate = float(config["training"]["learning_rate"])
    sensing_scale = float(config["training"].get("sensing_learning_rate_scale", 1.0))
    optimizer_parameters: list[dict[str, Any]] | list[torch.nn.Parameter]
    if str(spec["stage"]).lower() == "stage_c" and sensing_scale != 1.0:
        sensing_ids = {id(parameter) for parameter in model.sensing.parameters() if parameter.requires_grad}
        sensing_parameters = [parameter for parameter in parameters if id(parameter) in sensing_ids]
        compensation_parameters = [parameter for parameter in parameters if id(parameter) not in sensing_ids]
        optimizer_parameters = [
            {"params": sensing_parameters, "lr": learning_rate * sensing_scale, "name": "sensing"},
            {"params": compensation_parameters, "lr": learning_rate, "name": "csi_compensation"},
        ]
    else:
        optimizer_parameters = parameters
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=learning_rate,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(args.epochs or config["training"]["max_epochs"])
    if args.smoke:
        epochs = int(config["training"]["smoke_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    generator = torch.Generator().manual_seed(30_000 + seed)
    noise_generator = torch.Generator(device=device).manual_seed(40_000 + seed)
    output = _path(config["output"]["root"])
    stem = f"{'smoke_' if args.smoke else ''}{spec['method']}_{spec['stage']}_seed{seed}"
    status_path = output / "status" / f"{stem}.json"
    run_identity = {
        "pid": os.getpid(),
        "logical_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": None,
        "method": spec["method"],
        "stage": spec["stage"],
        "seed": seed,
        "output_directory": str(output.resolve()),
        "start_time_utc": started_at,
        "exit_status": None,
        "status": "running",
        "outer_test_accessed": False,
    }
    _write_json(status_path, run_identity)
    resolved = dict(config) | {
        "run": {
            "method": spec["method"],
            "stage": spec["stage"],
            "budget": spec["budget"],
            "history_frames": spec["history_frames"],
            "pilot_re_per_frame": int(spec["patterns"]) * int(spec["frequencies"]),
            "pilot_re_window": int(spec["patterns"]) * int(spec["frequencies"]) * int(spec["history_frames"]),
            "seed": seed,
            "device": str(device),
            "train_samples": len(train_indices),
            "unique_train_samples": len(unique_train_indices),
            "validation_samples": len(validation_indices),
            "optimizer_updates_target": optimizer_updates_target,
            "sensing_learning_rate_scale": sensing_scale,
            "teacher_evidence_representation": "centered_log_probability" if spec["loss_config"].residual_regression_weight else None,
            "initialization": initialization,
            "git_commit": _git_commit(),
            "outer_test_accessed": False,
        }
    }
    resolved_path = output / "resolved_configs" / f"{stem}.yaml"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    run_identity["config"] = str(resolved_path.resolve())
    _write_json(status_path, run_identity)
    history_rows = []
    best = {"all14_macro": float("-inf"), "all14_worst": float("-inf")}
    patience = 0
    stop_reason = "smoke_updates" if args.smoke else "max_epochs"
    optimizer_updates = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        totals = defaultdict(float)
        seen = 0
        for start in range(0, len(order), batch_size):
            current = order[start : start + batch_size]
            mask_ids = (current + epoch * 17 + start // batch_size).remainder(len(MASK_NAMES))
            availability = MASK_VALUES.index_select(0, mask_ids)
            snr = torch.empty(len(current), 5, device=device).uniform_(
                float(config["pilot"]["train_snr_db_min"]),
                float(config["pilot"]["train_snr_db_max"]),
                generator=noise_generator,
            )
            output_batch = _forward_batch(
                model,
                m4,
                train_feature,
                train_recovery,
                current,
                availability,
                spec,
                config,
                device=device,
                generator=noise_generator,
                snr_db=snr,
                dropout=float(config["pilot"]["train_dropout"]),
                frequency_metadata=frequency_metadata,
            )
            labels = train_recovery["labels_future"].index_select(0, current).to(device, non_blocking=True)
            teacher_probability = train_feature["p_full"].index_select(0, current).to(device, non_blocking=True)
            teacher_evidence = None
            if spec["loss_config"].residual_regression_weight:
                # p_full is the only audited cache contract.  log(p_full) is
                # a canonical evidence representation, not claimed as the
                # original pre-softmax M4 logits; the loss centers it first.
                teacher_evidence = teacher_probability.float().clamp_min(1e-12).log()
            losses = tspc_v2_losses(
                output_batch,
                labels,
                config=spec["loss_config"],
                prototype_bank=model.prototype_bank_for_loss(m4.prototype_bank),
                teacher_probability=teacher_probability,
                teacher_evidence=teacher_evidence,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, float(config["training"]["gradient_clip_norm"])))
            optimizer.step()
            optimizer_updates += 1
            count = len(current)
            seen += count
            for name, value in losses.items():
                totals[name] += float(value.detach()) * count
            totals["gradient_norm"] += gradient_norm * count
        validation = evaluate_model(
            model,
            m4,
            validation_feature,
            validation_recovery,
            spec,
            config,
            device=device,
            indices=validation_indices,
            seed=seed,
        )
        score = {
            "all14_macro": float(validation["groups"]["all14"]["top1_macro"]),
            "all14_worst": float(validation["groups"]["all14"]["top1_worst"]),
        }
        row = {
            "epoch": epoch,
            **{name: total / max(seen, 1) for name, total in totals.items()},
            "validation_all14_macro": score["all14_macro"],
            "validation_all14_worst": score["all14_worst"],
            "validation_missing_lidar": float(validation["missing_lidar"]["top1"]),
            "learning_rate": max(group["lr"] for group in optimizer.param_groups),
            "sensing_learning_rate": next(
                (group["lr"] for group in optimizer.param_groups if group.get("name") == "sensing"),
                optimizer.param_groups[0]["lr"],
            ),
            "optimizer_updates": optimizer_updates,
        }
        history_rows.append(row)
        _write_csv(output / "logs" / f"{stem}.csv", history_rows)
        scheduler.step()
        _save_checkpoint(
            _checkpoint_path(config, spec["method"], seed, "last.pt"),
            model,
            optimizer,
            scheduler,
            config=config,
            spec=spec,
            seed=seed,
            epoch=epoch,
            metrics=score,
        )
        improved = score["all14_macro"] > best["all14_macro"] or (
            score["all14_macro"] == best["all14_macro"] and score["all14_worst"] > best["all14_worst"]
        )
        if improved:
            best = score
            patience = 0
            _save_checkpoint(
                _checkpoint_path(config, spec["method"], seed, "best.pt"),
                model,
                optimizer,
                scheduler,
                config=config,
                spec=spec,
                seed=seed,
                epoch=epoch,
                metrics=score,
            )
        else:
            patience += 1
            if patience >= int(config["training"]["patience"]):
                stop_reason = "patience"
                break
    model, payload = _load_checkpoint(_checkpoint_path(config, spec["method"], seed, "best.pt"), config, device)
    result = evaluate_model(
        model,
        m4,
        validation_feature,
        validation_recovery,
        spec,
        config,
        device=device,
        indices=validation_indices,
        seed=seed,
    )
    result.update(
        {
            "method": spec["method"],
            "stage": spec["stage"],
            "seed": seed,
            "selected_epoch": int(payload["epoch"]),
            "stop_reason": stop_reason,
            "optimizer_updates": optimizer_updates,
            "unique_train_samples": len(unique_train_indices),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "pilot_re_per_frame_nominal": int(spec["patterns"]) * int(spec["frequencies"]),
            "pilot_history_frames": int(spec["history_frames"]),
            "pilot_re_window_nominal": int(spec["patterns"]) * int(spec["frequencies"]) * int(spec["history_frames"]),
            "prototype_mode": spec["prototype_label"],
            "residual_mode": spec["residual_label"],
            "sensing_architecture": spec["model_config"].sensing_architecture,
            "git_commit": _git_commit(),
            "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
            "validation_noise_seed": config["pilot"]["validation_noise_seed"],
            "future_channel_used_as_input": False,
            "outer_test_accessed": False,
            "smoke": bool(args.smoke),
            "pid": os.getpid(),
            "logical_device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "resolved_config": str(resolved_path.resolve()),
            "output_directory": str(output.resolve()),
            "start_time_utc": started_at,
            "end_time_utc": _utc_now(),
            "exit_status": 0,
            "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
            "codebook_sha256": config["source"]["codebook_sha256"],
            "train_feature_cache_sha256": config["source"]["train_feature_cache_sha256"],
            "validation_feature_cache_sha256": config["source"]["validation_feature_cache_sha256"],
            "train_records_sha256": config["source"]["train_records_sha256"],
            "validation_records_sha256": config["source"]["validation_records_sha256"],
            "model_config": dict(spec["model_config"].__dict__),
            "loss_config": dict(spec["loss_config"].__dict__),
            "training_config": dict(config["training"]),
            "teacher_evidence_representation": "centered_log_probability"
            if spec["loss_config"].residual_regression_weight
            else None,
        }
    )
    result_path = output / "seed_results" / f"{stem}.json"
    _write_json(result_path, result)
    _write_json(
        status_path,
        run_identity
        | {
            "config": str(resolved_path.resolve()),
            "end_time_utc": result["end_time_utc"],
            "exit_status": 0,
            "status": "complete",
        },
    )
    return result


def _checkpoint_for_args(args: argparse.Namespace, config: Mapping[str, Any], method: str) -> Path:
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is not None:
        return _path(checkpoint)
    return _checkpoint_path(config, method, int(args.seed), "best.pt")


def _validate_checkpoint_spec(payload: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    expected = {
        "method": spec["method"],
        "stage": spec["stage"],
        "budget": spec["budget"],
        "history_frames": int(spec["history_frames"]),
        "csi_enabled": bool(spec["csi_enabled"]),
    }
    actual = {name: payload.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"Checkpoint does not match the requested V2 method: {actual} != {expected}.")


def _evaluation_metadata(
    result: dict[str, Any],
    *,
    model: TSPCV2Model,
    payload: Mapping[str, Any],
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    result.update(
        {
            "method": spec["method"],
            "stage": spec["stage"],
            "seed": int(payload["seed"]),
            "selected_epoch": int(payload["epoch"]),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "pilot_re_per_frame_nominal": int(spec["patterns"]) * int(spec["frequencies"]),
            "pilot_history_frames": int(spec["history_frames"]),
            "pilot_re_window_nominal": int(spec["patterns"]) * int(spec["frequencies"]) * int(spec["history_frames"]),
            "prototype_mode": spec["prototype_label"],
            "residual_mode": spec["residual_label"],
            "sensing_architecture": spec["model_config"].sensing_architecture,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_config": dict(payload["model_config"]),
            "loss_config": dict(payload["loss_config"]),
            "training_config": dict(payload.get("training_config", config["training"])),
            "git_commit": _git_commit(),
            "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
            "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
            "validation_noise_seed": config["pilot"]["validation_noise_seed"],
            "future_channel_used_as_input": False,
            "outer_test_accessed": False,
            "smoke": False,
        }
    )
    return result


def evaluate_checkpoint(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one saved V2 checkpoint over all validation masks."""

    preflight(config)
    spec = _method_spec(config, args.method)
    device = torch.device(args.device)
    checkpoint = _checkpoint_for_args(args, config, str(spec["method"]))
    model, payload = _load_checkpoint(checkpoint, config, device)
    _validate_checkpoint_spec(payload, spec)
    m4 = _load_m4(config, device)
    feature, recovery = _load_role(config, "validation")
    indices = _stratified_indices(recovery["labels_future"], args.limit, 60_000 + int(payload["seed"]))
    result = evaluate_model(
        model,
        m4,
        feature,
        recovery,
        spec,
        config,
        device=device,
        indices=indices,
        seed=int(payload["seed"]),
    )
    result = _evaluation_metadata(
        result,
        model=model,
        payload=payload,
        spec=spec,
        config=config,
        checkpoint=checkpoint,
    )
    tag = str(args.tag).strip() if args.tag else "all_masks"
    path = _path(config["output"]["root"]) / "evaluations" / f"{spec['method']}_seed{payload['seed']}_{tag}.json"
    _write_json(path, result)
    return result


def _robustness_row(result: Mapping[str, Any], *, checkpoint: Path) -> dict[str, Any]:
    groups = result["groups"]["all14"]
    return {
        "method": result.get("method"),
        "stage": result.get("stage"),
        "seed": result.get("seed"),
        "checkpoint": str(checkpoint.resolve()),
        "snr_db": result["snr_db"],
        "dropout": result["dropout"],
        "all14_macro": groups["top1_macro"],
        "all14_worst": groups["top1_worst"],
        "missing_lidar": result["missing_lidar"]["top1"],
        "sensing_all14_macro": result["sensing_all14_macro"],
        "fix_rate": groups["fix_rate_macro"],
        "harm_rate": groups["harm_rate_macro"],
        "csi_only_top1": result["csi_only_top1"],
        "oracle_all14_macro": result["oracle_all14_macro"],
        "csi_only_status": "not_applicable_without_a_separate_v2_csi_classifier",
        "oracle_definition": result["oracle_definition"],
        "pilot_re_window_actual_mean": result["pilot_re_window_actual_mean"],
        "full_probability_max_abs_diff": result["full_probability_max_abs_diff"],
        "full_argmax_mismatch": result["full_argmax_mismatch"],
        "csi_off_evidence_max_abs_diff": result.get("csi_off_evidence_max_abs_diff"),
        "csi_off_probability_max_abs_diff": result["csi_off_probability_max_abs_diff"],
        "csi_off_argmax_mismatch": result["csi_off_argmax_mismatch"],
        "outer_test_accessed": False,
    }


def robustness(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Replay a checkpoint on the preregistered validation SNR/dropout grid."""

    preflight(config)
    spec = _method_spec(config, args.method)
    device = torch.device(args.device)
    checkpoint = _checkpoint_for_args(args, config, str(spec["method"]))
    model, payload = _load_checkpoint(checkpoint, config, device)
    _validate_checkpoint_spec(payload, spec)
    m4 = _load_m4(config, device)
    feature, recovery = _load_role(config, "validation")
    indices = _stratified_indices(recovery["labels_future"], args.limit, 70_000 + int(payload["seed"]))
    collections: dict[str, list[dict[str, Any]]] = {"snr": [], "dropout": []}
    for snr_db in config["robustness"]["snr_db"]:
        result = evaluate_model(
            model,
            m4,
            feature,
            recovery,
            spec,
            config,
            device=device,
            indices=indices,
            seed=int(payload["seed"]),
            snr_db=float(snr_db),
            dropout=0.0,
        )
        result = _evaluation_metadata(
            result,
            model=model,
            payload=payload,
            spec=spec,
            config=config,
            checkpoint=checkpoint,
        )
        collections["snr"].append(_robustness_row(result, checkpoint=checkpoint))
    for dropout in config["robustness"]["dropout"]:
        result = evaluate_model(
            model,
            m4,
            feature,
            recovery,
            spec,
            config,
            device=device,
            indices=indices,
            seed=int(payload["seed"]),
            snr_db=float(config["pilot"]["validation_snr_db"]),
            dropout=float(dropout),
        )
        result = _evaluation_metadata(
            result,
            model=model,
            payload=payload,
            spec=spec,
            config=config,
            checkpoint=checkpoint,
        )
        collections["dropout"].append(_robustness_row(result, checkpoint=checkpoint))
    root = _path(config["output"]["root"])
    for kind, rows in collections.items():
        _write_csv(root / "robustness" / f"{kind}_{spec['method']}_seed{payload['seed']}.csv", rows)
    _write_csv(root / "robustness_snr.csv", collections["snr"])
    _write_csv(root / "robustness_dropout.csv", collections["dropout"])
    return collections


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _flat_result(result: Mapping[str, Any]) -> dict[str, Any]:
    groups = result["groups"]
    all14 = groups["all14"]
    return {
        "method": result["method"],
        "stage": result["stage"],
        "seed": int(result["seed"]),
        "prototype_mode": result.get("prototype_mode"),
        "residual_mode": result.get("residual_mode"),
        "sensing_architecture": result.get("sensing_architecture"),
        "selected_epoch": result.get("selected_epoch"),
        "all14_macro": all14["top1_macro"],
        "all14_worst": all14["top1_worst"],
        "single_macro": groups["single"]["top1_macro"],
        "two_macro": groups["two"]["top1_macro"],
        "three_macro": groups["three"]["top1_macro"],
        "missing_lidar": result["missing_lidar"]["top1"],
        "sensing_all14_macro": result["sensing_all14_macro"],
        "full_top1": result["full"]["top1"],
        "full_probability_max_abs_diff": result["full_probability_max_abs_diff"],
        "full_argmax_mismatch": result["full_argmax_mismatch"],
        "csi_off_evidence_max_abs_diff": result.get("csi_off_evidence_max_abs_diff"),
        "csi_off_probability_max_abs_diff": result["csi_off_probability_max_abs_diff"],
        "csi_off_argmax_mismatch": result["csi_off_argmax_mismatch"],
        "pilot_re_per_frame_nominal": result.get("pilot_re_per_frame_nominal"),
        "pilot_history_frames": result.get("pilot_history_frames"),
        "pilot_re_window_nominal": result.get("pilot_re_window_nominal"),
        "pilot_re_window_actual_mean": result["pilot_re_window_actual_mean"],
        "model_parameters": result.get("model_parameters"),
        "trainable_parameters": result.get("trainable_parameters"),
        "latency_ms_per_sample_mask": result["latency_ms_per_sample_mask"],
        "csi_only_top1": result["csi_only_top1"],
        "oracle_all14_macro": result["oracle_all14_macro"],
        "outer_test_accessed": result["outer_test_accessed"],
    }


def _legacy_b0_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    section = config.get("legacy_b0", {})
    pattern = section.get("result_glob")
    if not pattern:
        return []
    target = _path(pattern)
    rows = []
    for path in sorted(target.parent.glob(target.name)):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("outer_test_accessed") is not False:
            raise ValueError(f"Legacy B0 reference accessed outer test: {path}.")
        if result.get("method") != "L2" or result.get("head_method") != "P0":
            raise ValueError(f"Legacy B0 reference is not the immutable P0/L2 result: {path}.")
        rows.append(
            {
                "method": "B0_legacy_tspc_p0_l2",
                "seed": result.get("seed"),
                "all14_macro": result["groups"]["all14"]["top1_macro"],
                "all14_worst": result["groups"]["all14"]["top1_worst"],
                "missing_lidar": result["missing_lidar"]["top1"],
                "csi_only_top1": result["csi_only"]["top1"],
                "oracle_all14_macro": result["oracle_all14_macro"],
                "pilot_re_per_frame": 4,
                "pilot_history_frames": 5,
                "pilot_re_window": 20,
                "source": str(path.resolve()),
                "source_sha256": sha256_file(path),
                "outer_test_accessed": False,
            }
        )
    return rows


def _format_percent(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%" if _finite(value) else "N/A"


def _summary_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(str(result["method"]), str(result["stage"]))].append(result)
    metric_names = (
        "all14_macro",
        "all14_worst",
        "missing_lidar",
        "sensing_all14_macro",
        "full_top1",
        "latency_ms_per_sample_mask",
        "pilot_re_window_actual_mean",
        "trainable_parameters",
    )
    rows = []
    for (method, stage), values in sorted(grouped.items()):
        row: dict[str, Any] = {"method": method, "stage": stage, "completed_seeds": len(values)}
        row["seeds"] = ";".join(str(value["seed"]) for value in sorted(values, key=lambda item: int(item["seed"])))
        for metric in metric_names:
            observed = [float(value[metric]) for value in values if _finite(value.get(metric))]
            row[f"{metric}_mean"] = float(np.mean(observed)) if observed else float("nan")
            row[f"{metric}_std"] = float(np.std(observed)) if observed else float("nan")
        rows.append(row)
    return rows


def _select_summary_candidate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the preregistered validation ordering without rewriting config."""

    candidates = [row for row in rows if _finite(row.get("all14_macro_mean"))]
    if not candidates:
        return {"selected": None, "reason": "no_formal_v2_results", "config_modified": False}
    best_macro = max(float(row["all14_macro_mean"]) for row in candidates)
    tolerance = float(config["selection"]["near_tie_pp"]) / 100.0
    near_ties = [row for row in candidates if float(row["all14_macro_mean"]) >= best_macro - tolerance]
    selected = max(
        near_ties,
        key=lambda row: (
            float(row["all14_worst_mean"]),
            float(row["missing_lidar_mean"]),
            -float(row["trainable_parameters_mean"]),
        ),
    )
    return {
        "selected": selected["method"],
        "stage": selected["stage"],
        "best_all14_macro": best_macro,
        "near_tie_pp": float(config["selection"]["near_tie_pp"]),
        "near_tie_methods": [row["method"] for row in near_ties],
        "tie_breakers": list(config["selection"]["tie_breakers"]),
        "config_modified": False,
        "outer_test_accessed": False,
    }


def summarize(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build only local development summaries; no test role is ever loaded."""

    root = _path(config["output"]["root"])
    for source_config in sorted((ROOT / "tools/configs/tspc_v2").glob("*.yaml")):
        target_config = root / "configs" / source_config.name
        target_config.parent.mkdir(parents=True, exist_ok=True)
        target_config.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    result_paths = sorted((root / "seed_results").glob("*.json"))
    raw_results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    if any(result.get("outer_test_accessed") is not False for result in raw_results):
        raise ValueError("A V2 result has invalid outer-test lineage.")
    formal_results = [result for result in raw_results if not bool(result.get("smoke", False))]
    flat_results = [_flat_result(result) for result in formal_results]
    per_mask_rows = [
        {"method": result["method"], "stage": result["stage"], "seed": result["seed"], **row}
        for result in formal_results
        for row in result["per_mask"]
    ]
    legacy_rows = _legacy_b0_rows(config)
    _write_csv(root / "seed_results.csv", flat_results)
    _write_csv(root / "ablation_summary.csv", flat_results)
    _write_csv(root / "seed_summary.csv", _summary_rows(flat_results))
    _write_csv(root / "per_mask_results.csv", per_mask_rows)
    _write_csv(
        root / "efficiency.csv",
        [
            {
                "method": row["method"],
                "stage": row["stage"],
                "seed": row["seed"],
                "model_parameters": row["model_parameters"],
                "trainable_parameters": row["trainable_parameters"],
                "latency_ms_per_sample_mask": row["latency_ms_per_sample_mask"],
                "pilot_re_per_frame_nominal": row["pilot_re_per_frame_nominal"],
                "pilot_history_frames": row["pilot_history_frames"],
                "pilot_re_window_nominal": row["pilot_re_window_nominal"],
                "pilot_re_window_actual_mean": row["pilot_re_window_actual_mean"],
            }
            for row in flat_results
        ],
    )
    _write_csv(root / "legacy_b0_summary.csv", legacy_rows)
    robustness_snr_rows = [
        row for path in sorted((root / "robustness").glob("snr_*.csv")) for row in _read_csv(path)
    ]
    robustness_dropout_rows = [
        row for path in sorted((root / "robustness").glob("dropout_*.csv")) for row in _read_csv(path)
    ]
    _write_csv(root / "robustness_snr.csv", robustness_snr_rows)
    _write_csv(root / "robustness_dropout.csv", robustness_dropout_rows)
    fairness = {
        "protocol": dict(config["protocol"]),
        "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
        "feature_cache_sha256": {
            "train": config["source"]["train_feature_cache_sha256"],
            "validation": config["source"]["validation_feature_cache_sha256"],
        },
        "validation_noise_seed": config["pilot"]["validation_noise_seed"],
        "c1_re": {"per_frame": 20, "history_frames": 1, "window": 20},
        "c2_re": {"per_frame": 4, "history_frames": 5, "window": 20},
        "full_hard_bypass": True,
        "csi_off_exact_fallback": True,
        "csi_only": "not_applicable_without_a_separate_v2_csi_classifier",
        "oracle_definition": "per_sample_upper_bound_between_sensing_and_compensated_final",
        "legacy_b0_is_read_only": bool(legacy_rows),
        "formal_v2_runs": len(formal_results),
        "outer_test_accessed": False,
    }
    _write_json(root / "fairness_metadata.json", fairness)

    summary = _summary_rows(flat_results)
    selection = _select_summary_candidate(summary, config)
    _write_json(root / "selection.json", selection)
    completed = defaultdict(set)
    for row in flat_results:
        completed[str(row["method"])].add(int(row["seed"]))
    status_records = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "status").glob("*.json"))
    ]
    if any(record.get("outer_test_accessed") is not False for record in status_records):
        raise ValueError("A V2 run status has invalid outer-test lineage.")
    failed_statuses = [record for record in status_records if record.get("status") == "failed"]
    method_inventory = (
        ("A0", "Flatten+MLP + shared frozen Prototype", "stage_a_sensing.yaml"),
        ("A1", "hierarchical + no Prototype", "stage_a_sensing.yaml"),
        ("A2", "hierarchical + shared frozen Prototype", "stage_a_sensing.yaml"),
        ("A3", "hierarchical + independent Prototype", "stage_a_sensing.yaml"),
        ("A4", "hierarchical + random frozen Prototype", "stage_a_sensing.yaml"),
        ("B0", "legacy independent CSI classifier + fixed 0.5", "legacy read-only"),
        ("B1", "Residual MLP", "stage_b_compensation.yaml"),
        ("B2", "Prototype cross-attention residual", "stage_b_compensation.yaml"),
        ("B3", "B2 without explicit mask token", "stage_b_compensation.yaml"),
        ("B4", "B2 with CSI context only", "stage_b_compensation.yaml"),
        ("B5", "B2 with last CSI state only", "stage_b_compensation.yaml"),
        ("C0", "no CSI", "stage_c_joint.yaml"),
        ("C1", "last frame 5x4", "stage_c_joint.yaml"),
        ("C2", "five frames, 2x2 each", "stage_c_joint.yaml"),
    )
    proof_results = formal_results or raw_results
    report = [
        "# TSPC-V2 开发集汇总",
        "",
        "本报告只读取 trajectory-disjoint development train/validation 产物；没有构建或访问 outer test。",
        "",
        "## 实际数据流",
        "",
        "```text",
        "M4 token cache [B,5,4,64] + availability [B,5,4]",
        "  -> fixed slots + modality/time/availability embeddings + missing tokens",
        "  -> per-frame asymmetric attention -> frame_features [B,5,64]",
        "  -> 2-layer LSTM -> z_sensing [B,64] -> Prototype evidence e_s [B,64]",
        "historical mother CSI [B,5,32,16] complex -> nested 2x2 selection [B,5,2,2]",
        "  -> SparsePilotEncoder(real/imag/index/time/validity) -> [B,5,128]",
        "  -> masked 2-layer LSTM -> csi_temporal_tokens [B,5,128]",
        "64 Beam Prototypes query sensing/CSI/missing context -> delta_e [B,64]",
        "  -> non-Full/CSI-on: e_final = e_s + delta_e",
        "  -> non-Full/CSI-off: e_final = e_s",
        "  -> Full: copy frozen M4 p_full directly; no CSI cache read, RE=0",
        "```",
        "",
        "## 两个创新点与代码",
        "",
        "- 分层缺失感知融合：`PrototypeGuidedHierarchicalSensingEncoder`，位于 "
        "`src/kd_sensing/models/prototype_guided_hierarchical_sensing.py`。",
        "- 时间分散 CSI 条件残差：`TemporalSparseCSIEncoder` 与 "
        "`PrototypeConditionedResidualCompensator`，位于 `src/kd_sensing/models/tspc_v2.py`。",
        "- Full/CSI-off 按样本分流：`TSPCV2Model.forward`；训练、评估和 cache guard：`tools/run_tspc_v2.py`。",
        "",
        "## 配置与训练协议",
        "",
        "- 配置：`tools/configs/tspc_v2/stage_a_sensing.yaml`、`stage_b_compensation.yaml`、"
        "`stage_c_joint.yaml`。",
        f"- train/validation={config['protocol']['expected_train_samples']}/{config['protocol']['expected_validation_samples']}，"
        f"轨迹={config['protocol']['expected_train_trajectories']}/{config['protocol']['expected_validation_trajectories']}。",
        f"- batch={config['training']['batch_size']}，AdamW，lr={config['training']['learning_rate']}，"
        f"weight_decay={config['training']['weight_decay']}，max_epochs={config['training']['max_epochs']}，"
        f"patience={config['training']['patience']}，AMP={config['training']['amp']}，"
        f"grad_clip={config['training']['gradient_clip_norm']}。Stage C sensing lr scale=0.1。",
        f"- M4 SHA256: `{config['source']['m4_checkpoint_sha256']}`；validation noise seed="
        f"{config['pilot']['validation_noise_seed']}。",
        "- C1=20 RE/frame x 1 frame=20 RE/window；C2=4 RE/frame x 5 frames=20 RE/window。",
        "- 主补偿 loss 为 hard CE + sensing CE + prototype alignment + temperature-correct KL + "
        "centered log-probability residual SmoothL1；B2_CE/KL/DELTA/KL_DELTA 可分别控制。",
        "",
        "## 消融覆盖",
        "",
        "| 方法 | 定义 | 配置 | 已完成正式 seeds |",
        "|---|---|---|---|",
    ]
    for method, definition, source in method_inventory:
        seeds = sorted(completed[method])
        if method == "B0":
            seeds = sorted(int(row["seed"]) for row in legacy_rows if row.get("seed") is not None)
        report.append(f"| {method} | {definition} | {source} | {','.join(map(str, seeds)) or '-'} |")
    report.extend(
        [
            "",
            "## Prototype 诊断边界",
            "",
            "A2 对应 shared_frozen/P0，A3 对应 independent/P1，A1 对应 no_prototype/P3，A4 对应 "
            "random_frozen/P4。必须分别判断 Prototype Evidence、共享约束、真实/随机 Prototype 和 "
            "cross-attention/Residual MLP；这些结论不能相互替代。",
            "",
            "## 正式 Seed 汇总结果",
            "",
        ]
    )
    if not summary:
        report.append("尚无完成的非 smoke V2 训练结果；该报告只记录可复现配置和边界。")
    else:
        report.extend(
            [
                "| 方法 | stage | seeds | All-14 mean +/- std | Worst mean +/- std | missing_lidar mean +/- std |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in summary:
            report.append(
                f"| {row['method']} | {row['stage']} | {row['completed_seeds']} | "
                f"{_format_percent(row['all14_macro_mean'])} +/- {_format_percent(row['all14_macro_std'])} | "
                f"{_format_percent(row['all14_worst_mean'])} +/- {_format_percent(row['all14_worst_std'])} | "
                f"{_format_percent(row['missing_lidar_mean'])} +/- {_format_percent(row['missing_lidar_std'])} |"
            )
        report.append(
            f"\n按 All-14 与 {selection['near_tie_pp']} pp 内 Worst/missing_lidar/参数量顺序，当前选择为 "
            f"`{selection['selected']}`；该诊断不改写正式配置。"
        )
    report.extend(["", "逐 seed 完整指标在 `seed_results/*.json`，逐 mask 指标在 `per_mask_results.csv`。", ""])
    report.extend(["## 旧 B0 只读参照", ""])
    if not legacy_rows:
        report.append("未配置或未找到旧 TSPC P0/L2 参照结果。")
    else:
        report.extend(["| seed | All-14 | Worst | missing_lidar | CSI-only |", "|---:|---:|---:|---:|---:|"])
        for row in legacy_rows:
            report.append(
                f"| {row['seed']} | {_format_percent(row['all14_macro'])} | {_format_percent(row['all14_worst'])} | "
                f"{_format_percent(row['missing_lidar'])} | {_format_percent(row['csi_only_top1'])} |"
            )
    report.extend(["", "## Full 与 CSI-off 证明", ""])
    if not proof_results:
        report.append("尚无任何 V2 forward 产物，数值证明待 smoke/正式运行生成。")
    else:
        reference = proof_results[-1]
        full_max = max(float(result["full_probability_max_abs_diff"]) for result in proof_results)
        full_mismatch = sum(int(result["full_argmax_mismatch"]) for result in proof_results)
        csi_probability_max = max(float(result["csi_off_probability_max_abs_diff"]) for result in proof_results)
        csi_evidence_values = [
            float(result["csi_off_evidence_max_abs_diff"])
            for result in proof_results
            if result.get("csi_off_evidence_max_abs_diff") is not None
        ]
        csi_evidence_max = max(csi_evidence_values) if csi_evidence_values else float("nan")
        csi_mismatch = sum(int(result["csi_off_argmax_mismatch"]) for result in proof_results)
        report.extend(
            [
                f"- 当前证明集合包含 {len(proof_results)} 个"
                f"{'正式' if formal_results else 'smoke'} run；Full Top-1/3/5="
                f"{_format_percent(reference['full']['top1'])}/{_format_percent(reference['full']['top3'])}/"
                f"{_format_percent(reference['full']['top5'])}。",
                f"- Full max_abs_probability_difference={full_max}，argmax_mismatch={full_mismatch}，CSI RE=0。",
                f"- CSI-off evidence max_abs_difference={csi_evidence_max}，probability max_abs_difference="
                f"{csi_probability_max}，argmax_mismatch={csi_mismatch}。",
            ]
        )
    report.extend(["", "## 失败实验与未完成项", ""])
    if failed_statuses:
        for record in failed_statuses:
            report.append(
                f"- {record.get('method')} seed {record.get('seed')}: {record.get('error_type')}: {record.get('error')}"
            )
    else:
        report.append("- 当前没有记录到 runner 失败退出。")
    if raw_results and not formal_results:
        report.append("- 现有 V2 结果全部是 smoke，明确排除在正式精度结论和 seed mean/std 之外。")
    seed1_complete = all(
        1 in completed[method]
        for method in ("A0", "A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "B5", "C0", "C1", "C2")
    )
    if seed1_complete:
        report.append("- A0-A4、B1-B5、C0-C2 的 seed1 筛选已完成；正式三种子和鲁棒性网格尚未运行。")
    else:
        report.append("- A0-A4、B1-B5、C0-C2 的 seed1 筛选尚未全部完成；正式三种子和鲁棒性网格尚未运行。")
    report.extend(
        [
            "",
            "## 不能声称的结论",
            "",
            "- 未完成正式三种子前，不能声称 V2 提升精度或稳定优于 B0。",
            "- shared_frozen 不等于精度必然最优；必须与 independent/no/random 控制比较。",
            "- V2 不含独立 CSI 分类头，因此 V2 CSI-only 标为 N/A；旧 B0 的 CSI-only 只能作为外部参照。",
            "- V2 oracle 定义为 sensing 与 compensated-final 的逐样本上界，不是 sensing/独立 CSI expert oracle。",
            "- cache 只有 p_full，没有原始 M4 logits；residual teacher 使用居中的 log(p_full) canonical evidence，"
            "不能称为恢复出的原始 logits。",
            "",
            "## 数据与公平性边界",
            "",
            "- `token_sequence` 与 `candidate_history` 通过 sample ID、标签、split 和 SHA256 fail-closed 对齐。",
            "- forward 不接收标签；future_channel/future_csi/outer_test 键递归拒绝，role 只允许 train/validation。",
            "- validation 只用于 checkpoint 选择和诊断，不自动改写正式配置。",
            "- `outer_test_accessed=false`。",
        ]
    )
    report.extend(
        [
            "",
            "## 产物",
            "",
            "逐 seed 与逐 mask 指标位于 `seed_results.csv`、`per_mask_results.csv`；消融表位于 "
            "`ablation_summary.csv`；参数、延迟和实际 RE 位于 `efficiency.csv`；SNR/dropout 重放位于 "
            "`robustness_snr.csv`、`robustness_dropout.csv`。",
        ]
    )
    (root / "final_report.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {
        "formal_v2_results": len(formal_results),
        "smoke_results": len(raw_results) - len(formal_results),
        "legacy_b0_results": len(legacy_rows),
        "selected": selection["selected"],
        "outer_test_accessed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    training = commands.add_parser("train")
    training.add_argument("--method", required=True)
    training.add_argument("--seed", type=int, default=1)
    training.add_argument("--device", default="cuda")
    training.add_argument("--limit", type=int)
    training.add_argument("--validation-limit", type=int)
    training.add_argument("--epochs", type=int)
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--initialize-from", type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--method", required=True)
    evaluation.add_argument("--seed", type=int, default=1)
    evaluation.add_argument("--checkpoint", type=Path)
    evaluation.add_argument("--device", default="cuda")
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--tag")
    quality = commands.add_parser("robustness")
    quality.add_argument("--method", required=True)
    quality.add_argument("--seed", type=int, default=1)
    quality.add_argument("--checkpoint", type=Path)
    quality.add_argument("--device", default="cuda")
    quality.add_argument("--limit", type=int)
    commands.add_parser("summarize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(_path(args.config))
    command_started_at = _utc_now()
    try:
        if args.command == "preflight":
            result = preflight(config)
        elif args.command == "train":
            result = train(args, config)
        elif args.command == "evaluate":
            result = evaluate_checkpoint(args, config)
        elif args.command == "robustness":
            result = robustness(args, config)
        else:
            result = summarize(config)
    except Exception as error:
        if args.command == "train":
            override = config.get("methods", {}).get(args.method, {})
            stage = str(override.get("stage", config.get("training", {}).get("stage", "unknown")))
            stem = f"{'smoke_' if args.smoke else ''}{args.method}_{stage}_seed{int(args.seed)}"
            _write_json(
                _path(config["output"]["root"]) / "status" / f"{stem}.json",
                {
                    "pid": os.getpid(),
                    "logical_device": str(args.device),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "config": str(_path(args.config).resolve()),
                    "method": args.method,
                    "stage": stage,
                    "seed": int(args.seed),
                    "output_directory": str(_path(config["output"]["root"]).resolve()),
                    "start_time_utc": command_started_at,
                    "end_time_utc": _utc_now(),
                    "exit_status": 1,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "outer_test_accessed": False,
                },
            )
        raise
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
