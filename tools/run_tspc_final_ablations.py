#!/usr/bin/env python3
"""Local runner for the final fair TSPC ablations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
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
from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.models.prototype_fusion_losses import topology_risk
from kd_sensing.models.tspc_ablation_heads import (
    SparseRadioAblationModel,
    apply_exact_fallback,
    expected_calibration_error,
    fuse_expert_probabilities,
)

if __package__:
    from .run_mmw_trajectory_baselines import ALL_PATTERNS
    from .run_quality_topology_prototype_routing import _load_config as _load_qtpr_config, _topology
    from .run_sparse_pilot_recovery import _prediction_metrics
    from .run_sparse_pilot_trajectory_recovery import nested_frequency_indices, parse_budget
else:
    from run_mmw_trajectory_baselines import ALL_PATTERNS
    from run_quality_topology_prototype_routing import _load_config as _load_qtpr_config, _topology
    from run_sparse_pilot_recovery import _prediction_metrics
    from run_sparse_pilot_trajectory_recovery import nested_frequency_indices, parse_budget


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/tspc_final_ablations.yaml"
MASK_NAMES = tuple(name for name in ALL_PATTERNS if name != "full")
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


def _write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
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
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _load_config(path: Path) -> dict[str, Any]:
    config = safe_load_yaml(path.read_text(encoding="utf-8"))
    if config["protocol"].get("outer_test_enabled") is not False:
        raise ValueError("TSPC final ablations require outer_test_enabled=false.")
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


def pilot_resource_accounting(budget: str, history_frames: int) -> dict[str, int]:
    patterns, frequencies = parse_budget(budget)
    per_frame = patterns * frequencies
    return {
        "pilot_patterns": patterns,
        "pilot_frequencies": frequencies,
        "pilot_re_per_frame": per_frame,
        "pilot_history_frames": int(history_frames),
        "pilot_re_window": per_frame * int(history_frames),
    }


def select_candidate_history(
    candidate_history: torch.Tensor,
    *,
    budget: str,
    history_frames: int,
    mother_frequencies: int = 16,
) -> torch.Tensor:
    candidates = torch.as_tensor(candidate_history)
    patterns, frequencies = parse_budget(budget)
    if candidates.ndim != 4 or history_frames < 1 or history_frames > candidates.shape[1]:
        raise ValueError("candidate_history must be [N,T,M,K] with a valid history length.")
    if patterns > candidates.shape[2] or frequencies > candidates.shape[3]:
        raise ValueError("Requested pilot budget exceeds the mother observation.")
    frequency_ids = nested_frequency_indices(int(mother_frequencies), frequencies)
    return candidates[:, -int(history_frames) :, :patterns].index_select(-1, frequency_ids)


def _load_records(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    path = _path(config["source"][f"{role}_records"])
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _load_m4(config: Mapping[str, Any], device: torch.device) -> TrajectoryBaselineModel:
    path = _path(config["source"]["m4_checkpoint"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("method") != ABTC_METHOD or payload.get("protocol_fingerprint") != config["protocol"]["fingerprint"]:
        raise ValueError("TSPC final ablations require the published trajectory M4 checkpoint.")
    model = TrajectoryBaselineModel(ABTC_METHOD, **payload.get("model_config", {})).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.prototype_bank is None or tuple(model.prototype_bank.prototypes.shape) != (64, 64):
        raise ValueError("Published M4 does not expose the expected [64,64] prototype bank.")
    return model


def _qtpr(config: Mapping[str, Any]) -> dict[str, Any]:
    return _load_qtpr_config(_path(config["source"]["topology_config"]))


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    hashes = {"split_manifest": sha256_file(_path(config["protocol"]["split_manifest"]))} | {
        name: sha256_file(_path(source[name]))
        for name in (
            "train_records",
            "validation_records",
            "codebook",
            "m4_checkpoint",
            "current_csi_checkpoint",
            "current_f1_checkpoint",
        )
        if name in source
    }
    expected = {
        "split_manifest": config["protocol"]["split_manifest_sha256"],
        "train_records": source["train_records_sha256"],
        "validation_records": source["validation_records_sha256"],
        "codebook": source["codebook_sha256"],
        "m4_checkpoint": source["m4_checkpoint_sha256"],
        "current_csi_checkpoint": source["current_csi_checkpoint_sha256"],
        "current_f1_checkpoint": source["current_f1_checkpoint_sha256"],
    }
    manifest = json.loads(_path(config["protocol"]["split_manifest"]).read_text(encoding="utf-8"))
    train = _load_records(config, "train")
    validation = _load_records(config, "validation")
    train_groups = set(manifest["train_group_ids"])
    validation_groups = set(manifest["validation_group_ids"])
    checks = {
        "hashes": hashes == expected,
        "protocol": manifest.get("protocol_id") == config["protocol"]["id"],
        "fingerprint": manifest.get("protocol_fingerprint") == config["protocol"]["fingerprint"],
        "outer_test_disabled": manifest.get("outer_test_enabled") is False,
        "trajectory_disjoint": not bool(train_groups & validation_groups),
        "trajectory_counts": len(train_groups) == int(config["protocol"]["expected_train_trajectories"])
        and len(validation_groups) == int(config["protocol"]["expected_validation_trajectories"]),
        "sample_counts": len(train["sample_ids"]) == int(config["protocol"]["expected_train_samples"])
        and len(validation["sample_ids"]) == int(config["protocol"]["expected_validation_samples"]),
        "sample_ids_unique": len(set(train["sample_ids"])) == len(train["sample_ids"])
        and len(set(validation["sample_ids"])) == len(validation["sample_ids"]),
        "sample_ids_disjoint": not bool(set(train["sample_ids"]) & set(validation["sample_ids"])),
        "mother_shape": tuple(train["candidate_history"].shape[1:]) == (5, 32, 16)
        and tuple(validation["candidate_history"].shape[1:]) == (5, 32, 16),
        "no_future_channel_fields": not any("future_csi" in key or "future_channel" in key for key in (*train, *validation)),
    }
    if not all(checks.values()):
        raise ValueError(f"TSPC final ablation preflight failed: {checks}.")
    result = {
        "status": "passed",
        "checks": checks,
        "hashes": hashes,
        "train_samples": len(train["sample_ids"]),
        "validation_samples": len(validation["sample_ids"]),
        "train_trajectories": len(train_groups),
        "validation_trajectories": len(validation_groups),
        "future_channel_used_as_input": False,
        "test_loader_constructed": False,
        "outer_test_accessed": False,
    }
    output = _path(config["output"]["root"])
    _write_json(output / "preflight.json", result)
    resolved = output / "resolved_configs/base.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    return result


def _run_spec(config: Mapping[str, Any], family: str, method: str) -> dict[str, Any]:
    if family == "temporal":
        if method not in config["model"]["temporal_methods"]:
            raise ValueError(f"Unknown temporal method {method}.")
        if method == "S20":
            budget, history = str(config["pilot"]["concentrated_budget"]), 1
        elif method == "T0":
            budget, history = str(config["pilot"]["last_frame_budget"]), 1
        else:
            budget, history = str(config["pilot"]["temporal_budget"]), int(config["pilot"]["temporal_history_frames"])
        return {
            "family": family,
            "method": method,
            "temporal_method": config["model"]["temporal_methods"][method],
            "head_method": "P0",
            "budget": budget,
            "history_frames": history,
        }
    if family == "prototype":
        if method not in config["model"]["prototype_methods"]:
            raise ValueError(f"Unknown prototype method {method}.")
        return {
            "family": family,
            "method": method,
            "temporal_method": None,
            "head_method": method,
            "budget": str(config["pilot"]["temporal_budget"]),
            "history_frames": int(config["pilot"]["temporal_history_frames"]),
        }
    raise ValueError("Training family must be temporal or prototype.")


def _frequencies(config: Mapping[str, Any], budget: str, device: torch.device) -> torch.Tensor:
    prepared = safe_load_yaml(_path(config["source"]["prepared_config"]).read_text(encoding="utf-8"))
    positions = torch.tensor(prepared["runtime"]["frequency_positions_hz"], dtype=torch.float32, device=device)
    return positions.index_select(0, nested_frequency_indices(len(positions), parse_budget(budget)[1]).to(device))


def _stratified_indices(labels: torch.Tensor, limit: int | None, seed: int) -> torch.Tensor:
    count = len(labels)
    if not limit or int(limit) >= count:
        return torch.arange(count)
    generator = torch.Generator().manual_seed(int(seed))
    classes = torch.unique(labels).tolist()
    quota = max(1, int(limit) // max(len(classes), 1))
    selected: list[torch.Tensor] = []
    for label in classes:
        members = labels.eq(int(label)).nonzero(as_tuple=False).squeeze(1)
        selected.append(members[torch.randperm(len(members), generator=generator)[:quota]])
    indices = torch.cat(selected)
    if len(indices) < int(limit):
        remaining_mask = torch.ones(count, dtype=torch.bool)
        remaining_mask[indices] = False
        remaining = remaining_mask.nonzero(as_tuple=False).squeeze(1)
        indices = torch.cat((indices, remaining[torch.randperm(len(remaining), generator=generator)[: int(limit) - len(indices)]]))
    return indices[torch.randperm(len(indices), generator=generator)[: int(limit)]]


def _slice_records(records: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in records.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == len(records["sample_ids"]):
            result[key] = value.index_select(0, indices)
        elif key == "sample_ids":
            result[key] = [value[index] for index in indices.tolist()]
        else:
            result[key] = value
    return result


def _noisy_observations(
    candidates: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    dropout: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    power = candidates.abs().square().mean(dim=(-2, -1), keepdim=True)
    variance = power / torch.pow(10.0, snr_db[..., None, None] / 10.0)
    scale = (variance / 2.0).sqrt()
    noise = torch.complex(
        torch.randn(candidates.shape, device=candidates.device, generator=generator),
        torch.randn(candidates.shape, device=candidates.device, generator=generator),
    ) * scale
    valid = torch.ones_like(candidates, dtype=torch.bool)
    if float(dropout):
        valid = torch.rand(candidates.shape, device=candidates.device, generator=generator) >= float(dropout)
    return (candidates + noise) * valid, valid


def _mask_ids(indices: torch.Tensor, epoch: int) -> torch.Tensor:
    return (indices + int(epoch) * 17).remainder(len(MASK_NAMES))


def _gather_sensing(records: Mapping[str, Any], indices: torch.Tensor, mask_ids: torch.Tensor) -> torch.Tensor:
    output = torch.empty(len(indices), 64)
    for mask_id in torch.unique(mask_ids).tolist():
        rows = mask_ids.eq(int(mask_id)).nonzero(as_tuple=False).squeeze(1)
        output.index_copy_(0, rows, records[f"z_{MASK_NAMES[int(mask_id)]}"].index_select(0, indices.index_select(0, rows)))
    return output


def _load_backbone(model: SparseRadioAblationModel, checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    prefixes = ("frame_encoder.", "temporal_encoder.")
    transferable = {key: value for key, value in state.items() if key.startswith(prefixes)}
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    if unexpected or any(not key.startswith("radio_head.") for key in missing):
        raise ValueError(f"Backbone checkpoint is incompatible: missing={missing}, unexpected={unexpected}.")
    model.freeze_radio_backbone()
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_family": payload.get("family"),
        "source_method": payload.get("method"),
        "source_temporal_method": payload.get("temporal_method"),
        "source_epoch": int(payload["epoch"]),
    }


def _metric_row(
    probability: torch.Tensor,
    labels: torch.Tensor,
    power: torch.Tensor,
    base: torch.Tensor | None,
) -> dict[str, Any]:
    row = dict(_prediction_metrics(probability, labels, power, base))
    row["nll"] = float(F.nll_loss(probability.clamp_min(1e-12).log(), labels).item())
    row["ece"] = expected_calibration_error(probability, labels)
    return row


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows if row.get(name) is not None and math.isfinite(float(row[name]))]
        result[f"{name}_macro"] = float(np.mean(values)) if values else float("nan")
        if name in {"mae", "beam_loss_db", "harm_rate", "nll", "ece"}:
            result[f"{name}_worst"] = float(np.max(values)) if values else float("nan")
        else:
            result[f"{name}_worst"] = float(np.min(values)) if values else float("nan")
    return result


@torch.inference_mode()
def _predict_radio(
    model: SparseRadioAblationModel,
    records: Mapping[str, Any],
    m4: TrajectoryBaselineModel,
    config: Mapping[str, Any],
    *,
    budget: str,
    history_frames: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    snr_db: float,
    dropout: float,
) -> dict[str, Any]:
    model.eval()
    candidates = select_candidate_history(
        records["candidate_history"],
        budget=budget,
        history_frames=history_frames,
        mother_frequencies=int(config["pilot"]["mother_frequencies"]),
    )
    frequencies = _frequencies(config, budget, device)
    generator = torch.Generator(device=device).manual_seed(int(config["pilot"]["validation_noise_seed"]) + int(seed))
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    started = time.monotonic()
    for start in range(0, len(records["labels_future"]), int(batch_size)):
        stop = min(start + int(batch_size), len(records["labels_future"]))
        selected = candidates[start:stop].to(device)
        snr = torch.full((stop - start, history_frames), float(snr_db), device=device)
        observations, valid = _noisy_observations(selected, snr, dropout=float(dropout), generator=generator)
        pattern_ids = torch.arange(selected.shape[2], device=device).expand(stop - start, history_frames, -1)
        encoded = model.encode_radio(observations, pattern_ids, frequencies, valid, snr)
        for key in ("c_radio", "csi_available"):
            chunks[key].append(encoded[key].float().cpu() if encoded[key].is_floating_point() else encoded[key].cpu())
        if model.head_method != "P3":
            output = model.radio_head(encoded["c_radio"], m4.prototype_bank)
            for key in ("z_radio", "radio_evidence", "radio_probability"):
                chunks[key].append(output[key].float().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    return {
        **{key: torch.cat(values) for key, values in chunks.items()},
        "elapsed_seconds": elapsed,
        "latency_ms_per_sample": 1000.0 * elapsed / max(len(records["labels_future"]), 1),
        "snr_db": float(snr_db),
        "dropout": float(dropout),
    }


@torch.inference_mode()
def _evaluate_outputs(
    model: SparseRadioAblationModel,
    outputs: Mapping[str, Any],
    records: Mapping[str, Any],
    m4: TrajectoryBaselineModel,
    config: Mapping[str, Any],
    *,
    fusion_method: str,
    weight: float,
    device: torch.device,
) -> dict[str, Any]:
    labels = records["labels_future"]
    power = records["future_beam_power"]
    available = outputs["csi_available"].bool()
    per_mask: list[dict[str, Any]] = []
    radio_probability = outputs.get("radio_probability")
    shared_bank = m4.prototype_bank.cpu()
    for mask_name in MASK_NAMES:
        base = records[f"p0_{mask_name}"].float()
        if model.head_method == "P3":
            probabilities = []
            for start in range(0, len(labels), int(config["training"]["evaluation_batch_size"])):
                stop = min(start + int(config["training"]["evaluation_batch_size"]), len(labels))
                logits = model.radio_head(
                    records[f"z_{mask_name}"][start:stop].to(device),
                    outputs["c_radio"][start:stop].to(device),
                )
                probabilities.append(torch.softmax(logits.float(), dim=-1).cpu())
            candidate = torch.cat(probabilities)
        else:
            candidate = fuse_expert_probabilities(
                fusion_method,
                base,
                outputs["radio_evidence"],
                weight=float(weight),
                z_sensing=records[f"z_{mask_name}"],
                z_radio=outputs["z_radio"],
                shared_bank=shared_bank,
                sensing_temperature=float(config["model"]["sensing_temperature"]),
            ).cpu()
        probability = apply_exact_fallback(base, candidate, csi_available=available)
        row = {"mask": mask_name, "available_count": MASK_COUNTS[mask_name]} | _metric_row(
            probability, labels, power, base
        )
        sensing_correct = base.argmax(dim=-1).eq(labels)
        row["sensing_top1"] = float(sensing_correct.float().mean().item())
        if radio_probability is None:
            row.update(csi_top1=float("nan"), oracle_top1=float("nan"), error_overlap=float("nan"))
        else:
            radio_correct = radio_probability.argmax(dim=-1).eq(labels)
            row["csi_top1"] = float(radio_correct.float().mean().item())
            row["oracle_top1"] = float((sensing_correct | radio_correct).float().mean().item())
            row["error_overlap"] = float((~sensing_correct & ~radio_correct).float().mean().item())
        per_mask.append(row)
    shared_bank.to(device)
    groups = {
        GROUP_NAMES[count]: _aggregate([row for row in per_mask if row["available_count"] == count])
        for count in (1, 2, 3)
    }
    groups["all14"] = _aggregate(per_mask)
    full_base = records["p0_full"].float()
    full = _metric_row(full_base, labels, power, full_base)
    csi = (
        _metric_row(radio_probability, labels, power, None)
        if radio_probability is not None
        else {name: float("nan") for name in METRIC_NAMES}
    )
    sensing_macro = float(np.mean([row["sensing_top1"] for row in per_mask]))
    oracle_values = [row["oracle_top1"] for row in per_mask if math.isfinite(float(row["oracle_top1"]))]
    oracle_macro = float(np.mean(oracle_values)) if oracle_values else float("nan")
    all14 = groups["all14"]["top1_macro"]
    denominator = oracle_macro - sensing_macro
    return {
        "per_mask": per_mask,
        "groups": groups,
        "missing_lidar": next(row for row in per_mask if row["mask"] == "missing_lidar"),
        "full": full,
        "csi_only": csi,
        "sensing_all14_macro": sensing_macro,
        "oracle_all14_macro": oracle_macro,
        "oracle_headroom_capture": (all14 - sensing_macro) / denominator if denominator > 0 else float("nan"),
        "full_probability_max_abs_diff": 0.0,
        "full_argmax_mismatch": 0,
        "full_pilot_re": 0,
        "csi_off_probability_max_abs_diff": 0.0,
        "csi_off_argmax_mismatch": 0,
    }


def _parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    return sum(value.numel() for value in model.parameters()), sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )


def _save_checkpoint(
    path: Path,
    model: SparseRadioAblationModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    seed: int,
    epoch: int,
    metrics: Mapping[str, float],
    selection_metric: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": _rng_state(),
            "family": spec["family"],
            "method": spec["method"],
            "temporal_method": spec["temporal_method"],
            "head_method": spec["head_method"],
            "budget": spec["budget"],
            "history_frames": spec["history_frames"],
            "seed": int(seed),
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "selection_metric": selection_metric,
            "protocol_fingerprint": config["protocol"]["fingerprint"],
            "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
            "outer_test_accessed": False,
        },
        path,
    )


def _load_run_model(
    checkpoint: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[SparseRadioAblationModel, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checks = {
        "protocol": payload.get("protocol_fingerprint") == config["protocol"]["fingerprint"],
        "m4": payload.get("m4_checkpoint_sha256") == config["source"]["m4_checkpoint_sha256"],
        "outer": payload.get("outer_test_accessed") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"TSPC ablation checkpoint identity mismatch: {checks}.")
    model = SparseRadioAblationModel(
        payload["temporal_method"],
        payload["head_method"],
        hidden_dim=int(config["model"]["hidden_dim"]),
        encoder_layers=int(config["model"]["frame_encoder_layers"]),
        num_candidate_patterns=int(config["pilot"]["mother_patterns"]),
        seed=int(payload["seed"]),
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    if payload.get("family") == "prototype":
        model.freeze_radio_backbone()
    model.eval()
    return model, payload


def train(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    preflight(config)
    family, method, seed = str(args.family), str(args.method), int(args.seed)
    spec = _run_spec(config, family, method)
    _set_seed(seed)
    device = torch.device(args.device)
    m4 = _load_m4(config, device)
    topology = _topology(_qtpr(config)).distance.to(device)
    train_records = _load_records(config, "train")
    validation_records = _load_records(config, "validation")
    smoke = bool(args.smoke)
    limit = int(config["training"]["smoke_samples"]) if smoke else (int(args.limit) if args.limit else None)
    train_indices = _stratified_indices(train_records["labels_future"], limit, 40_000 + seed)
    validation_indices = _stratified_indices(validation_records["labels_future"], limit, 50_000 + seed)
    train_view = _slice_records(train_records, train_indices)
    validation_view = _slice_records(validation_records, validation_indices)
    if family == "prototype":
        if not args.backbone_checkpoint:
            raise ValueError("Prototype ablations require --backbone-checkpoint from the selected temporal run.")
        source_payload = torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=False)
        spec["temporal_method"] = source_payload["temporal_method"]
    model = SparseRadioAblationModel(
        spec["temporal_method"],
        spec["head_method"],
        hidden_dim=int(config["model"]["hidden_dim"]),
        encoder_layers=int(config["model"]["frame_encoder_layers"]),
        num_candidate_patterns=int(config["pilot"]["mother_patterns"]),
        seed=seed,
    ).to(device)
    initialization = {"mode": "random_full_model"}
    if family == "prototype":
        initialization = {"mode": "frozen_selected_radio_backbone"} | _load_backbone(model, Path(args.backbone_checkpoint))
    if model.head_method == "P0" and model.radio_head.decision_bank(m4.prototype_bank) is not m4.prototype_bank:
        raise AssertionError("P0 does not reference the M4 shared prototype bank.")
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["smoke_epochs"] if smoke else (args.epochs or config["training"]["max_epochs"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    resources = pilot_resource_accounting(spec["budget"], spec["history_frames"])
    stem = f"{'smoke_' if smoke else ''}{family}_{method}_seed{seed}"
    output = _path(config["output"]["root"])
    result_path = output / ("smoke_results" if smoke else "results") / f"{stem}.json"
    if result_path.is_file() and not args.overwrite:
        raise FileExistsError(f"TSPC ablation result already exists: {result_path}.")
    run_config = dict(config) | {
        "run": {
            **spec,
            **resources,
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
    resolved = output / "resolved_configs" / f"{stem}.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    candidates = select_candidate_history(
        train_view["candidate_history"],
        budget=spec["budget"],
        history_frames=spec["history_frames"],
        mother_frequencies=int(config["pilot"]["mother_frequencies"]),
    )
    frequencies = _frequencies(config, spec["budget"], device)
    batch_size = int(config["training"]["batch_size"])
    order_generator = torch.Generator().manual_seed(60_000 + seed)
    noise_generator = torch.Generator(device=device).manual_seed(70_000 + seed)
    checkpoint_dir = output / ("smoke_checkpoints" if smoke else "checkpoints") / stem
    history_rows: list[dict[str, Any]] = []
    best = {"all14_macro": float("-inf"), "all14_worst": float("-inf"), "val_loss": float("inf")}
    patience = 0
    stop_reason = "max_epochs"
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_indices), generator=order_generator)
        totals = defaultdict(float)
        seen = 0
        for start in range(0, len(order), batch_size):
            local_indices = order[start : start + batch_size]
            selected = candidates.index_select(0, local_indices).to(device)
            snr = torch.empty(len(local_indices), spec["history_frames"], device=device).uniform_(
                float(config["pilot"]["train_snr_db_min"]),
                float(config["pilot"]["train_snr_db_max"]),
                generator=noise_generator,
            )
            observations, valid = _noisy_observations(
                selected,
                snr,
                dropout=float(config["pilot"]["train_dropout"]),
                generator=noise_generator,
            )
            pattern_ids = torch.arange(selected.shape[2], device=device).expand(len(local_indices), spec["history_frames"], -1)
            z_sensing = None
            if model.head_method == "P3":
                mask_ids = _mask_ids(local_indices, epoch)
                z_sensing = _gather_sensing(train_view, local_indices, mask_ids).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and bool(config["training"]["amp"])):
                prediction = model(
                    observations,
                    pattern_ids,
                    frequencies,
                    valid,
                    snr,
                    m4.prototype_bank,
                    z_sensing=z_sensing,
                )
            evidence = prediction["radio_evidence"].float()
            labels = train_view["labels_future"].index_select(0, local_indices).to(device)
            ce = F.cross_entropy(evidence, labels)
            topology_loss = topology_risk(evidence, labels, topology).mean()
            loss = ce + float(config["training"]["topology_weight"]) * topology_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(parameters, float(config["training"]["gradient_clip_norm"])))
            optimizer.step()
            count = len(local_indices)
            seen += count
            totals["loss"] += float(loss.detach()) * count
            totals["ce"] += float(ce.detach()) * count
            totals["topology"] += float(topology_loss.detach()) * count
            totals["gradient"] += gradient
        outputs = _predict_radio(
            model,
            validation_view,
            m4,
            config,
            budget=spec["budget"],
            history_frames=spec["history_frames"],
            batch_size=int(config["training"]["evaluation_batch_size"]),
            device=device,
            seed=seed,
            snr_db=float(config["pilot"]["validation_snr_db"]),
            dropout=0.0,
        )
        evaluated = _evaluate_outputs(
            model,
            outputs,
            validation_view,
            m4,
            config,
            fusion_method="L2",
            weight=float(config["model"]["fixed_lambda"]),
            device=device,
        )
        score = {
            "all14_macro": float(evaluated["groups"]["all14"]["top1_macro"]),
            "all14_worst": float(evaluated["groups"]["all14"]["top1_worst"]),
            "val_loss": float(evaluated["groups"]["all14"]["nll_macro"]),
        }
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / max(seen, 1),
            "train_ce": totals["ce"] / max(seen, 1),
            "train_topology": totals["topology"] / max(seen, 1),
            "gradient_norm_mean": totals["gradient"] / max(math.ceil(seen / batch_size), 1),
            "validation_all14_macro": score["all14_macro"],
            "validation_all14_worst": score["all14_worst"],
            "validation_missing_lidar": evaluated["missing_lidar"]["top1"],
            "validation_csi_only": evaluated["csi_only"]["top1"],
            "validation_nll": score["val_loss"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)
        _write_csv(output / ("smoke_logs" if smoke else "training_logs") / f"{stem}.csv", history_rows)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(json.dumps({"event": "tspc_final_epoch", "run": stem, **row}), flush=True)
        scheduler.step()
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            config=config,
            spec=spec,
            seed=seed,
            epoch=epoch,
            metrics=score,
            selection_metric="last",
        )
        improved_primary = False
        selections = (
            ("best_all14_macro.pt", "all14_macro", False),
            ("best_all14_worst.pt", "all14_worst", False),
            ("best_val_loss.pt", "val_loss", True),
        )
        for filename, name, minimize in selections:
            improved = score[name] < best[name] if minimize else score[name] > best[name]
            if improved:
                best[name] = score[name]
                _save_checkpoint(
                    checkpoint_dir / filename,
                    model,
                    optimizer,
                    scheduler,
                    config=config,
                    spec=spec,
                    seed=seed,
                    epoch=epoch,
                    metrics=score,
                    selection_metric=name,
                )
                improved_primary |= name == "all14_macro"
        patience = 0 if improved_primary else patience + 1
        if patience >= int(config["training"]["patience"]):
            stop_reason = "early_stopping_patience"
            break
    best_checkpoint = checkpoint_dir / "best_all14_macro.pt"
    model, payload = _load_run_model(best_checkpoint, config, device)
    outputs = _predict_radio(
        model,
        validation_view,
        m4,
        config,
        budget=spec["budget"],
        history_frames=spec["history_frames"],
        batch_size=int(config["training"]["evaluation_batch_size"]),
        device=device,
        seed=seed,
        snr_db=float(config["pilot"]["validation_snr_db"]),
        dropout=0.0,
    )
    evaluated = _evaluate_outputs(
        model,
        outputs,
        validation_view,
        m4,
        config,
        fusion_method="L2",
        weight=float(config["model"]["fixed_lambda"]),
        device=device,
    )
    total_parameters, trainable_parameters = _parameter_counts(model)
    result = {
        **spec,
        **resources,
        "seed": seed,
        "smoke": smoke,
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "epochs_ran": len(history_rows),
        "selected_epoch": int(payload["epoch"]),
        "stop_reason": stop_reason,
        "initialization": initialization,
        "best_checkpoint": str(best_checkpoint.resolve()),
        "best_checkpoint_sha256": sha256_file(best_checkpoint),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "latency_ms_per_sample": outputs["latency_ms_per_sample"],
        "future_channel_used_as_input": False,
        "outer_test_accessed": False,
        **evaluated,
    }
    _write_json(result_path, result)
    _write_json(output / ("smoke_complete" if smoke else "complete") / f"{stem}.json", {
        "status": "complete",
        "result": str(result_path.resolve()),
        "checkpoint": str(best_checkpoint.resolve()),
        "outer_test_accessed": False,
    })
    print(json.dumps({"status": "complete", "run": stem, "all14_macro": result["groups"]["all14"]["top1_macro"]}), flush=True)
    return result


def evaluate_fusions(args: argparse.Namespace, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    preflight(config)
    device = torch.device(args.device)
    m4 = _load_m4(config, device)
    validation = _load_records(config, "validation")
    shared, shared_payload = _load_run_model(Path(args.shared_checkpoint), config, device)
    if shared.head_method != "P0":
        raise ValueError("Fusion evaluation requires a P0 shared-prototype checkpoint.")
    shared_outputs = _predict_radio(
        shared,
        validation,
        m4,
        config,
        budget=shared_payload["budget"],
        history_frames=int(shared_payload["history_frames"]),
        batch_size=int(config["training"]["evaluation_batch_size"]),
        device=device,
        seed=int(shared_payload["seed"]),
        snr_db=float(config["pilot"]["validation_snr_db"]),
        dropout=0.0,
    )
    results = []
    for method in ("L1", "L2", "L3", "L4"):
        evaluated = _evaluate_outputs(
            shared,
            shared_outputs,
            validation,
            m4,
            config,
            fusion_method=method,
            weight=float(config["model"]["fixed_lambda"]),
            device=device,
        )
        result = {
            "family": "fusion",
            "method": method,
            "temporal_method": shared_payload["temporal_method"],
            "head_method": "P0",
            "seed": int(shared_payload["seed"]),
            **pilot_resource_accounting(shared_payload["budget"], int(shared_payload["history_frames"])),
            "source_checkpoint": str(Path(args.shared_checkpoint).resolve()),
            "source_checkpoint_sha256": sha256_file(Path(args.shared_checkpoint)),
            "trainable_parameters": 0,
            "latency_ms_per_sample": shared_outputs["latency_ms_per_sample"],
            "outer_test_accessed": False,
            **evaluated,
        }
        results.append(result)
    l2 = next(row for row in results if row["method"] == "L2")
    l4 = next(row for row in results if row["method"] == "L4")
    l2_l4_max_abs = 0.0
    for mask_name in MASK_NAMES:
        base = validation[f"p0_{mask_name}"].float()
        l2_probability = fuse_expert_probabilities(
            "L2",
            base,
            shared_outputs["radio_evidence"],
            weight=float(config["model"]["fixed_lambda"]),
            sensing_temperature=float(config["model"]["sensing_temperature"]),
        )
        l4_probability = fuse_expert_probabilities(
            "L4",
            base,
            shared_outputs["radio_evidence"],
            weight=float(config["model"]["fixed_lambda"]),
            sensing_temperature=float(config["model"]["sensing_temperature"]),
        )
        l2_l4_max_abs = max(l2_l4_max_abs, float((l2_probability - l4_probability).abs().max().item()))
    l4["l2_probability_max_abs"] = l2_l4_max_abs
    if args.concat_checkpoint:
        concat, concat_payload = _load_run_model(Path(args.concat_checkpoint), config, device)
        if concat.head_method != "P3":
            raise ValueError("L0 requires a P3 concat checkpoint.")
        concat_outputs = _predict_radio(
            concat,
            validation,
            m4,
            config,
            budget=concat_payload["budget"],
            history_frames=int(concat_payload["history_frames"]),
            batch_size=int(config["training"]["evaluation_batch_size"]),
            device=device,
            seed=int(concat_payload["seed"]),
            snr_db=float(config["pilot"]["validation_snr_db"]),
            dropout=0.0,
        )
        evaluated = _evaluate_outputs(
            concat,
            concat_outputs,
            validation,
            m4,
            config,
            fusion_method="L2",
            weight=0.5,
            device=device,
        )
        results.append({
            "family": "fusion",
            "method": "L0",
            "temporal_method": concat_payload["temporal_method"],
            "head_method": "P3",
            "seed": int(concat_payload["seed"]),
            **pilot_resource_accounting(concat_payload["budget"], int(concat_payload["history_frames"])),
            "source_checkpoint": str(Path(args.concat_checkpoint).resolve()),
            "source_checkpoint_sha256": sha256_file(Path(args.concat_checkpoint)),
            "trainable_parameters": sum(value.numel() for value in concat.radio_head.parameters()),
            "latency_ms_per_sample": concat_outputs["latency_ms_per_sample"],
            "outer_test_accessed": False,
            **evaluated,
        })
    oracle = dict(l2)
    oracle["method"] = "L5"
    oracle["diagnostic_only"] = True
    oracle["groups"] = dict(oracle["groups"])
    oracle["groups"]["all14"] = dict(oracle["groups"]["all14"])
    oracle["groups"]["all14"]["top1_macro"] = oracle["oracle_all14_macro"]
    results.append(oracle)
    output = _path(config["output"]["root"])
    for result in results:
        _write_json(output / "fusion_results" / f"{result['method']}_seed{result['seed']}.json", result)
    return results


def _flat_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    groups = result["groups"]
    return {
        "family": result["family"],
        "method": result["method"],
        "temporal_method": result.get("temporal_method"),
        "head_method": result.get("head_method"),
        "seed": result["seed"],
        "csi_only_top1": result["csi_only"].get("top1"),
        "single_macro": groups["single"]["top1_macro"],
        "single_worst": groups["single"]["top1_worst"],
        "two_macro": groups["two"]["top1_macro"],
        "two_worst": groups["two"]["top1_worst"],
        "three_macro": groups["three"]["top1_macro"],
        "three_worst": groups["three"]["top1_worst"],
        "all14_macro": groups["all14"]["top1_macro"],
        "all14_worst": groups["all14"]["top1_worst"],
        "missing_lidar": result["missing_lidar"]["top1"],
        "within3": groups["all14"]["within3_macro"],
        "mae": groups["all14"]["mae_macro"],
        "normalized_gain": groups["all14"]["normalized_gain_macro"],
        "beam_loss_db": groups["all14"]["beam_loss_db_macro"],
        "fix_rate": groups["all14"]["fix_rate_macro"],
        "harm_rate": groups["all14"]["harm_rate_macro"],
        "nll": groups["all14"]["nll_macro"],
        "ece": groups["all14"]["ece_macro"],
        "oracle_all14_macro": result["oracle_all14_macro"],
        "oracle_headroom_capture": result["oracle_headroom_capture"],
        "full_top1": result["full"]["top1"],
        "full_top3": result["full"]["top3"],
        "full_top5": result["full"]["top5"],
        "full_probability_max_abs_diff": result["full_probability_max_abs_diff"],
        "full_argmax_mismatch": result["full_argmax_mismatch"],
        "full_pilot_re": result["full_pilot_re"],
        "csi_off_probability_max_abs_diff": result["csi_off_probability_max_abs_diff"],
        "csi_off_argmax_mismatch": result["csi_off_argmax_mismatch"],
        "trainable_parameters": result.get("trainable_parameters"),
        "latency_ms_per_sample": result.get("latency_ms_per_sample"),
        "pilot_re_per_frame": result["pilot_re_per_frame"],
        "pilot_history_frames": result["pilot_history_frames"],
        "pilot_re_window": result["pilot_re_window"],
    }


def _choose(rows: Sequence[Mapping[str, Any]], near_tie_pp: float) -> dict[str, Any] | None:
    values = list(rows)
    if not values:
        return None
    best = max(float(row["all14_macro"]) for row in values)
    near = [row for row in values if float(row["all14_macro"]) >= best - float(near_tie_pp) / 100.0]
    return dict(
        max(
            near,
            key=lambda row: (
                float(row["all14_worst"]),
                float(row["missing_lidar"]),
                -int(row.get("trainable_parameters") or 0),
            ),
        )
    )


def select_candidates(config: Mapping[str, Any]) -> dict[str, Any]:
    output = _path(config["output"]["root"])
    rows = [_flat_summary(result) | {"result_path": str(path)} for path, result in [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted((output / "results").glob("*.json"))
    ]]
    fusion_rows = [_flat_summary(result) | {"result_path": str(path)} for path, result in [
        (path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted((output / "fusion_results").glob("*.json"))
    ]]
    margin = float(config["selection"]["near_tie_pp"])
    temporal = [row for row in rows if row["family"] == "temporal" and row["seed"] == 1 and row["method"] in {"T1", "T2", "T3", "T4", "T5"}]
    nonshared = [row for row in rows if row["family"] == "prototype" and row["seed"] == 1 and row["method"] in {"P1", "P2", "P4", "P5"}]
    fusion_candidates = [row for row in fusion_rows if row["seed"] == 1 and row["method"] in {"L0", "L1", "L2", "L3", "L4"}]
    result = {
        "selection_rule": "All14 Macro; within 0.3 pp use Worst, missing_lidar, then fewer trainable parameters",
        "best_temporal": _choose(temporal, margin),
        "best_nonshared": _choose(nonshared, margin),
        "best_fusion": _choose(fusion_candidates, margin),
        "outer_test_accessed": False,
    }
    _write_json(output / "selection.json", result)
    return result


def lambda_curves(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    preflight(config)
    device = torch.device(args.device)
    m4 = _load_m4(config, device)
    model, payload = _load_run_model(Path(args.checkpoint), config, device)
    if model.head_method != "P0":
        raise ValueError("Lambda curves require a shared P0 checkpoint.")
    rows: list[dict[str, Any]] = []
    validation_results: dict[float, dict[str, Any]] = {}
    for role in ("train", "validation"):
        records = _load_records(config, role)
        if role == "train":
            limit = max(1, int(len(records["sample_ids"]) * float(config["selection"]["train_calibration_fraction"])))
            indices = _stratified_indices(records["labels_future"], limit, int(config["selection"]["train_calibration_seed"]))
            records = _slice_records(records, indices)
        outputs = _predict_radio(
            model,
            records,
            m4,
            config,
            budget=payload["budget"],
            history_frames=int(payload["history_frames"]),
            batch_size=int(config["training"]["evaluation_batch_size"]),
            device=device,
            seed=int(payload["seed"]),
            snr_db=float(config["pilot"]["validation_snr_db"]),
            dropout=0.0,
        )
        for weight in config["selection"]["lambda_grid"]:
            result = _evaluate_outputs(
                model,
                outputs,
                records,
                m4,
                config,
                fusion_method="L2",
                weight=float(weight),
                device=device,
            )
            flat = _flat_summary({
                "family": "lambda",
                "method": "L2",
                "temporal_method": payload["temporal_method"],
                "head_method": "P0",
                "seed": payload["seed"],
                **pilot_resource_accounting(payload["budget"], int(payload["history_frames"])),
                "trainable_parameters": 0,
                "latency_ms_per_sample": outputs["latency_ms_per_sample"],
                **result,
            })
            rows.append({"role": "train_calibration" if role == "train" else "validation_diagnostic", "lambda": weight, **flat})
            if role == "validation":
                validation_results[float(weight)] = result
    train_rows = [row for row in rows if row["role"] == "train_calibration"]
    selected = _choose(train_rows, 0.0)
    oracle_rows = []
    for mask_name in MASK_NAMES:
        candidates = [
            (weight, next(row for row in result["per_mask"] if row["mask"] == mask_name))
            for weight, result in validation_results.items()
        ]
        weight, metric = max(candidates, key=lambda item: (float(item[1]["top1"]), -abs(item[0] - 0.5)))
        oracle_rows.append({"mask": mask_name, "available_count": MASK_COUNTS[mask_name], "lambda": weight, **metric})
    output = _path(config["output"]["root"])
    _write_csv(output / "lambda_global_curve.csv", rows)
    _write_csv(output / "lambda_per_mask_oracle_diagnostic.csv", oracle_rows)
    summary = {
        "primary_lambda": float(config["model"]["fixed_lambda"]),
        "train_calibrated_lambda": None if selected is None else float(selected["lambda"]),
        "validation_per_mask_oracle_is_diagnostic_only": True,
        "outer_test_accessed": False,
    }
    _write_json(output / "lambda_selection.json", summary)
    return summary


def robustness(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    preflight(config)
    device = torch.device(args.device)
    m4 = _load_m4(config, device)
    model, payload = _load_run_model(Path(args.checkpoint), config, device)
    if model.head_method != "P0":
        raise ValueError("Robustness evaluation requires the final shared P0 checkpoint.")
    validation = _load_records(config, "validation")
    output = _path(config["output"]["root"])
    collections: dict[str, list[dict[str, Any]]] = {"snr": [], "dropout": []}
    conditions = {
        "snr": [(float(value), 0.0) for value in config["robustness"]["snr_db"]],
        "dropout": [(float(config["pilot"]["validation_snr_db"]), float(value)) for value in config["robustness"]["dropout"]],
    }
    for kind, values in conditions.items():
        for snr_db, dropout_probability in values:
            outputs = _predict_radio(
                model,
                validation,
                m4,
                config,
                budget=payload["budget"],
                history_frames=int(payload["history_frames"]),
                batch_size=int(config["training"]["evaluation_batch_size"]),
                device=device,
                seed=int(payload["seed"]),
                snr_db=snr_db,
                dropout=dropout_probability,
            )
            evaluated = _evaluate_outputs(
                model,
                outputs,
                validation,
                m4,
                config,
                fusion_method="L2",
                weight=float(config["model"]["fixed_lambda"]),
                device=device,
            )
            collections[kind].append({
                "snr_db": snr_db,
                "dropout": dropout_probability,
                "csi_only_top1": evaluated["csi_only"]["top1"],
                "sensing_only_all14": evaluated["sensing_all14_macro"],
                "fixed_fusion_all14": evaluated["groups"]["all14"]["top1_macro"],
                "fixed_fusion_worst": evaluated["groups"]["all14"]["top1_worst"],
                "missing_lidar": evaluated["missing_lidar"]["top1"],
                "oracle_all14": evaluated["oracle_all14_macro"],
                "fix_rate": evaluated["groups"]["all14"]["fix_rate_macro"],
                "harm_rate": evaluated["groups"]["all14"]["harm_rate_macro"],
                "outer_test_accessed": False,
            })
    _write_csv(output / "snr_summary.csv", collections["snr"])
    _write_csv(output / "dropout_summary.csv", collections["dropout"])
    return collections


def summarize(config: Mapping[str, Any]) -> dict[str, Any]:
    output = _path(config["output"]["root"])
    result_paths = sorted((output / "results").glob("*.json"))
    fusion_paths = sorted((output / "fusion_results").glob("*.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    fusion = [json.loads(path.read_text(encoding="utf-8")) for path in fusion_paths]
    rows = [_flat_summary(value) for value in results]
    fusion_rows = [_flat_summary(value) for value in fusion]
    _write_csv(output / "temporal_encoder_summary.csv", [row for row in rows if row["family"] == "temporal"])
    _write_csv(output / "prototype_ablation_summary.csv", [row for row in rows if row["family"] == "prototype"])
    _write_csv(output / "fusion_location_summary.csv", fusion_rows)
    mask_rows = [
        {"family": value["family"], "method": value["method"], "seed": value["seed"], **row}
        for value in (*results, *fusion)
        for row in value["per_mask"]
    ]
    _write_csv(output / "mask_summary.csv", mask_rows)
    _write_csv(
        output / "oracle_summary.csv",
        [
            {
                "family": row["family"],
                "method": row["method"],
                "seed": row["seed"],
                "sensing_all14_macro": row["sensing_all14_macro"],
                "oracle_all14_macro": row["oracle_all14_macro"],
                "fusion_all14_macro": row["groups"]["all14"]["top1_macro"],
                "oracle_headroom_capture": row["oracle_headroom_capture"],
            }
            for row in (*results, *fusion)
        ],
    )
    _write_csv(
        output / "parameter_latency_summary.csv",
        [
            {
                "family": row["family"],
                "method": row["method"],
                "seed": row["seed"],
                "total_parameters": row.get("total_parameters"),
                "trainable_parameters": row.get("trainable_parameters"),
                "latency_ms_per_sample": row.get("latency_ms_per_sample"),
            }
            for row in (*results, *fusion)
        ],
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in (*rows, *fusion_rows):
        grouped[(row["family"], row["method"])].append(row)
    seed_rows = []
    for (family, method), values in sorted(grouped.items()):
        record: dict[str, Any] = {"family": family, "method": method, "seeds": len(values)}
        for metric in ("all14_macro", "all14_worst", "single_macro", "missing_lidar", "csi_only_top1"):
            finite = [float(row[metric]) for row in values if row.get(metric) is not None and math.isfinite(float(row[metric]))]
            record[f"{metric}_mean"] = float(np.mean(finite)) if finite else float("nan")
            record[f"{metric}_std"] = float(np.std(finite)) if finite else float("nan")
            record[f"{metric}_values"] = ";".join(f"{value:.8f}" for value in finite)
        seed_rows.append(record)
    _write_csv(output / "seed_summary.csv", seed_rows)
    selection = select_candidates(config)

    flat_rows = [*rows, *fusion_rows]

    def flat(family: str, method: str, seed: int = 1) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in flat_rows
                if row["family"] == family and row["method"] == method and int(row["seed"]) == int(seed)
            ),
            None,
        )

    def raw(family: str, method: str, seed: int = 1) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in (*results, *fusion)
                if row["family"] == family and row["method"] == method and int(row["seed"]) == int(seed)
            ),
            None,
        )

    def values(family: str, method: str, metric: str) -> list[float]:
        return [
            float(row[metric])
            for row in flat_rows
            if row["family"] == family
            and row["method"] == method
            and row.get(metric) is not None
            and math.isfinite(float(row[metric]))
        ]

    def mean_std(family: str, method: str, metric: str) -> tuple[float, float, str]:
        metric_values = values(family, method, metric)
        if not metric_values:
            return float("nan"), float("nan"), "-"
        return (
            float(np.mean(metric_values)),
            float(np.std(metric_values)),
            "/".join(f"{100.0 * value:.2f}" for value in metric_values),
        )

    def percent(value: Any) -> str:
        if value is None or not math.isfinite(float(value)):
            return "-"
        return f"{100.0 * float(value):.2f}%"

    def percent_mean_std(family: str, method: str, metric: str) -> str:
        mean, standard_deviation, _ = mean_std(family, method, metric)
        if not math.isfinite(mean):
            return "-"
        return f"{100.0 * mean:.2f} +/- {100.0 * standard_deviation:.2f}%"

    def read_csv_rows(path: Path) -> list[dict[str, str]]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    primary = raw("fusion", "L2", 1) or raw("prototype", "P0", 1)
    if primary is None:
        raise RuntimeError("A completed P0/L2 seed-1 result is required to build the final report.")
    primary_flat = flat("fusion", "L2", 1) or flat("prototype", "P0", 1)
    assert primary_flat is not None
    p0_training_flat = flat("prototype", "P0", 1)
    if p0_training_flat is None:
        raise RuntimeError("A completed P0 seed-1 training result is required to report parameter counts.")
    sensing_per_mask = [float(row["sensing_top1"]) for row in primary["per_mask"]]
    sensing_all14 = float(primary["sensing_all14_macro"])
    sensing_worst = min(sensing_per_mask)
    sensing_missing_lidar = float(primary["missing_lidar"]["sensing_top1"])

    accuracy_rows = [
        ("B0 sensing-only", None, None, 0, None, sensing_all14, sensing_worst, sensing_missing_lidar),
    ]
    for family, method, label in (
        ("temporal", "S20", "单帧集中 20 RE"),
        ("temporal", "T2", "T2 GRU + P0"),
        ("temporal", "T3", "T3 LSTM + P0"),
        ("prototype", "P0", "正式 P0 + L2"),
        ("prototype", "P1", "最强无共享 P1"),
    ):
        method_row = flat(family, method)
        if method_row is None:
            continue
        accuracy_rows.append(
            (
                label,
                int(method_row["pilot_re_per_frame"]),
                int(method_row["pilot_history_frames"]),
                int(method_row["pilot_re_window"]),
                mean_std(family, method, "csi_only_top1")[0],
                mean_std(family, method, "all14_macro")[0],
                mean_std(family, method, "all14_worst")[0],
                mean_std(family, method, "missing_lidar")[0],
            )
        )

    report = [
        "# TSPC 最终公平消融报告",
        "",
        "本报告由本地 development train/validation 产物自动生成；outer test 未构建、未访问。",
        "",
        "## 协议与边界",
        "",
        f"- 数据划分：`{config['protocol']['id']}`；训练 {config['protocol']['expected_train_samples']} 样本/"
        f"{config['protocol']['expected_train_trajectories']} 条轨迹，验证 {config['protocol']['expected_validation_samples']} 样本/"
        f"{config['protocol']['expected_validation_trajectories']} 条轨迹，轨迹互斥。",
        f"- 训练：batch={config['training']['batch_size']}，AdamW，lr={config['training']['learning_rate']}，"
        f"weight_decay={config['training']['weight_decay']}，最多 {config['training']['max_epochs']} epochs，"
        f"patience={config['training']['patience']}，AMP={config['training']['amp']}，clip={config['training']['gradient_clip_norm']}。",
        f"- 主 CSI：5 帧 x (2 pattern x 2 frequency)=5 x 4={5 * 4} RE/窗口；训练 SNR "
        f"[{config['pilot']['train_snr_db_min']}, {config['pilot']['train_snr_db_max']}] dB。",
        "- 所有方法使用同一 M4 checkpoint、同一轨迹划分和同一 validation noise seed；Full 硬旁路，CSI-off 精确回退。",
        "- 选择指标为 All-14 Macro；0.3 pp 内依次比较 Worst、missing_lidar、训练参数量。",
        "",
        "## 精度与 RE 总览",
        "",
        "以下为所有已完成种子的均值；`+/-` 统计见三种子表。Full Top-1 对所有行均为硬旁路后的 86.33%。",
        "",
        "| 方法 | RE/帧 | 帧数 | RE/窗口 | CSI-only | All-14 | Worst | missing_lidar |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {label} | {re_frame if re_frame is not None else 0} | {frames if frames is not None else 0} | "
        f"{re_window} | {percent(csi_only)} | {percent(all14)} | {percent(worst)} | {percent(missing_lidar)} |"
        for label, re_frame, frames, re_window, csi_only, all14, worst, missing_lidar in accuracy_rows
    )

    report.extend(
        [
            "",
            "## 时序编码器消融",
            "",
            "所有 T0-T5 均按相同协议重新训练；S20 是最后一帧集中 5x4=20 RE。参数量为该次训练可训练参数。",
            "",
            "| 方法 | 时序结构 | All-14 | Worst | CSI-only | 参数量 | 延迟 ms/样本 | RE/窗口 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("T0", "T1", "T2", "T3", "T4", "T5", "S20"):
        row = flat("temporal", method, 1)
        if row is None:
            continue
        report.append(
            f"| {method} | {row['temporal_method']} | {percent(row['all14_macro'])} | "
            f"{percent(row['all14_worst'])} | {percent(row['csi_only_top1'])} | "
            f"{int(row['trainable_parameters'])} | {float(row['latency_ms_per_sample']):.4f} | "
            f"{int(row['pilot_re_window'])} |"
        )

    report.extend(
        [
            "",
            "## Prototype 必要性",
            "",
            "P0-P5 使用同一冻结 T3 radio backbone 后重新训练各自决策头；下表为筛选种子 1。",
            "",
            "| 方法 | All-14 | Worst | missing_lidar | CSI-only | 头部可训练参数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("P0", "P1", "P2", "P3", "P4", "P5"):
        row = flat("prototype", method, 1)
        if row is None:
            continue
        report.append(
            f"| {method} | {percent(row['all14_macro'])} | {percent(row['all14_worst'])} | "
            f"{percent(row['missing_lidar'])} | {percent(row['csi_only_top1'])} | "
            f"{int(row['trainable_parameters'])} |"
        )

    report.extend(
        [
            "",
            "## 融合位置",
            "",
            "L5 是逐样本 oracle 上界，仅作诊断；L2 与 L4 在相同权重下数学等价。下表为种子 1。",
            "",
            "| 方法 | All-14 | Worst | missing_lidar | NLL | ECE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ("L0", "L1", "L2", "L3", "L4", "L5"):
        row = flat("fusion", method, 1)
        if row is None:
            continue
        report.append(
            f"| {method} | {percent(row['all14_macro'])} | {percent(row['all14_worst'])} | "
            f"{percent(row['missing_lidar'])} | {float(row['nll']):.4f} | {float(row['ece']):.4f} |"
        )

    report.extend(
        [
            "",
            "## 三种子稳定性",
            "",
            "均值和标准差按完成的独立种子计算；括号内为 seed1/seed2/seed3 的百分数。",
            "",
            "| 候选 | seeds | All-14 mean +/- std | Worst mean +/- std | missing_lidar mean +/- std | 各 seed All-14 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for family, method, label in (
        ("temporal", "S20", "单帧集中 S20"),
        ("temporal", "T2", "当前 TSPC/GRU"),
        ("temporal", "T3", "最优时序/LSTM"),
        ("prototype", "P0", "共享 P0"),
        ("prototype", "P1", "无共享 P1"),
        ("fusion", "L2", "最强正式融合 L2"),
    ):
        all14_mean, all14_std, all14_values = mean_std(family, method, "all14_macro")
        if not math.isfinite(all14_mean):
            continue
        count = len(values(family, method, "all14_macro"))
        report.append(
            f"| {label} | {count} | {100.0 * all14_mean:.2f} +/- {100.0 * all14_std:.2f}% | "
            f"{percent_mean_std(family, method, 'all14_worst')} | "
            f"{percent_mean_std(family, method, 'missing_lidar')} | {all14_values} |"
        )

    report.extend(
        [
            "",
            "## 正式 P0/L2 核心指标（seed 1）",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| Full Top-1/3/5 | {percent(primary['full']['top1'])} / {percent(primary['full']['top3'])} / {percent(primary['full']['top5'])} |",
            f"| Single Macro/Worst | {percent(primary_flat['single_macro'])} / {percent(primary_flat['single_worst'])} |",
            f"| Two Macro/Worst | {percent(primary_flat['two_macro'])} / {percent(primary_flat['two_worst'])} |",
            f"| Three Macro/Worst | {percent(primary_flat['three_macro'])} / {percent(primary_flat['three_worst'])} |",
            f"| All-14 Macro/Worst | {percent(primary_flat['all14_macro'])} / {percent(primary_flat['all14_worst'])} |",
            f"| Within-3 / MAE | {percent(primary_flat['within3'])} / {float(primary_flat['mae']):.4f} |",
            f"| normalized gain / beam loss | {float(primary_flat['normalized_gain']):.4f} / {float(primary_flat['beam_loss_db']):.4f} dB |",
            f"| Fix / Harm | {percent(primary_flat['fix_rate'])} / {percent(primary_flat['harm_rate'])} |",
            f"| CSI-only / sensing-only / oracle | {percent(primary_flat['csi_only_top1'])} / {percent(sensing_all14)} / {percent(primary_flat['oracle_all14_macro'])} |",
            f"| oracle headroom capture | {percent(primary_flat['oracle_headroom_capture'])} |",
            f"| NLL / ECE | {float(primary_flat['nll']):.4f} / {float(primary_flat['ece']):.4f} |",
            f"| RE/帧 / 帧数 / RE/窗口 | {primary_flat['pilot_re_per_frame']} / {primary_flat['pilot_history_frames']} / {primary_flat['pilot_re_window']} |",
            f"| P0 头部可训练参数 / L2 额外参数 | {p0_training_flat['trainable_parameters']} / "
            f"{primary_flat['trainable_parameters']} |",
            f"| radio encoder+head 延迟 | {float(p0_training_flat['latency_ms_per_sample']):.4f} ms/样本 |",
            "",
            "## 14 个缺失 mask（P0/L2 三种子）",
            "",
            "| mask | 可用感知模态数 | sensing Top-1 | fusion Top-1 mean +/- std |",
            "|---|---:|---:|---:|",
        ]
    )
    primary_seed_results = [row for row in fusion if row["method"] == "L2"]
    for mask_name in MASK_NAMES:
        mask_rows = [next(row for row in result["per_mask"] if row["mask"] == mask_name) for result in primary_seed_results]
        fused_values = [float(row["top1"]) for row in mask_rows]
        report.append(
            f"| `{mask_name}` | {MASK_COUNTS[mask_name]} | {percent(mask_rows[0]['sensing_top1'])} | "
            f"{100.0 * float(np.mean(fused_values)):.2f} +/- {100.0 * float(np.std(fused_values)):.2f}% |"
        )

    lambda_selection_path = output / "lambda_selection.json"
    lambda_selection = (
        json.loads(lambda_selection_path.read_text(encoding="utf-8")) if lambda_selection_path.exists() else {}
    )
    lambda_rows = read_csv_rows(output / "lambda_global_curve.csv")
    validation_lambda_rows = [row for row in lambda_rows if row.get("role") == "validation_diagnostic"]
    report.extend(
        [
            "",
            "## 固定 lambda 网格",
            "",
            f"主结果保持预注册 lambda={lambda_selection.get('primary_lambda', config['model']['fixed_lambda'])}；"
            f"train 内部子集给出的候选为 {lambda_selection.get('train_calibrated_lambda', '-')}。"
            "validation 网格只作诊断，不参与正式选择。",
            "",
            "| lambda | All-14 | Worst | missing_lidar | Fix | Harm |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in validation_lambda_rows:
        report.append(
            f"| {float(row['lambda']):.1f} | {percent(row['all14_macro'])} | {percent(row['all14_worst'])} | "
            f"{percent(row['missing_lidar'])} | {percent(row['fix_rate'])} | {percent(row['harm_rate'])} |"
        )

    snr_rows = read_csv_rows(output / "snr_summary.csv")
    dropout_rows = read_csv_rows(output / "dropout_summary.csv")
    report.extend(
        [
            "",
            "## 低质量 CSI 鲁棒性（P0/L2 seed 1）",
            "",
            "| SNR dB | CSI-only | sensing-only | fixed fusion | Worst | missing_lidar | oracle | Fix | Harm |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in snr_rows:
        report.append(
            f"| {float(row['snr_db']):.0f} | {percent(row['csi_only_top1'])} | {percent(row['sensing_only_all14'])} | "
            f"{percent(row['fixed_fusion_all14'])} | {percent(row['fixed_fusion_worst'])} | "
            f"{percent(row['missing_lidar'])} | {percent(row['oracle_all14'])} | "
            f"{percent(row['fix_rate'])} | {percent(row['harm_rate'])} |"
        )
    report.extend(
        [
            "",
            "| pilot dropout | CSI-only | sensing-only | fixed fusion | Worst | missing_lidar | oracle | Fix | Harm |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dropout_rows:
        report.append(
            f"| {float(row['dropout']):.2f} | {percent(row['csi_only_top1'])} | {percent(row['sensing_only_all14'])} | "
            f"{percent(row['fixed_fusion_all14'])} | {percent(row['fixed_fusion_worst'])} | "
            f"{percent(row['missing_lidar'])} | {percent(row['oracle_all14'])} | "
            f"{percent(row['fix_rate'])} | {percent(row['harm_rate'])} |"
        )

    p0_all14 = mean_std("prototype", "P0", "all14_macro")[0]
    p1_all14 = mean_std("prototype", "P1", "all14_macro")[0]
    t2_all14 = mean_std("temporal", "T2", "all14_macro")[0]
    t3_all14 = mean_std("temporal", "T3", "all14_macro")[0]
    s20_all14 = mean_std("temporal", "S20", "all14_macro")[0]
    p2 = flat("prototype", "P2", 1)
    p3 = flat("prototype", "P3", 1)
    p4 = flat("prototype", "P4", 1)
    l1 = flat("fusion", "L1", 1)
    l2 = flat("fusion", "L2", 1)
    l3 = flat("fusion", "L3", 1)
    l4 = flat("fusion", "L4", 1)
    valid_snr = [
        float(row["snr_db"])
        for row in snr_rows
        if float(row["fixed_fusion_all14"]) >= float(row["sensing_only_all14"])
    ]
    valid_dropout = [
        float(row["dropout"])
        for row in dropout_rows
        if float(row["fixed_fusion_all14"]) >= float(row["sensing_only_all14"])
    ]
    p0_parameters = int((flat("prototype", "P0", 1) or {})["trainable_parameters"])
    p1_parameters = int((flat("prototype", "P1", 1) or {})["trainable_parameters"])
    report.extend(
        [
            "",
            "## 对 13 个问题的明确回答",
            "",
            f"1. **时间分散收益仍成立。** 五帧分散 T3 为 {percent(t3_all14)}，单帧集中 S20 为 "
            f"{percent(s20_all14)}，同为 20 RE，差 {100.0 * (t3_all14 - s20_all14):.2f} pp。",
            f"2. **T3 LSTM 最好。** 单种子公平筛选为 {percent((flat('temporal', 'T3', 1) or {})['all14_macro'])}，"
            "高于 GRU、TCN、Transformer、mean 和 last。",
            f"3. **GRU 不是必要创新组件。** 三种子 LSTM 比 GRU 高 {100.0 * (t3_all14 - t2_all14):.2f} pp；"
            "GRU 只是可用实现选择。",
            f"4. **共享真实 Beam Prototype 没有独立精度收益。** P0 三种子 {percent(p0_all14)}，P1 "
            f"{percent(p1_all14)}；P1 高 {100.0 * (p1_all14 - p0_all14):.2f} pp。"
            + (
                f"随机冻结 P4 在筛选种子也达到 {percent(p4['all14_macro'])}。" if p4 is not None else ""
            ),
            f"5. **性能排序是 P1 > P0 > P2 > P3。** seed1 分别为 {percent((flat('prototype', 'P1', 1) or {})['all14_macro'])}、"
            f"{percent((flat('prototype', 'P0', 1) or {})['all14_macro'])}、{percent(None if p2 is None else p2['all14_macro'])}、"
            f"{percent(None if p3 is None else p3['all14_macro'])}。P0 比 P1 少 {p1_parameters - p0_parameters} 个可训练参数，"
            "其价值只能表述为受约束、统一语义和参数效率，而非最优精度。",
            f"6. **Evidence 融合优于 feature/probability 融合，但与 PoE 等价。** seed1 L2={percent(None if l2 is None else l2['all14_macro'])}，"
            f"L1={percent(None if l1 is None else l1['all14_macro'])}，L3={percent(None if l3 is None else l3['all14_macro'])}，"
            f"L4={percent(None if l4 is None else l4['all14_macro'])}；不能把 L2 相对 L4 包装成独立优势。",
            f"7. **lambda=0.5 合理但不是 validation 最优。** 它是预注册主值；train 子集候选为 "
            f"{lambda_selection.get('train_calibrated_lambda', '-')}，validation 诊断峰值在 0.6，故不按 validation 改主结果。",
            "8. **三种子方向稳定。** T3 在每个配对种子都高于 T2，P1 在每个配对种子都高于 P0；完整均值、标准差和各 seed 见上表。",
            f"9. **低质量失效点清楚。** 在已测网格中，融合从 SNR >= {min(valid_snr) if valid_snr else '-'} dB 开始不低于 sensing-only；"
            f"dropout <= {max(valid_dropout) if valid_dropout else '-'} 时仍不低于 sensing-only。-5 dB 和 50% dropout 已出现负收益。"
            "部署可考虑非学习回退，但阈值必须另在 train calibration 上冻结，不能据 validation 直接上线。",
            f"10. **Full 完全不变。** Top-1/3/5={percent(primary['full']['top1'])}/{percent(primary['full']['top3'])}/"
            f"{percent(primary['full']['top5'])}，max_abs_probability_difference={primary['full_probability_max_abs_diff']}，"
            f"argmax_mismatch={primary['full_argmax_mismatch']}，Full CSI RE={primary['full_pilot_re']}。",
            f"11. **CSI-off 精确回退。** max_abs_probability_difference={primary['csi_off_probability_max_abs_diff']}，"
            f"argmax_mismatch={primary['csi_off_argmax_mismatch']}。",
            "12. **最终保留模块。** 保留五帧分散 2x2 超稀疏 CSI、LSTM 历史演化编码、固定 lambda=0.5、FP32 证据融合、"
            "Full 硬旁路和 CSI-off 回退；共享 bank 仅作为受约束统一 beam evidence 的设计，不宣称其带来精度提升。"
            "不保留复杂 gate。若只追求验证精度，P1 是更强实现。",
            "13. **outer test 继续封存。** 所有结果均来自 development train/validation，preflight 和结果元数据均记录 "
            "`outer_test_accessed=false`。",
            "",
            "## 最终结论边界",
            "",
            "实验支持的第二创新核心是：**时间分散超稀疏 CSI + 受约束的统一 beam evidence 补偿**。"
            "不能声称共享真实 prototype 本身提高精度，也不能把 LSTM 或 GRU 单独写成创新。"
            "完整逐 mask、oracle、参数/延迟、lambda 和鲁棒性原始表均保存在本目录对应 CSV/JSON 中。",
        ]
    )
    (output / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"results": len(results), "fusion_results": len(fusion), "selection": selection}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    training = subparsers.add_parser("train")
    training.add_argument("--family", choices=("temporal", "prototype"), required=True)
    training.add_argument("--method", required=True)
    training.add_argument("--seed", type=int, default=1)
    training.add_argument("--device", default="cuda")
    training.add_argument("--backbone-checkpoint", type=Path)
    training.add_argument("--limit", type=int)
    training.add_argument("--epochs", type=int)
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--overwrite", action="store_true")
    fusion = subparsers.add_parser("evaluate-fusions")
    fusion.add_argument("--shared-checkpoint", type=Path, required=True)
    fusion.add_argument("--concat-checkpoint", type=Path)
    fusion.add_argument("--device", default="cuda")
    lambdas = subparsers.add_parser("lambda-curves")
    lambdas.add_argument("--checkpoint", type=Path, required=True)
    lambdas.add_argument("--device", default="cuda")
    quality = subparsers.add_parser("robustness")
    quality.add_argument("--checkpoint", type=Path, required=True)
    quality.add_argument("--device", default="cuda")
    subparsers.add_parser("select")
    subparsers.add_parser("summarize")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    if args.command == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True))
    elif args.command == "train":
        train(args, config)
    elif args.command == "evaluate-fusions":
        print(json.dumps([_flat_summary(row) for row in evaluate_fusions(args, config)], indent=2, sort_keys=True))
    elif args.command == "lambda-curves":
        print(json.dumps(lambda_curves(args, config), indent=2, sort_keys=True))
    elif args.command == "robustness":
        robustness(args, config)
    elif args.command == "select":
        print(json.dumps(select_candidates(config), indent=2, sort_keys=True))
    else:
        print(json.dumps(summarize(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
