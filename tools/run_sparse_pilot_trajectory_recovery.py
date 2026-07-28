#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from kd_sensing.baselines.full_pool_common import sha256_file
from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel
from kd_sensing.baselines.sparse_pilot_transition import SparsePilotInformationClassifier
from kd_sensing.channel.pilot_cache import PilotCache, PilotCacheSpec
from kd_sensing.channel.probe_codebook import generate_probe_codebook
from kd_sensing.channel.sparse_pilot_simulator import frequency_offsets_hz, pilot_subcarrier_indices
from kd_sensing.config.io import dump_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.mmw.trajectory_protocol import load_trajectory_protocol
from kd_sensing.data.transform_ops.gps import load_gps_scaler
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
if __package__:
    from .run_mmw_trajectory_baselines import ALL_PATTERNS, _fixed_loader, _inputs, build_config
    from .run_sparse_pilot_recovery import (
        _class_counts,
        _gradient_norm,
        _noisy_observations,
        _prediction_metrics,
        _save_checkpoint,
        _simulate_history,
        _write_csv,
    )
else:
    from run_mmw_trajectory_baselines import ALL_PATTERNS, _fixed_loader, _inputs, build_config
    from run_sparse_pilot_recovery import (
        _class_counts,
        _gradient_norm,
        _noisy_observations,
        _prediction_metrics,
        _save_checkpoint,
        _simulate_history,
        _write_csv,
    )


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "I1": {"input_time": "t", "target_time": "t", "history_length": 1, "concat": False},
    "I2": {"input_time": "t", "target_time": "t+1", "history_length": 1, "concat": False},
    "I3": {"input_time": "t-4:t", "target_time": "t+1", "history_length": 5, "concat": False},
    "I4": {"input_time": "M4+CSI_t", "target_time": "t+1", "history_length": 1, "concat": True},
    "I5": {"input_time": "M4+CSI_t-4:t", "target_time": "t+1", "history_length": 5, "concat": True},
}
SEVERE_MASKS = ("image_only", "lidar_only", "radar_only", "gps_only")
ALL_MISSING_MASKS = tuple(name for name in ALL_PATTERNS if name != "full")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_budget(value: str) -> tuple[int, int]:
    try:
        patterns, frequencies = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Pilot budget must use MxK syntax, for example 8x16.") from exc
    if patterns <= 0 or frequencies <= 0:
        raise ValueError("Pilot budget dimensions must be positive.")
    return patterns, frequencies


def nested_frequency_indices(maximum: int, count: int) -> torch.Tensor:
    if count <= 0 or count > maximum:
        raise ValueError("Nested frequency count must be in [1, maximum].")
    indices = np.rint(np.linspace(0, maximum - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise ValueError("Nested frequency construction produced duplicate indices.")
    return torch.from_numpy(indices)


def _load_m4(config: Mapping[str, Any], device: torch.device) -> tuple[TrajectoryBaselineModel, dict[str, Any]]:
    section = config["protocol"]
    trajectory_root = _path(section["trajectory_root"])
    manifest_path = trajectory_root / "protocol/split_manifest.json"
    protocol = load_trajectory_protocol(manifest_path)
    checkpoint_path = _path(section["checkpoint"])
    actual_sha = sha256_file(checkpoint_path)
    if actual_sha != section["checkpoint_sha256"]:
        raise ValueError("Trajectory M4 checkpoint SHA256 mismatch.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("method") != ABTC_METHOD:
        raise ValueError("Trajectory recovery requires the published M4 checkpoint.")
    if payload.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
        raise ValueError("Trajectory M4 checkpoint protocol fingerprint mismatch.")
    if payload.get("split_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Trajectory M4 checkpoint split manifest mismatch.")
    model = TrajectoryBaselineModel(ABTC_METHOD, **payload.get("model_config", {})).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "protocol": protocol,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": actual_sha,
        "checkpoint_best_epoch": int(payload["best_epoch"]),
    }


def _loaders(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    section = config["protocol"]
    trajectory_root = _path(section["trajectory_root"])
    protocol = load_trajectory_protocol(trajectory_root / "protocol/split_manifest.json")
    cfg = build_config(protocol, trajectory_root)
    cfg["data"]["dataset"].update(
        include_channel_ref=True,
        include_channel_history_refs=True,
        include_router_utility_targets=True,
        pilot_time_mode="last_input",
    )
    cfg["data"]["dataloader"].update(num_workers=4, persistent_workers=True, pin_memory=True)
    scaler = load_gps_scaler(trajectory_root / "artifacts/gps_scaler.npz")
    loaders = build_dataloaders(cfg, normalization_overrides={"gps_scaler": scaler})
    expected = {
        "train": int(section["expected_train_samples"]),
        "validation": int(section["expected_validation_samples"]),
    }
    if set(loaders) != set(expected) or any(len(loaders[role].dataset) != count for role, count in expected.items()):
        raise ValueError("Trajectory recovery loader split/count mismatch or test loader exposure.")
    return loaders, cfg


@torch.no_grad()
def _extract_role(
    loader,
    model: TrajectoryBaselineModel,
    *,
    config: Mapping[str, Any],
    device: torch.device,
    codebook,
    frequencies: np.ndarray,
    cache: PilotCache,
    cache_spec: PilotCacheSpec,
    role: str,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    keys = (
        "labels_current",
        "labels_future",
        "current_beam_power",
        "future_beam_power",
        "candidate_history",
        *(f"z_{name}" for name in ALL_PATTERNS),
        *(f"p0_{name}" for name in ALL_PATTERNS),
    )
    chunks: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    sample_ids: list[str] = []
    prepared_alias_mismatches = 0
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, 1):
        inputs = _inputs(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tokens = model.encode(inputs)
            count = len(batch["target_beam"])
            for name, mask in ALL_PATTERNS.items():
                output = model.forward_tokens(
                    tokens,
                    availability=torch.tensor(mask, device=device, dtype=torch.bool).expand(count, -1),
                )
                chunks[f"z_{name}"].append(output["fused_features"].float().cpu())
                chunks[f"p0_{name}"].append(output["logits"].float().softmax(dim=-1).cpu())
        current_power = torch.as_tensor(batch["current_beam_power"]).float()
        current_labels = current_power.argmax(dim=-1).long()
        prepared_alias_mismatches += int(
            torch.as_tensor(batch["prepared_beam_label"]).reshape(-1).long().ne(current_labels).sum().item()
        )
        chunks["labels_current"].append(current_labels)
        chunks["labels_future"].append(torch.as_tensor(batch["target_beam"]).reshape(-1).long())
        chunks["current_beam_power"].append(current_power)
        chunks["future_beam_power"].append(torch.as_tensor(batch["future_beam_power"]).float())
        chunks["candidate_history"].append(
            _simulate_history(
                batch["channel_history_refs"],
                codebook=codebook,
                frequencies=frequencies,
                cache=cache,
                cache_spec=cache_spec,
            )
        )
        sample_ids.extend(str(value) for value in batch["metadata"]["stable_sample_id"])
        if batch_index % 25 == 0 or batch_index == total_batches:
            status = {
                "status": "extracting",
                "role": role,
                "completed_batches": batch_index,
                "total_batches": total_batches,
                "outer_test_accessed": False,
            }
            (output_root / "prepare_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(json.dumps(status), flush=True)
    records = {key: torch.cat(values) for key, values in chunks.items()}
    records["sample_ids"] = sample_ids
    audit = {
        "sample_count": len(sample_ids),
        "sample_id_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "prepared_beam_label_alias_mismatch_count": prepared_alias_mismatches,
        "current_class_counts": _class_counts(records["labels_current"]),
        "future_class_counts": _class_counts(records["labels_future"]),
        "baseline": _baseline_summary(records),
    }
    return records, audit


def _baseline_summary(records: Mapping[str, Any]) -> dict[str, Any]:
    labels = records["labels_future"]
    power = records["future_beam_power"]
    result = {}
    for name in ALL_PATTERNS:
        result[name] = _prediction_metrics(records[f"p0_{name}"], labels, power, records[f"p0_{name}"])
    return result


def prepare(args: argparse.Namespace, config: dict[str, Any]) -> None:
    output_root = args.output_root
    record_dir = output_root / "records"
    if record_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Trajectory recovery records already exist: {record_dir}")
    record_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, identity = _load_m4(config, device)
    loaders, resolved_model_cfg = _loaders(config)
    fixed = {role: _fixed_loader(loader, workers=4) for role, loader in loaders.items()}
    pilot = config["pilot"]
    codebook = generate_probe_codebook(
        64,
        16,
        num_patterns=int(pilot["max_patterns"]),
        seed=int(pilot["codebook_seed"]),
        method=str(pilot["codebook_method"]),
    )
    codebook_path = output_root / "pilot_codebook_32x16.npz"
    codebook.save(codebook_path)
    subcarriers = pilot_subcarrier_indices(
        int(pilot["num_subcarriers"]), int(pilot["max_pilot_subcarriers"])
    )
    frequencies = frequency_offsets_hz(
        subcarriers,
        num_subcarriers=int(pilot["num_subcarriers"]),
        subcarrier_spacing_hz=float(pilot["subcarrier_spacing_hz"]),
        mode=str(pilot["frequency_index_mode"]),
    )
    cache = PilotCache(_path(config["output"]["cache_root"]))
    cache_spec = PilotCacheSpec(
        codebook.hash,
        tuple(frequencies),
        float(pilot["subcarrier_spacing_hz"]),
        str(pilot["frequency_index_mode"]),
        64,
        16,
    )
    audits = {}
    try:
        for role in ("train", "validation"):
            records, audits[role] = _extract_role(
                fixed[role],
                model,
                config=config,
                device=device,
                codebook=codebook,
                frequencies=frequencies,
                cache=cache,
                cache_spec=cache_spec,
                role=role,
                output_root=output_root,
            )
            torch.save(records, record_dir / f"{role}.pt")
    finally:
        for loader in (*fixed.values(), *loaders.values()):
            shutdown_dataloader_workers(loader)
    expected_full = float(config["protocol"]["expected_full_top1"])
    actual_full = float(audits["validation"]["baseline"]["full"]["top1"])
    if abs(actual_full - expected_full) > 1e-7:
        raise ValueError(f"Published trajectory M4 Full Top-1 mismatch: {actual_full} vs {expected_full}.")
    resolved = config | {
        "runtime": identity
        | {
            "codebook_hash": codebook.hash,
            "subcarrier_indices": subcarriers.tolist(),
            "frequency_positions_hz": frequencies.tolist(),
            "outer_test_accessed": False,
            "future_channel_used_as_input": False,
            "test_loader_constructed": False,
        },
        "trajectory_model_config": resolved_model_cfg,
    }
    dump_config(resolved, output_root / "resolved_configs/prepare.yaml")
    audit = {
        "status": "passed",
        "identity": identity,
        "roles": audits,
        "published_full_top1": expected_full,
        "recomputed_full_top1": actual_full,
        "outer_test_accessed": False,
        "test_loader_constructed": False,
        "legacy_full_pool_results_eligible": False,
    }
    (output_root / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "audit.md").write_text(
        "# Trajectory Recovery Audit\n\n"
        f"- Protocol: `{identity['protocol']['protocol_id']}` / `{identity['protocol']['protocol_fingerprint']}`\n"
        f"- Split: {audits['train']['sample_count']} train / {audits['validation']['sample_count']} validation\n"
        f"- M4 Full Top-1: published={expected_full:.8f}, recomputed={actual_full:.8f}\n"
        f"- Codebook: `{codebook.hash}`, maximum observation=32x16\n"
        "- Test loader/prediction: not constructed / not accessed\n"
        "- Old Full-pool recovery results: protocol-invalid for this question\n",
        encoding="utf-8",
    )
    status = {"status": "prepared", "roles": audits, "outer_test_accessed": False}
    (output_root / "prepare_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


def _task_records(
    records: Mapping[str, Any],
    task: str,
    budget: tuple[int, int],
    *,
    max_frequencies: int,
    masks: Sequence[str] | None = None,
) -> dict[str, Any]:
    spec = TASKS[task]
    patterns, frequency_count = budget
    frequency_indices = nested_frequency_indices(max_frequencies, frequency_count)
    candidates = records["candidate_history"][:, -int(spec["history_length"]) :, :patterns]
    candidates = candidates.index_select(-1, frequency_indices)
    labels = records["labels_current"] if spec["target_time"] == "t" else records["labels_future"]
    power = records["current_beam_power"] if spec["target_time"] == "t" else records["future_beam_power"]
    if not spec["concat"]:
        output = {"candidate_history": candidates, "labels": labels, "beam_power": power}
        if spec["target_time"] == "t+1":
            output["p0"] = records["p0_full"]
        return output
    if not masks:
        raise ValueError("Concat task records require at least one sensing mask.")
    names = tuple(str(name) for name in masks)
    return {
        "candidate_history": candidates.repeat(len(names), 1, 1, 1),
        "labels": labels.repeat(len(names)),
        "beam_power": power.repeat(len(names), 1),
        "sensing_feature": torch.cat([records[f"z_{name}"] for name in names]),
        "p0": torch.cat([records[f"p0_{name}"] for name in names]),
        "is_full": torch.cat(
            [torch.full((len(labels),), name == "full", dtype=torch.bool) for name in names]
        ),
        "physical_sample_count": torch.tensor(len(labels)),
        "mask_names": names,
    }


def _batch_forward(
    model: SparsePilotInformationClassifier,
    batch: Mapping[str, torch.Tensor],
    *,
    frequencies: torch.Tensor,
    snr: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    candidates = batch["candidate_history"]
    expanded_snr = snr[:, None].expand(-1, candidates.shape[1])
    observations, valid = _noisy_observations(candidates, expanded_snr, generator=generator)
    pattern_ids = torch.arange(candidates.shape[2], device=candidates.device).expand(
        candidates.shape[0], candidates.shape[1], -1
    )
    return model(
        observations,
        pattern_ids,
        frequencies,
        valid,
        expanded_snr,
        sensing_feature=batch.get("sensing_feature"),
        base_probabilities=batch.get("p0"),
    )


@torch.no_grad()
def _predict_probabilities(
    model: SparsePilotInformationClassifier,
    records: Mapping[str, Any],
    *,
    frequencies: torch.Tensor,
    snr_db: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probabilities, entropies = [], []
    generator = torch.Generator(device=device).manual_seed(100_000 + int(seed))
    for start in range(0, len(records["labels"]), int(batch_size)):
        stop = start + int(batch_size)
        batch = {
            key: value[start:stop].to(device)
            for key, value in records.items()
            if torch.is_tensor(value) and value.ndim > 0
        }
        snr = torch.full((len(batch["labels"]),), float(snr_db), device=device)
        output = _batch_forward(model, batch, frequencies=frequencies, snr=snr, generator=generator)
        probabilities.append(output["logits"].softmax(dim=-1).cpu())
        entropies.append(output["q_entropy"].cpu())
    probs = torch.cat(probabilities)
    entropy = torch.cat(entropies)
    return probs, entropy


@torch.no_grad()
def evaluate(
    model: SparsePilotInformationClassifier,
    records: Mapping[str, Any],
    *,
    frequencies: torch.Tensor,
    snr_db: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    probs, entropy = _predict_probabilities(
        model,
        records,
        frequencies=frequencies,
        snr_db=snr_db,
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    labels = records["labels"]
    power = records["beam_power"]
    base = records.get("p0")
    loss = float(F.nll_loss(probs.clamp_min(1e-12).log(), labels).item())
    physical_count = int(records.get("physical_sample_count", torch.tensor(0)).item())
    if not physical_count:
        metrics = _prediction_metrics(probs, labels, power, base)
        return metrics | {"validation_loss": loss, "q_entropy": float(entropy.mean().item())}
    per_mask = []
    for index, name in enumerate(records["mask_names"]):
        subset = slice(index * physical_count, (index + 1) * physical_count)
        per_mask.append(
            {"mask": name}
            | _prediction_metrics(
                probs[subset], labels[subset], power[subset], base[subset] if base is not None else None
            )
        )
    metric_names = (
        "top1",
        "top3",
        "top5",
        "within3",
        "mae",
        "normalized_gain",
        "beam_loss_db",
        "fix_rate",
        "harm_rate",
        "p_final_p0_kl",
    )
    aggregate = {
        key: float(np.mean([float(row[key]) for row in per_mask if row[key] is not None]))
        for key in metric_names
    }
    return aggregate | {
        "validation_loss": loss,
        "worst_top1": min(float(row["top1"]) for row in per_mask),
        "q_entropy": float(entropy.mean().item()),
        "per_mask": per_mask,
    }


def _fallback_probabilities(
    csi_probabilities: torch.Tensor,
    base_probabilities: torch.Tensor,
    mask_name: str,
) -> torch.Tensor:
    available = sum(int(value) for value in ALL_PATTERNS[mask_name])
    return csi_probabilities if available <= 2 else base_probabilities


def _aggregate_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics = (
        "top1",
        "top3",
        "top5",
        "within3",
        "mae",
        "normalized_gain",
        "beam_loss_db",
        "fix_rate",
        "harm_rate",
        "p_final_p0_kl",
    )
    return {
        key: float(np.mean([float(row[key]) for row in rows if row.get(key) is not None]))
        for key in metrics
    }


def _validate_run(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[int, int], str]:
    if args.round_name not in config["rounds"]:
        raise ValueError(f"Unknown recovery round: {args.round_name}")
    round_cfg = config["rounds"][args.round_name]
    if args.task not in round_cfg["tasks"] or args.seed is None or args.budget is None:
        raise ValueError("Train mode requires a round-allowed --task, --seed and --budget.")
    budget = parse_budget(args.budget)
    maximum = (int(config["pilot"]["max_patterns"]), int(config["pilot"]["max_pilot_subcarriers"]))
    if budget[0] > maximum[0] or budget[1] > maximum[1]:
        raise ValueError("Requested budget exceeds the prepared mother observation.")
    configured = round_cfg.get("budgets")
    if configured and args.budget not in configured:
        raise ValueError(f"Budget {args.budget} is not registered for {args.round_name}.")
    fusion_mode = str(round_cfg["fusion_mode"])
    return round_cfg, budget, fusion_mode


def _initialize_model(
    model: SparsePilotInformationClassifier,
    checkpoint: Path | None,
    *,
    fusion_mode: str,
) -> dict[str, Any]:
    if checkpoint is None:
        return {"mode": "random", "checkpoint": None, "checkpoint_sha256": None}
    source = _path(checkpoint)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    if fusion_mode == "replace":
        model.load_state_dict(state, strict=True)
        loaded = list(state)
        mode = "dense_to_sparse_full_state"
    else:
        prefixes = ("csi_encoder.", "temporal.", "classifier.0.", "classifier.1.")
        transferable = {
            key: value
            for key, value in state.items()
            if key.startswith(prefixes)
            and key in model.state_dict()
            and model.state_dict()[key].shape == value.shape
        }
        model.load_state_dict(transferable, strict=False)
        if not transferable:
            raise ValueError("Residual initialization did not find transferable CSI state.")
        if bool(model.classifier[-1].weight.detach().count_nonzero()) or bool(
            model.classifier[-1].bias.detach().count_nonzero()
        ):
            raise ValueError("Residual output layer must remain zero after initialization.")
        loaded = list(transferable)
        mode = "sparse_to_zero_residual_partial_state"
    return {
        "mode": mode,
        "checkpoint": str(source.resolve()),
        "checkpoint_sha256": sha256_file(source),
        "source_epoch": int(payload["epoch"]),
        "source_selection_metric": payload.get("selection_metric"),
        "loaded_state_keys": loaded,
    }


def train(args: argparse.Namespace, config: dict[str, Any]) -> None:
    round_cfg, budget, fusion_mode = _validate_run(args, config)
    task, seed = str(args.task), int(args.seed)
    stem = f"seed{seed}_{task}_{args.budget}_{fusion_mode}"
    round_root = args.output_root / args.round_name
    result_path = round_root / "results" / f"{stem}.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"Trajectory recovery result already exists: {result_path}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_raw = torch.load(args.output_root / "records/train.pt", map_location="cpu", weights_only=False)
    validation_raw = torch.load(
        args.output_root / "records/validation.pt", map_location="cpu", weights_only=False
    )
    max_frequencies = int(config["pilot"]["max_pilot_subcarriers"])
    spec = TASKS[task]
    train_masks = (*SEVERE_MASKS, "full") if fusion_mode == "residual" else SEVERE_MASKS
    train_records = _task_records(
        train_raw,
        task,
        budget,
        max_frequencies=max_frequencies,
        masks=train_masks if spec["concat"] else None,
    )
    validation_records = _task_records(
        validation_raw,
        task,
        budget,
        max_frequencies=max_frequencies,
        masks=SEVERE_MASKS if spec["concat"] else None,
    )
    all14_records = (
        _task_records(
            validation_raw,
            task,
            budget,
            max_frequencies=max_frequencies,
            masks=ALL_MISSING_MASKS,
        )
        if spec["concat"]
        else None
    )
    full_records = (
        _task_records(validation_raw, task, budget, max_frequencies=max_frequencies, masks=("full",))
        if spec["concat"]
        else None
    )
    device = torch.device(args.device)
    model = SparsePilotInformationClassifier(
        history_length=int(spec["history_length"]),
        sensing_dim=64 if spec["concat"] else 0,
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_candidate_patterns=int(config["pilot"]["max_patterns"]),
        encoder_layers=int(config["model"]["encoder_layers"]),
        fusion_mode=fusion_mode if spec["concat"] else "replace",
    ).to(device)
    initialization = _initialize_model(model, args.init_checkpoint, fusion_mode=fusion_mode)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["max_epochs"])
    )
    prepared = safe_load_yaml(
        (args.output_root / "resolved_configs/prepare.yaml").read_text(encoding="utf-8")
    )
    frequency_indices = nested_frequency_indices(max_frequencies, budget[1])
    frequencies = torch.tensor(prepared["runtime"]["frequency_positions_hz"], device=device).index_select(
        0, frequency_indices.to(device)
    )
    resolved = prepared | {
        "training": dict(training),
        "run": {
            "round": args.round_name,
            "task": task,
            "seed": seed,
            "budget": args.budget,
            "pilot_re": budget[0] * budget[1],
            "fusion_mode": fusion_mode,
            "train_masks": list(train_masks) if spec["concat"] else [],
            "frequency_indices": frequency_indices.tolist(),
            "device": str(device),
            "initialization": initialization,
            **spec,
        }
    }
    dump_config(resolved, round_root / "resolved_configs" / f"{stem}.yaml")
    history_rows: list[dict[str, Any]] = []
    checkpoint_dir = round_root / "checkpoints" / stem
    best = {
        "val_loss": float("inf"),
        "single_macro": float("-inf"),
        "single_worst": float("-inf"),
        "fix_rate": float("-inf"),
    }
    patience_count = 0
    stop_reason = "max_epochs"
    train_generator = torch.Generator(device=device).manual_seed(10_000 + seed)
    order_generator = torch.Generator().manual_seed(20_000 + seed)
    batch_size = int(training["batch_size"])
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_records["labels"]), generator=order_generator)
        loss_sum = ce_sum = preserve_sum = grad_encoder = grad_head = 0.0
        seen = batches = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = {
                key: value[indices].to(device)
                for key, value in train_records.items()
                if torch.is_tensor(value) and value.ndim > 0
            }
            snr = torch.empty(len(batch["labels"]), device=device).uniform_(
                float(training["snr_db_min"]),
                float(training["snr_db_max"]),
                generator=train_generator,
            )
            output = _batch_forward(model, batch, frequencies=frequencies, snr=snr, generator=train_generator)
            ce = F.cross_entropy(output["logits"], batch["labels"])
            preserve = ce.new_zeros(())
            full = batch.get("is_full")
            if fusion_mode == "residual" and full is not None and bool(full.any()):
                preserve = F.kl_div(
                    F.log_softmax(output["logits"][full], dim=-1),
                    batch["p0"][full],
                    reduction="batchmean",
                )
            loss = ce + float(training["full_preserve_weight"]) * preserve
            optimizer.zero_grad()
            loss.backward()
            grad_encoder += float(_gradient_norm(model.csi_encoder) or 0.0)
            grad_head += float(_gradient_norm(model.classifier) or 0.0)
            optimizer.step()
            count = len(batch["labels"])
            seen += count
            batches += 1
            loss_sum += float(loss.detach()) * count
            ce_sum += float(ce.detach()) * count
            preserve_sum += float(preserve.detach()) * count
        primary = evaluate(
            model,
            validation_records,
            frequencies=frequencies,
            snr_db=float(training["validation_snr_db"]),
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        single_macro = float(primary["top1"])
        single_worst = float(primary.get("worst_top1", primary["top1"]))
        row = {
            "epoch": epoch,
            "train_total_loss": loss_sum / seen,
            "train_final_ce": ce_sum / seen,
            "train_full_preserve_kl": preserve_sum / seen,
            "validation_loss": primary["validation_loss"],
            "validation_top1": primary["top1"],
            "validation_top3": primary["top3"],
            "validation_top5": primary["top5"],
            "validation_within3": primary["within3"],
            "validation_mae": primary["mae"],
            "single_macro": single_macro,
            "single_worst": single_worst,
            "all14_macro": None,
            "all14_worst": None,
            "full_top1": None,
            "fix_rate": primary["fix_rate"],
            "harm_rate": primary["harm_rate"],
            "p_final_p0_kl": primary["p_final_p0_kl"],
            "q_transition_entropy": primary["q_entropy"],
            "csi_encoder_gradient_norm": grad_encoder / batches,
            "transition_gradient_norm": grad_head / batches,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history_rows.append(row)
        _write_csv(round_root / "training_logs" / f"{stem}.csv", history_rows)
        print(json.dumps({"event": "trajectory_recovery_epoch", "run": stem, **row}), flush=True)
        scheduler.step()
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            resolved_config=resolved,
            selection_metric="last",
            selection_value=None,
        )
        selections = {
            "best_val_loss.pt": ("val_loss", float(primary["validation_loss"]), True),
            "best_single_macro.pt": ("single_macro", single_macro, False),
            "best_single_worst.pt": ("single_worst", single_worst, False),
            "best_fix_rate.pt": (
                "fix_rate",
                None if primary["fix_rate"] is None else float(primary["fix_rate"]),
                False,
            ),
        }
        worst_improved = False
        for filename, (name, value, minimize) in selections.items():
            improved = epoch == 1 or (
                value is not None and (value < best[name] if minimize else value > best[name])
            )
            if improved:
                if value is not None:
                    best[name] = value
                _save_checkpoint(
                    checkpoint_dir / filename,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    resolved_config=resolved,
                    selection_metric=name,
                    selection_value=value,
                )
                worst_improved |= name == "single_worst"
        patience_count = 0 if worst_improved else patience_count + 1
        if patience_count >= int(training["patience"]):
            stop_reason = "early_stopping_patience"
            break
    payload = torch.load(
        checkpoint_dir / "best_single_worst.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(payload["model_state"])
    final_primary = evaluate(
        model,
        validation_records,
        frequencies=frequencies,
        snr_db=float(training["validation_snr_db"]),
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    final_all14 = (
        evaluate(
            model,
            all14_records,
            frequencies=frequencies,
            snr_db=float(training["validation_snr_db"]),
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        if all14_records is not None
        else None
    )
    final_full = (
        evaluate(
            model,
            full_records,
            frequencies=frequencies,
            snr_db=float(training["validation_snr_db"]),
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        if full_records is not None
        else None
    )
    result = {
        "round": args.round_name,
        "task": task,
        "seed": seed,
        "budget": args.budget,
        "pilot_re": budget[0] * budget[1],
        "fusion_mode": fusion_mode,
        "initialization": initialization,
        "input_time": spec["input_time"],
        "target_time": spec["target_time"],
        "CSI_history_length": spec["history_length"],
        "selected_epoch": int(payload["epoch"]),
        "epochs_ran": len(history_rows),
        "stop_reason": stop_reason,
        "outer_test_accessed": False,
        **final_primary,
        "single_macro": final_primary["top1"],
        "single_worst": final_primary.get("worst_top1", final_primary["top1"]),
        "all14_macro": None if final_all14 is None else final_all14["top1"],
        "all14_worst": None if final_all14 is None else final_all14["worst_top1"],
        "all14_per_mask": None if final_all14 is None else final_all14["per_mask"],
        "full_top1": None if final_full is None else final_full["top1"],
        "full_fix_rate": None if final_full is None else final_full["fix_rate"],
        "full_harm_rate": None if final_full is None else final_full["harm_rate"],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def integrate_fallback(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.round_name != "round3_fallback" or args.seed is None or args.budget is None:
        raise ValueError("Fallback integration requires round3_fallback, --seed and --budget.")
    if args.init_checkpoint is None:
        raise ValueError("Fallback integration requires the selected sparse I3 --init-checkpoint.")
    budget = parse_budget(args.budget)
    maximum = (int(config["pilot"]["max_patterns"]), int(config["pilot"]["max_pilot_subcarriers"]))
    if budget[0] > maximum[0] or budget[1] > maximum[1]:
        raise ValueError("Requested budget exceeds the prepared mother observation.")
    seed = int(args.seed)
    round_root = args.output_root / args.round_name
    stem = f"seed{seed}_I3_{args.budget}_hard_fallback"
    result_path = round_root / "results" / f"{stem}.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"Trajectory fallback result already exists: {result_path}")
    validation = torch.load(
        args.output_root / "records/validation.pt", map_location="cpu", weights_only=False
    )
    records = _task_records(
        validation,
        "I3",
        budget,
        max_frequencies=int(config["pilot"]["max_pilot_subcarriers"]),
    )
    device = torch.device(args.device)
    model = SparsePilotInformationClassifier(
        history_length=5,
        sensing_dim=0,
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_candidate_patterns=int(config["pilot"]["max_patterns"]),
        encoder_layers=int(config["model"]["encoder_layers"]),
        fusion_mode="replace",
    ).to(device)
    initialization = _initialize_model(model, args.init_checkpoint, fusion_mode="replace")
    prepared = safe_load_yaml(
        (args.output_root / "resolved_configs/prepare.yaml").read_text(encoding="utf-8")
    )
    frequency_indices = nested_frequency_indices(
        int(config["pilot"]["max_pilot_subcarriers"]), budget[1]
    )
    frequencies = torch.tensor(
        prepared["runtime"]["frequency_positions_hz"], device=device
    ).index_select(0, frequency_indices.to(device))
    csi_probabilities, entropy = _predict_probabilities(
        model,
        records,
        frequencies=frequencies,
        snr_db=float(config["training"]["validation_snr_db"]),
        batch_size=int(config["training"]["batch_size"]),
        device=device,
        seed=seed,
    )
    labels = validation["labels_future"]
    power = validation["future_beam_power"]
    per_mask = []
    for name in ALL_MISSING_MASKS:
        base = validation[f"p0_{name}"]
        final = _fallback_probabilities(csi_probabilities, base, name)
        per_mask.append({"mask": name} | _prediction_metrics(final, labels, power, base))
    singles = [row for row in per_mask if row["mask"] in SEVERE_MASKS]
    single = _aggregate_prediction_rows(singles)
    all14 = _aggregate_prediction_rows(per_mask)
    full_base = validation["p0_full"]
    full_final = _fallback_probabilities(csi_probabilities, full_base, "full")
    full = _prediction_metrics(full_final, labels, power, full_base)
    result = {
        "round": args.round_name,
        "task": "I3",
        "seed": seed,
        "budget": args.budget,
        "pilot_re": budget[0] * budget[1],
        "fusion_mode": "hard_fallback",
        "input_time": TASKS["I3"]["input_time"],
        "target_time": TASKS["I3"]["target_time"],
        "CSI_history_length": 5,
        "initialization": initialization,
        "selected_epoch": initialization["source_epoch"],
        "epochs_ran": 0,
        "stop_reason": "fixed_availability_integration",
        "outer_test_accessed": False,
        **single,
        "validation_loss": float(
            F.nll_loss(csi_probabilities.clamp_min(1e-12).log(), labels).item()
        ),
        "q_entropy": float(entropy.mean().item()),
        "single_macro": single["top1"],
        "single_worst": min(float(row["top1"]) for row in singles),
        "single_per_mask": singles,
        "all14_macro": all14["top1"],
        "all14_worst": min(float(row["top1"]) for row in per_mask),
        "all14_per_mask": per_mask,
        "full_top1": full["top1"],
        "full_fix_rate": full["fix_rate"],
        "full_harm_rate": full["harm_rate"],
        "full_probability_max_abs_diff": float((full_final - full_base).abs().max().item()),
        "full_argmax_mismatch": int(
            (full_final.argmax(dim=-1) != full_base.argmax(dim=-1)).sum().item()
        ),
    }
    resolved = prepared | {
        "run": {
            "round": args.round_name,
            "task": "I3",
            "seed": seed,
            "budget": args.budget,
            "pilot_re": budget[0] * budget[1],
            "fusion_mode": "hard_fallback",
            "availability_rule": "csi_if_sensing_count_lte_2_else_m4",
            "frequency_indices": frequency_indices.tolist(),
            "initialization": initialization,
            "device": str(device),
        }
    }
    dump_config(resolved, round_root / "resolved_configs" / f"{stem}.yaml")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def _baseline_row(records: Mapping[str, Any]) -> dict[str, Any]:
    summary = _baseline_summary(records)
    singles = [summary[name] for name in SEVERE_MASKS]
    all14 = [summary[name] for name in ALL_MISSING_MASKS]
    return {
        "round": "baseline",
        "task": "I0",
        "seed": "frozen",
        "budget": "0x0",
        "pilot_re": 0,
        "fusion_mode": "frozen_m4",
        "top1": float(np.mean([row["top1"] for row in singles])),
        "top3": float(np.mean([row["top3"] for row in singles])),
        "top5": float(np.mean([row["top5"] for row in singles])),
        "within3": float(np.mean([row["within3"] for row in singles])),
        "mae": float(np.mean([row["mae"] for row in singles])),
        "single_macro": float(np.mean([row["top1"] for row in singles])),
        "single_worst": min(row["top1"] for row in singles),
        "all14_macro": float(np.mean([row["top1"] for row in all14])),
        "all14_worst": min(row["top1"] for row in all14),
        "full_top1": summary["full"]["top1"],
        "fix_rate": 0.0,
        "harm_rate": 0.0,
        "epochs_ran": 0,
        "selected_epoch": 0,
        "stop_reason": "frozen_m4",
    }


def summarize(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.round_name not in config["rounds"]:
        raise ValueError("Summarize mode requires a configured --round-name.")
    round_cfg = config["rounds"][args.round_name]
    budgets = list(round_cfg.get("budgets") or ([args.budget] if args.budget else []))
    if not budgets:
        raise ValueError("Round summary has no budget.")
    seeds = (
        [int(value) for value in args.seeds.split(",")]
        if args.seeds
        else [int(value) for value in config["training"]["seeds"]]
    )
    rows: list[dict[str, Any]] = []
    missing = []
    fusion = str(round_cfg["fusion_mode"])
    for budget in budgets:
        for task in round_cfg["tasks"]:
            for seed in seeds:
                path = args.output_root / args.round_name / "results" / f"seed{seed}_{task}_{budget}_{fusion}.json"
                if path.is_file():
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
                else:
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing round results:\n" + "\n".join(missing))
    validation = torch.load(
        args.output_root / "records/validation.pt", map_location="cpu", weights_only=False
    )
    baseline = _baseline_row(validation)
    csv_rows = [baseline]
    metrics = (
        "top1",
        "top3",
        "top5",
        "within3",
        "mae",
        "single_macro",
        "single_worst",
        "all14_macro",
        "all14_worst",
        "full_top1",
        "fix_rate",
        "harm_rate",
    )
    for row in rows:
        csv_rows.append({key: row.get(key) for key in baseline} | {key: row.get(key) for key in metrics})
    mean_rows = []
    for budget in budgets:
        for task in round_cfg["tasks"]:
            selected = [row for row in rows if row["budget"] == budget and row["task"] == task]
            mean = dict(selected[0])
            mean.update(seed="mean", epochs_ran="", selected_epoch="", stop_reason="")
            for metric in metrics:
                values = [float(row[metric]) for row in selected if row.get(metric) is not None]
                mean[metric] = float(np.mean(values)) if values else None
            mean_rows.append(mean)
            csv_rows.append({key: mean.get(key) for key in baseline} | {key: mean.get(key) for key in metrics})
    round_root = args.output_root / args.round_name
    _write_csv(round_root / "summary.csv", csv_rows)
    decision: dict[str, Any] = {
        "round": args.round_name,
        "baseline": baseline,
        "outer_test_accessed": False,
        "development_exploratory": True,
    }
    if args.round_name == "round1_dense":
        decision["means"] = {row["task"]: {key: row.get(key) for key in metrics} for row in mean_rows}
    elif args.round_name == "round2_sparse":
        candidate_task = str(round_cfg["tasks"][0])
        candidates = [row for row in mean_rows if row["task"] == candidate_task]
        feasible = [
            row
            for row in candidates
            if float(row["single_worst"]) > float(baseline["single_worst"])
            and float(row["single_macro"]) >= float(baseline["single_macro"])
        ]
        pool = feasible or candidates
        best_worst = max(float(row["single_worst"]) for row in pool)
        near_best = [row for row in pool if float(row["single_worst"]) >= best_worst - 0.005]
        selected = min(
            near_best,
            key=lambda row: (
                int(row["pilot_re"]),
                -float(row["single_worst"]),
                -float(row["single_macro"]),
            ),
        )
        decision.update(
            feasible_budget_found=bool(feasible),
            selected_budget=selected["budget"],
            selected_single_macro=selected["single_macro"],
            selected_single_worst=selected["single_worst"],
            selected_full_top1=baseline["full_top1"],
            full_catastrophic_regression=False,
        )
        (round_root / "selected_budget.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
        )
    else:
        selected = next(row for row in mean_rows if row["task"] == round_cfg["tasks"][0])
        decision.update(
            selected_budget=selected["budget"],
            single_macro_delta=float(selected["single_macro"]) - float(baseline["single_macro"]),
            single_worst_delta=float(selected["single_worst"]) - float(baseline["single_worst"]),
            all14_macro_delta=float(selected["all14_macro"]) - float(baseline["all14_macro"]),
            full_top1_delta=float(selected["full_top1"]) - float(baseline["full_top1"]),
            stable_single_gain=all(
                float(row["single_worst"]) > float(baseline["single_worst"])
                and float(row["single_macro"]) > float(baseline["single_macro"])
                for row in rows
                if row["task"] == round_cfg["tasks"][0]
            ),
        )
    (round_root / "analysis.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


def _return_code_path(args: argparse.Namespace) -> Path | None:
    if args.mode not in {"train", "integrate"} or not all(
        (args.round_name, args.task, args.budget, args.seed)
    ):
        return None
    fusion = args.fusion_mode or "configured"
    return args.output_root / args.round_name / "return_codes" / f"seed{args.seed}_{args.task}_{args.budget}_{fusion}.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trajectory-disjoint sparse-pilot recovery rounds.")
    parser.add_argument(
        "--mode", choices=("prepare", "train", "integrate", "summarize"), required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("tools/configs/sparse_pilot_trajectory_recovery.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--round-name", choices=("round1_dense", "round2_sparse", "round3_fallback")
    )
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget")
    parser.add_argument("--fusion-mode")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--seeds", help="Comma-separated seed subset for an interim summary.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = safe_load_yaml(_path(args.config).read_text(encoding="utf-8"))
    args.output_root = _path(args.output_root or config["output"]["root"])
    if args.round_name:
        args.fusion_mode = str(config["rounds"][args.round_name]["fusion_mode"])
    return_path = _return_code_path(args)
    try:
        if args.mode == "prepare":
            prepare(args, config)
        elif args.mode == "train":
            train(args, config)
        elif args.mode == "integrate":
            integrate_fallback(args, config)
        else:
            summarize(args, config)
    except Exception:
        if return_path is not None:
            return_path.parent.mkdir(parents=True, exist_ok=True)
            return_path.write_text("1\n", encoding="utf-8")
        raise
    if return_path is not None:
        return_path.parent.mkdir(parents=True, exist_ok=True)
        return_path.write_text("0\n", encoding="utf-8")


if __name__ == "__main__":
    main()
