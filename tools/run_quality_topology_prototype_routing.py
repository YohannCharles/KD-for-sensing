#!/usr/bin/env python3
"""Local trajectory workflow for quality-aware topology-smoothed prototype routing."""

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
import torch.nn as nn
import torch.nn.functional as F
import yaml

from kd_sensing.baselines.full_pool_bt_scl import BeamTopology, load_audited_topology
from kd_sensing.baselines.full_pool_common import sha256_file
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.models.dynamic_prototype_fusion import (
    DynamicPrototypeFusion,
    MatchedConcatHead,
    QTPR_METHODS,
)
from kd_sensing.models.prototype_fusion_losses import dynamic_fusion_loss, radio_alignment_loss

if __package__:
    from .run_csi_anchored_completion import (
        _autocast,
        _frequency_positions,
        _load_m4,
        _load_radio,
        _load_radio_training_teacher,
        _path,
        _radio_from_candidates,
        _recovery_record,
        _set_seed,
        _stratified_indices,
        _summary_from_per_mask,
        _write_csv,
        _write_json,
        preflight as completion_preflight,
        validate_cache_record,
    )
    from .run_mmw_trajectory_baselines import ALL_PATTERNS
    from .run_sparse_pilot_recovery import _prediction_metrics
else:
    from run_csi_anchored_completion import (
        _autocast,
        _frequency_positions,
        _load_m4,
        _load_radio,
        _load_radio_training_teacher,
        _path,
        _radio_from_candidates,
        _recovery_record,
        _set_seed,
        _stratified_indices,
        _summary_from_per_mask,
        _write_csv,
        _write_json,
        preflight as completion_preflight,
        validate_cache_record,
    )
    from run_mmw_trajectory_baselines import ALL_PATTERNS
    from run_sparse_pilot_recovery import _prediction_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/quality_topology_prototype_routing_trajectory.yaml"
METHODS = (*QTPR_METHODS, "B2-match")
MASK_BY_TUPLE = {tuple(bool(value) for value in mask): name for name, mask in ALL_PATTERNS.items()}
MASKS_BY_CARDINALITY = {count: tuple(mask for mask in MASK_BY_TUPLE if sum(mask) == count) for count in (1, 2, 3, 4)}
CHECKPOINT_NAMES = (
    "best_all14_macro.pt",
    "best_all14_worst.pt",
    "best_single_macro.pt",
    "best_val_loss.pt",
    "last.pt",
)


def _load_config(path: Path) -> dict[str, Any]:
    config = safe_load_yaml(path.read_text(encoding="utf-8"))
    if config["protocol"].get("outer_test_enabled") is not False:
        raise ValueError("QTPR requires the outer test to remain disabled.")
    return config


def _source_paths(config: Mapping[str, Any], role: str) -> tuple[Path, Path]:
    source = config["source_cache"]
    return _path(source["feature_root"]) / f"{role}.pt", _path(source["radio_root"]) / f"{role}.pt"


def _load_records(
    config: Mapping[str, Any],
    role: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    feature_path, radio_path = _source_paths(config, role)
    feature = torch.load(feature_path, map_location="cpu", weights_only=False, mmap=True)
    recovery = _recovery_record(config, role)
    use_cached_radio = bool(config["source_cache"].get("use_cached_radio", True))
    radio = (
        torch.load(radio_path, map_location="cpu", weights_only=False, mmap=True)
        if use_cached_radio
        else {"sample_ids": list(feature["sample_ids"])}
    )
    expected = int(config["protocol"][f"expected_{role}_samples"])
    validate_cache_record(feature, recovery, expected_count=expected)
    if list(feature["sample_ids"]) != list(radio["sample_ids"]):
        raise ValueError(f"{role} feature/radio stable sample IDs differ.")
    return feature, recovery, radio


def preflight(config: Mapping[str, Any], *, write_manifest: bool = True) -> dict[str, Any]:
    identity = completion_preflight(config)
    topology_cfg = config["topology"]
    topology_path = _path(topology_cfg["manifest"])
    topology_hashes = {
        "manifest": sha256_file(topology_path),
        "table": sha256_file(topology_path.parent / "topology_table.csv"),
        "edges": sha256_file(topology_path.parent / "topology_edges.csv"),
    }
    expected_topology = {
        "manifest": topology_cfg["manifest_sha256"],
        "table": topology_cfg["table_sha256"],
        "edges": topology_cfg["edges_sha256"],
    }
    if topology_hashes != expected_topology:
        raise ValueError(f"QTPR topology hash mismatch: actual={topology_hashes}, expected={expected_topology}.")
    topology = load_audited_topology(topology_path)
    if topology.descriptor_sha256 != topology_cfg["descriptor_sha256"] or tuple(topology.labels_by_position) != tuple(range(64)):
        raise ValueError("QTPR topology descriptor or phase-cycle label order changed.")
    if float(topology.distance[0, 63]) != 1.0:
        raise ValueError("The audited QTPR topology does not contain endpoint edge 0--63.")

    source = config["source_cache"]
    use_cached_radio = bool(source.get("use_cached_radio", True))
    source_manifest = _path(source["completion_manifest"])
    if sha256_file(source_manifest) != source["completion_manifest_sha256"]:
        raise ValueError("Completion feature manifest SHA256 mismatch.")
    roles: dict[str, Any] = {}
    trajectories: dict[str, set[str]] = {}
    for role in ("train", "validation"):
        feature_path, radio_path = _source_paths(config, role)
        feature_hash = sha256_file(feature_path)
        if feature_hash != source[f"{role}_feature_sha256"]:
            raise ValueError(f"{role} QTPR source feature cache SHA256 mismatch.")
        radio_hash = sha256_file(radio_path) if use_cached_radio else None
        if use_cached_radio and radio_hash != source[f"{role}_radio_sha256"]:
            raise ValueError(f"{role} QTPR source radio cache SHA256 mismatch.")
        feature, recovery, radio = _load_records(config, role)
        trajectories[role] = set(str(value) for value in feature["trajectory_ids"])
        roles[role] = {
            "sample_count": len(feature["sample_ids"]),
            "trajectory_count": len(trajectories[role]),
            "stable_sample_ids_exact": list(feature["sample_ids"]) == list(recovery["sample_ids"]) == list(radio["sample_ids"]),
            "targets_exact": bool(torch.equal(feature["target"], recovery["labels_future"])),
            "all_15_masks_same_identity": all(
                torch.is_tensor(value) and value.shape[0] == len(feature["sample_ids"])
                for name, value in recovery.items()
                if name.startswith("p0_")
            ),
            "feature_cache": str(feature_path.resolve()),
            "feature_cache_sha256": feature_hash,
            "radio_cache": str(radio_path.resolve()) if use_cached_radio else None,
            "radio_cache_sha256": radio_hash,
            "cached_radio_used": use_cached_radio,
        }
        if not all((roles[role]["stable_sample_ids_exact"], roles[role]["targets_exact"], roles[role]["all_15_masks_same_identity"])):
            raise ValueError(f"{role} QTPR cache identity validation failed.")
    overlap = trajectories["train"] & trajectories["validation"]
    if overlap:
        raise ValueError(f"QTPR train/validation trajectory overlap: {sorted(overlap)}.")
    manifest = {
        "version": "quality_topology_prototype_routing_cache_v1",
        "protocol_id": config["protocol"]["id"],
        "protocol_fingerprint": config["protocol"]["protocol_fingerprint"],
        "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
        "m4_checkpoint_sha256": config["protocol"]["m4_checkpoint_sha256"],
        "csi_checkpoint_sha256": config["radio_encoder"]["checkpoint_sha256"],
        "codebook_hash": config["radio_encoder"]["codebook_hash"],
        "source_completion_manifest_sha256": source["completion_manifest_sha256"],
        "cached_radio_used": use_cached_radio,
        "radio_view_source": "completion_cache" if use_cached_radio else "recovery_mother_observation",
        "topology_id": topology_cfg["id"],
        "topology_descriptor_sha256": topology.descriptor_sha256,
        "topology_manifest_sha256": topology_hashes["manifest"],
        "labels_by_position": list(topology.labels_by_position),
        "endpoint_0_63_adjacent": True,
        "roles": roles,
        "train_validation_trajectory_overlap_count": 0,
        "future_channel_used_as_input": False,
        "outer_test_accessed": False,
    }
    if write_manifest:
        output = _path(config["output"]["root"])
        _write_json(output / "cache_manifest.json", manifest)
        resolved = output / "resolved_configs/base.yaml"
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    return {"identity": identity, "cache": manifest, "outer_test_accessed": False}


def _topology(config: Mapping[str, Any]) -> BeamTopology:
    return load_audited_topology(_path(config["topology"]["manifest"]))


def _teacher_head_state(teacher: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in teacher.state_dict().items()}


def _build_fusion(
    config: Mapping[str, Any],
    method: str,
    topology: BeamTopology,
    device: torch.device,
) -> DynamicPrototypeFusion:
    model_cfg = config["model"]
    return DynamicPrototypeFusion(
        method,
        labels_by_position=topology.labels_by_position,
        radio_dim=int(model_cfg["radio_dim"]),
        prototype_dim=int(model_cfg["prototype_dim"]),
        radio_hidden_dim=int(model_cfg["radio_hidden_dim"]),
        trust_hidden_dim=int(model_cfg["trust_hidden_dim"]),
        gate_hidden_channels=int(model_cfg["gate_hidden_channels"]),
        gate_kernel_size=int(model_cfg["gate_kernel_size"]),
        gate_initial_probability=float(model_cfg["gate_initial_probability"]),
        structured_trust_base_bias=float(model_cfg["structured_trust_base_bias"]),
        structured_trust_raw_quality=float(model_cfg["structured_trust_raw_quality"]),
        sensing_temperature=float(model_cfg["sensing_temperature"]),
        radio_temperature=float(model_cfg["radio_temperature"]),
        fixed_weight=float(model_cfg["fixed_weight"]),
    ).to(device)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    config: Mapping[str, Any],
    method: str,
    budget: str,
    seed: int,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    metrics: Mapping[str, float],
    stage: str,
) -> None:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    if method != "B2-match" and any("classifier" in name for name in state):
        raise RuntimeError("A QTPR checkpoint unexpectedly contains classifier parameters.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": state,
            "method": method,
            "stage": stage,
            "budget": budget,
            "seed": int(seed),
            "epoch": int(epoch),
            "metrics": dict(metrics),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": _rng_state(),
            "protocol_fingerprint": config["protocol"]["protocol_fingerprint"],
            "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
            "m4_checkpoint_sha256": config["protocol"]["m4_checkpoint_sha256"],
            "csi_checkpoint_sha256": config["radio_encoder"]["checkpoint_sha256"],
            "topology_descriptor_sha256": config["topology"]["descriptor_sha256"],
            "outer_test_accessed": False,
        },
        path,
    )


def _load_checkpoint(model: nn.Module, path: Path, config: Mapping[str, Any], *, expected_method: str | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "protocol": payload.get("protocol_fingerprint") == config["protocol"]["protocol_fingerprint"],
        "split": payload.get("split_manifest_sha256") == config["protocol"]["split_manifest_sha256"],
        "m4": payload.get("m4_checkpoint_sha256") == config["protocol"]["m4_checkpoint_sha256"],
        "csi": payload.get("csi_checkpoint_sha256") == config["radio_encoder"]["checkpoint_sha256"],
        "topology": payload.get("topology_descriptor_sha256") == config["topology"]["descriptor_sha256"],
        "outer_test": payload.get("outer_test_accessed") is False,
        "method": expected_method is None or payload.get("method") == expected_method,
    }
    if not all(checks.values()):
        raise ValueError(f"QTPR checkpoint identity mismatch: {checks}.")
    model.load_state_dict(payload["model_state"], strict=True)
    return payload


def _load_alignment_state(fusion: DynamicPrototypeFusion, path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != "R1" or payload.get("outer_test_accessed") is not False:
        raise ValueError("QTPR fusion requires a valid R1 checkpoint.")
    if payload.get("csi_checkpoint_sha256") != config["radio_encoder"]["checkpoint_sha256"]:
        raise ValueError("R1 CSI checkpoint identity mismatch.")
    state = payload["model_state"]
    prefix = "radio_expert."
    radio_state = {name.removeprefix(prefix): value for name, value in state.items() if name.startswith(prefix)}
    fusion.radio_expert.load_state_dict(radio_state, strict=True)
    return payload


def _classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    topology_distance: torch.Tensor,
) -> dict[str, float]:
    values = logits.float()
    target = labels.to(values.device)
    probability = torch.softmax(values, dim=-1)
    prediction = probability.argmax(dim=-1)
    top = probability.topk(5, dim=-1).indices
    distance = topology_distance.to(values.device)[target, prediction].float()
    confidence = probability.max(dim=-1).values
    correct = prediction.eq(target)
    ece = values.new_zeros(())
    for left in torch.linspace(0.0, 1.0, 16, device=values.device)[:-1]:
        right = left + 1.0 / 15.0
        selected = (confidence >= left) & (confidence < right if float(right) < 1.0 else confidence <= right)
        if bool(selected.any()):
            ece = ece + selected.float().mean() * (confidence[selected].mean() - correct[selected].float().mean()).abs()
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    top_values = probability.topk(2, dim=-1).values
    return {
        "top1": float(correct.float().mean().item()),
        "top3": float(top[:, :3].eq(target[:, None]).any(dim=-1).float().mean().item()),
        "top5": float(top.eq(target[:, None]).any(dim=-1).float().mean().item()),
        "within3": float(distance.le(3).float().mean().item()),
        "mae": float(distance.mean().item()),
        "nll": float(F.cross_entropy(values, target).item()),
        "ece": float(ece.item()),
        "entropy": float(entropy.mean().item()),
        "top1_top2_margin": float((top_values[:, 0] - top_values[:, 1]).mean().item()),
    }


def _alignment_evaluation(
    fusion: DynamicPrototypeFusion,
    m4: nn.Module,
    teacher: nn.Module,
    c_radio: torch.Tensor,
    labels: torch.Tensor,
    topology: BeamTopology,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, float]]:
    fusion.eval()
    teacher.eval()
    student_chunks, teacher_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(labels), int(batch_size)):
            radio = c_radio[start : start + int(batch_size)].to(device)
            teacher_chunks.append(teacher(radio).float().cpu())
            student_chunks.append(fusion.radio_expert(radio, m4.prototype_bank)["radio_evidence"].float().cpu())
    student = torch.cat(student_chunks)
    teacher_logits = torch.cat(teacher_chunks)
    student_metrics = _classification_metrics(student, labels, topology.distance)
    student_metrics["radio_temperature"] = float(fusion.radio_expert.temperature().detach().cpu())
    teacher_metrics = _classification_metrics(teacher_logits, labels, topology.distance)
    teacher_probability = torch.softmax(teacher_logits, dim=-1)
    student_metrics["prototype_semantic_kl"] = float(
        F.kl_div(F.log_softmax(student, dim=-1), teacher_probability, reduction="batchmean").item()
    )
    teacher_metrics["prototype_semantic_kl"] = 0.0
    student_metrics["argmax_mismatch_vs_teacher"] = float(student.argmax(dim=-1).ne(teacher_logits.argmax(dim=-1)).sum().item())
    teacher_metrics["argmax_mismatch_vs_teacher"] = 0.0
    return student_metrics, teacher_metrics


def _alignment_paths(config: Mapping[str, Any], seed: int, budget: str) -> tuple[Path, Path]:
    root = _path(config["output"]["root"])
    run = root / "runs" / f"R1_seed{int(seed)}_{budget}"
    return run, run / "best_radio_top1.pt"


def train_alignment(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    preflight(config)
    seed, budget = int(args.seed), str(args.budget)
    device = torch.device(args.device)
    _set_seed(seed)
    topology = _topology(config)
    m4 = _load_m4(config, device)
    radio_encoder = _load_radio(config, device)
    teacher = _load_radio_training_teacher(config, device)
    fusion = _build_fusion(config, "F0", topology, device)
    initialization = fusion.radio_expert.initialize_from_teacher(m4.prototype_bank, _teacher_head_state(teacher))
    for parameter in fusion.parameters():
        parameter.requires_grad_(False)
    fusion.radio_expert.temperature.raw.requires_grad_(True)
    alignment_parameters = [fusion.radio_expert.temperature.raw]

    feature, recovery, _ = _load_records(config, "train")
    validation_feature, validation_recovery, validation_radio = _load_records(config, "validation")
    indices = _stratified_indices(feature, limit=int(args.limit) if args.limit else None, seed=90_000 + seed)
    training = config["training"]
    batch_size = int(args.batch_size or training["batch_size"])
    epochs = int(args.epochs if args.epochs is not None else training["max_epochs"])
    optimizer = torch.optim.AdamW(
        alignment_parameters,
        lr=float(training["radio_alignment_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    steps = max(math.ceil(len(indices) / batch_size), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs * steps, 1))
    frequencies = _frequency_positions(config, budget, device)
    order_generator = torch.Generator().manual_seed(91_000 + seed)
    noise_generator = torch.Generator(device=device).manual_seed(92_000 + seed)
    run_dir, best_path = _alignment_paths(config, seed, budget)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {"stage": "R1", "seed": seed, "budget": budget, "epochs": epochs, "sample_count": len(indices), **dict(config)}, sort_keys=False
        ),
        encoding="utf-8",
    )

    labels_validation = validation_feature["target"]
    cached_radio = _validation_radio_view(
        config,
        validation_recovery,
        validation_radio,
        radio_encoder,
        budget=budget,
        device=device,
        seed=seed,
    )["c_radio"]
    initial_student, teacher_metrics = _alignment_evaluation(
        fusion,
        m4,
        teacher,
        cached_radio,
        labels_validation,
        topology,
        device,
        batch_size=int(training["evaluation_batch_size"]),
    )
    threshold = float(config["evaluation"]["alignment_max_gap_pp"])
    best_nll = initial_student["nll"]
    best_metrics = dict(initial_student)
    _save_checkpoint(
        best_path,
        fusion,
        config=config,
        method="F0",
        budget=budget,
        seed=seed,
        epoch=0,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=initial_student,
        stage="R1",
    )
    rows: list[dict[str, Any]] = []
    patience = 0
    for epoch in range(1, epochs + 1):
        fusion.train()
        order = indices[torch.randperm(len(indices), generator=order_generator)]
        totals = defaultdict(float)
        seen = 0
        started = time.monotonic()
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            candidates = recovery["candidate_history"][batch_indices].to(device, non_blocking=True)
            snr = torch.empty(len(batch_indices), device=device).uniform_(
                float(training["snr_db_min"]),
                float(training["snr_db_max"]),
                generator=noise_generator,
            )
            # Keep the frozen sparse-pilot encoder in FP32. Its attention mask
            # uses the dtype minimum, which is not representable via BF16 fill.
            with torch.no_grad():
                radio = _radio_from_candidates(
                    radio_encoder,
                    candidates,
                    budget=budget,
                    frequencies=frequencies,
                    snr=snr,
                    generator=noise_generator,
                    dropout_probability=float(training["pilot_dropout_probability"]),
                )
            labels = feature["target"][batch_indices].to(device)
            with _autocast(device):
                student = fusion.radio_expert(radio["c_radio"], m4.prototype_bank)["radio_evidence"]
                terms = radio_alignment_loss(
                    student.float(),
                    labels,
                    topology.distance.to(device),
                    teacher_logits=None,
                    topology_weight=float(config["loss"]["topology"]),
                    distillation_weight=float(config["loss"]["radio_alignment_kd"]),
                    distillation_temperature=float(config["loss"]["radio_kd_temperature"]),
                )
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(alignment_parameters, float(training["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            count = len(batch_indices)
            seen += count
            for name, value in terms.items():
                totals[name] += float(value.detach()) * count
        student_metrics, _ = _alignment_evaluation(
            fusion,
            m4,
            teacher,
            cached_radio,
            labels_validation,
            topology,
            device,
            batch_size=int(training["evaluation_batch_size"]),
        )
        row = {
            "epoch": epoch,
            "train_samples": seen,
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in student_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - started,
        }
        rows.append(row)
        _write_csv(run_dir / "training_log.csv", rows)
        _save_checkpoint(
            run_dir / "last.pt",
            fusion,
            config=config,
            method="F0",
            budget=budget,
            seed=seed,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=student_metrics,
            stage="R1",
        )
        gap_pp_epoch = 100.0 * (teacher_metrics["top1"] - student_metrics["top1"])
        if gap_pp_epoch <= threshold and student_metrics["nll"] < best_nll:
            best_nll = student_metrics["nll"]
            best_metrics = dict(student_metrics)
            patience = 0
            _save_checkpoint(
                best_path,
                fusion,
                config=config,
                method="F0",
                budget=budget,
                seed=seed,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=student_metrics,
                stage="R1",
            )
        else:
            patience += 1
        if patience >= int(training["patience"]):
            break

    published = 0.9615082740783691 if budget == "16x16" else float("nan")
    gap_pp = 100.0 * (teacher_metrics["top1"] - best_metrics["top1"])
    passed = gap_pp <= threshold
    summary_rows = [
        {
            "expert": "published_csi_classifier",
            "budget": budget,
            "top1": published,
            "pilot_re": int(budget.split("x")[0]) * int(budget.split("x")[1]),
        },
        {"expert": "cached_csi_classifier", "budget": budget, **teacher_metrics},
        {"expert": "shared_prototype_initial", "budget": budget, **initial_student},
        {"expert": "shared_prototype_best", "budget": budget, **best_metrics},
    ]
    output = _path(config["output"]["root"])
    _write_csv(output / "radio_alignment_summary.csv", summary_rows)
    gate = {
        "status": "passed" if passed else "failed",
        "teacher_top1": teacher_metrics["top1"],
        "shared_prototype_top1": best_metrics["top1"],
        "shared_prototype_nll": best_metrics["nll"],
        "radio_temperature": best_metrics["radio_temperature"],
        "gap_pp": gap_pp,
        "maximum_gap_pp": threshold,
        "checkpoint": str(best_path.resolve()),
        "checkpoint_sha256": sha256_file(best_path),
        "initialization": initialization,
        "epochs_ran": len(rows),
        "outer_test_accessed": False,
    }
    _write_json(output / "radio_alignment_gate.json", gate)
    _write_json(run_dir / "complete.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)


def _sample_masks(count: int, config: Mapping[str, Any], generator: torch.Generator) -> torch.Tensor:
    probabilities_cfg = config["training"]["cardinality_probabilities"]
    cardinalities = torch.tensor((1, 2, 3, 4), dtype=torch.long)
    probabilities = torch.tensor(
        [float(probabilities_cfg.get(value, probabilities_cfg.get(str(value)))) for value in cardinalities.tolist()]
    )
    sampled = cardinalities[torch.multinomial(probabilities, int(count), replacement=True, generator=generator)]
    rows = []
    for cardinality in sampled.tolist():
        choices = MASKS_BY_CARDINALITY[int(cardinality)]
        weights = torch.ones(len(choices))
        if int(cardinality) == 3:
            for index, mask in enumerate(choices):
                if not mask[1]:
                    weights[index] = float(config["training"]["missing_lidar_weight"])
        rows.append(choices[int(torch.multinomial(weights, 1, generator=generator).item())])
    return torch.tensor(rows, dtype=torch.bool)


def _sensing_batch(
    recovery: Mapping[str, Any],
    indices: torch.Tensor,
    masks: torch.Tensor,
    m4: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = len(indices)
    embedding = torch.empty(count, 64, device=device)
    base_probability = torch.empty(count, 64, device=device)
    for mask in torch.unique(masks, dim=0):
        selected = masks.eq(mask).all(dim=-1).nonzero(as_tuple=False).squeeze(1)
        name = MASK_BY_TUPLE[tuple(bool(value) for value in mask.tolist())]
        source_indices = indices.index_select(0, selected)
        embedding.index_copy_(0, selected.to(device), recovery[f"z_{name}"][source_indices].to(device))
        base_probability.index_copy_(0, selected.to(device), recovery[f"p0_{name}"][source_indices].to(device))
    # Recovery records were extracted with BF16 prototype queries. Reusing the
    # same query precision exactly reproduces the frozen M4 evidence cache.
    with torch.no_grad(), _autocast(device):
        logits = m4.prototype_bank(embedding)
    logits = logits.float()
    if not torch.allclose(torch.softmax(logits, dim=-1), base_probability, atol=2e-6, rtol=1e-5):
        raise ValueError("Cached sensing embedding no longer reproduces M4 prototype evidence.")
    return embedding, logits, base_probability


def _nested_masks(masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    more, less = masks.clone(), masks.clone()
    for row in range(len(masks)):
        available = masks[row].nonzero(as_tuple=False).squeeze(1)
        missing = (~masks[row]).nonzero(as_tuple=False).squeeze(1)
        if len(available) > 1:
            more[row, available[-1]] = False
        elif len(missing):
            less[row, missing[0]] = True
    return more, less


def _method_loss_weights(config: Mapping[str, Any], method: str) -> dict[str, float]:
    values = {name: float(value) for name, value in config["loss"].items() if isinstance(value, (int, float))}
    if method == "F1":
        values.update(
            trust=0.0, missing_monotonic=0.0, quality_monotonic=0.0, gate_smooth=0.0, rescue=0.0, preserve=0.0, gate_usage=0.0, radio_kd=0.0
        )
    elif method == "F2":
        values.update(missing_monotonic=0.0, quality_monotonic=0.0, gate_smooth=0.0, rescue=0.0, preserve=0.0, gate_usage=0.0, radio_kd=0.0)
    elif method == "F3":
        values.update(missing_monotonic=0.0, quality_monotonic=0.0, gate_smooth=0.0, rescue=0.0, preserve=0.0, gate_usage=0.0, radio_kd=0.0)
    elif method == "F4":
        values.update(missing_monotonic=0.0, quality_monotonic=0.0, rescue=0.0, preserve=0.0, gate_usage=0.0, radio_kd=0.0)
    elif method in {"F5", "F7"}:
        values.update(rescue=0.0, preserve=0.0, radio_kd=0.0)
    return values


def _binary_auc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    values = scores.detach().float().cpu().numpy()
    labels = targets.detach().bool().cpu().numpy()
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


@torch.no_grad()
def _generated_radio_view(
    config: Mapping[str, Any],
    recovery: Mapping[str, Any],
    radio_encoder: nn.Module,
    *,
    budget: str,
    device: torch.device,
    snr_db: float,
    dropout_probability: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    batch_size = int(config["training"]["evaluation_batch_size"])
    frequencies = _frequency_positions(config, budget, device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
    for start in range(0, len(recovery["labels_future"]), batch_size):
        candidates = recovery["candidate_history"][start : start + batch_size].to(device)
        snr = torch.full((len(candidates),), float(snr_db), device=device)
        output = _radio_from_candidates(
            radio_encoder,
            candidates,
            budget=budget,
            frequencies=frequencies,
            snr=snr,
            generator=generator,
            dropout_probability=float(dropout_probability),
        )
        for name in ("c_radio", "csi_quality", "csi_available"):
            chunks[name].append(output[name].detach().cpu())
    return {name: torch.cat(values) for name, values in chunks.items()}


def _cached_radio_view(radio: Mapping[str, Any], budget: str) -> dict[str, torch.Tensor]:
    return {
        "c_radio": radio[f"c_radio_{budget}"],
        "csi_quality": radio[f"csi_quality_{budget}"],
        "csi_available": radio[f"csi_available_{budget}"],
    }


def _validation_radio_view(
    config: Mapping[str, Any],
    recovery: Mapping[str, Any],
    radio_cache: Mapping[str, Any],
    radio_encoder: nn.Module,
    *,
    budget: str,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    required = tuple(f"{name}_{budget}" for name in ("c_radio", "csi_quality", "csi_available"))
    if all(name in radio_cache for name in required):
        return _cached_radio_view(radio_cache, budget)
    return _generated_radio_view(
        config,
        recovery,
        radio_encoder,
        budget=budget,
        device=device,
        snr_db=float(config["training"]["validation_snr_db"]),
        dropout_probability=0.0,
        seed=int(config["evaluation"].get("validation_radio_seed", 100_000 + int(seed))),
    )


@torch.inference_mode()
def _evaluate(
    model: DynamicPrototypeFusion | MatchedConcatHead,
    m4: nn.Module,
    feature: Mapping[str, Any],
    recovery: Mapping[str, Any],
    radio_view: Mapping[str, torch.Tensor],
    topology: BeamTopology,
    device: torch.device,
    *,
    method: str,
    budget: str,
    seed: int,
    batch_size: int,
    diagnostic: str = "normal",
    collect_details: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if diagnostic not in {"normal", "csi_zero", "csi_shuffle", "sensing_shuffle"}:
        raise ValueError(f"Unknown QTPR diagnostic: {diagnostic}.")
    model.eval()
    count = len(feature["target"])
    generator = torch.Generator().manual_seed(300_000 + int(seed))
    permutation = torch.randperm(count, generator=generator)
    per_mask: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    all_rho, all_gate = [], []
    all_sensing, all_radio, all_final, all_labels = [], [], [], []
    examples: dict[str, list[torch.Tensor]] = defaultdict(list)
    started = time.monotonic()

    for mask_name, mask_values in ALL_PATTERNS.items():
        mask = torch.tensor(mask_values, dtype=torch.bool)
        available_count = int(mask.sum().item())
        base_probability = recovery[f"p0_{mask_name}"]
        if mask_name == "full":
            probability = base_probability.clone()
            row = {"mask": mask_name, "available_count": available_count} | _prediction_metrics(
                probability,
                feature["target"],
                feature["future_beam_power"],
                base_probability,
            )
            row["nll"] = float(F.nll_loss(probability.clamp_min(1e-12).log(), feature["target"]).item())
            row.update(rho_mean=0.0, rho_std=0.0, rho_p10=0.0, rho_p50=0.0, rho_p90=0.0, gate_variation=0.0)
            per_mask.append(row)
            continue

        probabilities, rho_chunks, gate_chunks = [], [], []
        sensing_chunks, radio_chunks, final_chunks, label_chunks = [], [], [], []
        for start in range(0, count, int(batch_size)):
            stop = min(start + int(batch_size), count)
            base_indices = torch.arange(start, stop)
            sensing_indices = permutation[start:stop] if diagnostic == "sensing_shuffle" else base_indices
            radio_indices = permutation[start:stop] if diagnostic == "csi_shuffle" else base_indices
            embedding = recovery[f"z_{mask_name}"][sensing_indices].to(device)
            with _autocast(device):
                sensing_logits = m4.prototype_bank(embedding)
            sensing_logits = sensing_logits.float()
            c_radio = radio_view["c_radio"][radio_indices].to(device)
            quality = radio_view["csi_quality"][radio_indices].to(device)
            available = radio_view["csi_available"][radio_indices].to(device)
            if diagnostic == "csi_zero":
                c_radio = torch.zeros_like(c_radio)
                quality = torch.zeros_like(quality)
                available = torch.zeros_like(available)
            physical = mask.to(device).expand(stop - start, -1)
            if method == "B2-match":
                final = model(embedding, c_radio)
                final = torch.where(available[:, None], final, sensing_logits)
                rho = available.to(final.dtype)
                gate = rho[:, None].expand_as(final)
                radio_logits = final.new_zeros(final.shape)
            else:
                output = model(
                    embedding,
                    sensing_logits,
                    c_radio,
                    quality,
                    available,
                    physical,
                    m4.prototype_bank,
                    topology.distance.to(device),
                )
                final = output["final_evidence"]
                rho = output["rho"]
                gate = output["prototype_gate"]
                radio_logits = output["radio_evidence_calibrated"]
            probabilities.append(torch.softmax(final.float(), dim=-1).cpu())
            rho_chunks.append(rho.float().cpu())
            gate_chunks.append(gate.float().cpu())
            if method != "B2-match":
                sensing_chunks.append(sensing_logits.float().cpu())
                radio_chunks.append(radio_logits.float().cpu())
                final_chunks.append(final.float().cpu())
                label_chunks.append(feature["target"][start:stop])
                if collect_details and sum(len(value) for value in examples.values()) < 500:
                    room = max(0, 100 - sum(chunk.shape[0] for chunk in examples["target"]))
                    take = min(room, stop - start)
                    if take:
                        examples["p_s"].append(torch.softmax(sensing_logits[:take].float(), dim=-1).cpu())
                        examples["p_c"].append(torch.softmax(radio_logits[:take].float(), dim=-1).cpu())
                        examples["g"].append(gate[:take].float().cpu())
                        examples["p_final"].append(torch.softmax(final[:take].float(), dim=-1).cpu())
                        examples["target"].append(feature["target"][start : start + take])

        probability = torch.cat(probabilities)
        rho_values = torch.cat(rho_chunks)
        gate_values = torch.cat(gate_chunks)
        row = {"mask": mask_name, "available_count": available_count} | _prediction_metrics(
            probability,
            feature["target"],
            feature["future_beam_power"],
            base_probability,
        )
        row["nll"] = float(F.nll_loss(probability.clamp_min(1e-12).log(), feature["target"]).item())
        quantiles = torch.quantile(rho_values, torch.tensor((0.1, 0.5, 0.9)))
        ordered_gate = gate_values[:, list(topology.labels_by_position)]
        variation = (ordered_gate - ordered_gate.roll(1, dims=-1)).abs().mean()
        row.update(
            rho_mean=float(rho_values.mean()),
            rho_std=float(rho_values.std(unbiased=False)),
            rho_p10=float(quantiles[0]),
            rho_p50=float(quantiles[1]),
            rho_p90=float(quantiles[2]),
            gate_variation=float(variation),
        )
        per_mask.append(row)
        gate_rows.append(
            {key: row[key] for key in ("mask", "available_count", "rho_mean", "rho_std", "rho_p10", "rho_p50", "rho_p90", "gate_variation")}
        )
        all_rho.append(rho_values)
        all_gate.append(gate_values)
        if method != "B2-match":
            all_sensing.append(torch.cat(sensing_chunks))
            all_radio.append(torch.cat(radio_chunks))
            all_final.append(torch.cat(final_chunks))
            all_labels.append(torch.cat(label_chunks))

    summary = _summary_from_per_mask(per_mask)
    elapsed = time.monotonic() - started
    result = {
        "method": method,
        "budget": budget,
        "seed": int(seed),
        "diagnostic": diagnostic,
        "sample_count": count,
        "per_mask": per_mask,
        **summary,
        "full_bypass_max_abs": 0.0,
        "full_bypass_argmax_mismatch": 0,
        "full_pilot_re": 0,
        "elapsed_seconds": elapsed,
        "latency_ms_per_sample_mask": 1000.0 * elapsed / max(count * len(ALL_PATTERNS), 1),
        "outer_test_accessed": False,
    }
    details: dict[str, Any] = {"gate_rows": gate_rows}
    if method != "B2-match" and all_rho:
        rho = torch.cat(all_rho)
        gate = torch.cat(all_gate)
        sensing = torch.cat(all_sensing)
        radio = torch.cat(all_radio)
        final = torch.cat(all_final)
        labels = torch.cat(all_labels)
        ce_s = F.cross_entropy(sensing, labels, reduction="none")
        ce_c = F.cross_entropy(radio, labels, reduction="none")
        ce_f = F.cross_entropy(final, labels, reduction="none")
        radio_better = ce_c < ce_s
        regret = ce_f - torch.minimum(ce_s, ce_c)
        pred_s, pred_c, pred_f = sensing.argmax(dim=-1), radio.argmax(dim=-1), final.argmax(dim=-1)
        groups = {
            "sensing_correct_radio_wrong": pred_s.eq(labels) & pred_c.ne(labels),
            "sensing_wrong_radio_correct": pred_s.ne(labels) & pred_c.eq(labels),
            "both_correct": pred_s.eq(labels) & pred_c.eq(labels),
            "both_wrong": pred_s.ne(labels) & pred_c.ne(labels),
        }
        disagreement_rows = []
        for name, selected in groups.items():
            disagreement_rows.append(
                {
                    "group": name,
                    "sample_count": int(selected.sum()),
                    "final_top1": float(pred_f[selected].eq(labels[selected]).float().mean()) if bool(selected.any()) else float("nan"),
                    "rho_mean": float(rho[selected].mean()) if bool(selected.any()) else float("nan"),
                    "fusion_regret_mean": float(regret[selected].mean()) if bool(selected.any()) else float("nan"),
                }
            )
        details.update(
            gate_auc=_binary_auc(rho, radio_better),
            gate_mean_by_beam=gate.mean(dim=0),
            gate_rows=gate_rows,
            disagreement_rows=disagreement_rows,
            fusion_regret={
                "mean": float(regret.mean()),
                "p50": float(torch.quantile(regret, 0.5)),
                "p90": float(torch.quantile(regret, 0.9)),
                "p95": float(torch.quantile(regret, 0.95)),
                "minimum": float(regret.min()),
                "maximum": float(regret.max()),
            },
            examples={name: torch.cat(values)[:100] for name, values in examples.items() if values},
        )
    return result, details


def _run_dir(config: Mapping[str, Any], method: str, seed: int, budget: str) -> Path:
    return _path(config["output"]["root"]) / "runs" / f"{method}_seed{int(seed)}_{budget}"


def _validation_score(result: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all14_macro": float(result["groups"]["all14"]["top1_macro"]),
        "all14_worst": float(result["groups"]["all14"]["top1_worst"]),
        "single_macro": float(result["groups"]["single"]["top1_macro"]),
        "val_loss": float(np.mean([row["nll"] for row in result["per_mask"] if row["mask"] != "full"])),
    }


def _alignment_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _path(config["output"]["root"]) / "radio_alignment_gate.json"
    if not path.is_file():
        raise FileNotFoundError("Stage R1 gate is absent; run alignment first.")
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("status") != "passed":
        raise RuntimeError(f"Stage R1 failed, so gate experiments are forbidden: {gate}.")
    checkpoint = Path(gate["checkpoint"])
    if sha256_file(checkpoint) != gate["checkpoint_sha256"]:
        raise ValueError("Stage R1 checkpoint SHA256 changed.")
    return gate


def train_fusion(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    preflight(config)
    gate = _alignment_gate(config)
    method, seed, budget = str(args.method), int(args.seed), str(args.budget)
    if method not in METHODS:
        raise ValueError(f"Unknown QTPR training method: {method}.")
    device = torch.device(args.device)
    _set_seed(seed)
    topology = _topology(config)
    m4 = _load_m4(config, device)
    radio_encoder = _load_radio(config, device)
    feature, recovery, _ = _load_records(config, "train")
    validation_feature, validation_recovery, validation_radio_cache = _load_records(config, "validation")
    training = config["training"]
    batch_size = int(args.batch_size or training["batch_size"])
    epochs = int(args.epochs if args.epochs is not None else training["short_epochs"])
    indices = _stratified_indices(feature, limit=int(args.limit) if args.limit else None, seed=400_000 + seed)
    run_dir = _run_dir(config, method, seed, budget)
    if run_dir.exists() and (run_dir / "complete.json").is_file() and not args.overwrite:
        raise FileExistsError(f"Completed QTPR run already exists: {run_dir}.")
    run_dir.mkdir(parents=True, exist_ok=True)

    fusion: DynamicPrototypeFusion | None = None
    concat: MatchedConcatHead | None = None
    if method == "B2-match":
        concat = MatchedConcatHead(
            sensing_dim=64,
            radio_dim=int(config["model"]["radio_dim"]),
            hidden_dim=int(config["model"]["radio_hidden_dim"]),
            num_beams=64,
        ).to(device)
        experiment_model: nn.Module = concat
    else:
        fusion = _build_fusion(config, method, topology, device)
        if method == "F0":
            _load_checkpoint(fusion, Path(gate["checkpoint"]), config, expected_method="F0")
        else:
            _load_alignment_state(fusion, Path(gate["checkpoint"]), config)
        experiment_model = fusion
    if args.checkpoint is not None:
        _load_checkpoint(experiment_model, args.checkpoint, config, expected_method=None)

    validation_radio = _validation_radio_view(
        config,
        validation_recovery,
        validation_radio_cache,
        radio_encoder,
        budget=budget,
        device=device,
        seed=seed,
    )
    if method == "F0":
        result, details = _evaluate(
            experiment_model,
            m4,
            validation_feature,
            validation_recovery,
            validation_radio,
            topology,
            device,
            method=method,
            budget=budget,
            seed=seed,
            batch_size=int(training["evaluation_batch_size"]),
            collect_details=True,
        )
        _write_evaluation_artifacts(run_dir, result, details, suffix="final")
        _write_json(run_dir / "complete.json", {"status": "completed", "stage": "R1", "outer_test_accessed": False})
        return

    if fusion is not None:
        for parameter in fusion.radio_expert.parameters():
            parameter.requires_grad_(False)
        fusion.radio_expert.temperature.raw.requires_grad_(True)
        if method in {"F6", "F7"}:
            for parameter in fusion.radio_expert.parameters():
                parameter.requires_grad_(True)
    parameters = [parameter for parameter in experiment_model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"{method} has no trainable parameters.")
    learning_rate = float(training["radio_joint_learning_rate"] if method in {"F6", "F7"} else training["gate_learning_rate"])
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=float(training["weight_decay"]))
    steps = max(math.ceil(len(indices) / batch_size), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs * steps, 1))
    teacher = _load_radio_training_teacher(config, device) if method in {"F6", "F7"} else None
    frequencies = _frequency_positions(config, budget, device)
    order_generator = torch.Generator().manual_seed(410_000 + seed)
    mask_generator = torch.Generator().manual_seed(420_000 + seed)
    noise_generator = torch.Generator(device=device).manual_seed(430_000 + seed)
    stage = "R3" if method in {"F6", "F7"} else "R2"
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": stage,
                "method": method,
                "seed": seed,
                "budget": budget,
                "epochs": epochs,
                "sample_count_per_epoch": len(indices),
                "alignment_checkpoint": gate["checkpoint"],
                "initialization_checkpoint": str(args.checkpoint) if args.checkpoint else None,
                **dict(config),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    initial_result, _ = _evaluate(
        experiment_model,
        m4,
        validation_feature,
        validation_recovery,
        validation_radio,
        topology,
        device,
        method=method,
        budget=budget,
        seed=seed,
        batch_size=int(training["evaluation_batch_size"]),
    )
    best_values = {
        "all14_macro": float("-inf"),
        "all14_worst": float("-inf"),
        "single_macro": float("-inf"),
        "val_loss": float("inf"),
    }
    history: list[dict[str, Any]] = []
    sampling_counts: dict[str, int] = defaultdict(int)
    expert_counts: dict[str, int] = defaultdict(int)
    patience = 0
    loss_weights = _method_loss_weights(config, method)
    cardinality_weights = training["task_cardinality_weights"]

    for epoch in range(1, epochs + 1):
        experiment_model.train()
        order = indices[torch.randperm(len(indices), generator=order_generator)]
        totals: dict[str, float] = defaultdict(float)
        seen = 0
        started = time.monotonic()
        floor_epochs = int(training["rho_floor_epochs"])
        rho_floor = 0.2 * max(0.0, 1.0 - (epoch - 1) / max(floor_epochs - 1, 1)) if epoch <= floor_epochs else 0.0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            masks = _sample_masks(len(batch_indices), config, mask_generator)
            for mask in masks.tolist():
                name = MASK_BY_TUPLE[tuple(bool(value) for value in mask)]
                sampling_counts[name] += 1
                sampling_counts[f"cardinality_{sum(mask)}"] += 1
            nonfull = ~masks.all(dim=-1)
            if not bool(nonfull.any()):
                continue
            batch_indices = batch_indices[nonfull]
            masks = masks[nonfull]
            labels = feature["target"][batch_indices].to(device)
            embedding, sensing_logits, _ = _sensing_batch(recovery, batch_indices, masks, m4, device)
            candidates = recovery["candidate_history"][batch_indices].to(device, non_blocking=True)
            snr = torch.empty(len(batch_indices), device=device).uniform_(
                float(training["snr_db_min"]),
                float(training["snr_db_max"]),
                generator=noise_generator,
            )
            with torch.no_grad():
                radio = _radio_from_candidates(
                    radio_encoder,
                    candidates,
                    budget=budget,
                    frequencies=frequencies,
                    snr=snr,
                    generator=noise_generator,
                    dropout_probability=float(training["pilot_dropout_probability"]),
                )
            optimizer.zero_grad(set_to_none=True)
            if concat is not None:
                with _autocast(device):
                    final = concat(embedding, radio["c_radio"])
                    items = F.cross_entropy(final.float(), labels, reduction="none")
                    cardinalities = masks.sum(dim=-1)
                    sample_weights = torch.tensor(
                        [
                            float(cardinality_weights.get(int(value), cardinality_weights.get(str(int(value)), 1.0)))
                            for value in cardinalities
                        ],
                        device=device,
                    )
                    total_loss = (items * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
                    terms: dict[str, torch.Tensor] = {"total": total_loss, "task": total_loss}
            else:
                assert fusion is not None
                with _autocast(device):
                    output = fusion(
                        embedding,
                        sensing_logits,
                        radio["c_radio"],
                        radio["csi_quality"],
                        radio["csi_available"],
                        masks.to(device),
                        m4.prototype_bank,
                        topology.distance.to(device),
                        rho_floor=rho_floor,
                    )
                rho_more = rho_less = rho_degraded = None
                if method in {"F5", "F6", "F7"}:
                    more_masks, less_masks = _nested_masks(masks)
                    more_embedding, more_logits, _ = _sensing_batch(recovery, batch_indices, more_masks, m4, device)
                    less_embedding, less_logits, _ = _sensing_batch(recovery, batch_indices, less_masks, m4, device)
                    with _autocast(device):
                        more_output = fusion(
                            more_embedding,
                            more_logits,
                            radio["c_radio"],
                            radio["csi_quality"],
                            radio["csi_available"],
                            more_masks.to(device),
                            m4.prototype_bank,
                            topology.distance.to(device),
                        )
                        less_output = fusion(
                            less_embedding,
                            less_logits,
                            radio["c_radio"],
                            radio["csi_quality"],
                            radio["csi_available"],
                            less_masks.to(device),
                            m4.prototype_bank,
                            topology.distance.to(device),
                        )
                    rho_more, rho_less = more_output["rho"], less_output["rho"]
                    degraded_snr = (snr - float(training["degraded_snr_delta_db"])).clamp_min(-20.0)
                    with torch.no_grad():
                        degraded = _radio_from_candidates(
                            radio_encoder,
                            candidates,
                            budget=budget,
                            frequencies=frequencies,
                            snr=degraded_snr,
                            generator=noise_generator,
                            dropout_probability=float(training["degraded_dropout_probability"]),
                        )
                    with _autocast(device):
                        degraded_output = fusion(
                            embedding,
                            sensing_logits,
                            degraded["c_radio"],
                            degraded["csi_quality"],
                            degraded["csi_available"],
                            masks.to(device),
                            m4.prototype_bank,
                            topology.distance.to(device),
                        )
                    rho_degraded = degraded_output["rho"]
                teacher_logits = teacher(radio["c_radio"]).detach() if teacher is not None else None
                terms = dynamic_fusion_loss(
                    {name: value.float() if torch.is_floating_point(value) else value for name, value in output.items()},
                    labels,
                    masks.to(device),
                    topology.distance.to(device),
                    weights=loss_weights,
                    cardinality_weights=cardinality_weights,
                    labels_by_position=topology.labels_by_position,
                    teacher_logits=teacher_logits.float() if teacher_logits is not None else None,
                    rho_more_missing=rho_more.float() if rho_more is not None else None,
                    rho_less_missing=rho_less.float() if rho_less is not None else None,
                    rho_clean=output["rho"].float() if rho_degraded is not None else None,
                    rho_degraded=rho_degraded.float() if rho_degraded is not None else None,
                    trust_temperature=float(config["loss"]["trust_temperature"]),
                    min_rho_std=float(config["loss"]["min_rho_std"]),
                )
                pred_s = output["sensing_evidence"].argmax(dim=-1)
                pred_c = output["radio_evidence_calibrated"].argmax(dim=-1)
                expert_counts["sensing_correct_radio_wrong"] += int((pred_s.eq(labels) & pred_c.ne(labels)).sum())
                expert_counts["sensing_wrong_radio_correct"] += int((pred_s.ne(labels) & pred_c.eq(labels)).sum())
                expert_counts["both_correct"] += int((pred_s.eq(labels) & pred_c.eq(labels)).sum())
                expert_counts["both_wrong"] += int((pred_s.ne(labels) & pred_c.ne(labels)).sum())
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip_norm"]))
            optimizer.step()
            scheduler.step()
            count = len(batch_indices)
            seen += count
            for name, value in terms.items():
                totals[name] += float(value.detach()) * count

        validation_result, _ = _evaluate(
            experiment_model,
            m4,
            validation_feature,
            validation_recovery,
            validation_radio,
            topology,
            device,
            method=method,
            budget=budget,
            seed=seed,
            batch_size=int(training["evaluation_batch_size"]),
        )
        scores = _validation_score(validation_result)
        row = {
            "epoch": epoch,
            "train_samples": seen,
            **{f"train_{name}": value / max(seen, 1) for name, value in totals.items()},
            **{f"validation_{name}": value for name, value in scores.items()},
            "validation_two_macro": validation_result["groups"]["two"]["top1_macro"],
            "validation_three_macro": validation_result["groups"]["three"]["top1_macro"],
            "validation_missing_lidar": validation_result["missing_lidar"]["top1"],
            "rho_floor": rho_floor,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - started,
        }
        history.append(row)
        _write_csv(run_dir / "training_log.csv", history)
        metric_payload = {
            **scores,
            "two_macro": row["validation_two_macro"],
            "three_macro": row["validation_three_macro"],
            "missing_lidar": row["validation_missing_lidar"],
        }
        _save_checkpoint(
            run_dir / "last.pt",
            experiment_model,
            config=config,
            method=method,
            budget=budget,
            seed=seed,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metric_payload,
            stage=stage,
        )
        improved = False
        for name, filename, maximize in (
            ("all14_macro", "best_all14_macro.pt", True),
            ("all14_worst", "best_all14_worst.pt", True),
            ("single_macro", "best_single_macro.pt", True),
            ("val_loss", "best_val_loss.pt", False),
        ):
            better = scores[name] > best_values[name] if maximize else scores[name] < best_values[name]
            if better:
                best_values[name] = scores[name]
                improved = improved or name == "all14_macro"
                _save_checkpoint(
                    run_dir / filename,
                    experiment_model,
                    config=config,
                    method=method,
                    budget=budget,
                    seed=seed,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=metric_payload,
                    stage=stage,
                )
        patience = 0 if improved else patience + 1
        if patience >= int(training["patience"]):
            break

    best_path = run_dir / "best_all14_macro.pt"
    _load_checkpoint(experiment_model, best_path, config, expected_method=method)
    final_result, details = _evaluate(
        experiment_model,
        m4,
        validation_feature,
        validation_recovery,
        validation_radio,
        topology,
        device,
        method=method,
        budget=budget,
        seed=seed,
        batch_size=int(training["evaluation_batch_size"]),
        collect_details=True,
    )
    _write_evaluation_artifacts(run_dir, final_result, details, suffix="final")
    sampling = {
        "sample_count_per_epoch": len(indices),
        "epochs_ran": len(history),
        "mask_counts": dict(sorted(sampling_counts.items())),
        "expert_outcome_counts": dict(sorted(expert_counts.items())),
        "front_n_sampling_used": False,
        "outer_test_accessed": False,
    }
    _write_json(run_dir / "sampling_statistics.json", sampling)
    complete = {
        "status": "completed",
        "method": method,
        "stage": stage,
        "budget": budget,
        "seed": seed,
        "epochs_ran": len(history),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256_file(best_path),
        "groups": final_result["groups"],
        "missing_lidar": final_result["missing_lidar"],
        "full": final_result["full"],
        "outer_test_accessed": False,
    }
    _write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, indent=2, sort_keys=True), flush=True)


def _write_evaluation_artifacts(
    run_dir: Path,
    result: Mapping[str, Any],
    details: Mapping[str, Any],
    *,
    suffix: str,
) -> None:
    _write_json(run_dir / f"evaluation_{suffix}.json", result)
    _write_csv(run_dir / f"mask_metrics_{suffix}.csv", result["per_mask"])
    if details.get("gate_rows"):
        _write_csv(run_dir / f"gate_statistics_{suffix}.csv", details["gate_rows"])
    if details.get("disagreement_rows"):
        _write_csv(run_dir / f"disagreement_groups_{suffix}.csv", details["disagreement_rows"])
    if details.get("fusion_regret"):
        _write_csv(run_dir / f"fusion_regret_{suffix}.csv", [{"method": result["method"], **details["fusion_regret"]}])
    gate_mean = details.get("gate_mean_by_beam")
    if torch.is_tensor(gate_mean):
        _write_csv(
            run_dir / f"gate_by_beam_{suffix}.csv",
            [{"beam": index, "mean_gate": float(value)} for index, value in enumerate(gate_mean.tolist())],
        )
    examples = details.get("examples")
    if isinstance(examples, Mapping) and examples:
        np.savez_compressed(
            run_dir / f"topology_examples_{suffix}.npz",
            **{name: value.detach().cpu().numpy() for name, value in examples.items()},
        )
    diagnostic = {
        "method": result["method"],
        "budget": result["budget"],
        "seed": result["seed"],
        "diagnostic": result["diagnostic"],
        "gate_auc": details.get("gate_auc"),
        "fusion_regret": details.get("fusion_regret"),
        "outer_test_accessed": False,
    }
    _write_json(run_dir / f"mechanism_{suffix}.json", diagnostic)


def _load_experiment_model(
    config: Mapping[str, Any],
    method: str,
    checkpoint: Path,
    topology: BeamTopology,
    device: torch.device,
) -> DynamicPrototypeFusion | MatchedConcatHead:
    if method == "B2-match":
        model: DynamicPrototypeFusion | MatchedConcatHead = MatchedConcatHead(
            sensing_dim=64,
            radio_dim=int(config["model"]["radio_dim"]),
            hidden_dim=int(config["model"]["radio_hidden_dim"]),
            num_beams=64,
        ).to(device)
    else:
        model = _build_fusion(config, method, topology, device)
    _load_checkpoint(model, checkpoint, config, expected_method=method)
    model.eval()
    return model


def _default_checkpoint(config: Mapping[str, Any], method: str, seed: int, budget: str) -> Path:
    if method == "F0":
        return Path(_alignment_gate(config)["checkpoint"])
    return _run_dir(config, method, seed, budget) / "best_all14_macro.pt"


def evaluate_checkpoint(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    preflight(config)
    method, seed, budget = str(args.method), int(args.seed), str(args.budget)
    device = torch.device(args.device)
    topology = _topology(config)
    checkpoint = args.checkpoint or _default_checkpoint(config, method, seed, budget)
    model = _load_experiment_model(config, method, checkpoint, topology, device)
    m4 = _load_m4(config, device)
    radio_encoder = _load_radio(config, device)
    feature, recovery, radio_cache = _load_records(config, "validation")
    radio_view = _validation_radio_view(
        config,
        recovery,
        radio_cache,
        radio_encoder,
        budget=budget,
        device=device,
        seed=seed,
    )
    result, details = _evaluate(
        model,
        m4,
        feature,
        recovery,
        radio_view,
        topology,
        device,
        method=method,
        budget=budget,
        seed=seed,
        batch_size=int(config["training"]["evaluation_batch_size"]),
        diagnostic=str(args.diagnostic),
        collect_details=True,
    )
    run_dir = _run_dir(config, method, seed, budget)
    suffix = "final" if args.diagnostic == "normal" else str(args.diagnostic)
    _write_evaluation_artifacts(run_dir, result, details, suffix=suffix)
    print(json.dumps({key: result[key] for key in ("method", "groups", "missing_lidar", "full")}, indent=2), flush=True)


def diagnose(args: argparse.Namespace, config: Mapping[str, Any]) -> None:
    preflight(config)
    method, seed, budget = str(args.method), int(args.seed), str(args.budget)
    if method == "B2-match":
        raise ValueError("Gate mechanism diagnosis is only defined for shared-prototype methods.")
    device = torch.device(args.device)
    topology = _topology(config)
    checkpoint = args.checkpoint or _default_checkpoint(config, method, seed, budget)
    model = _load_experiment_model(config, method, checkpoint, topology, device)
    m4 = _load_m4(config, device)
    radio_encoder = _load_radio(config, device)
    feature, recovery, radio_cache = _load_records(config, "validation")
    cached = _validation_radio_view(
        config,
        recovery,
        radio_cache,
        radio_encoder,
        budget=budget,
        device=device,
        seed=seed,
    )
    run_dir = _run_dir(config, method, seed, budget)
    output = _path(config["output"]["root"])

    shuffle_rows = []
    normal_details: dict[str, Any] | None = None
    for diagnostic in ("normal", "csi_shuffle", "sensing_shuffle", "csi_zero"):
        result, details = _evaluate(
            model,
            m4,
            feature,
            recovery,
            cached,
            topology,
            device,
            method=method,
            budget=budget,
            seed=seed,
            batch_size=int(config["training"]["evaluation_batch_size"]),
            diagnostic=diagnostic,
            collect_details=diagnostic == "normal",
        )
        _write_evaluation_artifacts(run_dir, result, details, suffix=diagnostic)
        shuffle_rows.append(
            {
                "method": method,
                "budget": budget,
                "seed": seed,
                "diagnostic": diagnostic,
                "single_macro": result["groups"]["single"]["top1_macro"],
                "two_macro": result["groups"]["two"]["top1_macro"],
                "three_macro": result["groups"]["three"]["top1_macro"],
                "all14_macro": result["groups"]["all14"]["top1_macro"],
                "all14_worst": result["groups"]["all14"]["top1_worst"],
                "missing_lidar": result["missing_lidar"]["top1"],
                "full": result["full"]["top1"],
            }
        )
        if diagnostic == "normal":
            normal_details = details
    _write_csv(output / "shuffle_diagnostics.csv", shuffle_rows)

    snr_rows = []
    for snr in config["evaluation"]["snr_db"]:
        radio_view = _generated_radio_view(
            config,
            recovery,
            radio_encoder,
            budget=budget,
            device=device,
            snr_db=float(snr),
            dropout_probability=0.0,
            seed=500_000 + seed + int((float(snr) + 20) * 10),
        )
        result, details = _evaluate(
            model,
            m4,
            feature,
            recovery,
            radio_view,
            topology,
            device,
            method=method,
            budget=budget,
            seed=seed,
            batch_size=int(config["training"]["evaluation_batch_size"]),
        )
        gate_rows = details.get("gate_rows", [])
        snr_rows.append(
            {
                "method": method,
                "budget": budget,
                "snr_db": float(snr),
                "rho_mean": float(np.mean([row["rho_mean"] for row in gate_rows])),
                "single_macro": result["groups"]["single"]["top1_macro"],
                "all14_macro": result["groups"]["all14"]["top1_macro"],
                "all14_worst": result["groups"]["all14"]["top1_worst"],
                "missing_lidar": result["missing_lidar"]["top1"],
            }
        )
    _write_csv(output / "snr_summary.csv", snr_rows)

    dropout_rows = []
    validation_snr = float(config["training"]["validation_snr_db"])
    for dropout in config["evaluation"]["pilot_dropout"]:
        radio_view = _generated_radio_view(
            config,
            recovery,
            radio_encoder,
            budget=budget,
            device=device,
            snr_db=validation_snr,
            dropout_probability=float(dropout),
            seed=600_000 + seed + int(float(dropout) * 1000),
        )
        result, details = _evaluate(
            model,
            m4,
            feature,
            recovery,
            radio_view,
            topology,
            device,
            method=method,
            budget=budget,
            seed=seed,
            batch_size=int(config["training"]["evaluation_batch_size"]),
        )
        gate_rows = details.get("gate_rows", [])
        dropout_rows.append(
            {
                "method": method,
                "budget": budget,
                "pilot_dropout": float(dropout),
                "rho_mean": float(np.mean([row["rho_mean"] for row in gate_rows])),
                "single_macro": result["groups"]["single"]["top1_macro"],
                "all14_macro": result["groups"]["all14"]["top1_macro"],
                "all14_worst": result["groups"]["all14"]["top1_worst"],
            }
        )
    _write_csv(output / "dropout_summary.csv", dropout_rows)
    if normal_details is not None:
        _write_csv(output / "gate_statistics.csv", normal_details.get("gate_rows", []))
        _write_csv(output / "fusion_regret.csv", [{"method": method, **normal_details.get("fusion_regret", {})}])
    _write_json(
        run_dir / "diagnostics_complete.json",
        {
            "status": "completed",
            "method": method,
            "budget": budget,
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "outer_test_accessed": False,
        },
    )


def _result_row(result: Mapping[str, Any], checkpoint: Path | None = None) -> dict[str, Any]:
    parameters = None
    if checkpoint is not None and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        parameters = sum(value.numel() for value in payload["model_state"].values())
    budget = str(result["budget"])
    pilot_re = int(budget.split("x")[0]) * int(budget.split("x")[1])
    return {
        "method": result["method"],
        "budget": budget,
        "seed": result["seed"],
        "pilot_re_missing": pilot_re,
        "pilot_re_full": 0,
        "trainable_parameters": parameters,
        "single_macro": result["groups"]["single"]["top1_macro"],
        "single_worst": result["groups"]["single"]["top1_worst"],
        "two_macro": result["groups"]["two"]["top1_macro"],
        "two_worst": result["groups"]["two"]["top1_worst"],
        "three_macro": result["groups"]["three"]["top1_macro"],
        "three_worst": result["groups"]["three"]["top1_worst"],
        "all14_macro": result["groups"]["all14"]["top1_macro"],
        "all14_worst": result["groups"]["all14"]["top1_worst"],
        "missing_lidar": result["missing_lidar"]["top1"],
        "full": result["full"]["top1"],
        "latency_ms_per_sample_mask": result["latency_ms_per_sample_mask"],
        "outer_test_accessed": False,
    }


def _low_re_quality_response(
    output: Path,
    method: str | None,
    *,
    minimum_delta: float,
) -> dict[str, Any]:
    result = {
        "method": method,
        "minimum_delta": float(minimum_delta),
        "snr_rho_delta": None,
        "dropout_rho_delta": None,
        "passed": False,
    }
    if method is None:
        return result
    views: dict[str, list[dict[str, str]]] = {}
    for name in ("snr", "dropout"):
        path = output / f"{name}_summary.csv"
        if not path.is_file():
            return result
        with path.open(encoding="utf-8", newline="") as handle:
            views[name] = [row for row in csv.DictReader(handle) if row["method"] == method]
    snr = {float(row["snr_db"]): float(row["rho_mean"]) for row in views["snr"]}
    dropout = {float(row["pilot_dropout"]): float(row["rho_mean"]) for row in views["dropout"]}
    if not {30.0, -10.0}.issubset(snr) or not {0.0, 0.5}.issubset(dropout):
        return result
    result["snr_rho_delta"] = snr[30.0] - snr[-10.0]
    result["dropout_rho_delta"] = dropout[0.0] - dropout[0.5]
    result["passed"] = bool(result["snr_rho_delta"] >= float(minimum_delta) and result["dropout_rho_delta"] >= float(minimum_delta))
    return result


def _baseline_rows() -> list[dict[str, Any]]:
    values = (
        ("B0", 0, 0.452121, 0.102121, 0.610631, 0.712176, 0.594355, 0.102121, 0.256088, 0.863315),
        ("B1", 256, 0.962399, 0.962399, 0.962399, 0.712176, 0.890906, 0.256088, 0.256088, 0.863315),
        ("B2-old", 512, 0.909688, 0.749149, 0.910448, 0.936201, 0.917589, 0.749149, 0.861273, 0.959937),
        ("B3", 256, 0.962399, 0.962399, 0.962399, 0.712176, 0.890906, 0.256088, 0.256088, 0.863315),
        ("B8D", 256, 0.690888, 0.274208, 0.672523, 0.738361, 0.696581, 0.274208, 0.400367, 0.863315),
    )
    rows = []
    for method, re, single, single_worst, two, three, all14, worst, lidar, full in values:
        rows.append(
            {
                "method": method,
                "budget": "none" if not re else ("32x16" if re == 512 else "16x16"),
                "seeds": 3 if method != "B0" else 1,
                "pilot_re_missing": re,
                "pilot_re_full": 0 if method != "B2-old" else 512,
                "single_macro": single,
                "single_worst": single_worst,
                "two_macro": two,
                "three_macro": three,
                "all14_macro": all14,
                "all14_worst": worst,
                "missing_lidar": lidar,
                "full": full,
                "source": "retained_baseline",
            }
        )
    return rows


def summarize(config: Mapping[str, Any]) -> None:
    output = _path(config["output"]["root"])
    run_rows, mask_rows, latency_rows = [], [], []
    for path in sorted((output / "runs").glob("*_seed*_*/*evaluation_final.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = path.parent / ("best_radio_top1.pt" if result["method"] == "F0" else "best_all14_macro.pt")
        row = _result_row(result, checkpoint if checkpoint.is_file() else None)
        run_rows.append(row)
        mask_rows.extend(
            {
                "method": result["method"],
                "budget": result["budget"],
                "seed": result["seed"],
                **mask,
            }
            for mask in result["per_mask"]
        )
        latency_rows.append(
            {
                "method": result["method"],
                "budget": result["budget"],
                "seed": result["seed"],
                "latency_ms_per_sample_mask": result["latency_ms_per_sample_mask"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(row["method"], row["budget"])].append(row)
    aggregate = _baseline_rows()
    for (method, budget), rows in sorted(grouped.items()):
        metric_names = (
            "single_macro",
            "single_worst",
            "two_macro",
            "two_worst",
            "three_macro",
            "three_worst",
            "all14_macro",
            "all14_worst",
            "missing_lidar",
            "full",
            "latency_ms_per_sample_mask",
        )
        aggregate.append(
            {
                "method": method,
                "budget": budget,
                "seeds": len(rows),
                "pilot_re_missing": rows[0]["pilot_re_missing"],
                "pilot_re_full": 0,
                "trainable_parameters": rows[0]["trainable_parameters"],
                **{name: float(np.mean([float(row[name]) for row in rows])) for name in metric_names},
                "source": "qtpr_development",
            }
        )
    _write_csv(output / "ablation_summary.csv", aggregate)
    _write_csv(output / "mask_summary.csv", mask_rows)
    _write_csv(output / "latency_summary.csv", latency_rows)
    _write_csv(
        output / "budget_summary.csv",
        [
            {
                "method": row["method"],
                "budget": row["budget"],
                "seeds": row.get("seeds"),
                "pilot_re_missing": row.get("pilot_re_missing"),
                "pilot_re_full": row.get("pilot_re_full"),
                "all14_macro": row.get("all14_macro"),
                "all14_worst": row.get("all14_worst"),
                "single_macro": row.get("single_macro"),
            }
            for row in aggregate
        ],
    )
    round_rows = []
    round_roots = (
        ("round1_uncalibrated", output / "round1_uncalibrated/runs"),
        ("round2_calibrated", output / "round2_calibrated/runs"),
        ("round3_quality_prior", output / "runs"),
    )
    for round_name, round_root in round_roots:
        for method in ("F1", "F2", "F3", "F4", "F5"):
            path = round_root / f"{method}_seed1_16x16/evaluation_final.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            round_rows.append(
                {
                    "round": round_name,
                    "method": method,
                    "pilot_re_missing": 256,
                    "pilot_re_full": 0,
                    "single_macro": result["groups"]["single"]["top1_macro"],
                    "all14_macro": result["groups"]["all14"]["top1_macro"],
                    "all14_worst": result["groups"]["all14"]["top1_worst"],
                    "missing_lidar": result["missing_lidar"]["top1"],
                    "full": result["full"]["top1"],
                }
            )
    _write_csv(output / "development_round_summary.csv", round_rows)
    by_method = {row["method"]: row for row in aggregate if row.get("source") == "qtpr_development"}
    low_re_mode = not bool(config["source_cache"].get("use_cached_radio", True))
    f5 = by_method.get("F5")
    f6 = by_method.get("F6")
    f5_beats_b8d = bool(f5 and f5["single_macro"] > 0.690888 and f5["all14_macro"] > 0.696581 and f5["missing_lidar"] > 0.400367)
    f5_beats_f2 = bool(f5 and "F2" in by_method and f5["all14_macro"] > by_method["F2"]["all14_macro"])
    f5_beats_f3 = bool(f5 and "F3" in by_method and f5["all14_macro"] > by_method["F3"]["all14_macro"])
    f5_stage_gate = f5_beats_b8d and f5_beats_f2 and f5_beats_f3
    dynamic_candidates = [by_method[method] for method in ("F2", "F5") if method in by_method]
    best_dynamic = max(dynamic_candidates, key=lambda row: float(row["all14_macro"])) if dynamic_candidates else None
    fixed = by_method.get("F1")
    low_re_dynamic_beats_fixed = bool(best_dynamic and fixed and float(best_dynamic["all14_macro"]) > float(fixed["all14_macro"]))
    low_re_dynamic_beats_b0 = bool(best_dynamic and float(best_dynamic["all14_macro"]) > 0.5943552910217217)
    low_re_quality = _low_re_quality_response(
        output,
        None if best_dynamic is None else str(best_dynamic["method"]),
        minimum_delta=float(config["loss"]["min_rho_std"]),
    )
    low_re_stage_gate = bool(low_re_dynamic_beats_fixed and low_re_dynamic_beats_b0 and low_re_quality["passed"])
    if low_re_mode:
        if not (low_re_dynamic_beats_fixed and low_re_dynamic_beats_b0):
            halt_reason = "Neither scalar F2 nor quality-topology F5 beat both fixed F1 and no-CSI B0."
        elif not low_re_quality["passed"]:
            halt_reason = "The best low-RE dynamic method did not meet the configured CSI-quality response threshold."
        else:
            halt_reason = None
    else:
        halt_reason = None if f5_stage_gate else "F5 did not beat both scalar F2 and independent-gate F3; F6 and expansion were skipped."
    gates = {
        "radio_alignment": json.loads((output / "radio_alignment_gate.json").read_text(encoding="utf-8"))
        if (output / "radio_alignment_gate.json").is_file()
        else None,
        "f5_beats_b8d": f5_beats_b8d,
        "f5_beats_f2": f5_beats_f2,
        "f5_beats_f3": f5_beats_f3,
        "f5_stage_gate": f5_stage_gate,
        "f6_ran": f6 is not None,
        "f6_beats_b8d": None
        if f6 is None
        else bool(f6["single_macro"] > 0.690888 and f6["all14_macro"] > 0.696581 and f6["missing_lidar"] > 0.400367),
        "f6_beats_f2": None if f6 is None else bool("F2" in by_method and f6["all14_macro"] > by_method["F2"]["all14_macro"]),
        "f6_beats_f3": None if f6 is None else bool("F3" in by_method and f6["all14_macro"] > by_method["F3"]["all14_macro"]),
        "f6_not_weaker_than_b2_match": None
        if f6 is None
        else bool("B2-match" in by_method and f6["all14_macro"] >= by_method["B2-match"]["all14_macro"]),
        "low_re_mode": low_re_mode,
        "low_re_best_dynamic_method": None if best_dynamic is None else best_dynamic["method"],
        "low_re_dynamic_beats_fixed_f1": low_re_dynamic_beats_fixed,
        "low_re_dynamic_beats_no_csi_b0": low_re_dynamic_beats_b0,
        "low_re_quality_response": low_re_quality,
        "low_re_dynamic_stage_gate": low_re_stage_gate,
        "halt_reason": halt_reason,
        "outer_test_accessed": False,
    }
    gates["expand_multi_seed"] = f5_stage_gate and all(
        gates[name] is True for name in ("f6_beats_b8d", "f6_beats_f2", "f6_beats_f3", "f6_not_weaker_than_b2_match")
    )
    _write_json(output / "success_gates.json", gates)
    table_rows = []
    for method in ("B0", "B8D", "F0", "F1", "F2", "F3", "F4", "F5"):
        row = next((value for value in aggregate if value["method"] == method), None)
        if row is None:
            continue
        table_rows.append(
            f"| {method} | {row.get('pilot_re_missing', '')} | {row.get('pilot_re_full', '')} | "
            f"{100 * float(row['single_macro']):.2f} | {100 * float(row['all14_macro']):.2f} | "
            f"{100 * float(row['all14_worst']):.2f} | {100 * float(row['missing_lidar']):.2f} | "
            f"{100 * float(row['full']):.2f} |"
        )
    report = [
        "# QTPR Development Report",
        "",
        "## Protocol",
        "",
        "- Shared prototype entry: yes; both sensing and radio evidence query the frozen M4 BeamPrototypeBank.",
        "- Inference CSI classifier: absent from every QTPR method and checkpoint.",
        "- Full path: hard bypass before radio/projection/trust/gate; pilot RE is zero.",
        "- Outer test: sealed and not accessed.",
        "",
        "## Accuracy and RE",
        "",
        "| Method | Missing RE | Full RE | Single Top-1 (%) | All-14 Top-1 (%) | Worst Top-1 (%) | Missing Lidar (%) | Full Top-1 (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        "## Required Answers",
        "",
        f"1. Shared-prototype CSI entry: {'validated' if gates['radio_alignment'] else 'pending'}.",
        f"2. Radio alignment loss: {gates['radio_alignment']['gap_pp']:.4f} pp."
        if gates["radio_alignment"]
        else "2. Radio alignment loss: pending.",
        f"3. Dynamic fusion comparisons available: {', '.join(sorted(by_method)) or 'none'}.",
        f"4. F5 prototype gate beats scalar F2: {gates['f5_beats_f2']}.",
        f"5. F5 topology route beats independent F3: {gates['f5_beats_f3']}.",
        f"5a. Low-RE best dynamic method: {gates['low_re_best_dynamic_method']}; "
        f"beats fixed F1: {gates['low_re_dynamic_beats_fixed_f1']}; "
        f"beats no-CSI B0: {gates['low_re_dynamic_beats_no_csi_b0']}.",
        f"5b. Low-RE quality response passed: {gates['low_re_quality_response']['passed']}; "
        f"SNR rho delta={gates['low_re_quality_response']['snr_rho_delta']}; "
        f"dropout rho delta={gates['low_re_quality_response']['dropout_rho_delta']}.",
        "6. Missingness response: see gate_statistics.csv.",
        "7. CSI quality response: see snr_summary.csv and dropout_summary.csv.",
        f"8. F5 beats B8D jointly: {gates['f5_beats_b8d']}; F6 ran: {gates['f6_ran']}.",
        "9. Full is exact by branch contract; every evaluation reports max_abs=0 and argmax mismatch=0.",
        f"10. 16x8 efficiency result present: {any(row.get('budget') == '16x8' for row in aggregate)}.",
        "11. CSI dominance/collapse checks: see shuffle_diagnostics.csv, gate_statistics.csv, and fusion_regret.csv.",
        f"12. Stop reason: {gates['halt_reason'] or 'all staged gates passed'}",
        "",
        "## Development Boundary",
        "",
        "Repeated validation use is exploratory. No outer-test claim is made.",
    ]
    (output / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(gates, indent=2, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("preflight", "align", "train", "evaluate", "diagnose", "summarize"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--method", choices=METHODS, default="F5")
    parser.add_argument("--budget", default="16x16")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--diagnostic", choices=("normal", "csi_zero", "csi_shuffle", "sensing_shuffle"), default="normal")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args.config)
    if args.mode == "preflight":
        print(json.dumps(preflight(config), indent=2, sort_keys=True), flush=True)
    elif args.mode == "align":
        train_alignment(args, config)
    elif args.mode == "train":
        train_fusion(args, config)
    elif args.mode == "evaluate":
        evaluate_checkpoint(args, config)
    elif args.mode == "diagnose":
        diagnose(args, config)
    else:
        summarize(config)


if __name__ == "__main__":
    main()
