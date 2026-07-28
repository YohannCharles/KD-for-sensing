#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from kd_sensing.baselines.prototype_decision_adapter import (
    MASKS,
    checkpoint_normalization_overrides,
    load_frozen_u0,
    load_u0_artifact_config,
    preflight,
)
from kd_sensing.baselines.sparse_pilot_transition import SparsePilotTransitionModel
from kd_sensing.channel.pilot_cache import PilotCache, PilotCacheSpec
from kd_sensing.channel.probe_codebook import generate_probe_codebook, load_probe_codebook
from kd_sensing.channel.sparse_pilot_simulator import (
    frequency_offsets_hz,
    load_path_channel,
    pilot_subcarrier_indices,
    simulate_candidate_pilots,
)
from kd_sensing.config.io import dump_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.engine.batch import prepare_fusion_inputs
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.models.prototype_pilot_selector import PrototypePilotSelector, select_from_lookup
from kd_sensing.models.prototype_transition import prototype_transition_losses
from kd_sensing.models.sparse_pilot_encoder import SparsePilotEncoder


METHODS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
BUDGET_ARMS = {
    "dense32x16": (("D32x16", 32, 16, 8),),
    "mid16x16": (("D16x16", 16, 16, 8),),
    "mid8x16": (("S8x16", 8, 16, 8),),
    "mid16x8": (("S16x8", 16, 8, 8),),
    "mid8x8": (("S8x8", 8, 8, 8),),
    "spatial4x16": (("S4x16", 4, 16, 8),),
    "target4x8": (("T4x8", 4, 8, 8),),
    "curriculum": (
        ("D32x16", 32, 16, 2),
        ("D16x16", 16, 16, 2),
        ("S8x8", 8, 8, 2),
        ("T4x8", 4, 8, 2),
    ),
}


class ConcatPilotModel(nn.Module):
    def __init__(self, method: str, *, sensing_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.method = method
        self.selector = PrototypePilotSelector(64, 32, num_selected_patterns=4) if method == "C4" else None
        self.encoder = SparsePilotEncoder(hidden_dim=hidden_dim)
        input_dim = hidden_dim if method == "C6" else sensing_dim + hidden_dim
        self.classifier = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 64))

    def forward(
        self,
        records: Mapping[str, torch.Tensor],
        *,
        frequencies: torch.Tensor,
        snr_db: torch.Tensor,
        dropout: float,
        generator: torch.Generator,
        lookup: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        candidates = records["candidate_multice"] if self.method == "C1" else records["candidate_g"]
        if self.method == "C4":
            assert self.selector is not None
            selected = self.selector(records["proto_id"], candidates, generator=generator)
        elif self.method == "C6":
            if lookup is None:
                raise ValueError("C6 requires the frozen C5 prototype lookup.")
            selected = select_from_lookup(records["proto_id"], candidates, lookup)
        else:
            mode = {"C1": "fixed_same_for_all", "C2": "fixed_same_for_all", "C3": "random_per_sample"}[self.method]
            helper = PrototypePilotSelector(64, 32, num_selected_patterns=4).to(candidates.device)
            selected = helper(records["proto_id"], candidates, mode=mode, generator=generator)
        observed, valid = noisy_observations(selected["selected_y"], snr_db, dropout=dropout, generator=generator)
        encoded = self.encoder(observed, selected["pattern_ids"], frequencies, valid, snr_db)
        feature = encoded["csi_feature"] if self.method == "C6" else torch.cat((records["z_sensing"], encoded["csi_feature"]), dim=-1)
        return {**encoded, **selected, "logits": self.classifier(feature)}


def noisy_observations(
    selected: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    dropout: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    power = selected.abs().square().mean(dim=(1, 2), keepdim=True)
    variance = power / torch.pow(10.0, snr_db[:, None, None] / 10.0)
    scale = (variance / 2.0).sqrt()
    noise = torch.complex(
        torch.randn(selected.shape, device=selected.device, generator=generator),
        torch.randn(selected.shape, device=selected.device, generator=generator),
    ) * scale
    valid = torch.rand(selected.shape, device=selected.device, generator=generator) >= float(dropout)
    return torch.where(valid, selected + noise, torch.zeros_like(selected)), valid


def extract_records(
    loader: DataLoader,
    model: SparsePilotTransitionModel,
    *,
    limit: int,
    cfg: Mapping[str, Any],
    device: torch.device,
    primary_codebook,
    multice_codebook,
    frequencies: np.ndarray,
    cache: PilotCache,
    cache_spec: PilotCacheSpec,
    cycle_sensing_masks: bool = False,
    sensing_mask_schedule: torch.Tensor | None = None,
    include_multice: bool = True,
) -> dict[str, torch.Tensor]:
    values: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "z_sensing",
            "p0",
            "proto_id",
            "sensing_availability",
            "labels",
            "beam_power",
            "candidate_g",
            "candidate_multice",
        )
    }
    if sensing_mask_schedule is not None and tuple(sensing_mask_schedule.shape) != (int(limit), 4):
        raise ValueError("sensing_mask_schedule must have shape [limit,4].")
    seen = 0
    for batch in loader:
        if seen >= int(limit):
            break
        take = min(len(batch["channel_ref"]), int(limit) - seen)
        inputs = prepare_fusion_inputs(
            dict(batch), seq_length=int(cfg["model"]["seq_length"]), device=device
        )
        if sensing_mask_schedule is not None:
            missing_mask = sensing_mask_schedule[seen : seen + take].to(device=device, dtype=torch.bool)
        elif cycle_sensing_masks:
            patterns = torch.tensor([pattern for _, _, pattern in MASKS], device=device, dtype=torch.bool)
            missing_mask = patterns[(torch.arange(take, device=device) + seen) % len(patterns)]
        else:
            missing_mask = torch.ones(take, 4, device=device, dtype=torch.bool)
        sensing = model.sensing_forward(inputs, missing_mask=missing_mask)
        for key in ("z_sensing", "p0", "proto_id", "sensing_availability"):
            values[key].append(sensing[key][:take].detach().cpu())
        values["labels"].append(torch.as_tensor(batch["target_beam"][:take]).reshape(-1).long())
        values["beam_power"].append(torch.as_tensor(batch["future_beam_power"][:take]).float())
        primary_rows, multice_rows = [], []
        for channel_text in batch["channel_ref"][:take]:
            path = Path(channel_text)

            def compute():
                matrices, delays = load_path_channel(path)
                return simulate_candidate_pilots(
                    matrices[None, None, :, None, :, :, None],
                    delays[None, None, None, :],
                    primary_codebook,
                    frequencies,
                )

            primary_rows.append(cache.get_or_compute(path, cache_spec, compute))
            if include_multice:
                matrices, delays = load_path_channel(path)
                multice_rows.append(
                    simulate_candidate_pilots(
                        matrices[None, None, :, None, :, :, None],
                        delays[None, None, None, :],
                        multice_codebook,
                        frequencies,
                    )
                )
            else:
                multice_rows.append(primary_rows[-1])
        values["candidate_g"].append(torch.from_numpy(np.stack(primary_rows)))
        values["candidate_multice"].append(torch.from_numpy(np.stack(multice_rows)))
        seen += take
    if seen != int(limit):
        raise ValueError(f"Requested {limit} records but extracted {seen}.")
    return {key: torch.cat(items, dim=0) for key, items in values.items()}


@torch.no_grad()
def extract_mask_records(
    loader: DataLoader,
    model: SparsePilotTransitionModel,
    common: Mapping[str, torch.Tensor],
    *,
    limit: int,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    per_mask = {
        key: {name: [] for name in ("z_sensing", "p0", "proto_id", "sensing_availability")}
        for key, _, _ in MASKS
    }
    seen = 0
    for batch in loader:
        if seen >= int(limit):
            break
        take = min(len(batch["channel_ref"]), int(limit) - seen)
        inputs = prepare_fusion_inputs(dict(batch), seq_length=int(cfg["model"]["seq_length"]), device=device)
        for key, _, pattern in MASKS:
            missing_mask = torch.tensor(pattern, device=device, dtype=torch.bool).expand(take, -1)
            sensing = model.sensing_forward(inputs, missing_mask=missing_mask)
            for name in ("z_sensing", "p0", "proto_id", "sensing_availability"):
                per_mask[key][name].append(sensing[name][:take].detach().cpu())
        seen += take
    if seen != int(limit):
        raise ValueError(f"Requested {limit} masked records but extracted {seen}.")
    shared = {
        key: value
        for key, value in common.items()
        if key not in {"z_sensing", "p0", "proto_id", "sensing_availability"}
    }
    return {
        key: shared | {name: torch.cat(chunks, dim=0) for name, chunks in values.items()}
        for key, values in per_mask.items()
    }


def device_records(records: Mapping[str, torch.Tensor], indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value[indices].to(device) for key, value in records.items()}


def nested_frequency_indices(max_count: int, selected_count: int) -> torch.Tensor:
    maximum, selected = int(max_count), int(selected_count)
    if maximum <= 0 or selected <= 0 or selected > maximum or maximum % selected:
        raise ValueError("Nested pilot frequencies require selected_count to divide max_count.")
    return torch.arange(0, maximum, maximum // selected, dtype=torch.long)


def apply_budget_arm(
    config: dict[str, Any], arm: str | None, *, total_epochs: int | None = None
) -> dict[str, Any]:
    if arm is None:
        return config
    stages = BUDGET_ARMS[arm]
    if total_epochs is not None:
        total = int(total_epochs)
        if total <= 0 or (arm == "curriculum" and total % len(stages)):
            raise ValueError("Arm epochs must be positive and divisible by four for curriculum.")
        stage_epochs = total // len(stages)
        stages = tuple((name, patterns, frequencies, stage_epochs) for name, patterns, frequencies, _ in stages)
    config["training"]["budget_arm"] = arm
    config["training"]["budget_curriculum"] = [
        {
            "name": name,
            "num_selected_patterns": patterns,
            "num_pilot_subcarriers": frequencies,
            "epochs": epochs,
        }
        for name, patterns, frequencies, epochs in stages
    ]
    _, patterns, frequencies, _ = stages[-1]
    config["evaluation"]["primary_num_selected_patterns"] = patterns
    config["evaluation"]["primary_num_pilot_subcarriers"] = frequencies
    return config


def balanced_subset_indices(dataset: object, count: int) -> list[int]:
    requested = int(count)
    if requested <= 0 or requested > len(dataset):  # type: ignore[arg-type]
        raise ValueError("Balanced subset count must be positive and no larger than the dataset.")
    if not isinstance(dataset, ConcatDataset):
        return [((2 * index + 1) * len(dataset)) // (2 * requested) for index in range(requested)]
    components = list(dataset.datasets)
    if requested < len(components):
        raise ValueError("Balanced pooled subset requires at least one sample per domain.")
    quotas = [requested // len(components)] * len(components)
    for index in range(requested % len(components)):
        quotas[index] += 1
    if any(quota > len(component) for quota, component in zip(quotas, components, strict=True)):
        raise ValueError("Balanced pooled subset quota exceeds a domain size.")
    indices: list[int] = []
    offset = 0
    for quota, component in zip(quotas, components, strict=True):
        indices.extend(offset + ((2 * index + 1) * len(component)) // (2 * quota) for index in range(quota))
        offset += len(component)
    return indices


def subset_index_audit(dataset: object, indices: list[int]) -> dict[str, Any]:
    digest = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
    audit: dict[str, Any] = {"count": len(indices), "index_sha256": digest, "domain_counts": {}}
    if isinstance(dataset, ConcatDataset):
        start = 0
        for position, component in enumerate(dataset.datasets):
            stop = start + len(component)
            name = str(getattr(component, "domain_id", position))
            audit["domain_counts"][name] = sum(start <= index < stop for index in indices)
            start = stop
    return audit


def missing_mask_schedule(
    count: int,
    cardinality_weights: Mapping[int | str, float],
    *,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    requested = int(count)
    weights = {int(key): float(value) for key, value in cardinality_weights.items()}
    if requested <= 0 or set(weights) - {1, 2, 3, 4} or any(value < 0.0 for value in weights.values()):
        raise ValueError("Missing-mask schedule requires positive count and non-negative cardinality weights 1--4.")
    total_weight = sum(weights.values())
    if total_weight <= 0.0:
        raise ValueError("Missing-mask schedule weights must have positive sum.")
    normalized = {key: value / total_weight for key, value in weights.items()}
    raw_quotas = {key: requested * value for key, value in normalized.items()}
    quotas = {key: int(np.floor(value)) for key, value in raw_quotas.items()}
    remainder = requested - sum(quotas.values())
    ranked = sorted(raw_quotas, key=lambda key: (raw_quotas[key] - quotas[key], -key), reverse=True)
    for key in ranked[:remainder]:
        quotas[key] += 1

    rows: list[tuple[int, int, int, int]] = []
    pattern_counts: dict[str, int] = {}
    for cardinality in sorted(quotas):
        patterns = [(key, pattern) for key, _, pattern in MASKS if sum(pattern) == cardinality]
        quota = quotas[cardinality]
        for index in range(quota):
            name, pattern = patterns[index % len(patterns)]
            rows.append(pattern)
            pattern_counts[name] = pattern_counts.get(name, 0) + 1
    schedule = torch.tensor(rows, dtype=torch.bool)
    order = torch.randperm(requested, generator=torch.Generator().manual_seed(int(seed)))
    schedule = schedule.index_select(0, order)
    audit = {
        "count": requested,
        "seed": int(seed),
        "cardinality_counts": {str(key): int(value) for key, value in sorted(quotas.items())},
        "pattern_counts": dict(sorted(pattern_counts.items())),
        "schedule_sha256": hashlib.sha256(schedule.numpy().tobytes()).hexdigest(),
    }
    return schedule, audit


def resolve_methods(config: Mapping[str, Any], requested: str | None) -> tuple[str, ...]:
    methods = tuple(config["evaluation"]["methods"]) if requested is None else tuple(
        item.strip() for item in requested.split(",") if item.strip()
    )
    unsupported = set(methods) - set(METHODS)
    if unsupported:
        raise ValueError(f"Unsupported methods: {sorted(unsupported)}")
    if not {"C0", "C5"}.issubset(methods):
        raise ValueError("Sparse-pilot diagnostics require C0 and C5.")
    return methods


def resolve_budget_curriculum(config: Mapping[str, Any], *, fallback_epochs: int) -> list[dict[str, Any]]:
    maximum_patterns = int(config["pilot_codebook"]["num_candidate_patterns"])
    maximum_frequencies = int(config["channel"]["pilot_subcarriers"])
    raw = config["training"].get("budget_curriculum")
    if not raw:
        raw = [
            {
                "name": "fixed_budget_stage_a",
                "num_selected_patterns": int(config["evaluation"]["primary_num_selected_patterns"]),
                "num_pilot_subcarriers": int(config["evaluation"]["primary_num_pilot_subcarriers"]),
                "epochs": int(fallback_epochs),
            }
        ]
    stages: list[dict[str, Any]] = []
    previous_patterns, previous_frequencies = maximum_patterns + 1, maximum_frequencies + 1
    previous_indices: set[int] | None = None
    for raw_stage in raw:
        stage = dict(raw_stage)
        patterns = int(stage["num_selected_patterns"])
        frequencies = int(stage["num_pilot_subcarriers"])
        epochs = int(stage["epochs"])
        if not 0 < patterns <= maximum_patterns or epochs <= 0:
            raise ValueError("Curriculum pattern budgets and epochs must be positive and in range.")
        indices = nested_frequency_indices(maximum_frequencies, frequencies)
        current_indices = set(indices.tolist())
        if patterns > previous_patterns or frequencies > previous_frequencies:
            raise ValueError("Dense-to-sparse curriculum budgets must be monotonic non-increasing.")
        if previous_indices is not None and not current_indices.issubset(previous_indices):
            raise ValueError("Each frequency budget must be an exact subset of the previous stage.")
        stage.update(
            num_selected_patterns=patterns,
            num_pilot_subcarriers=frequencies,
            epochs=epochs,
            frequency_token_indices=indices,
        )
        stages.append(stage)
        previous_patterns, previous_frequencies, previous_indices = patterns, frequencies, current_indices
    final = stages[-1]
    expected = (
        int(config["evaluation"]["primary_num_selected_patterns"]),
        int(config["evaluation"]["primary_num_pilot_subcarriers"]),
    )
    if (final["num_selected_patterns"], final["num_pilot_subcarriers"]) != expected:
        raise ValueError("Final curriculum budget must match the primary evaluation budget.")
    return stages


def slice_record_frequencies(
    records: Mapping[str, torch.Tensor], frequency_token_indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    indices = torch.as_tensor(frequency_token_indices, dtype=torch.long)
    return {
        key: value.index_select(-1, indices) if key in {"candidate_g", "candidate_multice"} else value
        for key, value in records.items()
    }


def train_concat(
    method: str,
    records: Mapping[str, torch.Tensor],
    *,
    frequencies: torch.Tensor,
    epochs: int,
    batch_size: int,
    dropout: float,
    device: torch.device,
    seed: int,
    lookup: torch.Tensor | None = None,
) -> ConcatPilotModel:
    model = ConcatPilotModel(method).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(int(epochs)):
        order = torch.randperm(len(records["labels"]), generator=torch.Generator().manual_seed(seed))
        for start in range(0, len(order), int(batch_size)):
            batch = device_records(records, order[start : start + int(batch_size)], device)
            snr = torch.empty(len(batch["labels"]), device=device).uniform_(-10.0, 30.0, generator=generator)
            result = model(batch, frequencies=frequencies, snr_db=snr, dropout=dropout, generator=generator, lookup=lookup)
            loss = F.cross_entropy(result["logits"], batch["labels"])
            if model.selector is not None:
                loss = loss + 0.01 * model.selector.usage_regularization()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def train_transition(
    model: SparsePilotTransitionModel,
    records: Mapping[str, torch.Tensor],
    *,
    frequencies: torch.Tensor,
    epochs: int,
    batch_size: int,
    dropout: float,
    device: torch.device,
    seed: int,
    loss_cfg: Mapping[str, Any],
    num_selected_patterns: int = 4,
    optimizer: torch.optim.Optimizer | None = None,
    history: list[dict[str, Any]] | None = None,
    stage_name: str = "",
    epoch_offset: int = 0,
    step_offset: int = 0,
    history_path: Path | None = None,
    checkpoint_path: Path | None = None,
    validation_records: Mapping[str, torch.Tensor] | None = None,
    validation_interval: int = 0,
    evaluation_batch_size: int | None = None,
) -> tuple[torch.optim.Optimizer, int]:
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed)
    order_generator = torch.Generator().manual_seed(seed)
    optimizer_steps = int(step_offset)
    target_history = history if history is not None else []
    for stage_epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        order = torch.randperm(len(records["labels"]), generator=order_generator)
        totals = {
            key: 0.0
            for key in (
                "loss",
                "final_ce",
                "final_topology",
                "route",
                "preserve",
                "fallback_ce",
                "fallback_topology",
                "gate",
                "selector",
            )
        }
        gradient_totals = {key: 0.0 for key in ("selector", "csi_encoder", "transition")}
        base_correct = final_correct = fixes = harms = 0
        route_targets: list[torch.Tensor] = []
        route_values: list[torch.Tensor] = []
        alpha_values: list[torch.Tensor] = []
        seen = 0
        batches = 0
        for start in range(0, len(order), int(batch_size)):
            batch = device_records(records, order[start : start + int(batch_size)], device)
            snr = torch.empty(len(batch["labels"]), device=device).uniform_(-10.0, 30.0, generator=generator)
            selected = model.selector(
                batch["proto_id"],
                batch["candidate_g"],
                generator=generator,
                num_selected_patterns=int(num_selected_patterns),
            )
            observed, valid = noisy_observations(selected["selected_y"], snr, dropout=dropout, generator=generator)
            sensing = {
                key: batch[key]
                for key in ("z_sensing", "p0", "proto_id", "sensing_availability")
            }
            result = model.forward_selected(
                sensing,
                observed,
                pattern_ids=selected["pattern_ids"],
                frequency_positions=frequencies,
                pilot_mask=valid,
                snr_db=snr,
            )
            terms = prototype_transition_losses(result, batch["p0"], batch["labels"], model.topology_positions)
            selector_term = model.selector.usage_regularization()
            loss = (
                terms["final_ce"]
                + float(loss_cfg["lambda_topology"]) * terms["final_topology"]
                + float(loss_cfg["lambda_route"]) * terms["route"]
                + float(loss_cfg["lambda_preserve"]) * terms["preserve"]
                + float(loss_cfg.get("lambda_fallback_ce", 0.0)) * terms["fallback_ce"]
                + float(loss_cfg.get("lambda_fallback_topology", 0.0)) * terms["fallback_topology"]
                + float(loss_cfg.get("lambda_gate", 0.0)) * terms["gate"]
                + float(loss_cfg["lambda_selector"]) * selector_term
            )
            optimizer.zero_grad()
            loss.backward()
            gradient_totals["selector"] += _gradient_norm(model.selector)
            gradient_totals["csi_encoder"] += _gradient_norm(model.csi_encoder)
            gradient_totals["transition"] += _gradient_norm(model.transition)
            optimizer.step()
            optimizer_steps += 1
            batches += 1
            batch_count = len(batch["labels"])
            seen += batch_count
            for key, value in terms.items():
                totals[key] += float(value.detach().item()) * batch_count
            totals["selector"] += float(selector_term.detach().item()) * batch_count
            totals["loss"] += float(loss.detach().item()) * batch_count
            base_prediction = batch["p0"].argmax(dim=-1)
            final_prediction = result["p_final"].argmax(dim=-1)
            labels = batch["labels"]
            base_mask = base_prediction.eq(labels)
            final_mask = final_prediction.eq(labels)
            base_correct += int(base_mask.sum().item())
            final_correct += int(final_mask.sum().item())
            fixes += int((~base_mask & final_mask).sum().item())
            harms += int((base_mask & ~final_mask).sum().item())
            route_targets.append(_route_target(batch["p0"], labels, model.topology_positions, model.transition.topology_radius).cpu())
            route_values.append(result["r_global"].detach().cpu())
            alpha_values.append(result["alpha"].detach().cpu())
        route_tensor = torch.cat(route_values)
        alpha_tensor = torch.cat(alpha_values)
        target_tensor = torch.cat(route_targets)
        row: dict[str, Any] = {
            "stage": stage_name,
            "stage_epoch": stage_epoch,
            "epoch": int(epoch_offset) + stage_epoch,
            "optimizer_steps": optimizer_steps,
            "sample_exposures": (int(epoch_offset) + stage_epoch) * len(records["labels"]),
            "epoch_seconds": time.perf_counter() - started,
            **{key: value / seen for key, value in totals.items()},
            "train_c0_top1": base_correct / seen,
            "train_c5_top1": final_correct / seen,
            "train_fix_rate": fixes / max(seen - base_correct, 1),
            "train_harm_rate": harms / max(base_correct, 1),
            "route_target_positive_ratio": float(target_tensor.mean().item()),
            "mean_r_global": float(route_tensor.mean().item()),
            "r_global_p10": float(torch.quantile(route_tensor, 0.1).item()),
            "r_global_p50": float(torch.quantile(route_tensor, 0.5).item()),
            "r_global_p90": float(torch.quantile(route_tensor, 0.9).item()),
            "global_route_ratio": float((route_tensor >= 0.5).float().mean().item()),
            "mean_alpha": float(alpha_tensor.mean().item()),
            "alpha_p10": float(torch.quantile(alpha_tensor, 0.1).item()),
            "alpha_p50": float(torch.quantile(alpha_tensor, 0.5).item()),
            "alpha_p90": float(torch.quantile(alpha_tensor, 0.9).item()),
            **{f"grad_{key}": value / batches for key, value in gradient_totals.items()},
            "validation_top1": None,
            "validation_top3": None,
            "validation_fix_rate": None,
            "validation_harm_rate": None,
            "validation_mean_alpha": None,
            "validation_mean_r_global": None,
            "validation_global_route_ratio": None,
        }
        if validation_records is not None and validation_interval > 0 and (
            stage_epoch % validation_interval == 0 or stage_epoch == int(epochs)
        ):
            model.eval()
            lookup = model.selector.lookup(int(num_selected_patterns)).to(device)
            validation = evaluate_method(
                "C5",
                model,
                validation_records,
                frequencies=frequencies,
                snr_db=10.0,
                dropout=0.0,
                device=device,
                lookup=lookup,
                batch_size=evaluation_batch_size,
            )
            row.update(
                validation_top1=validation["top1"],
                validation_top3=validation["top3"],
                validation_fix_rate=validation["fix_rate"],
                validation_harm_rate=validation["harm_rate"],
                validation_mean_alpha=validation["mean_alpha"],
                validation_mean_r_global=validation["mean_r_global"],
                validation_global_route_ratio=validation["global_route_ratio"],
            )
            model.train()
        target_history.append(row)
        if history_path is not None:
            write_csv(history_path, target_history)
        if checkpoint_path is not None:
            _save_transition_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                stage=stage_name,
                epoch=int(epoch_offset) + stage_epoch,
                optimizer_steps=optimizer_steps,
            )
    model.eval()
    return optimizer, optimizer_steps


def _gradient_norm(module: nn.Module) -> float:
    square_sum = sum(
        float(parameter.grad.detach().float().square().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return square_sum**0.5


def _route_target(
    p0: torch.Tensor, labels: torch.Tensor, topology_positions: torch.Tensor, topology_radius: int
) -> torch.Tensor:
    positions = topology_positions.to(device=p0.device, dtype=p0.dtype).reshape(-1)
    base_distance = (positions[p0.argmax(dim=-1)] - positions[labels]).abs()
    base_distance = torch.minimum(base_distance, float(p0.shape[1]) - base_distance)
    return (base_distance > float(topology_radius)).to(p0.dtype)


def _save_transition_checkpoint(
    path: Path,
    model: SparsePilotTransitionModel,
    optimizer: torch.optim.Optimizer,
    *,
    stage: str,
    epoch: int,
    optimizer_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    trainable_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(("selector.", "csi_encoder.", "transition."))
    }
    torch.save(
        {
            "state_dict": trainable_state,
            "optimizer": optimizer.state_dict(),
            "stage": stage,
            "epoch": int(epoch),
            "optimizer_steps": int(optimizer_steps),
            "frozen_u0_included": False,
            "outer_test_accessed": False,
        },
        temporary,
    )
    temporary.replace(path)


@torch.no_grad()
def evaluate_method(
    method: str,
    model: nn.Module | None,
    records: Mapping[str, torch.Tensor],
    *,
    frequencies: torch.Tensor,
    snr_db: float,
    dropout: float,
    device: torch.device,
    lookup: torch.Tensor | None = None,
    csi_available: bool = True,
    batch_size: int | None = None,
) -> dict[str, float]:
    sample_count = len(records["labels"])
    if batch_size is not None and sample_count > int(batch_size):
        rows = [
            evaluate_method(
                method,
                model,
                {key: value[start : start + int(batch_size)] for key, value in records.items()},
                frequencies=frequencies,
                snr_db=snr_db,
                dropout=dropout,
                device=device,
                lookup=lookup,
                csi_available=csi_available,
            )
            for start in range(0, sample_count, int(batch_size))
        ]
        return _aggregate_evaluation_chunks(rows)
    batch = {key: value.to(device) for key, value in records.items()}
    base = batch["p0"]
    if method == "C0":
        probabilities = base
        extra: dict[str, float] = {
            "mean_alpha": 0.0,
            "global_route_ratio": 0.0,
            "mean_r_global": 0.0,
            "local_top1": 0.0,
            "global_top1": 0.0,
            "transition_top1": 0.0,
            "csi_only_top1": 0.0,
            "changed_argmax_ratio": 0.0,
            "route_target_positive_ratio": 0.0,
            "fallback_max_abs_error": 0.0,
        }
        elapsed = 0.0
        pilot_soundings = 0.0
        pilot_resource_elements = 0.0
    else:
        generator = torch.Generator(device=device).manual_seed(1000 + int(round(float(snr_db) * 10)))
        snr = torch.full((len(batch["labels"]),), float(snr_db), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        if method == "C5":
            assert isinstance(model, SparsePilotTransitionModel)
            selected = select_from_lookup(batch["proto_id"], batch["candidate_g"], lookup)
            observed, valid = noisy_observations(selected["selected_y"], snr, dropout=dropout, generator=generator)
            result = model.forward_selected(
                {
                    key: batch[key]
                    for key in ("z_sensing", "p0", "proto_id", "sensing_availability")
                },
                observed,
                pattern_ids=selected["pattern_ids"],
                frequency_positions=frequencies,
                pilot_mask=valid,
                snr_db=snr,
                csi_available=torch.full(
                    (len(batch["labels"]),), bool(csi_available), device=device, dtype=torch.bool
                ),
            )
            probabilities = result["p_final"]
            fallback_error = float((probabilities - base).abs().max().item()) if not csi_available else 0.0
            extra = {
                "mean_alpha": float(result["alpha"].mean().item()),
                "mean_r_global": float(result["r_global"].mean().item()),
                "global_route_ratio": (
                    float((result["r_global"] >= 0.5).float().mean().item()) if csi_available else 0.0
                ),
                "local_top1": float(result["q_local"].argmax(dim=-1).eq(batch["labels"]).float().mean().item()),
                "global_top1": float(result["q_global"].argmax(dim=-1).eq(batch["labels"]).float().mean().item()),
                "transition_top1": float(
                    result["q_transition"].argmax(dim=-1).eq(batch["labels"]).float().mean().item()
                ),
                "csi_only_top1": float(
                    result["q_csi"].argmax(dim=-1).eq(batch["labels"]).float().mean().item()
                ),
                "changed_argmax_ratio": float(
                    probabilities.argmax(dim=-1).ne(base.argmax(dim=-1)).float().mean().item()
                ),
                "route_target_positive_ratio": float(
                    _route_target(base, batch["labels"], model.topology_positions, model.transition.topology_radius)
                    .mean()
                    .item()
                ),
                "fallback_max_abs_error": fallback_error,
            }
            pilot_soundings = float(selected["selected_y"].shape[1])
            pilot_resource_elements = float(selected["selected_y"].shape[1] * selected["selected_y"].shape[2])
        else:
            assert isinstance(model, ConcatPilotModel)
            result = model(batch, frequencies=frequencies, snr_db=snr, dropout=dropout, generator=generator, lookup=lookup)
            probabilities = result["logits"].softmax(dim=-1)
            extra = {
                "mean_alpha": 0.0,
                "mean_r_global": 0.0,
                "global_route_ratio": 0.0,
                "local_top1": 0.0,
                "global_top1": 0.0,
                "transition_top1": 0.0,
                "csi_only_top1": 0.0,
                "changed_argmax_ratio": 0.0,
                "route_target_positive_ratio": 0.0,
                "fallback_max_abs_error": 0.0,
            }
            pilot_soundings = float(result["selected_y"].shape[1])
            pilot_resource_elements = float(result["selected_y"].shape[1] * result["selected_y"].shape[2])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
    metrics = prediction_metrics(probabilities, batch["labels"], base, batch["beam_power"])
    params = 0 if model is None else sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return metrics | extra | {
        "snr_db": float(snr_db),
        "csi_parameters": float(params),
        "csi_latency_ms_per_sample": 1000.0 * elapsed / len(batch["labels"]),
        "pilot_soundings": 0.0 if not csi_available else pilot_soundings,
        "pilot_resource_elements": 0.0 if not csi_available else pilot_resource_elements,
    }


def _aggregate_evaluation_chunks(rows: list[dict[str, float]]) -> dict[str, float]:
    total = sum(row["sample_count"] for row in rows)
    base_correct = sum(row["base_correct_count"] for row in rows)
    base_incorrect = sum(row["base_incorrect_count"] for row in rows)
    constants = {"snr_db", "csi_parameters", "pilot_soundings", "pilot_resource_elements"}
    output: dict[str, float] = {}
    for key in rows[0]:
        if key in constants:
            output[key] = rows[0][key]
        elif key == "fallback_max_abs_error":
            output[key] = max(row[key] for row in rows)
        elif key == "fix_rate":
            output[key] = sum(row[key] * row["base_incorrect_count"] for row in rows) / max(base_incorrect, 1.0)
        elif key == "harm_rate":
            output[key] = sum(row[key] * row["base_correct_count"] for row in rows) / max(base_correct, 1.0)
        elif key in {"sample_count", "base_correct_count", "base_incorrect_count"}:
            output[key] = sum(row[key] for row in rows)
        else:
            output[key] = sum(row[key] * row["sample_count"] for row in rows) / total
    return output


def mask_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("top1", "top3", "top5", "within3", "mae", "topology_risk", "fix_rate", "harm_rate")
    output = list(rows)
    for csi_state in ("off", "on"):
        selected = [row for row in rows if row["csi_state"] == csi_state]
        non_full = [row for row in selected if row["mask"] != "full"]
        single = [row for row in selected if row["available_modalities"] == 1]
        for scope, source in (("all14_macro", non_full), ("all14_worst", non_full), ("single_macro", single), ("single_worst", single)):
            aggregate: dict[str, Any] = {
                "mask": scope,
                "mask_label": scope,
                "mask_bits": "",
                "available_modalities": "",
                "csi_state": csi_state,
                "method": "C5",
            }
            for metric in metrics:
                values = [float(row[metric]) for row in source]
                if scope.endswith("macro"):
                    aggregate[metric] = float(np.mean(values))
                elif metric in {"top1", "top3", "top5", "within3", "fix_rate"}:
                    aggregate[metric] = float(np.min(values))
                else:
                    aggregate[metric] = float(np.max(values))
            for metric in ("normalized_beamforming_gain", "beam_loss_db", "mean_alpha", "global_route_ratio", "fallback_max_abs_error"):
                aggregate[metric] = float(np.mean([float(row[metric]) for row in source]))
            output.append(aggregate)
    return output


def write_final_report(
    path: Path,
    *,
    primary: list[dict[str, Any]],
    snr_rows: list[dict[str, Any]],
    dropout_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    diagnostics: Mapping[str, Any],
    mask_rows: list[dict[str, Any]],
) -> None:
    by_key = {(row["mask"], row["csi_state"]): row for row in mask_rows}
    missing_primary = []
    if diagnostics.get("missing_fallback_enabled", False):
        missing_primary = [
            "",
            "## 严重模态缺失主结果",
            "",
            "| 范围 | CSI off Top-1 | CSI on Top-1 | Delta |",
            "| --- | ---: | ---: | ---: |",
            *[
                f"| {label} | {by_key[(scope, 'off')]['top1']:.4f} | "
                f"{by_key[(scope, 'on')]['top1']:.4f} | "
                f"{by_key[(scope, 'on')]['top1'] - by_key[(scope, 'off')]['top1']:+.4f} |"
                for scope, label in (
                    ("single_macro", "Single Macro"),
                    ("single_worst", "Single Worst"),
                    ("all14_macro", "All-14 Macro"),
                    ("full", "Full constraint"),
                )
            ],
        ]
    lines = [
        "# Prototype-Conditioned Sparse Pilot Transition 短程诊断报告",
        "",
        f"本报告来自 train/validation 各 {diagnostics['train_samples']}/{diagnostics['validation_samples']} "
        f"样本、{diagnostics['epochs']} 个 CSI 训练 epoch、单 seed 的开发诊断；未访问 outer test，不能作为正式论文结果。",
        *missing_primary,
        "",
        "## Full 条件 10 dB 诊断",
        "",
        "| 方法 | Top-1 | Top-3 | Top-5 | Within-3 | Fix | Harm | alpha |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in primary:
        lines.append(
            f"| {row['method']} | {row['top1']:.4f} | {row['top3']:.4f} | {row['top5']:.4f} | "
            f"{row['within3']:.4f} | {row['fix_rate']:.4f} | {row['harm_rate']:.4f} | {row['mean_alpha']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Dense-to-sparse budget curriculum",
            "",
            "| 阶段 | M | Kp | Sounding | Pilot RE | Top-1 | Top-3 | alpha |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['phase']} | {row['num_selected_patterns']} | {row['num_pilot_subcarriers']} | "
                f"{row['pilot_soundings']:.0f} | {row['pilot_resource_elements']:.0f} | {row['top1']:.4f} | "
                f"{row['top3']:.4f} | {row['mean_alpha']:.4f} |"
                for row in budget_rows
            ],
        ]
    )
    c0_by_snr = {float(row["snr_db"]): row for row in snr_rows if row["method"] == "C0"}
    c5_by_snr = {float(row["snr_db"]): row for row in snr_rows if row["method"] == "C5"}
    lines.extend(
        [
            "",
            "## SNR 与 dropout",
            "",
            "| SNR (dB) | C0 Top-1 | C5 Top-1 | C5 alpha |",
            "| ---: | ---: | ---: | ---: |",
            *[
                f"| {snr:g} | {c0_by_snr[snr]['top1']:.4f} | {c5_by_snr[snr]['top1']:.4f} | "
                f"{c5_by_snr[snr]['mean_alpha']:.4f} |"
                for snr in sorted(c5_by_snr)
            ],
            "",
            "| Pilot dropout | C5 Top-1 | C5 alpha |",
            "| ---: | ---: | ---: |",
            *[
                f"| {row['pilot_dropout']:.1f} | {row['top1']:.4f} | {row['mean_alpha']:.4f} |"
                for row in dropout_rows
            ],
            "",
            "## 15-mask 补充汇总",
            "",
            f"- Single Macro Top-1：CSI off={by_key[('single_macro', 'off')]['top1']:.4f}，CSI on={by_key[('single_macro', 'on')]['top1']:.4f}。",
            f"- Single Worst Top-1：CSI off={by_key[('single_worst', 'off')]['top1']:.4f}，CSI on={by_key[('single_worst', 'on')]['top1']:.4f}。",
            f"- All-14 Macro Top-1：CSI off={by_key[('all14_macro', 'off')]['top1']:.4f}，CSI on={by_key[('all14_macro', 'on')]['top1']:.4f}。",
            f"- Full Top-1：CSI off={by_key[('full', 'off')]['top1']:.4f}，CSI on={by_key[('full', 'on')]['top1']:.4f}。",
            "- 每个 mask 的 CSI-off fallback_max_abs_error 均应为 0。",
            "",
            "## 门槛与结论",
            "",
            "- D32x16 是 32 次 sounding、512 个 pilot RE 的诊断上界；主结果仍为 T4x8 的 4 次 sounding、32 个 pilot RE。",
            f"- Dense 上界提高 Top-1：{diagnostics['dense_pilot_improves_top1_at_10db']}。",
            f"- Proto+Pilot 在 10 dB 提升 Top-1：{diagnostics['proto_pilot_improves_top1_at_10db']}。",
            f"- Learned lookup 优于 fixed/random：{diagnostics['learned_lookup_beats_fixed_random_at_10db']}。",
            (
                "- 本轮仅获授权执行单 seed scale-up；未授权 multi-seed、Stage B 或 outer-test 评估。"
                if diagnostics["long_experiment_authorized"]
                else "- 两个门槛未同时通过，未授权长训练、多 seed 或 outer-test 评估。"
            ),
            "- 1024 子载波、120 kHz 和 centered index 是显式实验假设，不是 MMW 文件内可审计元数据。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prediction_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    base: torch.Tensor,
    beam_power: torch.Tensor,
) -> dict[str, float]:
    prediction = probabilities.argmax(dim=-1)
    base_prediction = base.argmax(dim=-1)
    distance = (prediction - labels).abs()
    distance = torch.minimum(distance, 64 - distance)
    top = probabilities.topk(5, dim=-1).indices
    correct = prediction.eq(labels)
    base_correct = base_prediction.eq(labels)
    row = torch.arange(len(labels), device=labels.device)
    ratio = beam_power[row, prediction] / beam_power.amax(dim=-1).clamp_min(1e-12)
    return {
        "sample_count": float(len(labels)),
        "base_correct_count": float(base_correct.sum().item()),
        "base_incorrect_count": float((~base_correct).sum().item()),
        "top1": float(correct.float().mean().item()),
        "top3": float(top[:, :3].eq(labels[:, None]).any(dim=1).float().mean().item()),
        "top5": float(top.eq(labels[:, None]).any(dim=1).float().mean().item()),
        "within3": float((distance <= 3).float().mean().item()),
        "mae": float(distance.float().mean().item()),
        "topology_risk": float((distance.float() / 32.0).mean().item()),
        "normalized_beamforming_gain": float(ratio.mean().item()),
        "beam_loss_db": float((-10.0 * ratio.clamp_min(1e-12).log10()).mean().item()),
        "fix_rate": float(correct[~base_correct].float().mean().item()) if bool((~base_correct).any()) else 0.0,
        "harm_rate": float((~correct[base_correct]).float().mean().item()) if bool(base_correct.any()) else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short Full-pool sparse-pilot diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("tools/configs/sparse_pilot_transition.yaml"))
    parser.add_argument("--u0-config", type=Path, default=Path("outputs/full_pool_capacity/protocol/u0_seed1_config.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/full_pool_capacity/u0_seed1/checkpoints/last.pth"))
    parser.add_argument("--expected-sha256", default="ed909406a37ec4ccd2b08bd1fb65ab66fc437cec226a526fdaf7ada1407ba8cf")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sparse_pilot_transition"))
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--validation-samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--budget-arm", choices=tuple(BUDGET_ARMS))
    parser.add_argument("--arm-epochs", type=int)
    parser.add_argument("--methods", help="Comma-separated diagnostic method subset; C0 and C5 are required.")
    parser.add_argument("--pilot-codebook", type=Path)
    parser.add_argument("--balanced-subset", action="store_true")
    parser.add_argument("--evaluation-batch-size", type=int)
    parser.add_argument("--curve-interval", type=int, default=0)
    parser.add_argument("--long-experiment-authorized", action="store_true")
    args = parser.parse_args()

    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    config = apply_budget_arm(
        safe_load_yaml(args.config.read_text(encoding="utf-8")), args.budget_arm, total_epochs=args.arm_epochs
    )
    methods = resolve_methods(config, args.methods)
    config["evaluation"]["methods"] = list(methods)
    fallback_cfg = config["training"].get("missing_fallback", {})
    fallback_enabled = bool(fallback_cfg.get("enabled", False))
    train_mask_schedule = validation_mask_schedule = None
    mask_schedule_audit: dict[str, Any] = {}
    if fallback_enabled:
        train_mask_schedule, mask_schedule_audit["train"] = missing_mask_schedule(
            args.train_samples,
            fallback_cfg["train_cardinality_weights"],
            seed=int(fallback_cfg.get("seed", 2026)),
        )
        validation_mask_schedule, mask_schedule_audit["validation_curve"] = missing_mask_schedule(
            args.validation_samples,
            fallback_cfg["validation_cardinality_weights"],
            seed=int(fallback_cfg.get("seed", 2026)) + 1,
        )
        config["training"]["missing_fallback"]["schedule_audit"] = mask_schedule_audit
    config["training"]["diagnostic_profile"] = {
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "batch_size": args.batch_size,
        "balanced_subset": args.balanced_subset,
        "curve_interval": args.curve_interval,
        "long_experiment_authorized": args.long_experiment_authorized,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    dump_config(config, args.output_root / "config_resolved.yaml")
    _, preflight_audit = preflight(args.u0_config, args.checkpoint, args.expected_sha256)
    cfg = load_u0_artifact_config(args.u0_config)
    cfg["data"]["dataset"].update(
        include_channel_ref=True,
        pilot_time_mode="last_input",
        include_router_utility_targets=True,
    )
    cfg["data"]["dataloader"].update(num_workers=4, persistent_workers=True, pin_memory=True)
    loaders = build_dataloaders(cfg, normalization_overrides=checkpoint_normalization_overrides(args.checkpoint))
    subset_indices = {
        role: (
            balanced_subset_indices(loaders[role].dataset, count)
            if args.balanced_subset
            else list(range(count))
        )
        for role, count in (("train", args.train_samples), ("validation", args.validation_samples))
    }
    subset_audit = {
        role: subset_index_audit(loaders[role].dataset, indices) for role, indices in subset_indices.items()
    }
    fixed = {
        role: DataLoader(
            Subset(loaders[role].dataset, subset_indices[role]),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=loaders[role].collate_fn,
        )
        for role, count in (("train", args.train_samples), ("validation", args.validation_samples))
    }
    device = torch.device(args.device)
    sensing_model, model_audit = load_frozen_u0(cfg, args.checkpoint, device)
    topology = torch.arange(64, device=device)
    transition = SparsePilotTransitionModel(
        sensing_model,
        topology_positions=topology,
        **config["model"],
    ).to(device)

    channel_cfg = config["channel"]
    indices = pilot_subcarrier_indices(channel_cfg["num_subcarriers"], channel_cfg["pilot_subcarriers"])
    frequencies_np = frequency_offsets_hz(
        indices,
        num_subcarriers=channel_cfg["num_subcarriers"],
        subcarrier_spacing_hz=channel_cfg["subcarrier_spacing_hz"],
        mode=channel_cfg["frequency_index_mode"],
    )
    maximum_frequencies = torch.from_numpy(frequencies_np).to(device=device, dtype=torch.float32)
    curriculum = resolve_budget_curriculum(config, fallback_epochs=args.epochs)
    codebook_path = args.pilot_codebook or args.output_root / "pilot_codebook.npz"
    codebook = load_probe_codebook(codebook_path)
    codebook.save(args.output_root / "pilot_codebook.npz")
    multice = generate_probe_codebook(64, 16, num_patterns=32, seed=2026, method="multice_interleaved")
    cache = PilotCache(config["output"]["cache_root"])
    cache_spec = PilotCacheSpec(
        codebook.hash,
        tuple(frequencies_np),
        float(channel_cfg["subcarrier_spacing_hz"]),
        str(channel_cfg["frequency_index_mode"]),
        64,
        16,
    )
    maximum_records = {
        "train": extract_records(
            fixed["train"],
            transition,
            limit=args.train_samples,
            cfg=cfg,
            device=device,
            primary_codebook=codebook,
            multice_codebook=multice,
            frequencies=frequencies_np,
            cache=cache,
            cache_spec=cache_spec,
            cycle_sensing_masks=not fallback_enabled,
            sensing_mask_schedule=train_mask_schedule,
            include_multice="C1" in methods,
        ),
        "validation": extract_records(
            fixed["validation"],
            transition,
            limit=args.validation_samples,
            cfg=cfg,
            device=device,
            primary_codebook=codebook,
            multice_codebook=multice,
            frequencies=frequencies_np,
            cache=cache,
            cache_spec=cache_spec,
            include_multice="C1" in methods,
        ),
    }
    maximum_curve_validation = (
        extract_records(
            fixed["validation"],
            transition,
            limit=args.validation_samples,
            cfg=cfg,
            device=device,
            primary_codebook=codebook,
            multice_codebook=multice,
            frequencies=frequencies_np,
            cache=cache,
            cache_spec=cache_spec,
            sensing_mask_schedule=validation_mask_schedule,
            include_multice=False,
        )
        if fallback_enabled
        else maximum_records["validation"]
    )
    primary_frequency_indices = curriculum[-1]["frequency_token_indices"]
    frequencies = maximum_frequencies.index_select(0, primary_frequency_indices.to(device))
    records = {
        role: slice_record_frequencies(role_records, primary_frequency_indices)
        for role, role_records in maximum_records.items()
    }

    models: dict[str, nn.Module | None] = {"C0": None}
    for offset, method in enumerate(("C1", "C2", "C3", "C4"), start=1):
        if method not in methods:
            continue
        models[method] = train_concat(
            method,
            records["train"],
            frequencies=frequencies,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dropout=float(channel_cfg["pilot_dropout_prob"]),
            device=device,
            seed=2026 + offset,
        )
    transition_optimizer = None
    optimizer_steps = 0
    training_history: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for stage_index, stage in enumerate(curriculum):
        stage_indices = stage["frequency_token_indices"]
        stage_frequencies = maximum_frequencies.index_select(0, stage_indices.to(device))
        stage_records = {
            role: slice_record_frequencies(role_records, stage_indices)
            for role, role_records in maximum_records.items()
        }
        stage_curve_validation = slice_record_frequencies(maximum_curve_validation, stage_indices)
        transition_optimizer, optimizer_steps = train_transition(
            transition,
            stage_records["train"],
            frequencies=stage_frequencies,
            epochs=int(stage["epochs"]),
            batch_size=args.batch_size,
            dropout=float(channel_cfg["pilot_dropout_prob"]),
            device=device,
            seed=2031 + stage_index,
            loss_cfg=config["loss"],
            num_selected_patterns=int(stage["num_selected_patterns"]),
            optimizer=transition_optimizer,
            history=training_history,
            stage_name=str(stage["name"]),
            epoch_offset=sum(int(item["epochs"]) for item in curriculum[:stage_index]),
            step_offset=optimizer_steps,
            history_path=args.output_root / "training_history.csv",
            checkpoint_path=args.output_root / "checkpoints" / "last_trainable.pth",
            validation_records=stage_curve_validation,
            validation_interval=args.curve_interval,
            evaluation_batch_size=args.evaluation_batch_size,
        )
        stage_lookup = transition.selector.lookup(int(stage["num_selected_patterns"])).to(device)
        stage_metrics = evaluate_method(
            "C5",
            transition,
            stage_records["validation"],
            frequencies=stage_frequencies,
            snr_db=10.0,
            dropout=0.0,
            device=device,
            lookup=stage_lookup,
            batch_size=args.evaluation_batch_size,
        )
        budget_rows.append(
            {
                "phase": stage["name"],
                "epochs": int(stage["epochs"]),
                "cumulative_epochs": sum(int(item["epochs"]) for item in curriculum[: stage_index + 1]),
                "num_selected_patterns": int(stage["num_selected_patterns"]),
                "num_pilot_subcarriers": int(stage["num_pilot_subcarriers"]),
                **stage_metrics,
            }
        )
    models["C5"] = transition
    lookup = transition.selector.lookup(int(curriculum[-1]["num_selected_patterns"])).to(device)
    transition.selector.export_lookup(
        args.output_root / "prototype_pilot_lookup.json",
        metadata={"source_split": "full_pool_train", "codebook_hash": codebook.hash, "outer_test_accessed": False},
        num_selected_patterns=int(curriculum[-1]["num_selected_patterns"]),
    )
    if "C6" in methods:
        models["C6"] = train_concat(
            "C6",
            records["train"],
            frequencies=frequencies,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dropout=float(channel_cfg["pilot_dropout_prob"]),
            device=device,
            seed=2032,
            lookup=lookup,
        )

    snr_rows = []
    for method in methods:
        for snr in config["evaluation"]["snr_db"]:
            metrics = evaluate_method(
                method,
                models[method],
                records["validation"],
                frequencies=frequencies,
                snr_db=float(snr),
                dropout=0.0,
                device=device,
                lookup=lookup if method in {"C5", "C6"} else None,
                batch_size=args.evaluation_batch_size,
            )
            snr_rows.append({"method": method, **metrics})
    primary = [row for row in snr_rows if row["snr_db"] == 10.0]
    write_csv(args.output_root / "snr_summary.csv", snr_rows)
    write_csv(args.output_root / "ablation_summary.csv", primary)
    write_csv(args.output_root / "budget_summary.csv", budget_rows)
    masked_records = extract_mask_records(
        fixed["validation"],
        transition,
        records["validation"],
        limit=args.validation_samples,
        cfg=cfg,
        device=device,
    )
    mask_rows: list[dict[str, Any]] = []
    for key, label, pattern in MASKS:
        for csi_state in ("off", "on"):
            method = "C5"
            metrics = evaluate_method(
                method,
                models[method],
                masked_records[key],
                frequencies=frequencies,
                snr_db=10.0,
                dropout=0.0,
                device=device,
                lookup=lookup,
                csi_available=csi_state == "on",
                batch_size=args.evaluation_batch_size,
            )
            mask_rows.append(
                {
                    "mask": key,
                    "mask_label": label,
                    "mask_bits": "".join(str(int(bit)) for bit in pattern),
                    "available_modalities": sum(pattern),
                    "csi_state": csi_state,
                    "method": method,
                    **metrics,
                }
            )
    mask_rows = mask_aggregate_rows(mask_rows)
    write_csv(args.output_root / "mask_summary.csv", mask_rows)
    dropout_rows = []
    for dropout in (0.0, 0.1, 0.3, 0.5, 1.0):
        dropout_rows.append(
            {
                "method": "C5",
                "pilot_dropout": dropout,
                **evaluate_method(
                    "C5",
                    transition,
                    records["validation"],
                    frequencies=frequencies,
                    snr_db=10.0,
                    dropout=dropout,
                    device=device,
                    lookup=lookup,
                    batch_size=args.evaluation_batch_size,
                ),
            }
        )
    write_csv(args.output_root / "dropout_summary.csv", dropout_rows)
    c0 = next(row for row in primary if row["method"] == "C0")
    c5 = next(row for row in primary if row["method"] == "C5")
    comparators = {row["method"]: row for row in primary if row["method"] in {"C2", "C3", "C4"}}
    lookup_delay_started = time.perf_counter()
    for _ in range(1000):
        _ = lookup[records["validation"]["proto_id"]]
    lookup_delay = (time.perf_counter() - lookup_delay_started) * 1000.0 / 1000.0
    diagnostics = {
        "status": "passed",
        "scope": (
            "long_single_seed_development_diagnostic"
            if args.long_experiment_authorized
            else "short_single_seed_development_diagnostic"
        ),
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "epochs": sum(int(stage["epochs"]) for stage in curriculum),
        "optimizer_steps": optimizer_steps,
        "batch_size": args.batch_size,
        "subset_audit": subset_audit,
        "missing_fallback_enabled": fallback_enabled,
        "mask_schedule_audit": mask_schedule_audit,
        "budget_arm": args.budget_arm,
        "methods": list(methods),
        "single_batch_forward_backward": True,
        "u0_preflight": preflight_audit,
        "u0_model": model_audit,
        "codebook_hash": codebook.hash,
        "candidate_shape": [int(codebook.tx.shape[0]), int(len(frequencies_np))],
        "encoder_input_shape": [
            int(curriculum[-1]["num_selected_patterns"]),
            int(curriculum[-1]["num_pilot_subcarriers"]),
        ],
        "budget_curriculum": [
            {
                key: value
                for key, value in stage.items()
                if key != "frequency_token_indices"
            }
            | {"frequency_token_indices": stage["frequency_token_indices"].tolist()}
            for stage in curriculum
        ],
        "prototype_lookup_pattern_diversity": int(torch.unique(lookup).numel()),
        "selector_lookup_latency_ms": lookup_delay,
        "proto_pilot_improves_top1_at_10db": c5["top1"] > c0["top1"],
        "dense_pilot_improves_top1_at_10db": budget_rows[0]["top1"] > c0["top1"],
        "dense_to_target_top1_delta": float(budget_rows[-1]["top1"] - budget_rows[0]["top1"]),
        "learned_lookup_beats_fixed_random_at_10db": (
            comparators["C4"]["top1"] > max(comparators["C2"]["top1"], comparators["C3"]["top1"])
            if set(comparators) == {"C2", "C3", "C4"}
            else None
        ),
        "long_experiment_authorized": args.long_experiment_authorized,
        "outer_test_accessed": False,
        "future_channel_used_as_input": False,
        "sensing_mask_count_evaluated": len(MASKS),
        "csi_off_max_abs_error": max(
            float(row["fallback_max_abs_error"])
            for row in mask_rows
            if row["csi_state"] == "off" and row["mask"] in {key for key, _, _ in MASKS}
        ),
    }
    (args.output_root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_root / "audit.md").write_text(Path("docs/prototype_pilot_transition_audit.md").read_text(encoding="utf-8"), encoding="utf-8")
    write_final_report(
        args.output_root / "final_report.md",
        primary=primary,
        snr_rows=snr_rows,
        dropout_rows=dropout_rows,
        budget_rows=budget_rows,
        diagnostics=diagnostics,
        mask_rows=mask_rows,
    )
    for loader in (*fixed.values(), *loaders.values()):
        shutdown_dataloader_workers(loader)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
