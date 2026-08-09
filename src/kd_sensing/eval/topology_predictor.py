from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from kd_sensing.engine.evaluation_pass_runtime import metadata_rows_from_batch, prepare_evaluation_batch
from kd_sensing.engine.runtime import prepare_task_labels, run_model_step
from kd_sensing.evaluation.metrics import beam_classification_circular_summary
from kd_sensing.modalities import MODALITY_ORDER
from kd_sensing.utils.missing_patterns import make_fixed_missing_mask, resolve_missing_patterns


def resolve_topology_missing_patterns(
    configured: str | Sequence[str] | None,
    modalities: Sequence[str] = MODALITY_ORDER,
) -> dict[str, list[int]]:
    names = tuple(str(value) for value in modalities)
    if names != tuple(MODALITY_ORDER):
        raise ValueError("Topology evaluation requires canonical image/radar/gps/lidar modalities.")
    patterns = resolve_missing_patterns(configured, list(names))
    masks = {tuple(int(value) for value in mask) for mask in patterns.values()}
    expected = {
        tuple(int(bool(bits & (1 << index))) for index in range(len(names)))
        for bits in range(1, 1 << len(names))
    }
    if len(patterns) != 15 or masks != expected:
        raise ValueError("Topology evaluation requires exactly the 15 non-empty four-modal masks.")
    return patterns


def collect_topology_observations(
    model: Any,
    dataloader: Any,
    cfg: Mapping[str, Any],
    *,
    device: str | torch.device,
    patterns: Mapping[str, Sequence[int]],
    max_batches: int | None = None,
) -> dict[str, Any]:
    modalities = tuple(str(value) for value in getattr(model, "modalities", ()))
    if modalities != tuple(MODALITY_ORDER):
        raise ValueError("Topology observation collection requires the native four-modal model.")
    chunks = {key: [] for key in ("labels", "available", "unimodal_logits", "unimodal_probabilities", "fused_probability", "final_prediction")}
    strings = {key: [] for key in ("weather", "domain", "pattern", "sample_id", "protocol_sample_id", "group_id")}
    model_cfg = cfg["model"]["primary"]
    task = str(cfg.get("experiment", {}).get("task", "fusion"))
    target_device = torch.device(device)
    model.to(target_device).eval()
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = prepare_evaluation_batch(raw_batch)
            labels = prepare_task_labels(
                batch,
                num_pred=int(model_cfg["num_pred"]),
                device=target_device,
            )[:, 0]
            metadata = metadata_rows_from_batch(batch.get("metadata"))
            if len(metadata) != int(labels.shape[0]):
                raise ValueError("Topology evaluation requires metadata for every validation sample.")
            for pattern_name, raw_pattern in patterns.items():
                mask = [int(value) for value in raw_pattern]
                forced = make_fixed_missing_mask(int(labels.shape[0]), mask, device=target_device)
                step = run_model_step(
                    model,
                    task,
                    batch,
                    seq_length=int(model_cfg["seq_length"]),
                    num_pred=int(model_cfg["num_pred"]),
                    device=target_device,
                    extra_model_kwargs={} if all(mask) else {"missing_mask": forced},
                )
                diagnostics = step.model_output.diagnostics
                probability = _tensor(diagnostics, "fused_probability").float()
                values = {
                    "labels": labels,
                    "available": _tensor(diagnostics, "available_modalities").bool(),
                    "unimodal_logits": _tensor(diagnostics, "unimodal_logits"),
                    "unimodal_probabilities": _tensor(diagnostics, "unimodal_probabilities").float(),
                    "fused_probability": probability,
                    "final_prediction": probability.argmax(dim=-1),
                }
                for key, value in values.items():
                    chunks[key].append(value.detach().cpu())
                for row in metadata:
                    weather = str(row.get("condition") or "unknown")
                    scenario = str(row.get("scenario") or row.get("sensor_scenario") or "unknown")
                    sample_id = str(row.get("stable_sample_id") or row.get("source_sample_id") or row.get("sample_id") or "")
                    protocol_id = str(row.get("source_sample_id") or row.get("sample_id") or "")
                    if not sample_id or not protocol_id:
                        raise ValueError("Topology evaluation metadata is missing stable/protocol sample identity.")
                    strings["weather"].append(weather)
                    strings["domain"].append(f"{weather}/{scenario}")
                    strings["pattern"].append(str(pattern_name))
                    strings["sample_id"].append(sample_id)
                    strings["protocol_sample_id"].append(protocol_id)
                    strings["group_id"].append(str(row.get("trajectory_group_id") or row.get("contiguous_segment_id") or sample_id))
    if not chunks["labels"]:
        raise ValueError("Topology evaluation observed zero validation batches.")
    return {
        **{key: torch.cat(values, dim=0) for key, values in chunks.items()},
        **strings,
        "modalities": list(modalities),
        "bounded_evaluation": max_batches is not None,
        "model_type": "four_modal_topology_predictor",
    }


def summarize_topology_matrix(
    records: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    dba_delta: float = 5.0,
) -> dict[str, Any]:
    patterns = [str(value) for value in records["pattern"]]
    domains = [str(value) for value in records["domain"]]
    probability = torch.as_tensor(records["fused_probability"], dtype=torch.float32)
    labels = torch.as_tensor(records["labels"], dtype=torch.long)
    per_pattern = {
        name: beam_classification_circular_summary(
            probability[[index for index, value in enumerate(patterns) if value == name]],
            labels[[index for index, value in enumerate(patterns) if value == name]],
            num_beams=64,
            dba_delta=float(dba_delta),
            topk=(1, 3, 5),
            distance_mode="circular",
        )
        for name in dict.fromkeys(patterns)
    }
    return {
        "schema_version": 1,
        "report_type": "four_modal_topology_validation_matrix",
        "model_type": "four_modal_topology_predictor",
        "patterns": per_pattern,
        "pattern_count": len(per_pattern),
        "domains": sorted(set(domains)),
        "domain_count": len(set(domains)),
        "bounded_evaluation": bool(records.get("bounded_evaluation", False)),
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "experiment_seed": int(provenance.get("experiment_seed", -1)),
        "provenance": dict(provenance),
    }


def write_observation_cache(records: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(records), target)


def _tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Topology predictor output is missing tensor {key!r}.")
    return value


__all__ = [
    "collect_topology_observations",
    "resolve_topology_missing_patterns",
    "summarize_topology_matrix",
    "write_observation_cache",
]
