from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kd_sensing.engine.evaluation_pass_runtime import metadata_rows_from_batch, prepare_evaluation_batch
from kd_sensing.engine.runtime import prepare_task_labels, run_model_step
from kd_sensing.evaluation.metrics import beam_classification_circular_summary
from kd_sensing.losses.pcpf_temporal_risk import topology_risk_target
from kd_sensing.models.pcpf_temporal_risk import analytic_fusion_weights
from kd_sensing.utils.missing_patterns import canonical_missing_pattern_name, make_fixed_missing_mask


def collect_pcpf_observations(
    model: Any,
    dataloader: Any,
    cfg: dict[str, Any],
    *,
    device: str | torch.device,
    patterns: Mapping[str, Sequence[int]],
    max_batches: int | None = None,
    direct_router_model: Any | None = None,
) -> dict[str, Any]:
    """Collect validation diagnostics without reading an outer-test loader."""
    device = torch.device(device)
    modalities = tuple(str(value) for value in getattr(model, "modalities", ()))
    if len(modalities) != 4:
        raise ValueError("PCPF diagnostics require four canonical modalities.")
    if direct_router_model is not None:
        if getattr(direct_router_model, "direct_router", None) is None:
            raise ValueError("The replacement control model does not contain a direct Router.")
        if model._expert_fingerprint() != direct_router_model._expert_fingerprint():
            raise ValueError("Direct Router and PCPF checkpoints do not share the Stage 1 expert fingerprint.")
        direct_router_model.to(device).eval()

    tensor_chunks: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "labels",
            "raw_risk",
            "target_risk",
            "available",
            "fusion_weights",
            "unimodal_probabilities",
            "fused_probability",
            "risk_components",
        )
    }
    replacement_probability: dict[str, list[torch.Tensor]] = {
        "uniform": [],
        "static_prior": [],
        "pcpf_analytic": [],
    }
    replacement_weights: dict[str, list[torch.Tensor]] = {key: [] for key in replacement_probability}
    if direct_router_model is not None:
        replacement_probability["direct_router_control"] = []
        replacement_weights["direct_router_control"] = []
    strings = {key: [] for key in ("weather", "domain", "pattern", "mask_group")}
    model.to(device).eval()
    model_cfg = cfg["model"]["primary"]
    task = cfg.get("experiment", {}).get("task", "fusion")
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = prepare_evaluation_batch(raw_batch)
            labels = prepare_task_labels(batch, num_pred=int(model_cfg["num_pred"]), device=device)[:, 0]
            metadata = metadata_rows_from_batch(batch.get("metadata"))
            if len(metadata) != int(labels.shape[0]):
                metadata = [{} for _ in range(int(labels.shape[0]))]
            for pattern_name, raw_pattern in patterns.items():
                pattern = [int(value) for value in raw_pattern]
                forced = make_fixed_missing_mask(int(labels.shape[0]), pattern, device=device)
                step = run_model_step(
                    model,
                    task,
                    batch,
                    seq_length=int(model_cfg["seq_length"]),
                    num_pred=int(model_cfg["num_pred"]),
                    device=device,
                    extra_model_kwargs={} if all(pattern) else {"missing_mask": forced},
                )
                diagnostics = step.model_output.diagnostics
                available = _tensor(diagnostics, "available_modalities").bool()
                probabilities = _tensor(diagnostics, "unimodal_probabilities").float()
                raw_risk = _tensor(diagnostics, "raw_risk").float()
                calibrated = _tensor(diagnostics, "calibrated_unimodal_probabilities").float()
                capability = _tensor(diagnostics, "static_capability").float()
                tau = _tensor(diagnostics, "fusion_tau").float()
                target_risk = topology_risk_target(
                    probabilities,
                    labels,
                    available,
                    topology_id=str(getattr(model, "prototype_topology_id", "cyclic_index_v1")),
                    topology_permutation=getattr(model, "prototype_topology_permutation", None),
                )
                uniform = analytic_fusion_weights(
                    risk=torch.zeros_like(raw_risk),
                    available=available,
                    static_capability=torch.ones_like(capability),
                    tau=1.0,
                )
                static = analytic_fusion_weights(
                    risk=torch.zeros_like(raw_risk),
                    available=available,
                    static_capability=capability,
                    tau=1.0,
                )
                analytic = analytic_fusion_weights(
                    risk=raw_risk,
                    available=available,
                    static_capability=capability,
                    tau=tau,
                )
                weights_by_name = {
                    "uniform": uniform,
                    "static_prior": static,
                    "pcpf_analytic": analytic,
                }
                if direct_router_model is not None:
                    weights_by_name["direct_router_control"] = _direct_router_weights(
                        direct_router_model,
                        diagnostics,
                        available,
                    )
                for name, weights in weights_by_name.items():
                    probability = (weights.unsqueeze(-1) * calibrated).sum(dim=1)
                    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    replacement_probability[name].append(probability.cpu())
                    replacement_weights[name].append(weights.cpu())

                values = {
                    "labels": labels,
                    "raw_risk": raw_risk,
                    "target_risk": target_risk,
                    "available": available,
                    "fusion_weights": _tensor(diagnostics, "fusion_weights"),
                    "unimodal_probabilities": probabilities,
                    "fused_probability": _tensor(diagnostics, "fused_probability"),
                    "risk_components": _tensor(diagnostics, "risk_components"),
                }
                for key, value in values.items():
                    tensor_chunks[key].append(value.detach().cpu())
                for row in metadata:
                    weather = str(row.get("condition") or "unknown")
                    scenario = str(row.get("scenario") or row.get("sensor_scenario") or "unknown")
                    strings["weather"].append(weather)
                    strings["domain"].append(f"{weather}/{scenario}")
                    strings["pattern"].append(str(pattern_name))
                    strings["mask_group"].append(_mask_group(str(pattern_name), pattern))

    if not tensor_chunks["labels"]:
        raise ValueError("PCPF evaluation observed zero batches.")
    return {
        **{key: torch.cat(chunks, dim=0) for key, chunks in tensor_chunks.items()},
        **strings,
        "modalities": list(modalities),
        "replacement_probability": {key: torch.cat(chunks, dim=0) for key, chunks in replacement_probability.items()},
        "replacement_weights": {key: torch.cat(chunks, dim=0) for key, chunks in replacement_weights.items()},
        "modality_temperatures": model.temperatures.detach().float().cpu(),
        "static_capability": model._static_capability().detach().float().cpu(),
        "fusion_tau": float(model.tau.detach().float().cpu().item()),
        "expert_fingerprint": model._expert_fingerprint(),
        "training_stage": str(model.training_stage),
        "bounded_evaluation": max_batches is not None,
    }


def fit_train_confidence_p90(records: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.as_tensor(records["unimodal_probabilities"], dtype=torch.float32)
    available = torch.as_tensor(records["available"], dtype=torch.bool)
    confidence = probabilities.amax(dim=-1)
    quantiles: list[torch.Tensor] = []
    counts: list[int] = []
    for index in range(confidence.shape[1]):
        values = confidence[:, index][available[:, index]]
        if values.numel() == 0:
            raise ValueError(f"Train confidence has no available values for modality index {index}.")
        quantiles.append(torch.quantile(values, 0.9))
        counts.append(int(values.numel()))
    return torch.stack(quantiles), torch.tensor(counts, dtype=torch.long)


def build_stage2_gate_report(
    records: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    train_confidence_p90: torch.Tensor,
    train_confidence_count: torch.Tensor,
    stage2_checkpoint_sha256: str,
    bounded_evaluation: bool,
) -> dict[str, Any]:
    modalities = list(records["modalities"])
    available = torch.as_tensor(records["available"], dtype=torch.bool)
    predicted = torch.as_tensor(records["raw_risk"], dtype=torch.float32)
    target = torch.as_tensor(records["target_risk"], dtype=torch.float32)
    overall = _correlations(predicted[available], target[available])
    per_modality = {
        name: _correlations(predicted[:, index][available[:, index]], target[:, index][available[:, index]])
        for index, name in enumerate(modalities)
    }
    weather = _group_correlations(records, "weather")
    domains = _group_correlations(records, "domain")
    mask_groups = _group_correlations(records, "mask_group")
    risk_std = _std_or_none(predicted[available])
    lower_upper = _lower_upper_gap(predicted[available], target[available])
    failures: list[str] = []
    overall_min = float(gate["overall_spearman_min"])
    if overall["spearman"] is None or float(overall["spearman"]) <= overall_min:
        failures.append(f"overall_spearman_not_above_{overall_min:g}")
    positive_modalities = sum(value["spearman"] is not None and float(value["spearman"]) > 0.0 for value in per_modality.values())
    if positive_modalities < int(gate["minimum_positive_modalities"]):
        failures.append("insufficient_positive_modality_spearman")
    if bool(gate["require_each_weather_positive"]) and any(
        value["spearman"] is None or float(value["spearman"]) <= 0.0 for value in weather.values()
    ):
        failures.append("weather_spearman_not_positive")
    if set(weather) != {"sunny", "rainy", "foggy"}:
        failures.append("weather_coverage_not_exact")
    if len(domains) != 15:
        failures.append("domain_coverage_not_15")
    if not {"full", "drop1", "drop2", "single"}.issubset(mask_groups):
        failures.append("mask_group_coverage_incomplete")
    gap_min = float(gate["upper_lower_gap_min"])
    if lower_upper["empirical_target_gap"] is None or float(lower_upper["empirical_target_gap"]) <= gap_min:
        failures.append(f"upper_lower_target_gap_not_above_{gap_min:g}")
    minimum_std = float(gate["minimum_risk_std"])
    if risk_std is None or risk_std < minimum_std:
        failures.append(f"risk_std_below_{minimum_std:g}")
    if bounded_evaluation:
        failures.append("bounded_evaluation_not_gate_eligible")
    confidence = _confident_wrong(records, train_confidence_p90)
    report = {
        "schema_version": 1,
        "report_type": "pcpf_stage2_risk_observability_gate",
        "source_training_stage": "stage2_risk",
        "source_split": "inner_validation",
        "train_confidence_source_split": "inner_train",
        "stage2_checkpoint_sha256": str(stage2_checkpoint_sha256),
        "expert_fingerprint": str(records["expert_fingerprint"]),
        "bounded_evaluation": bool(bounded_evaluation),
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "thresholds": dict(gate),
        "train_confidence_p90": {name: float(train_confidence_p90[index]) for index, name in enumerate(modalities)},
        "train_confidence_count": {name: int(train_confidence_count[index]) for index, name in enumerate(modalities)},
        "overall": {**overall, "risk_std": risk_std, **lower_upper},
        "modalities": per_modality,
        "weather": weather,
        "domains": domains,
        "mask_groups": mask_groups,
        "risk_deciles": _risk_deciles(predicted[available], target[available]),
        "confident_but_wrong": confidence,
        "positive_modality_count": int(positive_modalities),
        "failure_reasons": failures,
        "stage2_gate_passed": not failures,
    }
    return report


def summarize_pcpf_matrix(
    records: Mapping[str, Any],
    *,
    train_confidence_p90: torch.Tensor,
) -> dict[str, Any]:
    count = int(torch.as_tensor(records["labels"]).numel())
    all_rows = torch.ones(count, dtype=torch.bool)
    return {
        "schema_version": 1,
        "report_type": "pcpf_15_mask_diagnostics",
        "source_split": "inner_validation",
        "training_stage": records["training_stage"],
        "expert_fingerprint": records["expert_fingerprint"],
        "bounded_evaluation": bool(records["bounded_evaluation"]),
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "sample_pattern_count": count,
        "modality_temperatures": {name: float(records["modality_temperatures"][index]) for index, name in enumerate(records["modalities"])},
        "static_capability": {name: float(records["static_capability"][index]) for index, name in enumerate(records["modalities"])},
        "fusion_tau": float(records["fusion_tau"]),
        "overall": _matrix_group(records, all_rows, train_confidence_p90),
        "patterns": _group_matrix(records, "pattern", train_confidence_p90),
        "weather": _group_matrix(records, "weather", train_confidence_p90),
        "domains": _group_matrix(records, "domain", train_confidence_p90),
        "pattern_weather": _joint_group_matrix(records, "pattern", "weather", train_confidence_p90),
        "direct_router_status": ("evaluated" if "direct_router_control" in records["replacement_probability"] else "not_supplied"),
    }


def write_pcpf_report(report: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _matrix_group(
    records: Mapping[str, Any],
    rows: torch.Tensor,
    train_confidence_p90: torch.Tensor,
) -> dict[str, Any]:
    labels = torch.as_tensor(records["labels"], dtype=torch.long)[rows]
    probability = torch.as_tensor(records["fused_probability"], dtype=torch.float32)[rows]
    weights = torch.as_tensor(records["fusion_weights"], dtype=torch.float32)[rows]
    available = torch.as_tensor(records["available"], dtype=torch.bool)[rows]
    risk = torch.as_tensor(records["raw_risk"], dtype=torch.float32)[rows]
    static_weights = torch.as_tensor(records["replacement_weights"]["static_prior"], dtype=torch.float32)[rows]
    replacement = {
        name: _classification_metrics(torch.as_tensor(values)[rows], labels) for name, values in records["replacement_probability"].items()
    }
    return {
        **_classification_metrics(probability, labels),
        "weight_diagnostics": _weight_diagnostics(
            weights,
            risk,
            available,
            static_weights,
            list(records["modalities"]),
        ),
        "replacement_metrics": replacement,
        "confident_but_wrong": _confident_wrong(records, train_confidence_p90, rows=rows),
    }


def _classification_metrics(probability: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    if labels.numel() == 0:
        return {"count": 0}
    probability = probability.float().clamp_min(1e-12)
    logits = probability.log()
    summary = beam_classification_circular_summary(logits, labels, num_beams=probability.shape[-1])
    one_hot = F.one_hot(labels, num_classes=probability.shape[-1]).float()
    return {
        "count": int(labels.numel()),
        "top1": float(summary["top1"]),
        "top3": float(summary["top3"]),
        "top5": float(summary["top5"]),
        "within_3": float(summary["within_3"]),
        "circular_mae": float(summary["mean_error"]),
        "nll": float(F.nll_loss(logits, labels).item()),
        "brier": float((probability - one_hot).square().sum(dim=-1).mean().item()),
        "ece": _ece(probability, labels),
        "reliability_diagram": _reliability_diagram(probability, labels),
    }


def _weight_diagnostics(
    weights: torch.Tensor,
    risk: torch.Tensor,
    available: torch.Tensor,
    static_weights: torch.Tensor,
    modalities: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(modalities):
        values = weights[:, index][available[:, index]]
        result[name] = {
            "count": int(values.numel()),
            "mean": _mean_or_none(values),
            "sample_std": _std_or_none(values),
            "p10": _quantile_or_none(values, 0.1),
            "p50": _quantile_or_none(values, 0.5),
            "p90": _quantile_or_none(values, 0.9),
        }
    missing = weights[~available]
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=-1)
    valid = available & torch.isfinite(risk) & torch.isfinite(weights)
    return {
        "modalities": result,
        "missing_weight_max": float(missing.abs().max().item()) if missing.numel() else 0.0,
        "mean_effective_modalities": float(entropy.exp().mean().item()),
        "mean_absolute_dynamic_deviation_from_static": float((weights - static_weights).abs().mean().item()),
        "risk_weight_spearman": _correlations(risk[valid], weights[valid])["spearman"],
        "negative_risk_weight_spearman": _correlations(-risk[valid], weights[valid])["spearman"],
        "risk_weight_pair_order_agreement": _pair_order_agreement(risk, weights, available),
        "weight_row_sum_max_error": float((weights.sum(dim=-1) - 1.0).abs().max().item()),
    }


def _confident_wrong(
    records: Mapping[str, Any],
    threshold: torch.Tensor,
    *,
    rows: torch.Tensor | None = None,
) -> dict[str, Any]:
    probabilities = torch.as_tensor(records["unimodal_probabilities"], dtype=torch.float32)
    labels = torch.as_tensor(records["labels"], dtype=torch.long)
    available = torch.as_tensor(records["available"], dtype=torch.bool)
    risk = torch.as_tensor(records["raw_risk"], dtype=torch.float32)
    weights = torch.as_tensor(records["fusion_weights"], dtype=torch.float32)
    components = torch.as_tensor(records["risk_components"], dtype=torch.float32)
    fused = torch.as_tensor(records["fused_probability"], dtype=torch.float32)
    if rows is not None:
        probabilities, labels, available, risk, weights, components, fused = (
            value[rows] for value in (probabilities, labels, available, risk, weights, components, fused)
        )
    confidence, prediction = probabilities.max(dim=-1)
    result: dict[str, Any] = {}
    component_names = ("u_var", "u_proto", "u_temp", "u_conflict")
    for index, name in enumerate(records["modalities"]):
        wrong = available[:, index] & confidence[:, index].ge(float(threshold[index])) & prediction[:, index].ne(labels)
        correct = available[:, index] & prediction[:, index].eq(labels)
        result[name] = {
            "threshold": float(threshold[index]),
            "count": int(wrong.sum().item()),
            "mean_risk": _mean_or_none(risk[:, index][wrong]),
            "correct_mean_risk": _mean_or_none(risk[:, index][correct]),
            "pcpf_mean_weight": _mean_or_none(weights[:, index][wrong]),
            "static_prior_mean_weight": _mean_or_none(
                torch.as_tensor(records["replacement_weights"]["static_prior"], dtype=torch.float32)[rows][:, index][wrong]
                if rows is not None
                else torch.as_tensor(records["replacement_weights"]["static_prior"], dtype=torch.float32)[:, index][wrong]
            ),
            "final_fusion_correction_rate": _mean_or_none(fused.argmax(dim=-1)[wrong].eq(labels[wrong]).float()),
            "components": {
                component: _mean_or_none(components[:, index, component_index][wrong])
                for component_index, component in enumerate(component_names)
            },
        }
        if "direct_router_control" in records["replacement_weights"]:
            router = torch.as_tensor(records["replacement_weights"]["direct_router_control"], dtype=torch.float32)
            if rows is not None:
                router = router[rows]
            result[name]["old_router_mean_weight"] = _mean_or_none(router[:, index][wrong])
    return result


def _group_correlations(records: Mapping[str, Any], key: str) -> dict[str, Any]:
    values = list(records[key])
    predicted = torch.as_tensor(records["raw_risk"], dtype=torch.float32)
    target = torch.as_tensor(records["target_risk"], dtype=torch.float32)
    available = torch.as_tensor(records["available"], dtype=torch.bool)
    result = {}
    for group in sorted(set(values)):
        rows = torch.tensor([value == group for value in values], dtype=torch.bool)
        valid = available[rows]
        result[group] = _correlations(predicted[rows][valid], target[rows][valid])
    return result


def _group_matrix(
    records: Mapping[str, Any],
    key: str,
    threshold: torch.Tensor,
) -> dict[str, Any]:
    values = list(records[key])
    return {
        group: _matrix_group(
            records,
            torch.tensor([value == group for value in values], dtype=torch.bool),
            threshold,
        )
        for group in sorted(set(values))
    }


def _joint_group_matrix(
    records: Mapping[str, Any],
    left: str,
    right: str,
    threshold: torch.Tensor,
) -> dict[str, Any]:
    left_values = list(records[left])
    right_values = list(records[right])
    pairs = sorted(set(zip(left_values, right_values)))
    return {
        f"{first}|{second}": _matrix_group(
            records,
            torch.tensor(
                [left_value == first and right_value == second for left_value, right_value in zip(left_values, right_values)],
                dtype=torch.bool,
            ),
            threshold,
        )
        for first, second in pairs
    }


def _correlations(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int | None]:
    finite = torch.isfinite(left) & torch.isfinite(right)
    left, right = left[finite].float(), right[finite].float()
    return {
        "count": int(left.numel()),
        "pearson": _pearson(left, right),
        "spearman": _pearson(_rank(left), _rank(right)) if left.numel() else None,
        "predicted_std": _std_or_none(left),
        "target_std": _std_or_none(right),
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() < 2:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if not torch.isfinite(denominator) or denominator <= 1e-12:
        return None
    return float((left * right).sum().div(denominator).item())


def _rank(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.float()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values, dtype=torch.float32)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and bool(sorted_values[end].eq(sorted_values[start]).item()):
            end += 1
        ranks[order[start:end]] = 0.5 * float(start + end - 1)
        start = end
    return ranks


def _risk_deciles(predicted: torch.Tensor, target: torch.Tensor) -> list[dict[str, Any]]:
    order = torch.argsort(predicted, stable=True)
    chunks = torch.tensor_split(order, 10)
    return [
        {
            "decile": index + 1,
            "count": int(chunk.numel()),
            "mean_predicted_risk": _mean_or_none(predicted[chunk]),
            "mean_empirical_topology_risk": _mean_or_none(target[chunk]),
        }
        for index, chunk in enumerate(chunks)
    ]


def _lower_upper_gap(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float | None]:
    if predicted.numel() < 5:
        return {"lower_20_target_mean": None, "upper_20_target_mean": None, "empirical_target_gap": None}
    lower_cut = torch.quantile(predicted, 0.2)
    upper_cut = torch.quantile(predicted, 0.8)
    lower = target[predicted <= lower_cut]
    upper = target[predicted >= upper_cut]
    lower_mean, upper_mean = _mean_or_none(lower), _mean_or_none(upper)
    return {
        "lower_20_target_mean": lower_mean,
        "upper_20_target_mean": upper_mean,
        "empirical_target_gap": None if lower_mean is None or upper_mean is None else upper_mean - lower_mean,
    }


def _reliability_diagram(probability: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> list[dict[str, Any]]:
    confidence, prediction = probability.max(dim=-1)
    correct = prediction.eq(labels).float()
    result = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = confidence.ge(lower) & (confidence.le(upper) if index == bins - 1 else confidence.lt(upper))
        result.append(
            {
                "lower": lower,
                "upper": upper,
                "count": int(selected.sum().item()),
                "mean_confidence": _mean_or_none(confidence[selected]),
                "accuracy": _mean_or_none(correct[selected]),
            }
        )
    return result


def _ece(probability: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    total = max(int(labels.numel()), 1)
    return float(
        sum(
            row["count"] / total * abs(float(row["mean_confidence"]) - float(row["accuracy"]))
            for row in _reliability_diagram(probability, labels, bins)
            if row["count"] and row["mean_confidence"] is not None and row["accuracy"] is not None
        )
    )


def _pair_order_agreement(risk: torch.Tensor, weights: torch.Tensor, available: torch.Tensor) -> float | None:
    agreements: list[torch.Tensor] = []
    for left in range(risk.shape[1]):
        for right in range(left + 1, risk.shape[1]):
            valid = available[:, left] & available[:, right]
            if bool(valid.any().item()):
                risk_difference = risk[valid, left] - risk[valid, right]
                weight_difference = weights[valid, left] - weights[valid, right]
                informative = risk_difference.ne(0) & weight_difference.ne(0)
                agreements.append((risk_difference[informative] * weight_difference[informative]).lt(0).float())
    values = [value for value in agreements if value.numel()]
    return _mean_or_none(torch.cat(values)) if values else None


def _direct_router_weights(model: Any, diagnostics: Mapping[str, Any], available: torch.Tensor) -> torch.Tensor:
    probabilities = _tensor(diagnostics, "unimodal_probabilities").float()
    logits = _tensor(diagnostics, "unimodal_logits").float()
    components = _tensor(diagnostics, "risk_components").float()
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1, keepdim=True)
    top2 = probabilities.topk(2, dim=-1).values
    features = torch.cat(
        [
            components[..., 1:2],
            entropy,
            top2[..., :1] - top2[..., 1:2],
            top2[..., :1],
            logits.norm(dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    scores = model.direct_router(features).squeeze(-1).float().masked_fill(~available, -torch.inf)
    return torch.softmax(scores, dim=-1) * available.float()


def _mask_group(name: str, pattern: Sequence[int]) -> str:
    canonical = canonical_missing_pattern_name(name)
    available = sum(int(value) for value in pattern)
    if canonical == "full" or available == 4:
        return "full"
    if available == 3:
        return "drop1"
    if available == 2:
        return "drop2"
    if available == 1:
        return "single"
    return "other"


def _tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"PCPF diagnostics are missing tensor {key!r}.")
    return value


def _mean_or_none(values: torch.Tensor) -> float | None:
    values = values[torch.isfinite(values)]
    return float(values.mean().item()) if values.numel() else None


def _std_or_none(values: torch.Tensor) -> float | None:
    values = values[torch.isfinite(values)]
    return float(values.std(unbiased=False).item()) if values.numel() else None


def _quantile_or_none(values: torch.Tensor, quantile: float) -> float | None:
    values = values[torch.isfinite(values)]
    return float(torch.quantile(values, quantile).item()) if values.numel() else None


__all__ = [
    "build_stage2_gate_report",
    "collect_pcpf_observations",
    "fit_train_confidence_p90",
    "summarize_pcpf_matrix",
    "write_pcpf_report",
]
