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
    control_models: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect validation diagnostics without reading an outer-test loader."""
    device = torch.device(device)
    modalities = tuple(str(value) for value in getattr(model, "modalities", ()))
    if len(modalities) != 4:
        raise ValueError("PCPF diagnostics require four canonical modalities.")
    controls = dict(control_models or {})
    allowed_controls = {"uniform", "static_prior", "direct_router_control", "cuaf_local_adaptation"}
    for mode, control in controls.items():
        if mode not in allowed_controls or getattr(control, "fusion_mode", None) != mode:
            raise ValueError(f"Invalid PCPF replacement control {mode!r}.")
        if model._expert_fingerprint() != control._expert_fingerprint():
            raise ValueError(f"{mode} and PCPF checkpoints do not share the Stage 1 expert fingerprint.")
        control.to(device).eval()

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
        name: [] for name in ("uniform", "static_prior", "pcpf_analytic", *sorted(controls))
    }
    replacement_weights: dict[str, list[torch.Tensor]] = {key: [] for key in replacement_probability}
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
                weights_by_name = {
                    "uniform": uniform,
                    "static_prior": static,
                    "pcpf_analytic": _tensor(diagnostics, "fusion_weights").float(),
                }
                calibrated_by_name = {name: calibrated for name in weights_by_name}
                for name, control in controls.items():
                    control_weights, control_calibrated = _control_fusion_from_cached_evidence(
                        control,
                        diagnostics,
                        available,
                    )
                    weights_by_name[name] = control_weights
                    calibrated_by_name[name] = control_calibrated
                for name, weights in weights_by_name.items():
                    probability = (weights.unsqueeze(-1) * calibrated_by_name[name]).sum(dim=1)
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
        "trained_controls": sorted(controls),
        "replacement_parameters": {
            name: _fusion_parameter_summary(controls.get(name, model), source="control_checkpoint" if name in controls else "main_checkpoint")
            for name in replacement_probability
        },
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
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    required_provenance = {"checkpoint", "data_protocol", "normalization"}
    missing_provenance = sorted(required_provenance.difference(provenance))
    if missing_provenance:
        raise ValueError(f"PCPF matrix provenance is missing: {', '.join(missing_provenance)}.")
    if any(not isinstance(provenance[key], Mapping) for key in required_provenance):
        raise ValueError("PCPF matrix checkpoint, data_protocol, and normalization provenance must be mappings.")
    count = int(torch.as_tensor(records["labels"]).numel())
    all_rows = torch.ones(count, dtype=torch.bool)
    pattern_reports = _group_matrix(records, "pattern", train_confidence_p90)
    return {
        "schema_version": 1,
        "report_type": "pcpf_15_mask_diagnostics",
        "source_split": "inner_validation",
        "training_stage": records["training_stage"],
        "expert_fingerprint": records["expert_fingerprint"],
        "bounded_evaluation": bool(records["bounded_evaluation"]),
        "claim_ineligible": True,
        "outer_test_accessed": False,
        "provenance": dict(provenance),
        "sample_pattern_count": count,
        "modality_temperatures": {name: float(records["modality_temperatures"][index]) for index, name in enumerate(records["modalities"])},
        "static_capability": {name: float(records["static_capability"][index]) for index, name in enumerate(records["modalities"])},
        "fusion_tau": float(records["fusion_tau"]),
        "trained_controls": list(records.get("trained_controls", ())),
        "replacement_parameters": dict(records.get("replacement_parameters", {})),
        "overall": _matrix_group(records, all_rows, train_confidence_p90),
        "patterns": pattern_reports,
        "pattern_aggregates": _pattern_aggregates(pattern_reports),
        "weather": _group_matrix(records, "weather", train_confidence_p90),
        "domains": _group_matrix(records, "domain", train_confidence_p90),
        "pattern_weather": _joint_group_matrix(records, "pattern", "weather", train_confidence_p90),
        "direct_router_status": ("evaluated" if "direct_router_control" in records["replacement_probability"] else "not_supplied"),
        "cuaf_local_adaptation_status": (
            "evaluated" if "cuaf_local_adaptation" in records["replacement_probability"] else "not_supplied"
        ),
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


def _pattern_aggregates(patterns: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    single = [name for name in patterns if canonical_missing_pattern_name(name).endswith("_only")]
    all14 = [name for name in patterns if canonical_missing_pattern_name(name) != "full"]
    if len(single) != 4 or len(all14) != 14:
        raise ValueError("PCPF matrix aggregation requires four Single masks and exactly 14 non-Full masks.")
    higher_is_better = {"top1", "top3", "top5", "within_3"}
    metric_names = (*sorted(higher_is_better), "circular_mae", "nll", "brier", "ece")
    result: dict[str, Any] = {}
    for group_name, pattern_names in (("single", single), ("all14", all14)):
        methods = patterns[pattern_names[0]]["replacement_metrics"]
        result[group_name] = {}
        for method in methods:
            result[group_name][method] = {}
            for metric in metric_names:
                values = [float(patterns[name]["replacement_metrics"][method][metric]) for name in pattern_names]
                result[group_name][method][metric] = {
                    "macro": sum(values) / len(values),
                    "worst": min(values) if metric in higher_is_better else max(values),
                }
    return result


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


def _control_fusion_from_cached_evidence(
    model: Any,
    diagnostics: Mapping[str, Any],
    available: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, calibrated, _, effective_mode = model._fuse(
        _tensor(diagnostics, "unimodal_logits").float(),
        _tensor(diagnostics, "unimodal_probabilities").float(),
        _tensor(diagnostics, "raw_risk").float(),
        _tensor(diagnostics, "risk_components").float(),
        available,
    )
    if effective_mode != model.fusion_mode:
        raise ValueError(f"Control fusion mode changed from {model.fusion_mode!r} to {effective_mode!r}.")
    return weights, calibrated


def _fusion_parameter_summary(model: Any, *, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "fusion_mode": str(model.fusion_mode),
        "temperatures": model.temperatures.detach().float().cpu().tolist(),
        "tau": float(model.tau.detach().float().cpu().item()),
        "static_capability": model._static_capability().detach().float().cpu().tolist(),
    }


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
