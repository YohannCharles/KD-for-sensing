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
from kd_sensing.utils.missing_patterns import make_fixed_missing_mask, resolve_missing_patterns


PCPF_SENSING_MODALITIES = ("image", "radar", "gps", "lidar")
PCPF_SPARSE_CSI_MODALITIES = (*PCPF_SENSING_MODALITIES, "csi")


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
    if modalities not in {PCPF_SENSING_MODALITIES, PCPF_SPARSE_CSI_MODALITIES}:
        raise ValueError("PCPF diagnostics require canonical sensing modalities with optional sparse CSI.")
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
            "unimodal_logits",
            "unimodal_predictions",
            "unimodal_confidence",
            "unimodal_correct",
            "unimodal_circular_errors",
            "prototype_distance",
            "final_prediction",
            "raw_risk",
            "target_risk",
            "available",
            "fusion_weights",
            "unimodal_probabilities",
            "calibrated_unimodal_probabilities",
            "fused_probability",
            "risk_components",
            "normalized_risk_components",
        )
    }
    if modalities == PCPF_SPARSE_CSI_MODALITIES:
        tensor_chunks.update(
            csi_log_rms=[],
            csi_valid_ratio=[],
            csi_quality_confidence=[],
            csi_snr_available=[],
        )
    replacement_probability: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("uniform", "static_prior", "pcpf_analytic", *sorted(controls))
    }
    replacement_weights: dict[str, list[torch.Tensor]] = {key: [] for key in replacement_probability}
    strings = {key: [] for key in ("weather", "domain", "pattern", "mask_group", "sample_id", "group_id")}
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
                    "unimodal_logits": _tensor(diagnostics, "unimodal_logits"),
                    "unimodal_predictions": probabilities.argmax(dim=-1),
                    "unimodal_confidence": probabilities.amax(dim=-1),
                    "unimodal_correct": probabilities.argmax(dim=-1).eq(labels.unsqueeze(1)) & available,
                    "unimodal_circular_errors": _circular_class_errors(probabilities.argmax(dim=-1), labels),
                    "prototype_distance": (
                        1.0
                        - _tensor(diagnostics, "unimodal_logits").float().amax(dim=-1)
                        * float(model.prototype_bank.temperature)
                    )
                    * available.to(torch.float32),
                    "final_prediction": _tensor(diagnostics, "fused_probability").argmax(dim=-1),
                    "raw_risk": raw_risk,
                    "target_risk": target_risk,
                    "available": available,
                    "fusion_weights": _tensor(diagnostics, "fusion_weights"),
                    "unimodal_probabilities": probabilities,
                    "calibrated_unimodal_probabilities": calibrated,
                    "fused_probability": _tensor(diagnostics, "fused_probability"),
                    "risk_components": _tensor(diagnostics, "risk_components"),
                    "normalized_risk_components": _tensor(diagnostics, "normalized_risk_components"),
                }
                if modalities == PCPF_SPARSE_CSI_MODALITIES:
                    values.update(
                        csi_log_rms=_tensor(diagnostics, "csi_log_rms"),
                        csi_valid_ratio=_tensor(diagnostics, "csi_valid_ratio"),
                        csi_quality_confidence=_tensor(diagnostics, "csi_quality_confidence"),
                        csi_snr_available=_tensor(diagnostics, "csi_snr_available"),
                    )
                for key, value in values.items():
                    tensor_chunks[key].append(value.detach().cpu())
                for row in metadata:
                    weather = str(row.get("condition") or "unknown")
                    scenario = str(row.get("scenario") or row.get("sensor_scenario") or "unknown")
                    strings["weather"].append(weather)
                    strings["domain"].append(f"{weather}/{scenario}")
                    strings["pattern"].append(str(pattern_name))
                    strings["mask_group"].append(_mask_group(str(pattern_name), pattern, len(modalities)))
                    sample_id = str(row.get("stable_sample_id") or row.get("source_sample_id") or row.get("sample_id") or "unknown")
                    group_id = str(row.get("trajectory_group_id") or row.get("contiguous_segment_id") or sample_id)
                    strings["sample_id"].append(sample_id)
                    strings["group_id"].append(group_id)

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
        "risk_coefficients": model.risk_coefficients.detach().float().cpu(),
        "risk_component_mean": model.risk_component_mean.detach().float().cpu(),
        "risk_component_std": model.risk_component_std.detach().float().cpu(),
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
    diagnostics_config: Mapping[str, Any] | None = None,
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
    result = {
        "schema_version": 1,
        "report_type": f"pcpf_{len(pattern_reports)}_mask_diagnostics",
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
        "pattern_aggregates": _pattern_aggregates(pattern_reports, records=records),
        "expert_input_diagnostics": _expert_input_diagnostics(records),
        "weather": _group_matrix(records, "weather", train_confidence_p90),
        "domains": _group_matrix(records, "domain", train_confidence_p90),
        "pattern_weather": _joint_group_matrix(records, "pattern", "weather", train_confidence_p90),
        "pattern_domain": _joint_group_matrix(records, "pattern", "domain", train_confidence_p90),
        "direct_router_status": ("evaluated" if "direct_router_control" in records["replacement_probability"] else "not_supplied"),
        "cuaf_local_adaptation_status": (
            "evaluated" if "cuaf_local_adaptation" in records["replacement_probability"] else "not_supplied"
        ),
    }
    config = dict(diagnostics_config or {})
    mechanism = build_pcpf_mechanism_diagnostics(
        records,
        train_confidence_p90=train_confidence_p90,
        bootstrap_config=config.get("bootstrap"),
    )
    result["mechanism_diagnostics"] = mechanism
    result["R0_R7"] = (
        _r0_r7_summary(result, historical_reference=config.get("historical_reference_summary"))
        if tuple(records["modalities"]) == PCPF_SPARSE_CSI_MODALITIES
        else {"status": "not_applicable_without_sparse_csi"}
    )
    return result


def build_pcpf_mechanism_diagnostics(
    records: Mapping[str, Any],
    *,
    train_confidence_p90: torch.Tensor,
    bootstrap_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if tuple(records["modalities"]) != PCPF_SPARSE_CSI_MODALITIES:
        return {"status": "not_applicable_without_sparse_csi"}
    bootstrap = _bootstrap_config(bootstrap_config)
    dynamicity = _dynamicity_tests(records, bootstrap)
    return {
        "status": "computed",
        "source_split": "inner_validation",
        "claim_ineligible": True,
        "group_key": "trajectory_group_id_or_contiguous_segment_id_or_stable_sample_id",
        "bootstrap": bootstrap | {"group_count": len(set(records["group_id"]))},
        "dynamicity_test": dynamicity,
        "risk_and_component_diagnostics": _risk_and_component_diagnostics(
            records,
            train_confidence_p90=train_confidence_p90,
        ),
        "weight_transfer": _paired_weight_transfer(records),
        "replacement_top1_deltas": _bootstrap_replacement_deltas(records, bootstrap),
    }


def write_pcpf_observation_cache(records: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(records), target)
    return target


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
    true_errors = _record_circular_errors(records)[rows]
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
            true_errors,
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
    true_errors: torch.Tensor,
    modalities: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    top_weight = weights.masked_fill(~available, -torch.inf).argmax(dim=1)
    for index, name in enumerate(modalities):
        values = weights[:, index][available[:, index]]
        static_values = static_weights[:, index][available[:, index]]
        result[name] = {
            "count": int(values.numel()),
            "mean": _mean_or_none(values),
            "sample_std": _std_or_none(values),
            "p10": _quantile_or_none(values, 0.1),
            "p50": _quantile_or_none(values, 0.5),
            "p90": _quantile_or_none(values, 0.9),
            "mean_absolute_deviation_from_static": _mean_or_none((values - static_values).abs()),
            "top_weight_fraction": float(top_weight.eq(index).float().mean().item()),
        }
    missing = weights[~available]
    entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=-1)
    valid = available & torch.isfinite(risk) & torch.isfinite(weights)
    summary = {
        "modalities": result,
        "missing_weight_max": float(missing.abs().max().item()) if missing.numel() else 0.0,
        "mean_effective_modalities": float(entropy.exp().mean().item()),
        "mean_absolute_dynamic_deviation_from_static": float((weights - static_weights).abs().mean().item()),
        "risk_weight_spearman": _correlations(risk[valid], weights[valid])["spearman"],
        "negative_risk_weight_spearman": _correlations(-risk[valid], weights[valid])["spearman"],
        "risk_weight_pair_order_agreement": _pair_order_agreement(risk, weights, available),
        "true_error_weight_pair_order_agreement": _pair_order_agreement(true_errors, weights, available),
        "weight_row_sum_max_error": float((weights.sum(dim=-1) - 1.0).abs().max().item()),
    }
    if "csi" in modalities:
        csi = weights[:, modalities.index("csi")]
        csi_available = available[:, modalities.index("csi")]
        summary["csi_weight_threshold_fraction"] = {
            str(threshold): _mean_or_none(csi[csi_available].gt(threshold).float())
            for threshold in (0.5, 0.7, 0.9)
        }
    return summary


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


def _pattern_aggregates(
    patterns: Mapping[str, Mapping[str, Any]],
    *,
    records: Mapping[str, Any],
) -> dict[str, Any]:
    names = list(records["modalities"])
    masks = {pattern: _pattern_mask_from_name(pattern, names) for pattern in patterns}
    single = [name for name, mask in masks.items() if sum(mask) == 1]
    non_full = [name for name, mask in masks.items() if sum(mask) != len(names)]
    expected_non_full = (1 << len(names)) - 2
    if len(single) not in {4, 5} or len(non_full) != expected_non_full:
        raise ValueError("PCPF matrix aggregation requires every non-empty availability subset.")
    higher_is_better = {"top1", "top3", "top5", "within_3"}
    metric_names = (*sorted(higher_is_better), "circular_mae", "nll", "brier", "ece")
    result: dict[str, Any] = {}
    groups: dict[str, list[str]] = {
        "single": single,
        f"all{len(non_full)}": non_full,
    }
    if names == list(PCPF_SPARSE_CSI_MODALITIES):
        groups.update(
            single5=single,
            overall31=sorted(masks),
            multi_modal=[name for name, mask in masks.items() if sum(mask) >= 2],
            csi_present_with_sensing=[
                name for name, mask in masks.items() if mask[-1] and any(mask[:-1])
            ],
            csi_absent_legacy15=[
                name for name, mask in masks.items() if not mask[-1] and any(mask[:-1])
            ],
        )
        for count in range(1, len(names) + 1):
            groups[f"cardinality_n{count}"] = [name for name, mask in masks.items() if sum(mask) == count]
    for group_name, pattern_names in groups.items():
        if not pattern_names:
            continue
        methods = patterns[pattern_names[0]]["replacement_metrics"]
        result[group_name] = {}
        for method in methods:
            result[group_name][method] = {}
            for metric in metric_names:
                values = [float(patterns[name]["replacement_metrics"][method][metric]) for name in pattern_names]
                result[group_name][method][metric] = {
                    "macro": sum(values) / len(values),
                    "worst": min(values) if metric in higher_is_better else max(values),
                    "pattern_count": len(values),
                }
    return result


def _pattern_mask_from_name(pattern: str, modalities: list[str]) -> tuple[bool, ...]:
    if pattern == "full":
        return (True,) * len(modalities)
    if pattern.endswith("_only"):
        only = pattern.removesuffix("_only")
        if only not in modalities:
            raise ValueError(f"Unknown PCPF single-modality pattern {pattern!r}.")
        return tuple(name == only for name in modalities)
    if pattern.startswith("missing_"):
        missing = set(pattern.removeprefix("missing_").split("_"))
        if not missing or not missing.issubset(modalities):
            raise ValueError(f"Unknown PCPF missing pattern {pattern!r}.")
        return tuple(name not in missing for name in modalities)
    raise ValueError(f"Unsupported PCPF pattern name {pattern!r}.")


def _r0_r7_summary(
    report: Mapping[str, Any],
    *,
    historical_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    overall = report["overall"]["replacement_metrics"]
    patterns = report["patterns"]
    aggregates = report["pattern_aggregates"]

    def method(name: str) -> dict[str, Any]:
        if name not in overall:
            return {"status": "control_checkpoint_not_supplied"}
        return {
            "status": "evaluated",
            "overall31": _metric_slice(overall[name]),
            "full5": _metric_slice(patterns["full"]["replacement_metrics"][name]),
            "all30": aggregates["all30"][name],
            "single5": aggregates["single5"][name],
            "multi_modal": aggregates["multi_modal"][name],
            "csi_present_with_sensing": aggregates["csi_present_with_sensing"][name],
            "csi_absent_legacy15": aggregates["csi_absent_legacy15"][name],
        }

    r0 = (
        dict(historical_reference)
        if isinstance(historical_reference, Mapping)
        else {"status": "historical_reference_not_supplied", "requires_external_provenance": True}
    )

    return {
        "R0_four_modality_pcpf": r0,
        "R1_five_modality_checkpoint_csi_masked": {
            "status": "evaluated_on_all_legacy_nonempty_sensing_masks",
            "csi_absent_legacy15": aggregates["csi_absent_legacy15"]["pcpf_analytic"],
            "full_four_sensing_modalities": _metric_slice(
                patterns["missing_csi"]["replacement_metrics"]["pcpf_analytic"]
            ),
        },
        "R2_five_modality_uniform": method("uniform"),
        "R3_five_modality_static_prior": method("static_prior"),
        "R4_five_modality_direct_router": method("direct_router_control"),
        "R5_five_modality_cuaf_local_adaptation": method("cuaf_local_adaptation"),
        "R6_five_modality_pcpf_analytic": method("pcpf_analytic"),
        "R7_joint_checkpoint_csi_only": {
            "status": "evaluated",
            "csi_only": _metric_slice(patterns["csi_only"]["replacement_metrics"]["pcpf_analytic"]),
        },
    }


def _metric_slice(metrics: Mapping[str, Any]) -> dict[str, Any]:
    names = ("count", "top1", "top3", "top5", "within_3", "circular_mae", "nll", "brier", "ece")
    return {name: metrics[name] for name in names if name in metrics}


def _expert_input_diagnostics(records: Mapping[str, Any]) -> dict[str, Any]:
    patterns = list(records["pattern"])
    rows = torch.tensor([value == "full" for value in patterns], dtype=torch.bool)
    if not bool(rows.any().item()):
        rows = torch.ones(len(patterns), dtype=torch.bool)
    available = torch.as_tensor(records["available"], dtype=torch.bool)[rows]
    distances = _record_prototype_distance(records)[rows]
    modalities = list(records["modalities"])
    result: dict[str, Any] = {
        "source_mask": "full" if "full" in set(patterns) else "all_available_rows",
        "prototype_distance": {
            name: _distribution(distances[:, index][available[:, index]])
            for index, name in enumerate(modalities)
        },
    }
    if tuple(modalities) == PCPF_SPARSE_CSI_MODALITIES:
        csi_available = available[:, -1]
        result["csi_input_quality"] = {
            "valid_ratio": _distribution(torch.as_tensor(records["csi_valid_ratio"], dtype=torch.float32)[rows][csi_available]),
            "log_rms": _distribution(torch.as_tensor(records["csi_log_rms"], dtype=torch.float32)[rows][csi_available]),
            "quality_confidence": _distribution(
                torch.as_tensor(records["csi_quality_confidence"], dtype=torch.float32)[rows][csi_available]
            ),
            "snr_available_fraction": _mean_or_none(
                torch.as_tensor(records["csi_snr_available"], dtype=torch.bool)[rows][csi_available].float()
            ),
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


def _mask_group(name: str, pattern: Sequence[int], modality_count: int) -> str:
    available = sum(int(value) for value in pattern)
    if name == "full" or available == modality_count:
        return "full"
    if available == 1:
        return "single"
    return f"drop{modality_count - available}"


def resolve_pcpf_missing_patterns(
    patterns: str | list[str] | tuple[str, ...] | None,
    modalities: list[str] | tuple[str, ...],
) -> dict[str, list[int]]:
    names = tuple(str(value) for value in modalities)
    if names == PCPF_SENSING_MODALITIES:
        return resolve_missing_patterns(patterns, names)
    if names != PCPF_SPARSE_CSI_MODALITIES:
        raise ValueError("PCPF missing patterns require canonical sensing modalities with optional sparse CSI.")
    raw = [patterns] if isinstance(patterns, str) else list(patterns or ())
    if raw not in ([], ["all_nonempty"], ["default"]):
        raise ValueError("Five-modality PCPF evaluation requires patterns=[all_nonempty].")
    result: dict[str, list[int]] = {}
    for bits in range(1, 1 << len(names)):
        mask = [int(bool(bits & (1 << index))) for index in range(len(names))]
        available = [name for name, keep in zip(names, mask, strict=True) if keep]
        missing = [name for name, keep in zip(names, mask, strict=True) if not keep]
        if len(available) == len(names):
            pattern_name = "full"
        elif len(available) == 1:
            pattern_name = f"{available[0]}_only"
        else:
            pattern_name = "missing_" + "_".join(missing)
        result[pattern_name] = mask
    return result


def _bootstrap_config(value: Mapping[str, Any] | None) -> dict[str, int | float]:
    raw = dict(value or {})
    unknown = sorted(set(raw) - {"seed", "resamples", "confidence"})
    if unknown:
        raise ValueError(f"PCPF bootstrap config contains unsupported fields: {unknown}.")
    seed = int(raw.get("seed", 1))
    resamples = int(raw.get("resamples", 10_000))
    confidence = float(raw.get("confidence", 0.95))
    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("PCPF bootstrap requires positive resamples and confidence in (0,1).")
    return {"seed": seed, "resamples": resamples, "confidence": confidence}


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    finite = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if not finite.numel():
        return {"count": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "p10": float(torch.quantile(finite, 0.1).item()),
        "p50": float(torch.quantile(finite, 0.5).item()),
        "p90": float(torch.quantile(finite, 0.9).item()),
    }


def _bootstrap_scalar(
    values: torch.Tensor,
    group_ids: Sequence[str],
    *,
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    if tensor.numel() != len(group_ids):
        raise ValueError("Paired bootstrap values and group ids must have equal length.")
    grouped: dict[str, list[torch.Tensor]] = {}
    for group, value in zip(group_ids, tensor, strict=True):
        if torch.isfinite(value):
            grouped.setdefault(str(group), []).append(value)
    group_values = torch.stack([torch.stack(items).mean() for _, items in sorted(grouped.items())]) if grouped else tensor[:0]
    if not group_values.numel():
        return {"group_count": 0, "estimate": None, "ci_lower": None, "ci_upper": None}
    generator = torch.Generator().manual_seed(int(seed))
    sampled = torch.randint(
        0,
        group_values.numel(),
        (int(resamples), group_values.numel()),
        generator=generator,
    )
    estimates = group_values[sampled].mean(dim=1)
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "group_count": int(group_values.numel()),
        "estimate": float(group_values.mean().item()),
        "ci_lower": float(torch.quantile(estimates, tail).item()),
        "ci_upper": float(torch.quantile(estimates, 1.0 - tail).item()),
    }


def _bootstrap_replacement_deltas(records: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    labels = torch.as_tensor(records["labels"], dtype=torch.long)
    groups = [str(value) for value in records["group_id"]]
    baseline = torch.as_tensor(records["replacement_probability"]["pcpf_analytic"]).argmax(dim=-1).eq(labels).float()
    result = {}
    for offset, (name, probability) in enumerate(sorted(records["replacement_probability"].items())):
        if name == "pcpf_analytic":
            continue
        correct = torch.as_tensor(probability).argmax(dim=-1).eq(labels).float()
        interval = _bootstrap_scalar(
            baseline - correct,
            groups,
            seed=int(config["seed"]) + offset,
            resamples=int(config["resamples"]),
            confidence=float(config["confidence"]),
        )
        domain_deltas = []
        for domain in sorted(set(records["domain"])):
            rows = torch.tensor([value == domain for value in records["domain"]], dtype=torch.bool)
            domain_deltas.append(float((baseline[rows] - correct[rows]).mean().item()))
        interval["win_domain_count"] = sum(value > 0.0 for value in domain_deltas)
        interval["tie_domain_count"] = sum(value == 0.0 for value in domain_deltas)
        interval["domain_count"] = len(domain_deltas)
        result[f"pcpf_analytic_minus_{name}"] = interval
    return result


def _dynamicity_tests(records: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    risk = torch.as_tensor(records["raw_risk"], dtype=torch.float32)
    available = torch.as_tensor(records["available"], dtype=torch.bool)
    calibrated = torch.as_tensor(records["calibrated_unimodal_probabilities"], dtype=torch.float32)
    capability = torch.as_tensor(records["static_capability"], dtype=torch.float32)
    tau = float(records["fusion_tau"])
    shuffled = _groupwise_risk(records, risk, mode="shuffle", seed=int(bootstrap["seed"]))
    grouped_mean = _groupwise_risk(records, risk, mode="mean", seed=int(bootstrap["seed"]))
    d0_weights = torch.as_tensor(records["replacement_weights"]["pcpf_analytic"], dtype=torch.float32)
    d1_weights = analytic_fusion_weights(
        risk=shuffled,
        available=available,
        static_capability=capability,
        tau=tau,
    )
    d2_weights = analytic_fusion_weights(
        risk=grouped_mean,
        available=available,
        static_capability=capability,
        tau=tau,
    )
    d3_weights = torch.as_tensor(records["replacement_weights"]["static_prior"], dtype=torch.float32)
    probabilities = {
        "D0_original_sample_risk": _weighted_probability(d0_weights, calibrated),
        "D1_domain_mask_shuffled_risk": _weighted_probability(d1_weights, calibrated),
        "D2_domain_mask_mean_risk": _weighted_probability(d2_weights, calibrated),
        "D3_static_prior": _weighted_probability(d3_weights, calibrated),
    }
    methods = {name: _diagnostic_method_metrics(probability, records) for name, probability in probabilities.items()}
    labels = torch.as_tensor(records["labels"], dtype=torch.long)
    group_ids = [str(value) for value in records["group_id"]]
    d0_correct = probabilities["D0_original_sample_risk"].argmax(dim=-1).eq(labels).float()
    intervals: dict[str, Any] = {}
    for offset, (name, probability) in enumerate(probabilities.items()):
        if name == "D0_original_sample_risk":
            continue
        correct = probability.argmax(dim=-1).eq(labels).float()
        intervals[f"D0_minus_{name.split('_', 1)[0]}"] = _bootstrap_scalar(
            d0_correct - correct,
            group_ids,
            seed=int(bootstrap["seed"]) + offset,
            resamples=int(bootstrap["resamples"]),
            confidence=float(bootstrap["confidence"]),
        )
    return {
        "grouping": "domain+mask",
        "shuffle_seed": int(bootstrap["seed"]),
        "same_cached_unimodal_evidence": True,
        "methods": methods,
        "paired_group_bootstrap_top1": intervals,
    }


def _groupwise_risk(
    records: Mapping[str, Any],
    risk: torch.Tensor,
    *,
    mode: str,
    seed: int,
) -> torch.Tensor:
    if mode not in {"shuffle", "mean"}:
        raise ValueError("PCPF grouped risk replacement mode must be shuffle or mean.")
    result = torch.empty_like(risk)
    keys = list(zip(records["domain"], records["pattern"], strict=True))
    generator = torch.Generator().manual_seed(int(seed))
    for key in sorted(set(keys)):
        indices = torch.tensor([index for index, value in enumerate(keys) if value == key], dtype=torch.long)
        if mode == "shuffle":
            result[indices] = risk[indices[torch.randperm(indices.numel(), generator=generator)]]
        else:
            result[indices] = risk[indices].mean(dim=0, keepdim=True)
    return result


def _diagnostic_method_metrics(probability: torch.Tensor, records: Mapping[str, Any]) -> dict[str, Any]:
    count = int(torch.as_tensor(records["labels"]).numel())
    rows = torch.ones(count, dtype=torch.bool)
    result = {"overall": _compact_classification_metrics(probability, records, rows)}
    for output_key, values in (
        ("by_pattern", records["pattern"]),
        ("by_weather", records["weather"]),
        ("by_domain", records["domain"]),
        ("by_cardinality", [f"n={value}" for value in torch.as_tensor(records["available"]).sum(dim=1).tolist()]),
        ("by_csi_presence", ["present" if value else "absent" for value in torch.as_tensor(records["available"])[:, -1].tolist()]),
    ):
        result[output_key] = {
            group: _compact_classification_metrics(
                probability,
                records,
                torch.tensor([value == group for value in values], dtype=torch.bool),
            )
            for group in sorted(set(values))
        }
    return result


def _compact_classification_metrics(
    probability: torch.Tensor,
    records: Mapping[str, Any],
    rows: torch.Tensor,
) -> dict[str, Any]:
    metrics = _classification_metrics(
        torch.as_tensor(probability, dtype=torch.float32)[rows],
        torch.as_tensor(records["labels"], dtype=torch.long)[rows],
    )
    metrics.pop("reliability_diagram", None)
    return metrics


def _risk_and_component_diagnostics(
    records: Mapping[str, Any],
    *,
    train_confidence_p90: torch.Tensor,
) -> dict[str, Any]:
    modalities = list(records["modalities"])
    risk = torch.as_tensor(records["raw_risk"], dtype=torch.float32)
    components = torch.as_tensor(records["risk_components"], dtype=torch.float32)
    errors = _record_circular_errors(records)
    probabilities = torch.as_tensor(records["unimodal_probabilities"], dtype=torch.float32)
    available = torch.as_tensor(records["available"], dtype=torch.bool)
    labels = torch.as_tensor(records["labels"], dtype=torch.long)
    confidence, prediction = probabilities.max(dim=-1)
    wrong = prediction.ne(labels.unsqueeze(1)) & available
    threshold = torch.as_tensor(train_confidence_p90, dtype=torch.float32).view(1, -1)
    confident_wrong = wrong & confidence.ge(threshold)
    component_names = ("var", "proto", "temp", "conflict")
    per_modality = {}
    for index, name in enumerate(modalities):
        valid = available[:, index]
        correct = valid & ~wrong[:, index]
        per_modality[name] = {
            "risk_vs_true_circular_error": _correlations(risk[:, index][valid], errors[:, index][valid]),
            "risk_error_detection_auroc": _binary_auroc(risk[:, index][valid], wrong[:, index][valid]),
            "correct_risk": _distribution(risk[:, index][correct]),
            "wrong_risk": _distribution(risk[:, index][wrong[:, index]]),
            "confident_wrong_risk": _distribution(risk[:, index][confident_wrong[:, index]]),
            "components": {
                component_name: {
                    "correlations": _correlations(
                        components[:, index, component_index][valid],
                        errors[:, index][valid],
                    ),
                    "error_detection_auroc": _binary_auroc(
                        components[:, index, component_index][valid],
                        wrong[:, index][valid],
                    ),
                    "correct_mean": _mean_or_none(components[:, index, component_index][correct]),
                    "wrong_mean": _mean_or_none(components[:, index, component_index][wrong[:, index]]),
                    "confident_wrong_mean": _mean_or_none(
                        components[:, index, component_index][confident_wrong[:, index]]
                    ),
                }
                for component_index, component_name in enumerate(component_names)
            },
        }
    grouped = {}
    cardinality = [f"n={value}" for value in available.sum(dim=1).tolist()]
    csi_presence = ["present" if value else "absent" for value in available[:, -1].tolist()]
    for key, values in (
        ("weather", records["weather"]),
        ("domain", records["domain"]),
        ("csi_presence", csi_presence),
        ("availability_cardinality", cardinality),
    ):
        grouped[key] = {
            group: _group_risk_correlations(
                risk,
                errors,
                available,
                modalities,
                torch.tensor([value == group for value in values], dtype=torch.bool),
            )
            for group in sorted(set(values))
        }
    csi_index = len(modalities) - 1
    csi_valid = available[:, csi_index]
    csi_quality = torch.as_tensor(records["csi_quality_confidence"], dtype=torch.float32)
    if csi_quality.ndim > 1:
        csi_quality = csi_quality.mean(dim=tuple(range(1, csi_quality.ndim)))
    return {
        "true_error_definition": "hard_top1_circular_class_distance",
        "per_modality": per_modality,
        "grouped_risk_correlations": grouped,
        "csi_quality": {
            "quality_confidence": _distribution(csi_quality[csi_valid]),
            "one_minus_quality_vs_error": _correlations(1.0 - csi_quality[csi_valid], errors[:, csi_index][csi_valid]),
            "one_minus_quality_error_detection_auroc": _binary_auroc(
                1.0 - csi_quality[csi_valid],
                wrong[:, csi_index][csi_valid],
            ),
        },
    }


def _group_risk_correlations(
    risk: torch.Tensor,
    errors: torch.Tensor,
    available: torch.Tensor,
    modalities: list[str],
    rows: torch.Tensor,
) -> dict[str, Any]:
    valid = available[rows]
    return {
        "overall": _correlations(risk[rows][valid], errors[rows][valid]),
        "modalities": {
            name: _correlations(
                risk[rows, index][valid[:, index]],
                errors[rows, index][valid[:, index]],
            )
            for index, name in enumerate(modalities)
        },
    }


def _binary_auroc(score: torch.Tensor, positive: torch.Tensor) -> float | None:
    values = torch.as_tensor(score, dtype=torch.float32).reshape(-1)
    target = torch.as_tensor(positive, dtype=torch.bool).reshape(-1)
    finite = torch.isfinite(values)
    values, target = values[finite], target[finite]
    positives = int(target.sum().item())
    negatives = int((~target).sum().item())
    if positives == 0 or negatives == 0:
        return None
    rank_sum = _rank(values)[target].sum()
    return float(((rank_sum - positives * (positives - 1) / 2.0) / (positives * negatives)).item())


def _paired_weight_transfer(records: Mapping[str, Any]) -> dict[str, Any]:
    patterns = list(records["pattern"])
    sample_ids = list(records["sample_id"])
    weights = torch.as_tensor(records["fusion_weights"], dtype=torch.float32)
    available = torch.as_tensor(records["available"], dtype=torch.bool)
    full_indices = [index for index, name in enumerate(patterns) if name == "full"]
    full_by_sample = {sample_ids[index]: index for index in full_indices}
    if len(full_by_sample) != len(full_indices):
        raise ValueError("PCPF weight-transfer diagnostic requires unique sample ids within the Full mask.")
    result: dict[str, Any] = {}
    for pattern in sorted(set(patterns) - {"full"}):
        current_indices = [index for index, name in enumerate(patterns) if name == pattern]
        paired_full = [full_by_sample.get(sample_ids[index]) for index in current_indices]
        if any(index is None for index in paired_full):
            raise ValueError(f"PCPF weight-transfer diagnostic cannot pair pattern {pattern!r} to Full by sample id.")
        full = weights[torch.tensor(paired_full, dtype=torch.long)]
        current = weights[torch.tensor(current_indices, dtype=torch.long)]
        current_available = available[torch.tensor(current_indices, dtype=torch.long)]
        delta = current - full
        result[pattern] = {
            "sample_count": len(current_indices),
            "availability_count": int(current_available[0].sum().item()) if len(current_indices) else 0,
            "per_modality": {
                name: {
                    "delta_all_samples": _distribution(delta[:, modality_index]),
                    "delta_when_available": _distribution(delta[:, modality_index][current_available[:, modality_index]]),
                    "full_weight": _distribution(full[:, modality_index]),
                    "masked_weight_max": float(current[:, modality_index][~current_available[:, modality_index]].abs().max().item())
                    if bool((~current_available[:, modality_index]).any().item())
                    else 0.0,
                }
                for modality_index, name in enumerate(records["modalities"])
            },
            "weight_row_sum_max_error": float((current.sum(dim=1) - 1.0).abs().max().item()),
        }
    return result


def _weighted_probability(weights: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
    fused = (weights.unsqueeze(-1) * probability).sum(dim=1)
    return fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"PCPF diagnostics are missing tensor {key!r}.")
    return value


def _circular_class_errors(prediction: torch.Tensor, labels: torch.Tensor, *, classes: int = 64) -> torch.Tensor:
    predicted = torch.as_tensor(prediction, dtype=torch.long)
    target = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    if predicted.ndim != 2 or predicted.shape[0] != target.numel():
        raise ValueError("Unimodal predictions and labels must have shapes [B,M] and [B].")
    difference = (predicted - target.unsqueeze(1)).abs()
    return torch.minimum(difference, int(classes) - difference)


def _record_circular_errors(records: Mapping[str, Any]) -> torch.Tensor:
    value = records.get("unimodal_circular_errors")
    if value is not None:
        return torch.as_tensor(value, dtype=torch.float32)
    probability = torch.as_tensor(records["unimodal_probabilities"], dtype=torch.float32)
    labels = torch.as_tensor(records["labels"], dtype=torch.long)
    return _circular_class_errors(probability.argmax(dim=-1), labels).float()


def _record_prototype_distance(records: Mapping[str, Any]) -> torch.Tensor:
    value = records.get("prototype_distance")
    if value is not None:
        return torch.as_tensor(value, dtype=torch.float32)
    components = torch.as_tensor(records["risk_components"], dtype=torch.float32)
    return components[..., 1]


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
    "build_pcpf_mechanism_diagnostics",
    "build_stage2_gate_report",
    "collect_pcpf_observations",
    "fit_train_confidence_p90",
    "resolve_pcpf_missing_patterns",
    "summarize_pcpf_matrix",
    "write_pcpf_observation_cache",
    "write_pcpf_report",
]
