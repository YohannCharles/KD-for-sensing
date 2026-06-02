from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Mapping

from kd_sensing.engine.hist_beam_loso_comparisons import compare_adapter_to_source, compare_proto_to_full
from kd_sensing.engine.hist_beam_loso_artifacts import _write_json
from kd_sensing.engine.hist_beam_loso_config import DEFAULT_SOURCE_BASELINE_VARIANT
from kd_sensing.engine.run_lineage import run_lineage_metadata

def row_eligibility(
    row: Mapping[str, Any],
    adaptation_metrics: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = list(adaptation_metrics.get("eligibility_reasons", []) or [])
    status = str(adaptation_metrics.get("eligibility_status", "") or "").strip().lower()
    used_oracle_fields = list(adaptation_metrics.get("used_target_oracle_fields", []) or [])
    if status == "unknown_oracle_usage":
        reasons.append("unknown_oracle_usage")
    if used_oracle_fields:
        reasons.append("target_oracle_fields_used")
    if bool(adaptation_metrics.get("main_conclusion_eligible", True)) is False:
        reasons.append("run_marked_ineligible")
    if bool(row.get("distillation_enabled", False)) or str(row.get("method_family", "")) == "legacy_kd":
        reasons.append("legacy_kd_supplemental")
    if row.get("run_status") != "completed":
        reasons.append("run_not_completed")

    sensitive_usage = {
        "used_target_beam_power_for_training": "target_beam_power_supervision",
        "used_target_csi_for_training": "target_csi_supervision",
        "used_target_path_params_for_training": "target_path_params_supervision",
        "used_target_path_descriptor_for_training": "target_path_descriptor_supervision",
        "used_target_path_label_for_training": "target_path_label_supervision",
        "used_target_radio_label_for_training": "target_radio_label_supervision",
    }
    policy = adaptation_metrics.get("sensitive_field_policy")
    allow_sensitive_main = bool(
        policy.get("allow_target_sensitive_supervision_in_main_conclusion", False)
    ) if isinstance(policy, Mapping) else False
    if not allow_sensitive_main:
        for key, reason in sensitive_usage.items():
            if bool(row.get(key, False)):
                reasons.append(reason)

    if prototype_is_required(row) and prototype_is_no_op(row):
        reasons.append("prototype_no_op")
    if bool(primary_metrics.get("target_leakage", False)):
        reasons.append("target_leakage")
    if bool(adaptation_metrics.get("target_leakage", False)):
        reasons.append("target_leakage")
    reasons.extend(_consumed_oracle_reasons(row, adaptation_metrics))
    reasons.extend(_split_eligibility_reasons(row))

    unique = unique_reasons(reasons)
    return {
        "main_conclusion_eligible": len(unique) == 0,
        "eligibility_status": "eligible" if len(unique) == 0 else "ineligible",
        "eligibility_reasons": unique,
    }


def prototype_is_required(row: Mapping[str, Any]) -> bool:
    variant = str(row.get("variant"))
    return variant in {"v5_adapter_proto", "v6_radio_proto", "adapter_radio_proto", "v8_path_proto", "adapter_path_proto"}


def prototype_is_no_op(row: Mapping[str, Any]) -> bool:
    status = str(row.get("prototype_status") or "").strip().lower()
    if status in {"no_op", "unavailable", "skipped"}:
        return True
    coverage = row.get("prototype_coverage")
    if isinstance(coverage, (int, float)) and float(coverage) <= 0.0:
        return True
    used = row.get("prototype_used_sample_count")
    if isinstance(used, (int, float)) and float(used) <= 0.0:
        return True
    loss = row.get("prototype_loss_mean")
    if status == "" and prototype_is_required(row) and loss is None and row.get("proto_type") is not None:
        return True
    return False


def unique_reasons(reasons: Iterable[Any]) -> list[str]:
    unique: list[str] = []
    for item in reasons:
        text = str(item).strip()
        if text and text not in unique:
            unique.append(text)
    return unique


def _split_eligibility_reasons(row: Mapping[str, Any]) -> list[str]:
    if not _is_mmw_town10_row(row):
        return []
    reasons: list[str] = []
    strict = row.get("strict_validation_eligible")
    if strict is False:
        row_reasons = list(row.get("split_eligibility_reasons") or [])
        reasons.extend(row_reasons or ["split_not_strict_validation_eligible"])
    elif strict is None:
        reasons.append("split_eligibility_unknown")
        diagnostics = row.get("split_metadata_path") or row.get("leakage_diagnostics") or row.get("metrics_path")
        if diagnostics:
            reasons.append(f"split_eligibility_unknown_diagnostics:{diagnostics}")
        else:
            reasons.append("split_eligibility_unknown_missing_metadata_path")
    return reasons


def _consumed_oracle_reasons(row: Mapping[str, Any], adaptation_metrics: Mapping[str, Any]) -> list[str]:
    consumed = adaptation_metrics.get("consumed_fields")
    if not isinstance(consumed, Mapping):
        consumed = row.get("consumed_fields")
    if not isinstance(consumed, Mapping):
        return []
    disabled = {
        "gps",
        "lidar",
        "radar",
        "mmwave",
        "csi",
        "channel",
        "path",
        "beam_power",
        "beamspace_power_label",
        "radio_semantic_label",
        "path_semantic_label",
        "path_descriptor",
        "path_params",
    }
    reasons: list[str] = []
    for stage, stage_fields in consumed.items():
        if not isinstance(stage_fields, Mapping):
            continue
        fields = []
        for key in ("consumed_input_fields", "consumed_label_fields"):
            fields.extend(str(item) for item in stage_fields.get(key, []) or [])
        for field in fields:
            clean = field.split(":", 1)[0]
            leaf = clean.rsplit(".", 1)[-1]
            stage_text = str(stage)
            if leaf == "target_beam":
                allowed_support = stage_text == "target_adaptation" and clean.startswith("target_support.")
                allowed_eval = stage_text in {"source_only_target_test_eval", "adapted_target_test_eval", "target_test"} and field.endswith(":evaluation_only")
                if not (allowed_support or allowed_eval or clean.startswith("source.")):
                    reasons.append(f"target_oracle_consumed:{stage_text}:{field}")
                continue
            if leaf in disabled:
                reasons.append(f"target_oracle_consumed:{stage_text}:{field}")
    return unique_reasons(reasons)


def _is_mmw_town10_row(row: Mapping[str, Any]) -> bool:
    family = str(row.get("dataset_family") or "").strip().lower()
    if family != "mmw":
        return False
    values = [
        row.get("town"),
        row.get("condition"),
        row.get("target_scene"),
        row.get("fold"),
        *(row.get("source_scenes") or []),
    ]
    text = " ".join(str(item) for item in values if item is not None).lower()
    return "town10" in text or row.get("town") is not None


def reason_histogram(reason_lists: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for reasons in reason_lists:
        if isinstance(reasons, str):
            iterable = [reasons]
        else:
            iterable = list(reasons or [])
        for reason in unique_reasons(iterable):
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def excluded_run_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant": row.get("variant"),
        "target_scene": row.get("target_scene"),
        "budget": row.get("budget"),
        "seed": row.get("seed"),
        "eligibility_reasons": list(row.get("eligibility_reasons", []) or []),
        "split_strategy": row.get("split_strategy"),
        "strict_validation_eligible": row.get("strict_validation_eligible"),
        "split_metadata_path": row.get("split_metadata_path"),
        "metrics_path": row.get("metrics_path"),
        "prediction_hist_path": row.get("prediction_hist_path"),
        "collapse_diagnostics_path": row.get("collapse_diagnostics_path"),
        "run_id": row.get("run_id"),
        "eligibility_status": row.get("eligibility_status"),
        "used_target_oracle_fields": row.get("used_target_oracle_fields"),
        "target_oracle_usage_stage": row.get("target_oracle_usage_stage"),
        "oracle_usage_summary": row.get("oracle_usage_summary"),
    }


def conclusion_source_artifacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    artifacts = []
    for row in rows:
        source = row.get("eligibility_source_artifacts")
        if isinstance(source, Mapping):
            artifacts.append(
                {
                    "run_id": row.get("run_id"),
                    "variant": row.get("variant"),
                    "target_scene": row.get("target_scene"),
                    "budget": row.get("budget"),
                    "seed": row.get("seed"),
                    **dict(source),
                }
            )
    return artifacts



def write_quick_validation_conclusion(
    output_dir: str | Path,
    run_records: list[dict[str, Any]],
    summary_path: str | Path,
) -> Path:
    out_dir = Path(output_dir)
    rows = [_summary_row(record) for record in run_records]
    by_key = {
        (row["target_scene"], row["budget"], row["seed"], row["variant"]): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    groups = sorted({(row["target_scene"], row["budget"], row["seed"]) for row in rows})
    for target_scene, budget, seed in groups:
        baseline = by_key.get((target_scene, budget, seed, DEFAULT_SOURCE_BASELINE_VARIANT)) or by_key.get(
            (target_scene, budget, seed, "v0_flat")
        )
        baseline_variant = str(baseline.get("variant")) if isinstance(baseline, Mapping) else DEFAULT_SOURCE_BASELINE_VARIANT
        for variant in ("v4_adapter", "v5_adapter_proto", "v8_path_proto"):
            candidate = by_key.get((target_scene, budget, seed, variant))
            comparisons.append(
                compare_adapter_to_source(
                    target_scene=target_scene,
                    budget=budget,
                    seed=seed,
                    variant=variant,
                    baseline=baseline,
                    candidate=candidate,
                    baseline_variant=baseline_variant,
                )
            )
        comparisons.append(
            compare_proto_to_full(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                proto=by_key.get((target_scene, budget, seed, "v5_adapter_proto")),
                full=by_key.get((target_scene, budget, seed, "v6_full_finetune")),
            )
        )
        comparisons.append(
            _compare_coarse_to_radio(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                coarse=by_key.get((target_scene, budget, seed, "v5_adapter_proto")),
                radio=by_key.get((target_scene, budget, seed, "v6_radio_proto")),
            )
        )
        comparisons.append(
            _compare_radio_condition(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                off=by_key.get((target_scene, budget, seed, "adapter_radio_proto")),
                on=by_key.get((target_scene, budget, seed, "v6_radio_proto")),
            )
        )
        comparisons.append(
            _compare_radio_to_path(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                radio=by_key.get((target_scene, budget, seed, "v6_radio_proto")),
                path=by_key.get((target_scene, budget, seed, "v8_path_proto")),
            )
        )
        comparisons.append(
            _compare_path_to_full(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                path=by_key.get((target_scene, budget, seed, "v8_path_proto")),
                full=by_key.get((target_scene, budget, seed, "v6_full_finetune")),
            )
        )
        comparisons.append(
            _compare_path_condition(
                target_scene=target_scene,
                budget=budget,
                seed=seed,
                off=by_key.get((target_scene, budget, seed, "adapter_path_proto")),
                on=by_key.get((target_scene, budget, seed, "v8_path_proto")),
            )
        )
    excluded_runs = [_excluded_run_summary(row) for row in rows if not bool(row.get("main_conclusion_eligible", True))]
    inconclusive_count = sum(1 for item in comparisons if item.get("status") != "complete")
    payload = {
        "generated_at": _utc_now(),
        "summary_path": str(summary_path),
        "source_paths": {
            "summary_path": str(summary_path),
            "run_artifacts": _conclusion_source_artifacts(rows),
        },
        "eligible_run_count": len(rows) - len(excluded_runs),
        "excluded_run_count": len(excluded_runs),
        "eligible_run_count_zero_reason": _eligible_zero_reason(rows, excluded_runs),
        "inconclusive_comparison_count": inconclusive_count,
        "exclusion_reason_histogram": _reason_histogram(row.get("eligibility_reasons", []) for row in rows),
        "excluded_runs": excluded_runs,
        "status": "completed" if comparisons and all(item["status"] == "complete" for item in comparisons) else "inconclusive",
        "comparisons": comparisons,
    }
    path = out_dir / "quick_validation_conclusion.json"
    _write_json(path, payload)
    return path


def _eligible_zero_reason(rows: list[dict[str, Any]], excluded_runs: list[dict[str, Any]]) -> str | None:
    if not rows or len(rows) != len(excluded_runs):
        return None
    histogram = reason_histogram(row.get("eligibility_reasons", []) for row in rows)
    if not histogram:
        return "no_eligible_runs_without_machine_readable_reason"
    return ",".join(f"{key}:{value}" for key, value in sorted(histogram.items()))


def _claim_scope_from_rows(rows: list[dict[str, Any]]) -> str:
    scopes = sorted({str(row.get("claim_scope", "cross_scene")) for row in rows})
    if not scopes:
        return "unavailable"
    if len(scopes) == 1:
        return scopes[0]
    return "mixed"


def _summary_row(record: dict[str, Any]) -> dict[str, Any]:
    source_train_metrics = record.get("metrics", {}).get("source_train", {})
    source_metrics = record.get("metrics", {}).get("source_only_target_test_eval", {})
    adapted_metrics = record.get("metrics", {}).get("adapted_target_test_eval", {})
    adaptation_metrics = record.get("metrics", {}).get("target_adaptation", {})
    primary_metrics = adapted_metrics if adapted_metrics else source_metrics
    source_top1 = _metric(source_metrics, "top1")
    source_top3 = _metric(source_metrics, "top3")
    source_top5 = _metric(source_metrics, "top5")
    adapted_top1 = _metric(adapted_metrics, "top1")
    adapted_top3 = _metric(adapted_metrics, "top3")
    adapted_top5 = _metric(adapted_metrics, "top5")
    last_beam = _last_beam_summary(primary_metrics)
    cache_summary = _cache_summary(record, source_train_metrics)
    split_summary = _split_summary(record, source_metrics, adapted_metrics, primary_metrics)
    lineage = _lineage_summary(record, source_train_metrics, adaptation_metrics, primary_metrics)
    row = {
        "run_id": record.get("run_id"),
        "run_status": record.get("status"),
        "fold": record.get("fold"),
        "target_scene": record.get("target_scene"),
        "source_scenes": record.get("source_scenes"),
        "dataset_family": record.get("dataset_family") or record.get("scene_family") or "DeepSense6G",
        "condition": record.get("condition"),
        "town": record.get("town"),
        "profile": record.get("profile"),
        "modality_profile": record.get("modality_profile") or record.get("profile"),
        "matrix_scope": record.get("matrix_scope"),
        "quick_validation": record.get("quick_validation"),
        "enabled_modalities": list(record.get("enabled_modalities") or primary_metrics.get("enabled_modalities") or []),
        "excluded_sensitive_fields": record.get("excluded_sensitive_fields"),
        "cache_policy": cache_summary.get("cache_policy"),
        "lidar_cache_policy": cache_summary.get("lidar_cache_policy"),
        "lidar_cache_dir": cache_summary.get("lidar_cache_dir"),
        "num_workers": cache_summary.get("num_workers"),
        "cpu_threads": cache_summary.get("cpu_threads"),
        "claim_scope": record.get("claim_scope") or "cross_scene",
        "cross_scene_claim_allowed": True if record.get("cross_scene_claim_allowed") is None else record.get("cross_scene_claim_allowed"),
        "split_protocol": split_summary.get("split_protocol"),
        "split_strategy": split_summary.get("split_strategy"),
        "split_protocol_version": split_summary.get("split_protocol_version"),
        "split_metadata_path": split_summary.get("split_metadata_path"),
        "split_metadata_available": split_summary.get("split_metadata_available"),
        "strict_validation_eligible": split_summary.get("strict_validation_eligible"),
        "split_eligibility": split_summary.get("split_eligibility"),
        "split_eligibility_reasons": split_summary.get("eligibility_reasons"),
        "split_fix_hint": split_summary.get("fix_hint"),
        "split_seed": split_summary.get("split_seed"),
        "split_sequence_count": split_summary.get("split_sequence_count"),
        "split_num_samples": split_summary.get("split_num_samples"),
        "leakage_diagnostics": split_summary.get("leakage_diagnostics"),
        "variant": record.get("variant"),
        "budget": record.get("budget"),
        "seed": record.get("seed"),
        "distillation_enabled": lineage["distillation_enabled"],
        "distillation_type": lineage["distillation_type"],
        "teacher_checkpoint": lineage["teacher_checkpoint"],
        "teacher_source": lineage["teacher_source"],
        "student_model": lineage["student_model"],
        "distillation_lifecycle": lineage["distillation_lifecycle"],
        "baseline_role": lineage.get("baseline_role"),
        "reproduction_scope": lineage.get("reproduction_scope"),
        "stage_status": {stage["name"]: stage["status"] for stage in record.get("stages", [])},
        "failure_reason": record.get("failure_reason"),
        "metrics_path": _artifact(record, "adapted_target_test_eval.metrics_path") or _artifact(record, "source_only_target_test_eval.metrics_path"),
        "predictions_path": _artifact(record, "adapted_target_test_eval.predictions_path") or _artifact(record, "source_only_target_test_eval.predictions_path"),
        "prediction_hist_path": _artifact(record, "adapted_target_test_eval.prediction_hist_path") or _artifact(record, "source_only_target_test_eval.prediction_hist_path") or primary_metrics.get("prediction_hist_path"),
        "collapse_diagnostics_path": _artifact(record, "adapted_target_test_eval.collapse_diagnostics_path") or _artifact(record, "source_only_target_test_eval.collapse_diagnostics_path") or primary_metrics.get("collapse_diagnostics_path"),
        "source_checkpoint_path": _artifact(record, "source_train.source_checkpoint_path") or _artifact(record, "source_checkpoint_path"),
        "adaptation_checkpoint_path": _artifact(record, "target_adaptation.adaptation_checkpoint_path"),
        "source_prototype_path": _artifact(record, "target_adaptation.source_prototype_path") or _artifact(record, "source_train.source_prototype_path"),
        "source_beam_reference_path": _artifact(record, "source_train.source_beam_reference_path") or primary_metrics.get("source_beam_reference_path"),
        "top1": _metric(primary_metrics, "top1"),
        "top3": _metric(primary_metrics, "top3"),
        "top5": _metric(primary_metrics, "top5"),
        "source_top1": source_top1,
        "source_top3": source_top3,
        "source_top5": source_top5,
        "shared_top1": primary_metrics.get("shared_top1"),
        "shared_top3": primary_metrics.get("shared_top3"),
        "final_top1": primary_metrics.get("final_top1"),
        "final_top3": primary_metrics.get("final_top3"),
        "alpha_mean": primary_metrics.get("alpha_mean"),
        "alpha_std": primary_metrics.get("alpha_std"),
        "delta_norm": primary_metrics.get("delta_norm"),
        "phys_kl": primary_metrics.get("phys_kl"),
        "adapted_top1": adapted_top1,
        "adapted_top3": adapted_top3,
        "adapted_top5": adapted_top5,
        "adapted_source_top1_delta": _numeric_delta_from_values(adapted_top1, source_top1),
        "adapted_source_top3_delta": _numeric_delta_from_values(adapted_top3, source_top3),
        "adapted_source_top5_delta": _numeric_delta_from_values(adapted_top5, source_top5),
        "history_anchor_enabled": _bool_or_false(primary_metrics.get("history_anchor_enabled") or source_train_metrics.get("history_anchor_enabled")),
        "history_anchor_mode": primary_metrics.get("history_anchor_mode") or source_train_metrics.get("history_anchor_mode"),
        "residual_target_enabled": _bool_or_false(primary_metrics.get("residual_target_enabled") or source_train_metrics.get("residual_target_enabled")),
        "num_delta_classes": primary_metrics.get("num_delta_classes") or source_train_metrics.get("num_delta_classes"),
        "uses_input_beam_as_model_input": _bool_or_false(primary_metrics.get("uses_input_beam_as_model_input") or source_train_metrics.get("uses_input_beam_as_model_input")),
        "main_conclusion_profile": primary_metrics.get("main_conclusion_profile") or source_train_metrics.get("main_conclusion_profile"),
        "residual_accuracy": primary_metrics.get("residual_accuracy"),
        "reconstructed_absolute_top1": primary_metrics.get("reconstructed_absolute_top1_avg"),
        "reconstructed_absolute_top3": primary_metrics.get("reconstructed_absolute_top3_avg"),
        "reconstructed_absolute_top5": primary_metrics.get("reconstructed_absolute_top5_avg"),
        "markov_delta_top1": primary_metrics.get("markov_delta_top1"),
        "markov_delta_top3": primary_metrics.get("markov_delta_top3"),
        "markov_delta_top5": primary_metrics.get("markov_delta_top5"),
        "source_prior_collapse": primary_metrics.get("source_prior_collapse"),
        "unique_pred_beams": primary_metrics.get("unique_pred_beams"),
        "top1_pred_beam_ratio": primary_metrics.get("top1_pred_beam_ratio"),
        "top2_pred_beam_ratio": primary_metrics.get("top2_pred_beam_ratio"),
        "top5_pred_beam_ratio": primary_metrics.get("top5_pred_beam_ratio"),
        "within1": primary_metrics.get("within_1_acc"),
        "within2": primary_metrics.get("within_2_acc"),
        "within3": primary_metrics.get("within_3_acc"),
        "mae": primary_metrics.get("mean_abs_beam_error"),
        "bpl_db": primary_metrics.get("beam_power_loss_db"),
        "nrp": primary_metrics.get("normalized_received_power"),
        "confusion_by_true_beam_path": _artifact(record, "adapted_target_test_eval.confusion_by_true_beam_path") or _artifact(record, "source_only_target_test_eval.confusion_by_true_beam_path") or primary_metrics.get("confusion_by_true_beam_path"),
        "histogram_kl_pred_support": primary_metrics.get("kl_pred_support"),
        "histogram_kl_true_support": primary_metrics.get("kl_true_support"),
        "histogram_kl_pred_true": primary_metrics.get("kl_pred_true"),
        "beta_prior_initial": primary_metrics.get("beta_prior_initial"),
        "beta_prior_final": primary_metrics.get("beta_prior_final"),
        "beta_prior_effective": primary_metrics.get("beta_prior_effective"),
        "prototype_diagnostics": primary_metrics.get("prototype_diagnostics"),
        "coarse_accuracy": primary_metrics.get("coarse_accuracy"),
        "fine_accuracy": primary_metrics.get("fine_offset_accuracy"),
        "radio_semantic_accuracy": primary_metrics.get("radio_semantic_accuracy"),
        "radio_semantic_coverage": primary_metrics.get("radio_semantic_coverage"),
        "radio_metrics_unavailable_reason": primary_metrics.get("radio_metrics_unavailable_reason"),
        "path_semantic_accuracy": primary_metrics.get("path_semantic_accuracy"),
        "path_semantic_coverage": primary_metrics.get("path_semantic_coverage"),
        "path_metrics_unavailable_reason": primary_metrics.get("path_metrics_unavailable_reason"),
        "path_descriptor_regression_mse": primary_metrics.get("path_descriptor_regression_mse"),
        "prototype_assignment_confidence": primary_metrics.get("prototype_assignment_confidence"),
        "prototype_coverage_per_class": primary_metrics.get("prototype_coverage_per_class"),
        "source_target_path_class_histogram": primary_metrics.get("source_target_path_class_histogram"),
        "normalized_received_power": primary_metrics.get("normalized_received_power"),
        "beam_power_loss_db": primary_metrics.get("beam_power_loss_db"),
        "source_normalized_received_power": source_metrics.get("normalized_received_power"),
        "adapted_normalized_received_power": adapted_metrics.get("normalized_received_power"),
        "adapted_source_normalized_received_power_delta": _numeric_delta_from_values(
            adapted_metrics.get("normalized_received_power"),
            source_metrics.get("normalized_received_power"),
        ),
        "source_beam_power_loss_db": source_metrics.get("beam_power_loss_db"),
        "adapted_beam_power_loss_db": adapted_metrics.get("beam_power_loss_db"),
        "adapted_source_beam_power_loss_db_delta": _numeric_delta_from_values(
            adapted_metrics.get("beam_power_loss_db"),
            source_metrics.get("beam_power_loss_db"),
        ),
        "negative_transfer": _negative_transfer(adapted_top1, source_top1),
        "negative_transfer_metric": "top1" if _negative_transfer(adapted_top1, source_top1) is not None else None,
        "last_beam_top1": last_beam.get("top1"),
        "last_beam_top3": last_beam.get("top3"),
        "last_beam_avg_top1": last_beam.get("avg_top1"),
        "last_beam_avg_top3": last_beam.get("avg_top3"),
        "last_beam_available": last_beam.get("available"),
        "last_beam_baseline_type": "diagnostic",
        "last_beam_comparable_baseline": bool(primary_metrics.get("last_beam_comparable_baseline", False)),
        "power_metrics_unavailable_reason": primary_metrics.get("power_metrics_unavailable_reason"),
        "trainable_params": adaptation_metrics.get("trainable_params"),
        "total_params": adaptation_metrics.get("total_params"),
        "trainable_ratio": adaptation_metrics.get("trainable_ratio"),
        "adaptation_time_seconds": adaptation_metrics.get("adaptation_time_seconds"),
        "adaptation_time_per_epoch": adaptation_metrics.get("adaptation_time_per_epoch"),
        "source_training_duration_seconds": source_train_metrics.get("source_training_duration_seconds"),
        "prototype_generation_duration_seconds": source_train_metrics.get("prototype_generation_duration_seconds"),
        "prototype_coverage": adaptation_metrics.get("prototype_coverage"),
        "prototype_coverage_unavailable_reason": adaptation_metrics.get("prototype_coverage_unavailable_reason"),
        "prototype_status": adaptation_metrics.get("prototype_status") or source_train_metrics.get("prototype_status"),
        "prototype_skipped_reason": source_train_metrics.get("prototype_skipped_reason"),
        "prototype_confidence_mean": adaptation_metrics.get("prototype_confidence_mean"),
        "prototype_used_sample_count": adaptation_metrics.get("prototype_used_sample_count"),
        "prototype_loss_mean": _first_present(adaptation_metrics, "prototype_loss_mean", "prototype_loss"),
        "proto_type": adaptation_metrics.get("proto_type"),
        "label_budget": adaptation_metrics.get("label_budget", record.get("budget")),
        "target_labeled_subset_available": _bool_or_false(adaptation_metrics.get("target_labeled_subset_available")),
        "target_unlabeled_subset_available": _bool_or_false(adaptation_metrics.get("target_unlabeled_subset_available")),
        "sensitive_field_policy": adaptation_metrics.get("sensitive_field_policy", {}),
        "eligibility_status": adaptation_metrics.get("eligibility_status"),
        "used_target_oracle_fields": list(adaptation_metrics.get("used_target_oracle_fields", []) or []),
        "disabled_modalities": record.get("disabled_modalities") or primary_metrics.get("disabled_modalities") or adaptation_metrics.get("disabled_modalities"),
        "available_fields": primary_metrics.get("available_fields") or adaptation_metrics.get("available_fields") or source_train_metrics.get("available_fields"),
        "consumed_fields": _merge_consumed_fields(source_train_metrics, adaptation_metrics, primary_metrics),
        "target_oracle_usage_stage": adaptation_metrics.get("target_oracle_usage_stage", {}),
        "target_test_label_usage": primary_metrics.get("target_test_label_usage", adaptation_metrics.get("target_test_label_usage", "evaluation_only")),
        "used_target_labels": _bool_or_false(adaptation_metrics.get("used_target_labels")),
        "used_target_beam_for_training": _bool_or_false(adaptation_metrics.get("used_target_beam_for_training")),
        "used_target_beam_power_for_training": _bool_or_false(adaptation_metrics.get("used_target_beam_power_for_training")),
        "used_target_physical_label_for_training": _bool_or_false(adaptation_metrics.get("used_target_physical_label_for_training")),
        "target_physical_oracle_unused_reason": adaptation_metrics.get("target_physical_oracle_unused_reason"),
        "used_target_csi_for_training": _bool_or_false(adaptation_metrics.get("used_target_csi_for_training")),
        "used_target_path_params_for_training": _bool_or_false(adaptation_metrics.get("used_target_path_params_for_training")),
        "used_target_path_descriptor_for_training": _bool_or_false(adaptation_metrics.get("used_target_path_descriptor_for_training")),
        "used_target_path_label_for_training": _bool_or_false(adaptation_metrics.get("used_target_path_label_for_training")),
        "used_target_radio_label_for_training": _bool_or_false(adaptation_metrics.get("used_target_radio_label_for_training")),
        "radio_assignment_confidence_mean": adaptation_metrics.get("radio_assignment_confidence_mean"),
        "radio_assignment_used_sample_count": adaptation_metrics.get("radio_assignment_used_sample_count"),
        "path_assignment_confidence_mean": adaptation_metrics.get("path_assignment_confidence_mean"),
        "path_assignment_used_sample_count": adaptation_metrics.get("path_assignment_used_sample_count"),
        "target_private_initialized_count": adaptation_metrics.get("target_private_initialized_count"),
        "geometry_loss_coverage": primary_metrics.get("hist/geometry_consistency_coverage")
        or adaptation_metrics.get("geometry_consistency_coverage"),
    }
    row["method_family"] = "legacy_kd" if lineage["method_family"] == "legacy_kd" else _method_family(row)
    row["sensitive_field_usage"] = {
        key: row[key]
        for key in (
            "used_target_beam_for_training",
            "used_target_beam_power_for_training",
            "used_target_physical_label_for_training",
            "used_target_csi_for_training",
            "used_target_path_params_for_training",
            "used_target_path_descriptor_for_training",
            "used_target_path_label_for_training",
            "used_target_radio_label_for_training",
        )
    }
    eligibility = _row_eligibility(row, adaptation_metrics, primary_metrics)
    row["main_conclusion_eligible"] = eligibility["main_conclusion_eligible"]
    row["eligibility_status"] = eligibility["eligibility_status"]
    row["eligibility_reasons"] = eligibility["eligibility_reasons"]
    row["oracle_usage_summary"] = {
        "used_target_oracle_fields": row.get("used_target_oracle_fields", []),
        "target_oracle_usage_stage": row.get("target_oracle_usage_stage", {}),
        "target_test_label_usage": row.get("target_test_label_usage"),
    }
    row["eligibility_source_artifacts"] = {
        "metrics_path": row.get("metrics_path"),
        "adapt_log_path": _artifact(record, "target_adaptation.adapt_log_path"),
        "run_metadata_path": _artifact(record, "run_metadata_path"),
    }
    if record.get("status") == "failed":
        row["missing_reason"] = record.get("failure_reason")
    return row


def _method_family(row: Mapping[str, Any]) -> str:
    variant = str(row.get("variant"))
    if variant in {"v6_full_finetune", "full_finetune"}:
        return "full_finetuning_baseline"
    if variant in {"v6_radio_proto", "adapter_radio_proto"} or row.get("proto_type") == "radio_semantic":
        return "radio_semantic_prototype"
    if variant in {"v8_path_proto", "adapter_path_proto"} or row.get("proto_type") == "path":
        return "path_physical_prototype"
    if variant == "v7_shared_physical_private_residual":
        return "shared_physical_private_residual"
    if variant == "v9_input_conditioned_target_adaptation":
        return "input_conditioned_target_adaptation"
    if variant in {"v5_adapter_proto", "adapter_proto"}:
        return "coarse_prototype_baseline"
    return "source_or_adapter_baseline"


def _merge_consumed_fields(*sources: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        value = source.get("consumed_fields") if isinstance(source, Mapping) else None
        if isinstance(value, Mapping):
            merged.update(dict(value))
    return merged


def _lineage_summary(
    record: Mapping[str, Any],
    source_train_metrics: Mapping[str, Any],
    adaptation_metrics: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    merged = run_lineage_metadata(None, default_method_family="hist_beam_mainline")
    for source in (record, source_train_metrics, adaptation_metrics, primary_metrics):
        if not isinstance(source, Mapping):
            continue
        lineage = source.get("lineage") if isinstance(source.get("lineage"), Mapping) else source
        if bool(lineage.get("distillation_enabled", False)) or str(lineage.get("method_family", "")) == "legacy_kd":
            merged["distillation_enabled"] = True
            merged["method_family"] = "legacy_kd"
        for key in (
            "distillation_type",
            "teacher_checkpoint",
            "teacher_source",
            "student_model",
            "distillation_lifecycle",
            "baseline_role",
            "reproduction_scope",
            "main_conclusion_eligible",
        ):
            value = lineage.get(key)
            if value not in (None, ""):
                merged[key] = value
    if not merged["distillation_enabled"]:
        merged["method_family"] = "hist_beam_mainline"
        merged["main_conclusion_eligible"] = True
    else:
        merged["main_conclusion_eligible"] = False
        merged.setdefault("distillation_lifecycle", "legacy_kd")
    return merged


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _split_summary(
    record: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    adapted_metrics: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = []
    for mapping in (primary_metrics, adapted_metrics, source_metrics):
        if isinstance(mapping, Mapping):
            setup = mapping.get("prediction_setup")
            if isinstance(setup, Mapping):
                candidates.append(setup)
            candidates.append(mapping)
    candidates.append(record)
    for key in ("prediction_setup", "split_metadata", "split_protocol"):
        value = record.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    merged: dict[str, Any] = {}
    for candidate in candidates:
        extracted = _extract_split_summary(candidate)
        for key, value in extracted.items():
            if key not in merged and value is not None:
                merged[key] = value
    sidecar = merged.get("split_metadata")
    if isinstance(sidecar, Mapping) and "available" in sidecar:
        merged.setdefault("split_metadata_available", bool(sidecar.get("available")))
        merged.setdefault("fix_hint", sidecar.get("fix_hint") or sidecar.get("warning"))
    if "split_metadata_available" not in merged and merged.get("split_metadata_path"):
        merged["split_metadata_available"] = True
    strict = merged.get("strict_validation_eligible")
    if strict is True:
        merged["split_eligibility"] = "strict"
    elif strict is False:
        merged["split_eligibility"] = "ineligible"
    else:
        merged["split_eligibility"] = "unknown"
    return merged


def _extract_split_summary(candidate: Mapping[str, Any]) -> dict[str, Any]:
    split_payload = _preferred_split_payload(candidate)
    sidecar = candidate.get("split_metadata") if isinstance(candidate.get("split_metadata"), Mapping) else {}
    result = {
        "split_protocol": _first_non_none(candidate.get("split_protocol"), split_payload.get("split_protocol")),
        "split_strategy": _first_non_none(candidate.get("split_strategy"), split_payload.get("split_strategy")),
        "split_protocol_version": _first_non_none(
            candidate.get("split_protocol_version"),
            split_payload.get("split_protocol_version"),
        ),
        "split_metadata_path": _first_non_none(
            candidate.get("split_metadata_path"),
            split_payload.get("split_metadata_path"),
            sidecar.get("path"),
        ),
        "split_metadata_available": _first_non_none(
            candidate.get("split_metadata_available"),
            split_payload.get("split_metadata_available"),
            sidecar.get("available"),
        ),
        "strict_validation_eligible": _first_non_none(
            candidate.get("strict_validation_eligible"),
            split_payload.get("strict_validation_eligible"),
        ),
        "eligibility_reasons": _first_non_none(
            candidate.get("eligibility_reasons"),
            split_payload.get("eligibility_reasons"),
        ),
        "leakage_diagnostics": _first_non_none(
            candidate.get("leakage_diagnostics"),
            split_payload.get("leakage_diagnostics"),
        ),
        "split_seed": _first_non_none(candidate.get("split_seed"), split_payload.get("split_seed")),
        "split_sequence_count": _first_non_none(
            candidate.get("split_sequence_count"),
            split_payload.get("split_sequence_count"),
        ),
        "split_num_samples": _first_non_none(candidate.get("split_num_samples"), split_payload.get("split_num_samples")),
        "fix_hint": _first_non_none(candidate.get("fix_hint"), split_payload.get("fix_hint"), sidecar.get("warning")),
        "split_metadata": sidecar,
    }
    if result["eligibility_reasons"] is not None:
        result["eligibility_reasons"] = list(result["eligibility_reasons"] or [])
    return result


def _preferred_split_payload(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    splits = candidate.get("splits")
    if not isinstance(splits, Mapping):
        for split in ("target_test", "test", "validation", "val", "train"):
            payload = candidate.get(split)
            if isinstance(payload, Mapping):
                return payload
        return {}
    for split in ("target_test", "test", "validation", "val", "train"):
        payload = splits.get(split)
        if isinstance(payload, Mapping):
            return payload
    for payload in splits.values():
        if isinstance(payload, Mapping):
            return payload
    return {}


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _numeric_delta_from_values(lhs: Any, rhs: Any) -> float | None:
    if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)):
        return None
    return float(lhs) - float(rhs)


def _negative_transfer(adapted_top1: Any, source_top1: Any) -> bool | None:
    delta = _numeric_delta_from_values(adapted_top1, source_top1)
    return None if delta is None else bool(delta < 0.0)


def _bool_or_false(value: Any) -> bool:
    return bool(value) if value is not None else False


def _row_eligibility(
    row: Mapping[str, Any],
    adaptation_metrics: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return row_eligibility(row, adaptation_metrics, primary_metrics)


def _prototype_is_required(row: Mapping[str, Any]) -> bool:
    return prototype_is_required(row)


def _prototype_is_no_op(row: Mapping[str, Any]) -> bool:
    return prototype_is_no_op(row)


def _unique_reasons(reasons: list[Any]) -> list[str]:
    return unique_reasons(reasons)


def _reason_histogram(reason_lists: Any) -> dict[str, int]:
    return reason_histogram(reason_lists)


def _excluded_run_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return excluded_run_summary(row)


def _conclusion_source_artifacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return conclusion_source_artifacts(rows)


def _last_beam_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    baselines = metrics.get("degradation_baselines") if isinstance(metrics, Mapping) else None
    last = baselines.get("last_beam") if isinstance(baselines, Mapping) else None
    if not isinstance(last, Mapping):
        return {
            "available": False,
            "top1": None,
            "top3": None,
            "avg_top1": None,
            "avg_top3": None,
        }
    return {
        "available": bool(last.get("available", False)),
        "top1": last.get("top1"),
        "top3": last.get("top3"),
        "avg_top1": last.get("avg_top1"),
        "avg_top3": last.get("avg_top3"),
    }


def _cache_summary(record: Mapping[str, Any], source_train_metrics: Mapping[str, Any]) -> dict[str, Any]:
    throughput = source_train_metrics.get("throughput_config") if isinstance(source_train_metrics, Mapping) else None
    throughput = throughput if isinstance(throughput, Mapping) else {}
    return {
        "cache_policy": throughput.get("image_cache_policy") or throughput.get("cache_policy"),
        "lidar_cache_policy": throughput.get("lidar_cache_policy"),
        "lidar_cache_dir": throughput.get("lidar_cache_dir"),
        "num_workers": throughput.get("num_workers"),
        "cpu_threads": throughput.get("cpu_threads"),
    }


def _flatten_adaptation_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in diagnostics.items():
        short = str(key)
        if short.startswith("adaptation/"):
            short = short.split("/", 1)[1]
        if short == "prototype_loss":
            flat["prototype_loss_mean"] = value
        elif short == "prototype_used":
            flat["prototype_used_sample_count"] = value
        elif short == "prototype_status":
            flat["prototype_status"] = "effective" if float(value or 0.0) > 0 else "no_op"
        else:
            flat[short] = value
    if "prototype_status" not in flat and any(str(key).startswith("adaptation/prototype") for key in diagnostics):
        flat["prototype_status"] = "no_op"
    return flat


def _artifact(record: dict[str, Any], key: str) -> Any:
    return record.get("artifacts", {}).get(key)


def _metric(metrics: Mapping[str, Any], name: str) -> float | None:
    if not metrics:
        return None
    if name in metrics and isinstance(metrics[name], (int, float)):
        return float(metrics[name])
    mapping = {"top1": "val_top1_avg", "top3": "val_top3_avg", "top5": "val_top5_avg"}
    mapped = mapping.get(name)
    if mapped and isinstance(metrics.get(mapped), (int, float)):
        return float(metrics[mapped])
    topk = metrics.get("topk")
    if isinstance(topk, dict):
        k = name.removeprefix("top")
        values = topk.get(k) or topk.get(int(k)) if k.isdigit() else None
        if isinstance(values, list) and values:
            numeric = [float(value) for value in values if isinstance(value, (int, float))]
            return sum(numeric) / len(numeric) if numeric else None
    return None


def _compare_coarse_to_radio(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    coarse: dict[str, Any] | None,
    radio: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "v5_coarse_vs_v6_radio",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "v5_adapter_proto",
        "candidate_variant": "v6_radio_proto",
    }
    missing = _missing_comparison_inputs({"v5_adapter_proto": coarse, "v6_radio_proto": radio})
    radio_missing = _missing_radio_metrics(radio)
    if missing or radio_missing:
        return {**base, "status": "inconclusive", "missing": missing + radio_missing}
    deltas = _accuracy_deltas(radio, coarse)
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "radio_accuracy": radio.get("radio_semantic_accuracy"),
        "power": {
            "normalized_received_power": radio.get("normalized_received_power"),
            "beam_power_loss_db": radio.get("beam_power_loss_db"),
        },
        "prototype": {
            "coverage": radio.get("prototype_coverage"),
            "confidence_mean": radio.get("prototype_confidence_mean") or radio.get("radio_assignment_confidence_mean"),
        },
        "efficiency": _efficiency_summary(radio),
        "radio_prototype_better_than_coarse": _is_better(deltas),
    }


def _compare_radio_condition(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    off: dict[str, Any] | None,
    on: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "radio_condition_off_vs_on",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "adapter_radio_proto",
        "candidate_variant": "v6_radio_proto",
    }
    missing = _missing_comparison_inputs({"adapter_radio_proto": off, "v6_radio_proto": on})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(on, off)
    prediction_delta = 0 if all(value == 0 for value in deltas.values() if value is not None) else None
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "radio_condition_prediction_delta": prediction_delta,
        "radio_assignment": {
            "off_confidence_mean": off.get("radio_assignment_confidence_mean"),
            "on_confidence_mean": on.get("radio_assignment_confidence_mean"),
        },
    }


def _compare_radio_to_path(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    radio: dict[str, Any] | None,
    path: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "v6_radio_vs_v8_path",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "v6_radio_proto",
        "candidate_variant": "v8_path_proto",
    }
    missing = _missing_comparison_inputs({"v6_radio_proto": radio, "v8_path_proto": path})
    path_missing = _missing_path_metrics(path)
    if missing or path_missing:
        return {**base, "status": "inconclusive", "missing": missing + path_missing}
    deltas = _accuracy_deltas(path, radio)
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "path_accuracy": path.get("path_semantic_accuracy"),
        "path_descriptor_mse": path.get("path_descriptor_regression_mse"),
        "prototype": {
            "coverage": path.get("prototype_coverage"),
            "confidence_mean": path.get("prototype_confidence_mean") or path.get("path_assignment_confidence_mean"),
        },
        "efficiency": _efficiency_summary(path),
        "path_prototype_better_than_radio": _is_better(deltas),
    }


def _compare_path_to_full(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    path: dict[str, Any] | None,
    full: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "v8_path_vs_full_finetune",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "v6_full_finetune",
        "candidate_variant": "v8_path_proto",
    }
    missing = _missing_comparison_inputs({"v8_path_proto": path, "v6_full_finetune": full})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(path, full)
    trainable_delta = _numeric_delta(path, full, "trainable_ratio")
    time_delta = _numeric_delta(path, full, "adaptation_time_seconds")
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "trainable_ratio_delta": trainable_delta,
        "adaptation_time_seconds_delta": time_delta,
        "path_prototype_better_than_full_finetune": _is_better(deltas)
        and (trainable_delta is None or trainable_delta <= 0)
        and (time_delta is None or time_delta <= 0),
    }


def _compare_path_condition(
    *,
    target_scene: Any,
    budget: Any,
    seed: Any,
    off: dict[str, Any] | None,
    on: dict[str, Any] | None,
) -> dict[str, Any]:
    base = {
        "comparison": "path_condition_off_vs_on",
        "target_scene": target_scene,
        "budget": budget,
        "seed": seed,
        "baseline_variant": "adapter_path_proto",
        "candidate_variant": "v8_path_proto",
    }
    missing = _missing_comparison_inputs({"adapter_path_proto": off, "v8_path_proto": on})
    if missing:
        return {**base, "status": "inconclusive", "missing": missing}
    deltas = _accuracy_deltas(on, off)
    improves = _is_better(deltas)
    return {
        **base,
        "status": "complete",
        "accuracy_deltas": deltas,
        "path_condition_improved": improves,
        "diagnosis": None if improves else "path_prototype_may_be_more_effective_as_adaptation_anchor_than_beam_head_condition",
        "path_assignment": {
            "off_confidence_mean": off.get("path_assignment_confidence_mean"),
            "on_confidence_mean": on.get("path_assignment_confidence_mean"),
        },
    }


def _missing_radio_metrics(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if row is None:
        return []
    missing = []
    if row.get("radio_semantic_accuracy") is None:
        missing.append(
            {
                "variant": row.get("variant"),
                "reason": row.get("radio_metrics_unavailable_reason") or "radio_metrics_missing",
                "run_path": row.get("metrics_path"),
            }
        )
    if row.get("normalized_received_power") is None and row.get("beam_power_loss_db") is None:
        missing.append(
            {
                "variant": row.get("variant"),
                "reason": row.get("power_metrics_unavailable_reason") or "power_metrics_missing",
                "run_path": row.get("metrics_path"),
            }
        )
    return missing


def _missing_path_metrics(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if row is None:
        return []
    missing = []
    if row.get("path_semantic_accuracy") is None:
        missing.append(
            {
                "variant": row.get("variant"),
                "reason": row.get("path_metrics_unavailable_reason") or "path_metrics_missing",
                "run_path": row.get("metrics_path"),
            }
        )
    return missing


def _missing_comparison_inputs(items: Mapping[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    missing = []
    for variant, row in items.items():
        if row is None:
            missing.append({"variant": variant, "reason": "missing_run"})
            continue
        if row.get("run_status") != "completed":
            missing.append({"variant": variant, "reason": row.get("failure_reason") or "run_not_completed"})
            continue
        if bool(row.get("main_conclusion_eligible", True)) is False:
            missing.append(
                {
                    "variant": variant,
                    "reason": "run_excluded_from_main_conclusion",
                    "eligibility_reasons": list(row.get("eligibility_reasons", []) or []),
                    "run_path": row.get("metrics_path"),
                }
            )
            continue
        if all(row.get(metric) is None for metric in ("top1", "top3", "top5", "coarse_accuracy", "fine_accuracy")):
            missing.append({"variant": variant, "reason": "metrics_missing"})
    return missing


def _accuracy_deltas(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        metric: _numeric_delta(candidate, baseline, metric)
        for metric in ("top1", "top3", "top5", "coarse_accuracy", "fine_accuracy", "radio_semantic_accuracy", "path_semantic_accuracy")
    }


def _numeric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any], metric: str) -> float | None:
    lhs = candidate.get(metric)
    rhs = baseline.get(metric)
    if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)):
        return None
    return float(lhs) - float(rhs)


def _is_better(deltas: Mapping[str, float | None]) -> bool:
    available = [value for value in deltas.values() if value is not None]
    return bool(available) and all(value >= 0 for value in available) and any(value > 0 for value in available)


def _efficiency_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trainable_ratio": row.get("trainable_ratio"),
        "adaptation_time_seconds": row.get("adaptation_time_seconds"),
        "adaptation_time_per_epoch": row.get("adaptation_time_per_epoch"),
    }


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "conclusion_source_artifacts",
    "excluded_run_summary",
    "prototype_is_no_op",
    "prototype_is_required",
    "reason_histogram",
    "row_eligibility",
    "unique_reasons",
    "write_quick_validation_conclusion",
]
