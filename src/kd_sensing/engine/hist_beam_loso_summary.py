from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def row_eligibility(
    row: Mapping[str, Any],
    adaptation_metrics: Mapping[str, Any],
    primary_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = list(adaptation_metrics.get("eligibility_reasons", []) or [])
    if bool(adaptation_metrics.get("main_conclusion_eligible", True)) is False:
        reasons.append("run_marked_ineligible")
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
    reasons.extend(_split_eligibility_reasons(row))

    unique = unique_reasons(reasons)
    return {
        "main_conclusion_eligible": len(unique) == 0,
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
    return reasons


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
        "run_id": row.get("run_id"),
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


__all__ = [
    "conclusion_source_artifacts",
    "excluded_run_summary",
    "prototype_is_no_op",
    "prototype_is_required",
    "reason_histogram",
    "row_eligibility",
    "unique_reasons",
]
