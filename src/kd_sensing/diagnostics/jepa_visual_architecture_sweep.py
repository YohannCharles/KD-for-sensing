import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/jepa_visual_architecture_sweep")

REQUIRED_FAMILIES = {
    "baseline",
    "patch_granularity",
    "overlap_tokenizer",
    "conv_stem_tokenizer",
    "local_token_mixing",
    "cnn_tokens",
    "multi_scale_tokens",
    "frame_embedding_anchor",
    "pooler_core_ablation",
    "non_transformer_control",
}

STRICT_COMPARABILITY_FIELDS = (
    "split",
    "scene_set",
    "seed",
    "history_window",
    "gps_input_source_window",
    "prediction_horizon",
    "beam_label_space",
    "metric_profile",
    "distance_metric",
    "normalization_artifact",
    "difficulty_digest",
    "output_root",
)

SUMMARY_FIELDS = (
    "variant_id",
    "family",
    "run_tier",
    "evidence_scope",
    "checkpoint_policy",
    "token_count",
    "params_trainable",
    "compute_proxy",
    "top1",
    "top3",
    "top5",
    "dba",
    "adjacent_beam_error",
    "diagnostics_status",
    "claim_eligible",
    "claim_gate_reason",
)


class SweepManifestError(ValueError):
    """Raised when a JEPA visual architecture sweep manifest is incomplete."""


def load_sweep_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise SweepManifestError(f"Sweep manifest must be a mapping, got {type(payload).__name__}.")
    return validate_sweep_manifest(payload)


def validate_sweep_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(manifest)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise SweepManifestError("Sweep manifest requires a non-empty candidates list.")
    families = {str(candidate.get("family")) for candidate in candidates if isinstance(candidate, Mapping)}
    missing = sorted(REQUIRED_FAMILIES - families)
    if missing:
        raise SweepManifestError(f"Sweep manifest is missing required families: {missing}.")
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise SweepManifestError(f"Sweep candidate {index} must be a mapping.")
        variant_id = str(candidate.get("variant_id") or "")
        if not variant_id:
            raise SweepManifestError(f"Sweep candidate {index} is missing variant_id.")
        if variant_id in seen:
            raise SweepManifestError(f"Sweep manifest has duplicate variant_id {variant_id!r}.")
        seen.add(variant_id)
        for field in ("family", "visual_encoder", "pooler", "checkpoint_policy", "run_tier"):
            if field not in candidate:
                raise SweepManifestError(f"Sweep candidate {variant_id!r} is missing {field}.")
        visual_encoder = candidate.get("visual_encoder")
        pooler = candidate.get("pooler")
        if not isinstance(visual_encoder, Mapping) or "type" not in visual_encoder:
            raise SweepManifestError(f"Sweep candidate {variant_id!r} requires visual_encoder.type.")
        if not isinstance(pooler, Mapping) or "type" not in pooler:
            raise SweepManifestError(f"Sweep candidate {variant_id!r} requires pooler.type.")
    output_root = str(payload.get("output_root", DEFAULT_OUTPUT_ROOT))
    if not output_root.startswith("outputs/"):
        raise SweepManifestError("Sweep output_root must stay under ignored outputs/.")
    payload.setdefault("output_root", output_root)
    payload.setdefault("strict_comparability_fields", list(STRICT_COMPARABILITY_FIELDS))
    return payload


def strict_comparability_gate(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    baseline_fields = {field: baseline.get(field) for field in STRICT_COMPARABILITY_FIELDS} if baseline else None
    gated: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        reasons: list[str] = []
        if str(row.get("run_tier", "")).lower() != "strict":
            reasons.append("non_strict_run_tier")
        if str(row.get("evidence_scope", "")).lower() not in {"strict", "primary"}:
            reasons.append("non_primary_evidence_scope")
        if row.get("checkpoint_policy") in {"fresh_stage1_required", "supervised_only_anchor"} and not row.get(
            "stage1_checkpoint"
        ):
            reasons.append("missing_matching_stage1_checkpoint")
        missing = [field for field in STRICT_COMPARABILITY_FIELDS if row.get(field) in (None, "")]
        if missing:
            reasons.append("missing_strict_fields:" + ",".join(missing))
        if baseline_fields is not None:
            mismatched = [
                field
                for field, expected in baseline_fields.items()
                if row.get(field) not in (expected, None, "") and expected not in (None, "")
            ]
            if mismatched:
                reasons.append("strict_field_mismatch:" + ",".join(mismatched))
        row["claim_eligible"] = not reasons
        row["claim_gate_reason"] = ";".join(reasons) if reasons else "eligible"
        gated.append(row)
    return gated


def write_sweep_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    stem: str = "summary",
) -> dict[str, str]:
    root = Path(output_root)
    if not str(root).startswith("outputs/"):
        raise SweepManifestError("Sweep summary output_root must stay under ignored outputs/.")
    root.mkdir(parents=True, exist_ok=True)
    gated_rows = [dict(row) for row in rows]
    json_path = root / f"{stem}.json"
    csv_path = root / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(gated_rows, f, indent=2)
    fields = list(SUMMARY_FIELDS)
    for row in gated_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in gated_rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return {"json": str(json_path), "csv": str(csv_path)}


def summary_row_from_result(
    *,
    candidate: Mapping[str, Any],
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    diagnostics = dict(diagnostics or {})
    provenance = dict(provenance or {})
    visual_encoder = candidate.get("visual_encoder") if isinstance(candidate.get("visual_encoder"), Mapping) else {}
    return {
        "variant_id": candidate.get("variant_id"),
        "family": candidate.get("family"),
        "run_tier": candidate.get("run_tier"),
        "evidence_scope": candidate.get("evidence_scope", candidate.get("run_tier")),
        "checkpoint_policy": candidate.get("checkpoint_policy"),
        "token_count": candidate.get("token_count") or visual_encoder.get("token_count"),
        "params_trainable": metrics.get("params_trainable"),
        "compute_proxy": metrics.get("compute_proxy"),
        "top1": metrics.get("top1", metrics.get("top_1")),
        "top3": metrics.get("top3", metrics.get("top_3")),
        "top5": metrics.get("top5", metrics.get("top_5")),
        "dba": metrics.get("dba"),
        "adjacent_beam_error": metrics.get("adjacent_beam_error"),
        "diagnostics_status": diagnostics.get("status", "available" if diagnostics else "missing"),
        "attention_entropy": diagnostics.get("attention_entropy"),
        "attention_peakiness": diagnostics.get("attention_peakiness"),
        "branch_weights": diagnostics.get("branch_weights"),
        "gate_weights": diagnostics.get("gate_weights"),
        "wrong_gps_top1": metrics.get("wrong_gps_top1"),
        "counterfactual_gps_top1": metrics.get("counterfactual_gps_top1"),
        "condition_metrics": metrics.get("condition_metrics"),
        "command": provenance.get("command"),
        "metrics_path": provenance.get("metrics_path"),
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "REQUIRED_FAMILIES",
    "STRICT_COMPARABILITY_FIELDS",
    "SweepManifestError",
    "load_sweep_manifest",
    "strict_comparability_gate",
    "summary_row_from_result",
    "validate_sweep_manifest",
    "write_sweep_summary",
]
