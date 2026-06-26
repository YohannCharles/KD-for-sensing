from __future__ import annotations

import copy
import csv
import datetime as dt
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "wcl2025_missing_modality_reproduction.v1"
DEFAULT_OUTPUT_ROOT = Path("outputs/analysis/wcl2025_missing_modality_reproduction")
LOCAL_SUBSTITUTE_CONFIG = "configs/fusion/experiments/wcl2025_missing_modality/local_substitute.yaml"
CLAIM_STATUSES = (
    "official_reproduction",
    "local_substitute",
    "blocked",
    "pending",
    "unavailable",
    "not_comparable",
    "external_reference",
)
STRICT_COMPARABILITY_FIELDS = (
    "split",
    "scene_set",
    "label_space",
    "metric_profile",
    "sample_count",
    "seed",
    "difficulty_digest",
)
DEFAULT_MODALITIES = ("image", "radar", "gps", "lidar", "mmwave")
PAPER_KEY_MISSING_CONDITION_IDS = {
    "clean",
    "missing_image",
    "missing_radar",
    "missing_lidar",
    "missing_gps",
    "missing_image_lidar",
    "missing_radar_lidar",
    "missing_image_radar_lidar",
}

PAPER_METADATA = {
    "title": "Robust Multimodal Beam Prediction With Missing Modality",
    "venue": "IEEE Wireless Communications Letters",
    "year": 2025,
    "doi": "10.1109/LWC.2025.3591611",
    "ieee_document": "11089951",
    "sources": [
        "https://ieeexplore.ieee.org/document/11089951",
        "https://dblp.org/rec/journals/wcl/YaoMWSW25.html",
    ],
}


def build_source_audit_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    generated_at: str | None = None,
    command_args: Sequence[str] = (),
    official_code_url: str | None = None,
    source_commit: str | None = None,
    checkpoint_uri: str | None = None,
    training_recipe_available: bool = False,
    dataset_protocol_available: bool = False,
    metric_profile_available: bool = False,
) -> dict[str, Any]:
    """Build an auditable dry-run manifest without downloading external artifacts."""

    code_availability = "available" if official_code_url else "unavailable"
    checkpoint_availability = "available" if checkpoint_uri else "unavailable"
    recipe_availability = "available" if training_recipe_available else "pending"
    dataset_availability = "available" if dataset_protocol_available else "pending"
    metric_availability = "available" if metric_profile_available else "pending"
    official_status = _official_status(
        code_availability=code_availability,
        source_commit=source_commit,
        checkpoint_availability=checkpoint_availability,
        recipe_availability=recipe_availability,
        dataset_availability=dataset_availability,
        metric_availability=metric_availability,
    )
    local_model = build_local_substitute_model_config()
    conditions = default_missing_conditions(local_model["modalities"])

    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "generated_at": generated_at or _utc_now_iso(),
            "command": list(command_args),
            "output_root": str(Path(output_root)),
            "dry_run": True,
        },
        "paper": dict(PAPER_METADATA),
        "source_audit": {
            "code": {
                "url": official_code_url,
                "availability": code_availability,
                "source_commit": source_commit or "unavailable",
                "license": "unavailable" if not official_code_url else "pending",
            },
            "checkpoint": {
                "uri": checkpoint_uri,
                "availability": checkpoint_availability,
                "provenance": "unavailable" if not checkpoint_uri else "pending",
            },
            "dataset": {
                "name": "not disclosed in public metadata",
                "availability": dataset_availability,
                "local_substitute": "DeepSense6G-compatible multimodal sequence config",
                "split": "pending official protocol; config-declared local substitute",
            },
            "modalities": list(DEFAULT_MODALITIES),
            "split": {
                "official": "pending",
                "local_substitute": "config-declared",
            },
            "metric_profile": {
                "official": metric_availability,
                "local_substitute": "Top-K/DBA or beam distance condition summary",
            },
            "training_recipe": {
                "availability": recipe_availability,
                "local_substitute_config": LOCAL_SUBSTITUTE_CONFIG,
            },
            "missing_details": [
                "official code URL not found in consulted public metadata",
                "official checkpoint not found in consulted public metadata",
                "official exact split, seed, and metric profile remain pending",
            ],
        },
        "branches": {
            "official_code": {
                "status": official_status,
                "claim_status": claim_status_for_branch("official_code", official_status=official_status),
                "source_commit": source_commit,
                "checkpoint_provenance": checkpoint_uri,
            },
            "local_substitute": {
                "status": "available",
                "claim_status": "local_substitute",
                "config": LOCAL_SUBSTITUTE_CONFIG,
                "model": local_model,
                "deviation": [
                    "uses local modular_sequence components instead of unreleased official code",
                    "uses config-declared DeepSense6G-compatible protocol until official split is available",
                    "does not claim official WCL 2025 metrics",
                ],
            },
        },
        "missing_modality_conditions": conditions,
        "claim_status": claim_status_for_branch(
            "local_substitute",
            official_status=official_status,
            local_substitute_available=True,
        ),
    }


def run_source_audit_dry_run(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: str | Path | None = None,
    command_args: Sequence[str] = (),
    **manifest_kwargs: Any,
) -> dict[str, Any]:
    manifest = build_source_audit_manifest(
        output_root=output_root,
        command_args=command_args,
        **manifest_kwargs,
    )
    target = write_source_audit_manifest(manifest, output_path=manifest_path)
    manifest.setdefault("metadata", {})["manifest_path"] = str(target)
    return manifest


def write_source_audit_manifest(manifest: Mapping[str, Any], *, output_path: str | Path | None = None) -> Path:
    root = Path(manifest.get("metadata", {}).get("output_root") or DEFAULT_OUTPUT_ROOT)
    target = Path(output_path) if output_path is not None else root / "source_audit_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(dict(manifest))
    payload.setdefault("metadata", {})["manifest_path"] = str(target)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return target


def build_local_substitute_model_config(
    *,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    feature_size: int = 64,
    d_model: int = 64,
    num_classes: int = 64,
    num_pred: int = 1,
    fusion_type: str = "token_transformer",
) -> dict[str, Any]:
    enabled = [str(item) for item in modalities]
    heads = _num_heads(int(d_model))
    encoders = {
        modality: _encoder_config(modality, feature_size=int(feature_size))
        for modality in enabled
    }
    return {
        "type": "modular_sequence",
        "modalities": enabled,
        "feature_size": int(feature_size),
        "d_model": int(d_model),
        "num_classes": int(num_classes),
        "num_pred": int(num_pred),
        "encoders": encoders,
        "representation_core": {
            "type": str(fusion_type),
            "d_model": int(d_model),
            "num_heads": heads,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "heads": {
            "beam": {
                "type": "beam_head",
                "num_classes": int(num_classes),
                "dropout": 0.1,
            }
        },
        "paper_metadata": {
            "model_group": "RMBP-MM",
            "paper_alignment": "paper_aligned_local_substitute",
            "missing_modality_strategy": "zero_imputation_with_modality_dropout_training",
            "fusion_type": str(fusion_type),
            "enabled_modalities": enabled,
            "deviation": [
                "official implementation unavailable",
                "uses modular_sequence encoders and token_transformer fusion",
                "does not implement the paper-specific imputation and channel-attention modules",
            ],
        },
    }


def default_missing_conditions(modalities: Sequence[str] = DEFAULT_MODALITIES) -> list[dict[str, Any]]:
    enabled = [str(item) for item in modalities]
    rows: list[dict[str, Any]] = [{"condition_id": "clean", "affected_modalities": [], "paper_key_condition": True}]
    rows.extend(
        {
            "condition_id": f"missing_{modality}",
            "affected_modalities": [modality],
            "paper_key_condition": f"missing_{modality}" in PAPER_KEY_MISSING_CONDITION_IDS,
        }
        for modality in enabled
    )
    for combo in (("image", "lidar"), ("radar", "lidar"), ("image", "radar", "lidar")):
        if all(modality in enabled for modality in combo):
            condition_id = "missing_" + "_".join(combo)
            rows.append(
                {
                    "condition_id": condition_id,
                    "affected_modalities": list(combo),
                    "paper_key_condition": condition_id in PAPER_KEY_MISSING_CONDITION_IDS,
                }
            )
    return rows


def apply_missing_modality_condition(batch: Mapping[str, Any], affected_modalities: Sequence[str]) -> dict[str, Any]:
    """Return a zero-imputed batch for enabled missing-modality conditions."""

    output = dict(batch)
    for modality in affected_modalities:
        key = f"{modality}_batch"
        tensor = output.get(key)
        if tensor is None:
            continue
        zeroed = tensor.clone()
        zeroed.zero_()
        output[key] = zeroed
        if hasattr(tensor, "shape") and len(tensor.shape) >= 2:
            try:
                import torch

                output[f"{modality}_valid_mask"] = torch.zeros(
                    tuple(tensor.shape[:2]),
                    dtype=torch.bool,
                    device=tensor.device,
                )
            except Exception:
                pass
    return output


def build_condition_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    strict_protocol: Mapping[str, Any] | None = None,
    branch: str = "local_substitute",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    strict = dict(strict_protocol or {})
    normalized = [
        _condition_row(row, strict_protocol=strict, branch=branch, provenance=provenance)
        for row in rows
    ]
    strict_rows = [row for row in normalized if row["eligible_for_strict_ranking"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "branch": branch,
        "strict_protocol": strict,
        "conditions": normalized,
        "strict_ranking_rows": strict_rows,
        "summary": {
            "condition_count": len(normalized),
            "strict_ranking_count": len(strict_rows),
            "condition_types": sorted({row["condition_type"] for row in normalized}),
            "metric_means": _metric_means(normalized),
        },
    }


def write_condition_summary(
    summary: Mapping[str, Any],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, str]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "condition_summary.json"
    csv_path = root / "condition_metrics.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    rows = list(summary.get("conditions", []))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "condition_id",
            "condition_type",
            "affected_modalities",
            "sample_count",
            "split",
            "seed",
            "metric_profile",
            "claim_status",
            "comparability_status",
            "eligible_for_strict_ranking",
            "metrics",
            "provenance",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "condition_id": row.get("condition_id"),
                    "condition_type": row.get("condition_type"),
                    "affected_modalities": json.dumps(row.get("affected_modalities", []), sort_keys=True),
                    "sample_count": row.get("sample_count"),
                    "split": row.get("split"),
                    "seed": row.get("seed"),
                    "metric_profile": row.get("metric_profile"),
                    "claim_status": row.get("claim_status"),
                    "comparability_status": row.get("comparability_status"),
                    "eligible_for_strict_ranking": row.get("eligible_for_strict_ranking"),
                    "metrics": json.dumps(row.get("metrics", {}), sort_keys=True, default=_json_default),
                    "provenance": json.dumps(row.get("provenance", {}), sort_keys=True, default=_json_default),
                }
            )
    return {"json": str(json_path), "csv": str(csv_path)}


def claim_status_for_branch(
    branch: str,
    *,
    official_status: str = "blocked",
    local_substitute_available: bool = False,
    strict_mismatches: Sequence[Mapping[str, Any]] = (),
) -> str:
    if strict_mismatches:
        return "not_comparable"
    if branch == "official_code":
        return "official_reproduction" if official_status == "official_reproduction" else official_status
    if branch == "local_substitute":
        return "local_substitute" if local_substitute_available or official_status != "official_reproduction" else "pending"
    if branch == "external_reference":
        return "external_reference"
    return "pending"


def _official_status(
    *,
    code_availability: str,
    source_commit: str | None,
    checkpoint_availability: str,
    recipe_availability: str,
    dataset_availability: str,
    metric_availability: str,
) -> str:
    values = {
        "code": code_availability,
        "source_commit": "available" if source_commit else "unavailable",
        "checkpoint": checkpoint_availability,
        "training_recipe": recipe_availability,
        "dataset_protocol": dataset_availability,
        "metric_profile": metric_availability,
    }
    if all(value == "available" for value in values.values()):
        return "official_reproduction"
    if any(value == "pending" for value in values.values()) and values["code"] == "available":
        return "pending"
    return "blocked"


def _condition_row(
    row: Mapping[str, Any],
    *,
    strict_protocol: Mapping[str, Any],
    branch: str,
    provenance: Mapping[str, Any] | None,
) -> dict[str, Any]:
    affected = _as_list(row.get("affected_modalities", row.get("missing_modalities", [])))
    condition_id = str(row.get("condition_id") or _condition_id(affected))
    metrics = {
        key: row[key]
        for key in ("top1", "top3", "top5", "dba", "beam_distance", "mean_beam_distance")
        if key in row
    }
    payload = {
        "condition_id": condition_id,
        "affected_modalities": affected,
        "condition_type": _condition_type(affected),
        "paper_key_condition": bool(row.get("paper_key_condition", condition_id in PAPER_KEY_MISSING_CONDITION_IDS)),
        "metrics": metrics,
        "sample_count": int(row.get("sample_count", 0)),
        "split": row.get("split"),
        "scene_set": row.get("scene_set"),
        "label_space": row.get("label_space"),
        "metric_profile": row.get("metric_profile"),
        "seed": row.get("seed"),
        "difficulty_digest": row.get("difficulty_digest"),
        "provenance": dict(row.get("provenance") or provenance or {"branch": branch}),
    }
    mismatches = _strict_mismatches(payload, strict_protocol)
    payload["strict_mismatches"] = mismatches
    payload["comparability_status"] = "strict_comparable" if not mismatches else "not_comparable"
    payload["eligible_for_strict_ranking"] = not mismatches
    payload["claim_status"] = claim_status_for_branch(
        branch,
        official_status=str(row.get("official_status", "blocked")),
        local_substitute_available=branch == "local_substitute",
        strict_mismatches=mismatches,
    )
    return payload


def _strict_mismatches(row: Mapping[str, Any], strict_protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    for field in STRICT_COMPARABILITY_FIELDS:
        if field not in strict_protocol:
            continue
        expected = strict_protocol[field]
        actual = row.get(field)
        if _stable(expected) != _stable(actual):
            mismatches.append({"field": field, "expected": expected, "actual": actual})
    return mismatches


def _metric_means(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    means: dict[str, float] = {}
    for key in ("top1", "top3", "top5", "dba", "beam_distance", "mean_beam_distance"):
        values = [float(row["metrics"][key]) for row in rows if key in row.get("metrics", {})]
        if values:
            means[key] = float(mean(values))
    return means


def _encoder_config(modality: str, *, feature_size: int) -> dict[str, Any]:
    if modality == "image":
        return {
            "type": "resnet18_imagenet_rgb",
            "output_dim": feature_size,
            "pretrained": False,
            "weights": None,
            "freeze_backbone": True,
        }
    return {
        "radar": {"type": "radar_cnn", "output_dim": feature_size},
        "gps": {"type": "gps_mlp", "output_dim": feature_size},
        "lidar": {"type": "lidar_cnn", "output_dim": feature_size},
        "mmwave": {"type": "mmwave_mlp", "output_dim": feature_size},
        "csi": {"type": "pilot_dual_view_csi", "output_dim": feature_size},
    }[modality]


def _num_heads(d_model: int) -> int:
    for candidate in (8, 4, 2):
        if d_model % candidate == 0:
            return candidate
    return 1


def _condition_id(affected: Sequence[str]) -> str:
    return "clean" if not affected else "missing_" + "_".join(affected)


def _condition_type(affected: Sequence[str]) -> str:
    if not affected:
        return "clean"
    return "single_modality_missing" if len(affected) == 1 else "multi_modality_missing"


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _stable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)
