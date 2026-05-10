#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.data.scenes import scene_slug_from_config  # noqa: E402
from kd_sensing.diagnostics.communication_state_features import (  # noqa: E402
    DEFAULT_BUCKET_FEATURES,
    assign_buckets,
    communication_state_feature_records,
    fit_bucket_thresholds,
)
from kd_sensing.diagnostics.conditional_utility import (  # noqa: E402
    DEFAULT_ORACLE_CANDIDATES,
    WEAK_MODALITIES,
    aggregate_subset_metrics,
    build_conditional_utility_summary,
    compute_bucket_summary,
    compute_marginal_deltas,
    compute_subset_oracle,
    compute_teacher_complementarity,
    records_from_logits,
    write_json,
    write_table,
)
from kd_sensing.distillation.teacher_ensemble import build_g2d_teacher_ensemble  # noqa: E402
from kd_sensing.engine.batch import (  # noqa: E402
    forward_model,
    normalize_batch,
    prepare_fusion_inputs,
    prepare_labels,
)
from kd_sensing.engine.data_factory import build_dataloader, build_dataset, prepare_lidar_normalizer  # noqa: E402
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots  # noqa: E402
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts  # noqa: E402
from kd_sensing.engine.optim import build_device, build_model, build_task_criterion  # noqa: E402
from kd_sensing.engine.run_metadata import dataset_run_metadata, throughput_run_metadata  # noqa: E402
from kd_sensing.engine.runtime import autocast_context, resolve_amp_settings, transfer_non_blocking  # noqa: E402
from kd_sensing.evaluation.subset_specs import (  # noqa: E402
    CONDITIONAL_UTILITY_SUBSET_NAMES,
    resolve_conditional_utility_subset,
    subset_metadata,
)
from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities  # noqa: E402
from kd_sensing.utils.artifact_registry import resolve_evaluation_checkpoint  # noqa: E402
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state  # noqa: E402
from kd_sensing.utils.paths import resolve_path  # noqa: E402
from kd_sensing.utils.seed import set_seed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Conditional Utility Audit for a MARF fusion checkpoint.")
    parser.add_argument("--config", "-c", required=True, help="Audit YAML config.")
    parser.add_argument("--weights", help="Override checkpoint path.")
    parser.add_argument("--output-dir", help="Override conditional_utility output directory.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.override)
    result = run_conditional_utility_audit(cfg, weights=args.weights, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
    return result


def run_conditional_utility_audit(
    cfg: dict[str, Any],
    *,
    weights: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    set_seed(cfg.get("experiment", {}).get("seed", 0))
    cfg = deepcopy(cfg)
    cfg.setdefault("data", {}).setdefault("dataset", {})["return_metadata"] = True
    audit_cfg = cfg.get("conditional_utility", {})
    if not isinstance(audit_cfg, dict):
        audit_cfg = {}
    device = build_device(cfg)
    checkpoint_arg = weights or audit_cfg.get("checkpoint_path") or audit_cfg.get("weights")
    checkpoint_resolution = resolve_evaluation_checkpoint(cfg, str(checkpoint_arg) if checkpoint_arg else None)
    if checkpoint_resolution.path is None or not checkpoint_resolution.path.exists():
        raise FileNotFoundError(
            "Conditional Utility Audit requires a MARF checkpoint. "
            f"Resolution: {checkpoint_resolution.to_dict()}"
        )

    dataset_kwargs = _normalization_dataset_kwargs(cfg, checkpoint_resolution.metadata, checkpoint_resolution.path)
    dataset = build_dataset(cfg, "test", **dataset_kwargs)
    dataloader = build_dataloader(dataset, cfg["data"]["dataloader"], split="test")
    model = build_model(cfg["model"]["student"]).to(device)
    if cfg.get("experiment", {}).get("task", "image") != "fusion" or not getattr(
        model, "supports_force_modality_mask", False
    ):
        raise ValueError(
            "Conditional Utility Audit requires a fusion model with force_modality_mask support. "
            f"Got task={cfg.get('experiment', {}).get('task')} model={type(model).__name__}."
        )
    load_result = load_model_state(
        checkpoint_resolution.path,
        model,
        role="conditional_utility_audit",
        map_location=device,
        strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
    )
    checkpoint_load = checkpoint_load_summary(load_result)
    model.eval()
    criterion = build_task_criterion(cfg)
    _ = criterion

    model_modalities = _model_modalities(cfg)
    subset_specs = _resolve_subset_specs(audit_cfg, model_modalities)
    output_path = _resolve_output_dir(cfg, audit_cfg, output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_pred = int(cfg.get("model", {}).get("num_pred", 3))
    num_classes = int(cfg.get("model", {}).get("num_classes", 64))
    seq_length = int(cfg.get("model", {}).get("seq_length_student", 8))
    downsample_ratio = int(cfg.get("model", {}).get("downsample_ratio", 1))
    dba_delta = float(cfg.get("evaluation", {}).get("dba_delta", audit_cfg.get("dba_delta", 5)))
    horizon_names = list(audit_cfg.get("horizons") or [f"t+{idx + 1}" for idx in range(num_pred)])
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)

    teacher_ensemble = None
    dump_teacher = bool(audit_cfg.get("dump_teacher_predictions", True))
    if dump_teacher:
        teacher_ensemble = _build_teacher_ensemble_from_audit_config(cfg, audit_cfg, device, model_modalities)

    subset_records: list[dict[str, Any]] = []
    teacher_records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    sample_offset = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = normalize_batch(batch)
            labels = prepare_labels(
                batch,
                num_pred=num_pred,
                downsample_ratio=downsample_ratio,
                device=device,
                non_blocking=non_blocking,
            )
            metadata = batch.get("metadata")
            feature_records.extend(
                communication_state_feature_records(
                    batch,
                    labels=labels.detach().cpu(),
                    metadata=metadata,
                    horizon_names=horizon_names,
                    dataset_index_offset=sample_offset,
                )
            )
            with autocast_context(amp_enabled, device, amp_dtype):
                fusion_inputs = prepare_fusion_inputs(
                    batch,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                    modalities=model_modalities,
                    non_blocking=non_blocking,
                )
                for subset_name, spec in subset_specs.items():
                    mask = torch.tensor(spec.mask_for(model_modalities), dtype=torch.bool, device=device)
                    output = adapt_model_output(
                        forward_model(
                            model,
                            "fusion",
                            **fusion_inputs,
                            force_modality_mask=mask,
                        )
                    )
                    logits = select_prediction_slots(output.logits, num_pred)
                    subset_records.extend(
                        records_from_logits(
                            logits,
                            labels,
                            subset_name=subset_name,
                            modalities=spec.modalities,
                            metadata=metadata,
                            dba_delta=dba_delta,
                            horizon_names=horizon_names,
                            dataset_index_offset=sample_offset,
                        )
                    )
                if teacher_ensemble is not None:
                    teacher_outputs = teacher_ensemble(
                        batch,
                        seq_length=seq_length,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    for modality, teacher_output in teacher_outputs.items():
                        logits = select_prediction_slots(teacher_output.logits, num_pred)
                        if tuple(logits.shape[1:]) != (num_pred, num_classes):
                            raise ValueError(
                                f"Teacher '{modality}' logits must have shape [B,{num_pred},{num_classes}], "
                                f"got {tuple(logits.shape)}."
                            )
                        rows = records_from_logits(
                            logits,
                            labels,
                            subset_name=str(modality),
                            modalities=(str(modality),),
                            metadata=metadata,
                            dba_delta=dba_delta,
                            horizon_names=horizon_names,
                            dataset_index_offset=sample_offset,
                        )
                        for row in rows:
                            row["teacher_modality"] = str(modality)
                        teacher_records.extend(rows)
            sample_offset += int(labels.shape[0])

    subset_frame = pd.DataFrame(subset_records)
    subset_write = write_table(subset_frame, output_path, "subset_predictions")
    delta_frame = compute_marginal_deltas(subset_frame, weak_modalities=audit_cfg.get("weak_modalities", WEAK_MODALITIES))
    delta_write = write_table(delta_frame, output_path, "conditional_utility_per_sample_delta")
    aggregate_metrics = aggregate_subset_metrics(subset_frame)
    oracle_summary, oracle_rows = compute_subset_oracle(
        subset_frame,
        candidates=audit_cfg.get("oracle_candidates", DEFAULT_ORACLE_CANDIDATES),
    )
    write_json(oracle_summary, output_path / "oracle_subset_summary.json")

    teacher_summary: dict[str, Any] = {}
    teacher_rescue = pd.DataFrame()
    teacher_write = None
    if dump_teacher:
        teacher_frame = pd.DataFrame(teacher_records)
        teacher_write = write_table(teacher_frame, output_path, "teacher_predictions")
        teacher_summary, teacher_rescue = compute_teacher_complementarity(
            teacher_frame,
            subset_frame,
            weak_modalities=audit_cfg.get("weak_modalities", WEAK_MODALITIES),
        )
        write_json(teacher_summary, output_path / "teacher_complementarity_summary.json")

    feature_frame = pd.DataFrame(feature_records)
    feature_write = write_table(feature_frame, output_path, "communication_state_features")
    bucket_cfg = audit_cfg.get("bucket_features", {})
    feature_names = bucket_cfg.get("features") if isinstance(bucket_cfg, dict) else None
    quantiles = bucket_cfg.get("quantiles", (1.0 / 3.0, 2.0 / 3.0)) if isinstance(bucket_cfg, dict) else (1.0 / 3.0, 2.0 / 3.0)
    bucket_names = bucket_cfg.get("bucket_names", ("low", "mid", "high")) if isinstance(bucket_cfg, dict) else ("low", "mid", "high")
    thresholds = fit_bucket_thresholds(
        feature_frame,
        feature_names or DEFAULT_BUCKET_FEATURES,
        quantiles=tuple(float(q) for q in quantiles),
        bucket_names=tuple(str(name) for name in bucket_names),
    )
    bucketed_features = assign_buckets(feature_frame, thresholds)
    bucket_summary = compute_bucket_summary(
        delta_frame,
        bucketed_features,
        oracle_choices=oracle_rows,
        teacher_rescue=teacher_rescue,
        min_samples=int(audit_cfg.get("diagnosis_thresholds", {}).get("min_bucket_samples", 1)),
    )
    bucket_csv = output_path / "conditional_utility_by_bucket.csv"
    bucket_summary.to_csv(bucket_csv, index=False)

    run_name = str(audit_cfg.get("run_name") or cfg.get("output", {}).get("run_name") or cfg.get("experiment", {}).get("name"))
    summary = build_conditional_utility_summary(
        run_name=run_name,
        scene=cfg.get("data", {}).get("dataset", {}).get("scene_slug")
        or cfg.get("data", {}).get("dataset", {}).get("scene"),
        num_samples=len(dataset),
        horizons=horizon_names,
        aggregate_metrics=aggregate_metrics,
        deltas=delta_frame,
        oracle_summary=oracle_summary,
        teacher_summary=teacher_summary,
        bucket_summary=bucket_summary,
        metadata={
            "checkpoint_resolution": checkpoint_resolution.to_dict(),
            "checkpoint_load": checkpoint_load,
            "dataset": dataset_run_metadata(dataset),
            "throughput": throughput_run_metadata(cfg, {"test": dataloader}, device=device),
            "subsets": subset_metadata(model_modalities),
            "table_outputs": {
                "subset_predictions": subset_write,
                "conditional_utility_per_sample_delta": delta_write,
                "teacher_predictions": teacher_write,
                "communication_state_features": feature_write,
                "conditional_utility_by_bucket": {
                    "format": "csv",
                    "path": str(bucket_csv),
                    "num_rows": int(len(bucket_summary)),
                },
            },
            "bucket_thresholds": thresholds,
            "mmwave_source": "normalized_input" if cfg.get("data", {}).get("dataset", {}).get("mmwave_normalize", True) else "input",
        },
        diagnosis_thresholds=audit_cfg.get("diagnosis_thresholds", {}),
    )
    summary_path = write_json(summary, output_path / "conditional_utility_summary.json")
    return {
        "output_dir": str(output_path),
        "summary": str(summary_path),
        "subset_predictions": subset_write,
        "conditional_utility_per_sample_delta": delta_write,
        "teacher_predictions": teacher_write,
        "bucket_csv": str(bucket_csv),
    }


def _normalization_dataset_kwargs(
    cfg: dict[str, Any],
    checkpoint_metadata: dict[str, Any] | None,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    metadata = _merge_normalization_metadata(checkpoint_metadata, checkpoint_path)
    dataset_kwargs = load_normalization_artifacts(metadata)
    needs_train_gps = _uses_modality(cfg, "gps") and "gps_scaler" not in dataset_kwargs
    needs_train_lidar = _uses_modality(cfg, "lidar") and "lidar_normalizer" not in dataset_kwargs
    if _uses_modality(cfg, "mmwave") and cfg.get("data", {}).get("dataset", {}).get("mmwave_normalize", True):
        if "mmwave_scaler" not in dataset_kwargs:
            raise ValueError(
                "Conditional Utility Audit requires the train-fitted mmwave_scaler for normalized mmWave input. "
                "Use a checkpoint with normalization metadata or disable data.dataset.mmwave_normalize."
            )
    if needs_train_gps or needs_train_lidar:
        train_dataset = build_dataset(cfg, "train")
        prepare_lidar_normalizer(cfg, train_dataset)
        if needs_train_gps:
            dataset_kwargs["gps_scaler"] = getattr(train_dataset, "gps_scaler", None)
        if needs_train_lidar:
            dataset_kwargs["lidar_normalizer"] = getattr(train_dataset, "lidar_normalizer", None)
    return dataset_kwargs


def _merge_normalization_metadata(
    checkpoint_metadata: dict[str, Any] | None,
    checkpoint_path: Path | None,
) -> dict[str, Any] | None:
    fallback = _normalization_metadata_from_checkpoint_context(checkpoint_path)
    if checkpoint_metadata is None:
        return fallback
    if not fallback:
        return checkpoint_metadata
    merged = deepcopy(checkpoint_metadata)
    artifacts = dict(fallback.get("normalization_artifacts") or {})
    artifacts.update(merged.get("normalization_artifacts") or {})
    merged["normalization_artifacts"] = artifacts
    return merged


def _normalization_metadata_from_checkpoint_context(checkpoint_path: Path | None) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    artifacts_dir = run_dir / "artifacts"
    artifacts = {}
    for key, filename in (
        ("gps_scaler", "gps_scaler.npz"),
        ("lidar_normalizer", "lidar_normalizer.npz"),
        ("mmwave_scaler", "mmwave_scaler.npz"),
    ):
        candidate = artifacts_dir / filename
        if candidate.exists():
            artifacts[key] = str(candidate)
    if artifacts:
        return {"normalization_artifacts": artifacts}
    return None


def _uses_modality(cfg: dict[str, Any], modality: str) -> bool:
    if cfg.get("data", {}).get("dataset", {}).get(f"use_{modality}", False):
        return True
    if cfg.get("experiment", {}).get("task") == modality:
        return True
    if cfg.get("experiment", {}).get("task") != "fusion":
        return False
    return modality in _model_modalities(cfg)


def _model_modalities(cfg: dict[str, Any]) -> tuple[str, ...]:
    model_cfg = cfg.get("model", {})
    selected = model_cfg.get("modalities") or model_cfg.get("student", {}).get("modalities") or MODALITY_ORDER
    return normalize_modalities(tuple(selected), context="conditional audit model modalities")


def _resolve_subset_specs(audit_cfg: dict[str, Any], model_modalities: tuple[str, ...]) -> dict[str, Any]:
    names = list(audit_cfg.get("subsets") or CONDITIONAL_UTILITY_SUBSET_NAMES)
    specs = {}
    for name in names:
        spec = resolve_conditional_utility_subset(str(name), model_modalities)
        if spec is None:
            raise ValueError(
                f"Conditional Utility Audit subset '{name}' is not available for modalities {list(model_modalities)}."
            )
        specs[str(name)] = spec
    return specs


def _resolve_output_dir(cfg: dict[str, Any], audit_cfg: dict[str, Any], output_dir: str | None) -> Path:
    explicit = output_dir or audit_cfg.get("output_dir")
    if explicit:
        return resolve_path(explicit)
    base = resolve_path(cfg.get("output", {}).get("dir", "outputs"))
    scene_slug = scene_slug_from_config(cfg)
    if scene_slug and base.name != scene_slug:
        base = base / scene_slug
    run_name = str(audit_cfg.get("run_name") or cfg.get("output", {}).get("run_name") or cfg.get("experiment", {}).get("name", "run"))
    return base / run_name / "conditional_utility"


def _build_teacher_ensemble_from_audit_config(
    cfg: dict[str, Any],
    audit_cfg: dict[str, Any],
    device: torch.device,
    model_modalities: tuple[str, ...],
):
    teacher_cfg = deepcopy(cfg)
    registry_path = audit_cfg.get("teacher_registry_path") or cfg.get("teacher", {}).get("registry_path")
    strict = bool(audit_cfg.get("strict_teacher_load", True))
    required = bool(audit_cfg.get("require_teacher_checkpoints", True))
    if registry_path:
        registry = _load_teacher_registry(registry_path)
        registry_teachers = registry.get("teachers") or {}
        teachers = {}
        teacher_modalities = normalize_modalities(
            tuple(audit_cfg.get("teacher_modalities") or model_modalities),
            context="conditional audit teacher modalities",
        )
        for modality in teacher_modalities:
            item = registry_teachers.get(modality)
            if item is None:
                if required:
                    raise FileNotFoundError(
                        f"Teacher registry {registry_path} does not contain modality '{modality}'."
                    )
                continue
            checkpoint = item.get("checkpoint") or item.get("ckpt")
            if checkpoint is None:
                if required:
                    raise FileNotFoundError(
                        f"Teacher registry {registry_path} modality '{modality}' has no checkpoint/ckpt field."
                    )
                continue
            teachers[modality] = {
                "checkpoint": checkpoint,
                "strict_load": strict,
                "required": required,
            }
        teacher_cfg.setdefault("distillation", {}).setdefault("g2d", {})["teachers"] = teachers
    return build_g2d_teacher_ensemble(teacher_cfg, device)


def _load_teacher_registry(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Teacher registry not found: {path}")
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
