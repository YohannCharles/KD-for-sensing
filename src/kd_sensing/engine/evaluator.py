import json


from kd_sensing.config.io import dump_config
from kd_sensing.engine.data_factory import (
    build_dataloader,
    build_split_dataset,
    shutdown_dataloader_workers,
)
from kd_sensing.data.mmw.protocol import validate_mmw_config_protocol
from kd_sensing.engine.data_factory_scalers import fit_gps_scaler
from kd_sensing.engine.modality_resolution import (
    config_uses_gps,
    resolve_enabled_modalities,
)
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.optim import build_device, build_model, build_task_criterion
from kd_sensing.engine.objectives.metadata import objective_runtime_metadata
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata, throughput_run_metadata
from kd_sensing.engine.run_lineage import run_lineage_metadata
from kd_sensing.engine.run_status import (
    write_complete_status,
    write_failed_status_for_active_run,
    write_running_status,
)
from kd_sensing.engine.runtime import configure_cuda_performance_settings, configure_torch_runtime_threads
from kd_sensing.engine.trainer import create_eval_run_dir, final_config_with_runtime
from kd_sensing.engine.validator import validate
from kd_sensing.utils.artifact_registry import (
    resolve_evaluation_checkpoint,
    validate_evaluation_checkpoint_route,
    validate_evaluation_gps_checkpoint_provenance,
)
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state
from kd_sensing.utils.seed import set_seed


def evaluate(cfg: dict, weights: str | None = None, output_dir: str | None = None) -> dict:
    try:
        return _evaluate_inner(cfg, weights=weights, output_dir=output_dir)
    except Exception as exc:
        try:
            write_failed_status_for_active_run(cfg, exc, kind="evaluation")
        except Exception:
            pass
        raise


def _evaluate_inner(cfg: dict, weights: str | None = None, output_dir: str | None = None) -> dict:
    configure_torch_runtime_threads(cfg)
    set_seed(cfg.get("experiment", {}).get("seed", 0))
    device = build_device(cfg)
    configure_cuda_performance_settings(cfg, device)
    run_dir = create_eval_run_dir(cfg, output_dir=output_dir)
    write_running_status(run_dir, cfg, kind="evaluation")
    checkpoint_resolution = resolve_evaluation_checkpoint(cfg, weights)
    if checkpoint_resolution.path is None:
        raise FileNotFoundError(
            "Evaluation requires --weights or evaluation.weights with a checkpoint path. "
            f"Resolution: {checkpoint_resolution.to_dict()}"
        )
    validate_evaluation_gps_checkpoint_provenance(cfg, checkpoint_resolution.metadata)
    validate_evaluation_checkpoint_route(checkpoint_resolution.metadata)
    validate_normalization_artifact_fingerprint(cfg, checkpoint_resolution.metadata)
    mmw_protocol_audit = validate_mmw_config_protocol(cfg)
    evaluation_split = "validation" if mmw_protocol_audit is not None else "test"
    dataset_kwargs = load_normalization_artifacts(checkpoint_resolution.metadata)
    split_metadata = {}
    if checkpoint_resolution.metadata and checkpoint_resolution.metadata.get("split_metadata"):
        recorded_splits = checkpoint_resolution.metadata["split_metadata"]
        if "train" in recorded_splits:
            split_metadata["train"] = recorded_splits["train"]
    enabled_modalities = resolve_enabled_modalities(cfg)
    needs_train_gps = config_uses_gps(cfg) and "gps_scaler" not in dataset_kwargs
    if needs_train_gps:
        train_dataset = build_split_dataset(cfg, "train")
        fit_gps_scaler(train_dataset, source="train_split_streaming_fit")
        split_metadata["train"] = dataset_run_metadata(train_dataset)
        dataset_kwargs["gps_scaler"] = _dataset_attr_recursive(train_dataset, "gps_scaler")
        dataset = build_split_dataset(cfg, evaluation_split, **dataset_kwargs)
    else:
        dataset = build_split_dataset(cfg, evaluation_split, **dataset_kwargs)
    split_metadata[evaluation_split] = dataset_run_metadata(dataset)
    normalization_artifacts = {}
    if checkpoint_resolution.metadata:
        normalization_artifacts = checkpoint_resolution.metadata.get("normalization_artifacts", {})
    throughput_metadata = throughput_run_metadata(cfg, device=device)
    prediction_setup = prediction_setup_metadata(cfg, split_metadata=split_metadata)
    lineage = run_lineage_metadata(cfg)
    dump_config(
        final_config_with_runtime(
            cfg,
            run_dir=run_dir,
            split_metadata=split_metadata,
            normalization_artifacts=normalization_artifacts,
            evaluation_checkpoint=checkpoint_resolution.to_dict(),
            throughput_metadata=throughput_metadata,
        ),
        run_dir / "final_config.yaml",
    )
    loader_cfg = cfg["data"]["dataloader"]
    dataloader = build_dataloader(dataset, loader_cfg, split=evaluation_split)
    model = build_model(cfg["model"]["primary"]).to(device)
    checkpoint_load = None
    if checkpoint_resolution.path is not None:
        if not checkpoint_resolution.path.exists():
            raise FileNotFoundError(f"Evaluation checkpoint not found. Resolution: {checkpoint_resolution.to_dict()}")
        load_result = load_model_state(
            checkpoint_resolution.path,
            model,
            role="evaluation",
            map_location=device,
            strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
        )
        checkpoint_load = checkpoint_load_summary(load_result)
        if checkpoint_load is not None:
            checkpoint_load.update(
                {
                    "source": checkpoint_resolution.source,
                    "metadata": checkpoint_resolution.metadata,
                }
            )
    criterion = build_task_criterion(cfg)
    try:
        metrics = validate(model, dataloader, cfg, criterion, device, output_dir=run_dir)
    finally:
        shutdown_dataloader_workers(dataloader)
    report = {
        **metrics,
        "checkpoint_load": checkpoint_load,
        "split_protocol": _evaluation_split_protocol_report(split_metadata, evaluation_split),
        "runtime": {
            "run_dir": str(run_dir),
            "splits": split_metadata,
            "checkpoint_resolution": checkpoint_resolution.to_dict(),
            "normalization_artifacts": normalization_artifacts,
            "throughput": throughput_metadata,
            "prediction_objective": objective_runtime_metadata(cfg),
            "prediction_setup": prediction_setup,
            "enabled_modalities": list(enabled_modalities),
            "lineage": lineage,
        },
    }
    with (run_dir / f"{evaluation_split}_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_complete_status(
        run_dir,
        cfg,
        kind="evaluation",
        primary_metric=_evaluation_primary_metric(metrics),
        metrics_path=run_dir / "metrics.json",
        checkpoint=checkpoint_resolution.path,
    )
    return {
        "run_dir": str(run_dir),
        "metrics": metrics,
        "checkpoint_load": checkpoint_load,
        "checkpoint_resolution": checkpoint_resolution.to_dict(),
        "split_metadata": split_metadata,
        "split_protocol": report["split_protocol"],
        "throughput": throughput_metadata,
    }


def _evaluation_split_protocol_report(split_metadata: dict, split: str) -> dict:
    metadata = split_metadata.get(split, {}) if isinstance(split_metadata, dict) else {}
    sidecar = metadata.get("split_metadata", {}) if isinstance(metadata.get("split_metadata"), dict) else {}
    split_metadata_available = sidecar.get("available")
    if split_metadata_available is None:
        split_metadata_available = bool(metadata.get("split_metadata_path") or metadata.get("split_protocol"))
    split_metadata_path = metadata.get("split_metadata_path") or sidecar.get("path")
    strict_validation_eligible = metadata.get("strict_validation_eligible")
    reasons = list(metadata.get("eligibility_reasons") or [])
    warnings = []
    if split_metadata_available is False:
        warnings.append(
            {
                "code": "split_metadata_missing",
                "message": "Evaluation CSV has no split metadata sidecar; treat split eligibility as unknown.",
                "expected_path": sidecar.get("expected_path"),
                "fix_hint": "Regenerate or reference the prepared split metadata before using this run for strict conclusions.",
            }
        )
    elif strict_validation_eligible is False:
        warnings.append(
            {
                "code": "split_not_strict_validation_eligible",
                "message": "Split metadata marks this evaluation split as not eligible for strict validation.",
                "eligibility_reasons": reasons,
                "fix_hint": sidecar.get("fix_hint")
                or "Regenerate MMW splits with split_strategy=group_safe_time_block and a fresh strict split tag.",
            }
        )
    return {
        "evaluation_split": split,
        "csv": metadata.get("csv_path"),
        "csv_name": metadata.get("csv_name"),
        "num_samples": metadata.get("num_samples"),
        "split_metadata_available": bool(split_metadata_available),
        "split_metadata_path": split_metadata_path,
        "split_metadata_expected_path": sidecar.get("expected_path"),
        "split_protocol": metadata.get("split_protocol"),
        "split_strategy": metadata.get("split_strategy"),
        "split_protocol_version": metadata.get("split_protocol_version"),
        "strict_validation_eligible": strict_validation_eligible,
        "eligibility_reasons": reasons,
        "leakage_diagnostics": metadata.get("leakage_diagnostics"),
        "warnings": warnings,
    }


def _evaluation_primary_metric(metrics: dict) -> dict:
    return {"name": "loss", "value": metrics.get("loss")}


def _dataset_attr_recursive(dataset, attr: str):
    value = getattr(dataset, attr, None)
    if value is not None:
        return value
    for component in getattr(dataset, "datasets", []) or []:
        value = _dataset_attr_recursive(component, attr)
        if value is not None:
            return value
    parent = getattr(dataset, "dataset", None)
    if parent is not None:
        return _dataset_attr_recursive(parent, attr)
    return None
