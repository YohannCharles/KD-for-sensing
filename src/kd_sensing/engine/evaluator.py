import json

import torch

from kd_sensing.config.io import dump_config
from kd_sensing.engine.data_factory import (
    build_dataloader,
    build_protocol_split_datasets,
    build_split_dataset,
    prepare_lidar_normalizer,
)
from kd_sensing.engine.modality_resolution import (
    config_uses_csi,
    config_uses_gps,
    config_uses_lidar,
    config_uses_mmwave,
    resolve_enabled_modalities,
)
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.optim import build_device, build_model, build_task_criterion
from kd_sensing.engine.objectives.metadata import (
    objective_requires_occlusion,
    objective_requires_position,
    objective_runtime_metadata,
)
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata, throughput_run_metadata
from kd_sensing.engine.run_lineage import run_lineage_metadata
from kd_sensing.engine.run_status import (
    write_complete_status,
    write_failed_status_for_active_run,
    write_running_status,
)
from kd_sensing.engine.runtime import configure_torch_runtime_threads
from kd_sensing.engine.trainer import create_eval_run_dir, final_config_with_runtime
from kd_sensing.engine.validator import validate
from kd_sensing.utils.artifact_registry import (
    resolve_evaluation_checkpoint,
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
    run_dir = create_eval_run_dir(cfg, output_dir=output_dir)
    write_running_status(run_dir, cfg, kind="evaluation")
    checkpoint_resolution = resolve_evaluation_checkpoint(cfg, weights)
    if checkpoint_resolution.path is None:
        raise FileNotFoundError(
            "Evaluation checkpoint not found in the checkpoint registry. "
            "Run evaluation with --weights or set evaluation.weights to an absolute checkpoint path. "
            f"Resolution: {checkpoint_resolution.to_dict()}"
        )
    validate_evaluation_gps_checkpoint_provenance(cfg, checkpoint_resolution.metadata)
    validate_normalization_artifact_fingerprint(cfg, checkpoint_resolution.metadata)
    dataset_kwargs = load_normalization_artifacts(checkpoint_resolution.metadata)
    split_metadata = {}
    if checkpoint_resolution.metadata and checkpoint_resolution.metadata.get("split_metadata"):
        recorded_splits = checkpoint_resolution.metadata["split_metadata"]
        if "train" in recorded_splits:
            split_metadata["train"] = recorded_splits["train"]
    enabled_modalities = resolve_enabled_modalities(cfg)
    needs_train_gps = config_uses_gps(cfg) and "gps_scaler" not in dataset_kwargs
    needs_train_lidar = config_uses_lidar(cfg) and "lidar_normalizer" not in dataset_kwargs
    if _evaluation_uses_occlusion_target(cfg) and "occlusion_target_stats" not in dataset_kwargs:
        occlusion_target = cfg.get("data", {}).get("dataset", {}).get("occlusion_target", {})
        threshold = occlusion_target.get("threshold") if isinstance(occlusion_target, dict) else None
        if threshold is None:
            raise ValueError(
                "Occlusion evaluation requires train-fitted occlusion_target_stats. "
                "Use a registry checkpoint with auxiliary target artifacts or set data.dataset.occlusion_target.threshold."
            )
    if _evaluation_uses_position_target_scaler(cfg) and "position_target_scaler" not in dataset_kwargs:
        raise ValueError(
            "Position target evaluation with normalization requires a train-fitted position_target_scaler. "
            "Use a registry checkpoint with auxiliary target artifacts or disable data.dataset.position_target.normalize."
        )
    if config_uses_mmwave(cfg) and _mmwave_normalization_enabled(cfg) and "mmwave_scaler" not in dataset_kwargs:
        raise ValueError(
            "mmWave evaluation requires a train-fitted mmwave_scaler. "
            "Use a registry checkpoint with normalization metadata, provide a mmWave scaler artifact, "
            "or disable data.dataset.mmwave_normalize."
        )
    if config_uses_csi(cfg) and _csi_train_rms_enabled(cfg) and "csi_rms_normalizer" not in dataset_kwargs:
        raise ValueError(
            "CSI evaluation requires a train-fitted csi_rms_normalizer. "
            "Use a registry checkpoint with normalization metadata or disable data.dataset.csi_train_rms."
        )
    protocol_splits = build_protocol_split_datasets(cfg, **dataset_kwargs)
    if protocol_splits is not None:
        train_dataset = protocol_splits["train"]
        prepare_lidar_normalizer(cfg, train_dataset)
        split_metadata["train"] = dataset_run_metadata(train_dataset)
        dataset = protocol_splits["test"]
        if needs_train_gps:
            dataset_kwargs["gps_scaler"] = _dataset_attr_recursive(train_dataset, "gps_scaler")
        if needs_train_lidar and getattr(train_dataset, "use_lidar", False):
            dataset_kwargs["lidar_normalizer"] = _dataset_attr_recursive(train_dataset, "lidar_normalizer")
    elif needs_train_gps:
        train_dataset = build_split_dataset(cfg, "train")
        prepare_lidar_normalizer(cfg, train_dataset)
        split_metadata["train"] = dataset_run_metadata(train_dataset)
        dataset_kwargs["gps_scaler"] = _dataset_attr_recursive(train_dataset, "gps_scaler")
        if needs_train_lidar and getattr(train_dataset, "use_lidar", False):
            dataset_kwargs["lidar_normalizer"] = _dataset_attr_recursive(train_dataset, "lidar_normalizer")
        dataset = build_split_dataset(cfg, "test", **dataset_kwargs)
    elif needs_train_lidar:
        train_dataset = build_split_dataset(cfg, "train")
        prepare_lidar_normalizer(cfg, train_dataset)
        split_metadata["train"] = dataset_run_metadata(train_dataset)
        dataset_kwargs["lidar_normalizer"] = _dataset_attr_recursive(train_dataset, "lidar_normalizer")
        dataset = build_split_dataset(cfg, "test", **dataset_kwargs)
    else:
        dataset = build_split_dataset(cfg, "test", **dataset_kwargs)
    _apply_csi_rms_to_model_config(cfg, dataset)
    split_metadata["test"] = dataset_run_metadata(dataset)
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
            checkpoint_registry=checkpoint_resolution.to_dict(),
            throughput_metadata=throughput_metadata,
        ),
        run_dir / "final_config.yaml",
    )
    loader_cfg = cfg["data"]["dataloader"]
    dataloader = build_dataloader(dataset, loader_cfg, split="test")
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
                    "registry_dir": str(checkpoint_resolution.registry_dir)
                    if checkpoint_resolution.registry_dir is not None
                    else None,
                    "metadata": checkpoint_resolution.metadata,
                }
            )
    criterion = build_task_criterion(cfg)
    metrics = validate(model, dataloader, cfg, criterion, device, output_dir=run_dir)
    report = {
        **metrics,
        "checkpoint_load": checkpoint_load,
        "split_protocol": _evaluation_split_protocol_report(split_metadata),
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
    with (run_dir / "test_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_complete_status(
        run_dir,
        cfg,
        kind="evaluation",
        primary_metric=_evaluation_primary_metric(metrics),
        metrics_path=run_dir / "metrics.json",
        best_checkpoint=checkpoint_resolution.path,
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


def _evaluation_split_protocol_report(split_metadata: dict) -> dict:
    test = split_metadata.get("test", {}) if isinstance(split_metadata, dict) else {}
    sidecar = test.get("split_metadata", {}) if isinstance(test.get("split_metadata"), dict) else {}
    split_metadata_available = sidecar.get("available")
    if split_metadata_available is None:
        split_metadata_available = bool(test.get("split_metadata_path") or test.get("split_protocol"))
    split_metadata_path = test.get("split_metadata_path") or sidecar.get("path")
    strict_validation_eligible = test.get("strict_validation_eligible")
    reasons = list(test.get("eligibility_reasons") or [])
    warnings = []
    if split_metadata_available is False:
        warnings.append(
            {
                "code": "split_metadata_missing",
                "message": "Evaluation test CSV has no split metadata sidecar; treat split eligibility as unknown.",
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
        "test_csv": test.get("csv_path"),
        "test_csv_name": test.get("csv_name"),
        "test_num_samples": test.get("num_samples"),
        "split_metadata_available": bool(split_metadata_available),
        "split_metadata_path": split_metadata_path,
        "split_metadata_expected_path": sidecar.get("expected_path"),
        "split_protocol": test.get("split_protocol"),
        "split_strategy": test.get("split_strategy"),
        "split_protocol_version": test.get("split_protocol_version"),
        "strict_validation_eligible": strict_validation_eligible,
        "eligibility_reasons": reasons,
        "leakage_diagnostics": test.get("leakage_diagnostics"),
        "warnings": warnings,
    }


def _evaluation_primary_metric(metrics: dict) -> dict:
    objective = metrics.get("objective") if isinstance(metrics.get("objective"), dict) else {}
    name = objective.get("primary_metric") or "loss"
    value = metrics.get(name)
    if value is None and name == "top1" and isinstance(metrics.get("topk"), dict):
        value = metrics["topk"].get("1") or metrics["topk"].get(1)
    return {"name": name, "value": value}


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


def _mmwave_normalization_enabled(cfg: dict) -> bool:
    return bool(cfg.get("data", {}).get("dataset", {}).get("mmwave_normalize", True))


def _csi_train_rms_enabled(cfg: dict) -> bool:
    return bool(cfg.get("data", {}).get("dataset", {}).get("csi_train_rms", True))


def _apply_csi_rms_to_model_config(cfg: dict, dataset) -> None:
    normalizer = getattr(dataset, "csi_rms_normalizer", None)
    if normalizer is None:
        return
    rms = float(getattr(normalizer, "rms", normalizer))
    model_cfg = cfg.setdefault("model", {})
    model_cfg["csi_train_rms"] = rms
    primary_cfg = model_cfg.get("primary")
    if not isinstance(primary_cfg, dict) or "csi" not in primary_cfg.get("modalities", []):
        return
    primary_cfg["csi_train_rms"] = rms
    encoders = primary_cfg.get("encoders")
    if isinstance(encoders, dict) and isinstance(encoders.get("csi"), dict):
        encoders["csi"].setdefault("train_rms", rms)


def _evaluation_uses_occlusion_target(cfg: dict) -> bool:
    if objective_requires_occlusion(cfg):
        return True
    target = cfg.get("data", {}).get("dataset", {}).get("occlusion_target")
    return bool(target.get("enabled", False)) if isinstance(target, dict) else bool(target)


def _evaluation_uses_position_target_scaler(cfg: dict) -> bool:
    if objective_requires_position(cfg):
        target = cfg.get("data", {}).get("dataset", {}).get("position_target")
        if isinstance(target, dict):
            return bool(target.get("normalize", target.get("standardize", True)))
        return True
    target = cfg.get("data", {}).get("dataset", {}).get("position_target")
    if isinstance(target, dict):
        return bool(target.get("enabled", False)) and bool(target.get("normalize", target.get("standardize", True)))
    return bool(target)
