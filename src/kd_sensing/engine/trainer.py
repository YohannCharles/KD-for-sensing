from __future__ import annotations

from copy import deepcopy
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from kd_sensing.config.io import dump_config
from kd_sensing.config.lidar_normalization import canonicalize_lidar_normalization_config
from kd_sensing.data.scenes import scene_metadata_from_config, scene_slug_from_config
from kd_sensing.engine.craf_training import CrafTrainingExtension
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.debug_diagnostics import (
    ModuleHealthTracker,
    build_startup_summary,
    configure_csi_debug,
    consume_csi_debug_records,
    evaluate_pilot_noise_validity,
    print_startup_summary,
    set_csi_debug_batch_source,
    training_health_debug_enabled,
    write_config_diff_artifact,
    write_csi_debug_records,
    write_pilot_noise_validity_artifact,
    write_startup_summary,
)
from kd_sensing.engine.g2d_training import G2DTrainingExtension
from kd_sensing.engine.marf_training import MarfTrainingExtension
from kd_sensing.engine.normalization_artifacts import save_normalization_artifacts
from kd_sensing.engine.optim import (
    build_device,
    build_distiller,
    build_model,
    build_optimizer,
    build_scheduler,
    build_task_criterion,
    optimizer_param_group_summary,
)
from kd_sensing.engine.prediction_objectives import (
    compute_prediction_loss,
    normalize_objective_metric,
    objective_history_fields,
    objective_metric_mode,
    objective_optional_history_fields,
    objective_runtime_metadata,
    objective_tensorboard_scalars,
    prepare_prediction_targets,
    resolve_prediction_objective,
    validate_objective_metric_available,
)
from kd_sensing.engine.run_metadata import (
    dataloaders_run_metadata,
    prediction_setup_metadata,
    throughput_run_metadata,
)
from kd_sensing.engine.runtime import (
    autocast_context,
    make_grad_scaler,
    prepare_task_batch,
    prepare_task_auxiliary_targets,
    prepare_task_labels,
    resolve_amp_settings,
    run_model_step,
    transfer_non_blocking,
)
from kd_sensing.engine.training_extensions import (
    BaseLossResult,
    BatchState,
    EpochDiagnosticsAccumulator,
    ExtensionContext,
    ForwardControls,
    NoOpTrainingExtension,
    TrainingExtension,
)
from kd_sensing.engine.teacher_loader import (
    apply_selective_finetune,
    apply_teacher_priors,
    load_teacher_encoders,
    load_teacher_registry,
    trainable_parameter_count,
)
from kd_sensing.engine.validator import validate
from kd_sensing.evaluation.lidar_diagnostics import (
    LidarQualityAccumulator,
    lidar_preprocessing_metadata_from_dataset,
)
from kd_sensing.utils.artifact_registry import archive_best_checkpoint, resolve_teacher_checkpoint, write_sidecar
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_checkpoint, load_model_state, save_checkpoint
from kd_sensing.utils.paths import output_dir as resolve_output_dir, resolve_path
from kd_sensing.utils.plotting import plot_training_curves
from kd_sensing.utils.seed import set_seed
from kd_sensing.utils.teacher_registry import teacher_metrics_from_training


def create_run_dir(cfg: dict) -> Path:
    base = _scene_grouped_output_base(cfg)
    run_name = cfg.get("output", {}).get("run_name")
    if not run_name:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.get('experiment', {}).get('name', 'run')}_{timestamp}"
    path = base / run_name
    if _should_make_unique_run_dir(cfg, path):
        path = _unique_run_path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    return path


def _should_make_unique_run_dir(cfg: dict, path: Path) -> bool:
    if not path.exists():
        return False
    output_cfg = cfg.get("output", {})
    if output_cfg.get("overwrite", False):
        return False
    if cfg.get("training", {}).get("resume") is True:
        return False
    return True


def _unique_run_path(path: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}_{timestamp}")
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        indexed = path.with_name(f"{path.name}_{timestamp}_{index}")
        if not indexed.exists():
            return indexed
        index += 1


def create_eval_run_dir(cfg: dict, output_dir: str | None = None) -> Path:
    if output_dir:
        path = resolve_output_dir(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    base = _scene_grouped_output_base(cfg)
    run_name = cfg.get("output", {}).get("evaluation_run_name") or cfg.get("output", {}).get("run_name")
    if not run_name:
        run_name = cfg.get("experiment", {}).get("name", "run")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base / f"evaluation_{run_name}_{timestamp}"
    path = _unique_run_path(path) if path.exists() else path
    path.mkdir(parents=True, exist_ok=True)
    return path


def final_config_with_runtime(
    cfg: dict,
    *,
    run_dir: Path,
    split_metadata: dict | None = None,
    normalization_artifacts: dict | None = None,
    checkpoint_registry: dict | None = None,
    throughput_metadata: dict | None = None,
    teacher_prior: dict | None = None,
    early_stopping: dict | None = None,
    pilot_noise_validity: dict | None = None,
) -> dict:
    final_cfg = deepcopy(cfg)
    canonicalize_lidar_normalization_config(final_cfg)
    runtime = final_cfg.setdefault("runtime", {})
    runtime["run_dir"] = str(run_dir)
    runtime["output_overwrite"] = bool(cfg.get("output", {}).get("overwrite", False))
    runtime["prediction_objective"] = objective_runtime_metadata(cfg)
    if split_metadata is not None:
        runtime["splits"] = split_metadata
    if normalization_artifacts is not None:
        runtime["normalization_artifacts"] = normalization_artifacts
    if checkpoint_registry is not None:
        runtime["checkpoint_registry"] = checkpoint_registry
    if throughput_metadata is not None:
        runtime["throughput"] = throughput_metadata
        if isinstance(throughput_metadata, dict) and "cache" in throughput_metadata:
            runtime["cache"] = throughput_metadata["cache"]
    if teacher_prior is not None:
        runtime["teacher_prior"] = teacher_prior
    if early_stopping is not None:
        runtime["early_stopping"] = early_stopping
    if pilot_noise_validity is not None:
        runtime["pilot_noise_validity"] = pilot_noise_validity
    scene_metadata = scene_metadata_from_config(cfg)
    if scene_metadata:
        runtime["scene"] = scene_metadata
    runtime["prediction_setup"] = prediction_setup_metadata(cfg, split_metadata=split_metadata)
    return final_cfg


def _scene_grouped_output_base(cfg: dict) -> Path:
    base = resolve_output_dir(cfg.get("output", {}).get("dir", cfg.get("paths", {}).get("output_dir", "outputs")))
    output_cfg = cfg.get("output", {})
    if output_cfg.get("group_by_scene", True) is False:
        return base
    scene_slug = scene_slug_from_config(cfg)
    if not scene_slug or base.name == scene_slug:
        return base
    return base / scene_slug


def _teacher_enabled(cfg: dict) -> bool:
    return cfg.get("distillation", {}).get("type", "no_kd") != "no_kd"


def _g2d_enabled(cfg: dict) -> bool:
    return cfg.get("distillation", {}).get("type", "no_kd") == "g2d"


def _build_training_extensions(cfg: dict) -> list[TrainingExtension]:
    extensions: list[TrainingExtension] = [NoOpTrainingExtension()]
    if _g2d_enabled(cfg):
        extensions.append(G2DTrainingExtension())
    extensions.extend([CrafTrainingExtension(), MarfTrainingExtension()])
    return extensions


def _checkpoint_strict(cfg: dict) -> bool:
    return bool(cfg.get("checkpoint", {}).get("strict_load", True))


def _load_teacher_if_needed(cfg: dict, teacher_model, device: torch.device) -> dict | None:
    weight_name = cfg.get("distillation", {}).get("teacher_model_name")
    resolution = resolve_teacher_checkpoint(cfg, weight_name)
    if resolution.path is None and resolution.source == "none":
        return None
    if resolution.path is None or not resolution.path.exists():
        raise FileNotFoundError(
            "Teacher checkpoint not found in the checkpoint registry. "
            "Train and archive the teacher, or set distillation.teacher_model_name "
            f"to an absolute checkpoint path. Resolution: {resolution.to_dict()}"
        )
    load_result = load_model_state(
        resolution.path,
        teacher_model,
        role="teacher",
        map_location=device,
        strict=_checkpoint_strict(cfg),
    )
    summary = checkpoint_load_summary(load_result)
    if summary is not None:
        summary.update(
            {
                "source": resolution.source,
                "registry_dir": str(resolution.registry_dir) if resolution.registry_dir is not None else None,
                "metadata": resolution.metadata,
            }
        )
    return summary


def _load_stage_checkpoint_if_needed(cfg: dict, model, device: torch.device) -> dict | None:
    finetune_cfg = cfg.get("finetune", {})
    checkpoint_path = finetune_cfg.get("checkpoint_path") or finetune_cfg.get("stage2_checkpoint")
    if not checkpoint_path:
        return None
    resolved = resolve_path(checkpoint_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Stage checkpoint not found: {resolved}")
    load_result = load_model_state(
        resolved,
        model,
        role="stage_checkpoint",
        map_location=device,
        strict=bool(finetune_cfg.get("strict", cfg.get("checkpoint", {}).get("strict_load", True))),
    )
    summary = checkpoint_load_summary(load_result)
    if summary is not None:
        summary["source"] = "stage_checkpoint"
    return summary


def _apply_teacher_prior_initialization(cfg: dict, model, device: torch.device) -> dict | None:
    teacher_cfg = cfg.get("teacher", {})
    registry_path = teacher_cfg.get("registry_path") or teacher_cfg.get("teacher_registry")
    if not registry_path:
        return None
    registry = load_teacher_registry(registry_path)
    modalities = (
        cfg.get("model", {}).get("student", {}).get("modalities")
        or registry.get("modalities")
        or []
    )
    priors = apply_teacher_priors(model, registry, modalities)
    load_summaries = {}
    if bool(teacher_cfg.get("load_encoders", False)):
        load_summaries = load_teacher_encoders(
            model,
            registry,
            modalities,
            strict=bool(teacher_cfg.get("strict", cfg.get("checkpoint", {}).get("strict_load", True))),
            map_location=device,
            freeze_loaded=bool(teacher_cfg.get("freeze_encoders", False)),
        )
    frozen = _encoder_freeze_log(model)
    return {
        "registry_path": registry.get("_resolved_path"),
        "prior_mode": registry.get("prior_mode"),
        "priors": priors,
        "encoder_load": load_summaries,
        "encoder_freeze": frozen,
        "trainable_params": trainable_parameter_count(model),
        "teachers": {
            modality: {
                "checkpoint": item.get("ckpt") or item.get("checkpoint"),
                "prior": item.get("prior"),
            }
            for modality, item in (registry.get("teachers") or {}).items()
            if modality in set(priors)
        },
    }


def _apply_selective_finetune_if_needed(cfg: dict, model) -> dict | None:
    finetune_cfg = cfg.get("finetune", {})
    if not finetune_cfg.get("enabled", False):
        return None
    return apply_selective_finetune(
        model,
        unfreeze_modalities=finetune_cfg.get("unfreeze_modalities", []),
        freeze_modalities=finetune_cfg.get("freeze_modalities", []),
    )


def _encoder_freeze_log(model) -> dict[str, dict]:
    if not hasattr(model, "encoders"):
        return {}
    summary = {}
    for modality in getattr(model, "modalities", tuple(model.encoders.keys())):
        if modality not in model.encoders:
            continue
        params = list(model.encoders[modality].parameters())
        total = sum(param.numel() for param in params)
        trainable = sum(param.numel() for param in params if param.requires_grad)
        summary[modality] = {
            "frozen": trainable == 0 and total > 0,
            "total_params": int(total),
            "trainable_params": int(trainable),
        }
    return summary


def _add_distiller_params_to_optimizer(optimizer: torch.optim.Optimizer, distiller, cfg: dict) -> None:
    params = [param for param in distiller.parameters() if param.requires_grad]
    if not params:
        return
    optimizer.add_param_group(
        {
            "params": params,
            "name": "distiller",
            "lr": cfg.get("training", {}).get("lr", 7.5e-4),
            "param_count": int(sum(param.numel() for param in params)),
        }
    )


def _resolve_resume_checkpoint(cfg: dict, run_dir: Path) -> Path | None:
    resume = cfg.get("training", {}).get("resume", False)
    if not resume:
        return None
    if resume is True:
        if not cfg.get("output", {}).get("run_name"):
            raise ValueError("training.resume=true requires output.run_name so checkpoints/last.pth can be resolved.")
        checkpoint_path = run_dir / "checkpoints" / "last.pth"
    elif isinstance(resume, str):
        checkpoint_path = resolve_path(resume)
    else:
        raise ValueError("training.resume must be false, true, or a checkpoint path string.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _create_tensorboard_writer(cfg: dict, run_dir: Path):
    tensorboard_cfg = cfg.get("output", {}).get("tensorboard", {})
    if not tensorboard_cfg.get("enabled", True):
        return None

    log_dir = tensorboard_cfg.get("log_dir", "tensorboard") or "tensorboard"
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(run_dir / str(log_dir)))


def _write_tensorboard_scalars(
    writer,
    history: dict,
    step: int,
    *,
    objective: str = "beam",
    tensorboard_cfg: dict | None = None,
) -> None:
    if writer is None:
        return

    writer.add_scalar("loss/train", history["train_loss"][-1], step)
    writer.add_scalar("loss/train_task", history["train_task_loss"][-1], step)
    writer.add_scalar("loss/train_distill", history["train_distill_loss"][-1], step)
    writer.add_scalar("loss/val", history["val_loss"][-1], step)
    for tag, history_key in objective_tensorboard_scalars(objective):
        _add_latest_scalar(writer, tag, history, history_key, step)
    if _legacy_accuracy_tags_enabled(tensorboard_cfg):
        _write_tensorboard_legacy_accuracy_scalars(writer, history, step)
    writer.add_scalar("learning_rate/main", history["learning_rates"][-1], step)
    writer.flush()


def _write_tensorboard_legacy_accuracy_scalars(writer, history: dict, step: int) -> None:
    _add_latest_scalar(writer, "accuracy/train", history, "train_acc", step)
    _add_latest_scalar(writer, "accuracy/val", history, "val_acc", step)
    _add_latest_scalar(writer, "accuracy/val_atop3", history, "val_atop3", step)
    _add_latest_scalar(writer, "accuracy/val_atop5", history, "val_atop5", step)
    _add_latest_scalar(writer, "dba/val_adba", history, "val_adba", step)


def _legacy_accuracy_tags_enabled(tensorboard_cfg: dict | None) -> bool:
    if not isinstance(tensorboard_cfg, dict):
        return False
    return bool(tensorboard_cfg.get("legacy_accuracy_tags", False))


def _add_latest_scalar(writer, tag: str, history: dict, key: str, step: int) -> None:
    values = history.get(key)
    if not values:
        return
    value = _finite_float_or_none(values[-1])
    if value is None:
        return
    writer.add_scalar(tag, value, step)


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _write_tensorboard_craf_scalars(writer, epoch_log: dict, step: int) -> None:
    if writer is None:
        return
    reliability = epoch_log.get("craf_reliability")
    if isinstance(reliability, dict):
        for modality, value in reliability.items():
            writer.add_scalar(f"craf/reliability/{modality}", float(value), step)
    for key, value in epoch_log.items():
        if not isinstance(value, (int, float)):
            continue
        if key.startswith(("cf/", "craf/", "marf/", "loss/", "teacher/", "optimizer/", "val/subset/")):
            writer.add_scalar(key, float(value), step)
    writer.flush()


def _mean_valid_slots(values, totals) -> float:
    values_arr = np.asarray(values, dtype=float)
    totals_arr = np.asarray(totals, dtype=float)
    length = min(values_arr.size, totals_arr.size)
    if length == 0:
        return 0.0

    values_arr = values_arr[:length]
    valid_slots = totals_arr[:length] > 0
    if not np.any(valid_slots):
        return 0.0
    return float(np.mean(values_arr[valid_slots]))


def _aggregate_validation_metrics(val_metrics: dict) -> dict[str, float]:
    topk = val_metrics.get("topk", {})
    total = val_metrics.get("total", [])
    return {
        "val_atop3": _mean_valid_slots(topk.get("3", []), total),
        "val_atop5": _mean_valid_slots(topk.get("5", []), total),
        "val_adba": _mean_valid_slots(val_metrics.get("dba", []), total),
    }


def _normalize_early_stopping_metric(metric: object, *, objective: str = "beam") -> str:
    return normalize_objective_metric(metric, objective=objective)


def _resolve_early_stopping_mode(metric: str, mode: object | None) -> str:
    return objective_metric_mode(metric, mode)


def _configure_early_stopping(training_cfg: dict, objective: str = "beam") -> tuple[str, str]:
    metric = _normalize_early_stopping_metric(training_cfg.get("early_stopping_metric"), objective=objective)
    mode = _resolve_early_stopping_mode(metric, training_cfg.get("early_stopping_mode"))
    training_cfg["early_stopping_metric"] = metric
    training_cfg["early_stopping_mode"] = mode
    return metric, mode


def _initial_early_stopping_value(mode: str) -> float:
    return float("inf") if mode == "min" else float("-inf")


def _early_stopping_min_epoch(total_epochs: int) -> int:
    total_epochs = max(int(total_epochs), 0)
    return (total_epochs + 1) // 2


def _early_stopping_improved(current: float, best: float, *, mode: str, min_delta: float) -> bool:
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"Unsupported early stopping mode '{mode}'.")


def _early_stopping_metric_value(epoch_log: dict, metric: str) -> float:
    if metric not in epoch_log:
        raise ValueError(
            f"Early stopping metric '{metric}' is not available in validation metrics. "
            "Ensure validation produces DBA/ADBA metrics or configure "
            "training.early_stopping_metric to another supported metric such as val_loss."
        )
    value = epoch_log[metric]
    if value is None:
        raise ValueError(
            f"Early stopping metric '{metric}' is None. Configure a supported metric with a numeric value."
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Early stopping metric '{metric}' must be numeric, got {value!r}.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"Early stopping metric '{metric}' must be finite, got {numeric}.")
    return numeric


def _validate_early_stopping_source_available(val_metrics: dict, metric: str) -> None:
    validate_objective_metric_available(val_metrics, metric)


def _available_early_stopping_metrics(val_metrics: dict) -> set[str]:
    available = val_metrics.get("available_metrics", [])
    return set(available) if isinstance(available, list) else set()


def _legacy_early_stopping_value(checkpoint: dict, metric: str, default: float) -> float:
    if metric == "val_loss":
        return float(checkpoint.get("best_val_loss", checkpoint.get("test_loss", default)))
    if metric == "val_acc":
        return float(checkpoint.get("best_val_top1", default))
    return default


def _legacy_early_stopping_epoch(checkpoint: dict, metric: str, default: int) -> int:
    if metric == "val_acc":
        return int(checkpoint.get("best_top1_epoch", default))
    return default


def _early_stopping_state(
    *,
    metric: str,
    mode: str,
    best_value: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> dict:
    return {
        "metric": metric,
        "mode": mode,
        "best_value": float(best_value),
        "best_epoch": int(best_epoch),
        "epochs_without_improvement": int(epochs_without_improvement),
    }


def _validation_subset_epoch_scalars(val_metrics: dict) -> dict[str, float]:
    subset_metrics = val_metrics.get("modality_subsets")
    if not isinstance(subset_metrics, dict):
        return {}
    scalars: dict[str, float] = {}
    for subset_name, metrics in subset_metrics.items():
        if not isinstance(metrics, dict):
            continue
        prefix = f"val/subset/{subset_name}"
        scalars[f"{prefix}/loss"] = float(metrics.get("loss", 0.0))
        topk = metrics.get("topk", {})
        total = metrics.get("total", [])
        scalars[f"{prefix}/top1"] = _first_valid_slot(topk.get("1", []), total)
        scalars[f"{prefix}/atop3"] = _mean_valid_slots(topk.get("3", []), total)
        scalars[f"{prefix}/atop5"] = _mean_valid_slots(topk.get("5", []), total)
        scalars[f"{prefix}/adba"] = _mean_valid_slots(metrics.get("dba", []), total)
    return scalars


def _first_valid_slot(values, totals) -> float:
    values_arr = np.asarray(values, dtype=float)
    totals_arr = np.asarray(totals, dtype=float)
    length = min(values_arr.size, totals_arr.size)
    if length == 0:
        return 0.0
    for idx in range(length):
        if totals_arr[idx] > 0:
            return float(values_arr[idx])
    return 0.0


def _close_tensorboard_writer(writer) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()


def _progress_enabled(cfg: dict) -> bool:
    return cfg.get("output", {}).get("progress", {}).get("enabled", True)


def train(cfg: dict) -> dict:
    set_seed(cfg.get("experiment", {}).get("seed", 0))
    objective = resolve_prediction_objective(cfg)
    cfg.setdefault("experiment", {})["objective"] = objective
    objective_metadata = objective_runtime_metadata(cfg)
    training_cfg = cfg.setdefault("training", {})
    early_stopping_metric, early_stopping_mode = _configure_early_stopping(training_cfg, objective=objective)
    if cfg.get("training", {}).get("resume") is True and not cfg.get("output", {}).get("run_name"):
        raise ValueError("training.resume=true requires output.run_name so checkpoints/last.pth can be resolved.")
    run_dir = create_run_dir(cfg)
    dataloaders = build_dataloaders(cfg)
    _apply_csi_rms_to_model_config(cfg, dataloaders)
    split_metadata = dataloaders_run_metadata(dataloaders)
    normalization_artifacts = save_normalization_artifacts(dataloaders, run_dir)
    device = build_device(cfg)
    throughput_metadata = throughput_run_metadata(cfg, dataloaders, device)
    resolved_cfg = final_config_with_runtime(
        cfg,
        run_dir=run_dir,
        split_metadata=split_metadata,
        normalization_artifacts=normalization_artifacts,
        throughput_metadata=throughput_metadata,
    )
    dump_config(resolved_cfg, run_dir / "resolved_config.yaml")
    dump_config(resolved_cfg, run_dir / "final_config.yaml")
    config_diff = write_config_diff_artifact(cfg, resolved_cfg, run_dir)
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    num_classes = model_cfg.get("num_classes", 64)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    seq_length_student = model_cfg.get("seq_length_student", 8)
    seq_length_teacher = model_cfg.get("seq_length_teacher", seq_length_student)
    g2d_enabled = _g2d_enabled(cfg)

    student_model = build_model(model_cfg["student"]).to(device)
    teacher_model = None
    checkpoint_loads = []
    stage_checkpoint_load = _load_stage_checkpoint_if_needed(cfg, student_model, device)
    if stage_checkpoint_load is not None:
        checkpoint_loads.append(stage_checkpoint_load)
    teacher_prior_info = _apply_teacher_prior_initialization(cfg, student_model, device)
    selective_finetune_info = _apply_selective_finetune_if_needed(cfg, student_model)
    if teacher_prior_info is not None:
        teacher_prior_info["encoder_freeze"] = _encoder_freeze_log(student_model)
        if selective_finetune_info is not None:
            teacher_prior_info["selective_finetune"] = selective_finetune_info
        teacher_prior_info["trainable_params"] = trainable_parameter_count(student_model)
    if _teacher_enabled(cfg) and not g2d_enabled:
        teacher_model = build_model(model_cfg["teacher"]).to(device)
        teacher_load_info = _load_teacher_if_needed(cfg, teacher_model, device)
        if teacher_load_info is not None:
            checkpoint_loads.append(teacher_load_info)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    task_criterion = build_task_criterion(cfg)
    distiller = build_distiller(cfg, task_criterion).to(device)
    optimizer = build_optimizer(cfg, student_model)
    _add_distiller_params_to_optimizer(optimizer, distiller, cfg)
    scheduler = build_scheduler(cfg, optimizer)
    optimizer_groups = optimizer_param_group_summary(optimizer)
    configure_csi_debug(student_model, cfg)
    if teacher_model is not None:
        configure_csi_debug(teacher_model, cfg)
    startup_summary = build_startup_summary(cfg, student_model, optimizer, scheduler, device=device)
    write_startup_summary(run_dir, startup_summary)
    print_startup_summary(startup_summary)
    health_tracker = ModuleHealthTracker(student_model) if training_health_debug_enabled(cfg) else None
    csi_debug_records: list[dict] = []
    grad_scaler = make_grad_scaler(cfg, amp_enabled)
    extension_context = ExtensionContext(
        cfg=cfg,
        task=task,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        student_model=student_model,
        teacher_model=teacher_model,
        distiller=distiller,
        task_criterion=task_criterion,
        run_dir=run_dir,
        device=device,
        num_pred=num_pred,
        num_classes=num_classes,
        seq_length_student=seq_length_student,
        seq_length_teacher=seq_length_teacher,
        non_blocking=non_blocking,
    )
    extensions = _build_training_extensions(cfg)
    extension_states = [extension.setup(extension_context) for extension in extensions]
    for extension, state in zip(extensions, extension_states):
        checkpoint_loads.extend(extension.checkpoint_loads(state))
    best_val_loss = float("inf")
    best_val_top1 = float("-inf")
    best_top1_epoch = 0
    best_early_stopping_value = _initial_early_stopping_value(early_stopping_mode)
    best_early_stopping_epoch = 0
    registry_checkpoint = None
    epochs_without_improvement = 0
    history = {field: [] for field in objective_history_fields(objective, include_compat=True)}
    epoch_logs = []

    tensorboard_writer = _create_tensorboard_writer(cfg, run_dir)
    progress_enabled = _progress_enabled(cfg)
    start_epoch = training_cfg.get("start_epoch", 0)
    total_epochs = training_cfg.get("epochs", 100)
    early_stopping_min_epoch = _early_stopping_min_epoch(total_epochs)
    resume_path = _resolve_resume_checkpoint(cfg, run_dir)
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path,
            student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict=_checkpoint_strict(cfg),
            role="resume",
            map_location=device,
        )
        checkpoint_loads.append(checkpoint.get("_load_info"))
        start_epoch = int(checkpoint.get("epoch", start_epoch))
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("test_loss", best_val_loss)))
        best_val_top1 = float(checkpoint.get("best_val_top1", best_val_top1))
        best_top1_epoch = int(checkpoint.get("best_top1_epoch", best_top1_epoch))
        if "early_stopping_metric" in checkpoint:
            early_stopping_metric = _normalize_early_stopping_metric(checkpoint["early_stopping_metric"], objective=objective)
        if "early_stopping_mode" in checkpoint:
            early_stopping_mode = _resolve_early_stopping_mode(early_stopping_metric, checkpoint["early_stopping_mode"])
        training_cfg["early_stopping_metric"] = early_stopping_metric
        training_cfg["early_stopping_mode"] = early_stopping_mode
        best_early_stopping_value = float(
            checkpoint.get(
                "best_early_stopping_value",
                _legacy_early_stopping_value(checkpoint, early_stopping_metric, best_early_stopping_value),
            )
        )
        best_early_stopping_epoch = int(
            checkpoint.get(
                "best_early_stopping_epoch",
                _legacy_early_stopping_epoch(checkpoint, early_stopping_metric, best_early_stopping_epoch),
            )
        )
        registry_checkpoint = checkpoint.get("checkpoint_registry", registry_checkpoint)
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
    try:
        epoch_progress = tqdm(
            range(start_epoch, total_epochs),
            desc="Training",
            unit="epoch",
            disable=not progress_enabled,
        )
        for epoch in epoch_progress:
            _set_epoch_recursive(student_model, epoch)
            if teacher_model is not None:
                _set_epoch_recursive(teacher_model, epoch)
            student_model.train()
            running_loss = 0.0
            running_task_loss = 0.0
            running_distill_loss = 0.0
            running_beam_soft_loss = 0.0
            running_unimodal_loss = 0.0
            running_counterfactual_loss = 0.0
            running_prior_regularization_loss = 0.0
            running_reliability_kd_loss = 0.0
            running_occlusion_loss = 0.0
            running_position_loss = 0.0
            running_multitask_loss = 0.0
            running_acc = 0.0
            epoch_diagnostics = EpochDiagnosticsAccumulator()
            train_lidar_quality = LidarQualityAccumulator()
            saw_train_lidar = False
            batch_count = 0
            current_alpha = cfg["distillation"].get("alpha", 0.4)
            warmup_epochs = cfg["distillation"].get("alpha_warmup_epochs", 0)
            if warmup_epochs and epoch < warmup_epochs:
                current_alpha = current_alpha * (epoch / warmup_epochs)
            for extension, state in zip(extensions, extension_states):
                extension.before_epoch(extension_context, state, epoch=epoch)
            if health_tracker is not None:
                health_tracker.start_epoch()
            current_lr = optimizer.param_groups[0]["lr"]
            history["learning_rates"].append(current_lr)

            if progress_enabled:
                batch_progress = tqdm(
                    dataloaders["train"],
                    desc=f"Epoch {epoch + 1}/{total_epochs}",
                    unit="batch",
                    leave=False,
                )
            else:
                batch_progress = dataloaders["train"]
            for step, raw_batch in enumerate(batch_progress):
                batch_count = step + 1
                batch = prepare_task_batch(raw_batch)
                if "lidar" in batch:
                    saw_train_lidar = True
                    train_lidar_quality.update(batch["lidar"], raw_lidar=batch.get("lidar_raw"))
                labels = prepare_task_labels(
                    batch,
                    num_pred=num_pred,
                    downsample_ratio=downsample_ratio,
                    device=device,
                    non_blocking=non_blocking,
                )
                auxiliary_targets = prepare_task_auxiliary_targets(
                    batch,
                    num_pred=num_pred,
                    device=device,
                    non_blocking=non_blocking,
                )
                prediction_targets = prepare_prediction_targets(
                    labels=labels,
                    auxiliary_targets=auxiliary_targets,
                    cfg=cfg,
                )
                optimizer.zero_grad()
                with autocast_context(amp_enabled, device, amp_dtype):
                    controls = ForwardControls()
                    for extension, state in zip(extensions, extension_states):
                        controls = controls.merge(
                            extension.before_forward(
                                extension_context,
                                state,
                                batch,
                                labels,
                                epoch=epoch,
                            )
                        )
                    set_csi_debug_batch_source(student_model, "train")
                    student_step = run_model_step(
                        student_model,
                        task,
                        batch,
                        model_cfg=model_cfg["student"],
                        seq_length=seq_length_student,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                        force_modality_mask=controls.force_modality_mask,
                        force_reliability_gate=controls.force_reliability_gate,
                        gate_temperature=controls.gate_temperature,
                    )
                    student_model_output = student_step.model_output
                    student_outputs = student_step.logits
                    student_input_features = student_model_output.input_features
                    student_out_features = student_model_output.output_features
                    batch_state = BatchState(
                        epoch=epoch,
                        step=step,
                        batch=batch,
                        labels=labels,
                        student_output=student_model_output,
                        student_logits=student_outputs,
                        controls=controls,
                    )
                    base_loss: BaseLossResult | None = None
                    for extension, state in zip(extensions, extension_states):
                        extension_loss = extension.compute_base_loss(extension_context, state, batch_state)
                        if extension_loss is None:
                            continue
                        if base_loss is not None:
                            raise RuntimeError("Only one training extension may provide the base distillation loss.")
                        base_loss = extension_loss

                    if base_loss is None:
                        if teacher_model is not None:
                            with torch.no_grad():
                                set_csi_debug_batch_source(teacher_model, "train")
                                teacher_step = run_model_step(
                                    teacher_model,
                                    task,
                                    batch,
                                    model_cfg=model_cfg["teacher"],
                                    seq_length=seq_length_teacher,
                                    num_pred=num_pred,
                                    device=device,
                                    non_blocking=non_blocking,
                                )
                                teacher_model_output = teacher_step.model_output
                                teacher_outputs = teacher_step.logits
                                teacher_input_features = teacher_model_output.input_features
                                teacher_out_features = teacher_model_output.output_features
                                teacher_diagnostics = teacher_model_output.diagnostics
                        else:
                            teacher_outputs, teacher_input_features, teacher_out_features = _dummy_teacher(
                                student_outputs,
                                student_input_features,
                                student_out_features,
                            )
                            teacher_diagnostics = {}
                        batch_state.teacher_logits = teacher_outputs
                        batch_state.teacher_input_features = teacher_input_features
                        batch_state.teacher_output_features = teacher_out_features
                        batch_state.teacher_diagnostics = teacher_diagnostics
                        student_logits = student_outputs.reshape(-1, num_classes)
                        teacher_logits = teacher_outputs.reshape(-1, num_classes)
                        targets = labels.flatten()
                        student_input_window = _feature_prefix(
                            student_input_features,
                            seq_length_student - 1,
                            name="student input_features",
                        )
                        teacher_input_window = _feature_prefix(
                            teacher_input_features,
                            seq_length_teacher - 1,
                            name="teacher input_features",
                        )
                        student_output_window = _feature_tail(
                            student_out_features,
                            num_pred,
                            name="student output_features",
                        )
                        teacher_output_window = _feature_tail(
                            teacher_out_features,
                            num_pred,
                            name="teacher output_features",
                        )
                        total_loss, task_loss, distill_loss = distiller(
                            student_logits,
                            teacher_logits,
                            targets,
                            student_input_window,
                            teacher_input_window,
                            student_output_window,
                            teacher_output_window,
                            current_alpha,
                        )
                        base_loss = BaseLossResult(
                            total_loss=total_loss,
                            task_loss=task_loss,
                            distill_loss=distill_loss,
                            teacher_diagnostics=teacher_diagnostics,
                        )
                    else:
                        batch_state.teacher_diagnostics = base_loss.teacher_diagnostics

                    total_loss = base_loss.total_loss
                    task_loss = base_loss.task_loss
                    distill_loss = base_loss.distill_loss
                    batch_state.total_loss = total_loss
                    batch_state.task_loss = task_loss
                    batch_state.distill_loss = distill_loss
                    batch_state.active_modalities = base_loss.active_modalities
                    scalar_diagnostics = dict(base_loss.diagnostics)
                    extra_loss_values = {
                        "beam_soft": student_outputs.sum() * 0.0,
                        "unimodal": student_outputs.sum() * 0.0,
                        "counterfactual": student_outputs.sum() * 0.0,
                        "prior_regularization": student_outputs.sum() * 0.0,
                        "reliability_kd": student_outputs.sum() * 0.0,
                    }
                    for extension, state in zip(extensions, extension_states):
                        bundle = extension.after_forward(extension_context, state, batch_state)
                        if bundle is None:
                            continue
                        total_loss = total_loss + bundle.total
                        for key in extra_loss_values:
                            if key in bundle.components:
                                extra_loss_values[key] = bundle.components[key]
                        scalar_diagnostics.update(bundle.diagnostics)
                    prediction_loss = compute_prediction_loss(
                        student_model_output,
                        prediction_targets,
                        cfg,
                        reference=student_outputs,
                        beam_total_loss=total_loss,
                        beam_task_loss=task_loss,
                    )
                    total_loss = prediction_loss.total
                    task_loss = prediction_loss.primary
                    if objective not in {"beam", "multitask"}:
                        distill_loss = student_outputs.sum() * 0.0
                    scalar_diagnostics.update(prediction_loss.diagnostics)
                    batch_state.total_loss = total_loss
                    batch_state.task_loss = task_loss
                    batch_state.distill_loss = distill_loss
                grad_clip = training_cfg.get("grad_clip", None)
                if grad_scaler.is_enabled():
                    grad_scaler.scale(total_loss).backward()
                    if grad_clip or batch_state.active_modalities is not None or health_tracker is not None:
                        grad_scaler.unscale_(optimizer)
                    for extension, state in zip(extensions, extension_states):
                        extension.after_backward(extension_context, state, batch_state)
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), grad_clip)
                    if health_tracker is not None:
                        health_tracker.observe_gradients()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    total_loss.backward()
                    for extension, state in zip(extensions, extension_states):
                        extension.after_backward(extension_context, state, batch_state)
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), grad_clip)
                    if health_tracker is not None:
                        health_tracker.observe_gradients()
                    optimizer.step()
                csi_debug_records.extend(consume_csi_debug_records(student_model))
                prediction = torch.argmax(student_outputs, dim=-1)
                valid = torch.sum(labels != -100).item()
                acc = (prediction == labels).sum().item() / max(valid, 1)
                running_loss = (total_loss.item() + step * running_loss) / (step + 1)
                running_task_loss = (task_loss.item() + step * running_task_loss) / (step + 1)
                running_distill_loss = (distill_loss.item() + step * running_distill_loss) / (step + 1)
                running_beam_soft_loss = (
                    extra_loss_values["beam_soft"].item() + step * running_beam_soft_loss
                ) / (step + 1)
                running_unimodal_loss = (
                    extra_loss_values["unimodal"].item() + step * running_unimodal_loss
                ) / (step + 1)
                running_counterfactual_loss = (
                    extra_loss_values["counterfactual"].item() + step * running_counterfactual_loss
                ) / (step + 1)
                running_prior_regularization_loss = (
                    extra_loss_values["prior_regularization"].item() + step * running_prior_regularization_loss
                ) / (step + 1)
                running_reliability_kd_loss = (
                    extra_loss_values["reliability_kd"].item() + step * running_reliability_kd_loss
                ) / (step + 1)
                running_occlusion_loss = (prediction_loss.occlusion.item() + step * running_occlusion_loss) / (step + 1)
                running_position_loss = (prediction_loss.position.item() + step * running_position_loss) / (step + 1)
                running_multitask_loss = (
                    prediction_loss.multitask_total.item() + step * running_multitask_loss
                ) / (step + 1)
                running_acc = (acc + step * running_acc) / (step + 1)
                epoch_diagnostics.update(scalar_diagnostics)
                if progress_enabled:
                    batch_progress.set_postfix(
                        loss=f"{running_loss:.4f}",
                        task=f"{running_task_loss:.4f}",
                        distill=f"{running_distill_loss:.4f}",
                        acc=f"{running_acc:.4f}",
                        lr=f"{current_lr:.2e}",
                    )

            if scheduler is not None:
                scheduler.step()
            try:
                val_metrics = validate(
                    student_model,
                    dataloaders["test"],
                    cfg,
                    task_criterion,
                    device,
                    output_dir=run_dir,
                )
            finally:
                shutdown_dataloader_workers(dataloaders["test"])
            csi_debug_records.extend(consume_csi_debug_records(student_model))
            _validate_early_stopping_source_available(val_metrics, early_stopping_metric)
            val_loss = val_metrics["loss"]
            top1 = val_metrics["topk"].get("1", [0.0])
            val_acc = float(top1[0]) if top1 else 0.0
            validation_curve_metrics = _aggregate_validation_metrics(val_metrics)
            val_occlusion_accuracy = _finite_float_or_none(val_metrics.get("val_occlusion_accuracy"))
            val_occlusion_blocked_f1 = _finite_float_or_none(val_metrics.get("val_occlusion_blocked_f1"))
            val_position_rmse = _finite_float_or_none(val_metrics.get("val_position_rmse"))
            val_position_mae = _finite_float_or_none(val_metrics.get("val_position_mae"))
            val_multitask_loss = _finite_float_or_none(val_metrics.get("val_multitask_loss"))
            active_occlusion = val_occlusion_accuracy is not None or val_occlusion_blocked_f1 is not None
            active_position = val_position_rmse is not None or val_position_mae is not None
            train_occlusion_loss = (
                float(running_occlusion_loss) if objective in {"occlusion", "multitask"} or active_occlusion else None
            )
            train_position_loss = (
                float(running_position_loss) if objective in {"position", "multitask"} or active_position else None
            )
            train_multitask_loss = (
                float(running_multitask_loss)
                if objective == "multitask" or (objective == "beam" and (active_occlusion or active_position))
                else None
            )
            history["train_loss"].append(float(running_loss))
            history["train_task_loss"].append(float(running_task_loss))
            history["train_objective_loss"].append(float(running_task_loss))
            history["train_distill_loss"].append(float(running_distill_loss))
            history["train_beam_soft_loss"].append(float(running_beam_soft_loss))
            history["train_unimodal_loss"].append(float(running_unimodal_loss))
            history["train_counterfactual_loss"].append(float(running_counterfactual_loss))
            history["train_prior_regularization_loss"].append(float(running_prior_regularization_loss))
            history["train_reliability_kd_loss"].append(float(running_reliability_kd_loss))
            history["train_occlusion_loss"].append(train_occlusion_loss)
            history["train_position_loss"].append(train_position_loss)
            history["train_multitask_loss"].append(train_multitask_loss)
            history["train_acc"].append(float(running_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(val_acc)
            history["val_atop3"].append(validation_curve_metrics["val_atop3"])
            history["val_atop5"].append(validation_curve_metrics["val_atop5"])
            history["val_adba"].append(validation_curve_metrics["val_adba"])
            history["val_occlusion_accuracy"].append(val_occlusion_accuracy)
            history["val_occlusion_blocked_f1"].append(val_occlusion_blocked_f1)
            history["val_position_rmse"].append(val_position_rmse)
            history["val_position_mae"].append(val_position_mae)
            history["val_multitask_loss"].append(val_multitask_loss)
            early_stopping_candidates = {
                **validation_curve_metrics,
                "val_loss": float(val_loss),
                "val_acc": val_acc,
            }
            for key, value in {
                "val_occlusion_accuracy": val_occlusion_accuracy,
                "val_occlusion_blocked_f1": val_occlusion_blocked_f1,
                "val_position_rmse": val_position_rmse,
                "val_position_mae": val_position_mae,
                "val_multitask_loss": val_multitask_loss,
            }.items():
                if value is not None:
                    early_stopping_candidates[key] = value
            primary_metric_value = _early_stopping_metric_value(
                early_stopping_candidates,
                early_stopping_metric,
            )
            history["val_primary_metric"].append(float(primary_metric_value))
            epoch_log = {
                "epoch": epoch + 1,
                "total_epochs": total_epochs,
                "train_batches": batch_count,
                "objective": objective,
                "primary_loss": objective_metadata["primary_loss"],
                "primary_metric": early_stopping_metric,
                "primary_metric_mode": early_stopping_mode,
                "enabled_targets": objective_metadata["enabled_targets"],
                "enabled_heads": objective_metadata["enabled_heads"],
                "loss_weights": objective_metadata.get("loss_weights", {}),
                "train_loss": float(running_loss),
                "train_task_loss": float(running_task_loss),
                "train_objective_loss": float(running_task_loss),
                "train_distill_loss": float(running_distill_loss),
                "train_beam_soft_loss": float(running_beam_soft_loss),
                "train_unimodal_loss": float(running_unimodal_loss),
                "train_counterfactual_loss": float(running_counterfactual_loss),
                "train_prior_regularization_loss": float(running_prior_regularization_loss),
                "train_reliability_kd_loss": float(running_reliability_kd_loss),
                "train_occlusion_loss": train_occlusion_loss,
                "train_position_loss": train_position_loss,
                "train_multitask_loss": train_multitask_loss,
                "loss/occlusion": train_occlusion_loss,
                "loss/position": train_position_loss,
                "loss/multitask_total": train_multitask_loss,
                "train_acc": float(running_acc),
                "val_loss": float(val_loss),
                "val_acc": val_acc,
                "val_atop3": validation_curve_metrics["val_atop3"],
                "val_atop5": validation_curve_metrics["val_atop5"],
                "val_adba": validation_curve_metrics["val_adba"],
                "val_occlusion_accuracy": val_occlusion_accuracy,
                "val_occlusion_blocked_f1": val_occlusion_blocked_f1,
                "val_position_rmse": val_position_rmse,
                "val_position_mae": val_position_mae,
                "val_multitask_loss": val_multitask_loss,
                "val_primary_metric": float(primary_metric_value),
                "learning_rate": float(current_lr),
                "validation_metrics": deepcopy(val_metrics),
            }
            if saw_train_lidar:
                train_dataset = getattr(dataloaders["train"], "dataset", None)
                epoch_log["lidar_input_quality_train"] = train_lidar_quality.finalize(
                    split=getattr(train_dataset, "split", "train"),
                    preprocessing=lidar_preprocessing_metadata_from_dataset(train_dataset),
                )
            for group in optimizer_groups:
                group_name = group["name"]
                epoch_log[f"optimizer/lr/{group_name}"] = float(group["lr"])
                epoch_log[f"optimizer/params/{group_name}"] = float(group["param_count"])
            if teacher_prior_info is not None:
                epoch_log["teacher/trainable_params"] = float(teacher_prior_info.get("trainable_params", 0))
                for modality, freeze_info in teacher_prior_info.get("encoder_freeze", {}).items():
                    epoch_log[f"teacher/trainable_params/{modality}"] = float(
                        freeze_info.get("trainable_params", 0)
                    )
                    epoch_log[f"teacher/frozen/{modality}"] = 1.0 if freeze_info.get("frozen") else 0.0
            if health_tracker is not None:
                epoch_log.update(health_tracker.finish_epoch())
            epoch_log.update(_validation_subset_epoch_scalars(val_metrics))
            epoch_log.update(epoch_diagnostics.mean())
            for extension, state in zip(extensions, extension_states):
                epoch_log.update(extension.after_epoch(extension_context, state, epoch=epoch))
            early_stopping_value = _early_stopping_metric_value(epoch_log, early_stopping_metric)
            if float(val_loss) < best_val_loss:
                best_val_loss = float(val_loss)
            improved = _early_stopping_improved(
                early_stopping_value,
                best_early_stopping_value,
                mode=early_stopping_mode,
                min_delta=float(training_cfg.get("min_delta", 0.0)),
            )
            if improved:
                best_early_stopping_value = early_stopping_value
                best_early_stopping_epoch = epoch + 1
                epochs_without_improvement = 0
                best_objective_path = run_dir / "checkpoints" / "best.pth"
                torch.save(student_model.state_dict(), best_objective_path)
                write_sidecar(
                    best_objective_path,
                    {
                        "path": str(best_objective_path),
                        "source": "local",
                        "checkpoint_source": "objective-checkpoint",
                        "run_dir": str(run_dir),
                        "selection_metric": early_stopping_metric,
                        "selection_mode": "early_stopping",
                        "selected_epoch": best_early_stopping_epoch,
                        "objective_metric": {
                            "name": early_stopping_metric,
                            "mode": early_stopping_mode,
                            "value": early_stopping_value,
                        },
                        "task_metrics": _checkpoint_task_metrics(epoch_log),
                        "normalization_artifacts": normalization_artifacts,
                        "split_metadata": split_metadata,
                        "task": cfg.get("experiment", {}).get("task"),
                        "enabled_modalities": list(
                            getattr(dataloaders["train"].dataset, "enabled_modalities", [])
                        ),
                    },
                )
            else:
                epochs_without_improvement += 1
            top1_improved = val_acc > best_val_top1
            if top1_improved:
                best_val_top1 = val_acc
                best_top1_epoch = epoch + 1
                best_top1_path = run_dir / "checkpoints" / "best_top1.pth"
                torch.save(student_model.state_dict(), best_top1_path)
                registry_checkpoint = archive_best_checkpoint(
                    cfg,
                    source_checkpoint=best_top1_path,
                    val_top1=best_val_top1,
                    epoch=best_top1_epoch,
                    run_dir=run_dir,
                    split_metadata=split_metadata,
                    normalization_artifacts=normalization_artifacts,
                    objective_metric={
                        "name": early_stopping_metric,
                        "mode": early_stopping_mode,
                        "value": early_stopping_value,
                    },
                    task_metrics=_checkpoint_task_metrics(epoch_log),
                    selection_metric="val_acc_top1",
                    selection_mode="top1-selection",
                    checkpoint_source="top1-checkpoint",
                )
                write_sidecar(
                    best_top1_path,
                    {
                        "path": str(best_top1_path),
                        "source": "local",
                        "checkpoint_source": "top1-checkpoint",
                        "run_dir": str(run_dir),
                        "selection_metric": "val_acc_top1",
                        "selection_mode": "top1-selection",
                        "selected_epoch": best_top1_epoch,
                        "objective_metric": {
                            "name": early_stopping_metric,
                            "mode": early_stopping_mode,
                            "value": early_stopping_value,
                        },
                        "task_metrics": _checkpoint_task_metrics(epoch_log),
                        "normalization_artifacts": normalization_artifacts,
                        "split_metadata": split_metadata,
                        "task": cfg.get("experiment", {}).get("task"),
                        "enabled_modalities": list(
                            getattr(dataloaders["train"].dataset, "enabled_modalities", [])
                        ),
                    },
                )
            epoch_log.update(
                {
                    "early_stopping_metric": early_stopping_metric,
                    "early_stopping_mode": early_stopping_mode,
                    "early_stopping_value": early_stopping_value,
                    "early_stopping_improved": bool(improved),
                    "best_early_stopping_value": best_early_stopping_value,
                    "best_early_stopping_epoch": best_early_stopping_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                }
            )
            epoch_logs.append(epoch_log)
            if progress_enabled:
                epoch_progress.set_postfix(
                    train_loss=f"{running_loss:.4f}",
                    val_loss=f"{float(val_loss):.4f}",
                    val_acc=f"{val_acc:.4f}",
                    early_stop=f"{early_stopping_metric}:{early_stopping_value:.4f}",
                    lr=f"{current_lr:.2e}",
                )
            _write_tensorboard_scalars(
                tensorboard_writer,
                history,
                epoch + 1,
                objective=objective,
                tensorboard_cfg=cfg.get("output", {}).get("tensorboard", {}),
            )
            _write_tensorboard_craf_scalars(tensorboard_writer, epoch_log, epoch + 1)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": student_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "test_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_val_top1": best_val_top1,
                    "best_top1_epoch": best_top1_epoch,
                    "early_stopping_metric": early_stopping_metric,
                    "early_stopping_mode": early_stopping_mode,
                    "best_early_stopping_value": best_early_stopping_value,
                    "best_early_stopping_epoch": best_early_stopping_epoch,
                    "epochs_without_improvement": epochs_without_improvement,
                    "normalization_artifacts": normalization_artifacts,
                    "checkpoint_registry": registry_checkpoint,
                    "prediction_objective": objective_metadata,
                },
                run_dir / "checkpoints",
                "last.pth",
            )
            if (
                not improved
                and training_cfg.get("use_early_stopping", True)
                and epoch + 1 >= early_stopping_min_epoch
                and epochs_without_improvement >= training_cfg.get("patience", 20)
            ):
                break
    finally:
        for dataloader in dataloaders.values():
            shutdown_dataloader_workers(dataloader)
        _close_tensorboard_writer(tensorboard_writer)

    np.savez(run_dir / "training_outputs.npz", **_training_outputs_payload(history, objective_metadata, early_stopping_metric, early_stopping_mode))
    teacher_metrics = teacher_metrics_from_training(
        cfg,
        history,
        epoch_logs,
        best_selected_epoch=best_early_stopping_epoch,
        selection_metric=early_stopping_metric,
        selection_mode="early_stopping",
        checkpoint="checkpoints/best.pth",
        best_top1_epoch=best_top1_epoch,
    )
    if teacher_metrics is not None:
        with (run_dir / "teacher_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(teacher_metrics, f, indent=2)
    early_stopping_metadata = _early_stopping_state(
        metric=early_stopping_metric,
        mode=early_stopping_mode,
        best_value=best_early_stopping_value,
        best_epoch=best_early_stopping_epoch,
        epochs_without_improvement=epochs_without_improvement,
    )
    write_csi_debug_records(run_dir, csi_debug_records)
    pilot_noise_validity = evaluate_pilot_noise_validity(cfg, csi_debug_records)
    write_pilot_noise_validity_artifact(run_dir, pilot_noise_validity)
    train_log = {
        **history,
        "epoch_logs": epoch_logs,
        "early_stopping": early_stopping_metadata,
        "startup_summary": startup_summary,
        "config_diff": config_diff,
        "csi_first_batch_diagnostics": csi_debug_records,
        "pilot_noise_validity": pilot_noise_validity,
        "teacher_metrics": teacher_metrics,
        "checkpoint_loads": checkpoint_loads,
        "teacher_prior": teacher_prior_info,
        "optimizer_param_groups": optimizer_groups,
        "normalization_artifacts": normalization_artifacts,
        "checkpoint_registry": registry_checkpoint,
        "throughput": throughput_metadata,
        "prediction_objective": objective_metadata,
        "prediction_setup": prediction_setup_metadata(cfg, split_metadata=split_metadata),
        "runtime": {
            "run_dir": str(run_dir),
            "output_overwrite": bool(cfg.get("output", {}).get("overwrite", False)),
            "splits": split_metadata,
            "normalization_artifacts": normalization_artifacts,
            "checkpoint_registry": registry_checkpoint,
            "throughput": throughput_metadata,
            "teacher_prior": teacher_prior_info,
            "early_stopping": early_stopping_metadata,
            "startup_summary": startup_summary,
            "config_diff": config_diff,
            "pilot_noise_validity": pilot_noise_validity,
            "prediction_objective": objective_metadata,
            "prediction_setup": prediction_setup_metadata(cfg, split_metadata=split_metadata),
        },
    }
    with (run_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)
    plot_training_curves(history, run_dir)
    dump_config(
        final_config_with_runtime(
            cfg,
            run_dir=run_dir,
            split_metadata=split_metadata,
            normalization_artifacts=normalization_artifacts,
            checkpoint_registry=registry_checkpoint,
            throughput_metadata=throughput_metadata,
            teacher_prior=teacher_prior_info,
            early_stopping=early_stopping_metadata,
            pilot_noise_validity=pilot_noise_validity,
        ),
        run_dir / "final_config.yaml",
    )
    return {
        "run_dir": str(run_dir),
        "history": history,
        "epoch_logs": epoch_logs,
        "best_val_loss": best_val_loss,
        "best_val_top1": best_val_top1,
        "early_stopping": early_stopping_metadata,
        "best_early_stopping_value": best_early_stopping_value,
        "best_early_stopping_epoch": best_early_stopping_epoch,
        "checkpoint_registry": registry_checkpoint,
        "normalization_artifacts": normalization_artifacts,
        "checkpoint_loads": checkpoint_loads,
        "teacher_prior": teacher_prior_info,
        "optimizer_param_groups": optimizer_groups,
        "split_metadata": split_metadata,
        "throughput": throughput_metadata,
        "prediction_objective": objective_metadata,
        "startup_summary": startup_summary,
        "config_diff": config_diff,
        "csi_first_batch_diagnostics": csi_debug_records,
    }


def _apply_csi_rms_to_model_config(cfg: dict, dataloaders: dict) -> None:
    train_loader = dataloaders.get("train")
    dataset = getattr(train_loader, "dataset", None) if train_loader is not None else None
    normalizer = getattr(dataset, "csi_rms_normalizer", None)
    if normalizer is None:
        return
    rms = float(getattr(normalizer, "rms", normalizer))
    model_cfg = cfg.setdefault("model", {})
    model_cfg["csi_train_rms"] = rms
    for role in ("teacher", "student"):
        role_cfg = model_cfg.get(role)
        if not isinstance(role_cfg, dict):
            continue
        if "csi" not in role_cfg.get("modalities", []):
            continue
        role_cfg["csi_train_rms"] = rms
        encoders = role_cfg.get("encoders")
        if isinstance(encoders, dict):
            csi_cfg = encoders.get("csi")
            if isinstance(csi_cfg, dict):
                csi_cfg.setdefault("train_rms", rms)


def _set_epoch_recursive(module, epoch: int) -> None:
    setter = getattr(module, "set_epoch", None)
    if callable(setter):
        setter(int(epoch))
    children = getattr(module, "children", None)
    if not callable(children):
        return
    for child in children():
        _set_epoch_recursive(child, epoch)


def _training_outputs_payload(
    history: dict[str, list],
    objective_metadata: dict,
    early_stopping_metric: str,
    early_stopping_mode: str,
) -> dict[str, np.ndarray]:
    payload = {key: _history_array(key, value) for key, value in history.items()}
    payload["objective"] = np.asarray(objective_metadata["name"])
    payload["primary_loss"] = np.asarray(objective_metadata["primary_loss"])
    payload["primary_metric"] = np.asarray(early_stopping_metric)
    payload["primary_metric_mode"] = np.asarray(early_stopping_mode)
    payload["enabled_targets"] = np.asarray(objective_metadata["enabled_targets"], dtype=object)
    payload["enabled_heads"] = np.asarray(objective_metadata["enabled_heads"], dtype=object)
    weight_names = ("beam", "occlusion", "position")
    weights = objective_metadata.get("loss_weights", {})
    payload["loss_weight_names"] = np.asarray(weight_names, dtype=object)
    payload["loss_weights"] = np.asarray([float(weights.get(name, np.nan)) for name in weight_names], dtype=float)
    return payload


_OPTIONAL_HISTORY_KEYS = objective_optional_history_fields()


def _history_array(key: str, values: list) -> np.ndarray:
    if key not in _OPTIONAL_HISTORY_KEYS:
        return np.asarray(values)
    return np.asarray([np.nan if _finite_float_or_none(value) is None else float(value) for value in values], dtype=float)


def _checkpoint_task_metrics(epoch_log: dict) -> dict[str, float]:
    keys = (
        "val_acc",
        "val_adba",
        "val_loss",
        "val_occlusion_accuracy",
        "val_occlusion_blocked_f1",
        "val_position_rmse",
        "val_position_mae",
        "val_multitask_loss",
    )
    return {
        key: float(epoch_log[key])
        for key in keys
        if key in epoch_log and isinstance(epoch_log[key], (int, float))
    }


def _dummy_teacher(
    student_outputs: torch.Tensor,
    student_input_features: torch.Tensor | None,
    student_out_features: torch.Tensor | None,
):
    return (
        torch.zeros_like(student_outputs),
        torch.zeros_like(student_input_features) if torch.is_tensor(student_input_features) else None,
        torch.zeros_like(student_out_features) if torch.is_tensor(student_out_features) else None,
    )


def _feature_prefix(features: torch.Tensor | None, length: int, *, name: str) -> torch.Tensor | None:
    if features is None:
        return None
    if features.ndim < 2:
        raise ValueError(f"{name} must include a time dimension, got shape {tuple(features.shape)}.")
    if features.shape[1] < length:
        raise ValueError(f"{name} has {features.shape[1]} slots but {length} are required.")
    return features[:, :length, ...]


def _feature_tail(features: torch.Tensor | None, length: int, *, name: str) -> torch.Tensor | None:
    if features is None:
        return None
    if features.ndim < 2:
        raise ValueError(f"{name} must include a time dimension, got shape {tuple(features.shape)}.")
    if features.shape[1] < length:
        raise ValueError(f"{name} has {features.shape[1]} slots but {length} are required.")
    return features[:, -length:, ...]
