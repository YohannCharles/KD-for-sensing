from __future__ import annotations

import datetime as dt
from pathlib import Path

import torch
from tqdm.auto import tqdm

from kd_sensing.data.scenes import scene_slug_from_config
from kd_sensing.engine.artifacts import ArtifactWriter, final_config_with_runtime
from kd_sensing.engine.batch_step import (
    BatchStepRunner,
)
from kd_sensing.engine.checkpointing import (
    CheckpointManager,
    checkpoint_strict as _checkpoint_strict,
    resolve_resume_checkpoint as _resolve_resume_checkpoint,
)
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.debug_diagnostics import (
    ModuleHealthTracker,
    build_startup_summary,
    configure_csi_debug,
    consume_csi_debug_records,
    print_startup_summary,
    training_health_debug_enabled,
    write_config_diff_artifact,
    write_startup_summary,
)
from kd_sensing.engine.epoch_subsampling import epoch_subsampling_epoch_log, set_train_sampler_epoch
from kd_sensing.engine.normalization_artifacts import save_normalization_artifacts
from kd_sensing.engine.optim import (
    build_device,
    build_model,
    build_optimizer,
    build_scheduler,
    build_task_criterion,
    optimizer_param_group_summary,
)
from kd_sensing.engine.objectives.metadata import (
    objective_runtime_metadata,
    resolve_prediction_objective,
)
from kd_sensing.engine.run_metadata import (
    dataloaders_run_metadata,
    throughput_run_metadata,
)
from kd_sensing.engine.run_status import (
    write_complete_status,
    write_failed_status_for_active_run,
    write_running_status,
)
from kd_sensing.engine.runtime import (
    configure_torch_runtime_threads,
    make_grad_scaler,
    resolve_amp_settings,
    transfer_non_blocking,
)
from kd_sensing.engine.tensorboard_logging import (
    close_tensorboard_writer as _close_tensorboard_writer,
    create_tensorboard_writer as _create_tensorboard_writer,
    write_tensorboard_scalars as _write_tensorboard_scalars,
    write_tensorboard_startup_scalars as _write_tensorboard_startup_scalars,
)
from kd_sensing.engine.training_extensions import (
    ExtensionContext,
    NoOpTrainingExtension,
    TrainingExtension,
)
from kd_sensing.engine.training_metrics import EpochMetricsRecorder
from kd_sensing.engine.validator import validate
from kd_sensing.engine.training_state import (
    TrainingState,
    available_early_stopping_metrics as _available_early_stopping_metrics,
    configure_early_stopping as _configure_early_stopping,
    early_stopping_improved as _early_stopping_improved,
    early_stopping_metric_value as _early_stopping_metric_value,
    early_stopping_min_epoch as _early_stopping_min_epoch,
    early_stopping_state as _early_stopping_state,
    initial_early_stopping_value as _initial_early_stopping_value,
    legacy_early_stopping_epoch as _legacy_early_stopping_epoch,
    legacy_early_stopping_value as _legacy_early_stopping_value,
    normalize_early_stopping_metric as _normalize_early_stopping_metric,
    resolve_early_stopping_mode as _resolve_early_stopping_mode,
    validate_early_stopping_source_available as _validate_early_stopping_source_available,
)
from kd_sensing.evaluation.lidar_diagnostics import LidarQualityAccumulator
from kd_sensing.utils.paths import output_dir as resolve_output_dir, resolve_path
from kd_sensing.utils.seed import set_seed


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


def _scene_grouped_output_base(cfg: dict) -> Path:
    base = resolve_output_dir(cfg.get("output", {}).get("dir", cfg.get("paths", {}).get("output_dir", "outputs")))
    output_cfg = cfg.get("output", {})
    if output_cfg.get("group_by_scene", True) is False:
        return base
    scene_slug = scene_slug_from_config(cfg)
    if not scene_slug or base.name == scene_slug:
        return base
    return base / scene_slug


def _build_training_extensions(cfg: dict) -> list[TrainingExtension]:
    return [NoOpTrainingExtension()]


def _progress_enabled(cfg: dict) -> bool:
    return cfg.get("output", {}).get("progress", {}).get("enabled", True)


def train(cfg: dict) -> dict:
    try:
        return _train_inner(cfg)
    except Exception as exc:
        try:
            write_failed_status_for_active_run(cfg, exc, kind="training")
        except Exception:
            pass
        raise


def _train_inner(cfg: dict) -> dict:
    configure_torch_runtime_threads(cfg)
    set_seed(cfg.get("experiment", {}).get("seed", 0))
    objective = resolve_prediction_objective(cfg)
    cfg.setdefault("experiment", {})["objective"] = objective
    objective_metadata = objective_runtime_metadata(cfg)
    training_cfg = cfg.setdefault("training", {})
    early_stopping_metric, early_stopping_mode = _configure_early_stopping(training_cfg, objective=objective)
    if cfg.get("training", {}).get("resume") is True and not cfg.get("output", {}).get("run_name"):
        raise ValueError("training.resume=true requires output.run_name so checkpoints/last.pth can be resolved.")

    run_dir = create_run_dir(cfg)
    write_running_status(run_dir, cfg, kind="training")
    artifact_writer = ArtifactWriter(cfg=cfg, run_dir=run_dir)
    dataloaders = build_dataloaders(cfg)
    _apply_csi_rms_to_model_config(cfg, dataloaders)
    split_metadata = dataloaders_run_metadata(dataloaders)
    normalization_artifacts = save_normalization_artifacts(dataloaders, run_dir)
    device = build_device(cfg)
    throughput_metadata = throughput_run_metadata(cfg, dataloaders, device)
    resolved_cfg = artifact_writer.write_initial_configs(
        split_metadata=split_metadata,
        normalization_artifacts=normalization_artifacts,
        throughput_metadata=throughput_metadata,
    )
    config_diff = write_config_diff_artifact(cfg, resolved_cfg, run_dir)
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    num_classes = model_cfg.get("num_classes", 64)
    seq_length = model_cfg.get("seq_length", 8)

    primary_model = build_model(model_cfg["primary"]).to(device)
    state = TrainingState(
        start_epoch=training_cfg.get("start_epoch", 0),
        best_early_stopping_value=_initial_early_stopping_value(early_stopping_mode),
    )

    task_criterion = build_task_criterion(cfg)
    optimizer = build_optimizer(cfg, primary_model)
    scheduler = build_scheduler(cfg, optimizer)
    optimizer_groups = optimizer_param_group_summary(optimizer)
    configure_csi_debug(primary_model, cfg)
    startup_summary = build_startup_summary(cfg, primary_model, optimizer, scheduler, device=device)
    write_startup_summary(run_dir, startup_summary)
    print_startup_summary(startup_summary)
    health_tracker = ModuleHealthTracker(primary_model) if training_health_debug_enabled(cfg) else None
    csi_debug_records: list[dict] = []
    grad_scaler = make_grad_scaler(cfg, amp_enabled)
    extension_context = ExtensionContext(
        cfg=cfg,
        task=task,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        primary_model=primary_model,
        task_criterion=task_criterion,
        run_dir=run_dir,
        device=device,
        num_pred=num_pred,
        num_classes=num_classes,
        seq_length=seq_length,
        non_blocking=non_blocking,
    )
    extensions = _build_training_extensions(cfg)
    extension_states = [extension.setup(extension_context) for extension in extensions]
    for extension, extension_state in zip(extensions, extension_states):
        state.checkpoint_loads.extend(extension.checkpoint_loads(extension_state))

    recorder = EpochMetricsRecorder(
        objective=objective,
        objective_metadata=objective_metadata,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode,
    )
    state.history = recorder.history
    state.epoch_logs = recorder.epoch_logs
    checkpoint_manager = CheckpointManager(
        cfg=cfg,
        run_dir=run_dir,
        primary_model=primary_model,
        optimizer=optimizer,
        scheduler=scheduler,
        split_metadata=split_metadata,
        normalization_artifacts=normalization_artifacts,
        objective_metadata=objective_metadata,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode,
    )
    early_stopping_metric, early_stopping_mode = checkpoint_manager.restore_if_needed(
        state,
        objective=objective,
        device=device,
    )
    training_cfg["early_stopping_metric"] = early_stopping_metric
    training_cfg["early_stopping_mode"] = early_stopping_mode
    recorder.update_early_stopping(metric=early_stopping_metric, mode=early_stopping_mode)
    batch_runner = BatchStepRunner(
        cfg=cfg,
        task=task,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        optimizer=optimizer,
        grad_scaler=grad_scaler,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        extension_context=extension_context,
        extensions=extensions,
        extension_states=extension_states,
        health_tracker=health_tracker,
    )

    tensorboard_writer = _create_tensorboard_writer(cfg, run_dir)
    _write_tensorboard_startup_scalars(tensorboard_writer, startup_summary)
    progress_enabled = _progress_enabled(cfg)
    total_epochs = training_cfg.get("epochs", 100)
    early_stopping_min_epoch = _early_stopping_min_epoch(total_epochs)
    try:
        epoch_progress = tqdm(
            range(state.start_epoch, total_epochs),
            desc="Training",
            unit="epoch",
            disable=not progress_enabled,
        )
        for epoch in epoch_progress:
            _set_epoch_recursive(primary_model, epoch)
            set_train_sampler_epoch(dataloaders["train"], epoch)
            primary_model.train()
            train_lidar_quality = LidarQualityAccumulator()
            saw_train_lidar = False
            current_alpha = 0.0
            for extension, extension_state in zip(extensions, extension_states):
                extension.before_epoch(extension_context, extension_state, epoch=epoch)
            if health_tracker is not None:
                health_tracker.start_epoch()
            current_lr = optimizer.param_groups[0]["lr"]
            recorder.start_epoch(current_lr)

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
                batch_result = batch_runner.run(raw_batch, epoch=epoch, step=step, current_alpha=current_alpha)
                if "lidar" in batch_result.batch:
                    saw_train_lidar = True
                    train_lidar_quality.update(batch_result.batch["lidar"], raw_lidar=batch_result.batch.get("lidar_raw"))
                csi_debug_records.extend(consume_csi_debug_records(primary_model))
                progress_metrics = recorder.update_batch(batch_result, step)
                if progress_enabled:
                    batch_progress.set_postfix(
                        loss=f"{progress_metrics['loss']:.4f}",
                        task=f"{progress_metrics['task']:.4f}",
                        acc=f"{progress_metrics['acc']:.4f}",
                        lr=f"{current_lr:.2e}",
                    )

            if scheduler is not None:
                scheduler.step()
            try:
                val_metrics = validate(
                    primary_model,
                    dataloaders["test"],
                    cfg,
                    task_criterion,
                    device,
                    output_dir=run_dir,
                )
            finally:
                shutdown_dataloader_workers(dataloaders["test"])
            csi_debug_records.extend(consume_csi_debug_records(primary_model))
            _validate_early_stopping_source_available(val_metrics, early_stopping_metric)
            extension_metrics = {}
            for extension, extension_state in zip(extensions, extension_states):
                extension_metrics.update(extension.after_epoch(extension_context, extension_state, epoch=epoch))
            health_metrics = health_tracker.finish_epoch() if health_tracker is not None else None
            train_dataset = getattr(dataloaders["train"], "dataset", None)
            epoch_subsampling_log = epoch_subsampling_epoch_log(dataloaders["train"])
            epoch_log, val_loss, val_acc, _ = recorder.finish_epoch(
                epoch=epoch,
                total_epochs=total_epochs,
                val_metrics=val_metrics,
                current_lr=current_lr,
                optimizer_groups=optimizer_groups,
                train_lidar_quality=train_lidar_quality if saw_train_lidar else None,
                train_dataset=train_dataset,
                epoch_subsampling=epoch_subsampling_log,
                health_metrics=health_metrics,
                extension_metrics=extension_metrics,
            )
            checkpoint_update = checkpoint_manager.update_best_checkpoints(
                state=state,
                epoch=epoch,
                epoch_log=epoch_log,
                val_loss=val_loss,
                val_acc=val_acc,
                train_dataset=train_dataset,
            )
            epoch_log.update(
                {
                    "early_stopping_metric": early_stopping_metric,
                    "early_stopping_mode": early_stopping_mode,
                    "early_stopping_value": checkpoint_update.early_stopping_value,
                    "early_stopping_improved": bool(checkpoint_update.improved),
                    "best_early_stopping_value": state.best_early_stopping_value,
                    "best_early_stopping_epoch": state.best_early_stopping_epoch,
                    "epochs_without_improvement": state.epochs_without_improvement,
                }
            )
            if progress_enabled:
                metrics = recorder.progress_metrics()
                epoch_progress.set_postfix(
                    train_loss=f"{metrics['loss']:.4f}",
                    val_loss=f"{float(val_loss):.4f}",
                    val_acc=f"{val_acc:.4f}",
                    early_stop=f"{early_stopping_metric}:{checkpoint_update.early_stopping_value:.4f}",
                    lr=f"{current_lr:.2e}",
                )
            _write_tensorboard_scalars(
                tensorboard_writer,
                state.history,
                epoch + 1,
                objective=objective,
                tensorboard_cfg=cfg.get("output", {}).get("tensorboard", {}),
            )
            checkpoint_manager.save_last_checkpoint(state=state, epoch=epoch, val_loss=val_loss)
            if (
                not checkpoint_update.improved
                and training_cfg.get("use_early_stopping", True)
                and epoch + 1 >= early_stopping_min_epoch
                and state.epochs_without_improvement >= training_cfg.get("patience", 20)
            ):
                break
    finally:
        for dataloader in dataloaders.values():
            shutdown_dataloader_workers(dataloader)
        _close_tensorboard_writer(tensorboard_writer)

    final_artifacts = artifact_writer.write_final_artifacts(
        history=state.history,
        epoch_logs=state.epoch_logs,
        objective_metadata=objective_metadata,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode,
        best_early_stopping_value=state.best_early_stopping_value,
        best_early_stopping_epoch=state.best_early_stopping_epoch,
        epochs_without_improvement=state.epochs_without_improvement,
        checkpoint_loads=state.checkpoint_loads,
        optimizer_groups=optimizer_groups,
        normalization_artifacts=normalization_artifacts,
        checkpoint_registry=state.registry_checkpoint,
        throughput_metadata=throughput_metadata,
        split_metadata=split_metadata,
        startup_summary=startup_summary,
        config_diff=config_diff,
        csi_debug_records=csi_debug_records,
        best_top1_epoch=state.best_top1_epoch,
    )
    write_complete_status(
        run_dir,
        cfg,
        kind="training",
        primary_metric={
            "name": early_stopping_metric,
            "mode": early_stopping_mode,
            "value": float(state.best_early_stopping_value),
            "epoch": int(state.best_early_stopping_epoch),
        },
        metrics_path=run_dir / "metrics.json",
        best_checkpoint=_best_checkpoint_for_status(run_dir, state.registry_checkpoint),
    )
    return {
        "run_dir": str(run_dir),
        "history": state.history,
        "epoch_logs": state.epoch_logs,
        "best_val_loss": state.best_val_loss,
        "best_val_top1": state.best_val_top1,
        "early_stopping": final_artifacts["early_stopping"],
        "best_early_stopping_value": state.best_early_stopping_value,
        "best_early_stopping_epoch": state.best_early_stopping_epoch,
        "checkpoint_registry": state.registry_checkpoint,
        "normalization_artifacts": normalization_artifacts,
        "checkpoint_loads": state.checkpoint_loads,
        "optimizer_param_groups": optimizer_groups,
        "split_metadata": split_metadata,
        "throughput": throughput_metadata,
        "prediction_objective": objective_metadata,
        "startup_summary": startup_summary,
        "config_diff": config_diff,
        "csi_first_batch_diagnostics": csi_debug_records,
    }


def _best_checkpoint_for_status(run_dir: Path, registry_checkpoint: dict | None) -> Path | str | None:
    if isinstance(registry_checkpoint, dict) and registry_checkpoint.get("path"):
        return str(registry_checkpoint["path"])
    for name in ("best.pth", "best_top1.pth", "last.pth"):
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    return None


def _apply_csi_rms_to_model_config(cfg: dict, dataloaders: dict) -> None:
    train_loader = dataloaders.get("train")
    dataset = getattr(train_loader, "dataset", None) if train_loader is not None else None
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
