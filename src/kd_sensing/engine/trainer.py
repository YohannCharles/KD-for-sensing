import datetime as dt
from pathlib import Path

import torch

from kd_sensing.engine.artifacts import ArtifactWriter, final_config_with_runtime
from kd_sensing.engine.batch_step import (
    BatchStepRunner,
)
from kd_sensing.engine.checkpointing import CheckpointManager
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.debug_diagnostics import (
    ModuleHealthTracker,
    build_startup_summary,
    configure_csi_debug,
    print_startup_summary,
    training_health_debug_enabled,
    write_config_diff_artifact,
    write_startup_summary,
)
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
    configure_cuda_performance_settings,
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
from kd_sensing.engine.trainer_runtime_helpers import (
    _apply_csi_rms_to_model_config,
    _evaluate_final_test_split,
    run_training_epoch_loop,
    shutdown_all_dataloaders,
)
from kd_sensing.engine.training_run_context import TrainingRunContext
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
from kd_sensing.utils.runtime_output_layout import evaluation_output_base, scoped_output_base
from kd_sensing.utils.paths import output_dir as resolve_output_dir
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
    if cfg.get("training", {}).get("resume"):
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
    root = resolve_output_dir(cfg.get("output", {}).get("dir", cfg.get("paths", {}).get("output_dir", "outputs")))
    base = evaluation_output_base(root, cfg)
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
    return scoped_output_base(base, cfg, purpose="training")


def _build_training_extensions(cfg: dict) -> list[TrainingExtension]:
    u_mask_cfg = cfg.get("loss", {}).get("u_mask_beam_jepa", {}) if isinstance(cfg.get("loss"), dict) else {}
    if u_mask_cfg is True or (isinstance(u_mask_cfg, dict) and bool(u_mask_cfg.get("enabled", False))):
        from kd_sensing.losses.u_mask_beam_jepa import UMaskBeamJEPATrainingExtension

        return [UMaskBeamJEPATrainingExtension()]
    if resolve_prediction_objective(cfg) == "gps_conditioned_jepa":
        from kd_sensing.engine.jepa import JepaTrainingExtension

        return [JepaTrainingExtension()]
    physics_cfg = cfg.get("loss", {}).get("physics", {}) if isinstance(cfg.get("loss"), dict) else {}
    if isinstance(physics_cfg, dict) and bool(physics_cfg.get("enabled", False)):
        from kd_sensing.engine.physics_informed_extension import PhysicsInformedTrainingExtension

        return [PhysicsInformedTrainingExtension()]
    teacher_cfg = cfg.get("loss", {}).get("teacher_guidance", {}) if isinstance(cfg.get("loss"), dict) else {}
    teacher_enabled = teacher_cfg is True or (isinstance(teacher_cfg, dict) and bool(teacher_cfg.get("enabled", False)))
    if teacher_enabled:
        from kd_sensing.engine.teacher_guidance import TeacherGuidanceTrainingExtension

        return [TeacherGuidanceTrainingExtension()]
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
    context = _prepare_training_run_context(cfg)
    _build_training_resources(context)
    _restore_training_state(context)
    _run_training_loop_phase(context)
    return _finalize_training_run(context)


def _prepare_training_run_context(cfg: dict) -> TrainingRunContext:
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
    cuda_performance = configure_cuda_performance_settings(cfg, device)
    throughput_metadata = throughput_run_metadata(cfg, dataloaders, device)
    if cuda_performance:
        throughput_metadata["cuda_performance"] = cuda_performance
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
    return TrainingRunContext(
        cfg=cfg,
        objective=objective,
        objective_metadata=objective_metadata,
        training_cfg=training_cfg,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode,
        run_dir=run_dir,
        artifact_writer=artifact_writer,
        dataloaders=dataloaders,
        split_metadata=split_metadata,
        normalization_artifacts=normalization_artifacts,
        device=device,
        throughput_metadata=throughput_metadata,
        resolved_cfg=resolved_cfg,
        config_diff=config_diff,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        task=task,
        model_cfg=model_cfg,
        num_pred=num_pred,
        num_classes=num_classes,
        seq_length=seq_length,
    )


def _build_training_resources(context: TrainingRunContext) -> None:
    cfg = context.cfg
    training_cfg = context.training_cfg
    context.primary_model = build_model(context.model_cfg["primary"]).to(context.device)
    context.state = TrainingState(
        start_epoch=training_cfg.get("start_epoch", 0),
        best_early_stopping_value=_initial_early_stopping_value(context.early_stopping_mode),
    )
    context.task_criterion = build_task_criterion(cfg)
    context.optimizer = build_optimizer(cfg, context.primary_model)
    context.scheduler = build_scheduler(cfg, context.optimizer)
    context.optimizer_groups = optimizer_param_group_summary(context.optimizer)
    configure_csi_debug(context.primary_model, cfg)
    context.startup_summary = build_startup_summary(
        cfg,
        context.primary_model,
        context.optimizer,
        context.scheduler,
        device=context.device,
    )
    write_startup_summary(context.run_dir, context.startup_summary)
    print_startup_summary(context.startup_summary)
    context.health_tracker = ModuleHealthTracker(context.primary_model) if training_health_debug_enabled(cfg) else None
    context.grad_scaler = make_grad_scaler(cfg, context.amp_enabled)
    context.extension_context = ExtensionContext(
        cfg=cfg,
        task=context.task,
        model_cfg=context.model_cfg,
        training_cfg=training_cfg,
        primary_model=context.primary_model,
        task_criterion=context.task_criterion,
        run_dir=context.run_dir,
        device=context.device,
        num_pred=context.num_pred,
        num_classes=context.num_classes,
        seq_length=context.seq_length,
        non_blocking=context.non_blocking,
    )
    context.extensions = _build_training_extensions(cfg)
    context.extension_states = [
        extension.setup(context.extension_context)
        for extension in context.extensions
    ]
    for extension, extension_state in zip(context.extensions, context.extension_states):
        context.state.checkpoint_loads.extend(extension.checkpoint_loads(extension_state))
    context.recorder = EpochMetricsRecorder(
        objective=context.objective,
        objective_metadata=context.objective_metadata,
        early_stopping_metric=context.early_stopping_metric,
        early_stopping_mode=context.early_stopping_mode,
    )
    context.state.history = context.recorder.history
    context.state.epoch_logs = context.recorder.epoch_logs
    context.checkpoint_manager = CheckpointManager(
        cfg=cfg,
        run_dir=context.run_dir,
        primary_model=context.primary_model,
        optimizer=context.optimizer,
        scheduler=context.scheduler,
        split_metadata=context.split_metadata,
        normalization_artifacts=context.normalization_artifacts,
        objective_metadata=context.objective_metadata,
        early_stopping_metric=context.early_stopping_metric,
        early_stopping_mode=context.early_stopping_mode,
    )


def _restore_training_state(context: TrainingRunContext) -> None:
    context.early_stopping_metric, context.early_stopping_mode = context.checkpoint_manager.restore_if_needed(
        context.state,
        objective=context.objective,
        device=context.device,
    )
    context.training_cfg["early_stopping_metric"] = context.early_stopping_metric
    context.training_cfg["early_stopping_mode"] = context.early_stopping_mode
    context.recorder.update_early_stopping(
        metric=context.early_stopping_metric,
        mode=context.early_stopping_mode,
    )
    context.batch_runner = BatchStepRunner(
        cfg=context.cfg,
        task=context.task,
        model_cfg=context.model_cfg,
        training_cfg=context.training_cfg,
        optimizer=context.optimizer,
        grad_scaler=context.grad_scaler,
        amp_enabled=context.amp_enabled,
        amp_dtype=context.amp_dtype,
        extension_context=context.extension_context,
        extensions=context.extensions,
        extension_states=context.extension_states,
        health_tracker=context.health_tracker,
    )
    context.tensorboard_writer = _create_tensorboard_writer(context.cfg, context.run_dir)
    _write_tensorboard_startup_scalars(context.tensorboard_writer, context.startup_summary)
    context.progress_enabled = _progress_enabled(context.cfg)
    context.total_epochs = context.training_cfg.get("epochs", 100)
    context.early_stopping_min_epoch = _early_stopping_min_epoch(context.total_epochs)
    context.validation_loader = context.dataloaders.get("validation", context.dataloaders["test"])
    context.validation_split_name = "validation" if "validation" in context.dataloaders else "test"


def _run_training_loop_phase(context: TrainingRunContext) -> None:
    try:
        run_training_epoch_loop(
            cfg=context.cfg,
            dataloaders=context.dataloaders,
            primary_model=context.primary_model,
            optimizer=context.optimizer,
            scheduler=context.scheduler,
            batch_runner=context.batch_runner,
            recorder=context.recorder,
            checkpoint_manager=context.checkpoint_manager,
            state=context.state,
            extensions=context.extensions,
            extension_states=context.extension_states,
            extension_context=context.extension_context,
            health_tracker=context.health_tracker,
            csi_debug_records=context.csi_debug_records,
            tensorboard_writer=context.tensorboard_writer,
            objective=context.objective,
            task_criterion=context.task_criterion,
            device=context.device,
            run_dir=context.run_dir,
            training_cfg=context.training_cfg,
            early_stopping_metric=context.early_stopping_metric,
            early_stopping_mode=context.early_stopping_mode,
            optimizer_groups=context.optimizer_groups,
            progress_enabled=context.progress_enabled,
            total_epochs=context.total_epochs,
            early_stopping_min_epoch=context.early_stopping_min_epoch,
            validation_loader=context.validation_loader,
            validate_fn=validate,
        )
    finally:
        shutdown_all_dataloaders(context.dataloaders)
        _close_tensorboard_writer(context.tensorboard_writer)


def _finalize_training_run(context: TrainingRunContext) -> dict:
    context.final_test_metrics, context.final_test_checkpoint_load = _evaluate_final_test_split(
        context.primary_model,
        context.dataloaders["test"],
        context.cfg,
        context.task_criterion,
        context.device,
        run_dir=context.run_dir,
        validation_split_name=context.validation_split_name,
    )
    if context.final_test_checkpoint_load is not None:
        context.state.checkpoint_loads.append(context.final_test_checkpoint_load)

    context.final_artifacts = context.artifact_writer.write_final_artifacts(
        history=context.state.history,
        epoch_logs=context.state.epoch_logs,
        objective_metadata=context.objective_metadata,
        early_stopping_metric=context.early_stopping_metric,
        early_stopping_mode=context.early_stopping_mode,
        best_early_stopping_value=context.state.best_early_stopping_value,
        best_early_stopping_epoch=context.state.best_early_stopping_epoch,
        epochs_without_improvement=context.state.epochs_without_improvement,
        checkpoint_loads=context.state.checkpoint_loads,
        optimizer_groups=context.optimizer_groups,
        normalization_artifacts=context.normalization_artifacts,
        checkpoint_registry=context.state.registry_checkpoint,
        throughput_metadata=context.throughput_metadata,
        split_metadata=context.split_metadata,
        startup_summary=context.startup_summary,
        config_diff=context.config_diff,
        csi_debug_records=context.csi_debug_records,
        best_top1_epoch=context.state.best_top1_epoch,
        final_test_metrics=context.final_test_metrics,
        primary_model=context.primary_model,
    )
    write_complete_status(
        context.run_dir,
        context.cfg,
        kind="training",
        primary_metric={
            "name": context.early_stopping_metric,
            "mode": context.early_stopping_mode,
            "value": float(context.state.best_early_stopping_value),
            "epoch": int(context.state.best_early_stopping_epoch),
        },
        metrics_path=context.run_dir / "metrics.json",
        best_checkpoint=_best_checkpoint_for_status(context.run_dir, context.state.registry_checkpoint),
    )
    return {
        "run_dir": str(context.run_dir),
        "history": context.state.history,
        "epoch_logs": context.state.epoch_logs,
        "best_val_loss": context.state.best_val_loss,
        "best_val_top1": context.state.best_val_top1,
        "early_stopping": context.final_artifacts["early_stopping"],
        "best_early_stopping_value": context.state.best_early_stopping_value,
        "best_early_stopping_epoch": context.state.best_early_stopping_epoch,
        "checkpoint_registry": context.state.registry_checkpoint,
        "normalization_artifacts": context.normalization_artifacts,
        "checkpoint_loads": context.state.checkpoint_loads,
        "final_test_metrics": context.final_test_metrics,
        "optimizer_param_groups": context.optimizer_groups,
        "split_metadata": context.split_metadata,
        "throughput": context.throughput_metadata,
        "prediction_objective": context.objective_metadata,
        "startup_summary": context.startup_summary,
        "config_diff": context.config_diff,
        "csi_first_batch_diagnostics": context.csi_debug_records,
    }


def _best_checkpoint_for_status(run_dir: Path, registry_checkpoint: dict | None) -> Path | str | None:
    if isinstance(registry_checkpoint, dict) and registry_checkpoint.get("path"):
        return str(registry_checkpoint["path"])
    for name in ("best.pth", "best_top1.pth", "last.pth"):
        path = run_dir / "checkpoints" / name
        if path.exists():
            return path
    return None
