import datetime as dt
from pathlib import Path


from kd_sensing.engine.artifacts import ArtifactWriter
from kd_sensing.engine.batch_step import BatchStepRunner
from kd_sensing.engine.checkpointing import CheckpointManager, resolve_resume_checkpoint
from kd_sensing.engine.data_factory import build_dataloaders, final_test_enabled
from kd_sensing.data.mmw.trajectory_protocol import TRAJECTORY_PROTOCOL_MODE
from kd_sensing.engine.debug_diagnostics import (
    build_startup_summary,
    print_startup_summary,
    write_startup_summary,
)
from kd_sensing.engine.model_initialization import initialize_model_from_checkpoint
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    save_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.objectives.metadata import objective_runtime_metadata
from kd_sensing.engine.optim import (
    build_device,
    build_model,
    build_optimizer,
    build_scheduler,
    build_task_criterion,
    optimizer_param_group_summary,
)
from kd_sensing.engine.run_metadata import dataloaders_run_metadata, throughput_run_metadata
from kd_sensing.engine.run_status import write_complete_status, write_failed_status_for_active_run, write_running_status
from kd_sensing.engine.runtime import (
    configure_cuda_performance_settings,
    configure_torch_runtime_threads,
    make_grad_scaler,
    resolve_amp_settings,
    transfer_non_blocking,
)
from kd_sensing.engine.trainer_runtime_helpers import (
    _evaluate_final_test_split,
    run_training_epoch_loop,
    shutdown_all_dataloaders,
)
from kd_sensing.engine.training_extensions import ExtensionContext, NoOpTrainingExtension, TrainingExtension
from kd_sensing.engine.training_metrics import EpochMetricsRecorder
from kd_sensing.engine.training_run_context import TrainingRunContext
from kd_sensing.engine.training_state import TrainingState
from kd_sensing.engine.validator import validate
from kd_sensing.utils.paths import output_dir as resolve_output_dir
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.runtime_output_layout import evaluation_output_base
from kd_sensing.utils.seed import set_seed


def create_run_dir(cfg: dict) -> Path:
    base = resolve_output_dir(cfg["output"]["dir"])
    run_name = cfg.get("output", {}).get("run_name") or f"{cfg.get('experiment', {}).get('name', 'run')}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    path = base / run_name
    if path.exists() and not cfg.get("output", {}).get("overwrite", False) and not cfg.get("training", {}).get("resume"):
        path = _unique_run_path(path)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    return path


def create_eval_run_dir(cfg: dict, output_dir: str | None = None) -> Path:
    if output_dir:
        path = resolve_output_dir(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    root = resolve_output_dir(cfg["output"]["dir"])
    base = evaluation_output_base(root, cfg)
    run_name = cfg.get("output", {}).get("evaluation_run_name") or cfg.get("output", {}).get("run_name") or cfg.get("experiment", {}).get("name", "run")
    path = base / f"evaluation_{run_name}_{dt.datetime.now():%Y%m%d_%H%M%S}"
    path = _unique_run_path(path) if path.exists() else path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unique_run_path(path: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}_{timestamp}_{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to create a unique run directory for {path}.")


def _build_training_extensions(cfg: dict) -> list[TrainingExtension]:
    u_mask_cfg = cfg.get("loss", {}).get("u_mask_beam_jepa", {}) if isinstance(cfg.get("loss"), dict) else {}
    pcpf_cfg = cfg.get("loss", {}).get("pcpf_temporal_risk", {}) if isinstance(cfg.get("loss"), dict) else {}
    u_mask_enabled = isinstance(u_mask_cfg, dict) and bool(u_mask_cfg.get("enabled", False))
    pcpf_enabled = isinstance(pcpf_cfg, dict) and bool(pcpf_cfg.get("enabled", False))
    if u_mask_enabled and pcpf_enabled:
        raise ValueError("U0 and PCPF-T training extensions are mutually exclusive.")
    if pcpf_enabled:
        from kd_sensing.losses.pcpf_temporal_risk import PCPFTemporalRiskTrainingExtension

        return [PCPFTemporalRiskTrainingExtension()]
    if u_mask_enabled:
        from kd_sensing.losses.u_mask_beam_jepa import UMaskBeamJEPATrainingExtension

        return [UMaskBeamJEPATrainingExtension()]
    return [NoOpTrainingExtension()]


def train(cfg: dict) -> dict:
    try:
        context = _prepare_training_run_context(cfg)
        _build_training_resources(context)
        _restore_training_state(context)
        _run_training_loop(context)
        return _finalize_training_run(context)
    except Exception as exc:
        try:
            write_failed_status_for_active_run(cfg, exc, kind="training")
        except Exception:
            pass
        raise


def _prepare_training_run_context(cfg: dict) -> TrainingRunContext:
    configure_torch_runtime_threads(cfg)
    _print_mmw_split_binding(cfg)
    set_seed(cfg.get("experiment", {}).get("seed", 0))
    training_cfg = cfg.setdefault("training", {})
    if training_cfg.get("resume") is True and not cfg.get("output", {}).get("run_name"):
        raise ValueError("training.resume=true requires output.run_name so checkpoints/last.pth can be resolved.")
    run_dir = create_run_dir(cfg)
    write_running_status(run_dir, cfg, kind="training")
    artifact_writer = ArtifactWriter(cfg=cfg, run_dir=run_dir)
    resume_checkpoint = resolve_resume_checkpoint(cfg, run_dir)
    resume_metadata = load_checkpoint_metadata(resume_checkpoint) if resume_checkpoint is not None else None
    configured_artifacts = cfg.get("data", {}).get("normalization_artifacts")
    # Exact resume restores the checkpoint artifacts; the resume contract still
    # verifies that the resolved config has not drifted.
    normalization_metadata = (
        resume_metadata
        if resume_metadata is not None
        else {"normalization_artifacts": configured_artifacts}
        if configured_artifacts
        else None
    )
    validate_normalization_artifact_fingerprint(cfg, normalization_metadata)
    normalization_overrides = load_normalization_artifacts(normalization_metadata)
    dataloaders = build_dataloaders(cfg, normalization_overrides=normalization_overrides or None)
    recorded_split_metadata = (resume_metadata or {}).get("split_metadata")
    split_metadata = (
        dict(recorded_split_metadata)
        if isinstance(recorded_split_metadata, dict) and normalization_overrides
        else dataloaders_run_metadata(dataloaders)
    )
    if resume_metadata is not None and normalization_overrides:
        normalization_artifacts = dict((resume_metadata or {}).get("normalization_artifacts") or {})
    else:
        normalization_artifacts = save_normalization_artifacts(dataloaders, run_dir)
    device = build_device(cfg)
    cuda_performance = configure_cuda_performance_settings(cfg, device)
    throughput_metadata = throughput_run_metadata(cfg, dataloaders, device)
    if cuda_performance:
        throughput_metadata["cuda_performance"] = cuda_performance
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    artifact_writer.write_initial_configs(
        split_metadata=split_metadata,
        normalization_artifacts=normalization_artifacts,
        throughput_metadata=throughput_metadata,
    )
    return TrainingRunContext(
        cfg=cfg,
        objective_metadata=objective_runtime_metadata(),
        training_cfg=training_cfg,
        run_dir=run_dir,
        artifact_writer=artifact_writer,
        dataloaders=dataloaders,
        split_metadata=split_metadata,
        normalization_artifacts=normalization_artifacts,
        device=device,
        throughput_metadata=throughput_metadata,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        task=cfg["experiment"].get("task", "fusion"),
        model_cfg=cfg["model"],
        num_pred=cfg["model"].get("num_pred", 1),
        num_classes=cfg["model"].get("num_classes", 64),
        seq_length=cfg["model"].get("seq_length", 5),
        validation_loader=dataloaders.get("validation"),
    )


def _build_training_resources(context: TrainingRunContext) -> None:
    context.primary_model = build_model(context.model_cfg["primary"]).to(context.device)
    initialization_load = initialize_model_from_checkpoint(
        context.primary_model,
        context.training_cfg,
        map_location="cpu",
    )
    context.state = TrainingState(
        start_epoch=0 if initialization_load is not None else int(context.training_cfg.get("start_epoch", 0))
    )
    if initialization_load is not None:
        context.state.checkpoint_loads.append(initialization_load)
        context.cfg.setdefault("runtime", {})["model_initialization"] = initialization_load
    prepare_stage = getattr(context.primary_model, "prepare_training_stage", None)
    if callable(prepare_stage):
        preparation = prepare_stage(
            cfg=context.cfg,
            train_loader=context.dataloaders.get("train"),
            device=context.device,
            run_dir=context.run_dir,
            non_blocking=context.non_blocking,
        )
        context.cfg.setdefault("runtime", {})["pcpf_stage_preparation"] = preparation
    context.task_criterion = build_task_criterion(context.cfg)
    context.optimizer = build_optimizer(context.cfg, context.primary_model)
    context.scheduler = build_scheduler(context.cfg, context.optimizer)
    context.optimizer_groups = optimizer_param_group_summary(context.optimizer)
    context.startup_summary = build_startup_summary(
        context.cfg,
        context.primary_model,
        context.optimizer,
        context.scheduler,
        device=context.device,
    )
    write_startup_summary(context.run_dir, context.startup_summary)
    print_startup_summary(context.startup_summary)
    context.grad_scaler = make_grad_scaler(context.cfg, context.amp_enabled)
    context.extension_context = ExtensionContext(
        cfg=context.cfg,
        task=context.task,
        model_cfg=context.model_cfg,
        training_cfg=context.training_cfg,
        primary_model=context.primary_model,
        task_criterion=context.task_criterion,
        run_dir=context.run_dir,
        device=context.device,
        num_pred=context.num_pred,
        num_classes=context.num_classes,
        seq_length=context.seq_length,
        non_blocking=context.non_blocking,
    )
    context.extensions = _build_training_extensions(context.cfg)
    context.extension_states = [extension.setup(context.extension_context) for extension in context.extensions]
    for extension, state in zip(context.extensions, context.extension_states):
        context.state.checkpoint_loads.extend(extension.checkpoint_loads(state))
    context.recorder = EpochMetricsRecorder(objective_metadata=context.objective_metadata)
    context.state.history = context.recorder.history
    context.state.epoch_logs = context.recorder.epoch_logs
    context.checkpoint_manager = CheckpointManager(
        cfg=context.cfg,
        run_dir=context.run_dir,
        primary_model=context.primary_model,
        optimizer=context.optimizer,
        scheduler=context.scheduler,
        split_metadata=context.split_metadata,
        normalization_artifacts=context.normalization_artifacts,
        objective_metadata=context.objective_metadata,
        dataloaders=context.dataloaders,
        grad_scaler=context.grad_scaler,
        extensions=context.extensions,
        extension_states=context.extension_states,
    )


def _restore_training_state(context: TrainingRunContext) -> None:
    context.checkpoint_manager.restore_if_needed(context.state, device=context.device)
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
    )
    context.progress_enabled = bool(context.cfg.get("output", {}).get("progress", {}).get("enabled", True))
    context.total_epochs = int(context.training_cfg.get("epochs", 40))


def _run_training_loop(context: TrainingRunContext) -> None:
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
            task_criterion=context.task_criterion,
            device=context.device,
            run_dir=context.run_dir,
            training_cfg=context.training_cfg,
            optimizer_groups=context.optimizer_groups,
            progress_enabled=context.progress_enabled,
            total_epochs=context.total_epochs,
            validation_loader=context.validation_loader,
            validate_fn=validate,
        )
    finally:
        shutdown_all_dataloaders(context.dataloaders)


def _finalize_training_run(context: TrainingRunContext) -> dict:
    if final_test_enabled(context.cfg):
        test_loader = context.dataloaders.get("test")
        if test_loader is None:
            raise RuntimeError("Final test is enabled but the test dataloader was not constructed.")
        context.final_test_metrics, context.final_test_checkpoint_load = _evaluate_final_test_split(
            context.primary_model,
            test_loader,
            context.cfg,
            context.task_criterion,
            context.device,
            run_dir=context.run_dir,
        )
    else:
        context.final_test_metrics = {
            "status": "not_run",
            "reason": "training.final_test.enabled=false",
        }
        context.final_test_checkpoint_load = None
    if context.final_test_checkpoint_load is not None:
        context.state.checkpoint_loads.append(context.final_test_checkpoint_load)
    context.final_artifacts = context.artifact_writer.write_final_artifacts(
        history=context.state.history,
        epoch_logs=context.state.epoch_logs,
        objective_metadata=context.objective_metadata,
        checkpoint_loads=context.state.checkpoint_loads,
        optimizer_groups=context.optimizer_groups,
        normalization_artifacts=context.normalization_artifacts,
        throughput_metadata=context.throughput_metadata,
        split_metadata=context.split_metadata,
        startup_summary=context.startup_summary,
        final_test_metrics=context.final_test_metrics,
    )
    write_complete_status(
        context.run_dir,
        context.cfg,
        kind="training",
        metrics_path=context.run_dir / "metrics.json",
        checkpoint=context.run_dir / "checkpoints" / "last.pth",
    )
    return {
        "run_dir": str(context.run_dir),
        "history": context.state.history,
        "epoch_logs": context.state.epoch_logs,
        "normalization_artifacts": context.normalization_artifacts,
        "checkpoint_loads": context.state.checkpoint_loads,
        "final_test_metrics": context.final_test_metrics,
        "optimizer_param_groups": context.optimizer_groups,
        "split_metadata": context.split_metadata,
        "throughput": context.throughput_metadata,
        "prediction_objective": context.objective_metadata,
        "startup_summary": context.startup_summary,
        "data_protocol": dict(context.cfg.get("data_protocol", {})),
    }


def _print_mmw_split_binding(cfg: dict) -> None:
    protocol = cfg.get("data_protocol")
    if not isinstance(protocol, dict) or protocol.get("mode") != TRAJECTORY_PROTOCOL_MODE:
        return
    print(f"MMW split protocol: {protocol['protocol_id']}", flush=True)
    print(f"Protocol version: {int(protocol['protocol_version'])}", flush=True)
    print(f"Split seed: {int(protocol['split_seed'])}", flush=True)
    print(f"Train seed: {int(protocol.get('train_seed', cfg.get('experiment', {}).get('seed', 0)))}", flush=True)
    print(f"Block size: {int(protocol['block_size'])}", flush=True)
    print(f"Split manifest: {protocol.get('split_manifest', protocol.get('path'))}", flush=True)
    print(f"Train windows: {int(protocol['train_sample_count'])}", flush=True)
    print(f"Validation windows: {int(protocol['validation_sample_count'])}", flush=True)
    print(f"Test loaded: {str(bool(protocol.get('evaluate_test_requested', False))).lower()}", flush=True)
    print(f"Leakage validation: {protocol.get('leakage_validation', 'UNKNOWN')}", flush=True)
