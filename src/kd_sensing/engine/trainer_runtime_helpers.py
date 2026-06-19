from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from kd_sensing.engine.checkpointing import checkpoint_strict as _checkpoint_strict
from kd_sensing.engine.data_factory import shutdown_dataloader_workers
from kd_sensing.engine.debug_diagnostics import consume_csi_debug_records
from kd_sensing.engine.epoch_subsampling import epoch_subsampling_epoch_log, set_train_sampler_epoch
from kd_sensing.engine.tensorboard_logging import write_tensorboard_scalars as _write_tensorboard_scalars
from kd_sensing.engine.training_state import (
    validate_early_stopping_source_available as _validate_early_stopping_source_available,
)
from kd_sensing.engine.validator import validate
from kd_sensing.evaluation.lidar_diagnostics import LidarQualityAccumulator
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state


def run_training_epoch_loop(
    *,
    cfg: dict,
    dataloaders: dict[str, Any],
    primary_model,
    optimizer,
    scheduler,
    batch_runner,
    recorder,
    checkpoint_manager,
    state,
    extensions,
    extension_states,
    extension_context,
    health_tracker,
    csi_debug_records: list[dict],
    tensorboard_writer,
    objective: str,
    task_criterion,
    device,
    run_dir: Path,
    training_cfg: dict,
    early_stopping_metric: str,
    early_stopping_mode: str,
    optimizer_groups,
    progress_enabled: bool,
    total_epochs: int,
    early_stopping_min_epoch: int,
    validation_loader,
    validate_fn=validate,
) -> None:
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
            val_metrics = validate_fn(
                primary_model,
                validation_loader,
                cfg,
                task_criterion,
                device,
                output_dir=run_dir,
            )
        finally:
            shutdown_dataloader_workers(validation_loader)
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


def _evaluate_final_test_split(
    primary_model,
    test_loader,
    cfg: dict,
    task_criterion,
    device,
    *,
    run_dir: Path,
    validation_split_name: str,
) -> tuple[dict, dict | None]:
    checkpoint_load = None
    best_path = run_dir / "checkpoints" / "best.pth"
    if best_path.exists():
        load_result = load_model_state(
            best_path,
            primary_model,
            role="final-test-best",
            map_location=device,
            strict=_checkpoint_strict(cfg),
        )
        checkpoint_load = checkpoint_load_summary(load_result)
    primary_model.eval()
    try:
        metrics = validate(primary_model, test_loader, cfg, task_criterion, device, output_dir=run_dir)
    finally:
        shutdown_dataloader_workers(test_loader)
    metrics["model_selection_split"] = str(validation_split_name)
    metrics["evaluation_split"] = "test"
    metrics["checkpoint_for_test"] = str(best_path) if best_path.exists() else "last_in_memory"
    return metrics, checkpoint_load

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


def shutdown_all_dataloaders(dataloaders: dict[str, Any]) -> None:
    for dataloader in dataloaders.values():
        shutdown_dataloader_workers(dataloader)


__all__ = [
    "_apply_csi_rms_to_model_config",
    "_evaluate_final_test_split",
    "_set_epoch_recursive",
    "run_training_epoch_loop",
    "shutdown_all_dataloaders",
]
