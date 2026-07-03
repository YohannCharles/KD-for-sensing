from pathlib import Path
import csv
import json
import math
import resource
import sys
import time
from typing import Any

import torch
from tqdm.auto import tqdm

from kd_sensing.engine.checkpointing import CheckpointUpdate
from kd_sensing.engine.checkpointing import checkpoint_strict as _checkpoint_strict
from kd_sensing.engine.data_factory import shutdown_dataloader_workers
from kd_sensing.engine.debug_diagnostics import consume_csi_debug_records
from kd_sensing.engine.epoch_subsampling import epoch_subsampling_epoch_log, set_train_sampler_epoch
from kd_sensing.engine.tensorboard_logging import write_tensorboard_scalars as _write_tensorboard_scalars
from kd_sensing.engine.training_state import (
    validate_early_stopping_source_available as _validate_early_stopping_source_available,
)
from kd_sensing.engine.validator import validate
from kd_sensing.eval.export import format_results_markdown, save_results_csv, save_results_json, save_results_markdown
from kd_sensing.eval.missing_patterns import resolve_missing_patterns
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import evaluate_missing_matrix
from kd_sensing.evaluation.lidar_diagnostics import LidarQualityAccumulator
from kd_sensing.utils.artifact_registry import sanitize_slug
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state

try:
    import psutil
except ImportError:  # pragma: no cover - optional local dependency
    psutil = None


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
    validation_interval = _validation_interval_epochs(training_cfg)
    last_val_metrics: dict[str, Any] | None = None
    timing_logger = _TimingCsvLogger(cfg, run_dir)
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
        data_wait_start = time.perf_counter()
        for step, raw_batch in enumerate(batch_progress):
            batch_start = time.perf_counter()
            data_time = batch_start - data_wait_start
            batch_result = batch_runner.run(raw_batch, epoch=epoch, step=step, current_alpha=current_alpha)
            step_time = time.perf_counter() - batch_start
            if "lidar" in batch_result.batch:
                saw_train_lidar = True
                train_lidar_quality.update(batch_result.batch["lidar"], raw_lidar=batch_result.batch.get("lidar_raw"))
            csi_debug_records.extend(consume_csi_debug_records(primary_model))
            progress_metrics = recorder.update_batch(batch_result, step)
            timing_logger.maybe_log(
                epoch=epoch,
                batch=step,
                data_time=data_time,
                step_time=step_time,
                batch_result=batch_result,
                lr=current_lr,
            )
            if progress_enabled:
                batch_progress.set_postfix(
                    loss=f"{progress_metrics['loss']:.4f}",
                    task=f"{progress_metrics['task']:.4f}",
                    acc=f"{progress_metrics['acc']:.4f}",
                    lr=f"{current_lr:.2e}",
                )
            data_wait_start = time.perf_counter()

        if scheduler is not None:
            scheduler.step()
        validation_ran = _should_validate_epoch(epoch, total_epochs, validation_interval) or last_val_metrics is None
        if validation_ran:
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
            last_val_metrics = val_metrics
        else:
            val_metrics = dict(last_val_metrics)
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
        if validation_ran:
            checkpoint_update = checkpoint_manager.update_best_checkpoints(
                state=state,
                epoch=epoch,
                epoch_log=epoch_log,
                val_loss=val_loss,
                val_acc=val_acc,
                train_dataset=train_dataset,
            )
        else:
            checkpoint_update = CheckpointUpdate(
                early_stopping_value=float(epoch_log["val_primary_metric"]),
                improved=False,
                top1_improved=False,
            )
        epoch_log.update(
            {
                "validation_ran": bool(validation_ran),
                "validation_interval_epochs": int(validation_interval),
                "early_stopping_metric": early_stopping_metric,
                "early_stopping_mode": early_stopping_mode,
                "early_stopping_value": checkpoint_update.early_stopping_value,
                "early_stopping_improved": bool(checkpoint_update.improved),
                "best_early_stopping_value": state.best_early_stopping_value,
                "best_early_stopping_epoch": state.best_early_stopping_epoch,
                "epochs_without_improvement": state.epochs_without_improvement,
            }
        )
        _write_epoch_metrics_snapshot(run_dir, state.epoch_logs)
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
            and validation_ran
            and training_cfg.get("use_early_stopping", True)
            and epoch + 1 >= early_stopping_min_epoch
            and state.epochs_without_improvement >= training_cfg.get("patience", 20)
        ):
            early_stop_payload = {
                "early_stopped": True,
                "early_stop_epoch": epoch + 1,
                "early_stop_metric": early_stopping_metric,
            }
            epoch_log.update(early_stop_payload)
            if state.epoch_logs:
                state.epoch_logs[-1].update(early_stop_payload)
            _write_epoch_metrics_snapshot(run_dir, state.epoch_logs)
            tqdm.write(
                f"Early stopping triggered at epoch {epoch + 1}: "
                f"{early_stopping_metric} did not improve for {state.epochs_without_improvement} epochs."
            )
            break


def _validation_interval_epochs(training_cfg: dict[str, Any]) -> int:
    validation_cfg = training_cfg.get("validation")
    if isinstance(validation_cfg, dict):
        raw = validation_cfg.get("interval_epochs", 1)
    else:
        raw = training_cfg.get("validation_interval_epochs", 1)
    return max(1, int(raw or 1))


def _should_validate_epoch(epoch: int, total_epochs: int, interval_epochs: int) -> bool:
    if int(interval_epochs) <= 1:
        return True
    epoch_number = int(epoch) + 1
    return epoch_number == 1 or epoch_number == int(total_epochs) or epoch_number % int(interval_epochs) == 0


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
        missing_pattern_results = _write_missing_pattern_eval(primary_model, test_loader, cfg, device, run_dir=run_dir)
    finally:
        shutdown_dataloader_workers(test_loader)
    metrics["model_selection_split"] = str(validation_split_name)
    metrics["evaluation_split"] = "test"
    metrics["checkpoint_for_test"] = str(best_path) if best_path.exists() else "last_in_memory"
    if missing_pattern_results:
        metrics["missing_patterns"] = missing_pattern_results
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


class _TimingCsvLogger:
    def __init__(self, cfg: dict, run_dir: Path) -> None:
        timing_cfg = cfg.get("training", {}).get("timing", {})
        if not isinstance(timing_cfg, dict):
            timing_cfg = {}
        self.enabled = bool(timing_cfg.get("enabled", True))
        self.log_interval = max(1, int(timing_cfg.get("log_interval", cfg.get("output", {}).get("log_interval", 10)) or 10))
        self.slow_seconds = float(timing_cfg.get("slow_batch_seconds", 20.0))
        exp_name = sanitize_slug(str(cfg.get("output", {}).get("run_name") or cfg.get("experiment", {}).get("name", "run")))
        self.path = run_dir.parent / "logs" / f"{exp_name}_timing.csv"
        self.global_step = 0
        self.fieldnames = [
            "epoch",
            "batch",
            "global_step",
            "data_time",
            "forward_time",
            "loss_time",
            "backward_time",
            "optimizer_step_time",
            "step_time",
            "loss",
            "lr",
            "gpu_mem_alloc_mb",
            "gpu_mem_reserved_mb",
            "cpu_rss_mb",
            "slow_batch",
        ]
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                with self.path.open("w", encoding="utf-8", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=self.fieldnames).writeheader()

    def maybe_log(self, *, epoch: int, batch: int, data_time: float, step_time: float, batch_result, lr: float) -> None:
        if not self.enabled:
            self.global_step += 1
            return
        slow = data_time >= self.slow_seconds or step_time >= self.slow_seconds
        should_log = self.global_step % self.log_interval == 0 or slow
        if should_log:
            row = {
                "epoch": int(epoch) + 1,
                "batch": int(batch),
                "global_step": int(self.global_step),
                "data_time": float(data_time),
                "forward_time": float(batch_result.timings.get("forward_time", math.nan)),
                "loss_time": float(batch_result.timings.get("loss_time", math.nan)),
                "backward_time": float(batch_result.timings.get("backward_time", math.nan)),
                "optimizer_step_time": float(batch_result.timings.get("optimizer_step_time", math.nan)),
                "step_time": float(step_time),
                "loss": float(batch_result.total_loss.detach().cpu().item()),
                "lr": float(lr),
                "gpu_mem_alloc_mb": _gpu_memory_mb("allocated"),
                "gpu_mem_reserved_mb": _gpu_memory_mb("reserved"),
                "cpu_rss_mb": _cpu_rss_mb(),
                "slow_batch": bool(slow),
            }
            with self.path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self.fieldnames).writerow(row)
            message = (
                f"epoch={row['epoch']} batch={row['batch']} data_time={row['data_time']:.3f}s "
                f"step_time={row['step_time']:.3f}s loss={row['loss']:.4f} "
                f"gpu_reserved={row['gpu_mem_reserved_mb']:.1f}MB cpu_rss={row['cpu_rss_mb']:.1f}MB"
            )
            if slow:
                message = "[SLOW_BATCH] " + message
            tqdm.write(message)
        self.global_step += 1


def _gpu_memory_mb(kind: str) -> float:
    if not torch.cuda.is_available():
        return 0.0
    value = torch.cuda.max_memory_allocated() if kind == "allocated" else torch.cuda.max_memory_reserved()
    return float(value) / (1024.0 * 1024.0)


def _cpu_rss_mb() -> float:
    if psutil is not None:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return float(usage) / scale


def _write_epoch_metrics_snapshot(run_dir: Path, epoch_logs: list[dict[str, Any]]) -> None:
    if not epoch_logs:
        return
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"epoch_logs": epoch_logs, "latest": epoch_logs[-1]}, handle, indent=2)
    columns = sorted({key for row in epoch_logs for key, value in row.items() if _csv_scalar(value)})
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in epoch_logs:
            writer.writerow({key: row.get(key, "") for key in columns})


def _csv_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _write_missing_pattern_eval(model, dataloader, cfg: dict, device, *, run_dir: Path) -> list[dict]:
    eval_cfg = cfg.get("evaluation", {}).get("missing_patterns", {})
    if not isinstance(eval_cfg, dict) or not bool(eval_cfg.get("enabled", False)):
        return []
    if not getattr(model, "supports_force_modality_mask", False):
        return []
    modalities = list(cfg.get("model", {}).get("primary", {}).get("modalities") or ["image", "radar", "lidar", "gps"])
    pattern_names = eval_cfg.get("patterns") or [
        "full",
        "missing_gps",
        "missing_image",
        "missing_radar",
        "missing_lidar",
        "non_gps_only",
        "gps_only",
        "image_only",
        "radar_only",
        "lidar_only",
    ]
    patterns = resolve_missing_patterns(pattern_names, modalities)
    results = evaluate_missing_matrix(
        model,
        dataloader,
        device,
        modalities,
        patterns=patterns,
        random_missing=eval_cfg.get("random_missing"),
        prediction_index=eval_cfg.get("prediction_index", "last"),
        max_batches=eval_cfg.get("max_batches"),
        cfg=cfg,
    )
    exp_name = sanitize_slug(str(cfg.get("output", {}).get("run_name") or cfg.get("experiment", {}).get("name", "run")))
    output_dir = run_dir.parent / "eval"
    save_results_csv(results, output_dir / f"{exp_name}_missing_patterns.csv")
    save_results_json(results, output_dir / f"{exp_name}_missing_patterns.json")
    save_results_markdown(results, output_dir / f"{exp_name}_missing_patterns.md")
    tqdm.write(format_results_markdown(results))
    return results


def shutdown_all_dataloaders(dataloaders: dict[str, Any]) -> None:
    for dataloader in dataloaders.values():
        shutdown_dataloader_workers(dataloader)


__all__ = [
    "_apply_csi_rms_to_model_config",
    "_evaluate_final_test_split",
    "_set_epoch_recursive",
    "_should_validate_epoch",
    "_validation_interval_epochs",
    "run_training_epoch_loop",
    "shutdown_all_dataloaders",
]
