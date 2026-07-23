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

from kd_sensing.engine.checkpointing import checkpoint_strict as _checkpoint_strict
from kd_sensing.engine.data_factory import shutdown_dataloader_workers
from kd_sensing.engine.model_initialization import enforce_frozen_module_eval
from kd_sensing.engine.tensorboard_logging import (
    write_tensorboard_method_scalars as _write_tensorboard_method_scalars,
    write_tensorboard_scalars as _write_tensorboard_scalars,
)
from kd_sensing.engine.validator import validate
from kd_sensing.utils.missing_patterns import resolve_missing_patterns
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import (
    evaluate_missing_matrix,
    format_results_markdown,
    save_results_csv,
    save_results_json,
    save_results_markdown,
)
from kd_sensing.eval.missing_summary import save_missing_summary, summarize_missing_patterns
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
    tensorboard_writer,
    objective: str,
    task_criterion,
    device,
    run_dir: Path,
    training_cfg: dict,
    optimizer_groups,
    progress_enabled: bool,
    total_epochs: int,
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
    checkpoint_selection = _checkpoint_selection(training_cfg)
    best_validation_loss = _best_observed_validation_loss(state.epoch_logs)
    last_observed_validation: dict[str, Any] | None = None
    timing_logger = _TimingCsvLogger(cfg, run_dir, device=device)
    for epoch in _flush_timing_when_epoch_loop_exits(epoch_progress, timing_logger):
        _set_epoch_recursive(primary_model, epoch)
        _set_dataset_epoch_recursive(dataloaders["train"].dataset, epoch)
        sampler = getattr(dataloaders["train"], "sampler", None)
        if callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(epoch)
        primary_model.train()
        enforce_frozen_module_eval(primary_model)
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
        data_wait_start = timing_logger.host_now()
        for step, raw_batch in enumerate(batch_progress):
            batch_start = timing_logger.start_step()
            data_time = timing_logger.host_elapsed(data_wait_start)
            batch_result = batch_runner.run(raw_batch, epoch=epoch, step=step)
            step_time = timing_logger.finish_step(batch_start)
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
            data_wait_start = timing_logger.host_now()

        if scheduler is not None:
            scheduler.step()
        validation_ran = validation_loader is not None and (
            _should_validate_epoch(epoch, total_epochs, validation_interval) or last_observed_validation is None
        )
        if validation_ran:
            val_metrics = validate_fn(
                primary_model,
                validation_loader,
                cfg,
                task_criterion,
                device,
                output_dir=run_dir,
            )
            last_observed_validation = {"epoch": epoch + 1, "source": "validation"}
        else:
            val_metrics = None
        extension_metrics = {}
        for extension, extension_state in zip(extensions, extension_states):
            extension_metrics.update(extension.after_epoch(extension_context, extension_state, epoch=epoch))
        health_metrics = health_tracker.finish_epoch() if health_tracker is not None else None
        epoch_log, val_loss, val_acc = recorder.finish_epoch(
            epoch=epoch,
            total_epochs=total_epochs,
            val_metrics=val_metrics,
            current_lr=current_lr,
            optimizer_groups=optimizer_groups,
            health_metrics=health_metrics,
            extension_metrics=extension_metrics,
        )
        epoch_log.update(
            {
                "validation_ran": bool(validation_ran),
                "validation_interval_epochs": int(validation_interval),
                "last_observed_validation": (
                    dict(last_observed_validation) if last_observed_validation is not None else None
                ),
            }
        )
        timing_logger.flush()
        if progress_enabled:
            metrics = recorder.progress_metrics()
            postfix = {"train_loss": f"{metrics['loss']:.4f}", "lr": f"{current_lr:.2e}"}
            if val_loss is not None:
                postfix["val_loss"] = f"{float(val_loss):.4f}"
            if val_acc is not None:
                postfix["val_acc"] = f"{float(val_acc):.4f}"
            epoch_progress.set_postfix(**postfix)
        _write_tensorboard_scalars(
            tensorboard_writer,
            state.history,
            epoch + 1,
            objective=objective,
            tensorboard_cfg=cfg.get("output", {}).get("tensorboard", {}),
        )
        _write_tensorboard_method_scalars(tensorboard_writer, epoch_log, epoch + 1)
        last_checkpoint = checkpoint_manager.save_last_checkpoint(state=state, epoch=epoch, val_loss=val_loss)
        if (
            checkpoint_selection == "best_validation_loss"
            and val_loss is not None
            and math.isfinite(float(val_loss))
            and float(val_loss) < best_validation_loss
        ):
            best_validation_loss = float(val_loss)
            checkpoint_manager.save_best_checkpoint(state=state, epoch=epoch, val_loss=best_validation_loss)
            epoch_log["best_checkpoint_saved"] = True
            epoch_log["best_validation_loss"] = best_validation_loss
            epoch_log["best_checkpoint_epoch"] = int(epoch) + 1
        elif checkpoint_selection == "best_validation_loss":
            epoch_log["best_checkpoint_saved"] = False
            epoch_log["best_validation_loss"] = best_validation_loss if math.isfinite(best_validation_loss) else None
        _write_epoch_metrics_snapshot(run_dir, state.epoch_logs)


def _validation_interval_epochs(training_cfg: dict[str, Any]) -> int:
    return max(1, int(training_cfg["validation"]["interval_epochs"]))


def _should_validate_epoch(epoch: int, total_epochs: int, interval_epochs: int) -> bool:
    if int(interval_epochs) <= 1:
        return True
    epoch_number = int(epoch) + 1
    return epoch_number == 1 or epoch_number == int(total_epochs) or epoch_number % int(interval_epochs) == 0


def _checkpoint_selection(training_cfg: dict[str, Any]) -> str:
    value = str(training_cfg.get("checkpoint_selection", "last")).strip().lower()
    if value not in {"last", "best_validation_loss"}:
        raise ValueError("training.checkpoint_selection must be 'last' or 'best_validation_loss'.")
    return value


def _best_observed_validation_loss(epoch_logs: list[dict[str, Any]]) -> float:
    values = []
    for row in epoch_logs:
        value = row.get("val_loss")
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return min(values, default=math.inf)


def _evaluate_final_test_split(
    primary_model,
    test_loader,
    cfg: dict,
    task_criterion,
    device,
    *,
    run_dir: Path,
) -> tuple[dict, dict | None]:
    selection = _checkpoint_selection(cfg.get("training", {}))
    checkpoint_role = "validation_best" if selection == "best_validation_loss" else "last"
    checkpoint_path = run_dir / "checkpoints" / ("best.pth" if selection == "best_validation_loss" else "last.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Final test checkpoint not found: {checkpoint_path}")
    load_result = load_model_state(
        checkpoint_path,
        primary_model,
        role=f"final-test-{checkpoint_role}",
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
    metrics["evaluation_split"] = "test"
    metrics["checkpoint_for_test"] = str(checkpoint_path)
    metrics["selected_checkpoint"] = {"path": str(checkpoint_path), "checkpoint_role": checkpoint_role}
    if missing_pattern_results:
        metrics["missing_patterns"] = missing_pattern_results
    return metrics, checkpoint_load

def _set_epoch_recursive(module, epoch: int) -> None:
    setter = getattr(module, "set_epoch", None)
    if callable(setter):
        setter(int(epoch))
    children = getattr(module, "children", None)
    if not callable(children):
        return
    for child in children():
        _set_epoch_recursive(child, epoch)


def _set_dataset_epoch_recursive(dataset, epoch: int) -> None:
    setter = getattr(dataset, "set_epoch", None)
    if callable(setter):
        setter(int(epoch))
    nested = getattr(dataset, "datasets", None)
    if isinstance(nested, (list, tuple)):
        for child in nested:
            _set_dataset_epoch_recursive(child, epoch)
    parent = getattr(dataset, "dataset", None)
    if parent is not None:
        _set_dataset_epoch_recursive(parent, epoch)


def _flush_timing_when_epoch_loop_exits(epoch_progress, timing_logger):
    try:
        yield from epoch_progress
    finally:
        try:
            timing_logger.flush()
        except Exception:
            # Timing is optional observability; it must not mask a training failure.
            pass


class _TimingCsvLogger:
    def __init__(self, cfg: dict, run_dir: Path, *, device: torch.device | None = None) -> None:
        timing_cfg = cfg.get("training", {}).get("timing", {})
        if not isinstance(timing_cfg, dict):
            timing_cfg = {}
        self.enabled = bool(timing_cfg.get("enabled", False))
        self.profile: str | None = None
        if self.enabled:
            profile = str(timing_cfg.get("profile", "")).strip().lower()
            if profile not in {"host", "cuda_event"}:
                raise ValueError("training.timing.profile must be 'host' or 'cuda_event' when timing is enabled.")
            if profile == "cuda_event" and (device is None or torch.device(device).type != "cuda"):
                raise ValueError("training.timing.profile='cuda_event' requires a CUDA device.")
            self.profile = profile
        self.device = torch.device(device) if device is not None else None
        self.log_interval = max(1, int(timing_cfg.get("log_interval", 1) or 1))
        self.slow_seconds = float(timing_cfg.get("slow_batch_seconds", 20.0))
        self.path = run_dir / "timing.csv"
        self.global_step = 0
        self.rows: list[dict[str, Any]] = []
        self.fieldnames = [
            "profile",
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

    def host_now(self) -> float | None:
        if self.profile != "host" or not self._sample_current_step():
            return None
        return time.perf_counter()

    def host_elapsed(self, started_at: float | None) -> float | None:
        if started_at is None:
            return None
        return time.perf_counter() - started_at

    def start_step(self):
        if not self._sample_current_step():
            return None
        if self.profile == "host":
            return time.perf_counter()
        if self.profile == "cuda_event":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return start, end
        return None

    def finish_step(self, token) -> float | None:
        if token is None:
            return None
        if self.profile == "host":
            return time.perf_counter() - float(token)
        start, end = token
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)) / 1000.0

    def maybe_log(
        self,
        *,
        epoch: int,
        batch: int,
        data_time: float | None,
        step_time: float | None,
        batch_result,
        lr: float,
    ) -> None:
        if not self.enabled:
            return
        if data_time is not None or step_time is not None:
            slow = any(value is not None and value >= self.slow_seconds for value in (data_time, step_time))
            timings = getattr(batch_result, "timings", {})
            row = {
                "profile": self.profile,
                "epoch": int(epoch) + 1,
                "batch": int(batch),
                "global_step": int(self.global_step),
                "data_time": _timing_value(data_time),
                "forward_time": _timing_value(timings.get("forward_time")),
                "loss_time": _timing_value(timings.get("loss_time")),
                "backward_time": _timing_value(timings.get("backward_time")),
                "optimizer_step_time": _timing_value(timings.get("optimizer_step_time")),
                "step_time": _timing_value(step_time),
                "loss": float(batch_result.total_loss.detach().cpu().item()),
                "lr": float(lr),
                "gpu_mem_alloc_mb": _gpu_memory_mb("allocated"),
                "gpu_mem_reserved_mb": _gpu_memory_mb("reserved"),
                "cpu_rss_mb": _cpu_rss_mb(),
                "slow_batch": bool(slow),
            }
            self.rows.append(row)
        self.global_step += 1

    def flush(self) -> None:
        if not self.rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(self.rows)
        self.rows.clear()

    def _sample_current_step(self) -> bool:
        return self.enabled and self.global_step % self.log_interval == 0


def _timing_value(value: float | None) -> float:
    return float(value) if value is not None else math.nan


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
    eval_cfg = cfg["evaluation"]["missing_patterns"]
    if not eval_cfg["enabled"]:
        return []
    if not getattr(model, "supports_force_modality_mask", False):
        return []
    modalities = cfg["model"]["primary"]["modalities"]
    pattern_names = eval_cfg["patterns"]
    patterns = resolve_missing_patterns(pattern_names, modalities)
    results = evaluate_missing_matrix(
        model,
        dataloader,
        device,
        modalities,
        patterns=patterns,
        prediction_index=eval_cfg["prediction_index"],
        max_batches=eval_cfg.get("max_batches"),
        cfg=cfg,
    )
    exp_name = sanitize_slug(str(cfg.get("output", {}).get("run_name") or cfg.get("experiment", {}).get("name", "run")))
    output_dir = run_dir.parent / "eval"
    save_results_csv(results, output_dir / f"{exp_name}_missing_patterns.csv")
    save_results_json(results, output_dir / f"{exp_name}_missing_patterns.json")
    save_results_markdown(results, output_dir / f"{exp_name}_missing_patterns.md")
    summary = summarize_missing_patterns(results, modality_count=len(modalities))
    save_missing_summary(summary, output_dir / f"{exp_name}_missing_summary")
    tqdm.write(format_results_markdown(results))
    return results


def shutdown_all_dataloaders(dataloaders: dict[str, Any]) -> None:
    for dataloader in dataloaders.values():
        shutdown_dataloader_workers(dataloader)


__all__ = [
    "_evaluate_final_test_split",
    "_set_epoch_recursive",
    "_should_validate_epoch",
    "_validation_interval_epochs",
    "run_training_epoch_loop",
    "shutdown_all_dataloaders",
]
