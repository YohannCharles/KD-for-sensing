from __future__ import annotations

from copy import deepcopy
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from kd_sensing.config.io import dump_config
from kd_sensing.engine.batch import (
    forward_model,
    normalize_batch,
    prepare_fusion_inputs,
    prepare_gps_inputs,
    prepare_image_inputs,
    prepare_labels,
    prepare_lidar_inputs,
    prepare_mmwave_inputs,
    prepare_radar_inputs,
)
from kd_sensing.engine.builders import (
    build_dataloaders,
    build_device,
    build_distiller,
    build_model,
    build_optimizer,
    build_scheduler,
    build_task_criterion,
    dataloaders_run_metadata,
    save_normalization_artifacts,
    throughput_run_metadata,
)
from kd_sensing.engine.runtime import autocast_context, make_grad_scaler, resolve_amp_settings, transfer_non_blocking
from kd_sensing.engine.validator import validate
from kd_sensing.utils.artifact_registry import archive_best_checkpoint, resolve_teacher_checkpoint
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_checkpoint, load_model_state, save_checkpoint
from kd_sensing.utils.paths import output_dir as resolve_output_dir, resolve_path
from kd_sensing.utils.plotting import plot_training_curves
from kd_sensing.utils.seed import set_seed


def create_run_dir(cfg: dict) -> Path:
    base = resolve_output_dir(cfg.get("output", {}).get("dir", cfg.get("paths", {}).get("output_dir", "outputs")))
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
    base = resolve_output_dir(cfg.get("output", {}).get("dir", cfg.get("paths", {}).get("output_dir", "outputs")))
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
) -> dict:
    final_cfg = deepcopy(cfg)
    runtime = final_cfg.setdefault("runtime", {})
    runtime["run_dir"] = str(run_dir)
    runtime["output_overwrite"] = bool(cfg.get("output", {}).get("overwrite", False))
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
    return final_cfg


def _teacher_enabled(cfg: dict) -> bool:
    return cfg.get("distillation", {}).get("type", "no_kd") != "no_kd"


def _checkpoint_strict(cfg: dict) -> bool:
    return bool(cfg.get("checkpoint", {}).get("strict_load", True))


def _load_teacher_if_needed(cfg: dict, teacher_model, device: torch.device) -> dict | None:
    weight_name = cfg.get("distillation", {}).get("teacher_model_name")
    resolution = resolve_teacher_checkpoint(cfg, weight_name)
    if resolution.path is None and resolution.source == "none":
        return None
    if resolution.path is None or not resolution.path.exists():
        raise FileNotFoundError(f"Teacher weight not found. Resolution: {resolution.to_dict()}")
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
                "legacy_path": str(resolution.legacy_path) if resolution.legacy_path is not None else None,
                "metadata": resolution.metadata,
            }
        )
    return summary


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


def _write_tensorboard_scalars(writer, history: dict, step: int) -> None:
    if writer is None:
        return

    writer.add_scalar("loss/train", history["train_loss"][-1], step)
    writer.add_scalar("loss/train_task", history["train_task_loss"][-1], step)
    writer.add_scalar("loss/train_distill", history["train_distill_loss"][-1], step)
    writer.add_scalar("loss/val", history["val_loss"][-1], step)
    writer.add_scalar("accuracy/train", history["train_acc"][-1], step)
    writer.add_scalar("accuracy/val", history["val_acc"][-1], step)
    writer.add_scalar("accuracy/val_atop3", history["val_atop3"][-1], step)
    writer.add_scalar("accuracy/val_atop5", history["val_atop5"][-1], step)
    writer.add_scalar("dba/val_adba", history["val_adba"][-1], step)
    writer.add_scalar("learning_rate/main", history["learning_rates"][-1], step)
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


def _close_tensorboard_writer(writer) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()


def _progress_enabled(cfg: dict) -> bool:
    return cfg.get("output", {}).get("progress", {}).get("enabled", True)


def train(cfg: dict) -> dict:
    set_seed(cfg.get("experiment", {}).get("seed", 0))
    if cfg.get("training", {}).get("resume") is True and not cfg.get("output", {}).get("run_name"):
        raise ValueError("training.resume=true requires output.run_name so checkpoints/last.pth can be resolved.")
    run_dir = create_run_dir(cfg)
    dataloaders = build_dataloaders(cfg)
    split_metadata = dataloaders_run_metadata(dataloaders)
    normalization_artifacts = save_normalization_artifacts(dataloaders, run_dir)
    device = build_device(cfg)
    throughput_metadata = throughput_run_metadata(cfg, dataloaders, device)
    dump_config(
        final_config_with_runtime(
            cfg,
            run_dir=run_dir,
            split_metadata=split_metadata,
            normalization_artifacts=normalization_artifacts,
            throughput_metadata=throughput_metadata,
        ),
        run_dir / "final_config.yaml",
    )
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    num_classes = model_cfg.get("num_classes", 64)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    seq_length_student = model_cfg.get("seq_length_student", 8)
    seq_length_teacher = model_cfg.get("seq_length_teacher", seq_length_student)

    student_model = build_model(model_cfg["student"]).to(device)
    teacher_model = None
    checkpoint_loads = []
    if _teacher_enabled(cfg):
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
    scheduler = build_scheduler(cfg, optimizer)
    grad_scaler = make_grad_scaler(cfg, amp_enabled)
    training_cfg = cfg["training"]
    best_val_loss = float("inf")
    best_val_top1 = float("-inf")
    best_top1_epoch = 0
    registry_checkpoint = None
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "train_task_loss": [],
        "train_distill_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_atop3": [],
        "val_atop5": [],
        "val_adba": [],
        "learning_rates": [],
    }
    epoch_logs = []

    tensorboard_writer = _create_tensorboard_writer(cfg, run_dir)
    progress_enabled = _progress_enabled(cfg)
    start_epoch = training_cfg.get("start_epoch", 0)
    total_epochs = training_cfg.get("epochs", 100)
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
            student_model.train()
            running_loss = 0.0
            running_task_loss = 0.0
            running_distill_loss = 0.0
            running_acc = 0.0
            batch_count = 0
            current_alpha = cfg["distillation"].get("alpha", 0.4)
            warmup_epochs = cfg["distillation"].get("alpha_warmup_epochs", 0)
            if warmup_epochs and epoch < warmup_epochs:
                current_alpha = current_alpha * (epoch / warmup_epochs)
            current_lr = optimizer.param_groups[0]["lr"]
            history["learning_rates"].append(current_lr)

            batch_progress = tqdm(
                dataloaders["train"],
                desc=f"Epoch {epoch + 1}/{total_epochs}",
                unit="batch",
                leave=False,
                disable=not progress_enabled,
            )
            for step, raw_batch in enumerate(batch_progress):
                batch_count = step + 1
                batch = normalize_batch(raw_batch)
                labels = prepare_labels(
                    batch,
                    num_pred=num_pred,
                    downsample_ratio=downsample_ratio,
                    device=device,
                    non_blocking=non_blocking,
                )
                optimizer.zero_grad()
                with autocast_context(amp_enabled, device, amp_dtype):
                    student_outputs, student_input_features, student_out_features = _forward_for_task(
                        student_model,
                        task,
                        batch,
                        model_cfg=model_cfg["student"],
                        seq_length=seq_length_student,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_outputs, teacher_input_features, teacher_out_features = _forward_for_task(
                                teacher_model,
                                task,
                                batch,
                                model_cfg=model_cfg["teacher"],
                                seq_length=seq_length_teacher,
                                num_pred=num_pred,
                                device=device,
                                non_blocking=non_blocking,
                            )
                    else:
                        teacher_outputs, teacher_input_features, teacher_out_features = _dummy_teacher(
                            student_outputs,
                            student_input_features,
                            student_out_features,
                        )

                    student_outputs = student_outputs[:, -(num_pred + 1) :, :]
                    teacher_outputs = teacher_outputs[:, -(num_pred + 1) :, :]
                    student_logits = student_outputs.reshape(-1, num_classes)
                    teacher_logits = teacher_outputs.reshape(-1, num_classes)
                    targets = labels.flatten()
                    total_loss, task_loss, distill_loss = distiller(
                        student_logits,
                        teacher_logits,
                        targets,
                        student_input_features[:, : seq_length_student - 1, :],
                        teacher_input_features[:, : seq_length_teacher - 1, :],
                        student_out_features[:, -(num_pred + 1) :, :],
                        teacher_out_features[:, -(num_pred + 1) :, :],
                        current_alpha,
                    )
                grad_clip = training_cfg.get("grad_clip", None)
                if grad_scaler.is_enabled():
                    grad_scaler.scale(total_loss).backward()
                    if grad_clip:
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), grad_clip)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    total_loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), grad_clip)
                    optimizer.step()
                prediction = torch.argmax(student_outputs, dim=-1)
                valid = torch.sum(labels != -100).item()
                acc = (prediction == labels).sum().item() / max(valid, 1)
                running_loss = (total_loss.item() + step * running_loss) / (step + 1)
                running_task_loss = (task_loss.item() + step * running_task_loss) / (step + 1)
                running_distill_loss = (distill_loss.item() + step * running_distill_loss) / (step + 1)
                running_acc = (acc + step * running_acc) / (step + 1)
                batch_progress.set_postfix(
                    loss=f"{running_loss:.4f}",
                    task=f"{running_task_loss:.4f}",
                    distill=f"{running_distill_loss:.4f}",
                    acc=f"{running_acc:.4f}",
                    lr=f"{current_lr:.2e}",
                )

            if scheduler is not None:
                scheduler.step()
            val_metrics = validate(
                student_model,
                dataloaders["test"],
                cfg,
                task_criterion,
                device,
                output_dir=run_dir,
            )
            val_loss = val_metrics["loss"]
            top1 = val_metrics["topk"].get("1", [0.0])
            val_acc = float(top1[0]) if top1 else 0.0
            validation_curve_metrics = _aggregate_validation_metrics(val_metrics)
            history["train_loss"].append(float(running_loss))
            history["train_task_loss"].append(float(running_task_loss))
            history["train_distill_loss"].append(float(running_distill_loss))
            history["train_acc"].append(float(running_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(val_acc)
            history["val_atop3"].append(validation_curve_metrics["val_atop3"])
            history["val_atop5"].append(validation_curve_metrics["val_atop5"])
            history["val_adba"].append(validation_curve_metrics["val_adba"])
            epoch_logs.append(
                {
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "train_batches": batch_count,
                    "train_loss": float(running_loss),
                    "train_task_loss": float(running_task_loss),
                    "train_distill_loss": float(running_distill_loss),
                    "train_acc": float(running_acc),
                    "val_loss": float(val_loss),
                    "val_acc": val_acc,
                    "val_atop3": validation_curve_metrics["val_atop3"],
                    "val_atop5": validation_curve_metrics["val_atop5"],
                    "val_adba": validation_curve_metrics["val_adba"],
                    "learning_rate": float(current_lr),
                }
            )
            epoch_progress.set_postfix(
                train_loss=f"{running_loss:.4f}",
                val_loss=f"{float(val_loss):.4f}",
                val_acc=f"{val_acc:.4f}",
                lr=f"{current_lr:.2e}",
            )
            _write_tensorboard_scalars(tensorboard_writer, history, epoch + 1)
            improved = val_loss < best_val_loss - training_cfg.get("min_delta", 0.0)
            if improved:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(student_model.state_dict(), run_dir / "checkpoints" / "best.pth")
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
                )
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
                    "epochs_without_improvement": epochs_without_improvement,
                    "normalization_artifacts": normalization_artifacts,
                    "checkpoint_registry": registry_checkpoint,
                },
                run_dir / "checkpoints",
                "last.pth",
            )
            if (
                not improved
                and training_cfg.get("use_early_stopping", True)
                and epochs_without_improvement >= training_cfg.get("patience", 20)
            ):
                break
    finally:
        _close_tensorboard_writer(tensorboard_writer)

    np.savez(run_dir / "training_outputs.npz", **{k: np.asarray(v) for k, v in history.items()})
    train_log = {
        **history,
        "epoch_logs": epoch_logs,
        "checkpoint_loads": checkpoint_loads,
        "normalization_artifacts": normalization_artifacts,
        "checkpoint_registry": registry_checkpoint,
        "throughput": throughput_metadata,
        "runtime": {
            "run_dir": str(run_dir),
            "output_overwrite": bool(cfg.get("output", {}).get("overwrite", False)),
            "splits": split_metadata,
            "normalization_artifacts": normalization_artifacts,
            "checkpoint_registry": registry_checkpoint,
            "throughput": throughput_metadata,
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
        ),
        run_dir / "final_config.yaml",
    )
    return {
        "run_dir": str(run_dir),
        "history": history,
        "epoch_logs": epoch_logs,
        "best_val_loss": best_val_loss,
        "best_val_top1": best_val_top1,
        "checkpoint_registry": registry_checkpoint,
        "normalization_artifacts": normalization_artifacts,
        "checkpoint_loads": checkpoint_loads,
        "split_metadata": split_metadata,
        "throughput": throughput_metadata,
    }


def _dummy_teacher(student_outputs: torch.Tensor, student_input_features: torch.Tensor, student_out_features: torch.Tensor):
    return (
        torch.zeros_like(student_outputs),
        torch.zeros_like(student_input_features),
        torch.zeros_like(student_out_features),
    )


def _forward_for_task(
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict | None = None,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
):
    if task == "fusion":
        fusion_inputs = prepare_fusion_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            modalities=(model_cfg or {}).get("modalities"),
            non_blocking=non_blocking,
        )
        return forward_model(model, task, **fusion_inputs)
    if task == "radar":
        radar_batch = prepare_radar_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            non_blocking=non_blocking,
        )
        return forward_model(model, task, radar_batch=radar_batch)
    if task == "gps":
        gps_batch = prepare_gps_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            non_blocking=non_blocking,
        )
        return forward_model(model, task, gps_batch=gps_batch)
    if task == "lidar":
        lidar_batch = prepare_lidar_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            non_blocking=non_blocking,
        )
        return forward_model(model, task, lidar_batch=lidar_batch)
    if task == "mmwave":
        mmwave_batch = prepare_mmwave_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            non_blocking=non_blocking,
        )
        return forward_model(model, task, mmwave_batch=mmwave_batch)
    image_batch = prepare_image_inputs(
        batch,
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
    )
    return forward_model(model, task, image_batch)
