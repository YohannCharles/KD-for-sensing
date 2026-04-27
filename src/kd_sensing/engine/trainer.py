from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from kd_sensing.config.io import dump_config
from kd_sensing.engine.batch import (
    forward_model,
    normalize_batch,
    prepare_fusion_inputs,
    prepare_image_inputs,
    prepare_labels,
)
from kd_sensing.engine.builders import (
    build_dataloaders,
    build_device,
    build_distiller,
    build_model,
    build_optimizer,
    build_scheduler,
    build_task_criterion,
    resolve_weight_path,
)
from kd_sensing.engine.validator import validate
from kd_sensing.utils.checkpoint import save_checkpoint
from kd_sensing.utils.paths import output_dir as resolve_output_dir
from kd_sensing.utils.plotting import plot_training_curves
from kd_sensing.utils.seed import set_seed


def create_run_dir(cfg: dict) -> Path:
    base = resolve_output_dir(cfg.get("output", {}).get("dir", cfg.get("paths", {}).get("output_dir", "outputs")))
    run_name = cfg.get("output", {}).get("run_name")
    if not run_name:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.get('experiment', {}).get('name', 'run')}_{timestamp}"
    path = base / run_name
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    return path


def _teacher_enabled(cfg: dict) -> bool:
    return cfg.get("distillation", {}).get("type", "no_kd") != "no_kd"


def _load_teacher_if_needed(cfg: dict, teacher_model, device: torch.device) -> None:
    weight_name = cfg.get("distillation", {}).get("teacher_model_name")
    weight_path = resolve_weight_path(cfg, weight_name)
    if weight_path is None:
        return
    if not weight_path.exists():
        raise FileNotFoundError(f"Teacher weight not found: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    teacher_model.load_state_dict(state_dict, strict=False)


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
    writer.add_scalar("learning_rate/main", history["learning_rates"][-1], step)
    writer.flush()


def _close_tensorboard_writer(writer) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()


def train(cfg: dict) -> dict:
    set_seed(cfg.get("experiment", {}).get("seed", 0))
    run_dir = create_run_dir(cfg)
    dump_config(cfg, run_dir / "final_config.yaml")
    dataloaders = build_dataloaders(cfg)
    device = build_device(cfg)
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    num_classes = model_cfg.get("num_classes", 64)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    seq_length_student = model_cfg.get("seq_length_student", 8)
    seq_length_teacher = model_cfg.get("seq_length_teacher", seq_length_student)

    student_model = build_model(model_cfg["student"]).to(device)
    teacher_model = None
    if _teacher_enabled(cfg):
        teacher_model = build_model(model_cfg["teacher"]).to(device)
        _load_teacher_if_needed(cfg, teacher_model, device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    task_criterion = build_task_criterion(cfg)
    distiller = build_distiller(cfg, task_criterion).to(device)
    optimizer = build_optimizer(cfg, student_model)
    scheduler = build_scheduler(cfg, optimizer)
    training_cfg = cfg["training"]
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "train_task_loss": [],
        "train_distill_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
    }

    tensorboard_writer = _create_tensorboard_writer(cfg, run_dir)
    try:
        for epoch in range(training_cfg.get("start_epoch", 0), training_cfg.get("epochs", 100)):
            student_model.train()
            running_loss = 0.0
            running_task_loss = 0.0
            running_distill_loss = 0.0
            running_acc = 0.0
            current_alpha = cfg["distillation"].get("alpha", 0.4)
            warmup_epochs = cfg["distillation"].get("alpha_warmup_epochs", 0)
            if warmup_epochs and epoch < warmup_epochs:
                current_alpha = current_alpha * (epoch / warmup_epochs)
            history["learning_rates"].append(optimizer.param_groups[0]["lr"])

            for step, raw_batch in enumerate(dataloaders["train"]):
                batch = normalize_batch(raw_batch)
                labels = prepare_labels(
                    batch,
                    num_pred=num_pred,
                    downsample_ratio=downsample_ratio,
                    device=device,
                )
                optimizer.zero_grad()
                if task == "fusion":
                    student_image, student_radar = prepare_fusion_inputs(
                        batch,
                        seq_length=seq_length_student,
                        num_pred=num_pred,
                        device=device,
                    )
                    student_outputs, student_input_features, student_out_features = forward_model(
                        student_model,
                        task,
                        student_image,
                        student_radar,
                    )
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_image, teacher_radar = prepare_fusion_inputs(
                                batch,
                                seq_length=seq_length_teacher,
                                num_pred=num_pred,
                                device=device,
                            )
                            teacher_outputs, teacher_input_features, teacher_out_features = forward_model(
                                teacher_model,
                                task,
                                teacher_image,
                                teacher_radar,
                            )
                    else:
                        teacher_outputs, teacher_input_features, teacher_out_features = _dummy_teacher(
                            student_outputs,
                            student_input_features,
                            student_out_features,
                        )
                else:
                    student_image = prepare_image_inputs(
                        batch,
                        seq_length=seq_length_student,
                        num_pred=num_pred,
                        device=device,
                    )
                    student_outputs, student_input_features, student_out_features = forward_model(
                        student_model,
                        task,
                        student_image,
                    )
                    if teacher_model is not None:
                        with torch.no_grad():
                            teacher_image = prepare_image_inputs(
                                batch,
                                seq_length=seq_length_teacher,
                                num_pred=num_pred,
                                device=device,
                            )
                            teacher_outputs, teacher_input_features, teacher_out_features = forward_model(
                                teacher_model,
                                task,
                                teacher_image,
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
                total_loss.backward()
                grad_clip = training_cfg.get("grad_clip", None)
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
            history["train_loss"].append(float(running_loss))
            history["train_task_loss"].append(float(running_task_loss))
            history["train_distill_loss"].append(float(running_distill_loss))
            history["train_acc"].append(float(running_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(val_acc)
            _write_tensorboard_scalars(tensorboard_writer, history, epoch + 1)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": student_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "test_loss": val_loss,
                },
                run_dir / "checkpoints",
                "last.pth",
            )
            if val_loss < best_val_loss - training_cfg.get("min_delta", 0.0):
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(student_model.state_dict(), run_dir / "checkpoints" / "best.pth")
            else:
                epochs_without_improvement += 1
                if training_cfg.get("use_early_stopping", True) and epochs_without_improvement >= training_cfg.get("patience", 20):
                    break
    finally:
        _close_tensorboard_writer(tensorboard_writer)

    np.savez(run_dir / "training_outputs.npz", **{k: np.asarray(v) for k, v in history.items()})
    with (run_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, run_dir)
    return {
        "run_dir": str(run_dir),
        "history": history,
        "best_val_loss": best_val_loss,
    }


def _dummy_teacher(student_outputs: torch.Tensor, student_input_features: torch.Tensor, student_out_features: torch.Tensor):
    return (
        torch.zeros_like(student_outputs),
        torch.zeros_like(student_input_features),
        torch.zeros_like(student_out_features),
    )
