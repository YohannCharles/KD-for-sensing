from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from kd_sensing.config.io import dump_config
from kd_sensing.data.scenes import scene_metadata_from_config, scene_slug_from_config
from kd_sensing.diagnostics.g2d_diagnostics import G2DDiagnosticsAccumulator
from kd_sensing.distillation.craf_losses import (
    beam_soft_label_loss,
    counterfactual_sequence_ce,
    prior_regularization_loss,
    reliability_weighted_kd_loss,
    sequence_cross_entropy,
)
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
from kd_sensing.engine.craf_training import (
    generate_context_marginal_masks,
    generate_counterfactual_drop_masks,
    generate_modality_dropout_mask,
    loss_delta_to_binary_gate_target,
    masked_gate_mse_loss,
)
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.marf_training import (
    ModalitySubsetSampler,
    all_to_subset_kl_loss,
    marf_anchor_entropy,
    marf_anchor_prior_regularization_loss,
    marf_residual_norm_loss,
)
from kd_sensing.engine.model_output import ModelOutput, adapt_model_output, select_prediction_slots
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
from kd_sensing.engine.run_metadata import dataloaders_run_metadata, throughput_run_metadata
from kd_sensing.engine.runtime import autocast_context, make_grad_scaler, resolve_amp_settings, transfer_non_blocking
from kd_sensing.engine.teacher_loader import (
    apply_selective_finetune,
    apply_teacher_priors,
    load_teacher_encoders,
    load_teacher_registry,
    trainable_parameter_count,
)
from kd_sensing.engine.validator import validate
from kd_sensing.distillation.g2d_smp import apply_smp_gradient_mask
from kd_sensing.distillation.teacher_ensemble import build_g2d_teacher_ensemble
from kd_sensing.modalities import normalize_modalities
from kd_sensing.utils.artifact_registry import archive_best_checkpoint, resolve_teacher_checkpoint
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
    if teacher_prior is not None:
        runtime["teacher_prior"] = teacher_prior
    scene_metadata = scene_metadata_from_config(cfg)
    if scene_metadata:
        runtime["scene"] = scene_metadata
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
    if history.get("train_beam_soft_loss"):
        writer.add_scalar("loss/train_beam_soft", history["train_beam_soft_loss"][-1], step)
    if history.get("train_unimodal_loss"):
        writer.add_scalar("loss/train_unimodal_aux", history["train_unimodal_loss"][-1], step)
    if history.get("train_counterfactual_loss"):
        writer.add_scalar("loss/train_counterfactual_gate", history["train_counterfactual_loss"][-1], step)
    if history.get("train_prior_regularization_loss"):
        writer.add_scalar("loss/train_prior_regularization", history["train_prior_regularization_loss"][-1], step)
    if history.get("train_reliability_kd_loss"):
        writer.add_scalar("loss/train_reliability_kd", history["train_reliability_kd_loss"][-1], step)
    writer.flush()


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
    g2d_enabled = _g2d_enabled(cfg)

    student_model = build_model(model_cfg["student"]).to(device)
    teacher_model = None
    teacher_ensemble = None
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
    if _teacher_enabled(cfg):
        if g2d_enabled:
            teacher_ensemble = build_g2d_teacher_ensemble(cfg, device)
            checkpoint_loads.extend(teacher_ensemble.load_summary())
        else:
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
        "train_beam_soft_loss": [],
        "train_unimodal_loss": [],
        "train_counterfactual_loss": [],
        "train_prior_regularization_loss": [],
        "train_reliability_kd_loss": [],
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
            running_beam_soft_loss = 0.0
            running_unimodal_loss = 0.0
            running_counterfactual_loss = 0.0
            running_prior_regularization_loss = 0.0
            running_reliability_kd_loss = 0.0
            running_acc = 0.0
            reliability_sums: dict[str, float] = {}
            reliability_batches = 0
            craf_diag_sums: dict[str, float] = {}
            craf_diag_counts: dict[str, int] = {}
            g2d_accumulator = (
                G2DDiagnosticsAccumulator(
                    num_pred=num_pred,
                    horizon_names=getattr(distiller, "horizon_names", [f"t+{idx + 1}" for idx in range(num_pred)]),
                )
                if g2d_enabled and cfg.get("distillation", {}).get("g2d", {}).get("diagnostics", {}).get("enabled", True)
                else None
            )
            batch_count = 0
            current_alpha = cfg["distillation"].get("alpha", 0.4)
            warmup_epochs = cfg["distillation"].get("alpha_warmup_epochs", 0)
            if warmup_epochs and epoch < warmup_epochs:
                current_alpha = current_alpha * (epoch / warmup_epochs)
            supports_craf_controls = getattr(student_model, "supports_reliability_controls", False)
            gate_temperature = (
                _current_craf_gate_temperature(cfg, model_cfg["student"], epoch) if supports_craf_controls else 1.0
            )
            force_reliability_gate = _training_reliability_gate_override(
                training_cfg,
                student_model,
                epoch=epoch,
            )
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
                    force_modality_mask = _training_modality_mask(
                        training_cfg,
                        student_model,
                        model_cfg["student"],
                        batch_size=int(labels.shape[0]),
                        device=device,
                    )
                    student_raw = _forward_for_task(
                        student_model,
                        task,
                        batch,
                        model_cfg=model_cfg["student"],
                        seq_length=seq_length_student,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                        force_modality_mask=force_modality_mask,
                        force_reliability_gate=force_reliability_gate,
                        gate_temperature=gate_temperature if supports_craf_controls else None,
                    )
                    student_model_output = adapt_model_output(student_raw)
                    student_outputs = select_prediction_slots(student_model_output.logits, num_pred)
                    student_input_features = student_model_output.input_features
                    student_out_features = student_model_output.output_features
                    g2d_step_result = None
                    if g2d_enabled:
                        if teacher_ensemble is None:
                            raise ValueError("G2D training requires a teacher ensemble.")
                        teacher_outputs_by_modality = teacher_ensemble(
                            batch,
                            seq_length=seq_length_teacher,
                            num_pred=num_pred,
                            device=device,
                            non_blocking=non_blocking,
                        )
                        g2d_student_output = ModelOutput(
                            logits=student_outputs,
                            input_features=student_input_features,
                            output_features=student_out_features,
                            diagnostics=student_model_output.diagnostics,
                        )
                        g2d_step_result = distiller.compute(
                            g2d_student_output,
                            teacher_outputs_by_modality,
                            labels,
                            epoch=epoch,
                        )
                        total_loss = g2d_step_result.total_loss
                        task_loss = g2d_step_result.supervised_loss
                        distill_loss = g2d_step_result.distill_loss
                        teacher_diagnostics = {}
                    elif teacher_model is not None:
                        with torch.no_grad():
                            teacher_raw = _forward_for_task(
                                teacher_model,
                                task,
                                batch,
                                model_cfg=model_cfg["teacher"],
                                seq_length=seq_length_teacher,
                                num_pred=num_pred,
                                device=device,
                                non_blocking=non_blocking,
                            )
                            teacher_model_output = adapt_model_output(teacher_raw)
                            teacher_outputs = select_prediction_slots(teacher_model_output.logits, num_pred)
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

                    if not g2d_enabled:
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
                    extra_losses = _compute_craf_extra_losses(
                        cfg,
                        student_model,
                        task,
                        batch,
                        model_cfg=model_cfg["student"],
                        seq_length=seq_length_student,
                        num_pred=num_pred,
                        num_classes=num_classes,
                        labels=labels,
                        student_outputs=student_outputs,
                        diagnostics=student_model_output.diagnostics,
                        teacher_diagnostics=teacher_diagnostics,
                        epoch=epoch,
                        gate_temperature=gate_temperature,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    marf_losses = _compute_marf_extra_losses(
                        cfg,
                        student_model,
                        task,
                        batch,
                        model_cfg=model_cfg["student"],
                        seq_length=seq_length_student,
                        num_pred=num_pred,
                        num_classes=num_classes,
                        labels=labels,
                        student_outputs=student_outputs,
                        diagnostics=student_model_output.diagnostics,
                        task_criterion=task_criterion,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    total_loss = total_loss + extra_losses["total"] + marf_losses["total"]
                grad_clip = training_cfg.get("grad_clip", None)
                active_modalities = (
                    g2d_step_result.active_modalities
                    if g2d_step_result is not None and getattr(distiller, "smp_enabled", False)
                    else None
                )
                if grad_scaler.is_enabled():
                    grad_scaler.scale(total_loss).backward()
                    if grad_clip or active_modalities is not None:
                        grad_scaler.unscale_(optimizer)
                    if active_modalities is not None:
                        apply_smp_gradient_mask(student_model, active_modalities)
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), grad_clip)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    total_loss.backward()
                    if active_modalities is not None:
                        apply_smp_gradient_mask(student_model, active_modalities)
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(student_model.parameters(), grad_clip)
                    optimizer.step()
                prediction = torch.argmax(student_outputs, dim=-1)
                valid = torch.sum(labels != -100).item()
                acc = (prediction == labels).sum().item() / max(valid, 1)
                running_loss = (total_loss.item() + step * running_loss) / (step + 1)
                running_task_loss = (task_loss.item() + step * running_task_loss) / (step + 1)
                running_distill_loss = (distill_loss.item() + step * running_distill_loss) / (step + 1)
                running_beam_soft_loss = (
                    extra_losses["beam_soft"].item() + step * running_beam_soft_loss
                ) / (step + 1)
                running_unimodal_loss = (
                    extra_losses["unimodal"].item() + step * running_unimodal_loss
                ) / (step + 1)
                running_counterfactual_loss = (
                    extra_losses["counterfactual"].item() + step * running_counterfactual_loss
                ) / (step + 1)
                running_prior_regularization_loss = (
                    extra_losses["prior_regularization"].item() + step * running_prior_regularization_loss
                ) / (step + 1)
                running_reliability_kd_loss = (
                    extra_losses["reliability_kd"].item() + step * running_reliability_kd_loss
                ) / (step + 1)
                running_acc = (acc + step * running_acc) / (step + 1)
                reliability_summary = _batch_reliability_summary(student_model_output.diagnostics)
                if reliability_summary:
                    reliability_batches += 1
                    for modality, value in reliability_summary.items():
                        reliability_sums[modality] = reliability_sums.get(modality, 0.0) + value
                scalar_diagnostics = dict(extra_losses.get("_diagnostics", {}))
                scalar_diagnostics.update(marf_losses.get("_diagnostics", {}))
                if g2d_step_result is not None:
                    scalar_diagnostics.update(_g2d_scalar_diagnostics(g2d_step_result.diagnostics))
                    if g2d_accumulator is not None:
                        g2d_accumulator.update(g2d_step_result.diagnostics)
                _accumulate_scalar_diagnostics(
                    scalar_diagnostics,
                    sums=craf_diag_sums,
                    counts=craf_diag_counts,
                )
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
            history["train_beam_soft_loss"].append(float(running_beam_soft_loss))
            history["train_unimodal_loss"].append(float(running_unimodal_loss))
            history["train_counterfactual_loss"].append(float(running_counterfactual_loss))
            history["train_prior_regularization_loss"].append(float(running_prior_regularization_loss))
            history["train_reliability_kd_loss"].append(float(running_reliability_kd_loss))
            history["train_acc"].append(float(running_acc))
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(val_acc)
            history["val_atop3"].append(validation_curve_metrics["val_atop3"])
            history["val_atop5"].append(validation_curve_metrics["val_atop5"])
            history["val_adba"].append(validation_curve_metrics["val_adba"])
            epoch_log = {
                "epoch": epoch + 1,
                "total_epochs": total_epochs,
                "train_batches": batch_count,
                "train_loss": float(running_loss),
                "train_task_loss": float(running_task_loss),
                "train_distill_loss": float(running_distill_loss),
                "train_beam_soft_loss": float(running_beam_soft_loss),
                "train_unimodal_loss": float(running_unimodal_loss),
                "train_counterfactual_loss": float(running_counterfactual_loss),
                "train_prior_regularization_loss": float(running_prior_regularization_loss),
                "train_reliability_kd_loss": float(running_reliability_kd_loss),
                "train_acc": float(running_acc),
                "val_loss": float(val_loss),
                "val_acc": val_acc,
                "val_atop3": validation_curve_metrics["val_atop3"],
                "val_atop5": validation_curve_metrics["val_atop5"],
                "val_adba": validation_curve_metrics["val_adba"],
                "learning_rate": float(current_lr),
            }
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
            epoch_log.update(_validation_subset_epoch_scalars(val_metrics))
            if reliability_batches:
                epoch_log["craf_reliability"] = {
                    modality: float(value / reliability_batches)
                    for modality, value in reliability_sums.items()
                }
            epoch_log.update(_mean_scalar_diagnostics(craf_diag_sums, craf_diag_counts))
            if g2d_accumulator is not None:
                g2d_path = g2d_accumulator.write_epoch(run_dir, epoch=epoch + 1)
                g2d_payload = g2d_accumulator.finalize(epoch=epoch + 1)
                epoch_log["g2d_diagnostics_path"] = str(g2d_path)
                epoch_log["g2d_active_modalities"] = g2d_payload.get("active_modalities", [])
            epoch_logs.append(epoch_log)
            epoch_progress.set_postfix(
                train_loss=f"{running_loss:.4f}",
                val_loss=f"{float(val_loss):.4f}",
                val_acc=f"{val_acc:.4f}",
                lr=f"{current_lr:.2e}",
            )
            _write_tensorboard_scalars(tensorboard_writer, history, epoch + 1)
            _write_tensorboard_craf_scalars(tensorboard_writer, epoch_log, epoch + 1)
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
    teacher_metrics = teacher_metrics_from_training(
        cfg,
        history,
        epoch_logs,
        best_top1_epoch=best_top1_epoch,
    )
    if teacher_metrics is not None:
        with (run_dir / "teacher_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(teacher_metrics, f, indent=2)
    train_log = {
        **history,
        "epoch_logs": epoch_logs,
        "teacher_metrics": teacher_metrics,
        "checkpoint_loads": checkpoint_loads,
        "teacher_prior": teacher_prior_info,
        "optimizer_param_groups": optimizer_groups,
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
            "teacher_prior": teacher_prior_info,
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
        "teacher_prior": teacher_prior_info,
        "optimizer_param_groups": optimizer_groups,
        "split_metadata": split_metadata,
        "throughput": throughput_metadata,
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
    force_modality_mask: torch.Tensor | None = None,
    force_reliability_gate: torch.Tensor | float | None = None,
    gate_temperature: float | torch.Tensor | None = None,
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
        return forward_model(
            model,
            task,
            **fusion_inputs,
            force_modality_mask=force_modality_mask,
            force_reliability_gate=force_reliability_gate,
            gate_temperature=gate_temperature,
        )
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


def _training_modality_mask(
    training_cfg: dict,
    model,
    model_cfg: dict,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if not getattr(model, "supports_force_modality_mask", False):
        return None
    dropout_cfg = training_cfg.get("modality_dropout", {})
    enabled = bool(dropout_cfg.get("enabled", False))
    drop_prob = float(dropout_cfg.get("drop_prob", 0.0))
    if not enabled or drop_prob <= 0.0:
        return None
    modalities = normalize_modalities(model_cfg.get("modalities", ("image", "radar")), context="CRAF dropout modalities")
    available = torch.ones(batch_size, len(modalities), dtype=torch.bool, device=device)
    return generate_modality_dropout_mask(
        available,
        drop_prob=drop_prob,
        min_keep=int(dropout_cfg.get("min_keep", 1)),
    )


def _training_reliability_gate_override(training_cfg: dict, model, *, epoch: int) -> float | None:
    if not getattr(model, "supports_reliability_controls", False):
        return None
    return 1.0 if epoch < _craf_gate_start_epoch(training_cfg) else None


def _craf_gate_start_epoch(training_cfg: dict) -> int:
    counterfactual_cfg = training_cfg.get("counterfactual", {})
    warmup_epochs = int(training_cfg.get("warmup_epochs", 0))
    counterfactual_start = int(counterfactual_cfg.get("start_epoch", 0))
    return max(warmup_epochs, counterfactual_start)


def _current_craf_gate_temperature(cfg: dict, model_cfg: dict, epoch: int) -> float:
    reliability_cfg = model_cfg.get("reliability", {})
    base_temperature = float(reliability_cfg.get("gate_temperature", 1.0))
    start_temperature = float(reliability_cfg.get("gate_temperature_start", base_temperature))
    end_temperature = float(reliability_cfg.get("gate_temperature_end", base_temperature))
    start_epoch = int(reliability_cfg.get("gate_temperature_start_epoch", _craf_gate_start_epoch(cfg.get("training", {}))))
    default_anneal = max(int(cfg.get("training", {}).get("epochs", 0)) - start_epoch, 0)
    anneal_epochs = int(reliability_cfg.get("gate_temperature_anneal_epochs", default_anneal))
    if anneal_epochs <= 0:
        return max(end_temperature, 1e-6)
    progress = min(max((epoch - start_epoch) / float(anneal_epochs), 0.0), 1.0)
    return max(start_temperature + (end_temperature - start_temperature) * progress, 1e-6)


def _scheduled_unimodal_weight(loss_cfg: dict, model_cfg: dict, epoch: int, warmup_boundary: int) -> float:
    unimodal_cfg = loss_cfg.get("unimodal_aux", {})
    base_weight = float(unimodal_cfg.get("weight", model_cfg.get("unimodal_loss_weight", 0.0)))
    warmup_weight = _optional_float(
        loss_cfg.get("uni_weight_warmup", unimodal_cfg.get("weight_warmup", None))
    )
    after_weight = _optional_float(
        loss_cfg.get("uni_weight_after_warmup", unimodal_cfg.get("weight_after_warmup", None))
    )
    if epoch < warmup_boundary:
        return base_weight if warmup_weight is None else warmup_weight
    return base_weight if after_weight is None else after_weight


def _counterfactual_target_weight(loss_cfg: dict, counterfactual_cfg: dict) -> float:
    configured = _optional_float(loss_cfg.get("gate_weight", None))
    legacy = float(counterfactual_cfg.get("weight", counterfactual_cfg.get("gate_loss_weight", 0.0)))
    if configured is not None and configured > 0.0:
        return configured
    return legacy


def _scheduled_gate_loss_weight(
    target_weight: float,
    epoch: int,
    *,
    start_epoch: int,
    ramp_epochs: int,
) -> float:
    if target_weight <= 0.0 or epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return float(target_weight)
    progress = min(max((epoch - start_epoch + 1) / float(ramp_epochs), 0.0), 1.0)
    return float(target_weight) * progress


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _accumulate_scalar_diagnostics(
    diagnostics,
    *,
    sums: dict[str, float],
    counts: dict[str, int],
) -> None:
    if not isinstance(diagnostics, dict):
        return
    for key, value in diagnostics.items():
        if isinstance(value, (int, float)):
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1


def _mean_scalar_diagnostics(sums: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {
        key: float(value / max(counts.get(key, 0), 1))
        for key, value in sums.items()
        if counts.get(key, 0) > 0
    }


def _g2d_scalar_diagnostics(diagnostics: dict) -> dict[str, float]:
    scalars: dict[str, float] = {}
    for key, value in (diagnostics.get("loss") or {}).items():
        if isinstance(value, (int, float)):
            scalars[f"loss/g2d_{key}"] = float(value)
    for modality, values in (diagnostics.get("teacher_confidence") or {}).items():
        if isinstance(values, dict):
            avg = values.get("avg")
            if isinstance(avg, (int, float)):
                scalars[f"g2d/teacher_confidence/{modality}"] = float(avg)
    active = diagnostics.get("active_modalities")
    if isinstance(active, (list, tuple)):
        scalars["g2d/active_count"] = float(len(active))
    return scalars


def _compute_marf_extra_losses(
    cfg: dict,
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict,
    seq_length: int,
    num_pred: int,
    num_classes: int,
    labels: torch.Tensor,
    student_outputs: torch.Tensor,
    diagnostics: dict,
    task_criterion,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, torch.Tensor | dict[str, float]]:
    zero = student_outputs.sum() * 0.0
    scalar_diagnostics: dict[str, float] = {}
    losses = {
        "total": zero,
        "residual_norm": zero,
        "prior_regularization": zero,
        "anchor_entropy": zero,
        "subset_ce": zero,
        "subset_kd": zero,
        "_diagnostics": scalar_diagnostics,
    }
    if not getattr(model, "supports_marf_routing", False):
        return losses

    scalar_diagnostics.update(_marf_scalar_diagnostics(diagnostics))
    loss_cfg = cfg.get("loss", {}).get("marf", {})
    residual_cfg = loss_cfg.get("residual_norm", {})
    residual_weight = float(residual_cfg.get("weight", 0.0))
    residual_enabled = bool(residual_cfg.get("enabled", residual_weight > 0.0)) and residual_weight > 0.0
    scalar_diagnostics["loss/marf_residual_norm_weight"] = residual_weight if residual_enabled else 0.0
    if residual_enabled and torch.is_tensor(diagnostics.get("residual_delta")):
        losses["residual_norm"] = marf_residual_norm_loss(
            diagnostics["residual_delta"],
            diagnostics.get("residual_weights"),
            diagnostics.get("effective_modality_mask"),
        )
        losses["total"] = losses["total"] + residual_weight * losses["residual_norm"]
        scalar_diagnostics["loss/marf_residual_norm"] = float(losses["residual_norm"].detach().cpu().item())

    prior_cfg = loss_cfg.get("prior_regularization", cfg.get("loss", {}).get("prior_regularization", {}))
    prior_weight = float(prior_cfg.get("weight", 0.0))
    prior_enabled = bool(prior_cfg.get("enabled", prior_weight > 0.0)) and prior_weight > 0.0
    scalar_diagnostics["loss/marf_prior_regularization_weight"] = prior_weight if prior_enabled else 0.0
    if prior_enabled and torch.is_tensor(diagnostics.get("anchor_weights")) and torch.is_tensor(diagnostics.get("prior")):
        losses["prior_regularization"] = marf_anchor_prior_regularization_loss(
            diagnostics["anchor_weights"],
            diagnostics["prior"],
            diagnostics.get("effective_modality_mask"),
            loss_type=str(prior_cfg.get("loss_type", "mse")),
        )
        losses["total"] = losses["total"] + prior_weight * losses["prior_regularization"]
        scalar_diagnostics["loss/marf_prior_regularization"] = float(
            losses["prior_regularization"].detach().cpu().item()
        )

    entropy_cfg = loss_cfg.get("anchor_entropy", {})
    entropy_weight = float(entropy_cfg.get("weight", 0.0))
    entropy_enabled = bool(entropy_cfg.get("enabled", entropy_weight > 0.0)) and entropy_weight > 0.0
    scalar_diagnostics["loss/marf_anchor_entropy_weight"] = entropy_weight if entropy_enabled else 0.0
    if entropy_enabled and torch.is_tensor(diagnostics.get("anchor_weights")):
        entropy_value = marf_anchor_entropy(diagnostics["anchor_weights"], diagnostics.get("effective_modality_mask"))
        losses["anchor_entropy"] = entropy_value
        sign = -1.0 if bool(entropy_cfg.get("maximize", True)) else 1.0
        losses["total"] = losses["total"] + sign * entropy_weight * entropy_value
        scalar_diagnostics["loss/marf_anchor_entropy"] = float(entropy_value.detach().cpu().item())

    subset_cfg = cfg.get("training", {}).get("subset_training", {})
    subset_enabled = bool(subset_cfg.get("enabled", False))
    if not subset_enabled:
        scalar_diagnostics["loss/marf_subset_ce"] = 0.0
        scalar_diagnostics["loss/marf_subset_kd"] = 0.0
        return losses
    if task != "fusion":
        raise ValueError("training.subset_training.enabled=true requires experiment.task=fusion.")
    if not getattr(model, "supports_force_modality_mask", False):
        raise ValueError("training.subset_training.enabled=true requires force_modality_mask support.")

    modes = subset_cfg.get("modes") or subset_cfg.get("subsets") or []
    if isinstance(modes, str):
        modes = [modes]
    if not modes:
        return losses
    available = diagnostics.get("effective_modality_mask")
    if not torch.is_tensor(available):
        available = torch.ones(
            labels.shape[0],
            len(model_cfg.get("modalities", getattr(model, "modalities", ("image", "radar")))),
            dtype=torch.bool,
            device=device,
        )
    prior = _marf_prior_vector(diagnostics, available.shape[1], device=device)
    sampler = ModalitySubsetSampler(
        model_cfg.get("modalities", getattr(model, "modalities", ("image", "radar"))),
        prior,
        top_prior_k=int(subset_cfg.get("top_prior_k", 2)),
        min_keep=int(subset_cfg.get("min_keep", 1)),
        random_keep_prob=float(subset_cfg.get("random_keep_prob", 0.5)),
    )
    ce_weight = float(subset_cfg.get("ce_weight", loss_cfg.get("subset_ce", {}).get("weight", 0.0)))
    kd_weight = float(subset_cfg.get("kd_weight", loss_cfg.get("subset_kd", {}).get("weight", 0.0)))
    temperature = float(subset_cfg.get("temperature", loss_cfg.get("subset_kd", {}).get("temperature", 3.0)))
    ignore_index = int(subset_cfg.get("ignore_index", -100))
    subset_ce_losses = []
    subset_kd_losses = []
    max_subsets = int(subset_cfg.get("max_subsets_per_batch", len(modes)))
    for mode in list(modes)[: max(max_subsets, 0)]:
        spec = sampler.sample(str(mode), available_mask=available.detach(), device=device)
        if not torch.any(spec.mask):
            continue
        if str(mode) == "all" and torch.equal(spec.mask, available.detach()):
            subset_outputs = student_outputs
        else:
            subset_raw = _forward_for_task(
                model,
                task,
                batch,
                model_cfg=model_cfg,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
                force_modality_mask=spec.mask,
            )
            subset_output = adapt_model_output(subset_raw)
            subset_outputs = select_prediction_slots(subset_output.logits, num_pred)
        if ce_weight > 0.0:
            subset_ce_losses.append(task_criterion(subset_outputs.reshape(-1, num_classes), labels.flatten()))
        if kd_weight > 0.0:
            subset_kd_losses.append(
                all_to_subset_kl_loss(
                    subset_outputs,
                    student_outputs.detach(),
                    labels,
                    temperature=temperature,
                    ignore_index=ignore_index,
                )
            )
    if subset_ce_losses:
        losses["subset_ce"] = torch.stack(subset_ce_losses).mean()
        losses["total"] = losses["total"] + ce_weight * losses["subset_ce"]
    if subset_kd_losses:
        losses["subset_kd"] = torch.stack(subset_kd_losses).mean()
        losses["total"] = losses["total"] + kd_weight * losses["subset_kd"]
    scalar_diagnostics["loss/marf_subset_ce_weight"] = ce_weight if subset_ce_losses else 0.0
    scalar_diagnostics["loss/marf_subset_kd_weight"] = kd_weight if subset_kd_losses else 0.0
    scalar_diagnostics["loss/marf_subset_ce"] = float(losses["subset_ce"].detach().cpu().item())
    scalar_diagnostics["loss/marf_subset_kd"] = float(losses["subset_kd"].detach().cpu().item())
    return losses


def _compute_craf_extra_losses(
    cfg: dict,
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict,
    seq_length: int,
    num_pred: int,
    num_classes: int,
    labels: torch.Tensor,
    student_outputs: torch.Tensor,
    diagnostics: dict,
    teacher_diagnostics: dict | None = None,
    epoch: int,
    gate_temperature: float,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, torch.Tensor | dict[str, float]]:
    zero = student_outputs.sum() * 0.0
    record_craf_diagnostics = getattr(model, "supports_reliability_controls", False)
    scalar_diagnostics: dict[str, float] = {}
    if record_craf_diagnostics:
        scalar_diagnostics["craf/gate_temperature"] = float(gate_temperature)
    losses = {
        "total": zero,
        "beam_soft": zero,
        "unimodal": zero,
        "counterfactual": zero,
        "prior_regularization": zero,
        "reliability_kd": zero,
        "_diagnostics": scalar_diagnostics,
    }

    loss_cfg = cfg.get("loss", {})
    beam_cfg = loss_cfg.get("beam_soft", {})
    beam_weight = float(beam_cfg.get("weight", 0.0))
    beam_enabled = bool(beam_cfg.get("enabled", beam_weight > 0.0)) and beam_weight > 0.0
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/beam_soft_weight"] = beam_weight if beam_enabled else 0.0
    if beam_enabled:
        losses["beam_soft"] = beam_soft_label_loss(
            student_outputs,
            labels,
            sigma=float(beam_cfg.get("sigma", 2.0)),
            circular=bool(beam_cfg.get("circular", True)),
            ignore_index=int(beam_cfg.get("ignore_index", -100)),
        )
        losses["total"] = losses["total"] + beam_weight * losses["beam_soft"]

    unimodal_cfg = loss_cfg.get("unimodal_aux", {})
    warmup_boundary = _craf_gate_start_epoch(cfg.get("training", {}))
    unimodal_weight = _scheduled_unimodal_weight(loss_cfg, model_cfg, epoch, warmup_boundary)
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/unimodal_aux_weight"] = float(unimodal_weight)
    if unimodal_weight > 0.0:
        unimodal_loss = _unimodal_aux_loss(
            diagnostics.get("unimodal_logits"),
            labels,
            diagnostics.get("effective_modality_mask"),
            num_pred=num_pred,
            ignore_index=int(unimodal_cfg.get("ignore_index", -100)),
            zero=zero,
        )
        losses["unimodal"] = unimodal_loss
        losses["total"] = losses["total"] + unimodal_weight * unimodal_loss

    counterfactual_cfg = cfg.get("training", {}).get("counterfactual", {})
    counterfactual_weight = _counterfactual_target_weight(loss_cfg, counterfactual_cfg)
    counterfactual_effective_weight = _scheduled_gate_loss_weight(
        counterfactual_weight,
        epoch,
        start_epoch=warmup_boundary,
        ramp_epochs=int(loss_cfg.get("gate_ramp_epochs", counterfactual_cfg.get("gate_ramp_epochs", 0))),
    )
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/gate_weight_target"] = float(counterfactual_weight)
        losses["_diagnostics"]["loss/gate_weight_effective"] = float(counterfactual_effective_weight)
    counterfactual_enabled = bool(counterfactual_cfg.get("enabled", False))
    if (
        counterfactual_enabled
        and counterfactual_effective_weight > 0.0
        and epoch >= warmup_boundary
        and getattr(model, "supports_force_modality_mask", False)
    ):
        counterfactual_loss, counterfactual_diagnostics = _counterfactual_gate_loss(
            model,
            task,
            batch,
            model_cfg=model_cfg,
            seq_length=seq_length,
            num_pred=num_pred,
            labels=labels,
            full_outputs=student_outputs,
            reliability=diagnostics.get("reliability"),
            effective_modality_mask=diagnostics.get("effective_modality_mask"),
            modalities=diagnostics.get("modalities"),
            mode=str(counterfactual_cfg.get("mode", "sample_one")),
            ignore_delta_eps=float(counterfactual_cfg.get("ignore_delta_eps", 0.0)),
            num_drop_per_batch=int(counterfactual_cfg.get("num_drop_per_batch", 1)),
            min_keep=int(
                counterfactual_cfg.get(
                    "min_keep",
                    cfg.get("training", {}).get("modality_dropout", {}).get("min_keep", 1),
                )
            ),
            no_grad_drop_forward=bool(counterfactual_cfg.get("no_grad_drop_forward", True)),
            gate_temperature=gate_temperature,
            device=device,
            non_blocking=non_blocking,
            zero=zero,
        )
        losses["counterfactual"] = counterfactual_loss
        losses["total"] = losses["total"] + counterfactual_effective_weight * counterfactual_loss
        losses["_diagnostics"].update(counterfactual_diagnostics)

    prior_cfg = loss_cfg.get("prior_regularization", {})
    prior_weight = float(prior_cfg.get("weight", 0.0))
    prior_enabled = bool(prior_cfg.get("enabled", prior_weight > 0.0)) and prior_weight > 0.0
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/prior_regularization_weight"] = prior_weight if prior_enabled else 0.0
        losses["_diagnostics"].update(_craf_gate_scalar_diagnostics(diagnostics))
    if prior_enabled:
        gate = diagnostics.get("gate", diagnostics.get("reliability"))
        prior = diagnostics.get("prior")
        modality_mask = diagnostics.get("effective_modality_mask")
        if torch.is_tensor(gate) and torch.is_tensor(prior):
            losses["prior_regularization"] = prior_regularization_loss(
                gate,
                prior,
                modality_mask,
                loss_type=str(prior_cfg.get("loss_type", "mse")),
            )
            losses["total"] = losses["total"] + prior_weight * losses["prior_regularization"]

    kd_cfg = cfg.get("training", {}).get("reliability_kd", cfg.get("kd", {}))
    kd_weight = float(kd_cfg.get("weight", 0.0))
    kd_enabled = bool(kd_cfg.get("enabled", False)) and kd_weight > 0.0
    if record_craf_diagnostics:
        losses["_diagnostics"]["loss/reliability_kd_weight"] = kd_weight if kd_enabled else 0.0
    if kd_enabled:
        student_unimodal = diagnostics.get("unimodal_logits")
        teacher_unimodal = (teacher_diagnostics or {}).get("unimodal_logits")
        reliability = diagnostics.get("gate", diagnostics.get("reliability"))
        if not (torch.is_tensor(student_unimodal) and torch.is_tensor(teacher_unimodal) and torch.is_tensor(reliability)):
            raise ValueError("reliability_kd.enabled=true requires student and teacher unimodal CRAF logits.")
        losses["reliability_kd"] = reliability_weighted_kd_loss(
            student_unimodal,
            teacher_unimodal,
            reliability,
            modalities=diagnostics.get("modalities") or model_cfg.get("modalities") or [],
            use_modalities=kd_cfg.get("use_modalities", ["gps", "mmwave"]),
            temperature=float(kd_cfg.get("temperature", cfg.get("distillation", {}).get("temperature", 3.0))),
            modality_mask=diagnostics.get("effective_modality_mask"),
        )
        losses["total"] = losses["total"] + kd_weight * losses["reliability_kd"]
    return losses


def _unimodal_aux_loss(
    unimodal_logits,
    labels: torch.Tensor,
    effective_modality_mask,
    *,
    num_pred: int,
    ignore_index: int,
    zero: torch.Tensor,
) -> torch.Tensor:
    if not torch.is_tensor(unimodal_logits) or unimodal_logits.numel() == 0:
        return zero
    if unimodal_logits.ndim != 4:
        raise ValueError(f"unimodal_logits must have shape [B, K, H, C], got {tuple(unimodal_logits.shape)}.")
    horizon = num_pred
    if unimodal_logits.shape[2] != horizon:
        raise ValueError(
            "unimodal_logits horizon must exactly match num_pred future slots; "
            f"got {unimodal_logits.shape[2]} slots for num_pred={horizon}."
        )
    batch_size, modality_count, _, num_classes = unimodal_logits.shape
    expanded_labels = labels.unsqueeze(1).expand(batch_size, modality_count, -1)
    _, per_modality_loss = sequence_cross_entropy(
        unimodal_logits.reshape(batch_size * modality_count, horizon, num_classes),
        expanded_labels.reshape(batch_size * modality_count, horizon),
        ignore_index=ignore_index,
    )
    if torch.is_tensor(effective_modality_mask):
        mask = effective_modality_mask.to(device=unimodal_logits.device, dtype=torch.bool).reshape(-1)
    else:
        mask = torch.ones(batch_size * modality_count, dtype=torch.bool, device=unimodal_logits.device)
    if not torch.any(mask):
        return zero
    return per_modality_loss[mask].mean()


def _counterfactual_gate_loss(
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict,
    seq_length: int,
    num_pred: int,
    labels: torch.Tensor,
    full_outputs: torch.Tensor,
    reliability,
    effective_modality_mask,
    modalities,
    mode: str,
    ignore_delta_eps: float,
    num_drop_per_batch: int,
    min_keep: int,
    no_grad_drop_forward: bool,
    gate_temperature: float,
    device: torch.device,
    non_blocking: bool,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not torch.is_tensor(reliability) or not torch.is_tensor(effective_modality_mask):
        return zero, {}
    modality_names = _diagnostic_modalities(modalities, reliability.shape[1])
    available = effective_modality_mask.detach()
    gate_losses = []
    stats = _CounterfactualStats(modality_names)
    if mode == "shuffle":
        full_per_sample = counterfactual_sequence_ce(full_outputs, labels)
        for modality_idx, modality in enumerate(modality_names):
            target_mask = torch.zeros_like(available)
            target_mask[:, modality_idx] = available[:, modality_idx]
            if not torch.any(target_mask):
                continue
            context_manager = torch.no_grad() if no_grad_drop_forward else nullcontext()
            with context_manager:
                shuffled_batch = _shuffled_modality_batch(batch, modality)
                shuffled_per_sample = _counterfactual_forward_ce(
                    model,
                    task,
                    shuffled_batch,
                    model_cfg=model_cfg,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    labels=labels,
                    force_modality_mask=available,
                    gate_temperature=gate_temperature,
                    device=device,
                    non_blocking=non_blocking,
                )
                delta = shuffled_per_sample - full_per_sample
                target, valid_mask = loss_delta_to_binary_gate_target(
                    delta.detach(),
                    target_mask,
                    ignore_delta_eps=ignore_delta_eps,
                )
            stats.update(delta.detach(), target.detach(), target_mask, valid_mask)
            gate_losses.append(masked_gate_mse_loss(reliability, target, valid_mask))
    elif mode == "context_marginal":
        mask_specs = generate_context_marginal_masks(
            available,
            num_samples=num_drop_per_batch,
            min_keep=min_keep,
        )
        for context_mask, with_target_mask, target_mask in mask_specs:
            context_manager = torch.no_grad() if no_grad_drop_forward else nullcontext()
            with context_manager:
                context_per_sample = _counterfactual_forward_ce(
                    model,
                    task,
                    batch,
                    model_cfg=model_cfg,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    labels=labels,
                    force_modality_mask=context_mask,
                    gate_temperature=gate_temperature,
                    device=device,
                    non_blocking=non_blocking,
                )
                with_target_per_sample = _counterfactual_forward_ce(
                    model,
                    task,
                    batch,
                    model_cfg=model_cfg,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    labels=labels,
                    force_modality_mask=with_target_mask,
                    gate_temperature=gate_temperature,
                    device=device,
                    non_blocking=non_blocking,
                )
                delta = context_per_sample - with_target_per_sample
                target, valid_mask = loss_delta_to_binary_gate_target(
                    delta.detach(),
                    target_mask,
                    ignore_delta_eps=ignore_delta_eps,
                )
            stats.update(delta.detach(), target.detach(), target_mask, valid_mask)
            gate_losses.append(masked_gate_mse_loss(reliability, target, valid_mask))
    else:
        full_per_sample = counterfactual_sequence_ce(full_outputs, labels)
        drop_specs = generate_counterfactual_drop_masks(available, mode=mode)
        for keep_mask, dropped_mask in drop_specs:
            context_manager = torch.no_grad() if no_grad_drop_forward else nullcontext()
            with context_manager:
                drop_per_sample = _counterfactual_forward_ce(
                    model,
                    task,
                    batch,
                    model_cfg=model_cfg,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    labels=labels,
                    force_modality_mask=keep_mask,
                    gate_temperature=gate_temperature,
                    device=device,
                    non_blocking=non_blocking,
                )
                delta = drop_per_sample - full_per_sample
                target, valid_mask = loss_delta_to_binary_gate_target(
                    delta.detach(),
                    dropped_mask,
                    ignore_delta_eps=ignore_delta_eps,
                )
            stats.update(delta.detach(), target.detach(), dropped_mask, valid_mask)
            gate_losses.append(masked_gate_mse_loss(reliability, target, valid_mask))
    if not gate_losses:
        return zero, stats.to_diagnostics()
    return torch.stack(gate_losses).mean(), stats.to_diagnostics()


def _counterfactual_forward_ce(
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict,
    seq_length: int,
    num_pred: int,
    labels: torch.Tensor,
    force_modality_mask: torch.Tensor,
    gate_temperature: float,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    raw = _forward_for_task(
        model,
        task,
        batch,
        model_cfg=model_cfg,
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
        force_modality_mask=force_modality_mask,
        gate_temperature=gate_temperature,
    )
    output = adapt_model_output(raw)
    logits = select_prediction_slots(output.logits, num_pred)
    return counterfactual_sequence_ce(logits, labels)


def _shuffled_modality_batch(batch: dict[str, torch.Tensor], modality: str) -> dict[str, torch.Tensor]:
    keys_by_modality = {
        "image": ("image",),
        "radar": ("radar_ra", "radar_da"),
        "gps": ("gps",),
        "lidar": ("lidar",),
        "mmwave": ("mmwave",),
    }
    keys = keys_by_modality.get(str(modality), ())
    if not keys:
        return batch
    first = next((batch[key] for key in keys if key in batch and torch.is_tensor(batch[key])), None)
    if first is None or first.shape[0] <= 1:
        return batch
    order = torch.randperm(first.shape[0], device=first.device)
    shuffled = dict(batch)
    for key in keys:
        value = batch.get(key)
        if torch.is_tensor(value) and value.shape[0] == first.shape[0]:
            shuffled[key] = value.index_select(0, order)
    return shuffled


class _CounterfactualStats:
    def __init__(self, modalities: list[str]):
        self.modalities = modalities
        self.delta_sum = torch.zeros(len(modalities), dtype=torch.float64)
        self.delta_count = torch.zeros(len(modalities), dtype=torch.float64)
        self.target_sum = torch.zeros(len(modalities), dtype=torch.float64)
        self.valid_count = torch.zeros(len(modalities), dtype=torch.float64)
        self.candidate_count = torch.zeros(len(modalities), dtype=torch.float64)

    def update(
        self,
        delta: torch.Tensor,
        target: torch.Tensor,
        candidate_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        candidate = candidate_mask.detach().to(torch.bool).cpu()
        valid = valid_mask.detach().to(torch.bool).cpu()
        delta_cpu = delta.detach().double().cpu()
        target_cpu = target.detach().double().cpu()
        for idx in range(len(self.modalities)):
            candidate_idx = candidate[:, idx]
            if torch.any(candidate_idx):
                self.delta_sum[idx] += delta_cpu[candidate_idx].sum()
                self.delta_count[idx] += candidate_idx.sum()
                self.candidate_count[idx] += candidate_idx.sum()
            valid_idx = valid[:, idx]
            if torch.any(valid_idx):
                self.target_sum[idx] += target_cpu[:, idx][valid_idx].sum()
                self.valid_count[idx] += valid_idx.sum()

    def to_diagnostics(self) -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        for idx, modality in enumerate(self.modalities):
            if self.delta_count[idx].item() > 0:
                diagnostics[f"cf/delta_mean_{modality}"] = float(
                    (self.delta_sum[idx] / self.delta_count[idx]).item()
                )
            if self.valid_count[idx].item() > 0:
                diagnostics[f"cf/target_mean_{modality}"] = float(
                    (self.target_sum[idx] / self.valid_count[idx]).item()
                )
            if self.candidate_count[idx].item() > 0:
                diagnostics[f"cf/target_valid_rate_{modality}"] = float(
                    (self.valid_count[idx] / self.candidate_count[idx]).item()
                )
        return diagnostics


def _batch_reliability_summary(diagnostics: dict) -> dict[str, float]:
    reliability = diagnostics.get("reliability")
    modalities = diagnostics.get("modalities")
    if not torch.is_tensor(reliability) or reliability.ndim != 2:
        return {}
    modalities = _diagnostic_modalities(modalities, reliability.shape[1])
    means = reliability.detach().float().mean(dim=0).cpu()
    return {str(modality): float(means[idx].item()) for idx, modality in enumerate(modalities)}


def _marf_scalar_diagnostics(diagnostics: dict) -> dict[str, float]:
    anchor = diagnostics.get("anchor_weights")
    residual = diagnostics.get("residual_weights")
    prior = diagnostics.get("prior")
    mask = diagnostics.get("effective_modality_mask")
    modalities = diagnostics.get("modalities")
    if not torch.is_tensor(anchor) or anchor.ndim != 3:
        return {}
    modality_names = _diagnostic_modalities(modalities, anchor.shape[-1])
    if torch.is_tensor(mask):
        available = mask.detach().to(device=anchor.device, dtype=torch.bool)
    else:
        available = torch.ones(anchor.shape[0], anchor.shape[-1], dtype=torch.bool, device=anchor.device)
    scalars: dict[str, float] = {}
    for idx, modality in enumerate(modality_names):
        modality_mask = available[:, idx]
        if torch.any(modality_mask):
            values = anchor[:, :, idx][modality_mask]
            scalars[f"marf/anchor_mean/{modality}"] = float(values.detach().float().mean().cpu().item())
            for horizon_idx in range(anchor.shape[1]):
                horizon_values = anchor[:, horizon_idx, idx][modality_mask]
                scalars[f"marf/anchor_h{horizon_idx}/{modality}"] = float(
                    horizon_values.detach().float().mean().cpu().item()
                )
            if torch.is_tensor(residual) and residual.ndim == 3:
                residual_values = residual[:, :, idx][modality_mask]
                scalars[f"marf/residual_mean/{modality}"] = float(
                    residual_values.detach().float().mean().cpu().item()
                )
        if torch.is_tensor(prior):
            prior_values = prior[:, idx] if prior.ndim == 2 else prior[idx].view(1).expand(anchor.shape[0])
            scalars[f"marf/prior/{modality}"] = float(prior_values.detach().float().mean().cpu().item())
    return scalars


def _marf_prior_vector(diagnostics: dict, modality_count: int, *, device: torch.device) -> torch.Tensor:
    prior = diagnostics.get("prior")
    if torch.is_tensor(prior):
        values = prior.detach()
        if values.ndim == 2:
            values = values.mean(dim=0)
        return values.to(device=device, dtype=torch.float32).flatten()
    return torch.full((int(modality_count),), 1.0 / max(int(modality_count), 1), dtype=torch.float32, device=device)


def _craf_gate_scalar_diagnostics(diagnostics: dict) -> dict[str, float]:
    gate = diagnostics.get("gate", diagnostics.get("reliability"))
    prior = diagnostics.get("prior")
    residual = diagnostics.get("residual_logits")
    mask = diagnostics.get("effective_modality_mask")
    modalities = diagnostics.get("modalities")
    if not torch.is_tensor(gate) or gate.ndim != 2:
        return {}
    modality_names = _diagnostic_modalities(modalities, gate.shape[1])
    if torch.is_tensor(mask):
        available = mask.detach().to(device=gate.device, dtype=torch.bool)
    else:
        available = torch.ones_like(gate, dtype=torch.bool)
    scalars: dict[str, float] = {}
    for idx, modality in enumerate(modality_names):
        modality_mask = available[:, idx]
        if torch.any(modality_mask):
            scalars[f"craf/gate_mean/{modality}"] = float(gate[:, idx][modality_mask].detach().float().mean().cpu().item())
        if torch.is_tensor(prior):
            prior_values = prior[:, idx] if prior.ndim == 2 else prior[idx].view(1).expand(gate.shape[0])
            scalars[f"craf/prior/{modality}"] = float(prior_values.detach().float().mean().cpu().item())
        if torch.is_tensor(residual):
            residual_values = residual[:, idx]
            if torch.any(modality_mask):
                scalars[f"craf/residual_logit_mean/{modality}"] = float(
                    residual_values[modality_mask].detach().float().mean().cpu().item()
                )
    return scalars


def _diagnostic_modalities(modalities, modality_count: int) -> list[str]:
    if not isinstance(modalities, (tuple, list)) or len(modalities) != modality_count:
        return [f"modality_{idx}" for idx in range(modality_count)]
    return [str(modality) for modality in modalities]
