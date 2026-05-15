from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from kd_sensing.engine.marf_training import ModalitySubsetSampler
from kd_sensing.engine.prediction_objectives import (
    compute_prediction_loss,
    objective_runtime_metadata,
    prepare_prediction_targets,
    resolve_prediction_objective,
)
from kd_sensing.engine.runtime import (
    autocast_context,
    prepare_task_batch,
    prepare_task_auxiliary_targets,
    prepare_task_labels,
    resolve_amp_settings,
    run_model_step,
    transfer_non_blocking,
)
from kd_sensing.evaluation.lidar_diagnostics import (
    LidarQualityAccumulator,
    degradation_baselines_from_labels,
    lidar_degradation_report,
    lidar_preprocessing_metadata_from_dataset,
)
from kd_sensing.evaluation.metrics import (
    calculate_dba_score,
    calculate_occlusion_metrics,
    calculate_position_rmse,
    calculate_topk_accuracy,
)
from kd_sensing.evaluation.subset_specs import resolve_conditional_utility_subset


def validate(model, dataloader, cfg: dict, criterion, device: torch.device, output_dir: str | Path | None = None):
    model.eval()
    objective = resolve_prediction_objective(cfg)
    objective_metadata = objective_runtime_metadata(cfg)
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    seq_length = model_cfg.get("seq_length_student", 8)
    num_classes = model_cfg.get("num_classes", 64)
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    val_loss = 0.0
    val_occlusion_loss = 0.0
    val_position_loss = 0.0
    val_multitask_loss = 0.0
    all_outputs = []
    all_labels = []
    all_input_beams = []
    all_occlusion_logits = []
    all_occlusion_labels = []
    all_occlusion_valid = []
    all_position_outputs = []
    all_position_targets = []
    all_position_valid = []
    lidar_quality = LidarQualityAccumulator()
    saw_lidar = False
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_task_batch(batch)
            if "input_beam" in batch:
                all_input_beams.append(batch["input_beam"].detach().cpu())
            if "lidar" in batch:
                saw_lidar = True
                lidar_quality.update(batch["lidar"])
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
            with autocast_context(amp_enabled, device, amp_dtype):
                step = run_model_step(
                    model,
                    task,
                    batch,
                    model_cfg=cfg["model"]["student"],
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                    non_blocking=non_blocking,
                )
                outputs = step.logits
                beam_loss = criterion(outputs.reshape(-1, num_classes), labels.flatten())
                prediction_loss = compute_prediction_loss(
                    step.model_output,
                    prediction_targets,
                    cfg,
                    reference=outputs,
                    beam_total_loss=beam_loss,
                    beam_task_loss=beam_loss,
                )
                loss = prediction_loss.total
            val_loss += loss.item()
            val_occlusion_loss += prediction_loss.occlusion.item()
            val_position_loss += prediction_loss.position.item()
            val_multitask_loss += prediction_loss.multitask_total.item()
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if "occlusion_logits" in step.model_output.diagnostics and "occlusion_label" in auxiliary_targets:
                all_occlusion_logits.append(step.model_output.diagnostics["occlusion_logits"].detach().cpu())
                all_occlusion_labels.append(auxiliary_targets["occlusion_label"].detach().cpu())
                all_occlusion_valid.append(auxiliary_targets["occlusion_valid"].detach().cpu())
            if "position" in step.model_output.diagnostics and "position_target" in auxiliary_targets:
                all_position_outputs.append(step.model_output.diagnostics["position"].detach().cpu())
                all_position_targets.append(auxiliary_targets["position_target"].detach().cpu())
                all_position_valid.append(auxiliary_targets["position_valid"].detach().cpu())
    val_loss = val_loss / max(len(dataloader), 1)
    all_outputs_t = torch.cat(all_outputs, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    auxiliary_metrics = _auxiliary_metrics_from_outputs(
        dataloader,
        occlusion_logits=torch.cat(all_occlusion_logits, dim=0) if all_occlusion_logits else None,
        occlusion_labels=torch.cat(all_occlusion_labels, dim=0) if all_occlusion_labels else None,
        occlusion_valid=torch.cat(all_occlusion_valid, dim=0) if all_occlusion_valid else None,
        position_outputs=torch.cat(all_position_outputs, dim=0) if all_position_outputs else None,
        position_targets=torch.cat(all_position_targets, dim=0) if all_position_targets else None,
        position_valid=torch.cat(all_position_valid, dim=0) if all_position_valid else None,
    )
    metrics = _metrics_from_outputs(val_loss, all_outputs_t, all_labels_t, cfg)
    _attach_objective_metrics(
        metrics,
        auxiliary_metrics,
        objective=objective,
        dataloader_len=len(dataloader),
        val_occlusion_loss=val_occlusion_loss,
        val_position_loss=val_position_loss,
        val_multitask_loss=val_multitask_loss,
    )
    metrics["objective"] = objective_metadata
    metrics["available_metrics"] = _available_scalar_metrics(metrics)
    input_beams_t = torch.cat(all_input_beams, dim=0) if all_input_beams else None
    baselines = degradation_baselines_from_labels(
        all_labels_t,
        input_beams=input_beams_t,
        num_classes=num_classes,
        downsample_ratio=downsample_ratio,
    )
    metrics["degradation_baselines"] = baselines
    if saw_lidar:
        dataset = getattr(dataloader, "dataset", None)
        quality_summary = lidar_quality.finalize(
            split=getattr(dataset, "split", None),
            preprocessing=lidar_preprocessing_metadata_from_dataset(dataset),
        )
        metrics["lidar_input_quality"] = quality_summary
        metrics["degradation_risk"] = lidar_degradation_report(metrics, baselines, quality_summary)
    elif _cfg_uses_lidar(cfg):
        metrics["degradation_risk"] = lidar_degradation_report(metrics, baselines, None)
    subset_metrics = _validate_modality_subsets(model, dataloader, cfg, criterion, device, official_metrics=metrics)
    if subset_metrics:
        metrics["modality_subsets"] = subset_metrics
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def _metrics_from_outputs(loss: float, outputs: torch.Tensor, labels: torch.Tensor, cfg: dict) -> dict:
    topk_acc, total = calculate_topk_accuracy(
        outputs,
        labels,
        cfg.get("evaluation", {}).get("k_values", [1, 2, 3, 5, 10]),
    )
    dba_score = calculate_dba_score(
        outputs,
        labels,
        cfg.get("evaluation", {}).get("dba_delta", 5),
    )
    metrics = {
        "loss": float(loss),
        "topk": {str(k): v.tolist() for k, v in topk_acc.items()},
        "dba": dba_score.tolist(),
        "total": total.tolist(),
    }
    metrics.update(_flat_future_topk_metrics(topk_acc, total))
    return metrics


def _auxiliary_metrics_from_outputs(
    dataloader,
    *,
    occlusion_logits: torch.Tensor | None,
    occlusion_labels: torch.Tensor | None,
    occlusion_valid: torch.Tensor | None,
    position_outputs: torch.Tensor | None,
    position_targets: torch.Tensor | None,
    position_valid: torch.Tensor | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if occlusion_logits is not None and occlusion_labels is not None:
        metrics.update(calculate_occlusion_metrics(occlusion_logits, occlusion_labels, occlusion_valid))
    if position_outputs is not None and position_targets is not None:
        scaler = getattr(getattr(dataloader, "dataset", None), "position_target_scaler", None)
        mean = getattr(scaler, "mean_", None)
        scale = getattr(scaler, "scale_", None)
        metrics.update(
            calculate_position_rmse(
                position_outputs,
                position_targets,
                position_valid,
                mean=mean,
                scale=scale,
            )
        )
    return metrics


def _flat_future_topk_metrics(topk_acc: dict[int, object], total) -> dict[str, float]:
    scalars: dict[str, float] = {}
    total_arr = torch.as_tensor(total, dtype=torch.float32).cpu().numpy()
    horizon_names = [f"t{idx + 1}" for idx in range(len(total_arr))]
    for k in (1, 3, 5):
        if k not in topk_acc:
            continue
        values = torch.as_tensor(topk_acc[k], dtype=torch.float32).cpu().numpy()
        length = min(len(values), len(total_arr))
        valid = total_arr[:length] > 0
        for idx in range(length):
            scalars[f"val_top{k}_{horizon_names[idx]}"] = float(values[idx])
        scalars[f"val_top{k}_avg"] = float(values[:length][valid].mean()) if valid.any() else 0.0
    return scalars


def _cfg_uses_lidar(cfg: dict) -> bool:
    task = cfg.get("experiment", {}).get("task", "image")
    if task == "lidar":
        return True
    if task != "fusion":
        return False
    model_cfg = cfg.get("model", {})
    modalities = model_cfg.get("modalities")
    if modalities and "lidar" in modalities:
        return True
    for role in ("teacher", "student"):
        role_modalities = model_cfg.get(role, {}).get("modalities")
        if role_modalities and "lidar" in role_modalities:
            return True
    return bool(cfg.get("data", {}).get("dataset", {}).get("use_lidar", False))


def _validate_modality_subsets(
    model,
    dataloader,
    cfg: dict,
    criterion,
    device: torch.device,
    *,
    official_metrics: dict | None = None,
) -> dict[str, dict]:
    eval_cfg = cfg.get("evaluation", {}).get("modality_subsets", {})
    if not bool(eval_cfg.get("enabled", False)):
        return {}
    if cfg["experiment"].get("task", "image") != "fusion" or not getattr(model, "supports_force_modality_mask", False):
        return {}
    modalities = [str(name) for name in cfg["model"]["student"].get("modalities", ["image", "radar"])]
    requested = eval_cfg.get("subsets") or ["gps", "mmwave", "gps_mmwave", "strong_only", "weak_only", "all"]
    prior = _resolve_validation_prior(model, cfg, modalities, device)
    sampler = ModalitySubsetSampler(
        modalities,
        prior,
        top_prior_k=int(eval_cfg.get("top_prior_k", 2)),
        min_keep=int(eval_cfg.get("min_keep", 1)),
        random_keep_prob=float(eval_cfg.get("random_keep_prob", 0.5)),
    )
    results: dict[str, dict] = {}
    for name in requested:
        subset = _resolve_modality_subset(str(name), modalities, sampler, eval_cfg, device)
        if subset is None:
            continue
        mask = subset.mask[0] if subset.mask.ndim == 2 and subset.mask.shape[0] == 1 else subset.mask
        if str(name) == "all" and bool(mask.to(torch.bool).all()) and official_metrics is not None:
            results[str(name)] = deepcopy(official_metrics)
        else:
            results[str(name)] = _validate_with_force_mask(model, dataloader, cfg, criterion, device, mask)
        results[str(name)]["modalities"] = list(subset.modalities)
        results[str(name)]["mask"] = mask.detach().cpu().to(torch.bool).tolist()
    return results


def _validate_with_force_mask(model, dataloader, cfg: dict, criterion, device: torch.device, mask: torch.Tensor) -> dict:
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    seq_length = model_cfg.get("seq_length_student", 8)
    num_classes = model_cfg.get("num_classes", 64)
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    val_loss = 0.0
    val_occlusion_loss = 0.0
    val_position_loss = 0.0
    val_multitask_loss = 0.0
    all_outputs = []
    all_labels = []
    all_occlusion_logits = []
    all_occlusion_labels = []
    all_occlusion_valid = []
    all_position_outputs = []
    all_position_targets = []
    all_position_valid = []
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_task_batch(batch)
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
            with autocast_context(amp_enabled, device, amp_dtype):
                step = run_model_step(
                    model,
                    task,
                    batch,
                    model_cfg=cfg["model"]["student"],
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                    non_blocking=non_blocking,
                    force_modality_mask=mask,
                )
                outputs = step.logits
                beam_loss = criterion(outputs.reshape(-1, num_classes), labels.flatten())
                prediction_loss = compute_prediction_loss(
                    step.model_output,
                    prediction_targets,
                    cfg,
                    reference=outputs,
                    beam_total_loss=beam_loss,
                    beam_task_loss=beam_loss,
                )
                loss = prediction_loss.total
            val_loss += loss.item()
            val_occlusion_loss += prediction_loss.occlusion.item()
            val_position_loss += prediction_loss.position.item()
            val_multitask_loss += prediction_loss.multitask_total.item()
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if "occlusion_logits" in step.model_output.diagnostics and "occlusion_label" in auxiliary_targets:
                all_occlusion_logits.append(step.model_output.diagnostics["occlusion_logits"].detach().cpu())
                all_occlusion_labels.append(auxiliary_targets["occlusion_label"].detach().cpu())
                all_occlusion_valid.append(auxiliary_targets["occlusion_valid"].detach().cpu())
            if "position" in step.model_output.diagnostics and "position_target" in auxiliary_targets:
                all_position_outputs.append(step.model_output.diagnostics["position"].detach().cpu())
                all_position_targets.append(auxiliary_targets["position_target"].detach().cpu())
                all_position_valid.append(auxiliary_targets["position_valid"].detach().cpu())
    metrics = _metrics_from_outputs(
        val_loss / max(len(dataloader), 1),
        torch.cat(all_outputs, dim=0),
        torch.cat(all_labels, dim=0),
        cfg,
    )
    auxiliary_metrics = _auxiliary_metrics_from_outputs(
        dataloader,
        occlusion_logits=torch.cat(all_occlusion_logits, dim=0) if all_occlusion_logits else None,
        occlusion_labels=torch.cat(all_occlusion_labels, dim=0) if all_occlusion_labels else None,
        occlusion_valid=torch.cat(all_occlusion_valid, dim=0) if all_occlusion_valid else None,
        position_outputs=torch.cat(all_position_outputs, dim=0) if all_position_outputs else None,
        position_targets=torch.cat(all_position_targets, dim=0) if all_position_targets else None,
        position_valid=torch.cat(all_position_valid, dim=0) if all_position_valid else None,
    )
    objective = resolve_prediction_objective(cfg)
    _attach_objective_metrics(
        metrics,
        auxiliary_metrics,
        objective=objective,
        dataloader_len=len(dataloader),
        val_occlusion_loss=val_occlusion_loss,
        val_position_loss=val_position_loss,
        val_multitask_loss=val_multitask_loss,
    )
    metrics["objective"] = objective_runtime_metadata(cfg)
    metrics["available_metrics"] = _available_scalar_metrics(metrics)
    return metrics


def _attach_objective_metrics(
    metrics: dict,
    auxiliary_metrics: dict[str, float],
    *,
    objective: str,
    dataloader_len: int,
    val_occlusion_loss: float,
    val_position_loss: float,
    val_multitask_loss: float,
) -> None:
    auxiliary: dict[str, float] = dict(auxiliary_metrics)
    batches = max(dataloader_len, 1)
    has_occlusion = int(auxiliary_metrics.get("occlusion_total", 0)) > 0
    has_position = int(auxiliary_metrics.get("position_total", 0)) > 0

    if has_occlusion:
        auxiliary["loss_occlusion"] = float(val_occlusion_loss / batches)
        metrics["loss/occlusion"] = auxiliary["loss_occlusion"]
        if "occlusion_accuracy" in auxiliary_metrics:
            metrics["val_occlusion_accuracy"] = float(auxiliary_metrics["occlusion_accuracy"])
        if "occlusion_blocked_f1" in auxiliary_metrics:
            metrics["val_occlusion_blocked_f1"] = float(auxiliary_metrics["occlusion_blocked_f1"])

    if has_position:
        auxiliary["loss_position"] = float(val_position_loss / batches)
        metrics["loss/position"] = auxiliary["loss_position"]
        if "position_rmse" in auxiliary_metrics:
            metrics["val_position_rmse"] = float(auxiliary_metrics["position_rmse"])
        if "position_mae" in auxiliary_metrics:
            metrics["val_position_mae"] = float(auxiliary_metrics["position_mae"])

    if objective == "multitask":
        auxiliary["loss_multitask_total"] = float(val_multitask_loss / batches)
        metrics["loss/multitask_total"] = auxiliary["loss_multitask_total"]
        metrics["val_multitask_loss"] = auxiliary["loss_multitask_total"]

    if auxiliary:
        metrics["auxiliary"] = auxiliary


def _available_scalar_metrics(metrics: dict) -> list[str]:
    available = {"val_loss"}
    if "topk" in metrics and "1" in metrics.get("topk", {}):
        available.add("val_acc")
    if "dba" in metrics:
        available.add("val_adba")
    for key, value in metrics.items():
        if key.startswith("val_") and isinstance(value, (int, float)):
            available.add(key)
    return sorted(available)


def _modality_subset_definitions(modalities: list[str]) -> dict[str, list[str]]:
    strong = [name for name in ("gps", "mmwave") if name in modalities]
    weak = [name for name in ("image", "radar", "lidar") if name in modalities]
    return {
        "gps": ["gps"],
        "mmwave": ["mmwave"],
        "gps_mmwave": strong,
        "strong_only": strong,
        "weak_only": weak,
        "all": list(modalities),
    }


def _resolve_validation_prior(model, cfg: dict, modalities: list[str], device: torch.device) -> torch.Tensor:
    if hasattr(model, "router") and torch.is_tensor(getattr(model.router, "prior", None)):
        return model.router.prior.detach().to(device=device, dtype=torch.float32)
    estimator = getattr(model, "reliability_estimator", None)
    if estimator is not None and hasattr(estimator, "current_prior"):
        try:
            return estimator.current_prior(device=device, dtype=torch.float32).detach()
        except Exception:
            pass
    configured = (
        cfg.get("evaluation", {}).get("modality_subsets", {}).get("prior")
        or cfg.get("model", {}).get("student", {}).get("router", {}).get("dataset_prior")
        or cfg.get("model", {}).get("student", {}).get("reliability", {}).get("dataset_prior")
    )
    if isinstance(configured, dict):
        return torch.tensor([float(configured.get(name, 0.0)) for name in modalities], dtype=torch.float32, device=device)
    if isinstance(configured, (list, tuple)) and len(configured) == len(modalities):
        return torch.tensor([float(value) for value in configured], dtype=torch.float32, device=device)
    return torch.full((len(modalities),), 1.0 / max(len(modalities), 1), dtype=torch.float32, device=device)


def _resolve_modality_subset(
    name: str,
    modalities: list[str],
    sampler: ModalitySubsetSampler,
    eval_cfg: dict,
    device: torch.device,
):
    conditional = resolve_conditional_utility_subset(name, modalities)
    if conditional is not None:
        return sampler.explicit(name, conditional.modalities, device=device)
    if name == "all":
        return sampler.sample("all", device=device)
    if name == "top_prior":
        return sampler.sample("top_prior", device=device)
    if name == "single_best_prior":
        return sampler.sample("single_best_prior", device=device)
    if name in {"low_prior_only", "low_prior", "weak_only"}:
        low_k = eval_cfg.get("low_prior_k")
        return sampler.low_prior(name=name, k=None if low_k is None else int(low_k), device=device)
    if name in {"random", "random_with_top_prior", "drop_one"}:
        return sampler.sample(name, device=device)
    legacy = _modality_subset_definitions(modalities).get(name)
    if legacy:
        return sampler.explicit(name, legacy, device=device)
    if name in modalities:
        return sampler.explicit(name, [name], device=device)
    parts = [part for part in name.split("_") if part]
    if parts and all(part in modalities for part in parts):
        return sampler.explicit(name, parts, device=device)
    return None
