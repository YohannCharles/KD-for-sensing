from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

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
from kd_sensing.engine.marf_training import ModalitySubsetSampler
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots
from kd_sensing.engine.runtime import autocast_context, resolve_amp_settings, transfer_non_blocking
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy


def validate(model, dataloader, cfg: dict, criterion, device: torch.device, output_dir: str | Path | None = None):
    model.eval()
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    seq_length = model_cfg.get("seq_length_student", 8)
    num_classes = model_cfg.get("num_classes", 64)
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = normalize_batch(batch)
            labels = prepare_labels(
                batch,
                num_pred=num_pred,
                downsample_ratio=downsample_ratio,
                device=device,
                non_blocking=non_blocking,
            )
            with autocast_context(amp_enabled, device, amp_dtype):
                if task == "fusion":
                    fusion_inputs = prepare_fusion_inputs(
                        batch,
                        seq_length=seq_length,
                        num_pred=num_pred,
                        device=device,
                        modalities=cfg["model"]["student"].get("modalities"),
                        non_blocking=non_blocking,
                    )
                    model_output = adapt_model_output(forward_model(model, task, **fusion_inputs))
                    outputs = model_output.logits
                elif task == "radar":
                    radar_batch = prepare_radar_inputs(
                        batch,
                        seq_length=seq_length,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    model_output = adapt_model_output(forward_model(model, task, radar_batch=radar_batch))
                    outputs = model_output.logits
                elif task == "gps":
                    gps_batch = prepare_gps_inputs(
                        batch,
                        seq_length=seq_length,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    model_output = adapt_model_output(forward_model(model, task, gps_batch=gps_batch))
                    outputs = model_output.logits
                elif task == "lidar":
                    lidar_batch = prepare_lidar_inputs(
                        batch,
                        seq_length=seq_length,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    model_output = adapt_model_output(forward_model(model, task, lidar_batch=lidar_batch))
                    outputs = model_output.logits
                elif task == "mmwave":
                    mmwave_batch = prepare_mmwave_inputs(
                        batch,
                        seq_length=seq_length,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    model_output = adapt_model_output(forward_model(model, task, mmwave_batch=mmwave_batch))
                    outputs = model_output.logits
                else:
                    image_batch = prepare_image_inputs(
                        batch,
                        seq_length=seq_length,
                        num_pred=num_pred,
                        device=device,
                        non_blocking=non_blocking,
                    )
                    model_output = adapt_model_output(forward_model(model, task, image_batch))
                    outputs = model_output.logits
                outputs = select_prediction_slots(outputs, num_pred)
                loss = criterion(outputs.reshape(-1, num_classes), labels.flatten())
            val_loss += loss.item()
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
    val_loss = val_loss / max(len(dataloader), 1)
    all_outputs_t = torch.cat(all_outputs, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    metrics = _metrics_from_outputs(val_loss, all_outputs_t, all_labels_t, cfg)
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
    return {
        "loss": float(loss),
        "topk": {str(k): v.tolist() for k, v in topk_acc.items()},
        "dba": dba_score.tolist(),
        "total": total.tolist(),
    }


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
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = normalize_batch(batch)
            labels = prepare_labels(
                batch,
                num_pred=num_pred,
                downsample_ratio=downsample_ratio,
                device=device,
                non_blocking=non_blocking,
            )
            with autocast_context(amp_enabled, device, amp_dtype):
                fusion_inputs = prepare_fusion_inputs(
                    batch,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                    modalities=cfg["model"]["student"].get("modalities"),
                    non_blocking=non_blocking,
                )
                model_output = adapt_model_output(
                    forward_model(model, task, **fusion_inputs, force_modality_mask=mask)
                )
                outputs = select_prediction_slots(model_output.logits, num_pred)
                loss = criterion(outputs.reshape(-1, num_classes), labels.flatten())
            val_loss += loss.item()
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
    return _metrics_from_outputs(
        val_loss / max(len(dataloader), 1),
        torch.cat(all_outputs, dim=0),
        torch.cat(all_labels, dim=0),
        cfg,
    )


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
    if name == "all":
        return sampler.sample("all", device=device)
    if name in {"top_prior", "strong_only"}:
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
