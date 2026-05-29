from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from kd_sensing.engine.evaluation_pass import run_evaluation_pass
from kd_sensing.engine.marf_training import ModalitySubsetSampler
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata
from kd_sensing.evaluation.hist_beam_outputs import write_hist_beam_predictions
from kd_sensing.evaluation.subset_specs import resolve_named_modality_subset


def validate(model, dataloader, cfg: dict, criterion, device: torch.device, output_dir: str | Path | None = None):
    result = run_evaluation_pass(model, dataloader, cfg, criterion, device)
    metrics = result.metrics
    subset_metrics = _validate_modality_subsets(model, dataloader, cfg, criterion, device, official_metrics=metrics)
    if subset_metrics:
        metrics["modality_subsets"] = subset_metrics
    dataset = getattr(dataloader, "dataset", None)
    split_metadata = None
    if dataset is not None:
        split_metadata = {getattr(dataset, "split", "test"): dataset_run_metadata(dataset)}
    setup = prediction_setup_metadata(cfg, split_metadata=split_metadata)
    metrics["prediction_setup"] = setup
    for key in ("variant", "seq_len", "num_pred", "uses_temporal_core", "split_protocol"):
        if key in setup:
            metrics[key] = setup[key]
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if _hist_beam_output_enabled(cfg):
            write_hist_beam_predictions(
                target / "predictions.csv",
                result.outputs,
                result.labels,
                metadata=result.metadata,
                group_size=int(
                    cfg.get("hist_beam", {}).get(
                        "group_size",
                        cfg.get("model", {}).get("student", {}).get("group_size", 8),
                    )
                ),
                top_k=max(int(value) for value in cfg.get("evaluation", {}).get("k_values", [1, 3, 5])),
                variant_metadata=setup,
                radio_logits=result.radio_logits,
                radio_labels=result.radio_labels,
                path_logits=result.path_logits,
                path_labels=result.path_labels,
            )
    return metrics


def _hist_beam_output_enabled(cfg: dict) -> bool:
    hist_cfg = cfg.get("hist_beam")
    if isinstance(hist_cfg, dict) and hist_cfg.get("enabled") is not False:
        return True
    return cfg.get("model", {}).get("student", {}).get("type") == "hist_beam_fusion"


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
    return run_evaluation_pass(
        model,
        dataloader,
        cfg,
        criterion,
        device,
        force_modality_mask=mask,
    ).metrics


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
    if name.startswith("drop_"):
        drop = [part for part in name.removeprefix("drop_").split("_") if part]
        if drop and all(part in modalities for part in drop):
            keep = [modality for modality in modalities if modality not in set(drop)]
            if keep:
                return sampler.explicit(name, keep, device=device)
    if name == "top_prior":
        return sampler.sample("top_prior", device=device)
    if name == "single_best_prior":
        return sampler.sample("single_best_prior", device=device)
    if name == "weak_only":
        generic = resolve_named_modality_subset(name, modalities)
        if generic is not None:
            return sampler.explicit(name, generic.modalities, device=device)
        low_k = eval_cfg.get("low_prior_k")
        return sampler.low_prior(name=name, k=None if low_k is None else int(low_k), device=device)
    if name in {"low_prior_only", "low_prior"}:
        low_k = eval_cfg.get("low_prior_k")
        return sampler.low_prior(name=name, k=None if low_k is None else int(low_k), device=device)
    if name in {"random", "random_with_top_prior", "drop_one"}:
        return sampler.sample(name, device=device)
    generic = resolve_named_modality_subset(name, modalities)
    if generic is not None:
        return sampler.explicit(name, generic.modalities, device=device)
    return None
