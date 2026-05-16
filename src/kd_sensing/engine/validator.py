from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import torch

from kd_sensing.engine.evaluation_pass import run_evaluation_pass
from kd_sensing.engine.marf_training import ModalitySubsetSampler
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata
from kd_sensing.evaluation.subset_specs import resolve_conditional_utility_subset


def validate(model, dataloader, cfg: dict, criterion, device: torch.device, output_dir: str | Path | None = None):
    result = run_evaluation_pass(model, dataloader, cfg, criterion, device)
    metrics = result.metrics
    subset_metrics = _validate_modality_subsets(model, dataloader, cfg, criterion, device, official_metrics=metrics)
    if subset_metrics:
        metrics["modality_subsets"] = subset_metrics
    setup = prediction_setup_metadata(
        cfg,
        split_metadata={getattr(dataloader.dataset, "split", "test"): dataset_run_metadata(dataloader.dataset)},
    )
    metrics["prediction_setup"] = setup
    for key in ("variant", "seq_len", "num_pred", "uses_temporal_core", "split_protocol"):
        if key in setup:
            metrics[key] = setup[key]
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return metrics


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
