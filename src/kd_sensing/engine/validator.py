from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random

import torch

from kd_sensing.engine.evaluation_pass import run_evaluation_pass
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata
from kd_sensing.evaluation.subset_specs import resolve_named_modality_subset


@dataclass(frozen=True)
class _ModalitySubset:
    name: str
    mask: torch.Tensor
    modalities: tuple[str, ...]


class _ModalitySubsetSampler:
    def __init__(
        self,
        modalities: list[str],
        prior: torch.Tensor,
        *,
        top_prior_k: int = 2,
        min_keep: int = 1,
        random_keep_prob: float = 0.5,
    ) -> None:
        if not modalities:
            raise ValueError("Modality subset evaluation requires at least one modality.")
        self.modalities = tuple(str(name) for name in modalities)
        self.prior = prior.detach().to(dtype=torch.float32)
        if self.prior.numel() != len(self.modalities):
            raise ValueError(
                f"Modality prior length must match modalities: {self.prior.numel()} vs {len(self.modalities)}."
            )
        self.top_prior_k = max(1, int(top_prior_k))
        self.min_keep = max(1, int(min_keep))
        self.random_keep_prob = min(max(float(random_keep_prob), 0.0), 1.0)

    def explicit(self, name: str, selected: list[str] | tuple[str, ...], *, device: torch.device) -> _ModalitySubset:
        selected_set = set(str(item) for item in selected)
        mask = torch.tensor([name in selected_set for name in self.modalities], dtype=torch.bool, device=device)
        if not bool(mask.any()):
            return _ModalitySubset(str(name), mask, ())
        return _ModalitySubset(str(name), mask, tuple(name for name in self.modalities if name in selected_set))

    def low_prior(self, *, name: str = "low_prior_only", k: int | None = None, device: torch.device) -> _ModalitySubset:
        keep = min(max(int(k or max(len(self.modalities) - self.top_prior_k, 1)), 1), len(self.modalities))
        order = torch.argsort(self.prior, descending=False)[:keep]
        mask = torch.zeros(len(self.modalities), dtype=torch.bool, device=device)
        mask[order.to(device=device)] = True
        return _ModalitySubset(str(name), mask, self._modalities_from_mask(mask))

    def sample(self, mode: str, *, device: torch.device) -> _ModalitySubset:
        mode = str(mode)
        if mode == "all":
            mask = torch.ones(len(self.modalities), dtype=torch.bool, device=device)
        elif mode == "top_prior":
            mask = self._top_prior_mask(k=self.top_prior_k, device=device)
        elif mode == "single_best_prior":
            mask = self._top_prior_mask(k=1, device=device)
        elif mode == "random":
            mask = self._random_mask(include_top=False, device=device)
        elif mode == "random_with_top_prior":
            mask = self._random_mask(include_top=True, device=device)
        elif mode == "drop_one":
            mask = torch.ones(len(self.modalities), dtype=torch.bool, device=device)
            if len(self.modalities) > self.min_keep:
                mask[random.randrange(len(self.modalities))] = False
        else:
            raise ValueError(
                "Unsupported modality subset mode "
                f"'{mode}'. Expected all, top_prior, single_best_prior, random, random_with_top_prior, or drop_one."
            )
        if int(mask.sum().item()) < self.min_keep:
            mask = self._top_prior_mask(k=self.min_keep, device=device)
        return _ModalitySubset(mode, mask, self._modalities_from_mask(mask))

    def _top_prior_mask(self, *, k: int, device: torch.device) -> torch.Tensor:
        keep = min(max(int(k), self.min_keep), len(self.modalities))
        order = torch.argsort(self.prior, descending=True)[:keep]
        mask = torch.zeros(len(self.modalities), dtype=torch.bool, device=device)
        mask[order.to(device=device)] = True
        return mask

    def _random_mask(self, *, include_top: bool, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(len(self.modalities), dtype=torch.bool, device=device)
        selected: set[int] = set()
        if include_top:
            selected.add(int(torch.argmax(self.prior).item()))
        for idx in range(len(self.modalities)):
            if random.random() < self.random_keep_prob:
                selected.add(idx)
        while len(selected) < min(self.min_keep, len(self.modalities)):
            selected.add(random.randrange(len(self.modalities)))
        for idx in selected:
            mask[idx] = True
        return mask

    def _modalities_from_mask(self, mask: torch.Tensor) -> tuple[str, ...]:
        flat = mask.detach().cpu().to(torch.bool).flatten().tolist()
        return tuple(name for name, keep in zip(self.modalities, flat) if keep)


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
    for key in (
        "variant",
        "seq_len",
        "num_pred",
        "uses_temporal_core",
        "split_protocol",
        "split_strategy",
        "split_protocol_version",
        "strict_validation_eligible",
        "eligibility_reasons",
        "leakage_diagnostics",
        "split_metadata_path",
        "split_seed",
        "split_sequence_count",
        "split_num_samples",
    ):
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
    modalities = [str(name) for name in cfg["model"]["primary"].get("modalities", ["image", "radar"])]
    requested = eval_cfg.get("subsets") or ["gps", "mmwave", "gps_mmwave", "strong_only", "weak_only", "all"]
    prior = _resolve_validation_prior(model, cfg, modalities, device)
    sampler = _ModalitySubsetSampler(
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
        or cfg.get("model", {}).get("primary", {}).get("router", {}).get("dataset_prior")
        or cfg.get("model", {}).get("primary", {}).get("reliability", {}).get("dataset_prior")
    )
    if isinstance(configured, dict):
        return torch.tensor([float(configured.get(name, 0.0)) for name in modalities], dtype=torch.float32, device=device)
    if isinstance(configured, (list, tuple)) and len(configured) == len(modalities):
        return torch.tensor([float(value) for value in configured], dtype=torch.float32, device=device)
    return torch.full((len(modalities),), 1.0 / max(len(modalities), 1), dtype=torch.float32, device=device)


def _resolve_modality_subset(
    name: str,
    modalities: list[str],
    sampler: _ModalitySubsetSampler,
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
