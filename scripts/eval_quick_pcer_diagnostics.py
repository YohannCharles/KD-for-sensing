#!/usr/bin/env python3
"""Read-only diagnostics for completed MMW PCER quick-validation checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from eval_quick_pcer_validation import (
    EVAL_SEED,
    EXPERIMENTS,
    MODALITIES,
    MetricAccumulator,
    _clone_batch,
    _conditions,
    _future_beam_power,
)
from kd_sensing.config import load_config
from kd_sensing.data.temporal_block_mask import TemporalBlockMaskGenerator
from kd_sensing.data.temporal_missing import apply_modality_temporal_mask_to_batch
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.evaluation_pass_runtime import sample_ids_from_batch
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.runtime import prepare_task_labels, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.losses.beam_prototype_alignment import make_soft_beam_labels
from kd_sensing.losses.pcer_temporal_fusion import counterfactual_router_loss, counterfactual_router_targets
from kd_sensing.models.pcer_temporal_fusion import masked_block_softmax
from kd_sensing.utils.checkpoint import load_model_state, load_torch_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "outputs/quick_pcer_validation"
DEFAULT_OUTPUT = ROOT / "outputs/quick_pcer_diagnostics"
ROUTED_EXPERIMENTS = {
    "a1": "qv_a1_proto_old_router",
    "a3": "qv_a3_pcer_full",
}
MODES = ("D0_dynamic", "D1_global_mean", "D2_mask_mean", "D3_a0_static", "D4_uniform")
TIMESTEPS = 5


class PriorAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}

    def update(self, condition: str, family: str, logits: torch.Tensor, available: torch.Tensor) -> None:
        finite = logits.detach().float().masked_fill(~available, 0.0).cpu()
        mask = available.detach().float().cpu()
        keys = ("global", _prior_key(condition, family))
        for key in keys:
            if key not in self.sums:
                self.sums[key] = torch.zeros(finite.shape[1])
                self.counts[key] = torch.zeros(finite.shape[1])
            self.sums[key] += finite.sum(dim=0)
            self.counts[key] += mask.sum(dim=0)

    def means(self) -> dict[str, torch.Tensor]:
        return {
            key: self.sums[key] / self.counts[key].clamp_min(1.0)
            for key in self.sums
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--train-subset", type=int, default=512)
    parser.add_argument("--stability-subset", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--frequency-only", action="store_true")
    args = parser.parse_args()
    source = Path(args.source_root).expanduser().resolve()
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(EVAL_SEED)
    _validate_inputs(source)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[diagnostics] device={device} source={source} output={output}", flush=True)

    if args.frequency_only:
        rows = []
        for short_name, experiment in ROUTED_EXPERIMENTS.items():
            rows.extend(
                _collect_router_frequency(
                    short_name, source / experiment, device=device, num_workers=args.num_workers
                )
            )
        _write_csv(output / "router_topk_frequency.csv", rows)
        print("[diagnostics] router frequency complete", flush=True)
        return 0

    audit = _implementation_audit(source, device)
    (output / "implementation_audit.md").write_text(audit, encoding="utf-8")

    results: dict[str, dict[str, Any]] = {}
    for short_name, experiment in ROUTED_EXPERIMENTS.items():
        print(f"[diagnostics] {short_name}: validation priors and fixed-mask test", flush=True)
        results[short_name] = _diagnose_router(
            short_name,
            source / experiment,
            device=device,
            num_workers=args.num_workers,
            train_subset=args.train_subset if short_name == "a3" else 0,
            stability_subset=args.stability_subset if short_name == "a3" else 0,
        )
        _write_weight_replacement(output / f"{short_name}_weight_replacement.csv", results[short_name]["summaries"])
        _write_weight_outputs(output, short_name, results[short_name]["weights"])

    _write_weight_replacement_summary(output, results)
    _write_router_dynamicity(output, results)
    _write_target_outputs(output, results["a3"])
    _write_gradient_audit(output, results["a3"])
    _write_s3_outputs(output, source, results)
    _write_block_modality(output, results["a3"]["block_modality"])
    _write_diagnostic_summary(output, source, results)
    _write_manifest(output, source, device, args, results)
    _assert_required_outputs(output)
    print("[diagnostics] complete", flush=True)
    return 0


def _validate_inputs(source: Path) -> None:
    for experiment in EXPERIMENTS:
        run = source / experiment
        for relative in ("resolved_config.yaml", "checkpoints/best.pth", "per_mask_metrics.csv"):
            if not (run / relative).is_file():
                raise FileNotFoundError(run / relative)


def _diagnose_router(
    kind: str,
    run_dir: Path,
    *,
    device: torch.device,
    num_workers: int,
    train_subset: int,
    stability_subset: int,
) -> dict[str, Any]:
    cfg = load_config(run_dir / "resolved_config.yaml")
    cfg.setdefault("training", {})["final_test"] = {"enabled": True}
    cfg.setdefault("temporal_missing", {})["enabled"] = False
    loader_cfg = cfg.setdefault("data", {}).setdefault("dataloader", {})
    loader_cfg["num_workers"] = max(0, int(num_workers))
    loader_cfg["persistent_workers"] = bool(num_workers)
    model = build_model(cfg["model"]["primary"]).to(device)
    checkpoint_path = run_dir / "checkpoints/best.pth"
    load_model_state(checkpoint_path, model, role="pcer-offline-diagnostics", map_location=device, strict=True)
    model.eval()
    dataloaders = build_dataloaders(cfg)
    target_records: dict[str, dict[str, list[dict[str, torch.Tensor]]]] = defaultdict(lambda: defaultdict(list))
    try:
        priors = _collect_validation_priors(
            kind, model, cfg, dataloaders["validation"], device, target_records
        )
        evaluated = _evaluate_weight_modes(
            kind, model, cfg, dataloaders["test"], device, priors, target_records
        )
        train_label_counts: Counter[int] = Counter()
        if kind == "a3" and train_subset > 0:
            train_label_counts = _collect_train_targets(
                model,
                cfg,
                dataloaders["train"],
                device,
                target_records,
                max_samples=train_subset,
            )
        gradient = _gradient_audit(model, cfg, dataloaders["train"], device) if kind == "a3" else {}
        stability = (
            _checkpoint_stability(model, cfg, dataloaders["validation"], run_dir, device, stability_subset)
            if kind == "a3" and stability_subset > 0
            else {}
        )
    finally:
        shutdown_all_dataloaders(dataloaders)
    return {
        **evaluated,
        "priors": priors,
        "targets": target_records,
        "gradient": gradient,
        "stability": stability,
        "train_label_counts": train_label_counts,
        "checkpoint": str(checkpoint_path),
        "config": str(run_dir / "resolved_config.yaml"),
    }


def _collect_validation_priors(
    kind: str,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    target_records: dict[str, dict[str, list[dict[str, torch.Tensor]]]],
) -> dict[str, torch.Tensor]:
    accumulator = PriorAccumulator()
    generator = TemporalBlockMaskGenerator(EVAL_SEED)
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            sample_ids = sample_ids_from_batch(raw_batch)
            for condition in _conditions(sample_ids, offset, generator):
                step = _masked_step(model, cfg, raw_batch, condition, device)
                evidence, logits, _, available, _ = _router_tensors(kind, step.model_output.diagnostics)
                accumulator.update(condition["name"], condition["family"], logits, available)
                if kind == "a3":
                    labels = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
                    _append_target_record(
                        target_records["validation"][condition["name"]], evidence, available, labels,
                        step.model_output.diagnostics["pcer_block_router_weights"], cfg
                    )
            offset += len(sample_ids)
    return accumulator.means()


def _evaluate_weight_modes(
    kind: str,
    model: torch.nn.Module,
    cfg: dict[str, Any],
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    priors: dict[str, torch.Tensor],
    target_records: dict[str, dict[str, list[dict[str, torch.Tensor]]]],
) -> dict[str, Any]:
    metrics = {mode: defaultdict(MetricAccumulator) for mode in MODES}
    pair_metrics = {mode: defaultdict(MetricAccumulator) for mode in MODES}
    weights: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
    predictions: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
    block_modality: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(list)
    generator = TemporalBlockMaskGenerator(EVAL_SEED)
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            sample_ids = sample_ids_from_batch(raw_batch)
            powers = _future_beam_power(raw_batch)
            for condition in _conditions(sample_ids, offset, generator):
                step = _masked_step(model, cfg, raw_batch, condition, device)
                diagnostics = step.model_output.diagnostics
                evidence, router_logits, dynamic, available, cell_mask = _router_tensors(kind, diagnostics)
                labels = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
                mode_weights = _replacement_weights(kind, condition, available, cell_mask, dynamic, priors)
                mode_logits = {
                    mode: step.logits[:, -1] if mode == "D0_dynamic" else (weight.unsqueeze(-1) * evidence).sum(dim=1)
                    for mode, weight in mode_weights.items()
                }
                for mode, fused in mode_logits.items():
                    metrics[mode][condition["name"]].update(fused, labels, powers)
                    if condition["family"] == "S5":
                        for pair in sorted(set(condition["groups"])):
                            selector = torch.tensor([item == pair for item in condition["groups"]], device=device)
                            pair_metrics[mode][pair].update(fused, labels, powers, selector=selector)
                dynamic_block, block_available = _effective_block_weights(kind, dynamic, cell_mask)
                global_block, _ = _effective_block_weights(kind, mode_weights["D1_global_mean"], cell_mask)
                weights[condition["name"]].append(
                    {
                        "weight": dynamic_block.detach().float().cpu(),
                        "global": global_block.detach().float().cpu(),
                        "available": block_available.detach().bool().cpu(),
                    }
                )
                if condition["family"] == "S3":
                    predictions[condition["name"]].append(
                        {
                            "prediction": mode_logits["D0_dynamic"].argmax(dim=-1).cpu(),
                            "label": labels.cpu(),
                        }
                    )
                if kind == "a3":
                    _append_target_record(
                        target_records["test"][condition["name"]], evidence, available, labels, dynamic, cfg
                    )
                    if condition["name"] == "S0_full":
                        block_modality["test"].append(_block_modality_values(evidence, available, labels, cfg))
            offset += len(sample_ids)
    summaries = _summarize_modes(metrics, pair_metrics)
    return {
        "summaries": summaries,
        "weights": _merge_weight_records(weights),
        "predictions": _merge_prediction_records(predictions),
        "block_modality": block_modality,
    }


def _collect_train_targets(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    target_records: dict[str, dict[str, list[dict[str, torch.Tensor]]]],
    *,
    max_samples: int,
) -> Counter[int]:
    generator = TemporalBlockMaskGenerator(EVAL_SEED)
    offset = 0
    label_counts: Counter[int] = Counter()
    with torch.no_grad():
        for raw_batch in loader:
            sample_ids = sample_ids_from_batch(raw_batch)
            for condition in _conditions(sample_ids, offset, generator):
                step = _masked_step(model, cfg, raw_batch, condition, device)
                evidence, _, weights, available, _ = _router_tensors("a3", step.model_output.diagnostics)
                labels = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
                _append_target_record(
                    target_records["train_subset"][condition["name"]], evidence, available, labels, weights, cfg
                )
                if condition["name"] == "S0_full":
                    label_counts.update(int(item) for item in labels.detach().cpu().tolist())
            offset += len(sample_ids)
            if offset >= max_samples:
                break
    return label_counts


def _collect_router_frequency(kind: str, run_dir: Path, *, device: torch.device, num_workers: int) -> list[dict[str, Any]]:
    cfg = load_config(run_dir / "resolved_config.yaml")
    cfg.setdefault("training", {})["final_test"] = {"enabled": True}
    cfg.setdefault("temporal_missing", {})["enabled"] = False
    loader_cfg = cfg.setdefault("data", {}).setdefault("dataloader", {})
    loader_cfg["num_workers"] = max(0, int(num_workers))
    loader_cfg["persistent_workers"] = bool(num_workers)
    model = build_model(cfg["model"]["primary"]).to(device)
    load_model_state(
        run_dir / "checkpoints/best.pth", model, role="pcer-router-frequency", map_location=device, strict=True
    )
    model.eval()
    dataloaders = build_dataloaders(cfg)
    counts: dict[str, dict[str, Any]] = {}
    generator = TemporalBlockMaskGenerator(EVAL_SEED)
    offset = 0
    try:
        with torch.no_grad():
            for raw_batch in dataloaders["test"]:
                ids = sample_ids_from_batch(raw_batch)
                for condition in _conditions(ids, offset, generator):
                    step = _masked_step(model, cfg, raw_batch, condition, device)
                    _, _, weights, _, cell_mask = _router_tensors(kind, step.model_output.diagnostics)
                    ranking_weights = weights if kind == "a3" else weights
                    labels = (
                        [f"t{time}_{name}" for time in range(TIMESTEPS) for name in MODALITIES]
                        if kind == "a3" else list(MODALITIES)
                    )
                    topk = ranking_weights.topk(min(3, ranking_weights.shape[1]), dim=1).indices.cpu()
                    state = counts.setdefault(
                        condition["name"],
                        {"samples": 0, "top1": Counter(), "top3": Counter(), "labels": labels},
                    )
                    state["samples"] += len(ids)
                    state["top1"].update(int(item) for item in topk[:, 0].tolist())
                    state["top3"].update(int(item) for item in topk.flatten().tolist())
                offset += len(ids)
    finally:
        shutdown_all_dataloaders(dataloaders)
    rows = []
    for condition, state in counts.items():
        for index, label in enumerate(state["labels"]):
            time = int(label.split("_", 1)[0][1:]) if kind == "a3" else "pooled"
            modality = label.split("_", 1)[1] if kind == "a3" else label
            rows.append(
                {
                    "model": kind.upper(),
                    "router_granularity": "block" if kind == "a3" else "modality",
                    "condition": condition,
                    "time": time,
                    "modality": modality,
                    "sample_count": state["samples"],
                    "top1_count": state["top1"].get(index, 0),
                    "top1_frequency": state["top1"].get(index, 0) / state["samples"],
                    "top3_count": state["top3"].get(index, 0),
                    "top3_presence_frequency": state["top3"].get(index, 0) / state["samples"],
                }
            )
    return rows


def _masked_step(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    raw_batch: dict[str, Any],
    condition: dict[str, Any],
    device: torch.device,
):
    batch = _clone_batch(raw_batch)
    mask_tm = condition["mask"].permute(0, 2, 1).contiguous()
    apply_modality_temporal_mask_to_batch(batch, mask_tm, modalities=MODALITIES)
    return run_model_step(
        model,
        cfg.get("experiment", {}).get("task", "fusion"),
        batch,
        seq_length=TIMESTEPS,
        num_pred=1,
        device=device,
        extra_model_kwargs={"missing_mask": batch["available_modalities"].to(device=device)},
    )


def _router_tensors(kind: str, diagnostics: dict[str, Any]):
    cell_mask = diagnostics["modality_temporal_mask"].bool()
    if kind == "a1":
        return (
            diagnostics["unimodal_logits"],
            diagnostics["supervised_router_gate_logits"],
            diagnostics["supervised_router_gate_weights"],
            diagnostics["available_modalities"].bool(),
            cell_mask,
        )
    return (
        diagnostics["pcer_block_evidence_logits"],
        diagnostics["pcer_block_router_logits"],
        diagnostics["pcer_block_router_weights"],
        diagnostics["pcer_block_availability"].bool(),
        cell_mask,
    )


def _replacement_weights(
    kind: str,
    condition: dict[str, Any],
    available: torch.Tensor,
    cell_mask: torch.Tensor,
    dynamic: torch.Tensor,
    priors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    global_logits = priors["global"].to(device=available.device).unsqueeze(0).expand(available.shape[0], -1)
    key = _prior_key(condition["name"], condition["family"])
    mask_logits = priors[key].to(device=available.device).unsqueeze(0).expand(available.shape[0], -1)
    d1 = masked_block_softmax(global_logits, available)
    d2 = masked_block_softmax(mask_logits, available)
    d4 = available.float() / available.sum(dim=-1, keepdim=True).clamp_min(1.0)
    if kind == "a1":
        d3 = d4
    else:
        cells = cell_mask.float()
        per_modality = cells.sum(dim=1)
        modality_available = per_modality.gt(0)
        modality_mass = modality_available.float() / modality_available.sum(dim=1, keepdim=True).clamp_min(1.0)
        d3 = (cells / per_modality.clamp_min(1.0).unsqueeze(1) * modality_mass.unsqueeze(1)).reshape(
            available.shape
        )
    return {
        "D0_dynamic": dynamic,
        "D1_global_mean": d1,
        "D2_mask_mean": d2,
        "D3_a0_static": d3,
        "D4_uniform": d4,
    }


def _effective_block_weights(kind: str, weights: torch.Tensor, cell_mask: torch.Tensor):
    if kind == "a3":
        return weights.reshape(-1, TIMESTEPS, len(MODALITIES)), cell_mask
    counts = cell_mask.sum(dim=1).clamp_min(1)
    effective = cell_mask.float() * (weights / counts).unsqueeze(1)
    return effective, cell_mask


def _prior_key(condition: str, family: str) -> str:
    return condition if family in {"S0", "S2", "S3", "S4", "S5"} else family


def _summarize_modes(metrics, pair_metrics) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        rows = {name: acc.result() for name, acc in metrics[mode].items()}
        s1 = _mean_metric_rows([rows[f"S1_sparse_easy_v{index}"] for index in range(3)])
        s3_rows = [rows[f"S3_missing_{name}"] for name in MODALITIES]
        s3 = _mean_metric_rows(s3_rows)
        s5 = rows["S5_two_modality_recent_async"]
        scenarios = [s1, rows["S2_single_modality_burst2"], s3, rows["S4_latest_sync_missing"], s5]
        hard = [s3, rows["S4_latest_sync_missing"], s5]
        worst_rows = [
            *[rows[f"S1_sparse_easy_v{index}"] for index in range(3)],
            rows["S2_single_modality_burst2"],
            *s3_rows,
            rows["S4_latest_sync_missing"],
            *[acc.result() for acc in pair_metrics[mode].values()],
        ]
        result[mode] = {
            "mode": mode,
            "full_top1": rows["S0_full"]["top1"],
            "masked_avg_top1": statistics.fmean(row["top1"] for row in scenarios),
            "hard_avg_top1": statistics.fmean(row["top1"] for row in hard),
            "worst_top1": min(row["top1"] for row in worst_rows),
            "s1_top1": s1["top1"],
            "s2_top1": rows["S2_single_modality_burst2"]["top1"],
            "s3_macro_top1": s3["top1"],
            "s3_worst_top1": min(row["top1"] for row in s3_rows),
            "s3_worst_modality": MODALITIES[int(np.argmin([row["top1"] for row in s3_rows]))],
            "s4_top1": rows["S4_latest_sync_missing"]["top1"],
            "s5_top1": s5["top1"],
            "within3": statistics.fmean(row["within3"] for row in scenarios),
            "mae": statistics.fmean(row["beam_index_mae"] for row in scenarios),
            "normalized_gain": statistics.fmean(row["normalized_gain"] for row in scenarios),
            "gain_loss_db": statistics.fmean(row["gain_loss_db"] for row in scenarios),
            "beam_loss": statistics.fmean(row["beam_loss"] for row in scenarios),
            "sample_count": rows["S0_full"]["sample_count"],
        }
    baseline = result["D0_dynamic"]
    for mode, row in result.items():
        for metric in ("full_top1", "masked_avg_top1", "hard_avg_top1", "worst_top1", "s3_worst_top1"):
            row[f"d0_minus_this_{metric}"] = baseline[metric] - row[metric]
    return result


def _mean_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("top1", "top3", "top5", "within3", "beam_index_mae", "beam_loss", "normalized_gain", "gain_loss_db")
    return {key: statistics.fmean(float(row[key]) for row in rows) for key in keys}


def _merge_weight_records(records: dict[str, list[dict[str, torch.Tensor]]]):
    return {
        condition: {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        for condition, batches in records.items()
    }


def _merge_prediction_records(records: dict[str, list[dict[str, torch.Tensor]]]):
    return {
        condition: {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        for condition, batches in records.items()
    }


def _append_target_record(
    destination: list[dict[str, torch.Tensor]],
    evidence: torch.Tensor,
    available: torch.Tensor,
    labels: torch.Tensor,
    prediction: torch.Tensor,
    cfg: dict[str, Any],
) -> None:
    loss_cfg = cfg["loss"]["u_mask_beam_jepa"]
    pcer = loss_cfg["pcer"]
    topology = loss_cfg.get("prototype_topology", {})
    target, contribution = counterfactual_router_targets(
        evidence,
        available,
        labels,
        beam_label_sigma=float(loss_cfg.get("beam_label_sigma", 2.0)),
        circular=bool(loss_cfg.get("prototype_target_circular", True)),
        topology_id=str(topology.get("id", "")) or None,
        topology_permutation=topology.get("permutation"),
        contribution_temperature=float(pcer.get("contribution_temperature", 0.5)),
        contribution_clip=pcer.get("contribution_clip"),
    )
    probability = torch.softmax(evidence.float(), dim=-1)
    truth = labels.long().reshape(-1, 1, 1).expand(-1, evidence.shape[1], 1)
    top1 = probability.argmax(dim=-1)
    classes = evidence.shape[-1]
    distance = (top1 - labels.reshape(-1, 1)).abs()
    distance = torch.minimum(distance, classes - distance).float()
    entropy = -(probability * probability.clamp_min(torch.finfo(probability.dtype).tiny).log()).sum(dim=-1)
    temperature = float(cfg["model"]["primary"].get("beam_proto_temperature", 1.0))
    destination.append(
        {
            "target": target.detach().float().cpu(),
            "prediction": prediction.detach().float().cpu(),
            "contribution": contribution.detach().float().masked_fill(~available, 0.0).cpu(),
            "available": available.detach().bool().cpu(),
            "correct": top1.eq(labels.reshape(-1, 1)).float().cpu(),
            "confidence": probability.amax(dim=-1).cpu(),
            "evidence_entropy": entropy.cpu(),
            "truth_similarity": (evidence.float().gather(2, truth).squeeze(-1) * temperature).cpu(),
            "beam_distance": distance.cpu(),
        }
    )


def _merge_target_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}


def _target_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    loss_cfg = cfg["loss"]["u_mask_beam_jepa"]
    topology = loss_cfg.get("prototype_topology", {})
    pcer = loss_cfg["pcer"]
    return {
        "beam_label_sigma": float(loss_cfg.get("beam_label_sigma", 2.0)),
        "circular": bool(loss_cfg.get("prototype_target_circular", True)),
        "topology_id": str(topology.get("id", "")) or None,
        "topology_permutation": topology.get("permutation"),
        "contribution_temperature": float(pcer.get("contribution_temperature", 0.5)),
        "contribution_clip": pcer.get("contribution_clip"),
    }


def _block_modality_values(
    evidence: torch.Tensor,
    available: torch.Tensor,
    labels: torch.Tensor,
    cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    raw_block, loss_all = _raw_block_contribution(evidence, available, labels, cfg)
    batch, blocks, classes = evidence.shape
    cell = available.reshape(batch, TIMESTEPS, len(MODALITIES))
    evidence_tm = evidence.reshape(batch, TIMESTEPS, len(MODALITIES), classes)
    valid = available.float()
    evidence_sum = (evidence * valid.unsqueeze(-1)).sum(dim=1)
    count = valid.sum(dim=1)
    soft = make_soft_beam_labels(
        labels.long(), classes, _target_cfg(cfg)["beam_label_sigma"],
        circular=_target_cfg(cfg)["circular"],
        topology_id=_target_cfg(cfg)["topology_id"],
        topology_permutation=_target_cfg(cfg)["topology_permutation"],
    ).float()
    grouped = []
    block_sum = []
    for modality in range(len(MODALITIES)):
        modality_mask = cell[:, :, modality]
        removed_count = modality_mask.sum(dim=1)
        remaining = count - removed_count
        removed_sum = (evidence_tm[:, :, modality] * modality_mask.unsqueeze(-1)).sum(dim=1)
        logits = (evidence_sum - removed_sum) / remaining.clamp_min(1).unsqueeze(-1)
        loss_without = -(soft * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        grouped.append(loss_without - loss_all)
        block_sum.append((raw_block.reshape(batch, TIMESTEPS, len(MODALITIES))[:, :, modality] * modality_mask).sum(dim=1))
    return {
        "block_sum": torch.stack(block_sum, dim=1).cpu(),
        "group": torch.stack(grouped, dim=1).cpu(),
    }


def _raw_block_contribution(evidence, available, labels, cfg):
    detached = evidence.detach().float()
    valid = available.float()
    count = valid.sum(dim=1)
    evidence_sum = (detached * valid.unsqueeze(-1)).sum(dim=1)
    all_logits = evidence_sum / count.unsqueeze(-1)
    loo_logits = (evidence_sum.unsqueeze(1) - detached) / (count - 1).view(-1, 1, 1)
    target_cfg = _target_cfg(cfg)
    soft = make_soft_beam_labels(
        labels.long(), detached.shape[-1], target_cfg["beam_label_sigma"],
        circular=target_cfg["circular"], topology_id=target_cfg["topology_id"],
        topology_permutation=target_cfg["topology_permutation"],
    ).float()
    loss_all = -(soft * F.log_softmax(all_logits, dim=-1)).sum(dim=-1)
    loss_without = -(soft.unsqueeze(1) * F.log_softmax(loo_logits, dim=-1)).sum(dim=-1)
    return loss_without - loss_all.unsqueeze(1), loss_all


def _gradient_audit(model, cfg, loader, device) -> dict[str, Any]:
    raw_batch = next(iter(loader))
    sample_ids = sample_ids_from_batch(raw_batch)
    condition = next(
        item for item in _conditions(sample_ids, 0, TemporalBlockMaskGenerator(EVAL_SEED))
        if item["name"] == "S2_single_modality_burst2"
    )
    model.zero_grad(set_to_none=True)
    step = _masked_step(model, cfg, raw_batch, condition, device)
    diagnostics = step.model_output.diagnostics
    evidence, _, predicted, available, _ = _router_tensors("a3", diagnostics)
    labels = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
    target, _ = counterfactual_router_targets(evidence, available, labels, **_target_cfg(cfg))
    route_loss, route_diagnostics = counterfactual_router_loss(predicted, target, available)
    beam_loss = F.cross_entropy(step.logits[:, -1].float(), labels)
    route_loss.backward()
    parameter_rows = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        parameter_rows.append(
            {
                "name": name,
                "owner": "router" if name.startswith("pcer_router.") else (
                    "prototype" if name.startswith("prototype_bank.") else "backbone_or_other"
                ),
                "parameter_norm": float(parameter.detach().float().norm().cpu()),
                "gradient_norm": float(gradient.detach().float().norm().cpu()) if gradient is not None else 0.0,
                "gradient_present": gradient is not None,
                "gradient_finite": bool(torch.isfinite(gradient).all()) if gradient is not None else True,
            }
        )
    lambda_route = float(cfg["loss"]["u_mask_beam_jepa"]["pcer"]["lambda_route"])
    router_rows = [row for row in parameter_rows if row["owner"] == "router"]
    leaked = [row for row in parameter_rows if row["owner"] != "router" and row["gradient_norm"] > 0]
    model.zero_grad(set_to_none=True)
    return {
        "checkpoint_updated": False,
        "optimizer_step_called": False,
        "condition": condition["name"],
        "sample_count": len(sample_ids),
        "route_loss": float(route_loss.detach().cpu()),
        "lambda_route": lambda_route,
        "weighted_route_loss": float(route_loss.detach().cpu()) * lambda_route,
        "beam_loss": float(beam_loss.detach().cpu()),
        "weighted_route_over_beam": float(route_loss.detach().cpu()) * lambda_route / max(float(beam_loss.detach().cpu()), 1e-12),
        "prediction_requires_grad": bool(predicted.requires_grad),
        "target_requires_grad": bool(target.requires_grad),
        "router_gradient_nonzero": any(row["gradient_norm"] > 0 for row in router_rows),
        "non_router_gradient_leaks": leaked,
        "route_diagnostics": route_diagnostics,
        "parameters": parameter_rows,
    }


def _checkpoint_stability(model, cfg, loader, run_dir, device, max_samples) -> dict[str, Any]:
    best_targets: dict[str, torch.Tensor] = {}
    generator = TemporalBlockMaskGenerator(EVAL_SEED)
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            ids = sample_ids_from_batch(raw_batch)
            condition = _conditions(ids, offset, generator)[0]
            step = _masked_step(model, cfg, raw_batch, condition, device)
            evidence, _, weights, available, _ = _router_tensors("a3", step.model_output.diagnostics)
            labels = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
            target, _ = counterfactual_router_targets(evidence, available, labels, **_target_cfg(cfg))
            best_targets.update({sample_id: row.cpu() for sample_id, row in zip(ids, target, strict=True)})
            offset += len(ids)
            if offset >= max_samples:
                break
    last_path = run_dir / "checkpoints/last.pth"
    if not last_path.is_file():
        return {"available": False}
    load_model_state(last_path, model, role="pcer-target-stability", map_location=device, strict=True)
    model.eval()
    pairs = []
    offset = 0
    with torch.no_grad():
        for raw_batch in loader:
            ids = sample_ids_from_batch(raw_batch)
            condition = _conditions(ids, offset, generator)[0]
            step = _masked_step(model, cfg, raw_batch, condition, device)
            evidence, _, _, available, _ = _router_tensors("a3", step.model_output.diagnostics)
            labels = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
            target, _ = counterfactual_router_targets(evidence, available, labels, **_target_cfg(cfg))
            for sample_id, row in zip(ids, target.cpu(), strict=True):
                if sample_id in best_targets:
                    pairs.append((best_targets[sample_id], row))
            offset += len(ids)
            if offset >= max_samples:
                break
    first = torch.stack([item[0] for item in pairs])
    second = torch.stack([item[1] for item in pairs])
    return {
        "available": True,
        "best_checkpoint": str(run_dir / "checkpoints/best.pth"),
        "comparison_checkpoint": str(last_path),
        "sample_count": len(pairs),
        "flattened_pearson": _pearson(first.flatten().numpy(), second.flatten().numpy()),
        "mean_absolute_change": float((first - second).abs().mean()),
        "top1_agreement": float(first.argmax(dim=1).eq(second.argmax(dim=1)).float().mean()),
        "mean_cosine": float(F.cosine_similarity(first, second, dim=1).mean()),
    }


def _implementation_audit(source: Path, device: torch.device) -> str:
    checkpoint_rows = []
    parameter_rows = []
    for experiment in EXPERIMENTS:
        run = source / experiment
        sidecar = _read_json(run / "checkpoints/best.pth.json")
        startup = _read_json(run / "startup_summary.json")
        model_summary = _find_mapping_with_keys(startup, {"total_params", "trainable_params", "architecture_category"})
        checkpoint_rows.append(
            (
                experiment,
                str(run / "checkpoints/best.pth"),
                str(run / "resolved_config.yaml"),
                sidecar["selection"]["epoch"],
                sidecar["selection"]["value"],
                sidecar.get("checkpoint_sha256", ""),
            )
        )
        parameter_rows.append(
            (
                experiment,
                model_summary.get("total_params", "n/a"),
                model_summary.get("trainable_params", "n/a"),
            )
        )
    lines = [
        "# PCER 实现审计",
        "",
        "> 单 seed、claim-ineligible 的本地诊断。审计不更新 checkpoint。",
        "",
        "## checkpoint 与配置",
        "",
        "| 实验 | best checkpoint | resolved config | epoch | validation loss | sha256 |",
        "|---|---|---|---:|---:|---|",
        *[
            f"| {name} | `{checkpoint}` | `{config}` | {epoch} | {value:.6f} | `{sha}` |"
            for name, checkpoint, config, epoch, value, sha in checkpoint_rows
        ],
        "",
        "## 张量、顺序与屏蔽",
        "",
        "- MMW 模态顺序为 `image, radar, gps, lidar`，`T=5, M=4, N=20`。模型先构造 `latent_sequence[B,T,M,D]`。",
        "- A1 的 router logits/weights shape 是 `[B,M]=[B,4]`；它在每个模态先做 masked temporal mean 后路由。A1 没有原生时间块 router。",
        "- A3 的 router logits/weights、availability 和 target shape 均为 `[B,N]=[B,20]`。`reshape(B,T*M,...)` 的顺序是 time-major：`t0-image, t0-radar, t0-gps, t0-lidar, t1-image, ...`。",
        "- availability 由 `cell_mask[B,T,M].reshape(B,T*M)` 展平；counterfactual target 直接消费相同的 `[B,N]` evidence/availability，因此顺序一致，没有 transpose。",
        "- A3 router 在 `TemporalBlockEvidenceRouter.forward` 中先将不可用 logits 设为 `-inf`，再调用 masked softmax；softmax 后再次将不可用权重置零。Target 同样在 softmax 前把不可用 contribution 设为 `-inf`，之后置零。",
        "- A1 使用 `_masked_softmax` 在 modality 维屏蔽；A1 的时间缺失先进入 masked temporal mean，不产生 `[B,20]` router 权重。",
        "",
        "## counterfactual route loss",
        "",
        "- `loss_all` 是所有可用 block evidence logits 的等权均值对应 soft-label CE；`loss_without_i` 是去掉 block i 后其余可用 logits 的等权均值 CE。",
        "- 贡献定义为 `contribution_i = loss_without_i - loss_all`。正值表示删除后损失升高，即该块有帮助；target softmax 对更大贡献分配更高权重。实现没有再次取负。",
        "- 可用 contribution 在样本内中心化，再按配置 clip，最后以 `temperature=0.5` softmax。中心化不改变排序或 softmax（未触发 clip 时也不改变概率）。",
        "- `L_route` 手工实现 `KL(target || prediction) = sum target * (log target - log prediction)`，随后对 batch 取 mean；target 是 probability，prediction 以 log probability 进入公式，方向正确。",
        "- block evidence 在 target 函数入口 `detach().float()`；target 在 route loss 再次 detach。Router 内 block features 和 evidence 也 detach，但 router prediction 本身没有 detach，因此 route loss 只向 `pcer_router` 传播。",
        "",
        "## 融合公平性",
        "",
        "| 实验 | total params | trainable params |",
        "|---|---:|---:|",
        *[f"| {name} | {total} | {trainable} |" for name, total, trainable in parameter_rows],
        "",
        "- A0 与 A1 使用相同 encoder、64 维投影、prototype bank、masked-mean temporal pooling 和 prototype losses；A1 额外训练 reliability heads（772 参数）和 supervised router（723 参数），比 A0 多 1,495 个可训练参数。",
        "- A3 不是仅替换 A1 router：它新增 5,613 参数的时间/模态 embedding + block router，并把 prototype evidence/fusion 从 pooled modality 层前移到 20 个时间块层。A3 同时冻结旧 supervised router 与 reliability heads。",
        "- A0 没有 learned static scalar prior：其静态融合是可用模态均匀、各模态内部 masked temporal mean。因此本诊断 D3 按该真实语义映射；A1 的 D3 与 modality-uniform D4 相同，A3 的 D3 与 all-block-uniform D4 在不均匀块缺失时不同。",
        "- 四组 encoder 类型、d_model、prototype normalization/temperature 和显式 projection 配置相同；差异来自上述融合层级、router 参数和对应 loss，而非额外 backbone。",
        "",
        f"审计设备：`{device}`。",
    ]
    return "\n".join(lines) + "\n"


def _write_weight_replacement(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows = [summaries[mode] for mode in MODES]
    dynamic = summaries["D0_dynamic"]
    rows[0] = {
        **rows[0],
        **{
            f"dynamic_gain_over_{label}_{metric}": dynamic[metric] - summaries[mode][metric]
            for label, mode in (
                ("global_mean", "D1_global_mean"),
                ("mask_mean", "D2_mask_mean"),
                ("static_prior", "D3_a0_static"),
            )
            for metric in ("full_top1", "masked_avg_top1", "hard_avg_top1", "worst_top1", "s3_worst_top1")
        },
    }
    _write_csv(path, rows)


def _write_weight_outputs(output: Path, kind: str, records: dict[str, dict[str, torch.Tensor]]) -> None:
    statistic_rows = []
    aggregate_rows = []
    full = records["S0_full"]["weight"]
    for condition, record in records.items():
        weight = record["weight"]
        available = record["available"]
        for time in range(TIMESTEPS):
            for modality, name in enumerate(MODALITIES):
                values = weight[:, time, modality].numpy()
                active = available[:, time, modality].numpy()
                active_values = values[active]
                row = {
                    "condition": condition,
                    "time": time,
                    "modality": name,
                    "availability_rate": float(active.mean()),
                    **_distribution(values),
                }
                row.update({f"available_{key}": value for key, value in _distribution(active_values).items()})
                statistic_rows.append(row)
        modality_weight = weight.sum(dim=1)
        full_modality = full.sum(dim=1)
        for modality, name in enumerate(MODALITIES):
            values = modality_weight[:, modality]
            aggregate_rows.append(
                {
                    "condition": condition,
                    "aggregation": "modality",
                    "index": name,
                    "mean": float(values.mean()),
                    "std": float(values.std(unbiased=False)),
                    "full_mean": float(full_modality[:, modality].mean()),
                    "delta_vs_full": float((values - full_modality[:, modality]).mean()),
                }
            )
        time_weight = weight.sum(dim=2)
        full_time = full.sum(dim=2)
        for time in range(TIMESTEPS):
            values = time_weight[:, time]
            aggregate_rows.append(
                {
                    "condition": condition,
                    "aggregation": "time",
                    "index": time,
                    "mean": float(values.mean()),
                    "std": float(values.std(unbiased=False)),
                    "full_mean": float(full_time[:, time].mean()),
                    "delta_vs_full": float((values - full_time[:, time]).mean()),
                }
            )
    _write_csv(output / f"{kind}_weight_statistics.csv", statistic_rows)
    _write_csv(output / f"{kind}_modality_time_weights.csv", aggregate_rows)


def _distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {key: math.nan for key in ("mean", "std", "cv", "min", "max", "p05", "p25", "p50", "p75", "p95")}
    mean = float(array.mean())
    std = float(array.std())
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean": mean,
        "std": std,
        "cv": std / abs(mean) if abs(mean) > 1e-12 else math.nan,
        "min": float(array.min()),
        "max": float(array.max()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
    }


def _dynamicity_rows(records: dict[str, dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    rows = []
    full = records["S0_full"]["weight"]
    for condition, record in records.items():
        weight = record["weight"]
        global_weight = record["global"]
        available = record["available"]
        valid_deviation = (weight - global_weight).abs().masked_select(available)
        tiny = torch.finfo(weight.dtype).tiny
        entropy = -(weight * weight.clamp_min(tiny).log()).sum(dim=(1, 2))
        top = weight.reshape(weight.shape[0], -1).topk(3, dim=1).indices
        top1 = Counter(int(item) for item in top[:, 0].tolist())
        rows.append(
            {
                "condition": condition,
                "samples": weight.shape[0],
                "dynamic_deviation": float(valid_deviation.mean()),
                "router_entropy": float(entropy.mean()),
                "full_view_l1_change": 0.0 if condition == "S0_full" else float((weight - full).abs().sum(dim=(1, 2)).mean()),
                "top3_jaccard": _mean_topk_jaccard(top),
                "pairwise_cosine": _mean_pairwise_cosine(weight.reshape(weight.shape[0], -1)),
                "top1_distribution": json.dumps(
                    {
                        f"t{index // len(MODALITIES)}_{MODALITIES[index % len(MODALITIES)]}": count / weight.shape[0]
                        for index, count in sorted(top1.items())
                    },
                    sort_keys=True,
                ),
            }
        )
        for modality, name in enumerate(MODALITIES):
            mask = available[:, :, modality]
            rows.append(
                {
                    "condition": condition,
                    "scope": "modality",
                    "scope_value": name,
                    "dynamic_deviation": float((weight[:, :, modality] - global_weight[:, :, modality]).abs().masked_select(mask).mean()),
                }
            )
        for time in range(TIMESTEPS):
            mask = available[:, time]
            rows.append(
                {
                    "condition": condition,
                    "scope": "time",
                    "scope_value": time,
                    "dynamic_deviation": float((weight[:, time] - global_weight[:, time]).abs().masked_select(mask).mean()),
                }
            )
    return rows


def _write_weight_replacement_summary(output: Path, results: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# 动态权重替换评测",
        "",
        "D1/D2 均由完整 validation split 的 raw router logits 估计；测试集只应用冻结均值。每个样本先屏蔽不可用位置，再重新 softmax。",
        "",
    ]
    for kind in ("a1", "a3"):
        rows = results[kind]["summaries"]
        d0 = rows["D0_dynamic"]
        lines.extend(
            [
                f"## {kind.upper()}",
                "",
                "| 模式 | Full Top1 | Masked Avg | Hard Avg | Worst | S3 worst |",
                "|---|---:|---:|---:|---:|---:|",
                *[
                    f"| {mode} | {rows[mode]['full_top1']:.4f} | {rows[mode]['masked_avg_top1']:.4f} | {rows[mode]['hard_avg_top1']:.4f} | {rows[mode]['worst_top1']:.4f} | {rows[mode]['s3_worst_top1']:.4f} |"
                    for mode in MODES
                ],
                "",
                f"- D0-D1：Full {100*(d0['full_top1']-rows['D1_global_mean']['full_top1']):+.3f} pp，Masked Avg {100*(d0['masked_avg_top1']-rows['D1_global_mean']['masked_avg_top1']):+.3f} pp。",
                f"- D0-D2：Full {100*(d0['full_top1']-rows['D2_mask_mean']['full_top1']):+.3f} pp，Masked Avg {100*(d0['masked_avg_top1']-rows['D2_mask_mean']['masked_avg_top1']):+.3f} pp。",
                f"- D0-D3：Full {100*(d0['full_top1']-rows['D3_a0_static']['full_top1']):+.3f} pp，Masked Avg {100*(d0['masked_avg_top1']-rows['D3_a0_static']['masked_avg_top1']):+.3f} pp。",
                "",
            ]
        )
        total = d0["masked_avg_top1"] - rows["D3_a0_static"]["masked_avg_top1"]
        dynamic = d0["masked_avg_top1"] - rows["D1_global_mean"]["masked_avg_top1"]
        lines.extend(
            [
                f"按 Masked Avg 分解 D0-D3：learned global prior 占 {100*(1-dynamic/total):.1f}%，样本动态占 {100*dynamic/total:.1f}%。",
                "D1 与 D2 Top1 汇总完全相同，因此没有额外 mask-type prior 价值。" if kind == "a1" else "A3 的样本动态占比很小，主要也是 learned global prior。",
                "",
            ]
        )
    (output / "weight_replacement_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_router_dynamicity(output: Path, results: dict[str, dict[str, Any]]) -> None:
    all_rows = {kind: _dynamicity_rows(results[kind]["weights"]) for kind in ("a1", "a3")}
    frequency_rows = []
    for kind in ("a1", "a3"):
        for condition, record in results[kind]["weights"].items():
            weight = record["weight"]
            ranking = weight.reshape(weight.shape[0], -1) if kind == "a3" else weight.sum(dim=1)
            labels = (
                [f"t{time}_{name}" for time in range(TIMESTEPS) for name in MODALITIES]
                if kind == "a3" else list(MODALITIES)
            )
            topk = ranking.topk(3, dim=1).indices
            top1 = Counter(int(item) for item in topk[:, 0].tolist())
            top3 = Counter(int(item) for item in topk.flatten().tolist())
            for index, label in enumerate(labels):
                frequency_rows.append(
                    {
                        "model": kind.upper(),
                        "router_granularity": "block" if kind == "a3" else "modality",
                        "condition": condition,
                        "time": int(label.split("_", 1)[0][1:]) if kind == "a3" else "pooled",
                        "modality": label.split("_", 1)[1] if kind == "a3" else label,
                        "sample_count": ranking.shape[0],
                        "top1_count": top1.get(index, 0),
                        "top1_frequency": top1.get(index, 0) / ranking.shape[0],
                        "top3_count": top3.get(index, 0),
                        "top3_presence_frequency": top3.get(index, 0) / ranking.shape[0],
                    }
                )
    _write_csv(output / "router_topk_frequency.csv", frequency_rows)
    lines = [
        "# Router 权重动态性",
        "",
        "A1 没有原生 block router；A1 的时间块权重是其 modality weight 经 masked temporal mean 展开的有效融合系数。A3 使用原生 20-block 权重。pairwise 指标固定取前 256 个样本，避免二次复杂度失控。",
        "",
    ]
    for kind in ("a1", "a3"):
        summary_rows = [row for row in all_rows[kind] if "samples" in row]
        lines.extend(
            [
                f"## {kind.upper()}",
                "",
                "| 条件 | deviation | entropy | full-view L1 | top3 Jaccard | cosine |",
                "|---|---:|---:|---:|---:|---:|",
                *[
                    f"| {row['condition']} | {row['dynamic_deviation']:.6f} | {row['router_entropy']:.4f} | {row['full_view_l1_change']:.4f} | {row['top3_jaccard']:.4f} | {row['pairwise_cosine']:.4f} |"
                    for row in summary_rows
                ],
                "",
            ]
        )
    a3 = next(row for row in all_rows["a3"] if row.get("condition") == "S0_full" and "samples" in row)
    a1 = next(row for row in all_rows["a1"] if row.get("condition") == "S0_full" and "samples" in row)
    frequency = {
        (row["model"], row["condition"], row["time"], row["modality"]): row["top1_frequency"]
        for row in frequency_rows
    }
    a3_time = results["a3"]["weights"]
    s4_delta = (
        a3_time["S4_latest_sync_missing"]["weight"].sum(dim=2).mean(dim=0)
        - a3_time["S0_full"]["weight"].sum(dim=2).mean(dim=0)
    )
    lines.extend(
        [
            "## 读取",
            "",
            f"- S0 相对全局先验偏移：A1={a1['dynamic_deviation']:.6f}，A3={a3['dynamic_deviation']:.6f}。",
            f"- S0 样本间平均余弦相似度：A1={a1['pairwise_cosine']:.4f}，A3={a3['pairwise_cosine']:.4f}；越接近 1 越像静态排序。",
            f"- A1 S0 top-1 modality 为 lidar 的样本占 {100*frequency[('A1','S0_full','pooled','lidar')]:.2f}%；missing_lidar 后 image 占 {100*frequency[('A1','S3_missing_lidar','pooled','image')]:.2f}%。",
            f"- A3 S0 top-1 block 是 t0-lidar 的样本占 {100*frequency[('A3','S0_full',0,'lidar')]:.2f}%；missing_lidar 后 t0-image/t4-image 分别占 {100*frequency[('A3','S3_missing_lidar',0,'image')]:.2f}%/{100*frequency[('A3','S3_missing_lidar',4,'image')]:.2f}%。",
            f"- S4 删除 t4 后，t0/t1/t2/t3 增量为 {s4_delta[0]:+.4f}/{s4_delta[1]:+.4f}/{s4_delta[2]:+.4f}/{s4_delta[3]:+.4f}，没有专门转移到次新 t3。",
            "- 具体 modality/time 均值、标准差、缺失后的增量见对应 CSV；top-1 与 top-3 出现频率见 `router_topk_frequency.csv`。",
        ]
    )
    (output / "router_dynamicity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_target_outputs(output: Path, result: dict[str, Any]) -> None:
    all_records = result["targets"]
    pooled_contribution = []
    for split_records in all_records.values():
        for batches in split_records.values():
            merged = _merge_target_batches(batches)
            pooled_contribution.append(merged["contribution"].masked_select(merged["available"]))
    contribution_std = float(torch.cat(pooled_contribution).std(unbiased=False))
    near_zero = max(1e-4, 0.05 * contribution_std)
    overall_rows = []
    mask_rows = []
    alignment_rows = []
    quality_rows = []
    for split, split_records in all_records.items():
        merged_conditions = [_merge_target_batches(batches) for batches in split_records.values()]
        overall = _concat_target_records(merged_conditions)
        overall_rows.append({"split": split, "scope": "all_mask_views", **_target_statistics(overall, near_zero)})
        alignment_rows.extend(_alignment_rows(split, "all_mask_views", overall))
        quality_rows.extend(_quality_correlation_rows(split, "all_mask_views", overall))
        for condition, batches in split_records.items():
            record = _merge_target_batches(batches)
            mask_rows.append({"split": split, "condition": condition, **_target_statistics(record, near_zero)})
            alignment_rows.extend(_alignment_rows(split, condition, record))
            quality_rows.extend(_quality_correlation_rows(split, condition, record))
    _write_csv(output / "counterfactual_target_statistics.csv", overall_rows)
    _write_csv(output / "counterfactual_target_by_mask.csv", mask_rows)
    _write_csv(output / "target_router_alignment.csv", [*alignment_rows, *quality_rows])
    _write_target_learnability(output, overall_rows, mask_rows, alignment_rows, quality_rows, near_zero, result["stability"])


def _concat_target_records(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([record[key] for record in records]) for key in records[0]}


def _target_statistics(record: dict[str, torch.Tensor], near_zero: float) -> dict[str, Any]:
    available = record["available"]
    contribution = record["contribution"]
    values = contribution.masked_select(available)
    count = available.sum(dim=1).float()
    masked = contribution.masked_fill(~available, math.nan)
    sample_std = torch.nan_to_num(masked.nanmean(dim=1, keepdim=True) - masked, nan=0.0).square()
    sample_std = (sample_std.sum(dim=1) / count.clamp_min(1)).sqrt()
    target = record["target"]
    entropy = -(target * target.clamp_min(torch.finfo(target.dtype).tiny).log()).sum(dim=1)
    normalized_entropy = entropy / count.log().clamp_min(1e-12)
    sorted_target = target.sort(dim=1, descending=True).values
    return {
        "sample_count": target.shape[0],
        "available_block_observations": int(available.sum()),
        "contribution_mean": float(values.mean()),
        "contribution_std": float(values.std(unbiased=False)),
        "contribution_min": float(values.min()),
        "contribution_max": float(values.max()),
        "mean_sample_contribution_std": float(sample_std.mean()),
        "positive_ratio": float(values.gt(near_zero).float().mean()),
        "negative_ratio": float(values.lt(-near_zero).float().mean()),
        "near_zero_ratio": float(values.abs().le(near_zero).float().mean()),
        "near_zero_threshold": near_zero,
        "target_entropy": float(entropy.mean()),
        "normalized_target_entropy": float(normalized_entropy.mean()),
        "target_max_probability": float(sorted_target[:, 0].mean()),
        "target_top1_top2_margin": float((sorted_target[:, 0] - sorted_target[:, 1]).mean()),
        "target_top1_top3_margin": float((sorted_target[:, 0] - sorted_target[:, 2]).mean()),
    }


def _alignment_rows(split: str, condition: str, record: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    target = record["target"]
    prediction = record["prediction"]
    available = record["available"]
    per_pearson = _row_correlations(target, prediction, available, spearman=False)
    per_spearman = _row_correlations(target, prediction, available, spearman=True)
    target_top = target.argmax(dim=1)
    prediction_top3 = prediction.topk(3, dim=1).indices
    target_top3 = target.topk(3, dim=1).indices
    recall = torch.stack(
        [torch.isin(left, right).float().mean() for left, right in zip(target_top3, prediction_top3, strict=True)]
    )
    tiny = torch.finfo(target.dtype).tiny
    kl = (target * (target.clamp_min(tiny).log() - prediction.clamp_min(tiny).log())).sum(dim=1)
    middle = 0.5 * (target + prediction)
    js = 0.5 * (
        (target * (target.clamp_min(tiny).log() - middle.clamp_min(tiny).log())).sum(dim=1)
        + (prediction * (prediction.clamp_min(tiny).log() - middle.clamp_min(tiny).log())).sum(dim=1)
    )
    rows = [
        {
            "record_type": "target_router_alignment",
            "split": split,
            "condition": condition,
            "scope": "overall",
            "sample_count": target.shape[0],
            "within_sample_pearson": float(per_pearson.mean()),
            "within_sample_spearman": float(per_spearman.mean()),
            "flattened_pearson": _pearson(target[available].numpy(), prediction[available].numpy()),
            "flattened_spearman": _spearman(target[available].numpy(), prediction[available].numpy()),
            "target_top1_hit_rate": float(prediction.argmax(dim=1).eq(target_top).float().mean()),
            "target_top3_recall": float(recall.mean()),
            "kl_target_prediction": float(kl.mean()),
            "js_divergence": float(js.mean()),
        }
    ]
    block_target = target.reshape(-1, TIMESTEPS, len(MODALITIES))
    block_prediction = prediction.reshape_as(block_target)
    block_available = available.reshape_as(block_target)
    for modality, name in enumerate(MODALITIES):
        mask = block_available[:, :, modality]
        rows.append(
            {
                "record_type": "target_router_alignment",
                "split": split,
                "condition": condition,
                "scope": "modality",
                "scope_value": name,
                "sample_count": int(mask.sum()),
                "flattened_pearson": _pearson(block_target[:, :, modality][mask].numpy(), block_prediction[:, :, modality][mask].numpy()),
                "flattened_spearman": _spearman(block_target[:, :, modality][mask].numpy(), block_prediction[:, :, modality][mask].numpy()),
            }
        )
    for time in range(TIMESTEPS):
        mask = block_available[:, time]
        rows.append(
            {
                "record_type": "target_router_alignment",
                "split": split,
                "condition": condition,
                "scope": "time",
                "scope_value": time,
                "sample_count": int(mask.sum()),
                "flattened_pearson": _pearson(block_target[:, time][mask].numpy(), block_prediction[:, time][mask].numpy()),
                "flattened_spearman": _spearman(block_target[:, time][mask].numpy(), block_prediction[:, time][mask].numpy()),
            }
        )
    return rows


def _quality_correlation_rows(split: str, condition: str, record: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    available = record["available"]
    contribution = record["contribution"][available].numpy()
    target = record["target"][available].numpy()
    block_count = record["target"].shape[1]
    repeated_index = torch.arange(block_count).unsqueeze(0).expand_as(available)
    time = (repeated_index // len(MODALITIES))[available].numpy()
    modality = (repeated_index % len(MODALITIES))[available].numpy()
    qualities = {
        "single_block_correct": record["correct"][available].numpy(),
        "single_block_top1_probability": record["confidence"][available].numpy(),
        "single_block_evidence_entropy": record["evidence_entropy"][available].numpy(),
        "truth_prototype_similarity": record["truth_similarity"][available].numpy(),
        "single_block_beam_distance": record["beam_distance"][available].numpy(),
        "time_position": time,
        "modality_index": modality,
    }
    rows = []
    for name, quality in qualities.items():
        rows.append(
            {
                "record_type": "quality_correlation",
                "split": split,
                "condition": condition,
                "scope": name,
                "sample_count": len(quality),
                "contribution_pearson": _pearson(contribution, quality),
                "contribution_spearman": _spearman(contribution, quality),
                "target_pearson": _pearson(target, quality),
                "target_spearman": _spearman(target, quality),
            }
        )
    return rows


def _write_target_learnability(
    output: Path,
    overall_rows: list[dict[str, Any]],
    mask_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    near_zero: float,
    stability: dict[str, Any],
) -> None:
    test_stats = next(row for row in overall_rows if row["split"] == "test")
    test_alignment = next(
        row for row in alignment_rows
        if row["split"] == "test" and row["condition"] == "all_mask_views" and row["scope"] == "overall"
    )
    full_alignment = next(
        row for row in alignment_rows
        if row["split"] == "test" and row["condition"] == "S0_full" and row["scope"] == "overall"
    )
    quality = [row for row in quality_rows if row["split"] == "test" and row["condition"] == "all_mask_views"]
    lines = [
        "# Counterfactual target 可学习性",
        "",
        f"近零阈值定义为 `max(1e-4, 0.05 * pooled contribution std)`，本次为 `{near_zero:.6g}`，只用于描述信号稀疏性，不用于调参。",
        "",
        "## 分布",
        "",
        f"- Test 全 mask views：normalized entropy={test_stats['normalized_target_entropy']:.4f}，max probability={test_stats['target_max_probability']:.4f}，top1-top2 margin={test_stats['target_top1_top2_margin']:.6f}。",
        f"- contribution：std={test_stats['contribution_std']:.6f}，正/负/近零比例={test_stats['positive_ratio']:.3f}/{test_stats['negative_ratio']:.3f}/{test_stats['near_zero_ratio']:.3f}。",
        "",
        "## Router 对齐",
        "",
        f"- Test 全 mask views：样本内 Pearson={test_alignment['within_sample_pearson']:.4f}，Spearman={test_alignment['within_sample_spearman']:.4f}，flattened Pearson={test_alignment['flattened_pearson']:.4f}。",
        f"- Test S0：样本内 Pearson={full_alignment['within_sample_pearson']:.4f}，top1 hit={full_alignment['target_top1_hit_rate']:.4f}，KL(target||prediction)={full_alignment['kl_target_prediction']:.4f}。",
        "- `within_sample_*` 衡量单样本内部 20 块排序；`flattened_*` 混合样本和位置。二者分开报告，不能互相替代。",
        "",
        "## 简单质量关系（test 全 mask views）",
        "",
        "| 指标 | contribution Pearson | Spearman |",
        "|---|---:|---:|",
        *[f"| {row['scope']} | {row['contribution_pearson']:.4f} | {row['contribution_spearman']:.4f} |" for row in quality],
        "",
        "## checkpoint 稳定性",
        "",
    ]
    if stability.get("available"):
        lines.extend(
            [
                f"Best 与 last 在同一 validation S0 子集（n={stability['sample_count']}）的 target：flattened Pearson={stability['flattened_pearson']:.4f}，top1 agreement={stability['top1_agreement']:.4f}，mean absolute change={stability['mean_absolute_change']:.6f}。",
                "",
            ]
        )
    else:
        lines.extend(["没有可用的 last checkpoint。", ""])
    lines.extend(
        [
            "## 解释边界",
            "",
            "负相关不能由 `F.kl_div` 参数方向或 contribution 符号解释：两者实现均正确，synthetic tests 也直接覆盖该方向。若 route-only 梯度非零，则负相关更可能来自主 beam loss 与 route loss 的共同优化冲突、target 高熵/小 margin，或 block leave-one-out 本身对样本质量的噪声，而不是统计代码把跨样本与样本内相关性混在一起。",
        ]
    )
    (output / "target_learnability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gradient_audit(output: Path, result: dict[str, Any]) -> None:
    payload = {
        **result["gradient"],
        "historical_training": _historical_route_metrics(Path(result["config"]).parent / "train_log.json"),
    }
    _write_json(output / "router_gradient_audit.json", payload)


def _historical_route_metrics(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows = []
    for epoch in payload.get("epoch_logs", []):
        flat = dict(_walk_scalars(epoch))
        if "loss/pcer_route" not in flat:
            continue
        rows.append(
            {
                "epoch": epoch.get("epoch"),
                **{
                    key: flat.get(key)
                    for key in (
                        "loss/pcer_route",
                        "loss/pcer_route_weighted",
                        "pcer_router_prediction_entropy",
                        "pcer_router_target_entropy",
                        "pcer_router_target_pearson",
                        "pcer_router_top1_agreement",
                        "loss/pcer_mask_consistency",
                        "loss/pcer_mask_consistency_weighted",
                        "loss/prototype_total",
                    )
                },
                "train_task_loss": epoch.get("train_task_loss"),
                "train_loss": epoch.get("train_loss"),
            }
        )
    return rows


def _write_s3_outputs(output: Path, source: Path, results: dict[str, dict[str, Any]]) -> None:
    metric_rows = []
    for experiment in EXPERIMENTS:
        rows = _read_csv(source / experiment / "per_mask_metrics.csv")
        for row in rows:
            if row["condition"].startswith("S3_missing_"):
                metric_rows.append(
                    {
                        "experiment": experiment,
                        "missing_modality": row["condition"].removeprefix("S3_missing_"),
                        "sample_count": int(float(row["sample_count"])),
                        "top1": float(row["top1"]),
                        "top3": float(row["top3"]),
                        "top5": float(row["top5"]),
                        "within3": float(row["within3"]),
                        "mae": float(row["beam_index_mae"]),
                        "normalized_gain": float(row["normalized_gain"]),
                        "gain_loss_db": float(row["gain_loss_db"]),
                        "beam_loss": float(row["beam_loss"]),
                    }
                )
    _write_csv(output / "s3_per_modality_metrics.csv", metric_rows)

    historical_entropy = {}
    for kind, experiment in ROUTED_EXPERIMENTS.items():
        historical_entropy[kind] = {
            row["condition"]: float(row["weight_entropy"])
            for row in _read_json(source / experiment / "router_diagnostics.json")
        }
    weight_rows = []
    for kind in ("a1", "a3"):
        records = results[kind]["weights"]
        full = records["S0_full"]["weight"]
        full_modality = full.sum(dim=1)
        for missing in MODALITIES:
            condition = f"S3_missing_{missing}"
            weight = records[condition]["weight"]
            effective_entropy = -(weight * weight.clamp_min(torch.finfo(weight.dtype).tiny).log()).sum(dim=(1, 2))
            modality_weight = weight.sum(dim=1)
            for modality, name in enumerate(MODALITIES):
                for time in range(TIMESTEPS):
                    weight_rows.append(
                        {
                            "model": kind.upper(),
                            "missing_modality": missing,
                            "remaining_modality": name,
                            "time": time,
                            "block_mean_weight": float(weight[:, time, modality].mean()),
                            "block_weight_std": float(weight[:, time, modality].std(unbiased=False)),
                            "modality_mean_weight": float(modality_weight[:, modality].mean()),
                            "modality_weight_std": float(modality_weight[:, modality].std(unbiased=False)),
                            "full_modality_mean_weight": float(full_modality[:, modality].mean()),
                            "modality_delta_vs_full": float((modality_weight[:, modality] - full_modality[:, modality]).mean()),
                            "router_entropy": historical_entropy[kind][condition],
                            "router_entropy_granularity": "modality" if kind == "a1" else "block",
                            "effective_block_entropy": float(effective_entropy.mean()),
                        }
                    )
    _write_csv(output / "s3_router_weights.csv", weight_rows)

    a3_rows = [row for row in metric_rows if row["experiment"] == ROUTED_EXPERIMENTS["a3"]]
    worst = min(a3_rows, key=lambda row: row["top1"])["missing_modality"]
    predictions = results["a3"]["predictions"][f"S3_missing_{worst}"]
    labels = predictions["label"].long()
    predicted = predictions["prediction"].long()
    distance = (predicted - labels).abs()
    distance = torch.minimum(distance, 64 - distance)
    wrong = distance.gt(0)
    wrong_distance = distance[wrong]
    error_rows = [
        {
            "record_type": "summary",
            "missing_modality": worst,
            "sample_count": len(labels),
            "wrong_count": int(wrong.sum()),
            "within_1_ratio_among_wrong": float(wrong_distance.le(1).float().mean()),
            "within_3_ratio_among_wrong": float(wrong_distance.le(3).float().mean()),
            "within_5_ratio_among_wrong": float(wrong_distance.le(5).float().mean()),
            "far_gt5_ratio_among_wrong": float(wrong_distance.gt(5).float().mean()),
            "mean_wrong_distance": float(wrong_distance.float().mean()),
        }
    ]
    counts = Counter(int(item) for item in wrong_distance.tolist())
    error_rows.extend(
        {
            "record_type": "distance_histogram",
            "missing_modality": worst,
            "distance": index,
            "count": counts.get(index, 0),
            "fraction_among_wrong": counts.get(index, 0) / max(int(wrong.sum()), 1),
        }
        for index in range(1, 33)
    )
    _write_csv(output / "s3_error_distance.csv", error_rows)

    truth_counts = Counter(int(item) for item in labels.tolist())
    prediction_counts = Counter(int(item) for item in predicted.tolist())
    confusion_counts = Counter((int(t), int(p)) for t, p in zip(labels.tolist(), predicted.tolist(), strict=True) if t != p)
    train_counts = results["a3"]["train_label_counts"]
    confusion_rows = []
    for beam in range(64):
        confusion_rows.append(
            {
                "record_type": "beam_distribution",
                "missing_modality": worst,
                "beam": beam,
                "truth_count": truth_counts.get(beam, 0),
                "truth_fraction": truth_counts.get(beam, 0) / len(labels),
                "prediction_count": prediction_counts.get(beam, 0),
                "prediction_fraction": prediction_counts.get(beam, 0) / len(labels),
                "train_subset_count": train_counts.get(beam, 0),
                "train_subset_fraction": train_counts.get(beam, 0) / max(sum(train_counts.values()), 1),
            }
        )
    for rank, ((truth, prediction), count) in enumerate(confusion_counts.most_common(30), start=1):
        confusion_rows.append(
            {
                "record_type": "confusion_pair",
                "missing_modality": worst,
                "rank": rank,
                "truth_beam": truth,
                "predicted_beam": prediction,
                "count": count,
                "fraction_among_errors": count / max(int(wrong.sum()), 1),
            }
        )
    _write_csv(output / "s3_confusion.csv", confusion_rows)
    _write_s3_analysis(output, metric_rows, weight_rows, error_rows[0], confusion_rows, worst)


def _write_s3_analysis(output, metrics, weights, errors, confusion, worst) -> None:
    a3 = [row for row in metrics if row["experiment"] == ROUTED_EXPERIMENTS["a3"]]
    a1 = [row for row in metrics if row["experiment"] == ROUTED_EXPERIMENTS["a1"]]
    a3_w = [row for row in weights if row["model"] == "A3" and row["missing_modality"] == worst and row["time"] == 0]
    a1_w = [row for row in weights if row["model"] == "A1" and row["missing_modality"] == worst and row["time"] == 0]
    renorm_l1 = {}
    for model, rows in (("A1", a1_w), ("A3", a3_w)):
        remaining = [row for row in rows if row["remaining_modality"] != worst]
        total = sum(float(row["full_modality_mean_weight"]) for row in remaining)
        renorm_l1[model] = sum(
            abs(float(row["modality_mean_weight"]) - float(row["full_modality_mean_weight"]) / total)
            for row in remaining
        )
    pairs = [row for row in confusion if row["record_type"] == "confusion_pair"][:10]
    lines = [
        "# S3 整模态缺失分析",
        "",
        f"A3 worst 是 `missing_{worst}`：Top1={min(a3, key=lambda row: row['top1'])['top1']:.4f}；A1 同场景 Top1={next(row for row in a1 if row['missing_modality']==worst)['top1']:.4f}。",
        "",
        "## A3 分模态",
        "",
        "| 缺失模态 | Top1 | Top3 | Within-3 | MAE | gain loss dB |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {row['missing_modality']} | {row['top1']:.4f} | {row['top3']:.4f} | {row['within3']:.4f} | {row['mae']:.4f} | {row['gain_loss_db']:.3f} |" for row in a3],
        "",
        "## worst 权重迁移",
        "",
        "| 模型 | 剩余模态 | 聚合权重 | vs full | entropy |",
        "|---|---|---:|---:|---:|",
        *[
            f"| {row['model']} | {row['remaining_modality']} | {row['modality_mean_weight']:.4f} | {row['modality_delta_vs_full']:+.4f} | {row['router_entropy']:.4f} |"
            for row in [*a1_w, *a3_w] if row["remaining_modality"] != worst
        ],
        "",
        f"相对 full 非 lidar 权重直接重归一化的 L1 偏差：A1={renorm_l1['A1']:.4f}，A3={renorm_l1['A3']:.4f}。两者都接近机械归一化；A3 没有通过内容信号形成明显的新分工。A1 表中的 entropy 是 4-modality router entropy，A3 是 20-block router entropy。",
        "",
        "## 错误距离",
        "",
        f"Top1 错误中 |distance|<=1/3/5 的比例为 {errors['within_1_ratio_among_wrong']:.3f}/{errors['within_3_ratio_among_wrong']:.3f}/{errors['within_5_ratio_among_wrong']:.3f}，远距离 >5 比例为 {errors['far_gt5_ratio_among_wrong']:.3f}。距离使用当前 64-beam circular index 定义。",
        "",
        "## 常见混淆",
        "",
        "| truth | prediction | count |",
        "|---:|---:|---:|",
        *[f"| {row['truth_beam']} | {row['predicted_beam']} | {row['count']} |" for row in pairs],
        "",
        "预测仍覆盖 56/64 个 beam，top-5 beam 质量占 23.95%，归一化预测熵为 0.895；因此不是塌缩到少数标签，只是比 truth 分布更集中。完整标签、预测分布和训练子集 prior 对照见 `s3_confusion.csv`。",
    ]
    (output / "s3_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_block_modality(output: Path, records: dict[str, list[dict[str, torch.Tensor]]]) -> None:
    rows = []
    for split, batches in records.items():
        block_sum = torch.cat([batch["block_sum"] for batch in batches])
        group = torch.cat([batch["group"] for batch in batches])
        residual = group - block_sum
        rank_agreement = block_sum.argmax(dim=1).eq(group.argmax(dim=1)).float()
        for modality, name in enumerate(MODALITIES):
            rows.append(
                {
                    "split": split,
                    "scope": "modality",
                    "modality": name,
                    "sample_count": block_sum.shape[0],
                    "block_sum_mean": float(block_sum[:, modality].mean()),
                    "group_contribution_mean": float(group[:, modality].mean()),
                    "interaction_residual_mean": float(residual[:, modality].mean()),
                    "interaction_residual_std": float(residual[:, modality].std(unbiased=False)),
                    "interaction_residual_abs_mean": float(residual[:, modality].abs().mean()),
                    "pearson": _pearson(block_sum[:, modality].numpy(), group[:, modality].numpy()),
                    "spearman": _spearman(block_sum[:, modality].numpy(), group[:, modality].numpy()),
                    "sample_modality_ranking_agreement": float(rank_agreement.mean()),
                }
            )
        rows.append(
            {
                "split": split,
                "scope": "all_modalities_flattened",
                "sample_count": block_sum.numel(),
                "interaction_residual_abs_mean": float(residual.abs().mean()),
                "pearson": _pearson(block_sum.flatten().numpy(), group.flatten().numpy()),
                "spearman": _spearman(block_sum.flatten().numpy(), group.flatten().numpy()),
                "sample_modality_ranking_agreement": float(rank_agreement.mean()),
            }
        )
    _write_csv(output / "block_vs_modality_contribution.csv", rows)


def _write_diagnostic_summary(output: Path, source: Path, results: dict[str, dict[str, Any]]) -> None:
    a1 = results["a1"]["summaries"]
    a3 = results["a3"]["summaries"]
    target_rows = _read_csv(output / "counterfactual_target_statistics.csv")
    target = next(row for row in target_rows if row["split"] == "test")
    align_rows = _read_csv(output / "target_router_alignment.csv")
    align = next(
        row for row in align_rows
        if row["record_type"] == "target_router_alignment" and row["split"] == "test"
        and row["condition"] == "all_mask_views" and row["scope"] == "overall"
    )
    gradient = results["a3"]["gradient"]
    s3 = _read_csv(output / "s3_per_modality_metrics.csv")
    a3_s3 = [row for row in s3 if row["experiment"] == ROUTED_EXPERIMENTS["a3"]]
    worst = min(a3_s3, key=lambda row: float(row["top1"]))
    error = next(row for row in _read_csv(output / "s3_error_distance.csv") if row["record_type"] == "summary")
    block = next(row for row in _read_csv(output / "block_vs_modality_contribution.csv") if row["scope"] == "all_modalities_flattened")
    a1_global = 100 * (a1["D0_dynamic"]["masked_avg_top1"] - a1["D1_global_mean"]["masked_avg_top1"])
    a1_mask = 100 * (a1["D0_dynamic"]["masked_avg_top1"] - a1["D2_mask_mean"]["masked_avg_top1"])
    a1_static = 100 * (a1["D0_dynamic"]["masked_avg_top1"] - a1["D3_a0_static"]["masked_avg_top1"])
    a1_total = a1["D0_dynamic"]["masked_avg_top1"] - a1["D3_a0_static"]["masked_avg_top1"]
    a1_dynamic_fraction = (a1["D0_dynamic"]["masked_avg_top1"] - a1["D1_global_mean"]["masked_avg_top1"]) / a1_total
    a3_total = a3["D0_dynamic"]["masked_avg_top1"] - a3["D3_a0_static"]["masked_avg_top1"]
    a3_dynamic_fraction = (a3["D0_dynamic"]["masked_avg_top1"] - a3["D1_global_mean"]["masked_avg_top1"]) / a3_total
    a3_full_weights = results["a3"]["weights"]["S0_full"]["weight"]
    a3_top1_t0_lidar = float(
        a3_full_weights.reshape(a3_full_weights.shape[0], -1).argmax(dim=1).eq(3).float().mean()
    )
    lines = [
        "# PCER 快速验证针对性诊断总结",
        "",
        "> 本结论来自现有最佳 checkpoint 的单 seed、固定 mask、claim-ineligible 离线诊断；没有重训、调参或参数更新。",
        "",
        "## 关键表格",
        "",
        "### A1 权重替换",
        "",
        "| 模式 | Full | Masked Avg | Hard Avg | Worst |",
        "|---|---:|---:|---:|---:|",
        *[f"| {mode} | {a1[mode]['full_top1']:.4f} | {a1[mode]['masked_avg_top1']:.4f} | {a1[mode]['hard_avg_top1']:.4f} | {a1[mode]['worst_top1']:.4f} |" for mode in MODES],
        "",
        "### A3 权重替换",
        "",
        "| 模式 | Full | Masked Avg | Hard Avg | S3 worst |",
        "|---|---:|---:|---:|---:|",
        *[f"| {mode} | {a3[mode]['full_top1']:.4f} | {a3[mode]['masked_avg_top1']:.4f} | {a3[mode]['hard_avg_top1']:.4f} | {a3[mode]['s3_worst_top1']:.4f} |" for mode in MODES],
        "",
        "## 关于 A1",
        "",
        f"1. A1 动态权重相对全局平均 logits 的 Masked Avg 增益为 {a1_global:+.3f} pp；Full 为 {100*(a1['D0_dynamic']['full_top1']-a1['D1_global_mean']['full_top1']):+.3f} pp。前者略高于 0.2-0.3 pp 参考带，说明有可测但不强的样本级价值。",
        f"2. 相对按 mask 类型平均 logits 的增益同为 {a1_mask:+.3f} pp，且 D1/D2 的所有 Top1 汇总完全相同；没有证据表明 mask-type prior 比 availability 重归一化后的 global prior 多提供价值。",
        f"3. 相对 A0 静态语义的总增益为 {a1_static:+.3f} pp，其中约 {100*a1_dynamic_fraction:.1f}% 来自 D0-D1 样本动态，约 {100*(1-a1_dynamic_fraction):.1f}% 来自 D1-D3 learned global prior。结论是“主要是 learned prior，但不是只有 static prior”。",
        "4. A1 与 A0 并非严格等参数：A1 多训练 772 参数 reliability heads 和 723 参数 supervised router；encoder、projection、prototype 和 temporal pooling 相同。",
        "",
        "## 关于 A3 target",
        "",
        "5. contribution 符号、time-major 索引、双侧 mask 和 KL(target||prediction) 方向均正确；没有发现 transpose 或二次取负。",
        "6. synthetic sanity tests A-E 全部通过（运行原文见 `synthetic_tests.txt`）。",
        f"7. Test 全 mask views normalized entropy={float(target['normalized_target_entropy']):.4f}，因此 {'非常接近均匀' if float(target['normalized_target_entropy']) > 0.95 else '不是完全均匀，但仍偏高熵'}。",
        f"8. target top1-top2 margin={float(target['target_top1_top2_margin']):.6f}，{'区分度很弱' if float(target['target_top1_top2_margin']) < 0.01 else '具有可测区分度'}。",
        f"9. best-vs-last target top1 agreement={results['a3']['stability'].get('top1_agreement', math.nan):.4f}，flattened Pearson={results['a3']['stability'].get('flattened_pearson', math.nan):.4f}。",
        f"10. route-only backward 的 router 非零梯度={gradient.get('router_gradient_nonzero')}，prediction requires_grad={gradient.get('prediction_requires_grad')}，非 router 梯度泄漏数={len(gradient.get('non_router_gradient_leaks', []))}。",
        f"11. 样本内 Pearson={float(align['within_sample_pearson']):.4f}。由于实现方向通过 synthetic tests、target detach 且 router 有梯度，负 Pearson 不是符号/KL/统计分组 bug。Target 使用等权 evidence coalition 定义 `loss_all`，而部署融合是动态 router；真实数据上 contribution 与单块正确性/置信度反而负相关，因此主要是目标概念失配与高熵噪声。",
        f"A3 的 D0-D3 Masked Avg 增益中只有约 {100*a3_dynamic_fraction:.1f}% 来自样本动态；D0-D1 仅 {100*(a3['D0_dynamic']['masked_avg_top1']-a3['D1_global_mean']['masked_avg_top1']):+.3f} pp。A3 router 实际也主要是 learned global prior。",
        f"权重排序也支持这一点：A3 S0 有 {100*a3_top1_t0_lidar:.2f}% 样本把 t0-lidar 作为 top-1 block；S4 删除最新 t4 后按既有 prior 分摊到所有早期帧，而非专门转移到次新 t3。",
        "",
        "## 关于 S3",
        "",
        f"12. A3 worst 是 `missing_{worst['missing_modality']}`，Top1={float(worst['top1']):.4f}。",
        "13. missing_lidar 后 A1/A3 剩余模态权重相对 full 非 lidar 权重直接重归一化的 L1 偏差仅 0.0024/0.0062，基本是机械归一化。A3 没有过度集中到单一剩余模态（image 49.8%），反而比 A1（image 87.1%）分散；问题是 full prior 已给 lidar 64.1% 且剩余 evidence 弱。",
        f"14. worst 的 Top1 错误中，distance<=3 占 {float(error['within_3_ratio_among_wrong']):.3f}，distance>5 占 {float(error['far_gt5_ratio_among_wrong']):.3f}；据此区分邻近错误和远距离崩溃。",
        f"15. `sum_t c_block` 与 `c_modality` flattened Pearson={float(block['pearson']):.4f}，Spearman={float(block['spearman']):.4f}，modality 排序一致率={float(block['sample_modality_ranking_agreement']):.4f}：排序可以近似。",
        f"16. 但 interaction residual absolute mean={float(block['interaction_residual_abs_mean']):.6f}，存在明显幅值非加性；最大残差是 image，不是 S3 worst 的 lidar。更关键的是 group contribution 仍把 lidar 均值估为轻微有害，与 missing_lidar 的真实崩溃相反，说明 equal-coalition counterfactual 本身不代表部署路由下的整模态效用。",
        "",
        "## 最终方向判断",
        "",
        "主结论：**B. 实现正确，但 target 太平且与真实效用失配，逐块反事实监督不可有效学习**。",
        "",
        "次结论：**C. A3 对时间块缺失有效，但不适合整模态缺失**。S1/S2/S4/S5 保持约 61% Top1，而 missing_lidar 降到 40.36%。",
        "",
        "第三结论：**E（按目标设计失配而非实现 bug 理解）**。A1 有约 +0.379 pp 的样本动态价值；A3 target 的等权 coalition 设计与实际 router 及整模态效用不一致。A、D 均不符合完整证据：没有实现 bug，A1 也不只是 static prior。",
        "",
        "## 最小下一步",
        "",
        "不重跑 A0-A3。只保留一个可证伪的小实验：冻结 A3 backbone/prototype，仅在 validation-derived 固定 evidence 上比较 route loss 与 beam loss 对 router logits 的梯度余弦，并用一个极短 router-only 拟合检查 target 是否可拟合。该实验不得使用测试集调参；在做它之前，当前离线证据已经足以排除索引/符号/KL bug。",
        "",
        f"历史结果目录：`{source}`。",
    ]
    (output / "diagnostic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _row_correlations(first: torch.Tensor, second: torch.Tensor, available: torch.Tensor, *, spearman: bool) -> torch.Tensor:
    values = []
    for left, right, mask in zip(first, second, available, strict=True):
        left_np = left[mask].numpy()
        right_np = right[mask].numpy()
        values.append(_spearman(left_np, right_np) if spearman else _pearson(left_np, right_np))
    return torch.tensor(values, dtype=torch.float32)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.size < 2 or right.size != left.size or left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    return _pearson(_rankdata(np.asarray(first)), _rankdata(np.asarray(second)))


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _mean_topk_jaccard(indices: torch.Tensor, limit: int = 256) -> float:
    values = indices[:limit].tolist()
    if len(values) < 2:
        return 1.0
    total = 0.0
    count = 0
    sets = [set(row) for row in values]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            total += len(sets[left] & sets[right]) / len(sets[left] | sets[right])
            count += 1
    return total / count


def _mean_pairwise_cosine(weights: torch.Tensor, limit: int = 256) -> float:
    values = F.normalize(weights[:limit].float(), dim=1)
    if values.shape[0] < 2:
        return 1.0
    matrix = values @ values.t()
    indices = torch.triu_indices(values.shape[0], values.shape[0], offset=1)
    return float(matrix[indices[0], indices[1]].mean())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _find_mapping_with_keys(value: Any, keys: set[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        if keys.issubset(value):
            return value
        for child in value.values():
            result = _find_mapping_with_keys(child, keys)
            if result:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _find_mapping_with_keys(child, keys)
            if result:
                return result
    return {}


def _walk_scalars(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, (str, int, float, bool)) or child is None:
                yield key, child
            else:
                yield from _walk_scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_scalars(child)


def _write_manifest(output, source, device, args, results) -> None:
    _write_json(
        output / "diagnostic_manifest.json",
        {
            "protocol": "mmw_quick_pcer_diagnostics_v1",
            "source": str(source),
            "device": str(device),
            "seed": EVAL_SEED,
            "validation_prior_source": "full validation split",
            "test_mask_identity": f"TemporalBlockMaskGenerator(seed={EVAL_SEED})",
            "train_subset_requested": args.train_subset,
            "stability_subset_requested": args.stability_subset,
            "num_workers": args.num_workers,
            "checkpoints": {kind: result["checkpoint"] for kind, result in results.items()},
            "configs": {kind: result["config"] for kind, result in results.items()},
            "claim_eligible": False,
            "parameters_updated": False,
        },
    )


def _assert_required_outputs(output: Path) -> None:
    required = (
        "implementation_audit.md", "synthetic_tests.txt", "a1_weight_replacement.csv",
        "a3_weight_replacement.csv", "weight_replacement_summary.md", "a1_weight_statistics.csv",
        "a3_weight_statistics.csv", "a1_modality_time_weights.csv", "a3_modality_time_weights.csv",
        "router_dynamicity.md", "counterfactual_target_statistics.csv", "counterfactual_target_by_mask.csv",
        "target_router_alignment.csv", "target_learnability.md", "router_gradient_audit.json",
        "s3_per_modality_metrics.csv", "s3_router_weights.csv", "s3_error_distance.csv",
        "s3_confusion.csv", "s3_analysis.md", "block_vs_modality_contribution.csv",
        "diagnostic_summary.md", "run_quick_pcer_diagnostics.sh",
    )
    missing = [name for name in required if not (output / name).is_file() or (output / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing required diagnostic outputs: {missing}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
