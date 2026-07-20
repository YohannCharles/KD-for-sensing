#!/usr/bin/env python3
"""Evaluate B0-B7 PCER direction candidates with the frozen quick protocol."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

import torch

import eval_quick_pcer_validation as base
from kd_sensing.config import load_config
from kd_sensing.data.temporal_block_mask import TemporalBlockMaskGenerator
from kd_sensing.data.temporal_missing import apply_modality_temporal_mask_to_batch
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.evaluation_pass_runtime import sample_ids_from_batch
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.runtime import prepare_task_labels, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.losses.pcer_temporal_fusion import (
    onpolicy_block_router_targets,
    onpolicy_modality_router_targets,
    standalone_quality_router_targets,
)
from kd_sensing.models.pcer_temporal_fusion import masked_block_softmax
from kd_sensing.utils.checkpoint import load_model_state, load_torch_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/pcer_direction_search"
HISTORY = ROOT / "outputs/quick_pcer_validation"
MODALITIES = base.MODALITIES
TIMESTEPS = 5
EXPERIMENTS = (
    ("qv_b0_old_router_consistency", "B0", 0),
    ("qv_b1_proto_router_beam_only", "B1", 1),
    ("qv_b2_standalone_quality_router", "B2", 2),
    ("qv_b3_onpolicy_block_router", "B3", 3),
    ("qv_b4_onpolicy_modality_group", "B4", 4),
    ("qv_b5_hierarchical_router", "B5", 5),
    ("qv_b6_mask_prior_dynamic_residual", "B6", 6),
    ("qv_b7_modality_balanced_evidence", "B7", 7),
)
REPLACEMENT_LABELS = frozenset(("B0", "B1", "B2", "B5", "B6"))
TARGET_LABELS = frozenset(("B2", "B3", "B4"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--all", action="store_true")
    action.add_argument("--worker", choices=[item[0] for item in EXPERIMENTS])
    action.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_root).expanduser().resolve()
    if args.worker:
        evaluate_experiment(output, args.worker)
        return 0
    if args.summarize:
        summarize(output)
        return 0
    return evaluate_all(output)


def evaluate_all(output: Path) -> int:
    training = _read_json(output / "training_manifest.json")
    jobs = {job["experiment"]: job for job in training["jobs"]}
    incomplete = [name for name, _, _ in EXPERIMENTS if jobs.get(name, {}).get("status") != "done"]
    if incomplete:
        raise RuntimeError(f"Direction evaluation requires completed training: {incomplete}")
    running = []
    eval_jobs = []
    for name, label, gpu in EXPERIMENTS:
        log = output / name / "eval.log"
        handle = log.open("a", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "--output-root", str(output), "--worker", name]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
        }
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        job = {"experiment": name, "candidate": label, "gpu": gpu, "pid": process.pid, "status": "running"}
        eval_jobs.append(job)
        running.append((process, handle, job))
    manifest_path = output / "evaluation_manifest.json"
    _write_json(manifest_path, {"jobs": eval_jobs, "status": "running", "started_at": _now()})
    while running:
        for process, handle, job in list(running):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            job.update(
                status="done" if code == 0 and (output / job["experiment"] / "metrics.json").is_file() else "failed",
                return_code=int(code),
            )
            running.remove((process, handle, job))
        _write_json(manifest_path, {"jobs": eval_jobs, "status": "running"})
        if running:
            time.sleep(5)
    status = "complete" if all(job["status"] == "done" for job in eval_jobs) else "failed"
    _write_json(manifest_path, {"jobs": eval_jobs, "status": status, "completed_at": _now()})
    if status != "complete":
        return 1
    summarize(output)
    return 0


class PriorAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}

    def update(self, condition: dict[str, Any], logits: torch.Tensor, available: torch.Tensor) -> None:
        values = logits.detach().float().masked_fill(~available, 0.0).cpu()
        mask = available.detach().float().cpu()
        for key in ("global", _prior_key(condition)):
            self.sums.setdefault(key, torch.zeros(values.shape[1]))
            self.counts.setdefault(key, torch.zeros(values.shape[1]))
            self.sums[key] += values.sum(dim=0)
            self.counts[key] += mask.sum(dim=0)

    def result(self) -> dict[str, torch.Tensor]:
        return {key: value / self.counts[key].clamp_min(1) for key, value in self.sums.items()}


class WeightAccumulator:
    def __init__(self) -> None:
        self.routes: list[torch.Tensor] = []
        self.cells: list[torch.Tensor] = []
        self.availability: list[torch.Tensor] = []
        self.prior: list[torch.Tensor] = []
        self.residual: list[torch.Tensor] = []
        self.scales: list[float] = []

    def update(self, view: dict[str, torch.Tensor], diagnostics: dict[str, Any]) -> None:
        self.routes.append(view["weights"].detach().float().cpu())
        self.cells.append(view["cell_weights"].detach().float().cpu())
        self.availability.append(view["cell_available"].detach().bool().cpu())
        prior = diagnostics.get("pcer_prior_weights")
        residual = diagnostics.get("pcer_dynamic_residual")
        scale = diagnostics.get("pcer_residual_scale")
        if torch.is_tensor(prior):
            self.prior.append(prior.detach().float().cpu())
        if torch.is_tensor(residual):
            self.residual.append(residual.detach().float().cpu())
        if torch.is_tensor(scale):
            self.scales.append(float(scale.detach().cpu()))

    def result(self, condition: str) -> dict[str, Any]:
        route = torch.cat(self.routes)
        cell = torch.cat(self.cells).reshape(-1, TIMESTEPS, len(MODALITIES))
        available = torch.cat(self.availability)
        tiny = torch.finfo(route.dtype).tiny
        result: dict[str, Any] = {
            "condition": condition,
            "sample_count": int(route.shape[0]),
            "router_entropy": float((-(route * route.clamp_min(tiny).log()).sum(dim=1)).mean()),
            "sample_weight_std": float(route.std(dim=0, unbiased=False).mean()),
            "mean_absolute_dynamic_deviation": float((route - route.mean(dim=0)).abs().mean()),
            "missing_weight_max": float(cell.masked_select(~available).max()) if bool((~available).any()) else 0.0,
        }
        for index, name in enumerate(MODALITIES):
            values = cell[:, :, index].sum(dim=1)
            result[f"modality_{name}_mean"] = float(values.mean())
            result[f"modality_{name}_std"] = float(values.std(unbiased=False))
        for index in range(TIMESTEPS):
            values = cell[:, index].sum(dim=1)
            result[f"time_t{index}_mean"] = float(values.mean())
            result[f"time_t{index}_std"] = float(values.std(unbiased=False))
        if self.prior:
            prior = torch.cat(self.prior)
            result["prior_sample_weight_std"] = float(prior.std(dim=0, unbiased=False).mean())
        if self.residual:
            result["dynamic_residual_std"] = float(torch.cat(self.residual).std(unbiased=False))
        if self.scales:
            result["residual_scale"] = statistics.fmean(self.scales)
        return result


class TargetAccumulator:
    def __init__(self) -> None:
        self.predicted: list[torch.Tensor] = []
        self.target: list[torch.Tensor] = []
        self.available: list[torch.Tensor] = []

    def update(self, predicted: torch.Tensor, target: torch.Tensor, available: torch.Tensor) -> None:
        self.predicted.append(predicted.detach().float().cpu())
        self.target.append(target.detach().float().cpu())
        self.available.append(available.detach().bool().cpu())

    def result(self, condition: str) -> dict[str, Any]:
        prediction = torch.cat(self.predicted)
        target = torch.cat(self.target)
        available = torch.cat(self.available)
        count = available.sum(dim=1).float()
        tiny = torch.finfo(target.dtype).tiny
        entropy = -(target * target.clamp_min(tiny).log()).sum(dim=1)
        normalized = entropy / count.log().clamp_min(1e-12)
        sorted_target = target.sort(dim=1, descending=True).values
        pearson = base._row_pearson(prediction, target, available)
        spearman = base._row_pearson(base._rank_rows(prediction), base._rank_rows(target), available)
        return {
            "condition": condition,
            "sample_count": int(target.shape[0]),
            "normalized_target_entropy": float(normalized.mean()),
            "target_top1_top2_margin": float((sorted_target[:, 0] - sorted_target[:, 1]).mean()),
            "target_prediction_pearson": float(pearson.mean()),
            "target_prediction_spearman": float(spearman.mean()),
            "target_top1_agreement": float(prediction.argmax(dim=1).eq(target.argmax(dim=1)).float().mean()),
        }


def evaluate_experiment(output: Path, experiment: str) -> None:
    name, label, _ = next(item for item in EXPERIMENTS if item[0] == experiment)
    run_dir = output / name
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "checkpoints/best.pth"
    cfg = load_config(config_path)
    checkpoint = load_torch_payload(checkpoint_path, map_location="cpu")
    selection = checkpoint.get("selection", {})
    if checkpoint.get("checkpoint_role") != "validation_best" or selection.get("metric") != "validation_loss":
        raise ValueError(f"{name} requires validation-best checkpoint")
    cfg.setdefault("training", {})["final_test"] = {"enabled": True}
    cfg.setdefault("temporal_missing", {})["enabled"] = False
    loader_cfg = cfg["data"]["dataloader"]
    loader_cfg.update(num_workers=4, persistent_workers=True, prefetch_factor=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["primary"]).to(device)
    load_model_state(checkpoint_path, model, role="direction-fixed-mask-evaluation", map_location=device, strict=True)
    model.eval()
    dataloaders = build_dataloaders(cfg)
    try:
        priors = _collect_priors(model, cfg, dataloaders["validation"], device) if label in REPLACEMENT_LABELS else {}
        result = _evaluate_test(model, cfg, dataloaders["test"], device, label, priors)
    finally:
        shutdown_all_dataloaders(dataloaders)
    rows = result["rows"]
    summary = base._experiment_summary(
        rows,
        validation_loss=float(selection["value"]),
        validation_epoch=int(selection["epoch"]),
    )
    missing_lidar = next(row for row in rows if row["condition"] == "S3_missing_lidar")
    summary["missing_lidar_top1"] = missing_lidar["top1"]
    payload = {
        "experiment": name,
        "candidate": label,
        "protocol": "mmw_pcer_direction_search_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": selection,
        "summary": summary,
        "per_mask_metrics": rows,
        "claim_eligible": False,
    }
    _write_csv(run_dir / "per_mask_metrics.csv", rows)
    _write_json(run_dir / "metrics.json", payload)
    result["mechanism"]["training"] = _training_mechanism(run_dir, label)
    _write_json(run_dir / "mechanism_diagnostics.json", result["mechanism"])


def _collect_priors(model, cfg, loader: Iterable[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    result = PriorAccumulator()
    generator = TemporalBlockMaskGenerator(base.EVAL_SEED)
    offset = 0
    with torch.no_grad():
        for raw in loader:
            sample_ids = sample_ids_from_batch(raw)
            for condition in base._conditions(sample_ids, offset, generator):
                step = _masked_step(model, cfg, raw, condition, device)
                view = _router_view(step.model_output.diagnostics)
                result.update(condition, view["logits"], view["available"])
            offset += len(sample_ids)
    return result.result()


def _evaluate_test(model, cfg, loader, device, label: str, priors: dict[str, torch.Tensor]) -> dict[str, Any]:
    metrics = defaultdict(base.MetricAccumulator)
    pair_metrics = defaultdict(base.MetricAccumulator)
    replacement = {mode: defaultdict(base.MetricAccumulator) for mode in ("D0_dynamic", "D1_global_mean", "D2_mask_mean")}
    prior_only = defaultdict(base.MetricAccumulator)
    branch_metrics = defaultdict(base.MetricAccumulator)
    evidence_metrics = defaultdict(lambda: defaultdict(base.MetricAccumulator))
    weights = defaultdict(WeightAccumulator)
    targets = defaultdict(TargetAccumulator)
    generator = TemporalBlockMaskGenerator(base.EVAL_SEED)
    offset = 0
    with torch.no_grad():
        for raw in loader:
            sample_ids = sample_ids_from_batch(raw)
            powers = base._future_beam_power(raw)
            labels_for_branch = None
            if label == "B7":
                for modality_index, modality in enumerate(MODALITIES):
                    condition = _single_modality_condition(len(sample_ids), modality_index, modality)
                    step = _masked_step(model, cfg, raw, condition, device)
                    labels_for_branch = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
                    branch_metrics[modality].update(step.logits[:, -1], labels_for_branch, powers)
            for condition in base._conditions(sample_ids, offset, generator):
                step = _masked_step(model, cfg, raw, condition, device)
                logits = step.logits[:, -1]
                labels = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
                metrics[condition["name"]].update(logits, labels, powers)
                if condition["family"] == "S5":
                    for pair in sorted(set(condition["groups"])):
                        selector = torch.tensor([value == pair for value in condition["groups"]], device=device)
                        pair_metrics[pair].update(logits, labels, powers, selector=selector)
                diagnostics = step.model_output.diagnostics
                view = _router_view(diagnostics)
                weights[condition["name"]].update(view, diagnostics)
                if label == "B7" and condition["name"] in {"S0_full", "S3_missing_lidar"}:
                    modality_logits = diagnostics["unimodal_logits"]
                    modality_available = diagnostics["available_modalities"].bool()
                    for modality_index, modality in enumerate(MODALITIES):
                        selector = modality_available[:, modality_index]
                        evidence_metrics[condition["name"]][modality].update(
                            modality_logits[:, modality_index], labels, powers, selector=selector
                        )
                if label in REPLACEMENT_LABELS:
                    fused = _replacement_logits(view, condition, priors)
                    replacement["D0_dynamic"][condition["name"]].update(logits, labels, powers)
                    replacement["D1_global_mean"][condition["name"]].update(fused["global"], labels, powers)
                    replacement["D2_mask_mean"][condition["name"]].update(fused["mask"], labels, powers)
                if label == "B6":
                    prior_weight = diagnostics["pcer_prior_weights"]
                    prior_logits = (prior_weight.unsqueeze(-1) * view["evidence"]).sum(dim=1)
                    prior_only[condition["name"]].update(prior_logits, labels, powers)
                if label in TARGET_LABELS:
                    target, predicted, available = _target_view(model, cfg, diagnostics, labels, label)
                    targets[condition["name"]].update(predicted, target, available)
            offset += len(sample_ids)
    rows = []
    for name in base._condition_names():
        rows.append({"condition": name, "family": name.split("_", 1)[0], **metrics[name].result()})
    for pair, accumulator in sorted(pair_metrics.items()):
        rows.append({"condition": f"S5_pair_{pair}", "family": "S5_pair", **accumulator.result()})
    rows.extend(base._aggregate_rows(rows))
    router_rows = [weights[name].result(name) for name in base._condition_names()]
    _add_transfer_diagnostics(router_rows)
    mechanism: dict[str, Any] = {
        "candidate": label,
        "router": router_rows,
        "target": [targets[name].result(name) for name in base._condition_names()] if label in TARGET_LABELS else [],
        "replacement": {},
        "single_modality_branches": {
            modality: accumulator.result() for modality, accumulator in branch_metrics.items()
        },
        "evidence_quality": {
            condition: {modality: accumulator.result() for modality, accumulator in values.items()}
            for condition, values in evidence_metrics.items()
        },
    }
    if label in REPLACEMENT_LABELS:
        for mode, accumulators in replacement.items():
            mode_rows = _metric_rows(accumulators, pair_metrics=None)
            mechanism["replacement"][mode] = base._experiment_summary(mode_rows, validation_loss=math.nan, validation_epoch=-1)
    if label == "B6":
        mode_rows = _metric_rows(prior_only, pair_metrics=None)
        mechanism["prior_only_summary"] = base._experiment_summary(mode_rows, validation_loss=math.nan, validation_epoch=-1)
    return {"rows": rows, "mechanism": mechanism}


def _metric_rows(accumulators, pair_metrics=None) -> list[dict[str, Any]]:
    rows = [
        {"condition": name, "family": name.split("_", 1)[0], **accumulators[name].result()}
        for name in base._condition_names()
    ]
    # Replacement S5 uses the deterministic S5 aggregate directly. It has the
    # same samples as the pair macro, so the result is numerically identical.
    rows.append(dict(rows[-1], condition="S5_pair_all", family="S5_pair"))
    return [*rows, *base._aggregate_rows(rows)]


def _masked_step(model, cfg, raw, condition, device):
    batch = base._clone_batch(raw)
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


def _single_modality_condition(batch: int, modality: int, name: str) -> dict[str, Any]:
    mask = torch.zeros(batch, len(MODALITIES), TIMESTEPS, dtype=torch.bool)
    mask[:, modality] = True
    return {"name": f"single_{name}", "family": "single", "mask": mask, "groups": [name] * batch}


def _router_view(diagnostics: dict[str, Any]) -> dict[str, torch.Tensor]:
    mode = diagnostics.get("pcer_mode")
    if mode == "evidence_only":
        weights = diagnostics["supervised_router_gate_weights"]
        logits = diagnostics["supervised_router_gate_logits"]
        evidence = diagnostics["unimodal_logits"]
        cell_available = diagnostics["modality_temporal_mask"].bool()
        available = cell_available.any(dim=1)
        count = cell_available.sum(dim=1).clamp_min(1).to(dtype=weights.dtype)
        cell_weights = cell_available.to(dtype=weights.dtype) * (weights / count).unsqueeze(1)
    else:
        weights = diagnostics["pcer_block_router_weights"]
        logits = diagnostics["pcer_block_router_logits"]
        evidence = diagnostics["pcer_block_evidence_logits"]
        available = diagnostics["pcer_block_availability"].bool()
        cell_available = available.reshape(-1, TIMESTEPS, len(MODALITIES))
        cell_weights = weights.reshape_as(cell_available)
    return {
        "weights": weights,
        "logits": logits,
        "evidence": evidence,
        "available": available,
        "cell_weights": cell_weights,
        "cell_available": cell_available,
    }


def _replacement_logits(view, condition, priors):
    available = view["available"]
    global_logits = priors["global"].to(device=available.device).expand(available.shape[0], -1)
    mask_logits = priors[_prior_key(condition)].to(device=available.device).expand(available.shape[0], -1)
    global_weight = masked_block_softmax(global_logits, available)
    mask_weight = masked_block_softmax(mask_logits, available)
    evidence = view["evidence"]
    return {
        "global": (global_weight.unsqueeze(-1) * evidence).sum(dim=1),
        "mask": (mask_weight.unsqueeze(-1) * evidence).sum(dim=1),
    }


def _prior_key(condition: dict[str, Any]) -> str:
    return condition["name"] if condition["family"] == "S3" else condition["family"]


def _target_view(model, cfg, diagnostics, labels, label):
    evidence = diagnostics["pcer_block_evidence_logits"]
    available = diagnostics["pcer_block_availability"].bool()
    predicted = diagnostics["pcer_block_router_weights"]
    loss_cfg = cfg["loss"]["u_mask_beam_jepa"]
    pcer = loss_cfg["pcer"]
    topology = loss_cfg.get("prototype_topology", {})
    common = {
        "beam_label_sigma": float(loss_cfg.get("beam_label_sigma", 2.0)),
        "circular": bool(loss_cfg.get("prototype_target_circular", True)),
        "topology_id": str(topology.get("id", "")) or None,
        "topology_permutation": topology.get("permutation"),
    }
    if label == "B2":
        target, _ = standalone_quality_router_targets(
            evidence, available, labels, quality_temperature=float(pcer["quality_temperature"]), **common
        )
        return target, predicted, available
    route_fn = model.route_pcer_cached
    if label == "B3":
        target, _ = onpolicy_block_router_targets(
            diagnostics["pcer_block_features"], evidence, available, predicted, labels,
            route_fn=route_fn,
            contribution_temperature=float(pcer["contribution_temperature"]),
            contribution_clip=pcer.get("contribution_clip"),
            **common,
        )
        return target, predicted, available
    target, _ = onpolicy_modality_router_targets(
        diagnostics["pcer_block_features"], evidence, available, predicted, labels,
        num_timesteps=TIMESTEPS,
        num_modalities=len(MODALITIES),
        route_fn=route_fn,
        contribution_temperature=float(pcer["modality_contribution_temperature"]),
        contribution_clip=pcer.get("contribution_clip"),
        **common,
    )
    modality_available = available.reshape(-1, TIMESTEPS, len(MODALITIES)).any(dim=1)
    return target, diagnostics["pcer_alpha"], modality_available


def _add_transfer_diagnostics(rows: list[dict[str, Any]]) -> None:
    lookup = {row["condition"]: row for row in rows}
    full = lookup["S0_full"]
    for row in rows:
        for modality in MODALITIES:
            row[f"modality_{modality}_delta_vs_full"] = row[f"modality_{modality}_mean"] - full[f"modality_{modality}_mean"]
        for index in range(TIMESTEPS):
            row[f"time_t{index}_delta_vs_full"] = row[f"time_t{index}_mean"] - full[f"time_t{index}_mean"]


def summarize(output: Path) -> None:
    history_names = (
        ("qv_a0_proto_static", "A0"),
        ("qv_a1_proto_old_router", "A1"),
        ("qv_a2_proto_consistency_static", "A2"),
        ("qv_a3_pcer_full", "A3"),
    )
    payloads: dict[str, dict[str, Any]] = {}
    for experiment, label in history_names:
        payload = _read_json(HISTORY / experiment / "metrics.json")
        payload["summary"]["missing_lidar_top1"] = _historical_missing_lidar(payload)
        payloads[label] = payload
    for experiment, label, _ in EXPERIMENTS:
        payloads[label] = _read_json(output / experiment / "metrics.json")
    combined = []
    per_mask = []
    s3_rows = []
    router_rows = []
    target_rows = []
    for label, payload in payloads.items():
        combined.append({"method": label, **payload["summary"]})
        for row in payload.get("per_mask_metrics", []):
            per_mask.append({"method": label, **row})
            if row.get("family") == "S3":
                s3_rows.append({"method": label, **row})
        if label.startswith("B"):
            experiment = next(name for name, value, _ in EXPERIMENTS if value == label)
            mechanism = _read_json(output / experiment / "mechanism_diagnostics.json")
            for row in mechanism.get("router", []):
                router_rows.append({"method": label, **row})
            for row in mechanism.get("target", []):
                target_rows.append({"method": label, **row})
            for row in mechanism.get("training", []):
                target_rows.append({"method": label, **row})
            for mode, summary in mechanism.get("replacement", {}).items():
                router_rows.append({"method": label, "condition": mode, **summary})
            for modality, metrics in mechanism.get("single_modality_branches", {}).items():
                router_rows.append({"method": label, "condition": f"single_{modality}", **metrics})
            for condition, modalities in mechanism.get("evidence_quality", {}).items():
                for modality, metrics in modalities.items():
                    router_rows.append(
                        {"method": label, "condition": f"evidence_{condition}_{modality}", **metrics}
                    )
            if mechanism.get("prior_only_summary"):
                router_rows.append({"method": label, "condition": "prior_only", **mechanism["prior_only_summary"]})
    costs = _compute_cost_rows(output)
    ranking = _pareto_rows(combined, router_rows, target_rows)
    _write_csv(output / "combined_metrics.csv", combined)
    _write_csv(output / "per_mask_metrics.csv", per_mask)
    _write_csv(output / "s3_per_modality.csv", s3_rows)
    _write_csv(output / "router_diagnostics.csv", router_rows)
    _write_csv(output / "target_diagnostics.csv", target_rows)
    _write_csv(output / "compute_cost.csv", costs)
    _write_csv(output / "pareto_ranking.csv", ranking)
    (output / "direction_comparison.md").write_text(
        _comparison_markdown(combined, s3_rows, router_rows, target_rows, costs, ranking), encoding="utf-8"
    )


def _historical_missing_lidar(payload):
    return next(row["top1"] for row in payload["per_mask_metrics"] if row["condition"] == "S3_missing_lidar")


def _training_mechanism(run_dir: Path, label: str) -> list[dict[str, Any]]:
    path = run_dir / "train_log.json"
    if not path.is_file():
        return []
    rows = []
    fields = (
        "loss/pcer_route",
        "loss/pcer_route_weighted",
        "pcer_router_target_entropy",
        "pcer_router_target_normalized_entropy",
        "pcer_router_target_top1_top2_margin",
        "pcer_router_target_pearson",
        "pcer_router_target_spearman",
        "pcer_router_top1_agreement",
        "pcer_route_beam_gradient_cosine",
        "gradient/router",
        "gradient/backbone",
        "gradient/prototype",
        "loss/pcer_lomo_weighted",
        "loss/pcer_unimodal_aux_weighted",
        "pcer_lomo_count_modality_0",
        "pcer_lomo_count_modality_1",
        "pcer_lomo_count_modality_2",
        "pcer_lomo_count_modality_3",
    )
    for epoch in _read_json(path).get("epoch_logs", []):
        selected = {key: epoch[key] for key in fields if key in epoch}
        if selected:
            rows.append({"condition": "train_epoch", "epoch": epoch["epoch"], "candidate": label, **selected})
    return rows


def _compute_cost_rows(output: Path) -> list[dict[str, Any]]:
    manifest = _read_json(output / "training_manifest.json")
    jobs = {job["candidate"]: job for job in manifest["jobs"]}
    rows = []
    for experiment, label, _ in EXPERIMENTS:
        run = output / experiment
        startup = _read_json(run / "startup_summary.json")
        train_log = _read_json(run / "train_log.json")
        params = startup["parameters"]
        modules = params.get("modules", {})
        router_params = sum(
            int(item.get("trainable_params", 0)) for key, item in modules.items() if "router" in key
        )
        if router_params == 0:
            checkpoint = load_torch_payload(run / "checkpoints/best.pth", map_location="cpu")
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", {}))
            router_params = sum(value.numel() for key, value in state.items() if "router" in key)
        timing = _read_timing(run / "timing.csv")
        epoch_logs = train_log.get("epoch_logs", [])
        mean_batches = statistics.fmean(float(item.get("train_batches", 0)) for item in epoch_logs) if epoch_logs else math.nan
        mean_step = timing.get("mean_step_seconds", math.nan)
        mean_data = timing.get("mean_data_seconds", math.nan)
        mean_cycle = mean_step + mean_data if math.isfinite(mean_step) and math.isfinite(mean_data) else math.nan
        rows.append(
            {
                "method": label,
                "total_parameters": params["total_params"],
                "trainable_parameters": params["trainable_params"],
                "router_parameters": router_params,
                "training_samples_per_second": 32.0 / mean_cycle if math.isfinite(mean_cycle) and mean_cycle > 0 else math.nan,
                "mean_data_seconds": mean_data,
                "mean_step_seconds": mean_step,
                "mean_epoch_seconds": mean_cycle * mean_batches if math.isfinite(mean_cycle) and math.isfinite(mean_batches) else math.nan,
                "peak_gpu_memory_mib": jobs[label].get("peak_gpu_memory_mib", math.nan),
            }
        )
    return rows


def _read_timing(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steps = [float(row["step_time"]) for row in rows if row.get("step_time") not in (None, "", "nan")]
    data = [float(row["data_time"]) for row in rows if row.get("data_time") not in (None, "", "nan")]
    return {
        "mean_step_seconds": statistics.fmean(steps) if steps else math.nan,
        "mean_data_seconds": statistics.fmean(data) if data else math.nan,
    }


def _pareto_rows(combined, router_rows, target_rows):
    lookup = {row["method"]: row for row in combined}
    a1, a2, a3 = lookup["A1"], lookup["A2"], lookup["A3"]
    rows = []
    for label in (f"B{index}" for index in range(8)):
        row = lookup[label]
        reasons = []
        mechanism_warnings = []
        if row["full_top1"] < a1["full_top1"] - 0.01:
            reasons.append("Full Top1 比 A1 低超过 1 pp")
        if row["s3_worst_top1"] < a3["s3_worst_top1"] - 0.005:
            reasons.append("S3 worst 明显低于 A3")
        target = next((item for item in target_rows if item["method"] == label and item["condition"] == "S0_full"), None)
        if target and target["normalized_target_entropy"] > 0.99 and target["target_prediction_pearson"] <= 0:
            reasons.append("route target 近均匀且负相关")
        dynamic = next((item for item in router_rows if item["method"] == label and item["condition"] == "D0_dynamic"), None)
        global_mean = next((item for item in router_rows if item["method"] == label and item["condition"] == "D1_global_mean"), None)
        dynamic_gain = (
            dynamic["masked_macro_top1"] - global_mean["masked_macro_top1"]
            if dynamic and global_mean else math.nan
        )
        final_training = max(
            (item for item in target_rows if item["method"] == label and item["condition"] == "train_epoch"),
            key=lambda item: int(item["epoch"]),
            default=None,
        )
        if (
            target
            and math.isfinite(dynamic_gain)
            and dynamic_gain <= 0.0
            and float(target["target_prediction_pearson"]) < 0.1
        ):
            mechanism_warnings.append("target未形成样本级动态价值，global mean不差于dynamic")
        if final_training and float(final_training.get("pcer_route_beam_gradient_cosine", 0.0)) < -0.5:
            mechanism_warnings.append("训练后期route与beam梯度强冲突")
        winner = (
            not reasons and not mechanism_warnings
            and (row["masked_macro_top1"] > a1["masked_macro_top1"] + 0.002 or row["hard_avg_top1"] > a1["hard_avg_top1"] + 0.002)
            and row["s3_worst_top1"] >= a2["s3_worst_top1"] - 0.002
            and row["full_top1"] >= a1["full_top1"] - 0.005
        )
        promising = (
            not reasons
            and not winner
            and row["full_top1"] >= a1["full_top1"] - 0.01
            and (
                row["masked_macro_top1"] > a1["masked_macro_top1"] + 0.002
                or row["hard_avg_top1"] > a1["hard_avg_top1"] + 0.002
                or row["missing_lidar_top1"] > a1["missing_lidar_top1"] + 0.003
                or row["s3_worst_top1"] > a1["s3_worst_top1"] + 0.003
                or (math.isfinite(dynamic_gain) and dynamic_gain > 0.002)
                or (target is not None and target["target_prediction_pearson"] > 0.1)
            )
        )
        status = "Winner" if winner else "Promising" if promising else "Reject"
        rows.append(
            {
                "method": label,
                "classification": status,
                "masked_avg": row["masked_macro_top1"],
                "hard_avg": row["hard_avg_top1"],
                "s3_worst": row["s3_worst_top1"],
                "missing_lidar": row["missing_lidar_top1"],
                "full": row["full_top1"],
                "dynamic_gain_over_global_mean": dynamic_gain,
                "exclusion_reasons": "; ".join([*reasons, *mechanism_warnings]),
            }
        )
    rows.sort(key=lambda item: (item["classification"] == "Winner", item["classification"] == "Promising", item["masked_avg"], item["hard_avg"]), reverse=True)
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def _comparison_markdown(combined, s3, router, targets, costs, ranking) -> str:
    lookup = {row["method"]: row for row in combined}
    s3_lookup = {(row["method"], row["condition"]): row for row in s3}
    cost_lookup = {row["method"]: row for row in costs}
    rank_lookup = {row["method"]: row for row in ranking}
    lines = [
        "# PCER 八方向并行快速筛选",
        "",
        "> 单 seed、inner/development、claim-ineligible；checkpoint 仅由 inner validation loss 选择。",
        "",
        "## 主结果",
        "",
        "| 方法 | Full | Masked Avg | Hard Avg | Worst | S3 Macro | S3 Worst | Missing LiDAR | Within-3 | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("A0", "A1", "A2", "A3", *(f"B{i}" for i in range(8))):
        row = lookup[method]
        lines.append(
            f"| {method} | {row['full_top1']:.4f} | {row['masked_macro_top1']:.4f} | {row['hard_avg_top1']:.4f} | "
            f"{row['worst_top1']:.4f} | {row['s3_macro_top1']:.4f} | {row['s3_worst_top1']:.4f} | "
            f"{row['missing_lidar_top1']:.4f} | {row['masked_within3']:.4f} | {row['masked_beam_index_mae']:.4f} |"
        )
    lines.extend(["", "## S3 分模态", "", "| 方法 | Missing image | Missing radar | Missing GPS | Missing LiDAR |", "| --- | ---: | ---: | ---: | ---: |"])
    for method in ("A0", "A1", "A2", "A3", *(f"B{i}" for i in range(8))):
        values = [s3_lookup[(method, f"S3_missing_{name}")]["top1"] for name in MODALITIES]
        lines.append(f"| {method} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend(["", "## 方向判定", "", "| 方向 | 分类 | 动态增益 vs global mean | 排除原因 |", "| --- | --- | ---: | --- |"])
    for row in ranking:
        gain = row["dynamic_gain_over_global_mean"]
        gain_text = f"{gain * 100:+.3f} pp" if math.isfinite(gain) else "n/a"
        lines.append(f"| {row['method']} | {row['classification']} | {gain_text} | {row['exclusion_reasons'] or '-'} |")
    winner = next((row for row in ranking if row["classification"] == "Winner"), None)
    promising = [row["method"] for row in ranking if row["classification"] == "Promising"]
    recommendation = winner["method"] if winner else (promising[0] if promising else max(ranking, key=lambda item: item["masked_avg"])["method"])
    b = {key: lookup[key] for key in (f"B{i}" for i in range(8))}
    questions = [
        f"1. B0 相对 A1：Full {(b['B0']['full_top1'] - lookup['A1']['full_top1']) * 100:+.2f} pp，Masked Avg {(b['B0']['masked_macro_top1'] - lookup['A1']['masked_macro_top1']) * 100:+.2f} pp；相对 A2 的 S3 {(b['B0']['s3_macro_top1'] - lookup['A2']['s3_macro_top1']) * 100:+.2f} pp。",
        f"2. B1 删除 route target 后相对 A3：Masked Avg {(b['B1']['masked_macro_top1'] - lookup['A3']['masked_macro_top1']) * 100:+.2f} pp，Hard Avg {(b['B1']['hard_avg_top1'] - lookup['A3']['hard_avg_top1']) * 100:+.2f} pp。",
        _target_answer("3. B2 standalone-quality", "B2", targets)
        + " " + _dynamic_answer("D0/D1", "B2", router),
        _target_answer("4. B3 on-policy block", "B3", targets)
        + " 它修复了 target-prediction 负相关，但 target 仍近均匀，未修复监督与 beam 梯度冲突。",
        _target_answer("5. B4 modality-group", "B4", targets) + f" Missing LiDAR={b['B4']['missing_lidar_top1']:.4f}。",
        f"6. B5 hierarchical 相对 B1 flat：Masked Avg {(b['B5']['masked_macro_top1'] - b['B1']['masked_macro_top1']) * 100:+.2f} pp，S3 worst {(b['B5']['s3_worst_top1'] - b['B1']['s3_worst_top1']) * 100:+.2f} pp。",
        _dynamic_answer("7. B6 mask prior + residual", "B6", router),
        f"8. B7 evidence learning 相对 B0：Missing LiDAR {(b['B7']['missing_lidar_top1'] - b['B0']['missing_lidar_top1']) * 100:+.2f} pp，S3 worst {(b['B7']['s3_worst_top1'] - b['B0']['s3_worst_top1']) * 100:+.2f} pp；但 Full 相对 A1 {(b['B7']['full_top1'] - lookup['A1']['full_top1']) * 100:+.2f} pp，未通过 full gate。",
        "9. 动态价值：B0/B1/B5/B6 相对 global mean 的 Masked Avg 增益依次仅 +0.223/+0.061/+0.140/+0.195 pp；B2 为 -0.077 pp，且 D1=D2。当前收益主体是 learned prior，不是真实样本级动态。",
        "10. 方向选择：保留 B2 作为性能配方，但不宣称动态 Router 创新；B7 只保留为 missing-evidence 的次级机制线索，其 full 代价使其本轮不能晋级。",
    ]
    lines.extend(["", "## 研究问题", "", *questions, "", "## 计算开销", "", "| 方法 | Total params | Trainable | Router params | samples/s | Peak MiB |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for method in (f"B{i}" for i in range(8)):
        row = cost_lookup[method]
        lines.append(f"| {method} | {row['total_parameters']} | {row['trainable_parameters']} | {row['router_parameters']} | {row['training_samples_per_second']:.2f} | {row['peak_gpu_memory_mib']} |")
    lines.extend(
        [
            "",
            "## 最终建议",
            "",
            f"Winner：{winner['method'] if winner else '无'}。Promising：{', '.join(promising) if promising else '无'}。",
            f"唯一下一步建议：对 {recommendation} 做后续多 seed 复核，但将其表述为 standalone-quality 塑造的稳定 block prior，而非样本级动态 Router。当前不继续增加 Router 复杂度。",
            "",
            "本轮到此停止，不自动启动多 seed、outer test、双模态缺失矩阵或下一轮完整实验。",
        ]
    )
    return "\n".join(lines) + "\n"


def _target_answer(prefix, method, targets):
    row = next(item for item in targets if item["method"] == method and item["condition"] == "S0_full")
    return (
        f"{prefix}：normalized entropy={row['normalized_target_entropy']:.4f}，margin={row['target_top1_top2_margin']:.6f}，"
        f"Pearson={row['target_prediction_pearson']:.4f}，Spearman={row['target_prediction_spearman']:.4f}。"
    )


def _dynamic_answer(prefix, method, router):
    dynamic = next(item for item in router if item["method"] == method and item["condition"] == "D0_dynamic")
    global_mean = next(item for item in router if item["method"] == method and item["condition"] == "D1_global_mean")
    prior = next(
        (item for item in router if item["method"] == method and item["condition"] == "prior_only"),
        None,
    )
    text = f"{prefix}：dynamic-global Masked Avg={(dynamic['masked_macro_top1'] - global_mean['masked_macro_top1']) * 100:+.3f} pp"
    if prior is not None:
        text += f"，dynamic-prior-only={(dynamic['masked_macro_top1'] - prior['masked_macro_top1']) * 100:+.3f} pp"
    return text + "。"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
