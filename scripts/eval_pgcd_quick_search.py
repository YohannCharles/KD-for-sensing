#!/usr/bin/env python3
"""Evaluate E0-E5 and summarize the eight MMW PGCD quick-search runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from kd_sensing.config import load_config
from kd_sensing.data.sensor_degradation import (
    SensorDegradationGenerator,
    TRAIN_CORRUPTIONS,
    UNSEEN_CORRUPTIONS,
    assert_pgcd_channel_free,
)
from kd_sensing.data.temporal_block_mask import TemporalBlockMaskGenerator
from kd_sensing.data.temporal_missing import apply_modality_temporal_mask_to_batch
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.evaluation_pass_runtime import metadata_rows_from_batch, sample_ids_from_batch
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.runtime import prepare_task_labels, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.losses.pgcd import pgcd_degradation_targets
from kd_sensing.utils.checkpoint import load_model_state, load_torch_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/pgcd_quick_search"
MODALITIES = ("image", "radar", "gps", "lidar")
DISPLAY_MODALITIES = ("image", "lidar", "radar", "gps")
EVAL_SEED = 20260720
EXPERIMENTS = (
    "qv_c0_corrupt_global_prior",
    "qv_c1_severity_quality",
    "qv_c2_entropy_quality",
    "qv_c3_proto_drift_reg",
    "qv_c4_proto_drift_rank",
    "qv_c5_task_degradation",
    "qv_c6_combined_quality",
    "qv_c7_full_pgcd",
)
ABLATIONS = tuple(f"C{index}" for index in range(8))
REPLACEMENTS = ("D0_dynamic", "D1_global_mean", "D2_sensor_severity", "D3_prior_only")
QUALITY_VARIANTS = frozenset({"c1", "c3", "c4", "c5", "c6", "c7"})


@dataclass(frozen=True)
class Condition:
    name: str
    protocol: str
    family: str
    sensor: str = "all"
    corruption: str = "none"
    severity: int = 0
    fixed: Mapping[str, Any] | None = None
    mask_type: str | None = None


@dataclass(frozen=True)
class EvalBatch:
    batch: dict[str, Any]
    availability: torch.Tensor
    severity: torch.Tensor
    corrupted: torch.Tensor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--worker", choices=EXPERIMENTS)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--train-stat-batches", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output_root).expanduser().resolve()
    if args.worker:
        evaluate_experiment(output, args.worker, train_stat_batches=max(1, int(args.train_stat_batches)))
        return 0
    if not args.all:
        parser.error("select --all or --worker")
    gpus = tuple(int(item.strip()) for item in args.gpus.split(",") if item.strip())
    if gpus != (4, 5):
        parser.error("PGCD evaluation requires --gpus 4,5")
    return evaluate_all(output, gpus, train_stat_batches=max(1, int(args.train_stat_batches)))


def evaluate_all(output: Path, gpus: tuple[int, ...], *, train_stat_batches: int) -> int:
    training = _read_json(output / "training_manifest.json")
    jobs = {job["experiment"]: job for job in training["jobs"]}
    incomplete = [name for name in EXPERIMENTS if not _training_done(jobs.get(name, {}))]
    if incomplete:
        raise RuntimeError(f"PGCD evaluation requires completed best-validation checkpoints: {incomplete}")
    rows = [
        {"experiment": name, "gpu": gpus[index % len(gpus)], "status": "queued"}
        for index, name in enumerate(EXPERIMENTS)
    ]
    queues = {gpu: [row for row in rows if row["gpu"] == gpu] for gpu in gpus}
    running: dict[int, tuple[subprocess.Popen, Any, dict[str, Any]]] = {}
    manifest_path = output / "evaluation_manifest.json"
    _write_json(manifest_path, {"protocol": "mmw_pgcd_fixed_eval_v1", "status": "running", "jobs": rows})
    while running or any(queues.values()):
        for gpu in gpus:
            if gpu in running or not queues[gpu]:
                continue
            job = queues[gpu].pop(0)
            name = job["experiment"]
            log = (output / name / "eval.log").open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output-root",
                str(output),
                "--worker",
                name,
                "--train-stat-batches",
                str(train_stat_batches),
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "4"}
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            job.update(pid=process.pid, status="running", start_time=_now())
            running[gpu] = (process, log, job)
            _write_json(manifest_path, {"protocol": "mmw_pgcd_fixed_eval_v1", "status": "running", "jobs": rows})
        for gpu, (process, log, job) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            result = output / job["experiment"] / "metrics.json"
            job.update(status="done" if code == 0 and result.is_file() else "failed", return_code=int(code), end_time=_now())
            del running[gpu]
            _write_json(manifest_path, {"protocol": "mmw_pgcd_fixed_eval_v1", "status": "running", "jobs": rows})
        if running or any(queues.values()):
            time.sleep(5)
    status = "complete" if all(row["status"] == "done" for row in rows) else "failed"
    _write_json(manifest_path, {"protocol": "mmw_pgcd_fixed_eval_v1", "status": status, "jobs": rows, "completed_at": _now()})
    if status != "complete":
        return 1
    summarize(output)
    return 0


def evaluate_experiment(output: Path, experiment: str, *, train_stat_batches: int) -> None:
    run_dir = output / experiment
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "checkpoints/best.pth"
    cfg = load_config(config_path)
    assert_pgcd_channel_free(cfg)
    checkpoint = load_torch_payload(checkpoint_path, map_location="cpu")
    selection = checkpoint.get("selection", {}) if isinstance(checkpoint, dict) else {}
    if checkpoint.get("checkpoint_role") != "validation_best" or selection.get("metric") != "validation_loss":
        raise ValueError(f"{experiment} requires the validation_best/validation_loss checkpoint.")
    training_metrics = _read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").is_file() else {}
    cfg.setdefault("training", {})["final_test"] = {"enabled": True}
    cfg.setdefault("temporal_missing", {})["enabled"] = False
    cfg["data"]["dataloader"]["shuffle_train"] = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["primary"]).to(device)
    load_model_state(checkpoint_path, model, role="pgcd-fixed-evaluation", map_location=device, strict=True)
    model.eval()
    dataloaders = build_dataloaders(cfg)
    if "test" not in dataloaders:
        shutdown_all_dataloaders(dataloaders)
        raise ValueError("PGCD fixed evaluation requires the historical development test loader.")
    variant = str(cfg["model"]["primary"]["pgcd"]["variant"]).lower()
    generator = SensorDegradationGenerator(EVAL_SEED)
    mask_generator = TemporalBlockMaskGenerator(EVAL_SEED)
    train_stats = _fit_train_statistics(
        model, cfg, dataloaders["train"], generator, device=device, max_batches=train_stat_batches
    )
    conditions = _conditions()
    condition_rows: list[dict[str, Any]] = []
    dynamic_rows: list[dict[str, Any]] = []
    weather_acc = {name: MetricAccumulator() for name in ("all", "sunny", "rainy", "foggy")}
    weather_diagnostics = WeatherDiagnosticsAccumulator()
    scene_acc: dict[str, MetricAccumulator] = {}
    quality = QualityAccumulator(variant)
    try:
        for condition in conditions:
            accumulators = {name: MetricAccumulator() for name in REPLACEMENTS}
            for raw_batch in dataloaders["test"]:
                assert_pgcd_channel_free(cfg, raw_batch)
                degraded = _condition_batch(raw_batch, condition, generator, mask_generator)
                step = _forward(model, cfg, degraded, device)
                diagnostics = step.model_output.diagnostics
                labels = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
                logits = step.logits[:, -1]
                accumulators["D0_dynamic"].update(logits, labels)
                if variant in QUALITY_VARIANTS:
                    quality.observe_reliability(diagnostics, degraded, condition)
                if condition.protocol == "E0":
                    weather_rows = _weather_rows(raw_batch, len(labels))
                    scene_rows = _scene_rows(raw_batch, len(labels))
                    weather_acc["all"].update(logits, labels)
                    weather_diagnostics.update("all", diagnostics, labels)
                    for weather in ("sunny", "rainy", "foggy"):
                        selector = torch.tensor([item == weather for item in weather_rows], device=device)
                        weather_acc[weather].update(logits, labels, selector=selector)
                        weather_diagnostics.update(weather, diagnostics, labels, selector=selector)
                    for scene in sorted(set(scene_rows)):
                        selector = torch.tensor([item == scene for item in scene_rows], device=device)
                        scene_acc.setdefault(scene, MetricAccumulator()).update(logits, labels, selector=selector)
                if variant != "c0":
                    cached = _cached_tensors(diagnostics)
                    global_override = train_stats["global_degradation"].to(device).unsqueeze(0).expand(len(labels), -1)
                    d1 = model.route_pgcd_cached(*cached, degradation_override=global_override)["fused_logits"]
                    condition_override = _condition_override(degraded, train_stats["sensor_severity_degradation"], device)
                    d2 = model.route_pgcd_cached(*cached, degradation_override=condition_override)["fused_logits"]
                    d3 = model.route_pgcd_cached(*cached, use_dynamic=False)["fused_logits"]
                    accumulators["D1_global_mean"].update(d1, labels)
                    accumulators["D2_sensor_severity"].update(d2, labels)
                    accumulators["D3_prior_only"].update(d3, labels)
                if variant in QUALITY_VARIANTS and condition.protocol in {"E1", "E2", "E3", "E4"}:
                    clean = _condition_batch(raw_batch, _clean_condition(), generator, mask_generator)
                    clean_step = _forward(model, cfg, clean, device)
                    quality.update(
                        diagnostics,
                        clean_step.model_output.diagnostics,
                        labels,
                        degraded,
                        condition,
                        cfg,
                    )
            primary = {"condition": condition.name, "protocol": condition.protocol, "family": condition.family,
                       "sensor": condition.sensor, "corruption": condition.corruption, "severity": condition.severity,
                       **accumulators["D0_dynamic"].result()}
            condition_rows.append(primary)
            for replacement, accumulator in accumulators.items():
                if variant == "c0" and replacement != "D0_dynamic":
                    continue
                dynamic_rows.append({"condition": condition.name, "protocol": condition.protocol,
                                     "family": condition.family, "sensor": condition.sensor,
                                     "severity": condition.severity, "replacement": replacement,
                                     **accumulator.result()})
    finally:
        shutdown_all_dataloaders(dataloaders)

    summary = _summary(condition_rows)
    quality_result = quality.result(condition_rows)
    weather_rows = [
        {"weather": name, **acc.result(), **weather_diagnostics.result(name)}
        for name, acc in weather_acc.items()
    ]
    scene_rows = [{"scene": name, **acc.result()} for name, acc in sorted(scene_acc.items())]
    summary["scene_macro_top1"] = _mean(scene_rows, "top1")
    summary["worst_scene_top1"] = min((row["top1"] for row in scene_rows), default=math.nan)
    summary["worst_scene"] = min(scene_rows, key=lambda row: row["top1"])["scene"] if scene_rows else ""
    dynamic_summary = _dynamic_summary(dynamic_rows)
    gradient_rows = _gradient_alignment(training_metrics)
    compute = _compute_cost(model, training_metrics, run_dir)
    payload = {
        "experiment": experiment,
        "ablation": variant.upper(),
        "protocol": "mmw_pgcd_fixed_eval_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_selection": selection,
        "summary": summary,
        "condition_metrics": condition_rows,
        "weather_metrics": weather_rows,
        "scene_metrics": scene_rows,
        "quality_diagnostics": quality_result,
        "dynamic_replacement": dynamic_rows,
        "dynamic_summary": dynamic_summary,
        "gradient_alignment": gradient_rows,
        "compute_cost": compute,
        "train_fit_replacement_statistics": train_stats["audit"],
        "uses_channel_or_path_data": False,
        "claim_eligible": False,
    }
    _write_json(run_dir / "metrics.json", payload)
    _write_csv(run_dir / "per_sensor_corruption.csv", [row for row in condition_rows if row["protocol"] == "E1"])
    _write_csv(run_dir / "per_weather_metrics.csv", weather_rows)
    _write_json(run_dir / "quality_diagnostics.json", quality_result)
    _write_csv(run_dir / "dynamic_replacement.csv", dynamic_rows)


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums = {key: 0.0 for key in ("top1", "top3", "top5", "within3", "beam_index_mae", "beam_loss")}

    def update(self, logits: torch.Tensor, labels: torch.Tensor, *, selector: torch.Tensor | None = None) -> None:
        scores, target = logits.float(), labels.long()
        if selector is not None:
            scores, target = scores[selector], target[selector]
        if not target.numel():
            return
        top5 = scores.topk(min(5, scores.shape[-1]), dim=-1).indices
        prediction = top5[:, 0]
        distance = (prediction - target).abs()
        distance = torch.minimum(distance, scores.shape[-1] - distance).float()
        values = {
            "top1": prediction.eq(target).float(),
            "top3": top5[:, : min(3, top5.shape[1])].eq(target[:, None]).any(dim=1).float(),
            "top5": top5.eq(target[:, None]).any(dim=1).float(),
            "within3": distance.le(3).float(),
            "beam_index_mae": distance,
            "beam_loss": F.cross_entropy(scores, target, reduction="none"),
        }
        self.count += int(target.numel())
        for key, value in values.items():
            self.sums[key] += float(value.sum().cpu())

    def result(self) -> dict[str, Any]:
        return {"sample_count": self.count, **{key: value / self.count if self.count else math.nan for key, value in self.sums.items()}}


class QualityAccumulator:
    def __init__(self, variant: str) -> None:
        self.variant = variant
        self.predicted: list[torch.Tensor] = []
        self.reliability: list[torch.Tensor] = []
        self.target: list[torch.Tensor] = []
        self.topology_loss: list[torch.Tensor] = []
        self.available: list[torch.Tensor] = []
        self.row_pearson: list[torch.Tensor] = []
        self.row_spearman: list[torch.Tensor] = []
        self.entropy: list[torch.Tensor] = []
        self.margin: list[torch.Tensor] = []
        self.rank_correct = 0.0
        self.rank_count = 0
        self.severity_reliability: dict[tuple[str, int], list[float]] = {}
        self.severity_drift: dict[tuple[str, int], list[float]] = {}

    def observe_reliability(
        self,
        diagnostics: Mapping[str, Any],
        degraded: EvalBatch,
        condition: Condition,
    ) -> None:
        reliability = diagnostics["pgcd_predicted_reliability"].detach().float().reshape(-1, 5, 4)
        if condition.protocol == "E0":
            for index, sensor in enumerate(MODALITIES):
                self.severity_reliability.setdefault((sensor, 0), []).append(float(reliability[:, :, index].mean().cpu()))
        elif condition.protocol == "E1" and condition.sensor in MODALITIES:
            index = MODALITIES.index(condition.sensor)
            self.severity_reliability.setdefault((condition.sensor, condition.severity), []).append(
                float(reliability[:, :, index].mean().cpu())
            )

    def update(
        self,
        diagnostics: Mapping[str, Any],
        clean_diagnostics: Mapping[str, Any],
        labels: torch.Tensor,
        degraded: EvalBatch,
        condition: Condition,
        cfg: Mapping[str, Any],
    ) -> None:
        predicted = diagnostics["pgcd_predicted_degradation"].detach().float()
        reliability = diagnostics["pgcd_predicted_reliability"].detach().float()
        corrupted_logits = diagnostics["pgcd_block_evidence_logits"]
        clean_logits = clean_diagnostics["pgcd_block_evidence_logits"]
        available = degraded.availability.to(predicted.device).reshape_as(predicted)
        severity = degraded.severity.to(predicted.device).reshape_as(predicted)
        loss_cfg = cfg["loss"]["u_mask_beam_jepa"]
        topology = loss_cfg.get("prototype_topology", {})
        target_result = pgcd_degradation_targets(
            clean_logits,
            corrupted_logits,
            labels,
            available,
            severity=severity,
            target_mode="none",
            topology_id=str(topology.get("id", "linear_index_v1")),
            topology_permutation=topology.get("permutation"),
            beam_label_sigma=float(loss_cfg.get("beam_label_sigma", 2.0)),
        )
        target = _quality_target(self.variant, severity, target_result)
        active = available & degraded.corrupted.to(predicted.device).reshape_as(predicted)
        if not bool(active.any()):
            return
        self.predicted.append(predicted.cpu())
        self.reliability.append(reliability.cpu())
        self.target.append(target.cpu())
        self.topology_loss.append(target_result.corrupted_block_loss.cpu())
        self.available.append(active.cpu())
        self.row_pearson.append(_row_correlation(predicted, target, active).cpu())
        self.row_spearman.append(_row_correlation(_rank_rows(predicted), _rank_rows(target), active).cpu())
        probability = torch.softmax(-target.masked_fill(~active, 1e4), dim=-1)
        self.entropy.append((-(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)).cpu())
        top2 = probability.topk(min(2, probability.shape[-1]), dim=-1).values
        self.margin.append((top2[:, 0] - top2[:, -1]).cpu())
        pair_target = target[:, :, None] - target[:, None, :]
        pair_pred = predicted[:, :, None] - predicted[:, None, :]
        pair_mask = active[:, :, None] & active[:, None, :] & pair_target.abs().gt(0.02)
        self.rank_correct += float((pair_pred.sign() == pair_target.sign()).masked_select(pair_mask).float().sum().cpu())
        self.rank_count += int(pair_mask.sum().cpu())
        if condition.sensor in MODALITIES:
            sensor_index = MODALITIES.index(condition.sensor)
            mask = active.reshape(len(labels), 5, 4)[:, :, sensor_index]
            drift = target_result.topology_drift.reshape(len(labels), 5, 4)[:, :, sensor_index]
            values = drift.masked_select(mask)
            if values.numel():
                self.severity_drift.setdefault((condition.sensor, condition.severity), []).append(float(values.mean().cpu()))

    def result(self, condition_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.predicted:
            return {}
        predicted = torch.cat(self.predicted)
        reliability = torch.cat(self.reliability)
        target = torch.cat(self.target)
        topology = torch.cat(self.topology_loss)
        active = torch.cat(self.available)
        pred_flat, target_flat = predicted[active], target[active]
        rel_flat, topology_flat = reliability[active], topology[active]
        severity_means = {
            f"{sensor}_L{severity}": statistics.fmean(values)
            for (sensor, severity), values in sorted(self.severity_reliability.items())
        }
        drift_means = {
            f"{sensor}_L{severity}": statistics.fmean(values)
            for (sensor, severity), values in sorted(self.severity_drift.items())
        }
        violations = []
        drift_violations = []
        for sensor in MODALITIES:
            sequence = [severity_means.get(f"{sensor}_L{level}") for level in range(4)]
            pairs = [(first, second) for first, second in zip(sequence, sequence[1:]) if first is not None and second is not None]
            violations.extend(float(second > first) for first, second in pairs)
            drift_sequence = [drift_means.get(f"{sensor}_L{level}") for level in (1, 2, 3)]
            drift_pairs = [
                (first, second)
                for first, second in zip(drift_sequence, drift_sequence[1:])
                if first is not None and second is not None
            ]
            drift_violations.extend(float(second < first) for first, second in drift_pairs)
        return {
            "predicted_quality_target_pearson": _correlation(pred_flat, target_flat),
            "predicted_quality_target_spearman": _correlation(_rank_1d(pred_flat), _rank_1d(target_flat)),
            "per_sample_across_block_pearson": float(torch.cat(self.row_pearson).mean()),
            "per_sample_across_block_spearman": float(torch.cat(self.row_spearman).mean()),
            "predicted_reliability_topology_loss_pearson": _correlation(rel_flat, topology_flat),
            "quality_regression_mae": float((pred_flat - target_flat).abs().mean()),
            "quality_ranking_accuracy": self.rank_correct / self.rank_count if self.rank_count else math.nan,
            "target_normalized_entropy": float(torch.cat(self.entropy).mean()),
            "target_top1_top2_margin": float(torch.cat(self.margin).mean()),
            "severity_monotonicity": 1.0 - statistics.fmean(violations) if violations else math.nan,
            "monotonicity_violation_rate": statistics.fmean(violations) if violations else math.nan,
            "reliability_means": severity_means,
            "prototype_drift_means": drift_means,
            "prototype_drift_monotonicity": 1.0 - statistics.fmean(drift_violations) if drift_violations else math.nan,
        }


class WeatherDiagnosticsAccumulator:
    def __init__(self) -> None:
        self.sums: dict[tuple[str, str, str], float] = {}
        self.counts: dict[tuple[str, str, str], int] = {}

    def update(
        self,
        weather: str,
        diagnostics: Mapping[str, Any],
        labels: torch.Tensor,
        *,
        selector: torch.Tensor | None = None,
    ) -> None:
        evidence = diagnostics["pgcd_block_evidence_logits"].detach().float().reshape(-1, 5, 4, 64)
        reliability = diagnostics["pgcd_predicted_reliability"].detach().float().reshape(-1, 5, 4)
        weights = diagnostics["pgcd_block_router_weights"].detach().float().reshape(-1, 5, 4)
        target = labels.long()
        if selector is not None:
            evidence, reliability, weights, target = evidence[selector], reliability[selector], weights[selector], target[selector]
        if not target.numel():
            return
        probability = torch.softmax(evidence, dim=-1)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1) / math.log(evidence.shape[-1])
        topology_loss = F.cross_entropy(
            evidence.reshape(-1, evidence.shape[-1]),
            target[:, None, None].expand(-1, 5, 4).reshape(-1),
            reduction="none",
        ).reshape(-1, 5, 4)
        for index, sensor in enumerate(MODALITIES):
            values = {
                "prototype_entropy": entropy[:, :, index],
                "topology_loss": topology_loss[:, :, index],
                "predicted_reliability": reliability[:, :, index],
                "fusion_weight": weights[:, :, index].sum(dim=1),
            }
            for metric, value in values.items():
                key = (weather, sensor, metric)
                self.sums[key] = self.sums.get(key, 0.0) + float(value.sum().cpu())
                self.counts[key] = self.counts.get(key, 0) + int(value.numel())

    def result(self, weather: str) -> dict[str, float]:
        output = {}
        for sensor in MODALITIES:
            for metric in ("prototype_entropy", "topology_loss", "predicted_reliability", "fusion_weight"):
                key = (weather, sensor, metric)
                output[f"{sensor}_{metric}"] = self.sums.get(key, 0.0) / max(self.counts.get(key, 0), 1)
        return output


def _fit_train_statistics(
    model, cfg: dict[str, Any], loader, generator: SensorDegradationGenerator, *, device: torch.device, max_batches: int
) -> dict[str, Any]:
    global_sum = torch.zeros(20, device=device)
    global_count = torch.zeros(20, device=device)
    grouped_sum = torch.zeros(5, 4, 5, device=device)
    grouped_count = torch.zeros(5, 4, 5, device=device)
    sample_count = 0
    with torch.no_grad():
        for step_index, raw_batch in enumerate(loader):
            if step_index >= max_batches:
                break
            assert_pgcd_channel_free(cfg, raw_batch)
            result = generator.apply_batch(raw_batch, training=True, epoch=0, step=step_index)
            degraded = EvalBatch(result.corrupted_batch, result.availability_mask, result.severity, result.corrupted_mask)
            forward = _forward(model, cfg, degraded, device)
            diagnostics = forward.model_output.diagnostics
            predicted = diagnostics["pgcd_predicted_degradation"].detach().float().reshape(-1, 5, 4)
            available = result.availability_mask.to(device)
            severity = result.severity.to(device)
            global_sum += (predicted * available).reshape(-1, 20).sum(dim=0)
            global_count += available.reshape(-1, 20).sum(dim=0)
            for level in range(5):
                selected = available & severity.eq(level)
                grouped_sum[:, :, level] += (predicted * selected).sum(dim=0)
                grouped_count[:, :, level] += selected.sum(dim=0)
            sample_count += predicted.shape[0]
    global_mean = global_sum / global_count.clamp_min(1)
    global_grid = global_mean.reshape(5, 4)
    grouped = grouped_sum / grouped_count.clamp_min(1)
    grouped = torch.where(grouped_count.gt(0), grouped, global_grid.unsqueeze(-1))
    return {
        "global_degradation": global_mean.cpu(),
        "sensor_severity_degradation": grouped.cpu(),
        "audit": {
            "source": "training_loader_only",
            "batches": max_batches,
            "samples": sample_count,
            "global_observation_count": int(global_count.sum().cpu()),
            "test_statistics_used": False,
        },
    }


def _conditions() -> list[Condition]:
    rows = [_clean_condition()]
    for sensor in DISPLAY_MODALITIES:
        for corruption in TRAIN_CORRUPTIONS[sensor]:
            for severity in range(1, 5):
                rows.append(
                    Condition(
                        f"E1_{sensor}_{corruption}_L{severity}", "E1", "seen", sensor, corruption, severity,
                        {"sensor": sensor, "severity": severity, "corruption_type": corruption, "mode": "single_seen"},
                    )
                )
    mixed = (
        ("mild_severe", ("image", "lidar"), (1, 3)),
        ("medium_medium", ("radar", "gps"), (2, 2)),
        ("severe_missing", ("lidar", "gps"), (3, 4)),
    )
    for name, sensors, severities in mixed:
        corruptions = tuple(TRAIN_CORRUPTIONS[sensor][0] for sensor in sensors)
        rows.append(Condition(f"E2_{name}", "E2", "mixed", "+".join(sensors), "+".join(corruptions), max(severities),
                              {"sensors": sensors, "severities": severities, "corruption_types": corruptions, "mode": name}))
    for sensor in DISPLAY_MODALITIES:
        rows.append(Condition(f"E3_{sensor}_one_step_stale", "E3", "stale", sensor, "one_step_stale", 2,
                              {"sensor": sensor, "severity": 2, "corruption_type": "one_step_stale", "mode": "stale"}))
        unseen = UNSEEN_CORRUPTIONS[sensor]
        rows.append(Condition(f"E4_{sensor}_{unseen}", "E4", "unseen", sensor, unseen, 3,
                              {"sensor": sensor, "severity": 3, "corruption_type": unseen, "mode": "unseen"}))
    for index, mask_type in enumerate(("full", "sparse_easy", "single_modality_burst2", "single_modality_missing",
                                       "latest_sync_missing", "two_modality_recent_async")):
        rows.append(Condition(f"E5_S{index}_{mask_type}", "E5", "original_missing", "all", mask_type, 4, mask_type=mask_type))
    return rows


def _clean_condition() -> Condition:
    return Condition("E0_clean", "E0", "clean", fixed={"mode": "clean", "sensors": ()})


def _condition_batch(
    raw_batch: Mapping[str, Any],
    condition: Condition,
    generator: SensorDegradationGenerator,
    mask_generator: TemporalBlockMaskGenerator,
) -> EvalBatch:
    if condition.mask_type is None:
        result = generator.apply_batch(raw_batch, training=False, variant_id=0, fixed=condition.fixed)
        batch = result.corrupted_batch
        apply_modality_temporal_mask_to_batch(batch, result.availability_mask, modalities=MODALITIES)
        return EvalBatch(batch, result.availability_mask, result.severity, result.corrupted_mask)
    sample_ids = sample_ids_from_batch(raw_batch)
    variants = [int.from_bytes(item.encode("utf-8")[:4].ljust(4, b"0"), "little") for item in sample_ids]
    generated = mask_generator(
        batch_size=len(sample_ids), num_modalities=4, num_timesteps=5, sample_ids=sample_ids,
        mask_type=condition.mask_type, seed=EVAL_SEED, training=False, variant_ids=variants,
        source_frame_ids=raw_batch.get("source_frame_ids"),
    )
    availability = generated["availability_mask"].permute(0, 2, 1).contiguous()
    batch = _clone_batch(raw_batch)
    apply_modality_temporal_mask_to_batch(batch, availability, modalities=MODALITIES)
    severity = (~availability).long() * 4
    return EvalBatch(batch, availability, severity, ~availability)


def _forward(model, cfg: Mapping[str, Any], degraded: EvalBatch, device: torch.device):
    missing_mask = degraded.availability.any(dim=1).to(device)
    with torch.no_grad():
        return run_model_step(
            model, cfg.get("experiment", {}).get("task", "fusion"), degraded.batch,
            seq_length=5, num_pred=1, device=device, extra_model_kwargs={"missing_mask": missing_mask},
        )


def _cached_tensors(diagnostics: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        diagnostics["pgcd_block_features"],
        diagnostics["pgcd_block_evidence_logits"],
        diagnostics["pgcd_block_availability"],
    )


def _condition_override(degraded: EvalBatch, grouped: torch.Tensor, device: torch.device) -> torch.Tensor:
    severity = degraded.severity.to(device)
    table = grouped.to(device)
    result = torch.empty_like(severity, dtype=torch.float32)
    for level in range(5):
        values = table[:, :, level].unsqueeze(0).expand(severity.shape[0], -1, -1)
        result = torch.where(severity.eq(level), values, result)
    return result.reshape(severity.shape[0], -1)


def _quality_target(variant: str, severity: torch.Tensor, result) -> torch.Tensor:
    drift = result.topology_drift.clamp(0, 1)
    task = result.task_degradation.clamp(0, 4) / 4.0
    if variant == "c1":
        target = severity.float() / 4.0
    elif variant in {"c3", "c4"}:
        target = drift
    elif variant == "c5":
        target = task
    else:
        target = 0.5 * drift + 0.5 * task
    return target.masked_fill(severity.eq(4), 1.0)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = next(row for row in rows if row["protocol"] == "E0")
    seen = [row for row in rows if row["protocol"] == "E1"]
    severe = [row for row in seen if int(row["severity"]) >= 3]
    mixed = [row for row in rows if row["protocol"] == "E2"]
    unseen = [row for row in rows if row["protocol"] == "E4"]
    missing = [row for row in rows if row["protocol"] == "E5" and "S0_" not in row["condition"]]
    stressed = seen + mixed + unseen + missing
    return {
        "clean_top1": clean["top1"],
        "corrupted_avg_top1": _mean(seen, "top1"),
        "severe_avg_top1": _mean(severe, "top1"),
        "mixed_avg_top1": _mean(mixed, "top1"),
        "unseen_avg_top1": _mean(unseen, "top1"),
        "original_missing_avg_top1": _mean(missing, "top1"),
        "worst_sensor_corruption_top1": min(row["top1"] for row in seen + unseen),
        "worst_condition": min(seen + unseen, key=lambda row: row["top1"])["condition"],
        "within3": _mean(stressed, "within3"),
        "beam_index_mae": _mean(stressed, "beam_index_mae"),
        "corrupted_retention": _mean(seen, "top1") / max(clean["top1"], 1e-12),
        "severity_robustness_auc": _severity_auc(rows),
    }


def _dynamic_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = {
        "Clean": lambda row: row["protocol"] == "E0",
        "Seen corruption": lambda row: row["protocol"] == "E1",
        "Severe corruption": lambda row: row["protocol"] == "E1" and int(row["severity"]) >= 3,
        "Two-sensor mixed": lambda row: row["protocol"] == "E2",
        "Unseen corruption": lambda row: row["protocol"] == "E4",
        "Original missing": lambda row: row["protocol"] == "E5" and "S0_" not in row["condition"],
    }
    output = []
    for category, selected in categories.items():
        by_replacement = {name: [row for row in rows if row["replacement"] == name and selected(row)] for name in REPLACEMENTS}
        if not by_replacement["D0_dynamic"]:
            continue
        values = {name: _mean(items, "top1") if items else math.nan for name, items in by_replacement.items()}
        output.append({"category": category, **values,
                       "dynamic_gain": values["D0_dynamic"] - values["D1_global_mean"],
                       "quality_gain_over_prior": values["D0_dynamic"] - values["D3_prior_only"]})
    return output


def _severity_auc(rows: list[dict[str, Any]]) -> dict[str, float]:
    clean = next(row["top1"] for row in rows if row["protocol"] == "E0")
    output = {}
    for sensor in DISPLAY_MODALITIES:
        values = [clean]
        for level in range(1, 5):
            selected = [row for row in rows if row["protocol"] == "E1" and row["sensor"] == sensor and row["severity"] == level]
            values.append(_mean(selected, "top1"))
        output[sensor] = sum((values[index] + values[index + 1]) * 0.5 for index in range(4)) / 4.0
    return output


def _gradient_alignment(training_metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    logs = training_metrics.get("epoch_logs", []) if isinstance(training_metrics, Mapping) else []
    if not isinstance(logs, list) or not logs:
        return []
    values = [float(row.get("pgcd_quality_beam_gradient_cosine", math.nan)) for row in logs]
    indices = sorted({0, len(logs) // 2, len(logs) - 1})
    labels = {indices[0]: "early", indices[-1]: "late"}
    if len(indices) == 3:
        labels[indices[1]] = "middle"
    return [{"phase": labels[index], "epoch": int(logs[index].get("epoch", index + 1)),
             "quality_beam_gradient_cosine": values[index],
             "quality_gradient_norm": logs[index].get("gradient/quality_estimator"),
             "router_gradient_norm": logs[index].get("gradient/router_fusion"),
             "backbone_gradient_norm": logs[index].get("gradient/backbone")}
            for index in indices]


def _compute_cost(model, training_metrics: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    startup = _read_json(run_dir / "startup_summary.json") if (run_dir / "startup_summary.json").is_file() else {}
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "quality_router_parameter_count": sum(parameter.numel() for parameter in model.pgcd_router.parameters()),
        "startup_summary": startup,
        "training_history_epochs": len(training_metrics.get("epoch_logs", [])) if isinstance(training_metrics, Mapping) else 0,
    }


def summarize(output: Path) -> None:
    payloads = [_read_json(output / name / "metrics.json") for name in EXPERIMENTS]
    combined = [{"experiment": item["experiment"], "ablation": item["ablation"], **item["summary"]} for item in payloads]
    per_sensor = [{"experiment": item["experiment"], "ablation": item["ablation"], **row}
                  for item in payloads for row in item["condition_metrics"] if row["protocol"] == "E1"]
    per_severity = _aggregate_table(per_sensor, ("experiment", "ablation", "severity"))
    weather = [{"experiment": item["experiment"], "ablation": item["ablation"], **row}
               for item in payloads for row in item["weather_metrics"]]
    unseen = [{"experiment": item["experiment"], "ablation": item["ablation"], **row}
              for item in payloads for row in item["condition_metrics"] if row["protocol"] == "E4"]
    missing = [{"experiment": item["experiment"], "ablation": item["ablation"], **row}
               for item in payloads for row in item["condition_metrics"] if row["protocol"] == "E5"]
    quality = [{"experiment": item["experiment"], "ablation": item["ablation"], **item["quality_diagnostics"]}
               for item in payloads if item["quality_diagnostics"]]
    dynamic = [{"experiment": item["experiment"], "ablation": item["ablation"], **row}
               for item in payloads for row in item["dynamic_summary"]]
    gradient = [{"experiment": item["experiment"], "ablation": item["ablation"], **row}
                for item in payloads for row in item["gradient_alignment"]]
    compute = [{"experiment": item["experiment"], "ablation": item["ablation"], **item["compute_cost"]} for item in payloads]
    for path, rows in (
        ("combined_metrics.csv", combined), ("per_sensor_corruption.csv", per_sensor),
        ("per_severity_metrics.csv", per_severity), ("per_weather_metrics.csv", weather),
        ("unseen_corruption_metrics.csv", unseen), ("original_missing_metrics.csv", missing),
        ("quality_diagnostics.csv", quality), ("dynamic_replacement.csv", dynamic),
        ("gradient_alignment.csv", gradient), ("compute_cost.csv", compute),
    ):
        _write_csv(output / path, rows)
    gates = _quick_gates(combined, quality, dynamic)
    _write_json(output / "combined_metrics.json", {"experiments": combined, "quick_gates": gates, "claim_eligible": False})
    (output / "pgcd_comparison.md").write_text(_comparison_markdown(combined, weather, per_sensor, quality, dynamic, gates), encoding="utf-8")


def _quick_gates(combined, quality, dynamic) -> dict[str, Any]:
    c0, c7 = combined[0], combined[7]
    c7_dynamic = {row["category"]: row for row in dynamic if row["ablation"] == "C7"}
    severe_gain = c7_dynamic.get("Severe corruption", {}).get("dynamic_gain", -math.inf)
    mixed_gain = c7_dynamic.get("Two-sensor mixed", {}).get("dynamic_gain", -math.inf)
    c7_quality = next((row for row in quality if row["ablation"] == "C7"), {})
    gate1 = statistics.fmean((severe_gain, mixed_gain)) >= 0.005
    gate2 = float(c7_quality.get("predicted_quality_target_spearman", -math.inf)) > 0.3
    gate3 = (
        max(c7["corrupted_avg_top1"] - c0["corrupted_avg_top1"], c7["severe_avg_top1"] - c0["severe_avg_top1"],
            c7["worst_sensor_corruption_top1"] - c0["worst_sensor_corruption_top1"]) > 0
        and c7["clean_top1"] - c0["clean_top1"] >= -0.005
    )
    baselines = [combined[index]["unseen_avg_top1"] for index in (0, 1, 2)]
    gate4 = c7["unseen_avg_top1"] > max(baselines)
    passed = sum((gate1, gate2, gate3, gate4))
    return {"gate1_dynamic_value": gate1, "gate2_quality_predictable": gate2,
            "gate3_robustness": gate3, "gate4_unseen_generalization": gate4,
            "passed": passed, "required": 3, "continue_direction": passed >= 3}


def _comparison_markdown(combined, weather, per_sensor, quality, dynamic, gates) -> str:
    lines = [
        "# PGCD 快速搜索比较", "",
        "> 单 seed、inner/development、claim-ineligible；未运行 multi-seed 或 outer test。", "",
        "| 方法 | Clean | Corrupt Avg | Severe Avg | Mixed Avg | Unseen Avg | Missing Avg | Worst | Within-3 | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in combined:
        lines.append(f"| {row['ablation']} | {row['clean_top1']:.4f} | {row['corrupted_avg_top1']:.4f} | "
                     f"{row['severe_avg_top1']:.4f} | {row['mixed_avg_top1']:.4f} | {row['unseen_avg_top1']:.4f} | "
                     f"{row['original_missing_avg_top1']:.4f} | {row['worst_sensor_corruption_top1']:.4f} | "
                     f"{row['within3']:.4f} | {row['beam_index_mae']:.4f} |")
    lines.extend(["", "| 方法 | Dynamic | Global Mean | Prior Only | Dynamic Gain | Quality Spearman | Monotonic Violation |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for ablation in ABLATIONS[1:]:
        row = next((item for item in dynamic if item["ablation"] == ablation and item["category"] == "Severe corruption"), {})
        q = next((item for item in quality if item["ablation"] == ablation), {})
        lines.append(f"| {ablation} | {_fmt(row.get('D0_dynamic'))} | {_fmt(row.get('D1_global_mean'))} | "
                     f"{_fmt(row.get('D3_prior_only'))} | {_fmt(row.get('dynamic_gain'))} | "
                     f"{_fmt(q.get('predicted_quality_target_spearman'))} | {_fmt(q.get('monotonicity_violation_rate'))} |")
    lines.extend(["", "| 方法 | Sunny | Rainy | Foggy | Worst Weather |", "| --- | ---: | ---: | ---: | ---: |"])
    for ablation in ("C0", "C7"):
        values = {row["weather"]: row["top1"] for row in weather if row["ablation"] == ablation}
        listed = [values.get(name, math.nan) for name in ("sunny", "rainy", "foggy")]
        lines.append(f"| {ablation} | {_fmt(listed[0])} | {_fmt(listed[1])} | {_fmt(listed[2])} | {_fmt(min(listed))} |")
    lines.extend(["", "| 方法 | Image | LiDAR | Radar | GPS | Worst Sensor |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for ablation in ("C0", "C7"):
        values = {sensor: _mean([row for row in per_sensor if row["ablation"] == ablation and row["sensor"] == sensor], "top1")
                  for sensor in DISPLAY_MODALITIES}
        worst = min(values, key=values.get)
        lines.append(f"| {ablation} | {_fmt(values['image'])} | {_fmt(values['lidar'])} | {_fmt(values['radar'])} | "
                     f"{_fmt(values['gps'])} | {worst} ({values[worst]:.4f}) |")
    lines.extend(["", "## Quick Gates", "",
                  f"C7 通过 {gates['passed']}/4，要求至少 {gates['required']}/4："
                  f"{'方向可继续' if gates['continue_direction'] else '方向失败'}。",
                  f"Gate 1 动态价值：{gates['gate1_dynamic_value']}；Gate 2 质量可预测：{gates['gate2_quality_predictable']}；"
                  f"Gate 3 鲁棒性：{gates['gate3_robustness']}；Gate 4 未见退化：{gates['gate4_unseen_generalization']}。", "",
                  "本轮到此停止，不自动启动 multi-seed、outer test 或下一轮实验。", ""])
    return "\n".join(lines)


def _aggregate_table(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    metrics = ("top1", "top3", "top5", "within3", "beam_index_mae", "beam_loss")
    return [{**dict(zip(keys, key, strict=True)), **{metric: _mean(values, metric) for metric in metrics}}
            for key, values in sorted(groups.items())]


def _training_done(job: Mapping[str, Any]) -> bool:
    run_dir = Path(str(job.get("run_dir", "")))
    status = run_dir / "run_status.json"
    return (run_dir / "checkpoints/best.pth").is_file() and status.is_file() and _read_json(status).get("state") == "complete"


def _weather_rows(batch: Mapping[str, Any], size: int) -> list[str]:
    rows = metadata_rows_from_batch(batch.get("metadata"))
    result = [str(row.get("condition", "")).strip().lower() for row in rows]
    return result if len(result) == size else [""] * size


def _scene_rows(batch: Mapping[str, Any], size: int) -> list[str]:
    rows = metadata_rows_from_batch(batch.get("metadata"))
    result = [
        "/".join(filter(None, (str(row.get("condition", "")), str(row.get("town", "")), str(row.get("sensor_scenario", "")))))
        for row in rows
    ]
    return result if len(result) == size else [""] * size


def _rank_rows(values: torch.Tensor) -> torch.Tensor:
    ranks = torch.empty_like(values)
    order = values.argsort(dim=-1)
    base = torch.arange(values.shape[-1], device=values.device, dtype=values.dtype).expand_as(values)
    return ranks.scatter(-1, order, base)


def _rank_1d(values: torch.Tensor) -> torch.Tensor:
    ranks = torch.empty_like(values)
    return ranks.scatter(0, values.argsort(), torch.arange(values.numel(), device=values.device, dtype=values.dtype))


def _row_correlation(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    count = weight.sum(dim=-1).clamp_min(1)
    first_centered = (first - (first * weight).sum(dim=-1, keepdim=True) / count[:, None]) * weight
    second_centered = (second - (second * weight).sum(dim=-1, keepdim=True) / count[:, None]) * weight
    denominator = first_centered.square().sum(dim=-1).sqrt() * second_centered.square().sum(dim=-1).sqrt()
    return torch.where(denominator.gt(0), (first_centered * second_centered).sum(dim=-1) / denominator.clamp_min(1e-12), 0)


def _correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.numel() < 2:
        return math.nan
    first = first.float() - first.float().mean()
    second = second.float() - second.float().mean()
    denominator = first.square().sum().sqrt() * second.square().sum().sqrt()
    return float((first * second).sum() / denominator.clamp_min(1e-12)) if float(denominator) > 0 else 0.0


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values) if values else math.nan


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _clone_batch(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_batch(item) for item in value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
