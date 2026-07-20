#!/usr/bin/env python3
"""Evaluate and summarize the four MMW PCER quick-validation checkpoints."""

from __future__ import annotations

import argparse
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
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from kd_sensing.config import load_config
from kd_sensing.data.temporal_block_mask import TemporalBlockMaskGenerator
from kd_sensing.data.temporal_missing import apply_modality_temporal_mask_to_batch
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.evaluation_pass_runtime import metadata_rows_from_batch, sample_ids_from_batch
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.runtime import prepare_task_labels, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.losses.pcer_temporal_fusion import counterfactual_router_targets
from kd_sensing.utils.checkpoint import load_model_state, load_torch_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/quick_pcer_validation"
MODALITIES = ("image", "radar", "gps", "lidar")
EVAL_SEED = 20260720
EXPERIMENTS = (
    "qv_a0_proto_static",
    "qv_a1_proto_old_router",
    "qv_a2_proto_consistency_static",
    "qv_a3_pcer_full",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--worker", choices=EXPERIMENTS)
    parser.add_argument("--gpus", default="4,5,6,7")
    args = parser.parse_args()
    output_root = _path(args.output_root)
    if args.worker:
        evaluate_experiment(output_root, args.worker)
        return 0
    if not args.all:
        parser.error("select --all or --worker")
    gpus = tuple(int(item.strip()) for item in args.gpus.split(",") if item.strip())
    if gpus != (4, 5, 6, 7):
        parser.error("PCER evaluation requires --gpus 4,5,6,7")
    return evaluate_all(output_root, gpus)


def evaluate_all(output_root: Path, gpus: tuple[int, ...]) -> int:
    training = _read_json(output_root / "training_manifest.json")
    jobs = {job["experiment"]: job for job in training["jobs"]}
    incomplete = [name for name in EXPERIMENTS if jobs.get(name, {}).get("status") != "done"]
    if incomplete:
        raise RuntimeError(f"PCER evaluation requires completed training jobs: {incomplete}")
    running = []
    evaluation_jobs = []
    for name, gpu in zip(EXPERIMENTS, gpus, strict=True):
        log_path = output_root / name / "eval.log"
        handle = log_path.open("a", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "--output-root", str(output_root), "--worker", name]
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "OMP_NUM_THREADS": "4",
                "PYTHONUNBUFFERED": "1",
            }
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        job = {"experiment": name, "gpu": gpu, "pid": process.pid, "status": "running", "start_time": _now()}
        evaluation_jobs.append(job)
        running.append((process, handle, job))
    manifest_path = output_root / "evaluation_manifest.json"
    _write_json(manifest_path, {"jobs": evaluation_jobs, "status": "running"})
    while running:
        for process, handle, job in list(running):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            metrics = output_root / job["experiment"] / "metrics.json"
            job.update(
                status="done" if code == 0 and metrics.is_file() else "failed",
                return_code=int(code),
                end_time=_now(),
            )
            running.remove((process, handle, job))
            _write_json(manifest_path, {"jobs": evaluation_jobs, "status": "running"})
        if running:
            time.sleep(5.0)
    status = "complete" if all(job["status"] == "done" for job in evaluation_jobs) else "failed"
    _write_json(manifest_path, {"jobs": evaluation_jobs, "status": status, "completed_at": _now()})
    if status != "complete":
        return 1
    summarize(output_root)
    return 0


def evaluate_experiment(output_root: Path, experiment: str) -> None:
    run_dir = output_root / experiment
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "checkpoints/best.pth"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing PCER config or best checkpoint for {experiment}.")
    cfg = load_config(config_path)
    checkpoint = load_torch_payload(checkpoint_path, map_location="cpu")
    selection = checkpoint.get("selection", {}) if isinstance(checkpoint, dict) else {}
    if checkpoint.get("checkpoint_role") != "validation_best" or selection.get("metric") != "validation_loss":
        raise ValueError(f"{experiment} evaluator requires a validation_best checkpoint.")
    cfg.setdefault("training", {})["final_test"] = {"enabled": True}
    cfg.setdefault("temporal_missing", {})["enabled"] = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]["primary"]).to(device)
    load_model_state(checkpoint_path, model, role="pcer-fixed-mask-evaluation", map_location=device, strict=True)
    model.eval()
    dataloaders = build_dataloaders(cfg)
    loader = dataloaders["test"]
    accumulators = {name: MetricAccumulator() for name in _condition_names()}
    s5_pairs: dict[str, MetricAccumulator] = {}
    router_accumulators = {name: RouterAccumulator(MODALITIES, 5) for name in _condition_names()}
    generator = TemporalBlockMaskGenerator(EVAL_SEED)
    sample_offset = 0
    try:
        with torch.no_grad():
            for raw_batch in loader:
                sample_ids = sample_ids_from_batch(raw_batch)
                batch_size = len(sample_ids)
                if batch_size <= 0:
                    raise ValueError("PCER fixed-mask evaluation requires stable sample ids.")
                beam_powers = _future_beam_power(raw_batch)
                batch_full_weights = None
                for condition in _conditions(sample_ids, sample_offset, generator):
                    batch = _clone_batch(raw_batch)
                    mask_tm = condition["mask"].permute(0, 2, 1).contiguous()
                    apply_modality_temporal_mask_to_batch(batch, mask_tm, modalities=MODALITIES)
                    step = run_model_step(
                        model,
                        cfg.get("experiment", {}).get("task", "fusion"),
                        batch,
                        seq_length=5,
                        num_pred=1,
                        device=device,
                        extra_model_kwargs={"missing_mask": batch["available_modalities"].to(device=device)},
                    )
                    logits = step.logits[:, -1]
                    labels = prepare_task_labels(step.batch, num_pred=1, device=device)[:, -1]
                    accumulators[condition["name"]].update(logits, labels, beam_powers)
                    if condition["family"] == "S5":
                        for pair_name in sorted(set(condition["groups"])):
                            selector = torch.tensor(
                                [value == pair_name for value in condition["groups"]], device=logits.device
                            )
                            s5_pairs.setdefault(pair_name, MetricAccumulator()).update(
                                logits, labels, beam_powers, selector=selector
                            )
                    diagnostics = step.model_output.diagnostics
                    router = diagnostics.get("pcer_block_router_weights")
                    if torch.is_tensor(router) and condition["family"] == "S0":
                        batch_full_weights = router.detach()
                    router_accumulators[condition["name"]].update(
                        diagnostics,
                        labels,
                        full_weights=batch_full_weights,
                        cfg=cfg,
                    )
                sample_offset += batch_size
    finally:
        shutdown_all_dataloaders(dataloaders)

    rows = []
    for name in _condition_names():
        family = name.split("_", 1)[0]
        rows.append({"condition": name, "family": family, **accumulators[name].result()})
    for pair_name, accumulator in sorted(s5_pairs.items()):
        rows.append({"condition": f"S5_pair_{pair_name}", "family": "S5_pair", **accumulator.result()})
    rows.extend(_aggregate_rows(rows))
    summary = _experiment_summary(rows, validation_loss=float(selection["value"]), validation_epoch=int(selection["epoch"]))
    router_rows = []
    for name in _condition_names():
        result = router_accumulators[name].result()
        if result:
            router_rows.append({"experiment": experiment, "condition": name, **result})
    _write_csv(run_dir / "per_mask_metrics.csv", rows)
    _write_json(
        run_dir / "metrics.json",
        {
            "experiment": experiment,
            "protocol": "mmw_quick_pcer_validation_v1",
            "checkpoint": str(checkpoint_path),
            "checkpoint_selection": selection,
            "summary": summary,
            "per_mask_metrics": rows,
            "router_diagnostics": router_rows,
            "claim_eligible": False,
        },
    )
    _write_json(run_dir / "router_diagnostics.json", router_rows)


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums = {
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
            "within3": 0.0,
            "beam_index_mae": 0.0,
            "beam_loss": 0.0,
            "normalized_gain": 0.0,
            "gain_loss_db": 0.0,
        }

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        beam_powers: torch.Tensor,
        *,
        selector: torch.Tensor | None = None,
    ) -> None:
        scores = logits.float()
        target = labels.long()
        powers = beam_powers.to(device=scores.device, dtype=torch.float32)
        if selector is not None:
            scores, target, powers = scores[selector], target[selector], powers[selector]
        if not target.numel():
            return
        count = int(target.numel())
        top5 = scores.topk(min(5, scores.shape[-1]), dim=-1).indices
        prediction = top5[:, 0]
        distance = (prediction - target).abs()
        distance = torch.minimum(distance, scores.shape[-1] - distance).float()
        best_power = powers.amax(dim=-1)
        predicted_power = powers.gather(1, prediction.unsqueeze(1)).squeeze(1)
        gain = (predicted_power / best_power).clamp(min=torch.finfo(torch.float32).tiny, max=1.0)
        values = {
            "top1": prediction.eq(target).float(),
            "top3": top5[:, : min(3, top5.shape[1])].eq(target.unsqueeze(1)).any(dim=1).float(),
            "top5": top5.eq(target.unsqueeze(1)).any(dim=1).float(),
            "within3": distance.le(3).float(),
            "beam_index_mae": distance,
            "beam_loss": F.cross_entropy(scores, target, reduction="none"),
            "normalized_gain": gain,
            "gain_loss_db": -10.0 * torch.log10(gain),
        }
        self.count += count
        for key, value in values.items():
            self.sums[key] += float(value.sum().detach().cpu().item())

    def result(self) -> dict[str, Any]:
        return {
            "sample_count": self.count,
            **{key: value / self.count if self.count else math.nan for key, value in self.sums.items()},
        }


class RouterAccumulator:
    def __init__(self, modalities: tuple[str, ...], timesteps: int) -> None:
        self.modalities = modalities
        self.timesteps = int(timesteps)
        self.weights: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.availability: list[torch.Tensor] = []
        self.full_deltas: list[torch.Tensor] = []

    def update(
        self,
        diagnostics: dict[str, Any],
        labels: torch.Tensor,
        *,
        full_weights: torch.Tensor | None,
        cfg: dict[str, Any],
    ) -> None:
        weights = diagnostics.get("pcer_block_router_weights")
        availability = diagnostics.get("pcer_block_availability")
        if torch.is_tensor(weights) and torch.is_tensor(availability):
            self.weights.append(weights.detach().float().cpu())
            self.availability.append(availability.detach().bool().cpu())
            if full_weights is not None and tuple(full_weights.shape) == tuple(weights.shape):
                self.full_deltas.append((weights.detach() - full_weights).abs().float().cpu())
            evidence = diagnostics.get("pcer_block_evidence_logits")
            if torch.is_tensor(evidence) and diagnostics.get("pcer_mode") == "counterfactual_router":
                loss_cfg = cfg["loss"]["u_mask_beam_jepa"]
                topology = loss_cfg.get("prototype_topology", {})
                pcer = loss_cfg["pcer"]
                target, _ = counterfactual_router_targets(
                    evidence,
                    availability,
                    labels,
                    beam_label_sigma=float(loss_cfg.get("beam_label_sigma", 2.0)),
                    circular=bool(loss_cfg.get("prototype_target_circular", True)),
                    topology_id=str(topology.get("id", "")) or None,
                    topology_permutation=topology.get("permutation"),
                    contribution_temperature=float(pcer.get("contribution_temperature", 0.5)),
                    contribution_clip=pcer.get("contribution_clip"),
                )
                self.targets.append(target.detach().float().cpu())
            return
        modality_weights = diagnostics.get("supervised_router_gate_weights")
        if torch.is_tensor(modality_weights):
            self.weights.append(modality_weights.detach().float().cpu())
            self.availability.append(modality_weights.detach().gt(0).cpu())

    def result(self) -> dict[str, Any]:
        if not self.weights:
            return {}
        weights = torch.cat(self.weights)
        availability = torch.cat(self.availability)
        tiny = torch.finfo(weights.dtype).tiny
        result = {
            "router_type": "block" if weights.shape[1] == self.timesteps * len(self.modalities) else "modality",
            "weight_entropy": float((-(weights * weights.clamp_min(tiny).log()).sum(dim=-1)).mean()),
            "sample_weight_std": float(weights.std(dim=0, unbiased=False).mean()),
            "mean_absolute_dynamic_deviation": float((weights - weights.mean(dim=0)).abs().mean()),
            "missing_weight_max": float(weights.masked_select(~availability).max()) if bool((~availability).any()) else 0.0,
            "mask_weight_change_l1": float(torch.cat(self.full_deltas).mean()) if self.full_deltas else 0.0,
        }
        if result["router_type"] == "block":
            cell = weights.reshape(-1, self.timesteps, len(self.modalities))
            for index, name in enumerate(self.modalities):
                result[f"mean_weight_modality_{name}"] = float(cell[:, :, index].sum(dim=1).mean())
            for index in range(self.timesteps):
                result[f"mean_weight_time_{index}"] = float(cell[:, index].sum(dim=1).mean())
        else:
            for index, name in enumerate(self.modalities):
                result[f"mean_weight_modality_{name}"] = float(weights[:, index].mean())
        if self.targets:
            targets = torch.cat(self.targets)
            pearson = _row_pearson(weights, targets, availability)
            spearman = _row_pearson(_rank_rows(weights), _rank_rows(targets), availability)
            result.update(
                {
                    "target_pearson": float(pearson.mean()),
                    "target_spearman": float(spearman.mean()),
                    "target_top1_agreement": float(weights.argmax(dim=-1).eq(targets.argmax(dim=-1)).float().mean()),
                }
            )
        return result


def _conditions(
    sample_ids: list[str],
    offset: int,
    generator: TemporalBlockMaskGenerator,
) -> list[dict[str, Any]]:
    batch = len(sample_ids)
    indices = [offset + index for index in range(batch)]
    specs: list[tuple[str, str, Any]] = [("S0_full", "full", 0)]
    specs.extend((f"S1_sparse_easy_v{variant}", "sparse_easy", variant) for variant in range(3))
    specs.append(("S2_single_modality_burst2", "single_modality_burst2", [index % 16 for index in indices]))
    specs.extend(
        (f"S3_missing_{name}", "single_modality_missing", modality)
        for modality, name in enumerate(MODALITIES)
    )
    specs.append(("S4_latest_sync_missing", "latest_sync_missing", 0))
    specs.append(("S5_two_modality_recent_async", "two_modality_recent_async", [index % 12 for index in indices]))
    result = []
    for name, kind, variants in specs:
        generated = generator(
            batch_size=batch,
            num_modalities=len(MODALITIES),
            num_timesteps=5,
            sample_ids=sample_ids,
            mask_type=kind,
            severity=None,
            seed=EVAL_SEED,
            training=False,
            variant_ids=variants,
        )
        family = name.split("_", 1)[0]
        groups = [
            "-".join(MODALITIES[index] for index in metadata.get("modalities", [])) or "all"
            for metadata in generated["mask_metadata"]
        ]
        result.append({"name": name, "family": family, "mask": generated["availability_mask"], "groups": groups})
    return result


def _condition_names() -> list[str]:
    return [
        "S0_full",
        "S1_sparse_easy_v0",
        "S1_sparse_easy_v1",
        "S1_sparse_easy_v2",
        "S2_single_modality_burst2",
        *(f"S3_missing_{name}" for name in MODALITIES),
        "S4_latest_sync_missing",
        "S5_two_modality_recent_async",
    ]


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {row["condition"]: row for row in rows}
    s1 = [lookup[f"S1_sparse_easy_v{index}"] for index in range(3)]
    s3 = [lookup[f"S3_missing_{name}"] for name in MODALITIES]
    s5 = [row for row in rows if row["family"] == "S5_pair"]
    return [
        _macro_row("S1_macro", "S1_macro", s1),
        _macro_row("S3_macro", "S3_macro", s3),
        _macro_row("S5_macro", "S5_macro", s5),
    ]


def _macro_row(condition: str, family: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("top1", "top3", "top5", "within3", "beam_index_mae", "beam_loss", "normalized_gain", "gain_loss_db")
    return {
        "condition": condition,
        "family": family,
        "sample_count": sum(int(row["sample_count"]) for row in rows),
        **{key: statistics.fmean(float(row[key]) for row in rows) for key in metrics},
    }


def _experiment_summary(rows: list[dict[str, Any]], *, validation_loss: float, validation_epoch: int) -> dict[str, Any]:
    lookup = {row["condition"]: row for row in rows}
    scenario_rows = [
        lookup["S1_macro"],
        lookup["S2_single_modality_burst2"],
        lookup["S3_macro"],
        lookup["S4_latest_sync_missing"],
        lookup["S5_macro"],
    ]
    hard_rows = [lookup["S3_macro"], lookup["S4_latest_sync_missing"], lookup["S5_macro"]]
    fixed_subscenarios = [
        row
        for row in rows
        if row["family"] in {"S1", "S2", "S3", "S4", "S5_pair"}
    ]
    s3_rows = [lookup[f"S3_missing_{name}"] for name in MODALITIES]
    s5_rows = [row for row in rows if row["family"] == "S5_pair"]
    full = lookup["S0_full"]
    summary = {
        "validation_loss": validation_loss,
        "validation_epoch": validation_epoch,
        "full_top1": full["top1"],
        "full_top3": full["top3"],
        "full_top5": full["top5"],
        "masked_macro_top1": statistics.fmean(row["top1"] for row in scenario_rows),
        "hard_avg_top1": statistics.fmean(row["top1"] for row in hard_rows),
        "worst_top1": min(row["top1"] for row in fixed_subscenarios),
        "masked_within3": statistics.fmean(row["within3"] for row in scenario_rows),
        "masked_beam_index_mae": statistics.fmean(row["beam_index_mae"] for row in scenario_rows),
        "masked_normalized_gain": statistics.fmean(row["normalized_gain"] for row in scenario_rows),
        "s1_top1": lookup["S1_macro"]["top1"],
        "s2_top1": lookup["S2_single_modality_burst2"]["top1"],
        "s3_macro_top1": lookup["S3_macro"]["top1"],
        "s3_worst_top1": min(row["top1"] for row in s3_rows),
        "s3_worst_modality": min(s3_rows, key=lambda row: row["top1"])["condition"].removeprefix("S3_missing_"),
        "s4_top1": lookup["S4_latest_sync_missing"]["top1"],
        "s5_top1": lookup["S5_macro"]["top1"],
        "s5_worst_top1": min(row["top1"] for row in s5_rows),
        "s5_worst_pair": min(s5_rows, key=lambda row: row["top1"])["condition"].removeprefix("S5_pair_"),
    }
    for key in ("s1", "s2", "s3_macro", "s4", "s5"):
        summary[f"retention_{key}"] = summary[f"{key}_top1"] / max(summary["full_top1"], 1e-12)
    return summary


def summarize(output_root: Path) -> None:
    payloads = {name: _read_json(output_root / name / "metrics.json") for name in EXPERIMENTS}
    summaries = {name: payload["summary"] for name, payload in payloads.items()}
    a0 = summaries[EXPERIMENTS[0]]
    for summary in summaries.values():
        summary["full_drop_vs_a0"] = summary["full_top1"] - a0["full_top1"]
    combined_rows = [{"experiment": name, **summaries[name]} for name in EXPERIMENTS]
    _write_csv(output_root / "combined_metrics.csv", combined_rows)
    _write_json(output_root / "combined_metrics.json", {"experiments": combined_rows, "claim_eligible": False})
    router_rows = [row for payload in payloads.values() for row in payload.get("router_diagnostics", [])]
    _write_csv(output_root / "router_diagnostics.csv", router_rows)
    comparison = _comparison_markdown(summaries, router_rows)
    (output_root / "comparison.md").write_text(comparison, encoding="utf-8")


def _comparison_markdown(summaries: dict[str, dict[str, Any]], router_rows: list[dict[str, Any]]) -> str:
    names = dict(zip(EXPERIMENTS, ("A0 prototype static", "A1 old router", "A2 proto consistency static", "A3 full PCER"), strict=True))
    a0, a1, a2, a3 = (summaries[name] for name in EXPERIMENTS)
    a3_full_router = next(
        (row for row in router_rows if row.get("experiment") == EXPERIMENTS[3] and row.get("condition") == "S0_full"),
        {},
    )
    condition1 = a2["masked_macro_top1"] - a0["masked_macro_top1"] >= 0.005 and a2["full_drop_vs_a0"] >= -0.005
    condition2 = (
        (a3["hard_avg_top1"] - a2["hard_avg_top1"] >= 0.005 or a3["worst_top1"] - a2["worst_top1"] >= 0.005)
        and a3["hard_avg_top1"] - a1["hard_avg_top1"] >= 0.005
    )
    condition3 = (
        float(a3_full_router.get("target_pearson", 0.0)) > 0.1
        and float(a3_full_router.get("sample_weight_std", 0.0)) > 0.005
        and float(a3_full_router.get("missing_weight_max", 1.0)) <= 1e-7
    )
    passed = sum((condition1, condition2, condition3))
    gains = {
        "S1": max(a2["s1_top1"], a3["s1_top1"]) - a0["s1_top1"],
        "S2": max(a2["s2_top1"], a3["s2_top1"]) - a0["s2_top1"],
        "S3": max(a2["s3_macro_top1"], a3["s3_macro_top1"]) - a0["s3_macro_top1"],
        "S4": max(a2["s4_top1"], a3["s4_top1"]) - a0["s4_top1"],
        "S5": max(a2["s5_top1"], a3["s5_top1"]) - a0["s5_top1"],
    }
    best_mask = max(gains, key=gains.get)
    failed_mask = min(gains, key=gains.get)
    lines = [
        "# PCER 快速验证比较",
        "",
        "> 单 seed、inner/development、claim-ineligible；未使用冻结 outer evidence。",
        "",
        "| 方法 | Full Top1 | Masked Avg Top1 | Hard Avg Top1 | Worst Top1 | Within-3 | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in EXPERIMENTS:
        row = summaries[key]
        lines.append(
            f"| {names[key]} | {row['full_top1']:.4f} | {row['masked_macro_top1']:.4f} | "
            f"{row['hard_avg_top1']:.4f} | {row['worst_top1']:.4f} | {row['masked_within3']:.4f} | "
            f"{row['masked_beam_index_mae']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| 方法 | S1 | S2 | S3 macro | S3 worst | S4 | S5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key in EXPERIMENTS:
        row = summaries[key]
        lines.append(
            f"| {names[key]} | {row['s1_top1']:.4f} | {row['s2_top1']:.4f} | "
            f"{row['s3_macro_top1']:.4f} | {row['s3_worst_top1']:.4f} | {row['s4_top1']:.4f} | {row['s5_top1']:.4f} |"
        )
    questions = [
        f"1. A1 与 A0 是否接近：{'是' if abs(a1['masked_macro_top1'] - a0['masked_macro_top1']) < 0.005 else '否'}。",
        f"2. A2 masked macro 是否超过 A0：{'是' if a2['masked_macro_top1'] > a0['masked_macro_top1'] else '否'}，差值 {(a2['masked_macro_top1'] - a0['masked_macro_top1']) * 100:.2f} pp。",
        f"3. A2 是否改善 S4：{'是' if a2['s4_top1'] > a0['s4_top1'] else '否'}，差值 {(a2['s4_top1'] - a0['s4_top1']) * 100:.2f} pp。",
        f"4. A3 相对 A2：hard average {'提高' if a3['hard_avg_top1'] > a2['hard_avg_top1'] else '下降'} {(a3['hard_avg_top1'] - a2['hard_avg_top1']) * 100:+.2f} pp；worst-case {'提高' if a3['worst_top1'] > a2['worst_top1'] else '下降'} {(a3['worst_top1'] - a2['worst_top1']) * 100:+.2f} pp。",
        f"5. A3 是否超过旧 Router A1：hard average {'是' if a3['hard_avg_top1'] > a1['hard_avg_top1'] else '否'}（{(a3['hard_avg_top1'] - a1['hard_avg_top1']) * 100:+.2f} pp），worst-case {'是' if a3['worst_top1'] > a1['worst_top1'] else '否'}（{(a3['worst_top1'] - a1['worst_top1']) * 100:+.2f} pp）。",
        f"6. A2/A3 full 性能是否明显下降：{'是' if min(a2['full_drop_vs_a0'], a3['full_drop_vs_a0']) < -0.005 else '否'}。",
        f"7. A3 Router 是否形成有效动态路由：{'是' if condition3 else '否'}；full sample weight std={float(a3_full_router.get('sample_weight_std', math.nan)):.6f}，mean absolute dynamic deviation={float(a3_full_router.get('mean_absolute_dynamic_deviation', math.nan)):.6f}，missing max={float(a3_full_router.get('missing_weight_max', math.nan)):.2e}。",
        f"8. Router 与反事实 target 是否正相关：{'是' if float(a3_full_router.get('target_pearson', 0.0)) > 0 else '否'}；Pearson={float(a3_full_router.get('target_pearson', math.nan)):.4f}，Spearman={float(a3_full_router.get('target_spearman', math.nan)):.4f}。",
        f"9. A2/A3 最优包络相对 A0：得益最大为 {best_mask}（{gains[best_mask] * 100:.2f} pp），最弱为 {failed_mask}（{gains[failed_mask] * 100:.2f} pp）；A3 自身的 S3 macro 相对 A2 变化 {(a3['s3_macro_top1'] - a2['s3_macro_top1']) * 100:+.2f} pp。",
        f"10. 是否值得继续：{'是' if passed >= 2 else '否'}；三项 quick gate 通过 {passed}/3。",
    ]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            *questions,
            "",
            f"快速验证判定：{'成功' if passed >= 2 else '未成功'}（条件1={condition1}，条件2={condition2}，条件3={condition3}）。",
            "",
            "本轮到此停止，不自动启动多 seed、双模态全缺失或完整消融。",
        ]
    )
    return "\n".join(lines) + "\n"


def _future_beam_power(batch: dict[str, Any]) -> torch.Tensor:
    rows = metadata_rows_from_batch(batch.get("metadata"))
    paths = [row.get("future_beam_path") for row in rows]
    if not paths or any(not value for value in paths):
        raise ValueError("PCER evaluation metadata is missing future_beam_path.")
    powers = [torch.as_tensor(np.loadtxt(Path(str(value))), dtype=torch.float32).reshape(-1) for value in paths]
    if any(int(value.numel()) != 64 for value in powers):
        raise ValueError("MMW future beam-power vectors must contain 64 values.")
    return torch.stack(powers)


def _rank_rows(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(dim=-1)
    ranks = torch.empty_like(values)
    rank_values = torch.arange(values.shape[-1], dtype=values.dtype).expand_as(values)
    return ranks.scatter(1, order, rank_values)


def _row_pearson(first: torch.Tensor, second: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    mask = available.to(dtype=first.dtype)
    count = mask.sum(dim=-1).clamp_min(1.0)
    first_centered = (first - (first * mask).sum(dim=-1, keepdim=True) / count.unsqueeze(-1)) * mask
    second_centered = (second - (second * mask).sum(dim=-1, keepdim=True) / count.unsqueeze(-1)) * mask
    numerator = (first_centered * second_centered).sum(dim=-1)
    denominator = first_centered.square().sum(dim=-1).sqrt() * second_centered.square().sum(dim=-1).sqrt()
    return torch.where(denominator.gt(0), numerator / denominator.clamp_min(1e-12), torch.zeros_like(numerator))


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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
