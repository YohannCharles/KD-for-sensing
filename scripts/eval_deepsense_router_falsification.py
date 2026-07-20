#!/usr/bin/env python3
"""同 checkpoint 诊断 DeepSense6G Router 是否优于静态模态先验。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from kd_sensing.config.io import load_config
from kd_sensing.data.deepsense_twc import load_protocol, sha256_file, sha256_payload
from kd_sensing.data.temporal_missing import (
    DEFAULT_TEMPORAL_MODALITIES,
    apply_modality_temporal_mask_to_batch,
    build_random_balanced_modality_frame_masks,
)
from kd_sensing.engine.data_factory import (
    build_dataloader,
    build_dataloaders,
    shutdown_dataloader_workers,
)
from kd_sensing.engine.evaluation_pass_runtime import prepare_evaluation_batch, sample_ids_from_batch
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_cuda_performance_settings, prepare_task_labels, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.checkpoint import load_model_state, load_torch_payload
from kd_sensing.utils.seed import set_seed

import eval_mmw_all_weather_matrix as matrix
from eval_mmw_router_oracle_gap import metrics_for, oracle_gap_branches


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "deepsense6g_router_falsification_v1"
PARENT_PROTOCOL_ID = "deepsense6g_twc_secondary_v1"
SCENES = (31, 32, 33, 34)
DAY_SCENES = (31, 32)
NIGHT_SCENES = (33, 34)
TOKEN_RATES = (0.2, 0.4, 0.6, 0.8, 0.9)
TOKEN_MASKS_PER_RATE = 10
MASK_SEED = 20260720
EXPECTED_CHECKPOINT_EPOCH = 20
MODALITIES = tuple(DEFAULT_TEMPORAL_MODALITIES)
DEFAULT_PARENT = ROOT / "outputs/cache/deepsense6g_twc_secondary_v1/protocol_manifest.json"
DEFAULT_CONFIG = ROOT / "outputs/deepsense6g_twc_secondary_v1/generated_configs/T2_seed1.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs/deepsense6g_twc_secondary_v1/T2/seed1/checkpoints/last.pth"
DEFAULT_CACHE_ROOT = ROOT / "outputs/cache/deepsense6g_router_falsification_v1"
DEFAULT_OUTPUT = ROOT / "outputs/deepsense6g_router_falsification_v1"
SUMMARY_METRICS = (
    "adba",
    "top1",
    "top3",
    "top5",
    "normalized_gain",
    "gain_loss_db",
    "spectral_efficiency_ratio_0db",
    "spectral_efficiency_ratio_10db",
    "spectral_efficiency_ratio_20db",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="冻结开发诊断 manifest 与 fixed masks。")
    _common_paths(prepare)

    evaluate = subparsers.add_parser("evaluate", help="评估一个 Scene shard。")
    _common_paths(evaluate)
    evaluate.add_argument("--scene", required=True, type=int, choices=SCENES)
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--max-batches", type=int)
    evaluate.add_argument("--overwrite", action="store_true")

    summarize = subparsers.add_parser("summarize", help="从四个 Scene trace 离线计算静态先验并汇总。")
    _common_paths(summarize)
    summarize.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    paths = _paths(args)
    if args.command == "prepare":
        result = prepare_falsification(**paths)
    elif args.command == "evaluate":
        if args.batch_size <= 0 or args.batch_size % 16:
            parser.error("--batch-size must be a positive multiple of 16")
        result = evaluate_scene(
            **paths,
            scene=int(args.scene),
            batch_size=int(args.batch_size),
            max_batches=args.max_batches,
            overwrite=bool(args.overwrite),
        )
    else:
        result = summarize_falsification(**paths, overwrite=bool(args.overwrite))
    print(json.dumps(result, indent=2))
    return 0


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent-protocol", default=str(DEFAULT_PARENT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "parent_path": Path(args.parent_protocol).resolve(),
        "config_path": Path(args.config).resolve(),
        "checkpoint_path": Path(args.checkpoint).resolve(),
        "cache_root": Path(args.cache_root).resolve(),
        "output_root": Path(args.output_root).resolve(),
    }


def build_falsification_cache(parent_cache: Mapping[str, Any], *, seed: int = MASK_SEED) -> dict[str, Any]:
    if tuple(parent_cache.get("modalities", ())) != MODALITIES or int(parent_cache.get("history_window", -1)) != 5:
        raise ValueError("DeepSense6G Router falsification requires the parent 5x4 mask contract.")
    whole = [dict(item) for item in parent_cache.get("conditions", ()) if item.get("family") == "whole_modality"]
    if len(whole) != 15 or sorted(int(item["drop_count"]) for item in whole) != [0] + [1] * 4 + [2] * 6 + [3] * 4:
        raise ValueError("Parent DeepSense6G cache must contain Clean and all 14 non-empty whole-modality subsets.")
    conditions = whole[:]
    audits = []
    for rate in TOKEN_RATES:
        matrices = build_random_balanced_modality_frame_masks(
            mask_count=TOKEN_MASKS_PER_RATE,
            missing_rate=rate,
            seed=int(seed),
            history_window=5,
            modality_count=len(MODALITIES),
        )
        rate_conditions = [_token_condition(matrix, rate=rate, index=index) for index, matrix in enumerate(matrices)]
        conditions.extend(rate_conditions)
        retained = np.asarray([item["modality_temporal_mask"] for item in rate_conditions], dtype=np.int64).sum(axis=0)
        audits.append(
            {
                "requested_missing_rate": rate,
                "mask_count": TOKEN_MASKS_PER_RATE,
                "retained_per_cell": retained.tolist(),
                "retained_per_modality": retained.sum(axis=0).tolist(),
                "retained_per_frame": retained.sum(axis=1).tolist(),
                "exact_cell_balance": bool(np.all(retained == retained.reshape(-1)[0])),
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "generator": "deepsense6g_seeded_random_kof20_exact_panel_balance_v1",
        "balance_policy": "minimum_cell_swap_exact_aggregate_cell_marginals_v1",
        "seed": int(seed),
        "history_window": 5,
        "modalities": list(MODALITIES),
        "token_rates": list(TOKEN_RATES),
        "token_masks_per_rate": TOKEN_MASKS_PER_RATE,
        "conditions": conditions,
        "rate_balance_audit": audits,
    }
    payload["checksum"] = sha256_payload(payload)
    validate_falsification_cache(payload)
    return payload


def validate_falsification_cache(cache: Mapping[str, Any]) -> None:
    checksum = str(cache.get("checksum", ""))
    body = {key: value for key, value in cache.items() if key != "checksum"}
    if cache.get("protocol_id") != PROTOCOL_ID or checksum != sha256_payload(body):
        raise ValueError("Invalid or drifted DeepSense6G Router falsification cache.")
    conditions = list(cache.get("conditions", ()))
    if len(conditions) != 15 + len(TOKEN_RATES) * TOKEN_MASKS_PER_RATE:
        raise ValueError("DeepSense6G Router falsification cache must contain exactly 65 conditions.")
    digests = [str(item.get("mask_digest", "")) for item in conditions]
    if any(not value for value in digests) or len(set(digests)) != len(digests):
        raise ValueError("DeepSense6G Router falsification masks must have unique identities.")
    for audit in cache.get("rate_balance_audit", ()):
        retained = np.asarray(audit["retained_per_cell"])
        if retained.shape != (5, 4) or not bool(audit.get("exact_cell_balance")) or not np.all(retained == retained[0, 0]):
            raise ValueError("Token panel is not exactly balanced across modality-frame cells.")


def _token_condition(matrix: list[list[bool]], *, rate: float, index: int) -> dict[str, Any]:
    canonical = {"modalities": list(MODALITIES), "modality_temporal_mask": matrix}
    observed = sum(not bool(value) for row in matrix for value in row) / 20.0
    return {
        "family": "temporal_missing",
        "pattern": f"token{int(round(rate * 100)):02d}_{index:02d}",
        "mask_type": "modality_frame",
        "requested_missing_rate": float(rate),
        "observed_missing_rate": observed,
        "available_modalities": list(MODALITIES),
        "drop_count": 0,
        "modality_temporal_mask": matrix,
        "mask_digest": sha256_payload(canonical),
    }


def prepare_falsification(
    *,
    parent_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    for path in (parent_path, config_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = load_protocol(parent_path)
    if parent.get("protocol_id") != PARENT_PROTOCOL_ID:
        raise ValueError("Router falsification must bind the DeepSense6G secondary protocol.")
    parent_cache_path = Path(parent["fixed_mask_cache"]["path"])
    parent_cache = json.loads(parent_cache_path.read_text(encoding="utf-8"))
    cache = build_falsification_cache(parent_cache)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "fixed_mask_cache.json"
    _write_immutable_json(cache_path, cache)

    checkpoint = load_torch_payload(checkpoint_path, map_location="cpu")
    epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    if epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError(f"This development diagnostic is frozen to epoch {EXPECTED_CHECKPOINT_EPOCH}, got {epoch}.")
    metadata = load_checkpoint_metadata(checkpoint_path) or {}
    checkpoint_sha = str(metadata.get("checkpoint_sha256", ""))
    if not checkpoint_sha:
        checkpoint_sha = sha256_file(checkpoint_path)
    request = {
        "protocol_id": PROTOCOL_ID,
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "parent_protocol_manifest_sha256": parent["manifest_sha256"],
        "parent_component_inventory_sha256": parent["pooled_dataset"]["component_inventory_sha256"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": epoch,
        "training_complete": False,
        "split": "pooled_test_scene_shards",
        "scenes": list(SCENES),
        "day_scenes": list(DAY_SCENES),
        "night_scenes": list(NIGHT_SCENES),
        "sample_count": int(parent["pooled_dataset"]["test_row_count"]),
        "mask_seed": MASK_SEED,
        "mask_cache_checksum": cache["checksum"],
        "condition_count": len(cache["conditions"]),
        "fusion_branches": ["learned", "uniform", "global_clean_prior", "oracle"],
        "claim_eligible": False,
        "gpu_allowlist": [4, 5, 6, 7],
    }
    manifest = {
        "schema_version": 1,
        "request": request,
        "request_sha256": sha256_payload(request),
        "fixed_mask_cache": {
            "path": str(cache_path),
            "sha256": sha256_file(cache_path),
            "checksum": cache["checksum"],
        },
    }
    manifest["manifest_sha256"] = sha256_payload(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "diagnostic_manifest.json"
    _write_immutable_json(manifest_path, manifest)
    return {"status": "prepared", "manifest": str(manifest_path), "condition_count": len(cache["conditions"])}


def evaluate_scene(
    *,
    parent_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    cache_root: Path,
    output_root: Path,
    scene: int,
    batch_size: int,
    max_batches: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    prepare_falsification(
        parent_path=parent_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        cache_root=cache_root,
        output_root=output_root,
    )
    manifest = _load_manifest(output_root / "diagnostic_manifest.json")
    cache = _load_cache(Path(manifest["fixed_mask_cache"]["path"]))
    scene_dir = output_root / f"scene{scene}"
    complete = scene_dir / "complete.json"
    partial = max_batches is not None
    if complete.is_file() and not overwrite:
        return {"status": "already_complete", "scene": scene, "output": str(scene_dir)}
    if overwrite and complete.is_file():
        raise ValueError("Completed scene evidence is immutable; use a new output root.")
    scene_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    cfg.setdefault("training", {})["final_test"] = {"enabled": True}
    cfg.setdefault("temporal_missing", {}).update({"enabled": False, "mode": "none"})
    cfg["data"]["dataloader"].update(test_batch_size=batch_size, validation_batch_size=batch_size)
    metadata = load_checkpoint_metadata(checkpoint_path)
    normalization = load_normalization_artifacts(metadata)
    dataloaders = None
    started = time.monotonic()
    try:
        dataloaders = build_dataloaders(cfg, normalization_overrides=normalization)
        pooled = dataloaders["test"].dataset
        components = list(getattr(pooled, "datasets", ()))
        inventory = list(getattr(pooled, "domain_inventory", ()))
        selected = [(component, item) for component, item in zip(components, inventory) if int(item["scene"]) == scene]
        if len(selected) != 1:
            raise ValueError(f"Expected one pooled test component for Scene{scene}, got {len(selected)}.")
        component, domain = selected[0]
        loader = build_dataloader(component, cfg["data"]["dataloader"], split="test", experiment_seed=1)
        try:
            set_seed(1)
            device = build_device(cfg)
            configure_cuda_performance_settings(cfg, device)
            model = build_model(cfg["model"]["primary"]).to(device)
            loaded = load_model_state(
                checkpoint_path,
                model,
                role="DeepSense6G Router falsification epoch20 checkpoint",
                map_location=device,
                strict=True,
            )
            if int(loaded["checkpoint"].get("epoch", -1)) != EXPECTED_CHECKPOINT_EPOCH:
                raise ValueError("Loaded checkpoint epoch differs from the frozen falsification manifest.")
            model.eval()
            traces = _evaluate_conditions(
                model,
                loader,
                cfg,
                device,
                cache["conditions"],
                max_batches=max_batches,
                scene=scene,
            )
        finally:
            shutdown_dataloader_workers(loader)

        metric_rows = []
        trace_records = []
        for index, (condition, trace) in enumerate(zip(cache["conditions"], traces)):
            trace_path = scene_dir / "traces" / f"{index:03d}_{condition['pattern']}.npz"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(trace_path, **trace)
            trace_records.append(
                {
                    "condition_index": index,
                    "pattern": condition["pattern"],
                    "path": str(trace_path),
                    "sha256": sha256_file(trace_path),
                }
            )
            for fusion, key in (("learned", "learned_logits"), ("uniform", "uniform_logits"), ("oracle", "oracle_logits")):
                metric_rows.append(
                    _metric_row(
                        trace,
                        cfg,
                        condition=condition,
                        condition_index=index,
                        scene=scene,
                        fusion=fusion,
                        logits=trace[key],
                        weights=_branch_weights(trace, fusion),
                    )
                )
        _write_csv(scene_dir / "metrics.csv", metric_rows)
        provenance = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "manifest_sha256": manifest["manifest_sha256"],
            "scene": scene,
            "domain_id": str(domain["id"]),
            "day_night": "day" if scene in DAY_SCENES else "night",
            "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
            "training_complete": False,
            "sample_count": len(component) if not partial else int(traces[0]["target"].shape[0]),
            "condition_count": len(traces),
            "batch_size": batch_size,
            "partial": partial,
            "claim_eligible": False,
            "trace_files": trace_records,
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(scene_dir / "provenance.json", provenance)
        if not partial:
            _write_json(complete, {"status": "complete", **provenance})
        return {"status": "partial" if partial else "complete", "scene": scene, **provenance}
    finally:
        if dataloaders is not None:
            shutdown_all_dataloaders(dataloaders)


def _evaluate_conditions(
    model,
    loader,
    cfg: dict[str, Any],
    device: torch.device,
    conditions: list[dict[str, Any]],
    *,
    max_batches: int | None,
    scene: int,
) -> list[dict[str, np.ndarray]]:
    masks = [matrix._mask_in_model_order(model, item, MODALITIES)[0] for item in conditions]
    chunks: list[dict[str, list[np.ndarray]]] = [defaultdict(list) for _ in conditions]
    model_cfg = cfg["model"]["primary"]
    seq_length = int(model_cfg.get("seq_length", cfg.get("model", {}).get("seq_length", 5)))
    num_pred = int(model_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1)))
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            prepared = prepare_evaluation_batch(raw_batch)
            powers = matrix._load_future_beam_power(prepared).to(device=device)
            ids = np.asarray(sample_ids_from_batch(prepared), dtype=str)
            for condition_index, mask in enumerate(masks):
                batch = matrix._clone_batch(prepared)
                apply_modality_temporal_mask_to_batch(batch, mask, modalities=tuple(model.modalities))
                availability = batch["modality_mask"].to(device=device, dtype=torch.bool)
                step = run_model_step(
                    model,
                    cfg["experiment"].get("task", "fusion"),
                    batch,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                    extra_model_kwargs={"missing_mask": availability},
                )
                labels = prepare_task_labels(step.batch, num_pred=num_pred, device=device)
                target = labels[:, -1].reshape(-1) if labels.ndim > 1 else labels.reshape(-1)
                diagnostics = step.model_output.diagnostics
                unimodal = diagnostics["unimodal_logits"]
                router = diagnostics["supervised_router_gate_weights"]
                learned = step.logits[:, -1, :] if step.logits.ndim == 3 else step.logits
                available = diagnostics["available_modalities"].to(dtype=torch.bool)
                branches = oracle_gap_branches(unimodal, router, powers, available)
                reconstructed = (router.unsqueeze(-1) * unimodal).sum(dim=1)
                if not torch.allclose(learned, reconstructed, atol=1e-5, rtol=1e-5):
                    raise ValueError("Learned logits cannot be reconstructed from the saved Router state.")
                if len(ids) != int(target.numel()):
                    raise ValueError("Sample identity count differs from batch target count.")
                values = {
                    "sample_id": ids,
                    "target": target.detach().cpu().numpy().astype(np.int64),
                    "beam_powers": powers.detach().cpu().numpy().astype(np.float32),
                    "available": available.detach().cpu().numpy().astype(np.bool_),
                    "unimodal_logits": unimodal.detach().cpu().numpy().astype(np.float32),
                    "router_weights": router.detach().cpu().numpy().astype(np.float32),
                    "learned_logits": learned.detach().cpu().numpy().astype(np.float32),
                    **{key: value.detach().cpu().numpy() for key, value in branches.items()},
                }
                for key, value in values.items():
                    chunks[condition_index][key].append(value)
            print(
                f"Scene{scene}: batch {batch_index + 1}/{len(loader)}, conditions={len(conditions)}",
                flush=True,
            )
    result = []
    for condition_chunks in chunks:
        if not condition_chunks:
            raise ValueError("DeepSense6G Router falsification produced an empty condition.")
        result.append({key: np.concatenate(values, axis=0) for key, values in condition_chunks.items()})
    return result


def static_prior_branch(
    unimodal_logits: np.ndarray,
    available: np.ndarray,
    prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(unimodal_logits, dtype=np.float64)
    availability = np.asarray(available, dtype=bool)
    base = np.asarray(prior, dtype=np.float64).reshape(1, -1)
    if logits.ndim != 3 or availability.shape != logits.shape[:2] or base.shape[1] != logits.shape[1]:
        raise ValueError("Static-prior fusion expects logits [N,M,C], availability [N,M], and prior [M].")
    weights = availability * base
    totals = weights.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Every sample must keep at least one positive-prior modality available.")
    weights = weights / totals
    return np.sum(weights[..., None] * logits, axis=1).astype(np.float32), weights.astype(np.float32)


def summarize_falsification(
    *,
    parent_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    cache_root: Path,
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    del parent_path, checkpoint_path, cache_root
    manifest = _load_manifest(output_root / "diagnostic_manifest.json")
    summary_dir = output_root / "summary"
    summary_json = summary_dir / "summary.json"
    if summary_json.is_file() and not overwrite:
        return json.loads(summary_json.read_text(encoding="utf-8"))
    cache = _load_cache(Path(manifest["fixed_mask_cache"]["path"]))
    cfg = load_config(config_path)
    scene_traces: dict[int, list[dict[str, np.ndarray]]] = {}
    clean_router = []
    for scene in SCENES:
        complete = output_root / f"scene{scene}/complete.json"
        if not complete.is_file():
            raise FileNotFoundError(f"Scene{scene} falsification is incomplete: {complete}")
        provenance = json.loads(complete.read_text(encoding="utf-8"))
        if provenance.get("manifest_sha256") != manifest["manifest_sha256"] or provenance.get("partial"):
            raise ValueError(f"Scene{scene} provenance differs from the frozen falsification manifest.")
        traces = [_load_trace(output_root / f"scene{scene}", record) for record in provenance["trace_files"]]
        if len(traces) != len(cache["conditions"]):
            raise ValueError(f"Scene{scene} condition coverage is incomplete.")
        scene_traces[scene] = traces
        clean_router.append(traces[0]["router_weights"])
    global_prior = np.concatenate(clean_router, axis=0).mean(axis=0).astype(np.float64)
    global_prior = global_prior / global_prior.sum()

    rows = []
    for scene, traces in scene_traces.items():
        for index, (condition, trace) in enumerate(zip(cache["conditions"], traces)):
            static_logits, static_weights = static_prior_branch(
                trace["unimodal_logits"], trace["available"], global_prior
            )
            branch_data = (
                ("learned", trace["learned_logits"], _branch_weights(trace, "learned")),
                ("uniform", trace["uniform_logits"], _branch_weights(trace, "uniform")),
                ("global_clean_prior", static_logits, static_weights),
                ("oracle", trace["oracle_logits"], _branch_weights(trace, "oracle")),
            )
            for fusion, logits, weights in branch_data:
                rows.append(
                    _metric_row(
                        trace,
                        cfg,
                        condition=condition,
                        condition_index=index,
                        scene=scene,
                        fusion=fusion,
                        logits=logits,
                        weights=weights,
                    )
                )
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(summary_dir / "scene_condition_metrics.csv", rows)
    grouped = aggregate_summary_rows(rows)
    _write_csv(summary_dir / "cell_summary.csv", grouped)
    decision = build_decision(grouped)
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
        "training_complete": False,
        "claim_eligible": False,
        "global_clean_prior": {name: float(global_prior[index]) for index, name in enumerate(MODALITIES)},
        "decision": decision,
        "artifacts": {
            "scene_condition_metrics": str(summary_dir / "scene_condition_metrics.csv"),
            "cell_summary": str(summary_dir / "cell_summary.csv"),
            "readme": str(summary_dir / "README.md"),
        },
    }
    _write_json(summary_json, payload)
    (summary_dir / "README.md").write_text(_markdown_report(payload, grouped), encoding="utf-8")
    return payload


def _metric_row(
    trace: Mapping[str, np.ndarray],
    cfg: dict[str, Any],
    *,
    condition: Mapping[str, Any],
    condition_index: int,
    scene: int,
    fusion: str,
    logits: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    values = metrics_for(logits, trace["target"], trace["beam_powers"], cfg)
    modality_gain = np.asarray(trace["unimodal_normalized_gain"], dtype=np.float64)
    branch_weights = np.asarray(weights, dtype=np.float64)
    oracle_gain = modality_gain.max(axis=1)
    selected = modality_gain[np.arange(len(modality_gain)), branch_weights.argmax(axis=1)]
    expected = np.sum(branch_weights * modality_gain, axis=1)
    oracle_ids = np.asarray(trace["oracle_modality"], dtype=np.int64)
    return {
        "protocol_id": PROTOCOL_ID,
        "scene": scene,
        "day_night": "day" if scene in DAY_SCENES else "night",
        "condition_index": condition_index,
        "cell": _cell_name(condition),
        "family": condition["family"],
        "pattern": condition["pattern"],
        "mask_type": condition["mask_type"],
        "requested_missing_rate": condition["requested_missing_rate"],
        "mask_digest": condition["mask_digest"],
        "fusion": fusion,
        "sample_count": int(len(trace["target"])),
        **values,
        "soft_oracle_regret": float(np.mean(oracle_gain - expected)),
        "selection_oracle_regret": float(np.mean(oracle_gain - selected)),
        **{f"weight_{name}": float(branch_weights[:, index].mean()) for index, name in enumerate(MODALITIES)},
        **{f"oracle_frequency_{name}": float(np.mean(oracle_ids == index)) for index, name in enumerate(MODALITIES)},
    }


def _branch_weights(trace: Mapping[str, np.ndarray], fusion: str) -> np.ndarray:
    available = np.asarray(trace["available"], dtype=bool)
    if fusion == "learned":
        weights = np.asarray(trace["router_weights"], dtype=np.float64)
        weights = np.where(available, weights, 0.0)
        return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)
    if fusion == "uniform":
        weights = available.astype(np.float64)
        return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)
    if fusion == "oracle":
        weights = np.zeros_like(available, dtype=np.float32)
        weights[np.arange(len(weights)), np.asarray(trace["oracle_modality"], dtype=np.int64)] = 1.0
        return weights
    raise ValueError(f"Unsupported branch {fusion!r}.")


def aggregate_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes: dict[str, set[int]] = {
        "pooled": set(SCENES),
        "day": set(DAY_SCENES),
        "night": set(NIGHT_SCENES),
        **{f"scene{scene}": {scene} for scene in SCENES},
    }
    result = []
    cells = ("Clean", "Drop1", "Drop2", "Drop3", "Token20", "Token40", "Token60", "Token80", "Token90")
    for scope, scope_scenes in scopes.items():
        for cell in cells:
            for fusion in ("learned", "uniform", "global_clean_prior", "oracle"):
                selected = [row for row in rows if row["scene"] in scope_scenes and row["cell"] == cell and row["fusion"] == fusion]
                if not selected:
                    continue
                weights = np.asarray([int(row["sample_count"]) for row in selected], dtype=np.float64)
                payload: dict[str, Any] = {
                    "scope": scope,
                    "cell": cell,
                    "fusion": fusion,
                    "scene_count": len({int(row["scene"]) for row in selected}),
                    "condition_count": len({int(row["condition_index"]) for row in selected}),
                    "sample_condition_count": int(weights.sum()),
                }
                numeric = (*SUMMARY_METRICS, "soft_oracle_regret", "selection_oracle_regret")
                numeric += tuple(f"weight_{name}" for name in MODALITIES)
                numeric += tuple(f"oracle_frequency_{name}" for name in MODALITIES)
                for key in numeric:
                    payload[key] = float(np.average([float(row[key]) for row in selected], weights=weights))
                result.append(payload)
    return result


def build_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stress = ("Drop2", "Drop3", "Token60", "Token80", "Token90")
    lookup = {(row["scope"], row["cell"], row["fusion"]): row for row in rows}

    def delta(scope: str, metric: str) -> float:
        return float(
            np.mean(
                [
                    float(lookup[(scope, cell, "learned")][metric])
                    - float(lookup[(scope, cell, "global_clean_prior")][metric])
                    for cell in stress
                ]
            )
        )

    scene_deltas = {f"scene{scene}": delta(f"scene{scene}", "adba") for scene in SCENES}
    day_learned = lookup[("day", "Clean", "learned")]
    night_learned = lookup[("night", "Clean", "learned")]
    day_oracle = np.asarray([day_learned[f"oracle_frequency_{name}"] for name in MODALITIES])
    night_oracle = np.asarray([night_learned[f"oracle_frequency_{name}"] for name in MODALITIES])
    day_weight = np.asarray([day_learned[f"weight_{name}"] for name in MODALITIES])
    night_weight = np.asarray([night_learned[f"weight_{name}"] for name in MODALITIES])
    oracle_shift = night_oracle - day_oracle
    weight_shift = night_weight - day_weight
    denominator = float(np.linalg.norm(oracle_shift) * np.linalg.norm(weight_shift))
    shift_cosine = float(np.dot(oracle_shift, weight_shift) / denominator) if denominator > 0 else math.nan
    adba_delta = delta("pooled", "adba")
    gain_delta = delta("pooled", "normalized_gain")
    positive_scenes = sum(value > 0 for value in scene_deltas.values())
    proceed = adba_delta >= 0.002 and gain_delta >= 0.005 and positive_scenes >= 3 and shift_cosine > 0
    return {
        "stress_cells": list(stress),
        "learned_minus_global_clean_prior_adba": adba_delta,
        "learned_minus_global_clean_prior_normalized_gain": gain_delta,
        "scene_adba_deltas": scene_deltas,
        "positive_scene_count": positive_scenes,
        "day_night_weight_oracle_shift_cosine": shift_cosine,
        "thresholds": {
            "adba": 0.002,
            "normalized_gain": 0.005,
            "positive_scene_count": 3,
            "shift_cosine_exclusive_min": 0.0,
        },
        "recommendation": "resume_formal_40epoch_router_evidence" if proceed else "do_not_resume_router_specific_full_matrix",
    }


def _markdown_report(payload: Mapping[str, Any], rows: list[dict[str, Any]]) -> str:
    lookup = {(row["scope"], row["cell"], row["fusion"]): row for row in rows}
    lines = [
        "# DeepSense6G Router Falsification",
        "",
        "该结果复用中断在 epoch20/40 的 T2 seed1 checkpoint，属于开发诊断，`claim_eligible=false`。",
        "该 checkpoint 仍使用旧版 `router_oracle_target_mode=hard_first`，不是后续 MMW H2R-JointCE 配方，",
        "因此只能判断 DeepSense6G 的 Router 压力与旧 Router 行为，不能作为 H2R-JointCE 的最终结论。",
        "",
        "| Cell | Uniform ADBA | Static ADBA | Learned ADBA | Oracle ADBA | L-U ADBA | L-S ADBA | L-S norm. gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in ("Clean", "Drop1", "Drop2", "Drop3", "Token20", "Token40", "Token60", "Token80", "Token90"):
        uniform = lookup[("pooled", cell, "uniform")]
        static = lookup[("pooled", cell, "global_clean_prior")]
        learned = lookup[("pooled", cell, "learned")]
        oracle = lookup[("pooled", cell, "oracle")]
        lines.append(
            f"| {cell} | {uniform['adba']:.4f} | {static['adba']:.4f} | {learned['adba']:.4f} | "
            f"{oracle['adba']:.4f} | {learned['adba'] - uniform['adba']:+.4f} | "
            f"{learned['adba'] - static['adba']:+.4f} | "
            f"{learned['normalized_gain'] - static['normalized_gain']:+.4f} |"
        )
    decision = payload["decision"]
    lines.extend(
        [
            "",
            "## 权重诊断",
            "",
            "每个单元格为 `Learned Router 平均权重 / 逐样本最优模态频率`。",
            "",
            "| Cell | Image | Radar | GPS | LiDAR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cell in ("Clean", "Drop2", "Token60", "Token80", "Token90"):
        learned = lookup[("pooled", cell, "learned")]
        values = [
            f"{learned[f'weight_{name}']:.4f} / {learned[f'oracle_frequency_{name}']:.4f}"
            for name in MODALITIES
        ]
        lines.append(f"| {cell} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 场景一致性",
            "",
            "下表为预注册压力单元上的平均 `Learned - GlobalCleanPrior ADBA`。",
            "",
            "| Scene | 31 (day) | 32 (day) | 33 (night) | 34 (night) |",
            "|---|---:|---:|---:|---:|",
            "| Delta | "
            + " | ".join(f"{decision['scene_adba_deltas'][f'scene{scene}']:+.4f}" for scene in SCENES)
            + " |",
        ]
    )
    lines.extend(
        [
            "",
            "## 决策",
            "",
            f"- 建议：`{decision['recommendation']}`",
            f"- 压力单元 Learned-Static ADBA：{decision['learned_minus_global_clean_prior_adba']:+.4f}",
            f"- 压力单元 Learned-Static normalized gain：{decision['learned_minus_global_clean_prior_normalized_gain']:+.4f}",
            f"- 正增益 Scene 数：{decision['positive_scene_count']}/4",
            f"- day/night Router 与 Oracle shift cosine：{decision['day_night_weight_oracle_shift_cosine']:+.4f}",
            f"- ADBA 门槛：{decision['thresholds']['adba']:+.4f}",
            f"- normalized gain 门槛：{decision['thresholds']['normalized_gain']:+.4f}",
            "- 本轮未通过 normalized gain 门槛，因此不恢复 Router 专属的正式 40 epoch 全矩阵。",
            "",
        ]
    )
    return "\n".join(lines)


def _cell_name(condition: Mapping[str, Any]) -> str:
    if condition["family"] == "whole_modality":
        return "Clean" if int(condition["drop_count"]) == 0 else f"Drop{int(condition['drop_count'])}"
    return f"Token{int(round(float(condition['requested_missing_rate']) * 100))}"


def _load_trace(scene_dir: Path, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = scene_dir / path
    if sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"Trace SHA256 mismatch: {path}")
    with np.load(path) as payload:
        return {key: payload[key] for key in payload.files}


def _load_cache(path: Path) -> dict[str, Any]:
    cache = json.loads(path.read_text(encoding="utf-8"))
    validate_falsification_cache(cache)
    return cache


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(payload.get("manifest_sha256", ""))
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload.get("request", {}).get("protocol_id") != PROTOCOL_ID or recorded != sha256_payload(body):
        raise ValueError(f"Invalid DeepSense6G Router falsification manifest: {path}")
    if payload["request"].get("claim_eligible") is not False:
        raise ValueError("DeepSense6G Router falsification must remain claim-ineligible.")
    return payload


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Refusing to mutate immutable diagnostic identity: {path}")
        return
    _write_json(path, payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
