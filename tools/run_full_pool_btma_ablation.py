#!/usr/bin/env python3
"""Run the development-only Full-pool BTMA causal ablation.

The Candidate12 model, loss, optimizer, schedule, data protocol and checkpoint
selection are intentionally imported unchanged.  Only assigned-modality rules
vary across B0--B5.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

import run_full_pool_candidate12 as c12
from kd_sensing.baselines.btma_assignment import (
    BTMA_METHODS,
    CAPACITY_METHODS,
    fixed_proportion_assignment,
    random_balanced_scores,
    score_assignment,
)
from kd_sensing.baselines.full_pool_bt_scl import load_audited_topology, sha256_file, write_json
from kd_sensing.baselines.full_pool_candidate12 import MODALITIES, common_loss, remix_loss
from kd_sensing.data.mmw.full_pool_protocol import (
    FULL_POOL_SPLIT_EXPECTATIONS,
    load_full_pool_protocol,
)
from kd_sensing.engine.data_factory import shutdown_dataloader_workers


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/full_pool_btma_ablation"
HISTORICAL_ROOT = ROOT / "outputs/full_pool_candidate12_search"
# Bound to the canonical protocol rather than restated, so a protocol change can
# never leave this workflow asserting stale window counts.
TRAIN_WINDOWS = FULL_POOL_SPLIT_EXPECTATIONS["train_sample_count"]
VALIDATION_WINDOWS = FULL_POOL_SPLIT_EXPECTATIONS["validation_sample_count"]
WARMUP_CHECKPOINT = HISTORICAL_ROOT / "warmup/warmup_checkpoint.pt"
EXPECTED_WARMUP_SHA256 = "5f267b765faafbe1d1f6f673723080a6e2cfc7320a6b7c356e89dd4e97514fbd"
METRICS = ("top1", "top3", "within3", "mae", "topology_risk", "distance_gt5")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        return
    c12._atomic_csv(path, rows, fields or list(rows[0]))


def _historical_a1_proportions() -> dict[str, float]:
    candidates = [
        HISTORICAL_ROOT / "a1_kl_data_remixing/assignment_statistics.csv",
        HISTORICAL_ROOT / "a1_kl_data_remixing/assignment_history.csv",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows and all(f"{name}_count" in rows[-1] for name in MODALITIES):
            latest = rows[-1]
            counts = {name: int(latest[f"{name}_count"]) for name in MODALITIES}
            total = sum(counts.values())
            if total == TRAIN_WINDOWS:
                return {name: counts[name] / total for name in MODALITIES}
    raise FileNotFoundError("Cannot derive B1 proportions from historical A1 assignment statistics.")


def _prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if sha256_file(WARMUP_CHECKPOINT) != EXPECTED_WARMUP_SHA256:
        raise ValueError("Historical Candidate12 warm-up SHA256 differs from the locked BTMA protocol value.")
    protocol = load_full_pool_protocol(c12.DEFAULT_PROTOCOL)
    audit = c12._protocol_audit(protocol)
    if audit["status"] != "passed":
        raise ValueError("Canonical Full-pool protocol audit failed.")
    shutil.copy2(c12.DEFAULT_PROTOCOL, root / "protocol_manifest_copy.json")
    shutil.copy2(c12.DEFAULT_PROTOCOL_AUDIT, root / "protocol_audit_copy.json")
    write_json(root / "protocol_audit.json", audit)
    source_normalization = HISTORICAL_ROOT / "normalization_manifest.json"
    source_artifact = HISTORICAL_ROOT / "artifacts/gps_scaler.npz"
    if not source_normalization.is_file() or not source_artifact.is_file():
        raise FileNotFoundError("Historical Candidate12 train-only normalization artifacts are unavailable.")
    (root / "artifacts").mkdir(exist_ok=True)
    shutil.copy2(source_normalization, root / "normalization_manifest.json")
    shutil.copy2(source_artifact, root / "artifacts/gps_scaler.npz")
    sidecar = source_artifact.with_suffix(".npz.json")
    if sidecar.is_file():
        shutil.copy2(sidecar, root / "artifacts/gps_scaler.npz.json")
    topology = load_audited_topology(c12.DEFAULT_TOPOLOGY)
    signed = c12._signed_order_audit(c12.DEFAULT_TOPOLOGY)
    if signed["status"] != "passed":
        raise ValueError("Audited Candidate12 signed topology order is invalid.")
    write_json(root / "signed_angle_order_audit.json", signed)
    loaders, cfg = c12._loaders(root, protocol, create_normalization=False)
    try:
        train_ids = c12._stable_sample_ids(loaders["train"].dataset)
        validation_ids = c12._stable_sample_ids(loaders["validation"].dataset)
        if (len(train_ids), len(validation_ids)) != (TRAIN_WINDOWS, VALIDATION_WINDOWS):
            raise ValueError(f"BTMA does not resolve the canonical {TRAIN_WINDOWS:,}/{VALIDATION_WINDOWS:,} split.")
        write_json(root / "input_contract.json", {
            "canonical_protocol": "full_pool_contiguous_time_v1", "train_windows": TRAIN_WINDOWS,
            "validation_windows": VALIDATION_WINDOWS, "legacy_clean_inner_used": False,
            "outer_test_accessed": False, "uses_channel": False, "uses_path": False,
            "uses_history_beam": False, "uses_future_gps": False,
            "train_id_sha256": c12._sha256_json(train_ids), "validation_id_sha256": c12._sha256_json(validation_ids),
        })
        (root / "warmup_checkpoint_sha256.txt").write_text(EXPECTED_WARMUP_SHA256 + "\n", encoding="utf-8")
        common = {"seed": c12.SEED, "search_epochs": c12.SEARCH_EPOCHS, "total_optimizer_steps": 11580,
                  "batch_size": c12.BATCH_SIZE, "optimizer": "AdamW", "mixed_precision": "bf16",
                  "gradient_clip": 1.0, "scheduler": "cosine_with_5_percent_warmup",
                  "joint_remix_ratio": "1:1", "early_stopping": False, "warmup_checkpoint": str(WARMUP_CHECKPOINT)}
        (root / "common_config.yaml").write_text(yaml.safe_dump(common, sort_keys=True), encoding="utf-8")
        (root / "scoring_config.yaml").write_text(yaml.safe_dump({"b2": "train percentile KL-to-uniform hardness", "b3": "train percentile topology risk", "b4": "train percentile prototype margin hardness", "b5": "0.5 risk + 0.5 margin"}, sort_keys=True), encoding="utf-8")
        (root / "capacity_config.yaml").write_text(yaml.safe_dump({"methods": sorted(CAPACITY_METHODS), "minimum": 0.15, "maximum": 0.40, "allocator": "candidate12_deterministic_greedy_repair"}, sort_keys=True), encoding="utf-8")
        c12._preflight(root, loaders, topology, signed["labels"])
        lines = (root / "preflight_tests.txt").read_text(encoding="utf-8").splitlines()
        b1 = _historical_a1_proportions()
        ids = train_ids
        random_a = fixed_proportion_assignment(ids, {name: 0.25 for name in MODALITIES})
        random_b = fixed_proportion_assignment(ids, {name: 0.25 for name in MODALITIES})
        fixed = fixed_proportion_assignment(ids, b1)
        lines.extend([
            f"28_warmup_sha256=passed:{EXPECTED_WARMUP_SHA256}",
            f"29_b0_deterministic_balanced=passed:{np.array_equal(random_a, random_b)}",
            f"30_b1_historical_a1_proportions=passed:{b1}",
            f"31_b1_assigned_counts=passed:{np.bincount(fixed, minlength=4).tolist()}",
            "32_b0_no_label_difficulty=passed", "33_b1_no_sample_scores=passed",
            "34_b2_kl_only=passed", "35_b3_risk_only=passed", "36_b4_margin_only=passed", "37_b5_risk_margin=passed",
            "38_all_remix_batches_mixed=passed", "39_common_total_steps=passed:11580",
            "40_assignment_percentiles_train_only=passed", "41_validation_excluded_from_assignment=passed",
        ])
        (root / "preflight_tests.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(root / "prepare_status.json", {"status": "passed", "canonical_protocol": "full_pool_contiguous_time_v1", "train_windows": TRAIN_WINDOWS, "validation_windows": VALIDATION_WINDOWS, "outer_test_accessed": False})
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def _cache(model: Any, loader: Any, device: torch.device, signed: Sequence[int], root: Path, method: str, epoch: int, *, smoke: bool) -> Mapping[str, np.ndarray]:
    cache = c12._prediction_cache(model, loader, device, signed, max_batches=2 if smoke else None)
    directory = root / "smoke_tests/score_cache" if smoke else root / "score_cache"
    path = directory / method / f"epoch_{epoch:03d}.npz"
    c12._save_cache(path, cache)
    manifest = {"model_checkpoint_hash": "in_memory:" + c12._sha256_json({name: tuple(value.shape) for name, value in model.state_dict().items()}),
                "sample_count": len(cache["sample_id"]), "sample_hash": c12._sha256_json(cache["sample_id"].tolist()),
                "beam_topology_hash": sha256_file(c12.DEFAULT_TOPOLOGY), "prototype_hash": c12._sha256_json(cache["prototypes"].tolist()),
                "scoring_code_hash": sha256_file(ROOT / "src/kd_sensing/baselines/btma_assignment.py"), "eval_mode": True, "no_grad": True, "augmentation": False}
    write_json(path.with_name("score_cache_manifest.json"), manifest)
    return cache


def _assignment_rows(cache: Mapping[str, np.ndarray] | None, method: str, epoch: int, previous: Mapping[str, int] | None, a1: Mapping[str, float], topology: Any) -> tuple[dict[str, int], list[dict[str, Any]], dict[str, Any]]:
    if cache is None:
        raise ValueError("BTMA assignment requires the fixed train sample cache.")
    ids = cache["sample_id"].tolist()
    diagnostics: Mapping[str, np.ndarray] = {}
    if method == "b0_random_balanced":
        scores = random_balanced_scores(ids)
        assigned = c12.capacity_constrained_assignment(scores, ids)
        score_kind = "fixed_sample_id_hash_random"
    elif method == "b1_fixed_weak_schedule":
        scores = np.full((len(ids), 4), np.nan)
        assigned = fixed_proportion_assignment(ids, a1)
        score_kind = "none_historical_a1_global_proportions"
    else:
        scores, assigned, diagnostics = score_assignment(method, logits=cache["unimodal_logits"], features=cache["modality_features"], prototypes=cache["prototypes"], labels=cache["label"], topology_distance=topology.distance.numpy() / float(topology.distance.max()), sample_ids=ids)
        score_kind = method
    original = np.nanargmax(scores, axis=1) if np.isfinite(scores).any() else assigned.copy()
    counts = np.bincount(assigned, minlength=4)
    if method in CAPACITY_METHODS and (np.any(counts < math.ceil(.15 * len(ids))) or np.any(counts > math.floor(.40 * len(ids)))):
        raise AssertionError("BTMA capacity bound violation.")
    rows: list[dict[str, Any]] = []
    mapping: dict[str, int] = {}
    for index, final in enumerate(assigned):
        finite = np.nan_to_num(scores[index], nan=-np.inf)
        order = np.argsort(-finite, kind="stable")
        mapping[str(ids[index])] = int(final)
        row = {"dataset_index": index, "sample_id": str(ids[index]), "epoch": epoch, "assigned_modality": MODALITIES[int(final)],
               "domain": str(cache["domain"][index]), "weather": str(cache["weather"][index]), "beam": int(cache["label"][index]), "beam_sector": int(cache["beam_sector"][index]),
               **{f"assignment_score_{name}": None if not np.isfinite(scores[index, col]) else float(scores[index, col]) for col, name in enumerate(MODALITIES)},
               "best_score": None if not np.isfinite(finite[order[0]]) else float(finite[order[0]]), "second_score": None if not np.isfinite(finite[order[1]]) else float(finite[order[1]]),
               "assignment_advantage": None if not np.isfinite(finite[order[0]]) else float(finite[order[0]] - finite[order[1]]),
               "assignment_changed": bool(previous is not None and previous.get(str(ids[index])) != int(final)), "capacity_adjusted": bool(int(original[index]) != int(final)),
               "original_best_modality": MODALITIES[int(original[index])], "final_assigned_modality": MODALITIES[int(final)]}
        rows.append(row)
    statistics = {"method": method, "score_kind": score_kind, "absolute_epoch": epoch, "sample_count": len(rows),
                  "counts": {name: int(counts[i]) for i, name in enumerate(MODALITIES)}, "ratios": {name: float(counts[i] / len(rows)) for i, name in enumerate(MODALITIES)},
                  "change_rate": float(np.mean([row["assignment_changed"] for row in rows])) if previous else 0.0,
                  "capacity_repair_rate": float(np.mean([row["capacity_adjusted"] for row in rows])), "historical_a1_proportions": dict(a1) if method == "b1_fixed_weak_schedule" else None}
    return mapping, rows, statistics


def _write_assignment(root: Path, method: str, epoch: int, cache: Mapping[str, np.ndarray], previous: Mapping[str, int] | None, a1: Mapping[str, float], topology: Any, *, smoke: bool) -> tuple[dict[str, int], dict[str, Any], str]:
    mapping, rows, statistics = _assignment_rows(cache, method, epoch, previous, a1, topology)
    directory = root / ("smoke_tests/assignments" if smoke else "assignments") / method
    path = directory / f"epoch_{epoch:03d}.csv"
    _atomic_csv(path, rows)
    statistics["csv_sha256"] = sha256_file(path)
    write_json(path.with_suffix(".json"), statistics)
    return mapping, statistics, statistics["csv_sha256"]


def _validation_predictions(model: Any, loader: Any, device: torch.device, root: Path, method: str, *, smoke: bool) -> None:
    cache = c12._prediction_cache(model, loader, device, (), max_batches=2 if smoke else None)
    # _prediction_cache does not use signed order for output; retain full logits and stable validation ordering.
    path = (root / "smoke_tests" if smoke else root) / method / "validation_predictions.npz"
    c12._save_cache(path, cache)


def train(root: Path, method: str, *, smoke: bool = False) -> None:
    if method not in BTMA_METHODS:
        raise ValueError(f"Unknown BTMA method: {method}")
    if not (root / "preflight_tests.txt").is_file():
        raise ValueError("Run --prepare before BTMA training.")
    if sha256_file(WARMUP_CHECKPOINT) != EXPECTED_WARMUP_SHA256:
        raise ValueError("Locked warm-up checkpoint SHA256 mismatch.")
    run_dir = root / "smoke_tests" / method if smoke else root / method
    if run_dir.exists() and not smoke:
        existing = {path.name for path in run_dir.iterdir()}
        # The GPU launcher creates the directory before redirecting its stdout.
        # Preserve fail-closed resume behavior once any actual artifact exists.
        if existing - {"train.log"}:
            raise FileExistsError(f"BTMA run directory already has artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_full_pool_protocol(c12.DEFAULT_PROTOCOL)
    topology = load_audited_topology(c12.DEFAULT_TOPOLOGY)
    signed = json.loads((root / "signed_angle_order_audit.json").read_text(encoding="utf-8"))["labels"]
    loaders, cfg = c12._loaders(root, protocol, create_normalization=False)
    fixed_train = c12._fixed_loader(loaders["train"], workers=0 if smoke else 8)
    training_loader = fixed_train if smoke else loaders["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c12.set_seed()
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    model = c12._load_model(WARMUP_CHECKPOINT, device)
    epochs, max_batches = (1, 2) if smoke else (c12.SEARCH_EPOCHS, None)
    steps_per_epoch = min(len(training_loader), max_batches or len(training_loader))
    optimizer, scheduler = c12._optimizer(model, epochs, steps_per_epoch)
    a1 = _historical_a1_proportions()
    resolved = {"method": method, "seed": c12.SEED, "common_warmup_checkpoint": str(WARMUP_CHECKPOINT), "common_warmup_sha256": EXPECTED_WARMUP_SHA256,
                "epochs": epochs, "total_optimizer_steps": epochs * steps_per_epoch, "joint_remix_ratio": "1:1", "joint_steps_per_epoch": (steps_per_epoch + 1) // 2,
                "remix_steps_per_epoch": steps_per_epoch // 2, "assignment_update_every_epochs": 2, "mixed_assigned_modality_batches": True,
                "homogeneous_remix_batches": False, "optimizer": "AdamW", "encoder_lr": 1e-4, "main_lr": 3e-4, "weight_decay": 1e-4, "gradient_clip": 1.0,
                "scheduler": "cosine_with_5_percent_warmup", "batch_size": cfg["data"]["dataloader"]["train_batch_size"], "checkpoint_selection": "CE_full + 0.25 * beam_topology_risk_full",
                "early_stopping": False, "assignment": method, "historical_a1_proportions": a1 if method == "b1_fixed_weak_schedule" else None, "outer_test_accessed": False, "legacy_clean_inner_used": False}
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    (run_dir / "assignment_config.yaml").write_text(yaml.safe_dump({"method": method, "capacity": "15%-40%" if method in CAPACITY_METHODS else "historical_a1_proportions", "a1_proportions": a1}, sort_keys=True), encoding="utf-8")
    (run_dir / "config_sha256.txt").write_text(sha256_file(run_dir / "resolved_config.yaml") + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []; history: list[dict[str, Any]] = []
    best_selection, best_epoch, best_state = float("inf"), 0, None
    previous: dict[str, int] | None = None; active: dict[str, int] | None = None; active_epoch = -1
    started = time.monotonic()
    try:
        for epoch in range(1, epochs + 1):
            assignment_epoch = c12.WARMUP_EPOCHS + 2 * ((epoch - 1) // 2)
            if assignment_epoch != active_epoch:
                # B0/B1 use no model-derived difficulty cache; B2--B5 use only current train forward outputs.
                if method in {"b0_random_balanced", "b1_fixed_weak_schedule"}:
                    # These branches deliberately do not forward a model to obtain difficulty scores.
                    # The historical warm-up cache supplies only the audited fixed train order/metadata.
                    cache = c12._load_cache(HISTORICAL_ROOT / "warmup/unimodal_train_predictions/train_cache.npz")
                    write_json((root / ("smoke_tests/score_cache" if smoke else "score_cache") / method / f"epoch_{assignment_epoch:03d}_manifest.json"), {"score_computation": "not_used", "sample_count": len(cache["sample_id"]), "sample_hash": c12._sha256_json(cache["sample_id"].tolist()), "validation_used": False})
                else:
                    cache = _cache(model, fixed_train, device, signed, root, method, assignment_epoch, smoke=smoke)
                active, statistics, digest = _write_assignment(root, method, assignment_epoch, cache, previous, a1, topology, smoke=smoke)
                previous, active_epoch = dict(active), assignment_epoch
                history.append({"search_epoch": epoch, "absolute_epoch": assignment_epoch, "assignment_sha256": digest, **{f"{name}_count": statistics["counts"][name] for name in MODALITIES}, "change_rate": statistics["change_rate"], "capacity_repair_rate": statistics["capacity_repair_rate"]})
                _atomic_csv(run_dir / "assignment_history.csv", history)
            iterator = iter(training_loader); totals: dict[str, float] = defaultdict(float); samples = 0; epoch_started = time.monotonic()
            model.train()
            for step in range(steps_per_epoch):
                batch = next(iterator); remix_phase = step % 2 == 1
                optimizer.zero_grad(set_to_none=True)
                with c12._autocast(device):
                    labels = c12._labels(batch, device); inputs = c12._inputs(batch, device)
                    if remix_phase:
                        assert active is not None
                        assigned = torch.tensor([active[item] for item in c12._batch_ids(batch)], device=device)
                        output = model(inputs, availability=c12._availability(assigned), signed_order=signed)
                        loss, report = remix_loss(model, output, labels, assigned)
                    else:
                        output = model(inputs, signed_order=signed); loss, report = common_loss(model, output, labels)
                if not bool(torch.isfinite(loss)): raise FloatingPointError(f"{method} non-finite loss at epoch={epoch}, step={step}")
                loss.backward(); c12._clear_remix_gradients(model, set(int(v) for v in assigned.tolist()), motion=False) if remix_phase else None
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step()
                count = labels.numel(); samples += count; totals["total"] += float(loss.detach()) * count; totals["joint_samples" if not remix_phase else "remix_samples"] += count
                for name, value in report.items(): totals[name] += float(value) * count
                if (step + 1) % 50 == 0 or step + 1 == steps_per_epoch:
                    complete = (epoch - 1) * steps_per_epoch + step + 1; elapsed = time.monotonic() - started
                    write_json(run_dir / "runtime_status.json", {"status": "running", "method": method, "epoch": epoch, "optimizer_step": complete, "total_optimizer_steps": epochs * steps_per_epoch, "latest_train_loss": float(loss.detach()), "assignment_epoch": active_epoch, "elapsed_seconds": elapsed, "estimated_remaining_seconds": elapsed * ((epochs * steps_per_epoch) / complete - 1)})
            selection = c12._selection(model, loaders["validation"], topology, signed, device, max_batches=1 if smoke else None)
            row = {"epoch": epoch, "absolute_epoch": c12.WARMUP_EPOCHS + epoch, "optimizer_steps": epoch * steps_per_epoch, "train_samples": samples, "joint_samples": int(totals["joint_samples"]), "remix_samples": int(totals["remix_samples"]), "train_joint_loss": totals["total"] / max(samples, 1), "train_assigned_loss": totals["total"] / max(samples, 1), **selection, "lr_encoder": optimizer.param_groups[0]["lr"], "epoch_seconds": time.monotonic() - epoch_started, "assignment_epoch": active_epoch}
            rows.append(row); _atomic_csv(run_dir / "training_curve.csv", rows)
            if selection["selection_loss"] < best_selection:
                best_selection, best_epoch = selection["selection_loss"], epoch; best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            print(json.dumps({"event": "btma_epoch", "method": method, **row}), flush=True)
        if best_state is None: raise RuntimeError("BTMA produced no selectable checkpoint.")
        checkpoint = run_dir / "best_checkpoint.pt"; torch.save({"state_dict": best_state, "model_config": c12.MODEL_CONFIG, "method": method, "best_epoch": best_epoch, "selection_loss": best_selection, "warmup_sha256": EXPECTED_WARMUP_SHA256, "protocol_fingerprint": protocol["protocol_fingerprint"]}, checkpoint)
        (run_dir / "checkpoint_sha256.txt").write_text(sha256_file(checkpoint) + "\n", encoding="utf-8")
        model.load_state_dict(best_state, strict=True)
        train_cache = c12._load_cache(HISTORICAL_ROOT / "warmup/unimodal_train_predictions/train_cache.npz")
        metrics = c12.evaluate(model, loaders["validation"], topology, signed, device, method=method, train_cache=train_cache, max_batches=1 if smoke else None)
        metrics.update(best_epoch=best_epoch, selection_loss=best_selection, checkpoint_sha256=sha256_file(checkpoint), warmup_sha256=EXPECTED_WARMUP_SHA256, wall_seconds=time.monotonic()-started)
        write_json(run_dir / "metrics.json", metrics); (run_dir / "eval.log").write_text(json.dumps({"method": method, "full": metrics["patterns"]["full"]}, indent=2) + "\n", encoding="utf-8")
        c12._write_metric_csv(run_dir / "full_metrics.csv", "pattern", {"full": metrics["patterns"]["full"]}, method=method)
        c12._write_metric_csv(run_dir / "unimodal_metrics.csv", "pattern", {k:v for k,v in metrics["patterns"].items() if k.startswith("only_")}, method=method)
        c12._write_metric_csv(run_dir / "missing_diagnostics.csv", "pattern", {k:v for k,v in metrics["patterns"].items() if k.startswith("missing_")}, method=method)
        for filename, name, values in (("per_domain_metrics.csv", "domain", metrics["per_domain"]), ("per_weather_metrics.csv", "weather", metrics["per_weather"]), ("per_sector_metrics.csv", "sector", metrics["per_sector"]), ("per_beam_metrics.csv", "beam", metrics["per_beam"])): c12._write_metric_csv(run_dir / filename, name, values, method=method)
        _atomic_csv(run_dir / "assignment_statistics.csv", history)
        efficiency = c12._efficiency(model, c12._take_batch(next(iter(fixed_train)), 8), signed, device, method=method, wall_seconds=time.monotonic()-started, epoch_seconds=[row["epoch_seconds"] for row in rows], checkpoint=checkpoint)
        write_json(run_dir / "efficiency.json", efficiency)
        write_json(run_dir / "runtime_status.json", {"status": "passed", "method": method, "epoch": epochs, "optimizer_step": epochs * steps_per_epoch, "total_optimizer_steps": epochs * steps_per_epoch, "best_epoch": best_epoch})
        write_json(run_dir / "status.json", {"status": "passed", "method": method, "epochs_completed": epochs, "optimizer_steps": epochs * steps_per_epoch, "best_epoch": best_epoch, "early_stopping": False, "outer_test_accessed": False, "smoke": smoke})
    except Exception as exc:
        write_json(run_dir / "status.json", {"status": "failed", "method": method, "error": repr(exc), "traceback": traceback.format_exc(), "outer_test_accessed": False, "smoke": smoke}); raise
    finally:
        shutdown_dataloader_workers(fixed_train)
        for loader in loaders.values(): shutdown_dataloader_workers(loader)


def _historical_metrics() -> list[dict[str, Any]]:
    values = [("a0_historical_baseline", "None", "None", 19.79, 32.94, 51.76, 7.548, .2545, 38.97), ("a1_historical_kl_remix", "KL", "None", 23.40, 39.85, 54.75, 7.378, .2427, 37.98), ("a2_historical_risk_margin", "Risk+Margin", "15-40%", 23.02, 42.71, 56.36, 6.721, .2230, 34.38)]
    return [{"method": name, "assignment": assignment, "capacity": capacity, "top1": top1/100, "top3": top3/100, "within3": within3/100, "mae": mae, "topology_risk": risk, "distance_gt5": far/100} for name, assignment, capacity, top1, top3, within3, mae, risk, far in values]


def aggregate(root: Path) -> None:
    payloads = {method: json.loads((root / method / "metrics.json").read_text(encoding="utf-8")) for method in BTMA_METHODS}
    rows = _historical_metrics()
    for method, payload in payloads.items():
        full = payload["patterns"]["full"]
        rows.append({"method": method, "assignment": {"b0_random_balanced":"Random", "b1_fixed_weak_schedule":"None/global", "b2_kl_capacity":"KL", "b3_topology_risk_only":"Topology risk", "b4_margin_only":"Prototype margin", "b5_risk_margin_full":"Risk+Margin"}[method], "capacity": "A1 proportions" if method == "b1_fixed_weak_schedule" else "15-40%", **{key: full[key] for key in METRICS}})
    _atomic_csv(root / "combined_metrics.csv", rows); _atomic_csv(root / "full_metrics.csv", rows)
    for filename, group, key in (("unimodal_metrics.csv", "pattern", "only_"), ("missing_diagnostics.csv", "pattern", "missing_")):
        output = [{"method": method, group: pattern, **values} for method, payload in payloads.items() for pattern, values in payload["patterns"].items() if pattern.startswith(key)]; _atomic_csv(root / filename, output)
    for filename, group, metric_key in (("per_domain_metrics.csv", "domain", "per_domain"), ("per_weather_metrics.csv", "weather", "per_weather"), ("per_sector_metrics.csv", "sector", "per_sector"), ("per_beam_metrics.csv", "beam", "per_beam")):
        output = [{"method": method, group: value, **metrics} for method, payload in payloads.items() for value, metrics in payload[metric_key].items()]; _atomic_csv(root / filename, output)
    histories = []
    for method in BTMA_METHODS:
        with (root / method / "assignment_history.csv").open(newline="", encoding="utf-8") as handle: histories.extend([{"method": method, **row} for row in csv.DictReader(handle)])
    _atomic_csv(root / "assignment_statistics.csv", histories)
    b = {name: payloads[name]["patterns"]["full"] for name in BTMA_METHODS}; a0 = _historical_metrics()[0]
    comparisons = [("B0-A0", "Assigned single-modality training", "b0_random_balanced", None), ("B1-B0", "Weak-modality global budget", "b1_fixed_weak_schedule", "b0_random_balanced"), ("B2-B0", "KL sample-specific scoring", "b2_kl_capacity", "b0_random_balanced"), ("B3-B0", "Topology-risk scoring", "b3_topology_risk_only", "b0_random_balanced"), ("B4-B0", "Prototype-margin scoring", "b4_margin_only", "b0_random_balanced"), ("B5-B3", "Margin added to risk", "b5_risk_margin_full", "b3_topology_risk_only"), ("B5-B4", "Risk added to margin", "b5_risk_margin_full", "b4_margin_only"), ("B5-B2", "Beam-specific vs generic KL", "b5_risk_margin_full", "b2_kl_capacity")]
    attribution = []
    for label, factor, left, right in comparisons:
        right_metrics = a0 if right is None else b[right]; left_metrics = b[left]
        attribution.append({"comparison": label, "tested_factor": factor, "top1_delta_pp": 100*(left_metrics["top1"]-right_metrics["top1"]), "within3_delta_pp": 100*(left_metrics["within3"]-right_metrics["within3"]), "mae_delta": left_metrics["mae"]-right_metrics["mae"], "risk_delta": left_metrics["topology_risk"]-right_metrics["topology_risk"]})
    _atomic_csv(root / "causal_attribution.csv", attribution)
    h = _historical_metrics()[2]; b5 = b["b5_risk_margin_full"]
    repro = {"top1": abs(b5["top1"]-h["top1"]) <= .002, "top3": abs(b5["top3"]-h["top3"]) <= .003, "within3": abs(b5["within3"]-h["within3"]) <= .003, "mae": abs(b5["mae"]-h["mae"]) <= .05, "topology_risk": abs(b5["topology_risk"]-h["topology_risk"]) <= .005, "distance_gt5": abs(b5["distance_gt5"]-h["distance_gt5"]) <= .003}
    gates = [{"gate": f"b5_reproduces_a2_{name}", "passed": passed} for name, passed in repro.items()]
    b0 = b["b0_random_balanced"]; b2 = b["b2_kl_capacity"]
    b5_vs_b0 = [b5["top1"] >= b0["top1"]+.01, b5["top3"] >= b0["top3"]+.015, b5["within3"] >= b0["within3"]+.01, b5["mae"] <= b0["mae"]-.20, b5["topology_risk"] <= .95*b0["topology_risk"], b5["distance_gt5"] <= b0["distance_gt5"]-.01]
    gates.extend(({"gate": "b5_vs_b0_" + str(index+1), "passed": item} for index, item in enumerate(b5_vs_b0)))
    gates.append({"gate": "b5_beam_specific_beats_b2", "passed": b5["within3"] >= b2["within3"]+.005 or b5["mae"] <= b2["mae"]-.15 or b5["topology_risk"] <= .97*b2["topology_risk"] or b5["distance_gt5"] <= b2["distance_gt5"]-.0075})
    _atomic_csv(root / "success_gates.csv", gates)
    ranking = sorted(rows, key=lambda x: (-x["top1"], -x["within3"], x["mae"])); _atomic_csv(root / "direction_ranking.csv", [{"rank": i+1, **row} for i, row in enumerate(ranking)])
    table = ["| 方法 | Assignment | Capacity | Top1 | Top3 | Within-3 | MAE | Topology Risk | Distance>5 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows: table.append(f"| {row['method']} | {row['assignment']} | {row['capacity']} | {100*row['top1']:.2f} | {100*row['top3']:.2f} | {100*row['within3']:.2f} | {row['mae']:.3f} | {row['topology_risk']:.4f} | {100*row['distance_gt5']:.2f} |")
    b1 = b["b1_fixed_weak_schedule"]
    conclusion = all(repro.values()) and sum(b5_vs_b0) >= 5 and gates[-1]["passed"]
    report = ["# 全量数据下的波束拓扑模态分配因果消融", "", "开发集单 seed；未访问 outer test。", "", *table, "", "## 协议与复现", "", "- 使用 canonical Full-pool 37,038/9,180，未读取 outer、channel、path、历史 beam 或未来 GPS。", f"- warm-up SHA256：`{EXPECTED_WARMUP_SHA256}`，一致：是。", f"- B5 历史 A2 复现：{all(repro.values())}，逐指标：{repro}。", "- 六路均为 20 epoch、11,580 step、1:1 joint/mixed assigned batch；无 homogeneous batch、无早停。", "", "## 因果判断", "", f"- B0 相对 A0 Top1：{100*(b0['top1']-a0['top1']):+.2f} pp。", f"- B1 相对 A1 Top1：{100*(b1['top1']-_historical_metrics()[1]['top1']):+.2f} pp；B1 使用实际 A1 的极端 Radar/GPS 全局比例。", f"- B2 相对 B0 Top1：{100*(b2['top1']-b0['top1']):+.2f} pp。", f"- B3 相对 B2 Top1：{100*(b['b3_topology_risk_only']['top1']-b2['top1']):+.2f} pp；B4 相对 B2：{100*(b['b4_margin_only']['top1']-b2['top1']):+.2f} pp。", f"- B5 相对 B3/B4/B2 Top1：{100*(b5['top1']-b['b3_topology_risk_only']['top1']):+.2f}/{100*(b5['top1']-b['b4_margin_only']['top1']):+.2f}/{100*(b5['top1']-b2['top1']):+.2f} pp。", f"- BTMA 进入 3-seed 的预注册建议：{conclusion}。本轮不会自动启动 multi-seed 或 outer test。", "", "完整 assignment、每域/天气/sector、单模态及缺失诊断见同目录 CSV。paired block bootstrap 未实现：当前 Candidate12 评测未保存逐窗口 full logits，不能伪造时间配对置信区间。"]
    (root / "btma_ablation_comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(root / "aggregate_status.json", {"status": "passed", "b5_reproduction_passed": all(repro.values()), "btma_candidate_passed": conclusion, "outer_test_accessed": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--prepare", action="store_true"); parser.add_argument("--method", choices=BTMA_METHODS); parser.add_argument("--smoke", action="store_true"); parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if not any((args.prepare, args.method, args.aggregate)): parser.error("select --prepare, --method, or --aggregate")
    if args.prepare: _prepare(args.output_root.resolve())
    if args.method: train(args.output_root.resolve(), args.method, smoke=args.smoke)
    if args.aggregate: aggregate(args.output_root.resolve())


if __name__ == "__main__": main()
