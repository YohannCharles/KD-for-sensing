#!/usr/bin/env python3
"""Prepare, train, and summarize the four sealed trajectory baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from kd_sensing.baselines.full_pool_bt_scl import load_audited_topology, topology_risk
from kd_sensing.baselines.full_pool_common import atomic_csv, now, sha256_file, write_json
from kd_sensing.baselines.mmw_trajectory import (
    METHODS,
    RANDOM_BALANCED_METHOD,
    TrajectoryBaselineModel,
    availability_for_assignments,
    baseline_loss,
    model_contract,
    random_balanced_assignment,
)
from kd_sensing.config import load_config
from kd_sensing.data.mmw.trajectory_protocol import (
    AUDIT_IDENTITIES,
    TRAJECTORY_PROTOCOL_ID,
    TRAJECTORY_PROTOCOL_MODE,
    build_trajectory_protocol,
    load_trajectory_protocol,
    protocol_dataset_domains,
)
from kd_sensing.data.transform_ops.gps import load_gps_coordinate_cache, load_gps_scaler, read_gps_latlon
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices
from kd_sensing.engine.normalization_artifacts import (
    save_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/mmw_trajectory_split"
DEFAULT_PROTOCOL = DEFAULT_OUTPUT / "protocol/split_manifest.json"
DEFAULT_AUDIT = DEFAULT_OUTPUT / "protocol/split_audit.json"
DEFAULT_TOPOLOGY = ROOT / "outputs/cache/mmw_codebook_topology/v1/a692c2b43365b483/topology_manifest.json"
SEED = 2026
EPOCHS = 20
BATCH_SIZE = 64
PATTERNS = {
    "full": (1, 1, 1, 1),
    "only_image": (1, 0, 0, 0),
    "only_lidar": (0, 1, 0, 0),
    "only_radar": (0, 0, 1, 0),
    "only_gps": (0, 0, 0, 1),
    "missing_image": (0, 1, 1, 1),
    "missing_lidar": (1, 0, 1, 1),
    "missing_radar": (1, 1, 0, 1),
    "missing_gps": (1, 1, 1, 0),
}
METRICS = ("top1", "top3", "top5", "within1", "within3", "mae", "topology_risk", "distance_gt5", "ce_loss")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(protocol: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    cfg = load_config(ROOT / "configs/mmw/u0.yaml")
    cfg["experiment"].update(name="MMW_Trajectory_Disjoint", seed=SEED, device="auto")
    cfg["data"]["dataset"].update(
        domains=protocol_dataset_domains(protocol),
        portion=1.0,
        frame_cache_root=str((ROOT / "outputs/cache/MMW").resolve()),
        frame_cache_strict=True,
        gps_coordinate_cache_root=str((output_root / "cache/gps_coordinates").resolve()),
        include_router_utility_targets=False,
        include_router_corruption_metadata=False,
        lidar_augment=False,
    )
    cfg["data"]["domain_balanced_sampling"]["enabled"] = False
    cfg["data"]["dataloader"].update(
        train_batch_size=BATCH_SIZE,
        validation_batch_size=BATCH_SIZE,
        test_batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        train_drop_last=False,
        test_drop_last=False,
    )
    cfg["data_protocol"] = {
        "mode": TRAJECTORY_PROTOCOL_MODE,
        "path": str((output_root / "protocol/split_manifest.json").resolve()),
        "audit_report": str((output_root / "protocol/split_audit.json").resolve()),
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "train_role": protocol["train_role"],
        "validation_role": protocol["validation_role"],
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
        "allow_test_evaluation": False,
    }
    cfg["training"].update(epochs=EPOCHS, max_epochs=EPOCHS, lr=3e-4, weight_decay=1e-4, resume=False)
    cfg["training"]["final_test"] = {"enabled": False}
    cfg["output"].update(dir=str(output_root), run_name="trajectory_baselines", overwrite=False)
    return cfg


def _gps_paths(csv_path: str | Path) -> set[str]:
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [name for name in (reader.fieldnames or []) if name.startswith("gps") or name.startswith("bs_gps")]
        return {value for row in reader for name in columns if (value := str(row.get(name, "")).strip())}


def ensure_gps_coordinate_caches(output_root: Path, protocol: Mapping[str, Any]) -> Path:
    source_root = ROOT / "outputs/full_pool_capacity/cache/gps_coordinates"
    target_root = output_root / "cache/gps_coordinates"
    target_root.mkdir(parents=True, exist_ok=True)
    records = []
    for domain in protocol["domains"]:
        split_paths = [domain[key] for key in ("train_split", "validation_split") if domain.get(key)]
        if not split_paths:
            continue
        required = set().union(*(_gps_paths(path) for path in split_paths))
        name = f"{domain['condition']}__{domain['scene']}.npz"
        source = source_root / name
        target = target_root / name
        coordinates = load_gps_coordinate_cache(target) if target.is_file() else {}
        inherited = load_gps_coordinate_cache(source)
        reused = required & (coordinates.keys() | inherited.keys())
        missing = sorted(required - reused)
        coordinates.update({path: inherited[path] for path in required & inherited.keys()})
        coordinates.update({path: read_gps_latlon(domain["data_root"], path) for path in missing})
        if set(coordinates) != required:
            coordinates = {path: coordinates[path] for path in required}
        ordered = sorted(required)
        if not target.is_file() or set(load_gps_coordinate_cache(target)) != required or missing:
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    paths=np.asarray(ordered),
                    coordinates=np.asarray([coordinates[path] for path in ordered], dtype=np.float64),
                )
            temporary.replace(target)
        records.append(
            {
                "domain_id": domain["id"],
                "coordinate_count": len(required),
                "reused_coordinate_count": len(reused),
                "parsed_coordinate_count": len(missing),
            }
        )
    write_json(
        target_root / "manifest.json",
        {
            "schema_version": 1,
            "protocol_fingerprint": protocol["protocol_fingerprint"],
            "source_cache_root": str(source_root.resolve()),
            "strict_cache_coverage": True,
            "outer_test_accessed": False,
            "domains": records,
        },
    )
    return target_root


def _loaders(
    output_root: Path, protocol: Mapping[str, Any], *, create_normalization: bool
) -> tuple[dict[str, DataLoader], dict[str, Any], dict[str, Any]]:
    cfg = build_config(protocol, output_root)
    artifact = output_root / "artifacts/gps_scaler.npz"
    manifest_path = output_root / "normalization_manifest.json"
    if artifact.is_file():
        loaders = build_dataloaders(cfg, normalization_overrides={"gps_scaler": load_gps_scaler(artifact)})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif create_normalization:
        loaders = build_dataloaders(cfg)
        manifest = save_normalization_artifacts(loaders, output_root)
        write_json(manifest_path, manifest)
    else:
        raise FileNotFoundError("Trajectory train-only GPS scaler is absent; run --prepare first.")
    validate_normalization_artifact_fingerprint(cfg, {"normalization_artifacts": manifest})
    metadata = manifest.get("metadata", {})
    if metadata.get("source_split") != "train" or int(metadata.get("effective_sample_count", -1)) != int(protocol["train_window_count"]):
        raise ValueError("Trajectory normalization is not bound to the complete train split.")
    return loaders, cfg, manifest


def _fixed_loader(loader: DataLoader, *, workers: int) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=workers,
        pin_memory=bool(workers),
        persistent_workers=bool(workers),
        prefetch_factor=2 if workers else None,
        collate_fn=loader.collate_fn,
        drop_last=False,
    )


def _stable_train_ids(dataset: Any) -> list[str]:
    values: list[str] = []
    for leaf, indices in leaf_datasets_with_indices(dataset):
        rows = getattr(getattr(leaf, "samples", None), "rows", None)
        if not isinstance(rows, list):
            raise ValueError("Trajectory assignment requires prepared sample rows.")
        for index in indices:
            sample_id = str(rows[int(index)].get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError("Trajectory assignment requires stable sample ids.")
            values.append(f"mmw:{leaf.condition}:{leaf.scene_slug}:train:{sample_id}")
    if len(values) != len(set(values)):
        raise ValueError("Trajectory train sample ids are not unique.")
    return values


def prepare(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = build_trajectory_protocol(output_root)
    audit = json.loads((output_root / "protocol/split_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("outer_test_accessed") is not False:
        raise ValueError("Trajectory prepare requires a passed, prediction-unaccessed split audit.")
    ensure_gps_coordinate_caches(output_root, protocol)
    loaders, cfg, normalization = _loaders(output_root, protocol, create_normalization=True)
    try:
        checks = {
            "protocol_id": protocol["protocol_id"] == TRAJECTORY_PROTOCOL_ID,
            "candidate_windows": int(protocol["candidate_window_count"]) == 46_860,
            "group_count": int(protocol["trajectory_group_count"]) == 15,
            "group_split": tuple(int(protocol[f"{role}_group_count"]) for role in ("train", "validation", "test")) == (12, 2, 1),
            "window_split": sum(int(protocol[f"{role}_window_count"]) for role in ("train", "validation", "test")) == 46_860,
            "pairwise_resource_zero": all(
                int(value["count"]) == 0 for pair in audit["pairwise_overlaps"].values() for value in pair.values()
            ),
            "all_resource_families_audited": all(set(pair) == set(AUDIT_IDENTITIES) for pair in audit["pairwise_overlaps"].values()),
            "test_loader_absent": set(loaders) == {"train", "validation"},
            "train_windows": len(loaders["train"].dataset) == int(protocol["train_window_count"]),
            "validation_windows": len(loaders["validation"].dataset) == int(protocol["validation_window_count"]),
            "normalization_train_only": normalization["metadata"]["sample_id_hash"] == audit["train_sample_id_hash"],
            "outer_test_unaccessed": protocol["outer_test_accessed"] is False,
            "no_channel_model_input": set(cfg["model"]["modalities"]) == {"image", "radar", "gps", "lidar"},
        }
        contracts = {}
        backbone_hashes = {}
        for method in METHODS:
            set_seed()
            model = TrajectoryBaselineModel(method)
            contracts[method] = model_contract(model)
            backbone_hashes[method] = _state_hash(
                {key: value for key, value in model.state_dict().items() if not key.startswith(("linear_head", "prototype_bank"))}
            )
        checks["same_encoder_fusion_initialization"] = len(set(backbone_hashes.values())) == 1
        checks["methods_exact"] = tuple(contracts) == METHODS
        checks["motion_absent"] = all(not value["motion_branch_present"] for value in contracts.values())
        if not all(checks.values()):
            raise ValueError(f"Trajectory preflight failed: {checks}")
        write_json(output_root / "model_contracts.json", {"contracts": contracts, "backbone_hashes": backbone_hashes})
        lines = [f"{index:02d}_{name}=passed:{value}" for index, (name, value) in enumerate(checks.items(), 1)]
        (output_root / "preflight_tests.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (output_root / "implementation_notes.md").write_text(
            "# Implementation Notes\n\n"
            "- Protocol: mmw_trajectory_disjoint_v1; 5 history frames, 1 future beam label, 64 classes.\n"
            "- Inputs: Image, LiDAR, Radar, GPS only; channel/path/beam power/history beam/future GPS are excluded.\n"
            "- Ordinary training exposes train and validation only; sealed test prediction is unavailable.\n"
            "- M0--M3 share Candidate12 encoders/fusion and differ only in the declared head/loss/training mask.\n",
            encoding="utf-8",
        )
        write_json(output_root / "prepare_status.json", {"status": "passed", "checks": checks, "outer_test_accessed": False, "at": now()})
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def _state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    image = torch.as_tensor(batch["image"]).to(device=device, non_blocking=True)
    lidar = torch.as_tensor(batch["lidar"]).to(device=device, non_blocking=True)
    radar_ra = torch.as_tensor(batch["radar_ra"]).to(device=device, non_blocking=True)
    radar_da = torch.as_tensor(batch["radar_da"]).to(device=device, non_blocking=True)
    if radar_ra.ndim == 4:
        radar_ra, radar_da = radar_ra.unsqueeze(2), radar_da.unsqueeze(2)
    radar = torch.cat((radar_ra, radar_da), dim=2)
    gps = torch.as_tensor(batch["gps"]).to(device=device, non_blocking=True)
    return {"image": image, "lidar": lidar, "radar": radar, "gps": gps}


def _labels(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    labels = torch.as_tensor(batch["target_beam"], dtype=torch.long, device=device).reshape(-1)
    if not bool(((labels >= 0) & (labels < 64)).all()):
        raise ValueError("Trajectory labels must be in [0, 63].")
    return labels


def _batch_ids(batch: Mapping[str, Any]) -> list[str]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping) or "stable_sample_id" not in metadata:
        raise ValueError("Trajectory batch lacks stable sample identity metadata.")
    return [str(value) for value in metadata["stable_sample_id"]]


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _optimizer(model: TrajectoryBaselineModel, epochs: int, steps_per_epoch: int):
    encoder_ids = {id(parameter) for parameter in model.encoders.parameters()}
    encoder = [parameter for parameter in model.parameters() if id(parameter) in encoder_ids]
    main = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": 1e-4}, {"params": main, "lr": 3e-4}], weight_decay=1e-4)
    total = max(1, epochs * steps_per_epoch)
    warmup = max(1, int(total * 0.05))

    def factor(step: int) -> float:
        current = min(step + 1, total)
        if current <= warmup:
            return current / warmup
        progress = (current - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _new_bucket() -> dict[str, float]:
    return defaultdict(float)


def _metric_values(logits: torch.Tensor, labels: torch.Tensor, topology: Any) -> torch.Tensor:
    prediction = logits.argmax(-1)
    distance = topology.distance.to(logits.device)[labels, prediction].float()
    top = logits.topk(5, dim=-1).indices
    return (
        torch.stack(
            (
                prediction.eq(labels),
                top[:, :3].eq(labels[:, None]).any(-1),
                top.eq(labels[:, None]).any(-1),
                distance.le(1),
                distance.le(3),
                distance,
                topology_risk(logits.float(), labels, topology),
                distance.gt(5),
                F.cross_entropy(logits.float(), labels, reduction="none"),
            ),
            dim=1,
        )
        .double()
        .cpu()
    )


def _update_bucket(bucket: dict[str, float], values: torch.Tensor) -> None:
    bucket["count"] += values.shape[0]
    for name, value in zip(METRICS, values.sum(0).tolist()):
        bucket[f"{name}_sum"] += float(value)


def _finish_bucket(bucket: Mapping[str, float]) -> dict[str, float]:
    count = max(float(bucket.get("count", 0)), 1.0)
    return {
        "sample_count": int(bucket.get("count", 0)),
        **{name: float(bucket.get(f"{name}_sum", 0.0)) / count for name in METRICS},
    }


def _indices_by(values: Sequence[Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        result[str(value)].append(index)
    return result


def evaluate(
    model: TrajectoryBaselineModel,
    loader: DataLoader,
    topology: Any,
    device: torch.device,
    *,
    train_beam_support: np.ndarray,
    detailed: bool,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    patterns = PATTERNS if detailed else {"full": PATTERNS["full"]}
    pattern_buckets = {name: _new_bucket() for name in patterns}
    group_buckets = {name: defaultdict(_new_bucket) for name in ("trajectory", "domain", "weather", "sector", "beam", "frequency")}
    positions = np.empty(64, dtype=np.int64)
    positions[np.asarray(topology.labels_by_position, dtype=np.int64)] = np.arange(64)
    order = np.argsort(train_beam_support, kind="stable")
    frequency = np.empty(64, dtype="<U4")
    frequency[order[:21]], frequency[order[21:43]], frequency[order[43:]] = "tail", "mid", "head"
    started = time.monotonic()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            with _autocast(device):
                labels = _labels(batch, device)
                tokens = model.encode(_inputs(batch, device))
                outputs = {
                    name: model.forward_tokens(
                        tokens,
                        availability=torch.tensor(mask, device=device, dtype=torch.bool).expand(labels.numel(), -1),
                    )
                    for name, mask in patterns.items()
                }
            values = {}
            for name, output in outputs.items():
                metric_values = _metric_values(output["logits"].float(), labels, topology)
                values[name] = metric_values
                _update_bucket(pattern_buckets[name], metric_values)
            if detailed:
                metadata = batch["metadata"]
                weather = [str(value) for value in metadata["condition"]]
                scenario = [str(value) for value in metadata["scenario"]]
                trajectory = [str(value) for value in metadata["trajectory_group_id"]]
                labels_cpu = labels.cpu().numpy()
                group_values = {
                    "trajectory": trajectory,
                    "domain": [f"{left}/{right}" for left, right in zip(weather, scenario)],
                    "weather": weather,
                    "sector": (positions[labels_cpu] // 8).tolist(),
                    "beam": labels_cpu.tolist(),
                    "frequency": frequency[labels_cpu].tolist(),
                }
                for group_name, items in group_values.items():
                    for key, indices in _indices_by(items).items():
                        _update_bucket(group_buckets[group_name][key], values["full"][indices])
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return {
        "patterns": {name: _finish_bucket(bucket) for name, bucket in pattern_buckets.items()},
        **{f"per_{name}": {key: _finish_bucket(bucket) for key, bucket in buckets.items()} for name, buckets in group_buckets.items()},
        "evaluation_seconds": time.monotonic() - started,
        "outer_test_accessed": False,
    }


def _write_group_csv(path: Path, method: str, group: str, values: Mapping[str, Mapping[str, Any]]) -> None:
    rows = [{"method": method, group: key, **metrics} for key, metrics in values.items()]
    if rows:
        atomic_csv(path, rows, list(rows[0]))


def train(output_root: Path, method: str, *, smoke: bool) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown trajectory method: {method}")
    if not (output_root / "preflight_tests.txt").is_file():
        raise ValueError("Run trajectory --prepare before training.")
    run_dir = output_root / "smoke_tests" / method if smoke else output_root / method
    if run_dir.exists() and not smoke and ({path.name for path in run_dir.iterdir()} - {"train.log"}):
        raise FileExistsError(f"Trajectory run directory already has artifacts: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_trajectory_protocol(output_root / "protocol/split_manifest.json")
    topology = load_audited_topology(DEFAULT_TOPOLOGY)
    loaders, cfg, normalization = _loaders(output_root, protocol, create_normalization=False)
    fixed_validation = _fixed_loader(loaders["validation"], workers=0 if smoke else 8)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed()
    model = TrajectoryBaselineModel(method).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epochs, max_batches = (1, 2) if smoke else (EPOCHS, None)
    steps_per_epoch = min(len(loaders["train"]), max_batches or len(loaders["train"]))
    optimizer, scheduler = _optimizer(model, epochs, steps_per_epoch)
    assignments = random_balanced_assignment(_stable_train_ids(loaders["train"].dataset)) if method == RANDOM_BALANCED_METHOD else {}
    split_sha = sha256_file(output_root / "protocol/split_manifest.json")
    resolved = {
        "method": method,
        "seed": SEED,
        "epochs": epochs,
        "total_optimizer_steps": epochs * steps_per_epoch,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "encoder_lr": 1e-4,
        "head_lr": 3e-4,
        "weight_decay": 1e-4,
        "mixed_precision": "bf16",
        "gradient_clip": 1.0,
        "drop_last": False,
        "early_stopping": False,
        "checkpoint_selection": "minimum validation CE loss",
        "split_manifest_sha256": split_sha,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "normalization_fingerprint": normalization["metadata"]["normalization_fingerprint"],
        "model": model_contract(model),
        "random_balanced_ratio": "1:1 joint/single-modality optimizer steps" if assignments else None,
        "outer_test_accessed": False,
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    (run_dir / "split_manifest_sha256.txt").write_text(split_sha + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    started = time.monotonic()
    epoch_times: list[float] = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            totals: dict[str, float] = defaultdict(float)
            samples = 0
            epoch_started = time.monotonic()
            for step, batch in enumerate(loaders["train"]):
                if max_batches is not None and step >= max_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                labels = _labels(batch, device)
                availability = None
                single_phase = bool(assignments) and step % 2 == 1
                if single_phase:
                    availability = availability_for_assignments([assignments[value] for value in _batch_ids(batch)], device)
                with _autocast(device):
                    output = model(_inputs(batch, device), availability=availability)
                    loss, report = baseline_loss(model, output, labels)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"{method} produced non-finite loss at epoch {epoch}, step {step}.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                count = labels.numel()
                samples += count
                totals["loss"] += float(loss.detach()) * count
                totals["joint_samples" if not single_phase else "single_samples"] += count
                for name, value in report.items():
                    totals[name] += float(value) * count
                complete = (epoch - 1) * steps_per_epoch + step + 1
                if complete % 50 == 0 or step + 1 == steps_per_epoch:
                    elapsed = time.monotonic() - started
                    write_json(
                        run_dir / "runtime_status.json",
                        {
                            "status": "running",
                            "method": method,
                            "epoch": epoch,
                            "optimizer_step": complete,
                            "latest_train_loss": float(loss.detach()),
                            "elapsed_seconds": elapsed,
                            "estimated_remaining_seconds": elapsed * (epochs * steps_per_epoch / complete - 1),
                            "outer_test_accessed": False,
                        },
                    )
            validation = evaluate(
                model,
                fixed_validation,
                topology,
                device,
                train_beam_support=np.ones(64),
                detailed=False,
                max_batches=max_batches,
            )["patterns"]["full"]
            epoch_time = time.monotonic() - epoch_started
            epoch_times.append(epoch_time)
            row = {
                "epoch": epoch,
                "optimizer_steps": epoch * steps_per_epoch,
                "train_samples": samples,
                "joint_samples": int(totals["joint_samples"]),
                "single_modality_samples": int(totals["single_samples"]),
                "train_loss": totals["loss"] / max(samples, 1),
                "train_ce": totals["ce"] / max(samples, 1),
                "train_topology_alignment": totals["topology_alignment"] / max(samples, 1),
                "validation_loss": validation["ce_loss"],
                **{f"validation_{name}": validation[name] for name in ("top1", "top3", "within3", "mae", "topology_risk", "distance_gt5")},
                "lr": optimizer.param_groups[-1]["lr"],
                "epoch_seconds": epoch_time,
            }
            rows.append(row)
            atomic_csv(run_dir / "training_curve.csv", rows, list(rows[0]))
            print(json.dumps({"event": "trajectory_epoch", "method": method, **row}), flush=True)
            if validation["ce_loss"] < best_loss:
                best_loss, best_epoch = validation["ce_loss"], epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError("Trajectory training did not produce a checkpoint state.")
        checkpoint = run_dir / "best_checkpoint.pt"
        torch.save(
            {
                "state_dict": best_state,
                "method": method,
                "best_epoch": best_epoch,
                "validation_loss": best_loss,
                "split_manifest_sha256": split_sha,
                "protocol_fingerprint": protocol["protocol_fingerprint"],
                "model_config": {"d_model": 64, "seq_len": 5, "dropout": 0.1},
            },
            checkpoint,
        )
        reloaded = TrajectoryBaselineModel(method)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reloaded.load_state_dict(saved["state_dict"], strict=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.to(device)
        train_support = pd_read_beam_support(output_root / "protocol/train_beam_distribution.csv")
        metrics = evaluate(
            model,
            fixed_validation,
            topology,
            device,
            train_beam_support=train_support,
            detailed=True,
            max_batches=max_batches,
        )
        metrics.update(
            method=method,
            best_epoch=best_epoch,
            selection_loss=best_loss,
            checkpoint_sha256=sha256_file(checkpoint),
            split_manifest_sha256=split_sha,
            optimizer_steps=epochs * steps_per_epoch,
            outer_test_accessed=False,
        )
        write_json(run_dir / "metrics.json", metrics)
        for group in ("trajectory", "domain", "weather", "sector", "beam", "frequency"):
            _write_group_csv(run_dir / f"per_{group}_metrics.csv", method, group, metrics[f"per_{group}"])
        _write_group_csv(
            run_dir / "missing_diagnostics.csv",
            method,
            "pattern",
            {name: value for name, value in metrics["patterns"].items() if name != "full"},
        )
        efficiency = {
            "total_params": sum(value.numel() for value in model.parameters()),
            "trainable_params": sum(value.numel() for value in model.parameters() if value.requires_grad),
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0,
            "mean_epoch_seconds": float(np.mean(epoch_times)),
            "total_wall_seconds": time.monotonic() - started,
            "samples_per_second": float(epochs * int(protocol["train_window_count"]) / max(time.monotonic() - started, 1e-8)),
            "checkpoint_size_mib": checkpoint.stat().st_size / 1024**2,
        }
        write_json(run_dir / "efficiency.json", efficiency)
        (run_dir / "checkpoint_sha256.txt").write_text(metrics["checkpoint_sha256"] + "\n", encoding="utf-8")
        (run_dir / "eval.log").write_text(json.dumps(metrics["patterns"]["full"], indent=2) + "\n", encoding="utf-8")
        write_json(
            run_dir / "status.json",
            {
                "status": "passed",
                "method": method,
                "smoke": smoke,
                "epochs_completed": epochs,
                "optimizer_steps": epochs * steps_per_epoch,
                "best_epoch": best_epoch,
                "outer_test_accessed": False,
            },
        )
    except Exception as exc:
        write_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "method": method,
                "smoke": smoke,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "outer_test_accessed": False,
            },
        )
        raise
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)


def pd_read_beam_support(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([int(row["window_count"]) for row in rows], dtype=np.int64)


def verify_smokes(output_root: Path) -> None:
    checks = {}
    split_hashes = set()
    for method in METHODS:
        run_dir = output_root / "smoke_tests" / method
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        checks[method] = status.get("status") == "passed" and int(status.get("optimizer_steps", 0)) == 2
        checks[f"{method}_finite"] = all(math.isfinite(float(value)) for value in metrics["patterns"]["full"].values())
        checks[f"{method}_checkpoint"] = (run_dir / "best_checkpoint.pt").is_file()
        split_hashes.add(metrics["split_manifest_sha256"])
    checks["same_split_hash"] = len(split_hashes) == 1
    if not all(checks.values()):
        raise ValueError(f"Trajectory smoke verification failed: {checks}")
    with (output_root / "preflight_tests.txt").open("a", encoding="utf-8") as handle:
        for name, value in checks.items():
            handle.write(f"smoke_{name}=passed:{value}\n")
    write_json(output_root / "smoke_tests/status.json", {"status": "passed", "checks": checks, "outer_test_accessed": False})


def aggregate(output_root: Path) -> None:
    available = [method for method in METHODS if (output_root / method / "metrics.json").is_file()]
    if not available:
        raise FileNotFoundError("No trajectory baseline metrics are available.")
    rows, payloads = [], {}
    for method in available:
        payload = json.loads((output_root / method / "metrics.json").read_text(encoding="utf-8"))
        payloads[method] = payload
        rows.append(
            {
                "method": method,
                **payload["patterns"]["full"],
                "best_epoch": payload["best_epoch"],
                "optimizer_steps": payload["optimizer_steps"],
            }
        )
    atomic_csv(output_root / "combined_metrics.csv", rows, list(rows[0]))
    for group in ("trajectory", "domain", "weather", "sector", "beam"):
        combined = []
        for method in available:
            path = output_root / method / f"per_{group}_metrics.csv"
            if path.is_file():
                with path.open(newline="", encoding="utf-8") as handle:
                    combined.extend(csv.DictReader(handle))
        if combined:
            atomic_csv(output_root / f"per_{group}_metrics.csv", combined, list(combined[0]))
    missing = []
    for method in available:
        path = output_root / method / "missing_diagnostics.csv"
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                missing.extend(csv.DictReader(handle))
    if missing:
        atomic_csv(output_root / "missing_diagnostics.csv", missing, list(missing[0]))
    efficiency = []
    for method in available:
        path = output_root / method / "efficiency.json"
        if path.is_file():
            efficiency.append({"method": method, **json.loads(path.read_text(encoding="utf-8"))})
    if efficiency:
        atomic_csv(output_root / "efficiency.csv", efficiency, list(efficiency[0]))

    comparisons = []
    for name, left, right in (("M1-M0", METHODS[1], METHODS[0]), ("M2-M1", METHODS[2], METHODS[1]), ("M3-M2", METHODS[3], METHODS[2])):
        if left not in payloads or right not in payloads:
            continue
        left_metrics, right_metrics = payloads[left]["patterns"]["full"], payloads[right]["patterns"]["full"]
        comparisons.append(
            {
                "comparison": name,
                **{f"delta_{metric}": float(left_metrics[metric]) - float(right_metrics[metric]) for metric in METRICS},
            }
        )
    if comparisons:
        atomic_csv(output_root / "first_innovation_ablation.csv", comparisons, list(comparisons[0]))
    _write_analysis(output_root, rows, comparisons, payloads)
    write_json(
        output_root / "aggregate_status.json",
        {"status": "passed" if len(available) == 4 else "partial", "methods": available, "outer_test_accessed": False, "at": now()},
    )


def _write_analysis(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    protocol = load_trajectory_protocol(output_root / "protocol/split_manifest.json")
    group_audit = json.loads((output_root / "protocol/trajectory_group_audit.json").read_text(encoding="utf-8"))
    split_audit = json.loads((output_root / "protocol/split_audit.json").read_text(encoding="utf-8"))
    exposure = json.loads((output_root / "protocol/historical_exposure_summary.json").read_text(encoding="utf-8"))
    comparison = {row["comparison"]: row for row in comparisons}
    m2m1 = comparison.get("M2-M1", {})
    first_holds = bool(
        float(m2m1.get("delta_top1", 0)) > 0
        or float(m2m1.get("delta_within3", 0)) >= 0.005
        or float(m2m1.get("delta_mae", 0)) <= -0.10
        or float(m2m1.get("delta_topology_risk", 0))
        <= -0.03 * float(payloads.get(METHODS[1], {}).get("patterns", {}).get("full", {}).get("topology_risk", 0))
        or float(m2m1.get("delta_distance_gt5", 0)) <= -0.0075
    )
    lines = [
        "# MMW Trajectory Split Analysis",
        "",
        "## Protocol",
        "",
        f"1. Trajectory groups: **{protocol['trajectory_group_count']}**.",
        f"2. Definition: {protocol['trajectory_definition']}; {protocol['resource_coupling_rule']}.",
        f"3. Different CAVs share RSU resources: **yes**, all {group_audit['shared_rsu_cross_cav_group_count']} groups contain multiple CAVs.",
        "4. CAV-level leakage is prevented by resource connected components: **yes**.",
        "5. The data does not contain 50 groups, so 40/5/5 is not applicable.",
        f"6. Actual group split: **{protocol['train_group_count']}/{protocol['validation_group_count']}/{protocol['test_group_count']}**.",
        f"7. All pairwise resource intersections are zero: **{split_audit['status'] == 'passed'}**.",
        "8. Trajectories are strictly disjoint: **yes**.",
        "9. Chronological tail split is used: **no**.",
        "10. Random window split is used: **no**.",
        f"11. Window split: **{protocol['train_window_count']}/{protocol['validation_window_count']}/{protocol['test_window_count']}**; distributions are in protocol CSV/summary files.",
        "12. clean-inner measures local interpolation; Full-pool chronological measures tail extrapolation; this protocol measures unseen complete-trajectory generalization.",
        f"13. The sealed test has historical exposure: **{exposure['test_previously_exposed_for_training_or_selection']}**.",
        f"14. claim_eligible: **{str(protocol['claim_eligible']).lower()}**.",
        "15. Test prediction results accessed in this run: **no**.",
        "",
        "## Validation Results",
        "",
        "| Method | Head | Training | Top1 | Top3 | Within-3 | MAE | Topology Risk | D>5 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {
        METHODS[0]: ("Linear", "Joint"),
        METHODS[1]: ("Ordinary Prototype", "Joint"),
        METHODS[2]: ("Topology Prototype", "Joint"),
        METHODS[3]: ("Topology Prototype", "Random Balanced"),
    }
    indexed = {str(row["method"]): row for row in rows}
    for method in METHODS:
        if method not in indexed:
            lines.append(
                f"| {method} | {names[method][0]} | {names[method][1]} | pending | pending | pending | pending | pending | pending |"
            )
            continue
        row = indexed[method]
        lines.append(
            f"| {method} | {names[method][0]} | {names[method][1]} | {100 * float(row['top1']):.2f} | {100 * float(row['top3']):.2f} | {100 * float(row['within3']):.2f} | {float(row['mae']):.3f} | {float(row['topology_risk']):.4f} | {100 * float(row['distance_gt5']):.2f} |"
        )
    lines.extend(
        (
            "",
            f"16. M0 result is listed above ({'complete' if METHODS[0] in indexed else 'pending'}).",
            f"17. M1 improves M0 Top1: **{float(comparison.get('M1-M0', {}).get('delta_top1', 0)) > 0}**.",
            f"18. M2 improves M1 Top1: **{float(m2m1.get('delta_top1', 0)) > 0}**.",
            f"19. First innovation passes the preregistered trajectory criteria: **{first_holds}**.",
            f"20. M3 improves M2 Top1: **{float(comparison.get('M3-M2', {}).get('delta_top1', 0)) > 0}**.",
            "21. Hardest trajectory/weather/sector can be read from the corresponding sorted per-group CSV files.",
            f"22. Search a second innovation on this protocol: **{'yes' if first_holds else 'no'}**, based only on validation evidence.",
            "23. This workflow did not access test predictions, run multi-seed, or start a subsequent method search.",
        )
    )
    (output_root / "trajectory_split_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verify-smokes", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--allow-test-evaluation", action="store_true", help="Explicit sealed-test gate; unavailable in this run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.allow_test_evaluation:
        raise ValueError("This change is validation-only; sealed test evaluation is not implemented or executed.")
    if not any((args.prepare, args.method, args.verify_smokes, args.aggregate)):
        raise ValueError("Select --prepare, --method, --verify-smokes, or --aggregate.")
    output_root = Path(args.output_root).resolve()
    if args.prepare:
        prepare(output_root)
    if args.method:
        train(output_root, args.method, smoke=args.smoke)
    if args.verify_smokes:
        verify_smokes(output_root)
    if args.aggregate:
        aggregate(output_root)


if __name__ == "__main__":
    main()
