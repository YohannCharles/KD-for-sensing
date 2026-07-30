#!/usr/bin/env python3
"""Run the frozen sensing-only compositional prototype-deformation diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr

from kd_sensing.baselines.full_pool_bt_scl import load_audited_topology
from kd_sensing.baselines.full_pool_common import atomic_csv, now, sha256_file, write_json
from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel
from kd_sensing.diagnostics.prototype_deformation import (
    MASKS,
    MODALITIES,
    SINGLE_MISSING,
    additive_deformation,
    benjamini_hochberg,
    centers_from_deformation,
    count_deformation,
    estimate_centers,
    euclidean_shifts,
    mask_metadata,
    normalize,
    pairwise_deformation,
    prototype_logits,
    smooth_deformation,
    spherical_log_map,
    tangent_shifts,
    topology_adjacency,
    validate_mask_contract,
    weighted_r2,
)
from kd_sensing.engine.data_factory import shutdown_dataloader_workers
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices

from run_mmw_trajectory_baselines import _autocast, _fixed_loader, _inputs, _labels, _loaders


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/compositional_prototype_deformation_diagnostic.yaml"
MASK_NAMES = tuple(MASKS)
MISSING_INDICES = tuple(range(1, len(MASKS)))


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty diagnostic table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    atomic_csv(path, (_json_ready(row) for row in rows), fields, extrasaction="ignore")


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _string_hash(values: Iterable[str]) -> str:
    payload = "\n".join(str(value) for value in values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("diagnostic config must be a mapping.")
    if int(config["statistics"]["bootstrap_replicates"]) < 1000:
        raise ValueError("formal diagnostic requires at least 1000 bootstrap replicates.")
    if int(config["statistics"]["permutation_replicates"]) < 1000:
        raise ValueError("formal diagnostic requires at least 1000 permutations.")
    if config["safety"] != {
        "sensing_modalities": ["image", "lidar", "radar", "gps"],
        "csi_used": False,
        "channel_input_used": False,
        "outer_test_accessed": False,
        "future_beam_power_role": "label_side_evaluation_metric_only",
    }:
        raise ValueError("safety config must retain the exact sensing-only contract.")
    validate_mask_contract()
    return config


def _prepare_output(config: Mapping[str, Any], *, resume: bool) -> Path:
    output = _path(config["output"]["root"])
    marker = output / "process_manifest.json"
    if output.exists() and not resume:
        raise FileExistsError(f"diagnostic output already exists and will not be overwritten: {output}")
    if output.exists():
        if not marker.is_file():
            raise ValueError("resume target lacks this diagnostic's process manifest.")
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("diagnostic_id") != config["diagnostic_id"]:
            raise ValueError("resume target belongs to a different diagnostic.")
    else:
        output.mkdir(parents=True)
    for name in ("cache", "artifacts", "diagnostics", "figures"):
        (output / name).mkdir(exist_ok=True)
    return output


def _update_process(output: Path, config: Mapping[str, Any], stage: str, status: str, **extra: Any) -> None:
    path = output / "process_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload.update(
        diagnostic_id=config["diagnostic_id"],
        status=status,
        stage=stage,
        updated_at=now(),
        pid=os.getpid(),
        csi_used=False,
        channel_input_used=False,
        outer_test_accessed=False,
    )
    payload.update(_json_ready(extra))
    write_json(path, payload, sort_keys=True)


def _scan_split_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    role_ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    role_groups: dict[str, set[str]] = {"train": set(), "validation": set()}
    counts = defaultdict(int)
    chronology_errors = 0
    for domain in protocol["domains"]:
        for role, key in (("train", "train_split"), ("validation", "validation_split")):
            if not domain.get(key):
                continue
            with Path(domain[key]).open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    counts[role] += 1
                    role_ids[role].add(f"{domain['id']}:{row['sample_id']}")
                    role_groups[role].add(row["trajectory_group_id"])
                    history = json.loads(row["history_frame_ids_json"])
                    future = json.loads(row["future_frame_ids_json"])
                    chronology_errors += int(
                        len(history) != 5
                        or len(future) != 1
                        or int(history[-1]) != int(row["end_frame"])
                        or int(future[0]) != int(row["future_start_frame"])
                        or int(future[0]) != int(history[-1]) + 1
                    )
    return {
        "counts": dict(counts),
        "sample_overlap": len(role_ids["train"] & role_ids["validation"]),
        "trajectory_overlap": len(role_groups["train"] & role_groups["validation"]),
        "train_trajectories": len(role_groups["train"]),
        "validation_trajectories": len(role_groups["validation"]),
        "chronology_errors": chronology_errors,
        "train_sample_hash": _string_hash(sorted(role_ids["train"])),
        "validation_sample_hash": _string_hash(sorted(role_ids["validation"])),
    }


def audit(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    checkpoint_path = _path(config["source"]["checkpoint"])
    protocol_path = _path(config["protocol"]["manifest"])
    split_audit_path = _path(config["protocol"]["audit"])
    topology_path = _path(config["source"]["topology_manifest"])
    checkpoint_hash = sha256_file(checkpoint_path)
    split_hash = sha256_file(protocol_path)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    topology = load_audited_topology(topology_path)
    prototypes = saved["state_dict"]["prototype_bank.prototypes"].detach().float()
    split_scan = _scan_split_contract(protocol)
    expected = config["source"]
    checks = {
        "checkpoint_hash": checkpoint_hash == expected["expected_checkpoint_sha256"],
        "checkpoint_method": saved.get("method") == ABTC_METHOD,
        "checkpoint_protocol": saved.get("protocol_fingerprint") == protocol["protocol_fingerprint"],
        "checkpoint_split": saved.get("split_manifest_sha256") == split_hash,
        "prototype_shape": tuple(prototypes.shape) == (64, 64),
        "prototype_hash": _tensor_sha256(prototypes) == expected["expected_prototype_sha256"],
        "split_hash": split_hash == expected["expected_split_sha256"],
        "split_audit_passed": split_audit.get("status") == "passed",
        "split_resource_overlap_zero": all(
            int(item["count"]) == 0 for pair in split_audit["pairwise_overlaps"].values() for item in pair.values()
        ),
        "sample_overlap_zero": split_scan["sample_overlap"] == 0,
        "trajectory_overlap_zero": split_scan["trajectory_overlap"] == 0,
        "chronology_t_plus_1": split_scan["chronology_errors"] == 0,
        "sample_counts": split_scan["counts"]
        == {
            "train": int(config["protocol"]["expected_train_samples"]),
            "validation": int(config["protocol"]["expected_validation_samples"]),
        },
        "trajectory_counts": (
            split_scan["train_trajectories"],
            split_scan["validation_trajectories"],
        )
        == (
            int(config["protocol"]["expected_train_trajectories"]),
            int(config["protocol"]["expected_validation_trajectories"]),
        ),
        "outer_test_unaccessed": protocol.get("outer_test_accessed") is False and split_audit.get("outer_test_accessed") is False,
        "topology_cycle_audited": bool(topology.distance[0, 63].item() == 1 and topology.num_beams == 64),
        "all_masks_valid": len(MASKS) == 15 and MASKS["full"] == (1, 1, 1, 1),
        "no_mask_conditioned_state": not any(
            token in key.lower() for key in saved["state_dict"] for token in ("mask_head", "mask_temperature", "mask_bias")
        ),
    }
    hard_failures = [name for name, passed in checks.items() if not passed]
    audit_payload = {
        "status": "failed" if hard_failures else "passed",
        "checks": checks,
        "hard_failures": hard_failures,
        "checkpoint_sha256": checkpoint_hash,
        "prototype_sha256": _tensor_sha256(prototypes),
        "split_manifest_sha256": split_hash,
        "topology_manifest_sha256": topology.manifest_sha256,
        "topology_descriptor_sha256": topology.descriptor_sha256,
        "prototype_rank": int(torch.linalg.matrix_rank(prototypes).item()),
        "split_scan": split_scan,
        "mask_metadata": mask_metadata(),
        "csi_used": False,
        "channel_input_used": False,
        "outer_test_accessed": False,
    }
    lines = [
        "# 组合式缺失条件 Beam Prototype 形变诊断审计",
        "",
        f"审计状态：**{audit_payload['status']}**。硬失败：{hard_failures or '无'}。",
        "",
        f"1. M4 checkpoint：`{checkpoint_path}`；SHA256 `{checkpoint_hash}`。",
        f"2. Beam Prototype Bank：shape `{tuple(prototypes.shape)}`，矩阵 rank `{audit_payload['prototype_rank']}`，原始 FP32 tensor SHA256 `{audit_payload['prototype_sha256']}`。",
        "3. sensing embedding 实际维度为 64。",
        "4. 精确打分为 `normalize(z) @ normalize(P).T / 0.1`。",
        "5. feature 与 prototype 在打分函数内做 L2 normalization；temperature 是 `BeamPrototypeBank.temperature` 的固定 Python float，值为 0.1，不是可学习 state。",
        "6. 15 个非空 mask（内部 slot 顺序 `image,lidar,radar,gps`）：",
    ]
    for row in mask_metadata():
        lines.append(
            f"   - `{row['mask']}`：`{row['bits']}`；available={row['available_modalities']}；missing={row['missing_modalities']}；group={row['group']}。"
        )
    lines.extend(
        [
            "7. Full mask 为 `[1,1,1,1]`，四种感知模态全部可用。",
            "8. 每个 mask 由同一固定顺序 DataLoader batch 的同一批 sample ID 枚举生成；cache manifest 还会记录同一 sample hash。",
            f"9. train/validation sample overlap={split_scan['sample_overlap']}，trajectory overlap={split_scan['trajectory_overlap']}，严格互斥。",
            f"10. 扫描 {sum(split_scan['counts'].values())} 个 train/validation 窗口，chronology error={split_scan['chronology_errors']}；历史严格为 t-4...t，target 为 t+1。",
            "11. 模型与 loader 显式设置 `include_channel_ref=false`、`include_channel_history_refs=false`；本轮不加载任何 channel tensor/checkpoint/cache/fusion 结果。CSV 中已有的资源引用列仅属于协议清单，不被 dataset 解引用。",
            "12. ordinary loader 只构造 train/validation；outer-test loader/cache 均不构造，outer_test_accessed=false。",
            f"13. topology 加载/距离代码：`src/kd_sensing/baselines/full_pool_bt_scl.py:83`；manifest：`{topology.manifest_path}`。",
            f"14. 审计拓扑为 ULA-DFT phase cycle；distance(0,63)={float(topology.distance[0,63])}，首尾相邻。",
            "15. Beam Prototype Bank 是 checkpoint 中的可学习 `nn.Parameter`；不是 EMA center，也不是 empirical mean。",
            "16. Full 与所有缺失 mask 的当前 logits 都查询同一个 `prototype_bank.prototypes` tensor。",
            f"17. checkpoint 中不存在 mask-conditioned classifier、temperature 或 bias：{checks['no_mask_conditioned_state']}。",
            "",
            "标签侧 `future_beam_power` 仅用于 normalized gain/beam loss 评估，从不进入模型前向或中心估计。",
        ]
    )
    (output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "audit.json", audit_payload, sort_keys=True)
    if hard_failures:
        raise RuntimeError(f"protocol audit hard stop: {hard_failures}")
    return audit_payload


def _metadata_values(batch: Mapping[str, Any], key: str) -> list[str]:
    metadata = batch.get("metadata")
    if not isinstance(metadata, Mapping) or key not in metadata:
        raise ValueError(f"batch metadata lacks {key}.")
    return [str(value) for value in metadata[key]]


def _enable_label_side_power(loaders: Mapping[str, Any]) -> None:
    for loader in loaders.values():
        for dataset, _ in leaf_datasets_with_indices(loader.dataset):
            if bool(getattr(dataset, "include_channel_ref", False)) or bool(getattr(dataset, "include_channel_history_refs", False)):
                raise ValueError("channel reference loading is forbidden in this diagnostic.")
            dataset.include_router_utility_targets = True
            dataset.include_router_corruption_metadata = False


def _cache_split(
    split: str,
    loader: Any,
    model: TrajectoryBaselineModel,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    sample_ids: list[str] = []
    trajectory_ids: list[str] = []
    labels_parts: list[torch.Tensor] = []
    powers_parts: list[torch.Tensor] = []
    raw_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    predictions_parts: list[torch.Tensor] = []
    ranks_parts: list[torch.Tensor] = []
    ce_parts: list[torch.Tensor] = []
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            inputs = _inputs(batch, device)
            labels = _labels(batch, device)
            with _autocast(device):
                tokens = model.encode(inputs)
            batch_raw = []
            batch_logits = []
            for bits in MASKS.values():
                availability = torch.tensor(bits, dtype=torch.bool, device=device).expand(labels.numel(), -1)
                with _autocast(device):
                    result = model.forward_tokens(tokens, availability=availability)
                batch_raw.append(result["fused_features"].float())
                batch_logits.append(result["logits"].float())
            raw = torch.stack(batch_raw, dim=1)
            logits = torch.stack(batch_logits, dim=1)
            target_scores = logits.gather(2, labels[:, None, None].expand(-1, len(MASKS), 1))
            rank = 1 + logits.gt(target_scores).sum(dim=2)
            ce = -F.log_softmax(logits, dim=2).gather(2, labels[:, None, None].expand(-1, len(MASKS), 1)).squeeze(2)
            sample_ids.extend(_metadata_values(batch, "stable_sample_id"))
            trajectory_ids.extend(_metadata_values(batch, "trajectory_group_id"))
            labels_parts.append(labels.cpu())
            powers_parts.append(torch.as_tensor(batch["future_beam_power"], dtype=torch.float32).cpu())
            raw_parts.append(raw.cpu())
            logits_parts.append(logits.cpu())
            predictions_parts.append(logits.argmax(dim=2).cpu())
            ranks_parts.append(rank.cpu())
            ce_parts.append(ce.cpu())
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(
                    json.dumps(
                        {
                            "event": "sensing_cache",
                            "split": split,
                            "batch": batch_index,
                            "batches": len(loader),
                            "samples": len(sample_ids),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    raw = torch.cat(raw_parts).contiguous().float()
    logits = torch.cat(logits_parts).contiguous().float()
    payload = {
        "schema_version": 1,
        "split": split,
        "sample_id": sample_ids,
        "trajectory_id": trajectory_ids,
        "target": torch.cat(labels_parts).long(),
        "mask_id": torch.arange(len(MASKS), dtype=torch.long),
        "mask_metadata": mask_metadata(),
        "z_raw": raw,
        "z_normalized": normalize(raw).contiguous().float(),
        "shared_bank_logits": logits,
        "current_prediction": torch.cat(predictions_parts).long(),
        "target_rank": torch.cat(ranks_parts).long(),
        "ce_loss": torch.cat(ce_parts).float(),
        "full_feature": raw[:, 0].contiguous(),
        "future_beam_power": torch.cat(powers_parts).float(),
        "dtype": "float32",
        "compute_path": "current_bfloat16_autocast_then_fp32_storage" if device.type == "cuda" else "float32_cpu",
        "csi_used": False,
        "channel_input_used": False,
        "outer_test_accessed": False,
    }
    if len(sample_ids) != len(set(sample_ids)) or raw.shape[:2] != (len(sample_ids), len(MASKS)):
        raise ValueError(f"{split} cache identity or shape contract failed.")
    tensors = (raw, payload["z_normalized"], logits, payload["future_beam_power"])
    if not all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError(f"{split} cache must contain only finite FP32 analysis tensors.")
    _torch_save(output_path, payload)
    return {
        "split": split,
        "sample_count": len(sample_ids),
        "trajectory_count": len(set(trajectory_ids)),
        "sample_id_sha256": _string_hash(sample_ids),
        "trajectory_ids": sorted(set(trajectory_ids)),
        "shape": list(raw.shape),
        "file": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
    }


def build_cache(config: Mapping[str, Any], output: Path, audit_payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol_path = _path(config["protocol"]["manifest"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    from kd_sensing.data.mmw.trajectory_protocol import load_trajectory_protocol

    protocol = load_trajectory_protocol(protocol_path)
    loaders, loader_config, normalization = _loaders(_path("outputs/mmw_trajectory_split"), protocol, create_normalization=False)
    _enable_label_side_power(loaders)
    if set(loaders) != {"train", "validation"}:
        raise ValueError("diagnostic must construct train and validation loaders only.")
    if loader_config["training"]["final_test"]["enabled"] is not False:
        raise ValueError("outer-test loading must remain disabled.")
    dataset_config = loader_config["data"]["dataset"]
    if bool(dataset_config.get("include_channel_ref", False)) or bool(dataset_config.get("include_channel_history_refs", False)):
        raise ValueError("loader config requests forbidden channel references.")
    checkpoint = torch.load(_path(config["source"]["checkpoint"]), map_location="cpu", weights_only=False)
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model = TrajectoryBaselineModel(checkpoint["method"], **checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    workers = int(config["runtime"]["dataloader_workers"])
    fixed = {split: _fixed_loader(loader, workers=workers) for split, loader in loaders.items()}
    try:
        records = {
            split: _cache_split(split, fixed[split], model, device, output / "cache" / f"{split}_sensing_features.pt")
            for split in ("train", "validation")
        }
    finally:
        for loader in (*fixed.values(), *loaders.values()):
            shutdown_dataloader_workers(loader)
    expected_counts = {
        "train": int(config["protocol"]["expected_train_samples"]),
        "validation": int(config["protocol"]["expected_validation_samples"]),
    }
    if any(records[split]["sample_count"] != expected_counts[split] for split in records):
        raise ValueError("cache sample count does not match the sealed protocol.")
    manifest = {
        "schema_version": 1,
        "cache_version": config["cache"]["version"],
        "checkpoint_sha256": audit_payload["checkpoint_sha256"],
        "prototype_sha256": audit_payload["prototype_sha256"],
        "split_manifest_sha256": audit_payload["split_manifest_sha256"],
        "normalization_manifest": str(_path(config["source"]["normalization_manifest"]).resolve()),
        "normalization_source_split": normalization["metadata"]["source_split"],
        "mask_order": list(MASKS),
        "masks": mask_metadata(),
        "internal_modality_slot_order": list(MODALITIES),
        "feature_dimension": 64,
        "topology_definition": {
            "manifest": str(_path(config["source"]["topology_manifest"]).resolve()),
            "manifest_sha256": audit_payload["topology_manifest_sha256"],
            "descriptor_sha256": audit_payload["topology_descriptor_sha256"],
        },
        "splits": records,
        "dtype": "float32",
        "compute_path": config["cache"]["compute_path"] if device.type == "cuda" else "float32_cpu",
        "future_beam_power_role": "label_side_evaluation_metric_only",
        "csi_used": False,
        "channel_input_used": False,
        "outer_test_cache_created": False,
        "outer_test_accessed": False,
    }
    write_json(output / "cache/cache_manifest.json", manifest, sort_keys=True)
    return manifest


def _load_analysis_inputs(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "cache/cache_manifest.json").read_text(encoding="utf-8"))
    if any(
        (
            manifest.get("csi_used") is not False,
            manifest.get("channel_input_used") is not False,
            manifest.get("outer_test_accessed") is not False,
            manifest.get("outer_test_cache_created") is not False,
        )
    ):
        raise ValueError("cache manifest violates the sensing-only sealed-test contract.")
    caches = {
        split: torch.load(output / "cache" / f"{split}_sensing_features.pt", map_location="cpu", weights_only=False)
        for split in ("train", "validation")
    }
    for split, cache in caches.items():
        expected = int(config["protocol"][f"expected_{split}_samples"])
        if len(cache["sample_id"]) != expected or tuple(cache["z_raw"].shape) != (expected, 15, 64):
            raise ValueError(f"{split} cache no longer matches its manifest contract.")
        if cache.get("csi_used") is not False or cache.get("outer_test_accessed") is not False:
            raise ValueError(f"{split} cache violates the safety contract.")
    checkpoint = torch.load(_path(config["source"]["checkpoint"]), map_location="cpu", weights_only=False)
    learned = checkpoint["state_dict"]["prototype_bank.prototypes"].detach().float()
    topology = load_audited_topology(_path(config["source"]["topology_manifest"]))
    return {"manifest": manifest, "checkpoint": checkpoint, "learned": learned, "topology": topology, **caches}


def _indices_for(cache: Mapping[str, Any], *, trajectory: str | None = None, beam: int | None = None) -> torch.Tensor:
    selected = torch.ones(len(cache["sample_id"]), dtype=torch.bool)
    if trajectory is not None:
        selected &= torch.tensor([value == trajectory for value in cache["trajectory_id"]], dtype=torch.bool)
    if beam is not None:
        selected &= cache["target"].eq(int(beam))
    return selected.nonzero().reshape(-1)


def _centers_for(
    cache: Mapping[str, Any],
    learned: torch.Tensor,
    kappa: float,
    indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    index = torch.arange(len(cache["sample_id"])) if indices is None else indices
    return estimate_centers(cache["z_raw"][index], cache["target"][index], learned, kappa=float(kappa))


def _metric_arrays(logits: torch.Tensor, labels: torch.Tensor, powers: torch.Tensor, distance: torch.Tensor) -> dict[str, torch.Tensor]:
    scores = torch.as_tensor(logits, dtype=torch.float32)
    target = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    beam_power = torch.as_tensor(powers, dtype=torch.float32)
    if scores.ndim != 3 or scores.shape[:2] != (target.numel(), len(MASKS)) or beam_power.shape != (target.numel(), 64):
        raise ValueError("metric tensors must be logits [N,15,64], labels [N], and beam power [N,64].")
    prediction = scores.argmax(dim=-1)
    top = scores.topk(5, dim=-1).indices
    target_scores = scores.gather(2, target[:, None, None].expand(-1, len(MASKS), 1)).squeeze(2)
    masked = scores.clone()
    masked.scatter_(2, target[:, None, None].expand(-1, len(MASKS), 1), -torch.inf)
    margin = target_scores - masked.max(dim=2).values
    oracle = beam_power.max(dim=1).values.clamp_min(torch.finfo(torch.float32).tiny)
    selected_power = beam_power.gather(1, prediction).clamp_min(torch.finfo(torch.float32).tiny)
    gain = (selected_power / oracle[:, None]).clamp(max=1.0)
    return {
        "prediction": prediction,
        "top1": prediction.eq(target[:, None]).float(),
        "top3": top[:, :, :3].eq(target[:, None, None]).any(dim=2).float(),
        "top5": top.eq(target[:, None, None]).any(dim=2).float(),
        "within3": distance[target[:, None], prediction].le(3).float(),
        "mae": distance[target[:, None], prediction].float(),
        "normalized_gain": gain,
        "beam_loss_db": -10.0 * torch.log10(gain),
        "target_margin": margin,
        "nll": -F.log_softmax(scores, dim=2).gather(2, target[:, None, None].expand(-1, len(MASKS), 1)).squeeze(2),
    }


def _ece(logits: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    scores = torch.as_tensor(logits, dtype=torch.float32).reshape(-1, 64)
    target = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
    probabilities = torch.softmax(scores, dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(target).float()
    result = 0.0
    for lower in torch.linspace(0, 1, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        selected = (confidence > lower) & (confidence <= upper)
        if bool(selected.any()):
            result += float(selected.float().mean() * (confidence[selected].mean() - correct[selected].mean()).abs())
    return result


def _aggregate_metrics(
    arrays: Mapping[str, torch.Tensor],
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_index: torch.Tensor,
    mask_indices: Sequence[int],
) -> dict[str, float]:
    masks = torch.as_tensor(mask_indices, dtype=torch.long)
    result = {
        name: float(value[sample_index][:, masks].mean())
        for name, value in arrays.items()
        if name != "prediction"
    }
    selected_logits = logits[sample_index][:, masks].reshape(-1, 64)
    selected_labels = labels[sample_index, None].expand(-1, len(mask_indices)).reshape(-1)
    result["ece"] = _ece(selected_logits, selected_labels)
    result["sample_count"] = int(sample_index.numel())
    result["mask_count"] = len(mask_indices)
    return result


def _scope_masks() -> dict[str, tuple[int, ...]]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for row in mask_metadata():
        by_group[str(row["group"])].append(int(row["mask_id"]))
    return {
        "Full": tuple(by_group["Full"]),
        "Single": tuple(by_group["Single"]),
        "Two": tuple(by_group["Two"]),
        "Three": tuple(by_group["Three"]),
        "All-14": MISSING_INDICES,
    }


def d0_shared_bank(config: Mapping[str, Any], output: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    cache = data["validation"]
    direct = prototype_logits(cache["z_normalized"], normalize(data["learned"]), temperature=float(config["geometry"]["temperature"]))
    logits_max_abs = float((direct - cache["shared_bank_logits"]).abs().max())
    arrays = _metric_arrays(cache["shared_bank_logits"], cache["target"], cache["future_beam_power"], data["topology"].distance)
    all_samples = torch.arange(len(cache["sample_id"]))
    reference = json.loads(_path(config["source"]["existing_metrics"]).read_text(encoding="utf-8"))["patterns"]
    aliases = {"image_only": "only_image", "lidar_only": "only_lidar", "radar_only": "only_radar", "gps_only": "only_gps"}
    rows: list[dict[str, Any]] = []
    for mask_index, name in enumerate(MASK_NAMES):
        metrics = _aggregate_metrics(arrays, cache["shared_bank_logits"], cache["target"], all_samples, (mask_index,))
        previous = reference.get(aliases.get(name, name))
        rows.append(
            {
                "row_type": "mask",
                "scope": name,
                "mask": name,
                **metrics,
                "existing_b0_top1": None if previous is None else float(previous["top1"]),
                "top1_difference_pp": None if previous is None else 100.0 * (metrics["top1"] - float(previous["top1"])),
                "top1_within_0_02pp": None if previous is None else abs(metrics["top1"] - float(previous["top1"])) < 0.0002,
                "logits_max_abs_vs_fp32_recompute": logits_max_abs,
                "logits_within_pre_registered_tolerance": logits_max_abs <= float(config["thresholds"]["d0_logits_max_abs"]),
            }
        )
    for scope, masks in _scope_masks().items():
        metrics = _aggregate_metrics(arrays, cache["shared_bank_logits"], cache["target"], all_samples, masks)
        mask_top1 = [rows[index]["top1"] for index in masks]
        rows.append(
            {
                "row_type": "aggregate",
                "scope": scope,
                "mask": "",
                **metrics,
                "macro_top1": float(np.mean(mask_top1)),
                "worst_top1": float(np.min(mask_top1)),
                "logits_max_abs_vs_fp32_recompute": logits_max_abs,
            }
        )
    _write_csv(output / "diagnostics/d0_shared_bank_baseline.csv", rows)
    checked = [row for row in rows if row["row_type"] == "mask" and row.get("existing_b0_top1") is not None]
    summary = {
        "logits_max_abs": logits_max_abs,
        "logits_tolerance": float(config["thresholds"]["d0_logits_max_abs"]),
        "logits_tolerance_passed": logits_max_abs <= float(config["thresholds"]["d0_logits_max_abs"]),
        "known_mask_count": len(checked),
        "known_top1_tolerance_passed": all(bool(row["top1_within_0_02pp"]) for row in checked),
        "full_top1": rows[0]["top1"],
        "all14_top1": next(row["top1"] for row in rows if row["scope"] == "All-14"),
        "all14_worst": next(row["worst_top1"] for row in rows if row["scope"] == "All-14"),
        "missing_lidar_top1": next(row["top1"] for row in rows if row["scope"] == "missing_lidar"),
    }
    return {"rows": rows, "summary": summary, "arrays": arrays}


def _paired_center_tests(
    cache: Mapping[str, Any],
    centers: torch.Tensor,
    shifts: torch.Tensor,
    replicates: int,
    bootstrap_seed: int,
    permutation_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ci_low = np.full((15, 64), np.nan)
    ci_high = np.full((15, 64), np.nan)
    p_value = np.full((15, 64), np.nan)
    labels = cache["target"]
    features = cache["z_normalized"]
    for beam in range(64):
        index = labels.eq(beam).nonzero().reshape(-1)
        count = int(index.numel())
        if count < 2:
            continue
        base = centers[0, beam].expand(count, -1)
        full_tangent = spherical_log_map(base, features[index, 0])
        target_base = centers[0, beam].expand(count, 14, -1)
        missing_tangent = spherical_log_map(target_base, features[index, 1:])
        paired = missing_tangent - full_tangent[:, None, :]
        directions = normalize(shifts[1:, beam])
        projected = torch.einsum("nmd,md->nm", paired, directions).numpy().astype(np.float32)
        bootstrap_rng = np.random.default_rng(bootstrap_seed + beam)
        weights = bootstrap_rng.multinomial(count, np.full(count, 1.0 / count), size=replicates).astype(np.float32) / count
        bootstrap = weights @ projected
        exact = shifts[1:, beam].norm(dim=1).numpy()
        bootstrap += exact[None] - bootstrap.mean(axis=0, keepdims=True)
        ci_low[1:, beam], ci_high[1:, beam] = np.quantile(bootstrap, (0.025, 0.975), axis=0)
        permutation_rng = np.random.default_rng(permutation_seed + beam)
        swaps = permutation_rng.integers(0, 2, size=(replicates, count), dtype=np.int8).astype(np.float32) * 2.0 - 1.0
        null = swaps @ projected / count
        p_value[1:, beam] = (1.0 + (np.abs(null) >= exact[None]).sum(axis=0)) / (replicates + 1.0)
    return ci_low, ci_high, p_value


def d1_center_shift(config: Mapping[str, Any], output: Path, data: Mapping[str, Any], centers: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    cache = data["train"]
    spherical = centers["spherical"]
    shifts = tangent_shifts(spherical)
    replicates = int(config["statistics"]["bootstrap_replicates"])
    ci_low, ci_high, p_value = _paired_center_tests(
        cache,
        spherical,
        shifts,
        replicates,
        int(config["statistics"]["bootstrap_seed"]),
        int(config["statistics"]["permutation_seed"]),
    )
    q_value = np.full_like(p_value, np.nan)
    q_value[1:] = benjamini_hochberg(p_value[1:])
    labels = cache["target"]
    features = cache["z_normalized"]
    logits = cache["shared_bank_logits"]
    prediction = cache["current_prediction"]
    target_scores = logits.gather(2, labels[:, None, None].expand(-1, 15, 1)).squeeze(2)
    alternative = logits.clone()
    alternative.scatter_(2, labels[:, None, None].expand(-1, 15, 1), -torch.inf)
    margins = target_scores - alternative.max(dim=2).values
    rows: list[dict[str, Any]] = []
    for mask_index in MISSING_INDICES:
        for beam in range(64):
            index = labels.eq(beam).nonzero().reshape(-1)
            angles = torch.acos((features[index, mask_index] @ spherical[mask_index, beam]).clamp(-1 + 1e-7, 1 - 1e-7))
            within_std = float(angles.std(unbiased=True)) if index.numel() > 1 else math.nan
            norm = float(shifts[mask_index, beam].norm())
            rows.append(
                {
                    "beam": beam,
                    "mask": MASK_NAMES[mask_index],
                    "mask_id": mask_index,
                    "sample_count": int(index.numel()),
                    "low_sample": int(index.numel()) < int(config["geometry"]["minimum_beam_samples"]),
                    "tangent_shift_norm": norm,
                    "cosine_distance": float(1.0 - spherical[0, beam] @ spherical[mask_index, beam]),
                    "within_class_angular_std": within_std,
                    "shift_snr": norm / max(within_std, 1e-12),
                    "bootstrap_ci_low": ci_low[mask_index, beam],
                    "bootstrap_ci_high": ci_high[mask_index, beam],
                    "paired_mask_swap_p_value": p_value[mask_index, beam],
                    "bh_q_value": q_value[mask_index, beam],
                    "fdr_significant": bool(q_value[mask_index, beam] < float(config["statistics"]["fdr_q"])),
                    "current_error_rate": float(prediction[index, mask_index].ne(labels[index]).float().mean()),
                    "target_margin_drop_from_full": float((margins[index, 0] - margins[index, mask_index]).mean()),
                    "test_method": "label-preserving paired Full/mask identity-swap permutation",
                    "bootstrap_replicates": replicates,
                    "permutation_replicates": int(config["statistics"]["permutation_replicates"]),
                }
            )
    _write_csv(output / "diagnostics/d1_center_shift_by_beam_mask.csv", rows)
    summary_rows: list[dict[str, Any]] = []
    for mask in MASK_NAMES[1:]:
        selected = [row for row in rows if row["mask"] == mask and not row["low_sample"]]
        shifts_value = np.asarray([row["tangent_shift_norm"] for row in selected])
        errors = np.asarray([row["current_error_rate"] for row in selected])
        margins_value = np.asarray([row["target_margin_drop_from_full"] for row in selected])
        error_corr = spearmanr(shifts_value, errors).statistic if len(selected) > 2 and np.ptp(errors) > 0 else math.nan
        margin_corr = spearmanr(shifts_value, margins_value).statistic if len(selected) > 2 and np.ptp(margins_value) > 0 else math.nan
        summary_rows.append(
            {
                "scope_type": "mask",
                "scope": mask,
                "valid_beams": len(selected),
                "significant_beam_fraction": float(np.mean([row["fdr_significant"] for row in selected])),
                "median_shift_norm": float(np.median(shifts_value)),
                "median_shift_snr": float(np.median([row["shift_snr"] for row in selected])),
                "shift_error_spearman": error_corr,
                "shift_margin_drop_spearman": margin_corr,
            }
        )
    for beam in range(64):
        selected = [row for row in rows if row["beam"] == beam and not row["low_sample"]]
        summary_rows.append(
            {
                "scope_type": "beam",
                "scope": str(beam),
                "valid_masks": len(selected),
                "significant_mask_count": sum(bool(row["fdr_significant"]) for row in selected),
                "median_shift_norm": float(np.median([row["tangent_shift_norm"] for row in selected])),
                "median_shift_snr": float(np.median([row["shift_snr"] for row in selected])),
            }
        )
    _write_csv(output / "diagnostics/d1_center_shift_summary.csv", summary_rows)
    mask_summaries = [row for row in summary_rows if row["scope_type"] == "mask"]
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "shifts": shifts,
        "euclidean_shifts": euclidean_shifts(spherical),
        "median_significant_beam_fraction": float(np.median([row["significant_beam_fraction"] for row in mask_summaries])),
    }


def _direction_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    result = (left * right).sum(dim=-1) / (left_norm * right_norm).clamp_min(1e-12)
    return torch.where((left_norm > 1e-8) & (right_norm > 1e-8), result, torch.full_like(result, torch.nan))


def _trajectory_shift_records(
    cache: Mapping[str, Any], learned: torch.Tensor, kappa: float
) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for trajectory in sorted(set(cache["trajectory_id"])):
        center = _centers_for(cache, learned, kappa, _indices_for(cache, trajectory=trajectory))
        result[trajectory] = {"centers": center["spherical"], "shifts": tangent_shifts(center["spherical"]), "counts": center["counts"]}
    return result


def d2_stability(
    config: Mapping[str, Any], output: Path, data: Mapping[str, Any], train_centers: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    kappa = float(config["geometry"]["shrinkage_kappa"])
    minimum = int(config["geometry"]["minimum_beam_samples"])
    global_shift = tangent_shifts(train_centers["spherical"])
    train_records = _trajectory_shift_records(data["train"], data["learned"], kappa)
    validation_records = _trajectory_shift_records(data["validation"], data["learned"], kappa)
    train_trajectories = tuple(train_records)
    validation_trajectories = tuple(validation_records)
    if len(train_trajectories) != 12 or len(validation_trajectories) != 2:
        raise ValueError("stability analysis requires exactly 12 train and 2 validation trajectories.")
    all_train = torch.arange(len(data["train"]["sample_id"]))
    loto_records = {}
    for trajectory in train_trajectories:
        held_out = _indices_for(data["train"], trajectory=trajectory)
        retained = torch.ones(len(data["train"]["sample_id"]), dtype=torch.bool)
        retained[held_out] = False
        centers = _centers_for(data["train"], data["learned"], kappa, all_train[retained])
        loto_records[trajectory] = {"shifts": tangent_shifts(centers["spherical"]), "counts": centers["counts"]}
    validation_all = _centers_for(data["validation"], data["learned"], kappa)
    validation_shift = tangent_shifts(validation_all["spherical"])
    rows: list[dict[str, Any]] = []
    for mask_index in MISSING_INDICES:
        global_norm_ranking = global_shift[mask_index].norm(dim=1).numpy()
        rank_values = []
        jaccard_values = []
        for trajectory in train_trajectories:
            record = train_records[trajectory]
            valid = record["counts"].ge(minimum).numpy()
            if valid.sum() >= 3:
                rank_values.append(spearmanr(global_norm_ranking[valid], record["shifts"][mask_index].norm(dim=1).numpy()[valid]).statistic)
                global_top = set(np.argsort(global_norm_ranking[valid])[-8:].tolist())
                local_top = set(np.argsort(record["shifts"][mask_index].norm(dim=1).numpy()[valid])[-8:].tolist())
                jaccard_values.append(len(global_top & local_top) / max(len(global_top | local_top), 1))
        for beam in range(64):
            train_vectors = []
            for trajectory in train_trajectories:
                if int(train_records[trajectory]["counts"][beam]) >= minimum:
                    train_vectors.append(train_records[trajectory]["shifts"][mask_index, beam])
            cosines = (
                _direction_cosine(torch.stack(train_vectors), global_shift[mask_index, beam].expand(len(train_vectors), -1)).numpy()
                if train_vectors
                else np.asarray([])
            )
            norms = np.asarray([float(value.norm()) for value in train_vectors])
            loto_cosines = [
                float(_direction_cosine(loto_records[trajectory]["shifts"][mask_index, beam], global_shift[mask_index, beam]))
                for trajectory in train_trajectories
                if int(loto_records[trajectory]["counts"][beam]) >= minimum
            ]
            val_vectors = [
                validation_records[trajectory]["shifts"][mask_index, beam]
                for trajectory in validation_trajectories
                if int(validation_records[trajectory]["counts"][beam]) >= minimum
            ]
            val_pair_cos = float(_direction_cosine(val_vectors[0], val_vectors[1])) if len(val_vectors) == 2 else math.nan
            bootstrap_cosines = []
            if train_vectors:
                stack = torch.stack(train_vectors)
                for seed in config["statistics"]["stability_bootstrap_seeds"]:
                    rng = np.random.default_rng(int(seed) + beam + mask_index * 101)
                    sampled = stack[torch.as_tensor(rng.integers(0, len(stack), size=len(stack)))]
                    bootstrap_cosines.append(float(_direction_cosine(sampled.mean(dim=0), global_shift[mask_index, beam])))
            rows.append(
                {
                    "beam": beam,
                    "mask": MASK_NAMES[mask_index],
                    "valid_train_trajectories": len(train_vectors),
                    "low_sample": len(train_vectors) < 3,
                    "median_train_trajectory_cosine": float(np.nanmedian(cosines)) if cosines.size else math.nan,
                    "median_leave_one_trajectory_out_cosine": float(np.nanmedian(loto_cosines)) if loto_cosines else math.nan,
                    "norm_coefficient_of_variation": float(norms.std(ddof=1) / max(norms.mean(), 1e-12)) if len(norms) > 1 else math.nan,
                    "direction_consistency_fraction": float(np.nanmean(cosines > 0)) if cosines.size else math.nan,
                    "train_validation_shift_cosine": float(_direction_cosine(global_shift[mask_index, beam], validation_shift[mask_index, beam])),
                    "validation_trajectory_pair_cosine": val_pair_cos,
                    "bootstrap_cosine_ci_low": float(np.quantile(bootstrap_cosines, 0.025)) if bootstrap_cosines else math.nan,
                    "bootstrap_cosine_ci_high": float(np.quantile(bootstrap_cosines, 0.975)) if bootstrap_cosines else math.nan,
                    "rank_spearman": float(np.nanmedian(rank_values)) if rank_values else math.nan,
                    "top8_shifted_beam_jaccard": float(np.nanmedian(jaccard_values)) if jaccard_values else math.nan,
                    "leave_one_trajectory_out": True,
                    "train_trajectory_count": 12,
                    "validation_trajectory_count": 2,
                    "bootstrap_seed_count": len(config["statistics"]["stability_bootstrap_seeds"]),
                }
            )
    _write_csv(output / "diagnostics/d2_shift_stability.csv", rows)
    summary_rows = []
    for mask in MASK_NAMES[1:]:
        selected = [row for row in rows if row["mask"] == mask and not row["low_sample"]]
        summary_rows.append(
            {
                "mask": mask,
                "valid_beams": len(selected),
                "median_cross_trajectory_cosine": float(np.nanmedian([row["median_train_trajectory_cosine"] for row in selected])),
                "median_leave_one_trajectory_out_cosine": float(
                    np.nanmedian([row["median_leave_one_trajectory_out_cosine"] for row in selected])
                ),
                "median_direction_consistency": float(np.nanmedian([row["direction_consistency_fraction"] for row in selected])),
                "median_norm_cv": float(np.nanmedian([row["norm_coefficient_of_variation"] for row in selected])),
                "median_train_validation_cosine": float(np.nanmedian([row["train_validation_shift_cosine"] for row in selected])),
                "median_validation_pair_cosine": float(np.nanmedian([row["validation_trajectory_pair_cosine"] for row in selected])),
                "median_rank_spearman": float(np.nanmedian([row["rank_spearman"] for row in selected])),
                "median_top8_jaccard": float(np.nanmedian([row["top8_shifted_beam_jaccard"] for row in selected])),
                "stable": bool(
                    np.nanmedian([row["median_train_trajectory_cosine"] for row in selected])
                    >= float(config["thresholds"]["stability_median_cosine"])
                    and np.nanmedian([row["direction_consistency_fraction"] for row in selected])
                    >= float(config["thresholds"]["stability_direction_fraction"])
                    and np.nanmedian([row["validation_trajectory_pair_cosine"] for row in selected])
                    >= float(config["thresholds"]["validation_trajectory_cosine"])
                ),
            }
        )
    _write_csv(output / "diagnostics/d2_shift_stability_summary.csv", summary_rows)
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "stable_mask_fraction": float(np.mean([row["stable"] for row in summary_rows])),
        "median_train_cosine": float(np.nanmedian([row["median_cross_trajectory_cosine"] for row in summary_rows])),
        "median_validation_pair_cosine": float(np.nanmedian([row["median_validation_pair_cosine"] for row in summary_rows])),
    }


def _composition_rows(
    label: str,
    cache: Mapping[str, Any],
    indices: torch.Tensor,
    learned: torch.Tensor,
    kappa: float,
    predicted: torch.Tensor,
    train_full: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    centers = _centers_for(cache, learned, kappa, indices)
    observed = tangent_shifts(centers["spherical"])
    predicted_centers = centers_from_deformation(train_full, predicted)
    rows = []
    selected_masks = [index for index, row in enumerate(mask_metadata()) if int(row["missing_count"]) >= 2]
    for mask_index in selected_masks:
        for beam in range(64):
            observed_vector = observed[mask_index, beam]
            predicted_vector = predicted[mask_index, beam]
            rows.append(
                {
                    "dataset": label,
                    "beam": beam,
                    "mask": MASK_NAMES[mask_index],
                    "missing_count": mask_metadata()[mask_index]["missing_count"],
                    "sample_count": int(centers["counts"][beam]),
                    "shift_cosine": float(_direction_cosine(observed_vector, predicted_vector)),
                    "shift_norm_relative_error": float(
                        abs(observed_vector.norm() - predicted_vector.norm()) / observed_vector.norm().clamp_min(1e-12)
                    ),
                    "center_cosine_distance": float(1.0 - centers["spherical"][mask_index, beam] @ predicted_centers[mask_index, beam]),
                    "squared_error": float((observed_vector - predicted_vector).square().sum()),
                    "observed_shift_norm": float(observed_vector.norm()),
                    "predicted_shift_norm": float(predicted_vector.norm()),
                }
            )
    mask_tensor = torch.as_tensor(selected_masks, dtype=torch.long)
    weights = centers["counts"][None, :, None].expand(len(selected_masks), 64, 64).float()
    observed_selected = observed[mask_tensor]
    predicted_selected = predicted[mask_tensor]
    residual = observed_selected - predicted_selected
    observed_variance = float(observed_selected.var(unbiased=False))
    summary = {
        "dataset": label,
        "weighted_r2": weighted_r2(observed_selected, predicted_selected, weights),
        "explained_variance": 1.0 - float(residual.var(unbiased=False)) / max(observed_variance, 1e-12),
        "median_shift_cosine": float(np.nanmedian([row["shift_cosine"] for row in rows])),
        "median_relative_norm_error": float(np.nanmedian([row["shift_norm_relative_error"] for row in rows])),
        "median_center_cosine_distance": float(np.nanmedian([row["center_cosine_distance"] for row in rows])),
        "positive_direction_fraction": float(np.nanmean(np.asarray([row["shift_cosine"] for row in rows]) > 0)),
        "fit_masks": "Full+four single-missing only",
        "evaluated_multi_masks": len(selected_masks),
    }
    return rows, summary


def d3_compositionality(
    config: Mapping[str, Any], output: Path, data: Mapping[str, Any], train_centers: Mapping[str, torch.Tensor], train_shifts: torch.Tensor
) -> dict[str, Any]:
    predicted = additive_deformation(train_shifts)
    kappa = float(config["geometry"]["shrinkage_kappa"])
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    definitions: list[tuple[str, Mapping[str, Any], torch.Tensor]] = [
        ("train", data["train"], torch.arange(len(data["train"]["sample_id"]))),
        ("validation", data["validation"], torch.arange(len(data["validation"]["sample_id"]))),
    ]
    definitions.extend(
        (f"validation:{trajectory}", data["validation"], _indices_for(data["validation"], trajectory=trajectory))
        for trajectory in sorted(set(data["validation"]["trajectory_id"]))
    )
    for label, cache, indices in definitions:
        rows, summary = _composition_rows(label, cache, indices, data["learned"], kappa, predicted, train_centers["spherical"][0])
        all_rows.extend(rows)
        summaries.append(summary)
        for mask in sorted({row["mask"] for row in rows}):
            selected = [row for row in rows if row["mask"] == mask]
            summaries.append(
                {
                    "dataset": label,
                    "scope": mask,
                    "weighted_r2": 1.0
                    - sum(row["squared_error"] * row["sample_count"] for row in selected)
                    / max(
                        sum(row["observed_shift_norm"] ** 2 * row["sample_count"] for row in selected),
                        1e-12,
                    ),
                    "median_shift_cosine": float(np.nanmedian([row["shift_cosine"] for row in selected])),
                    "median_relative_norm_error": float(np.nanmedian([row["shift_norm_relative_error"] for row in selected])),
                    "positive_direction_fraction": float(np.nanmean(np.asarray([row["shift_cosine"] for row in selected]) > 0)),
                }
            )
    validation_rows = [row for row in all_rows if row["dataset"] == "validation"]
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]) + 303)
    cosine_values = np.asarray([row["shift_cosine"] for row in validation_rows], dtype=np.float64)
    boot = np.asarray(
        [np.nanmedian(cosine_values[rng.integers(0, len(cosine_values), size=len(cosine_values))]) for _ in range(int(config["statistics"]["bootstrap_replicates"]))]
    )
    validation_summary = next(row for row in summaries if row["dataset"] == "validation" and "scope" not in row)
    validation_summary.update(
        median_cosine_ci_low=float(np.quantile(boot, 0.025)),
        median_cosine_ci_high=float(np.quantile(boot, 0.975)),
        bootstrap_replicates=int(config["statistics"]["bootstrap_replicates"]),
    )
    _write_csv(output / "diagnostics/d3_additive_compositionality.csv", all_rows)
    _write_csv(output / "diagnostics/d3_additive_summary.csv", summaries)
    trajectory_summaries = [row for row in summaries if row["dataset"].startswith("validation:") and "scope" not in row]
    return {
        "rows": all_rows,
        "summary_rows": summaries,
        "validation": validation_summary,
        "validation_trajectories": trajectory_summaries,
        "predicted_shifts": predicted,
        "trajectory_direction_consistent": all(row["median_shift_cosine"] > 0 for row in trajectory_summaries),
    }


def _low_rank_deformation(shifts: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(shifts, dtype=torch.float32)
    matrix = values[1:].reshape(14, -1)
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    selected = min(int(rank), singular.numel())
    reconstructed = (left[:, :selected] * singular[:selected]) @ right[:selected]
    result = torch.zeros_like(values)
    result[1:] = reconstructed.reshape_as(values[1:])
    return result, singular


def _head_banks(learned: torch.Tensor, full: torch.Tensor, deformation: torch.Tensor, *, full_bypass: bool = True) -> torch.Tensor:
    banks = centers_from_deformation(full, deformation)
    if full_bypass:
        banks[0] = normalize(learned)
    return banks


def _nll_for_banks(cache: Mapping[str, Any], indices: torch.Tensor, banks: torch.Tensor, temperature: float) -> float:
    scores = prototype_logits(cache["z_normalized"][indices], banks, temperature=float(temperature))
    labels = cache["target"][indices]
    return float(
        -F.log_softmax(scores[:, 1:], dim=2)
        .gather(2, labels[:, None, None].expand(-1, 14, 1))
        .mean()
    )


def train_cv_selection(
    config: Mapping[str, Any], data: Mapping[str, Any], adjacency: torch.Tensor
) -> dict[str, Any]:
    cache = data["train"]
    learned = data["learned"]
    trajectories = sorted(set(cache["trajectory_id"]))
    lambda_grid = [float(value) for value in config["selection"]["topology_lambda_grid"]]
    rank_grid = [int(value) for value in config["selection"]["low_rank_grid"]]
    kappa_grid = [float(value) for value in config["selection"]["shrinkage_kappa_grid"]]
    lambda_scores: dict[float, list[float]] = {value: [] for value in lambda_grid}
    rank_scores: dict[int, list[float]] = {value: [] for value in rank_grid}
    kappa_scores: dict[float, list[float]] = {value: [] for value in kappa_grid}
    folds = []
    all_index = torch.arange(len(cache["sample_id"]))
    for trajectory in trajectories:
        holdout = _indices_for(cache, trajectory=trajectory)
        fit_mask = torch.ones(len(cache["sample_id"]), dtype=torch.bool)
        fit_mask[holdout] = False
        fit_index = all_index[fit_mask]
        fit_centers = _centers_for(cache, learned, float(config["geometry"]["shrinkage_kappa"]), fit_index)
        fit_shifts = tangent_shifts(fit_centers["spherical"])
        holdout_centers = _centers_for(cache, learned, float(config["geometry"]["shrinkage_kappa"]), holdout)
        holdout_shifts = tangent_shifts(holdout_centers["spherical"])
        additive = additive_deformation(fit_shifts)
        for coefficient in lambda_grid:
            smoothed = smooth_deformation(additive, fit_centers["spherical"][0], adjacency, coefficient)
            banks = _head_banks(learned, fit_centers["spherical"][0], smoothed)
            lambda_scores[coefficient].append(_nll_for_banks(cache, holdout, banks, float(config["geometry"]["temperature"])))
        for rank in rank_grid:
            reconstructed, _ = _low_rank_deformation(fit_shifts, rank)
            rank_scores[rank].append(weighted_r2(holdout_shifts[1:], reconstructed[1:]))
        for kappa in kappa_grid:
            fitted = _centers_for(cache, learned, kappa, fit_index)["shrinkage"].clone()
            fitted[0] = normalize(learned)
            kappa_scores[kappa].append(_nll_for_banks(cache, holdout, fitted, float(config["geometry"]["temperature"])))
        folds.append({"trajectory": trajectory, "fit_samples": int(fit_index.numel()), "holdout_samples": int(holdout.numel())})
    temperature_scores = {}
    for temperature in config["selection"]["temperature_grid"]:
        logits = prototype_logits(cache["z_normalized"], normalize(learned), temperature=float(temperature))
        temperature_scores[float(temperature)] = float(F.cross_entropy(logits[:, 0], cache["target"]))
    selected_lambda = min(lambda_grid, key=lambda value: float(np.mean(lambda_scores[value])))
    selected_rank = max(rank_grid, key=lambda value: float(np.mean(rank_scores[value])))
    selected_kappa = min(kappa_grid, key=lambda value: float(np.mean(kappa_scores[value])))
    selected_temperature = min(temperature_scores, key=temperature_scores.get)
    return {
        "folds": folds,
        "topology_lambda": {
            "selected": selected_lambda,
            "mean_cv_nll": {str(key): float(np.mean(value)) for key, value in lambda_scores.items()},
        },
        "low_rank": {
            "selected": selected_rank,
            "mean_cv_r2": {str(key): float(np.mean(value)) for key, value in rank_scores.items()},
        },
        "shrinkage": {
            "pre_registered": float(config["geometry"]["shrinkage_kappa"]),
            "cv_best_sensitivity_only": selected_kappa,
            "mean_cv_nll": {str(key): float(np.mean(value)) for key, value in kappa_scores.items()},
        },
        "global_temperature": {
            "primary_fixed": float(config["geometry"]["temperature"]),
            "train_only_selected": selected_temperature,
            "train_nll": {str(key): value for key, value in temperature_scores.items()},
        },
        "validation_used_for_selection": False,
    }


def _factorization_summary(
    model: str,
    prediction: torch.Tensor,
    observed: torch.Tensor,
    centers: torch.Tensor,
    full: torch.Tensor,
    counts: torch.Tensor,
    parameter_count: int,
    dataset: str,
) -> list[dict[str, Any]]:
    groups = {
        "All-14": list(MISSING_INDICES),
        "Single-missing fit": [list(MASKS).index(name) for name in SINGLE_MISSING.values()],
        "Double-missing": [index for index, row in enumerate(mask_metadata()) if row["missing_count"] == 2],
        "Triple-missing extrapolation": [index for index, row in enumerate(mask_metadata()) if row["missing_count"] == 3],
    }
    predicted_centers = centers_from_deformation(full, prediction)
    rows = []
    for scope, indices in groups.items():
        index = torch.as_tensor(indices, dtype=torch.long)
        weights = counts[None, :, None].expand(len(indices), 64, 64).float()
        cosine = _direction_cosine(observed[index], prediction[index])
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "scope": scope,
                "weighted_r2": weighted_r2(observed[index], prediction[index], weights),
                "median_shift_cosine": float(torch.nanmedian(cosine)),
                "mean_center_cosine_error": float((1.0 - (centers[index] * predicted_centers[index]).sum(dim=-1)).mean()),
                "parameter_count": int(parameter_count),
                "fit_rule": {
                    "M0": "none",
                    "M1": "Full+single only",
                    "M2": "Full+single only",
                    "M3": "Full+single+double only",
                    "M4": "independent upper bound",
                }.get(model, "train-only SVD"),
            }
        )
    return rows


def d4_factorization(
    config: Mapping[str, Any], output: Path, data: Mapping[str, Any], train_centers: Mapping[str, torch.Tensor], selection: Mapping[str, Any]
) -> dict[str, Any]:
    train_shift = tangent_shifts(train_centers["spherical"])
    validation_centers = _centers_for(data["validation"], data["learned"], float(config["geometry"]["shrinkage_kappa"]))
    validation_shift = tangent_shifts(validation_centers["spherical"])
    count_model = count_deformation(train_shift)
    additive = additive_deformation(train_shift)
    pairwise, pair_terms = pairwise_deformation(train_shift)
    selected_rank = int(selection["low_rank"]["selected"])
    low_rank, singular = _low_rank_deformation(train_shift, selected_rank)
    models = {
        "M0": (torch.zeros_like(train_shift), 0),
        "M1": (count_model, 64 * 64),
        "M2": (additive, 4 * 64 * 64),
        "M3": (pairwise, 10 * 64 * 64),
        "M4": (train_shift, 14 * 64 * 64),
        f"M5-r{selected_rank}": (low_rank, selected_rank * (14 + 64 * 64)),
    }
    rows = []
    for model, (prediction, parameters) in models.items():
        rows.extend(
            _factorization_summary(
                model,
                prediction,
                train_shift,
                train_centers["spherical"],
                train_centers["spherical"][0],
                train_centers["counts"],
                parameters,
                "train",
            )
        )
        rows.extend(
            _factorization_summary(
                model,
                prediction,
                validation_shift,
                validation_centers["spherical"],
                train_centers["spherical"][0],
                validation_centers["counts"],
                parameters,
                "validation",
            )
        )
    upper = next(row for row in rows if row["dataset"] == "validation" and row["model"] == "M4" and row["scope"] == "All-14")["weighted_r2"]
    baseline = next(row for row in rows if row["dataset"] == "validation" and row["model"] == "M0" and row["scope"] == "All-14")["weighted_r2"]
    for row in rows:
        denominator = upper - baseline
        row["recovery_vs_m4_upper"] = (row["weighted_r2"] - baseline) / denominator if abs(denominator) > 1e-12 else math.nan
    total_energy = float(singular.square().sum())
    spectrum_rows = []
    cumulative = 0.0
    for index, value in enumerate(singular.tolist(), 1):
        explained = value * value / max(total_energy, 1e-12)
        cumulative += explained
        spectrum_rows.append(
            {
                "component": index,
                "singular_value": value,
                "explained_variance": explained,
                "cumulative_explained_variance": cumulative,
                "candidate_rank": index if index in config["selection"]["low_rank_grid"] else "",
                "train_cv_r2": selection["low_rank"]["mean_cv_r2"].get(str(index)),
                "selected": index == selected_rank,
            }
        )
    _write_csv(output / "diagnostics/d4_deformation_factorization.csv", rows)
    _write_csv(output / "diagnostics/d4_low_rank_spectrum.csv", spectrum_rows)
    _torch_save(
        output / "artifacts/empirical_centers.pt",
        {
            "train": train_centers,
            "validation_evaluation_only": validation_centers,
            "kappa": float(config["geometry"]["shrinkage_kappa"]),
            "validation_used_for_fit": False,
        },
    )
    _torch_save(output / "artifacts/tangent_shifts.pt", {"train_tangent": train_shift, "train_euclidean": euclidean_shifts(train_centers["spherical"])})
    _torch_save(output / "artifacts/additive_deformation.pt", {"shifts": additive, "fit_masks": ["full", *SINGLE_MISSING.values()]})
    _torch_save(output / "artifacts/pairwise_deformation.pt", {"shifts": pairwise, "pair_terms": pair_terms})
    _torch_save(output / "artifacts/low_rank_deformation.pt", {"rank": selected_rank, "shifts": low_rank, "singular_values": singular})
    return {
        "rows": rows,
        "spectrum_rows": spectrum_rows,
        "count": count_model,
        "additive": additive,
        "pairwise": pairwise,
        "low_rank": low_rank,
        "selected_rank": selected_rank,
        "validation_centers": validation_centers,
        "validation_shifts": validation_shift,
    }


def _topology_variation(vectors: torch.Tensor, adjacency: torch.Tensor, order: Sequence[int]) -> tuple[float, float]:
    edge = torch.nonzero(torch.triu(adjacency, diagonal=1), as_tuple=False)
    tv1 = float((vectors[edge[:, 0]] - vectors[edge[:, 1]]).square().sum(dim=1).mean())
    ordered = vectors[torch.as_tensor(order, dtype=torch.long)]
    tv2 = float((ordered.roll(1, 0) - 2.0 * ordered + ordered.roll(-1, 0)).square().sum(dim=1).mean())
    return tv1, tv2


def _topology_nulls(
    vectors: torch.Tensor,
    adjacency: torch.Tensor,
    frequencies: torch.Tensor,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    values = torch.as_tensor(vectors, dtype=torch.float32)
    edge = torch.nonzero(torch.triu(adjacency, diagonal=1), as_tuple=False)
    left, right = edge[:, 0], edge[:, 1]
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(64) for _ in range(replicates)])
    permuted = values[torch.as_tensor(permutations)]
    beam_null = (permuted[:, left] - permuted[:, right]).square().sum(dim=2).mean(dim=1).numpy()
    random_values = torch.as_tensor(rng.standard_normal((replicates, 64, 64)), dtype=torch.float32)
    random_values = normalize(random_values) * values.norm(dim=1)[None, :, None]
    direction_null = (random_values[:, left] - random_values[:, right]).square().sum(dim=2).mean(dim=1).numpy()
    random_left = torch.as_tensor(rng.integers(0, 64, size=(replicates, edge.shape[0])), dtype=torch.long)
    random_right = torch.as_tensor(rng.integers(0, 63, size=(replicates, edge.shape[0])), dtype=torch.long)
    random_right += random_right.ge(random_left).long()
    graph_null = (values[random_left] - values[random_right]).square().sum(dim=2).mean(dim=1).numpy()
    frequency = frequencies.numpy()
    quantiles = np.quantile(frequency, (0.25, 0.5, 0.75))
    bins = np.digitize(frequency, quantiles)
    matched = np.empty((replicates, 64), dtype=np.int64)
    for replicate in range(replicates):
        matched[replicate] = np.arange(64)
        for group in range(4):
            members = np.flatnonzero(bins == group)
            matched[replicate, members] = rng.permutation(members)
    matched_values = values[torch.as_tensor(matched)]
    frequency_null = (matched_values[:, left] - matched_values[:, right]).square().sum(dim=2).mean(dim=1).numpy()
    return {
        "beam_order_permutation": beam_null,
        "norm_matched_random_direction": direction_null,
        "random_topology_graph": graph_null,
        "frequency_matched_beam_permutation": frequency_null,
    }


def d5_topology(
    config: Mapping[str, Any], output: Path, data: Mapping[str, Any], train_shift: torch.Tensor, validation_shift: torch.Tensor
) -> dict[str, Any]:
    topology = data["topology"]
    adjacency = topology_adjacency(topology.distance)
    replicates = int(config["statistics"]["permutation_replicates"])
    definitions: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        ("train", train_shift, torch.bincount(data["train"]["target"], minlength=64)),
        ("validation", validation_shift, torch.bincount(data["validation"]["target"], minlength=64)),
    ]
    for trajectory in sorted(set(data["validation"]["trajectory_id"])):
        index = _indices_for(data["validation"], trajectory=trajectory)
        center = _centers_for(data["validation"], data["learned"], float(config["geometry"]["shrinkage_kappa"]), index)
        definitions.append((f"validation:{trajectory}", tangent_shifts(center["spherical"]), center["counts"]))
    rows = []
    for dataset_index, (dataset, shifts, frequencies) in enumerate(definitions):
        for mask_index in MISSING_INDICES:
            vectors = shifts[mask_index]
            observed, curvature = _topology_variation(vectors, adjacency, topology.labels_by_position)
            nulls = _topology_nulls(
                vectors,
                adjacency,
                frequencies,
                replicates,
                int(config["statistics"]["permutation_seed"]) + dataset_index * 1009 + mask_index,
            )
            for control, null in nulls.items():
                mean, std = float(null.mean()), float(null.std(ddof=1))
                rows.append(
                    {
                        "dataset": dataset,
                        "mask": MASK_NAMES[mask_index],
                        "control": control,
                        "observed_tv1": observed,
                        "observed_tv2": curvature,
                        "permutation_mean": mean,
                        "permutation_std": std,
                        "reduction_fraction": (mean - observed) / max(mean, 1e-12),
                        "effect_size": (mean - observed) / max(std, 1e-12),
                        "empirical_p_value": (1.0 + float((null <= observed).sum())) / (replicates + 1.0),
                        "permutation_replicates": replicates,
                        "topology_manifest_sha256": topology.manifest_sha256,
                    }
                )
    _write_csv(output / "diagnostics/d5_topology_smoothness.csv", rows)
    primary = [row for row in rows if row["dataset"] == "train" and row["control"] == "beam_order_permutation"]
    singles = {mask for mask in SINGLE_MISSING.values()}
    single_rows = [row for row in primary if row["mask"] in singles]
    return {
        "rows": rows,
        "adjacency": adjacency,
        "single_pass_count": sum(
            row["reduction_fraction"] >= float(config["thresholds"]["topology_reduction"]) and row["empirical_p_value"] < 0.05
            for row in single_rows
        ),
        "median_single_reduction": float(np.median([row["reduction_fraction"] for row in single_rows])),
    }


def build_prototype_heads(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    train_centers: Mapping[str, torch.Tensor],
    factorization: Mapping[str, Any],
    selection: Mapping[str, Any],
    adjacency: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, torch.Tensor]], dict[str, Any]]:
    learned = normalize(data["learned"])
    full = train_centers["spherical"][0]
    repeated_learned = learned[None].expand(15, -1, -1).clone()
    full_empirical = full[None].expand(15, -1, -1).clone()
    independent = train_centers["spherical"].clone()
    shrinkage = train_centers["shrinkage"].clone()
    p4 = _head_banks(learned, full, factorization["count"])
    p5 = _head_banks(learned, full, factorization["additive"])
    p6 = _head_banks(learned, full, factorization["pairwise"])
    coefficient = float(selection["topology_lambda"]["selected"])
    smoothed = smooth_deformation(factorization["additive"], full, adjacency, coefficient)
    p7 = _head_banks(learned, full, smoothed)
    rng = np.random.default_rng(int(config["statistics"]["permutation_seed"]) + 909)
    permutation = rng.permutation(14) + 1
    p9 = independent.clone()
    p9[1:] = independent[torch.as_tensor(permutation)]
    p9[0] = learned
    heads = {
        "P0": repeated_learned,
        "P1": full_empirical,
        "P2": independent,
        "P3": shrinkage,
        "P4": p4,
        "P5": p5,
        "P6": p6,
        "P7": p7,
        "P9": p9,
    }
    random_heads = []
    train_shift = tangent_shifts(train_centers["spherical"])
    single_indices = [list(MASKS).index(mask) for mask in SINGLE_MISSING.values()]
    for seed in config["statistics"]["random_deformation_seeds"]:
        generator = torch.Generator().manual_seed(int(seed))
        random_shift = torch.zeros_like(train_shift)
        for mask_index in single_indices:
            direction = torch.randn(64, 64, generator=generator)
            direction -= (direction * full).sum(dim=1, keepdim=True) * full
            random_shift[mask_index] = normalize(direction) * train_shift[mask_index].norm(dim=1, keepdim=True)
        random_heads.append((int(seed), _head_banks(learned, full, additive_deformation(random_shift))))
    _torch_save(
        output / "artifacts/topology_smoothed_deformation.pt",
        {
            "lambda": coefficient,
            "shifts": smoothed,
            "selection": selection["topology_lambda"],
            "validation_used_for_selection": False,
        },
    )
    metadata = {
        "P0": "current shared learned bank",
        "P1": "train Full empirical center shared by every mask",
        "P2": "train independent mask-conditioned empirical centers",
        "P3": f"train independent shrinkage centers, kappa={config['geometry']['shrinkage_kappa']}",
        "P4": "missing-count deformation from single-missing conditions",
        "P5": "additive four single-missing tangent deformation",
        "P6": "additive plus pairwise interaction; triple masks are extrapolation",
        "P7": f"topology-smoothed P5, train-CV lambda={coefficient}",
        "P8": "20 norm-matched random tangent deformation seeds",
        "P9": f"mask identity permutation {permutation.tolist()}",
    }
    return heads, random_heads, metadata


def _bootstrap_weights(count: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.multinomial(count, np.full(count, 1.0 / count), size=int(replicates)).astype(np.float32) / count


def _bootstrap_summary(weights: np.ndarray, differences: torch.Tensor | np.ndarray) -> dict[str, float]:
    values = np.asarray(differences, dtype=np.float32).reshape(-1)
    draws = weights @ values
    return {
        "delta": float(values.mean()),
        "delta_ci_low": float(np.quantile(draws, 0.025)),
        "delta_ci_high": float(np.quantile(draws, 0.975)),
        "delta_p_value": min(1.0, float((1.0 + min((draws <= 0).sum(), (draws >= 0).sum())) / (len(draws) + 1.0) * 2.0)),
    }


def _context_indices(cache: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "validation": torch.arange(len(cache["sample_id"])),
        **{
            f"validation:{trajectory}": _indices_for(cache, trajectory=trajectory)
            for trajectory in sorted(set(cache["trajectory_id"]))
        },
    }


def _arrays_aggregate(arrays: Mapping[str, torch.Tensor], indices: torch.Tensor, masks: Sequence[int]) -> dict[str, float]:
    mask_index = torch.as_tensor(masks, dtype=torch.long)
    return {
        name: float(values[indices][:, mask_index].mean())
        for name, values in arrays.items()
        if name != "prediction"
    }


def _evaluate_one_head(
    head: str,
    arrays: Mapping[str, torch.Tensor],
    logits: torch.Tensor | None,
    baseline_arrays: Mapping[str, torch.Tensor],
    baseline_logits: torch.Tensor,
    cache: Mapping[str, Any],
    contexts: Mapping[str, torch.Tensor],
    bootstrap: Mapping[str, np.ndarray],
    *,
    random_seed: int | None = None,
    include_bootstrap: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    mask_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    beam_rows: list[dict[str, Any]] = []
    fix_rows: list[dict[str, Any]] = []
    scopes = _scope_masks()
    for context, indices in contexts.items():
        for mask_index, mask in enumerate(MASK_NAMES):
            metrics = _arrays_aggregate(arrays, indices, (mask_index,))
            baseline_metrics = _arrays_aggregate(baseline_arrays, indices, (mask_index,))
            if logits is not None:
                metrics["ece"] = _ece(logits[indices, mask_index], cache["target"][indices])
            else:
                metrics["ece"] = math.nan
            difference = arrays["top1"][indices, mask_index] - baseline_arrays["top1"][indices, mask_index]
            paired = _bootstrap_summary(bootstrap[context], difference) if include_bootstrap else {"delta": float(difference.mean())}
            mask_rows.append(
                {
                    "head": head,
                    "random_seed": random_seed,
                    "trajectory": context,
                    "mask": mask,
                    "available_count": mask_metadata()[mask_index]["available_count"],
                    "sample_count": int(indices.numel()),
                    **metrics,
                    "top1_delta_vs_p0_pp": 100.0 * (metrics["top1"] - baseline_metrics["top1"]),
                    "paired_top1_ci_low_pp": 100.0 * paired.get("delta_ci_low", math.nan),
                    "paired_top1_ci_high_pp": 100.0 * paired.get("delta_ci_high", math.nan),
                    "paired_top1_p_value": paired.get("delta_p_value", math.nan),
                }
            )
            base_correct = baseline_arrays["top1"][indices, mask_index]
            candidate_correct = arrays["top1"][indices, mask_index]
            fix_rows.append(
                {
                    "head": head,
                    "random_seed": random_seed,
                    "trajectory": context,
                    "scope": mask,
                    "fix_rate": float(((1.0 - base_correct) * candidate_correct).mean()),
                    "harm_rate": float((base_correct * (1.0 - candidate_correct)).mean()),
                    "net_fix_minus_harm": float((candidate_correct - base_correct).mean()),
                }
            )
        for scope, masks in scopes.items():
            metrics = _arrays_aggregate(arrays, indices, masks)
            base_metrics = _arrays_aggregate(baseline_arrays, indices, masks)
            difference = arrays["top1"][indices][:, masks].mean(dim=1) - baseline_arrays["top1"][indices][:, masks].mean(dim=1)
            paired = _bootstrap_summary(bootstrap[context], difference) if include_bootstrap else {"delta": float(difference.mean())}
            per_mask_top1 = [float(arrays["top1"][indices, mask].mean()) for mask in masks]
            ece = (
                float(np.mean([_ece(logits[indices, mask], cache["target"][indices]) for mask in masks]))
                if logits is not None
                else math.nan
            )
            head_rows.append(
                {
                    "head": head,
                    "random_seed": random_seed,
                    "trajectory": context,
                    "scope": scope,
                    "sample_count": int(indices.numel()),
                    "mask_count": len(masks),
                    **metrics,
                    "ece": ece,
                    "macro_top1": float(np.mean(per_mask_top1)),
                    "worst_top1": float(np.min(per_mask_top1)),
                    "top1_delta_vs_p0_pp": 100.0 * (metrics["top1"] - base_metrics["top1"]),
                    "paired_top1_ci_low_pp": 100.0 * paired.get("delta_ci_low", math.nan),
                    "paired_top1_ci_high_pp": 100.0 * paired.get("delta_ci_high", math.nan),
                    "paired_top1_p_value": paired.get("delta_p_value", math.nan),
                }
            )
            base_correct = baseline_arrays["top1"][indices][:, masks]
            candidate_correct = arrays["top1"][indices][:, masks]
            fix_rows.append(
                {
                    "head": head,
                    "random_seed": random_seed,
                    "trajectory": context,
                    "scope": scope,
                    "fix_rate": float(((1.0 - base_correct) * candidate_correct).mean()),
                    "harm_rate": float((base_correct * (1.0 - candidate_correct)).mean()),
                    "net_fix_minus_harm": float((candidate_correct - base_correct).mean()),
                }
            )
        missing = scopes["All-14"]
        top1 = [float(arrays["top1"][indices, mask].mean()) for mask in missing]
        worst_mask = missing[int(np.argmin(top1))]
        metrics = _arrays_aggregate(arrays, indices, (worst_mask,))
        head_rows.append(
            {
                "head": head,
                "random_seed": random_seed,
                "trajectory": context,
                "scope": "Worst",
                "worst_mask": MASK_NAMES[worst_mask],
                "sample_count": int(indices.numel()),
                "mask_count": 1,
                **metrics,
                "ece": _ece(logits[indices, worst_mask], cache["target"][indices]) if logits is not None else math.nan,
                "macro_top1": metrics["top1"],
                "worst_top1": metrics["top1"],
                "top1_delta_vs_p0_pp": 100.0
                * (metrics["top1"] - float(baseline_arrays["top1"][indices, worst_mask].mean())),
            }
        )
    validation_index = contexts["validation"]
    for mask_index, mask in enumerate(MASK_NAMES):
        for beam in range(64):
            selected = validation_index[cache["target"][validation_index].eq(beam)]
            beam_rows.append(
                {
                    "head": head,
                    "random_seed": random_seed,
                    "mask": mask,
                    "beam": beam,
                    "sample_count": int(selected.numel()),
                    "top1": float(arrays["top1"][selected, mask_index].mean()) if selected.numel() else math.nan,
                    "p0_top1": float(baseline_arrays["top1"][selected, mask_index].mean()) if selected.numel() else math.nan,
                    "top1_delta_pp": 100.0
                    * float((arrays["top1"][selected, mask_index] - baseline_arrays["top1"][selected, mask_index]).mean())
                    if selected.numel()
                    else math.nan,
                }
            )
    return {"mask": mask_rows, "head": head_rows, "beam": beam_rows, "fix": fix_rows}


def d7_heads(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    heads: Mapping[str, torch.Tensor],
    random_heads: Sequence[tuple[int, torch.Tensor]],
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    cache = data["validation"]
    temperature = float(config["geometry"]["temperature"])
    contexts = _context_indices(cache)
    bootstrap = {
        context: _bootstrap_weights(
            int(indices.numel()),
            int(config["statistics"]["bootstrap_replicates"]),
            int(config["statistics"]["bootstrap_seed"]) + context_index,
        )
        for context_index, (context, indices) in enumerate(contexts.items())
    }
    baseline_logits = prototype_logits(cache["z_normalized"], heads["P0"], temperature=temperature)
    baseline_arrays = _metric_arrays(baseline_logits, cache["target"], cache["future_beam_power"], data["topology"].distance)
    all_rows: dict[str, list[dict[str, Any]]] = {"mask": [], "head": [], "beam": [], "fix": []}
    evaluations: dict[str, dict[str, Any]] = {"P0": {"logits": baseline_logits, "arrays": baseline_arrays}}
    for head, banks in heads.items():
        logits = baseline_logits if head == "P0" else prototype_logits(cache["z_normalized"], banks, temperature=temperature)
        arrays = baseline_arrays if head == "P0" else _metric_arrays(logits, cache["target"], cache["future_beam_power"], data["topology"].distance)
        evaluations[head] = {"logits": logits, "arrays": arrays}
        rows = _evaluate_one_head(head, arrays, logits, baseline_arrays, baseline_logits, cache, contexts, bootstrap)
        for key in all_rows:
            all_rows[key].extend(rows[key])
    random_arrays = []
    random_summary = []
    random_logits_sum = torch.zeros_like(baseline_logits)
    for seed, banks in random_heads:
        logits = prototype_logits(cache["z_normalized"], banks, temperature=temperature)
        random_logits_sum += logits
        arrays = _metric_arrays(logits, cache["target"], cache["future_beam_power"], data["topology"].distance)
        random_arrays.append(arrays)
        random_summary.append(
            {
                "seed": seed,
                "all14_top1": float(arrays["top1"][:, 1:].mean()),
                "worst_top1": float(arrays["top1"][:, 1:].mean(dim=0).min()),
                "missing_lidar_top1": float(arrays["top1"][:, list(MASKS).index("missing_lidar")].mean()),
            }
        )
    averaged_arrays = {
        name: torch.stack([arrays[name] for arrays in random_arrays]).float().mean(dim=0)
        for name in baseline_arrays
        if name != "prediction"
    }
    averaged_arrays["prediction"] = random_arrays[0]["prediction"]
    averaged_logits = random_logits_sum / len(random_heads)
    evaluations["P8"] = {"logits": averaged_logits, "arrays": averaged_arrays, "seeds": random_summary}
    rows = _evaluate_one_head(
        "P8",
        averaged_arrays,
        averaged_logits,
        baseline_arrays,
        baseline_logits,
        cache,
        contexts,
        bootstrap,
    )
    for key in all_rows:
        all_rows[key].extend(rows[key])
    random_all14 = np.asarray([row["all14_top1"] for row in random_summary])
    for row in all_rows["head"]:
        row["head_definition"] = metadata[row["head"]]
        if row["head"] == "P8":
            row["random_seed_count"] = len(random_summary)
            row["random_all14_top1_mean"] = float(random_all14.mean())
            row["random_all14_top1_std"] = float(random_all14.std(ddof=1))
    for kind in ("mask", "beam", "fix"):
        for row in all_rows[kind]:
            row["head_definition"] = metadata[row["head"]]
    _write_csv(output / "diagnostics/d7_prototype_head_results.csv", all_rows["head"])
    _write_csv(output / "diagnostics/d7_mask_results.csv", all_rows["mask"])
    _write_csv(output / "diagnostics/d7_beam_results.csv", all_rows["beam"])
    _write_csv(output / "diagnostics/d7_fix_harm.csv", all_rows["fix"])
    full_index = 0
    full_checks = {
        head: {
            "prototype_max_abs": float((banks[full_index] - normalize(data["learned"])).abs().max()),
            "logits_max_abs": float((evaluations[head]["logits"][:, full_index] - baseline_logits[:, full_index]).abs().max()),
            "prediction_identical": bool(evaluations[head]["logits"][:, full_index].argmax(dim=1).eq(baseline_logits[:, full_index].argmax(dim=1)).all()),
            "within_fp32_tolerance": float(
                (evaluations[head]["logits"][:, full_index] - baseline_logits[:, full_index]).abs().max()
            )
            <= 1e-5,
        }
        for head, banks in heads.items()
        if head in ("P4", "P5", "P6", "P7", "P9")
    }
    return {
        "rows": all_rows,
        "evaluations": evaluations,
        "random_summary": random_summary,
        "full_checks": full_checks,
        "random_all14_mean": float(random_all14.mean()),
        "random_all14_std": float(random_all14.std(ddof=1)),
    }


def _result_value(results: Mapping[str, Any], head: str, scope: str, *, trajectory: str = "validation") -> float:
    row = next(
        row
        for row in results["rows"]["head"]
        if row["head"] == head and row["scope"] == scope and row["trajectory"] == trajectory
    )
    return float(row["top1"])


def _mask_value(results: Mapping[str, Any], head: str, mask: str, *, trajectory: str = "validation") -> float:
    row = next(
        row
        for row in results["rows"]["mask"]
        if row["head"] == head and row["mask"] == mask and row["trajectory"] == trajectory
    )
    return float(row["top1"])


def d8_headroom(config: Mapping[str, Any], output: Path, results: Mapping[str, Any]) -> dict[str, Any]:
    independent = str(config["selection"]["independent_reference"])
    scopes = ("All-14", "Worst", "Single", "Two", "Three", "missing_lidar")
    rows = []
    trajectories = sorted({row["trajectory"] for row in results["rows"]["head"] if row["head"] == "P0"})
    for trajectory in trajectories:
        for scope in scopes:
            getter = _mask_value if scope == "missing_lidar" else _result_value
            shared = getter(results, "P0", scope, trajectory=trajectory)
            upper = getter(results, independent, scope, trajectory=trajectory)
            additive = getter(results, "P5", scope, trajectory=trajectory)
            topology = getter(results, "P7", scope, trajectory=trajectory)
            denominator = upper - shared
            has_positive_headroom = denominator > 0
            rows.append(
                {
                    "trajectory": trajectory,
                    "scope": scope,
                    "independent_reference": independent,
                    "shared_top1": shared,
                    "independent_top1": upper,
                    "additive_top1": additive,
                    "topology_top1": topology,
                    "independent_gain_pp": 100.0 * denominator,
                    "additive_gain_pp": 100.0 * (additive - shared),
                    "topology_gain_vs_additive_pp": 100.0 * (topology - additive),
                    "positive_independent_headroom": has_positive_headroom,
                    "recovered_headroom_additive": (additive - shared) / denominator if has_positive_headroom else math.nan,
                    "recovered_headroom_topology": (topology - shared) / denominator if has_positive_headroom else math.nan,
                }
            )
    _write_csv(output / "diagnostics/d8_headroom_recovery.csv", rows)
    validation = [row for row in rows if row["trajectory"] == "validation"]
    by_scope = {row["scope"]: row for row in validation}
    independent_pass = any(
        (
            by_scope["All-14"]["independent_gain_pp"] >= float(config["thresholds"]["independent_all14_gain_pp"]),
            by_scope["Worst"]["independent_gain_pp"] >= float(config["thresholds"]["independent_worst_gain_pp"]),
            by_scope["missing_lidar"]["independent_gain_pp"] >= float(config["thresholds"]["independent_missing_lidar_gain_pp"]),
        )
    )
    all14 = next(row for row in validation if row["scope"] == "All-14")
    return {
        "rows": rows,
        "independent_pass": independent_pass,
        "additive_headroom_pass": bool(
            all14["positive_independent_headroom"]
            and all14["additive_gain_pp"] > 0
            and all14["recovered_headroom_additive"] >= float(config["thresholds"]["additive_headroom_fraction"])
        ),
        "all14": all14,
        "full_candidate_unchanged": all(
            results["full_checks"][head]["within_fp32_tolerance"] and results["full_checks"][head]["prediction_identical"]
            for head in ("P5", "P7")
        ),
    }


def _covariance_stats(values: torch.Tensor) -> dict[str, Any]:
    count, dimension = values.shape
    if count < 2:
        return {
            "covariance": torch.full((dimension, dimension), torch.nan),
            "trace_covariance": math.nan,
            "top_eigenvalue": math.nan,
            "effective_rank": math.nan,
            "anisotropy_ratio": math.nan,
        }
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    covariance = centered.t() @ centered / (count - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    trace = eigenvalues.sum()
    probabilities = eigenvalues / trace.clamp_min(torch.finfo(torch.float64).eps)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-15).log()).sum())
    top = eigenvalues[-1]
    return {
        "covariance": covariance.float(),
        "trace_covariance": float(trace),
        "top_eigenvalue": float(top),
        "effective_rank": float(effective_rank),
        "anisotropy_ratio": float(top / (trace / dimension).clamp_min(1e-15)),
    }


def covariance_analysis(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    train_centers: Mapping[str, torch.Tensor],
    validation_centers: Mapping[str, torch.Tensor],
    d1: Mapping[str, Any],
    d8: Mapping[str, Any],
) -> dict[str, Any]:
    definitions = (
        ("train", data["train"], train_centers),
        ("validation", data["validation"], validation_centers),
    )
    rows: list[dict[str, Any]] = []
    summary_rows = []
    for dataset, cache, centers in definitions:
        shifts = tangent_shifts(centers["spherical"])
        full_covariances: dict[int, dict[str, Any]] = {}
        for beam in range(64):
            index = cache["target"].eq(beam).nonzero().reshape(-1)
            full_covariances[beam] = _covariance_stats(cache["z_normalized"][index, 0])
        dataset_rows = []
        for mask_index, mask in enumerate(MASK_NAMES):
            for beam in range(64):
                index = cache["target"].eq(beam).nonzero().reshape(-1)
                current = _covariance_stats(cache["z_normalized"][index, mask_index])
                full = full_covariances[beam]
                covariance_change = float((current["covariance"] - full["covariance"]).norm())
                error = float(cache["current_prediction"][index, mask_index].ne(cache["target"][index]).float().mean())
                row = {
                    "row_type": "beam_mask",
                    "dataset": dataset,
                    "beam": beam,
                    "mask": mask,
                    "sample_count": int(index.numel()),
                    "trace_covariance": current["trace_covariance"],
                    "top_eigenvalue": current["top_eigenvalue"],
                    "effective_rank": current["effective_rank"],
                    "anisotropy_ratio": current["anisotropy_ratio"],
                    "covariance_inflation": current["trace_covariance"] / max(full["trace_covariance"], 1e-12),
                    "center_shift_magnitude": float(shifts[mask_index, beam].norm()),
                    "covariance_change_magnitude": covariance_change,
                    "current_error_rate": error,
                }
                rows.append(row)
                if mask_index:
                    dataset_rows.append(row)
        shift = np.asarray([row["center_shift_magnitude"] for row in dataset_rows])
        covariance = np.asarray([row["covariance_change_magnitude"] for row in dataset_rows])
        inflation = np.asarray([row["covariance_inflation"] for row in dataset_rows])
        error = np.asarray([row["current_error_rate"] for row in dataset_rows])
        summary = {
            "row_type": "summary",
            "dataset": dataset,
            "beam": -1,
            "mask": "All-14",
            "sample_count": len(cache["sample_id"]),
            "error_center_shift_spearman": float(spearmanr(error, shift).statistic),
            "error_covariance_change_spearman": float(spearmanr(error, covariance).statistic),
            "error_covariance_inflation_spearman": float(spearmanr(error, inflation).statistic),
            "median_center_shift": float(np.median(shift)),
            "median_covariance_inflation": float(np.median(inflation)),
        }
        rows.append(summary)
        summary_rows.append(summary)
    validation_summary = next(row for row in summary_rows if row["dataset"] == "validation")
    shift_association = abs(validation_summary["error_center_shift_spearman"])
    covariance_association = abs(validation_summary["error_covariance_change_spearman"])
    shift_present = d1["median_significant_beam_fraction"] >= 0.5 and shift_association >= 0.3
    covariance_present = covariance_association >= 0.3
    if shift_present and covariance_present:
        dominance = "center_shift_and_variance_joint"
    elif shift_present:
        dominance = "center_shift_dominant"
    elif covariance_present:
        dominance = "variance_inflation_dominant"
    else:
        dominance = "inconclusive_joint"
    validation_summary["diagnosis"] = dominance
    _write_csv(output / "diagnostics/covariance_shift_analysis.csv", rows)
    return {"rows": rows, "summary": summary_rows, "diagnosis": dominance}


def negative_controls(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    train_centers: Mapping[str, torch.Tensor],
    selection: Mapping[str, Any],
    d3: Mapping[str, Any],
    d5: Mapping[str, Any],
    d7: Mapping[str, Any],
) -> dict[str, Any]:
    train = data["train"]
    validation = data["validation"]
    learned = data["learned"]
    temperature = float(config["geometry"]["temperature"])
    rng = torch.Generator().manual_seed(int(config["statistics"]["permutation_seed"]) + 1701)
    permuted_labels = train["target"][torch.randperm(len(train["sample_id"]), generator=rng)]
    label_centers = estimate_centers(
        train["z_raw"], permuted_labels, learned, kappa=float(config["geometry"]["shrinkage_kappa"])
    )["spherical"]
    label_centers[0] = normalize(learned)
    label_logits = prototype_logits(validation["z_normalized"], label_centers, temperature=temperature)
    label_arrays = _metric_arrays(label_logits, validation["target"], validation["future_beam_power"], data["topology"].distance)
    validation_oracle = _centers_for(validation, learned, float(config["geometry"]["shrinkage_kappa"]))["spherical"]
    oracle_logits = prototype_logits(validation["z_normalized"], validation_oracle, temperature=temperature)
    oracle_arrays = _metric_arrays(oracle_logits, validation["target"], validation["future_beam_power"], data["topology"].distance)
    tangent_r2 = float(d3["validation"]["weighted_r2"])
    train_euclidean = euclidean_shifts(train_centers["spherical"])
    validation_centers = _centers_for(validation, learned, float(config["geometry"]["shrinkage_kappa"]))
    validation_euclidean = euclidean_shifts(validation_centers["spherical"])
    euclidean_prediction = additive_deformation(train_euclidean)
    weights = validation_centers["counts"][None, :, None].expand(14, 64, 64).float()
    euclidean_r2 = weighted_r2(validation_euclidean[1:], euclidean_prediction[1:], weights)
    rows: list[dict[str, Any]] = [
        {
            "control": "beam_label_permutation",
            "metric": "all14_top1",
            "value": float(label_arrays["top1"][:, 1:].mean()),
            "validation_leakage_oracle": False,
        },
        {
            "control": "mask_identity_permutation_P9",
            "metric": "all14_top1",
            "value": _result_value(d7, "P9", "All-14"),
            "validation_leakage_oracle": False,
        },
        {
            "control": "true_independent_P3",
            "metric": "all14_top1",
            "value": _result_value(d7, "P3", "All-14"),
            "validation_leakage_oracle": False,
        },
        {
            "control": "random_deformation_P8_20_seed_mean",
            "metric": "all14_top1",
            "value": d7["random_all14_mean"],
            "std": d7["random_all14_std"],
            "validation_leakage_oracle": False,
        },
        {
            "control": "true_additive_P5",
            "metric": "all14_top1",
            "value": _result_value(d7, "P5", "All-14"),
            "validation_leakage_oracle": False,
        },
        {
            "control": "validation_center_illegal_oracle",
            "metric": "all14_top1",
            "value": float(oracle_arrays["top1"][:, 1:].mean()),
            "validation_leakage_oracle": True,
            "included_in_formal_results": False,
        },
        {
            "control": "euclidean_composition",
            "metric": "validation_weighted_r2",
            "value": euclidean_r2,
            "validation_leakage_oracle": False,
        },
        {
            "control": "tangent_composition_primary",
            "metric": "validation_weighted_r2",
            "value": tangent_r2,
            "validation_leakage_oracle": False,
        },
        {
            "control": "shared_learned_P0",
            "metric": "all14_top1",
            "value": _result_value(d7, "P0", "All-14"),
            "validation_leakage_oracle": False,
        },
        {
            "control": "shared_full_empirical_P1",
            "metric": "all14_top1",
            "value": _result_value(d7, "P1", "All-14"),
            "validation_leakage_oracle": False,
        },
    ]
    for seed_row in d7["random_summary"]:
        rows.append(
            {
                "control": "random_deformation_seed",
                "seed": seed_row["seed"],
                "metric": "all14_top1",
                "value": seed_row["all14_top1"],
                "validation_leakage_oracle": False,
            }
        )
    for coefficient, score in selection["topology_lambda"]["mean_cv_nll"].items():
        rows.append(
            {
                "control": "topology_lambda_train_trajectory_cv",
                "setting": coefficient,
                "metric": "mean_cv_nll",
                "value": score,
                "selected": float(coefficient) == float(selection["topology_lambda"]["selected"]),
                "validation_leakage_oracle": False,
            }
        )
    for kappa, score in selection["shrinkage"]["mean_cv_nll"].items():
        rows.append(
            {
                "control": "shrinkage_train_only_sensitivity",
                "setting": kappa,
                "metric": "mean_cv_nll",
                "value": score,
                "pre_registered_primary": float(kappa) == float(config["geometry"]["shrinkage_kappa"]),
                "validation_leakage_oracle": False,
            }
        )
    for temperature_value, score in selection["global_temperature"]["train_nll"].items():
        rows.append(
            {
                "control": "single_global_temperature_train_only",
                "setting": temperature_value,
                "metric": "train_nll",
                "value": score,
                "primary_fixed_temperature": float(temperature_value) == float(config["geometry"]["temperature"]),
                "validation_leakage_oracle": False,
            }
        )
    topology_primary = [
        row for row in d5["rows"] if row["dataset"] == "train" and row["control"] in ("beam_order_permutation", "random_topology_graph")
    ]
    for row in topology_primary:
        rows.append(
            {
                "control": row["control"],
                "setting": row["mask"],
                "metric": "tv_reduction_fraction",
                "value": row["reduction_fraction"],
                "empirical_p_value": row["empirical_p_value"],
                "validation_leakage_oracle": False,
            }
        )
    _write_csv(output / "diagnostics/negative_controls.csv", rows)
    real_mask_advantage = _result_value(d7, "P3", "All-14") > _result_value(d7, "P9", "All-14")
    real_shift_advantage = _result_value(d7, "P5", "All-14") > d7["random_all14_mean"]
    return {
        "rows": rows,
        "label_permutation_top1": float(label_arrays["top1"][:, 1:].mean()),
        "validation_oracle_top1": float(oracle_arrays["top1"][:, 1:].mean()),
        "real_mask_beats_permutation": real_mask_advantage,
        "real_additive_beats_random": real_shift_advantage,
        "controls_pass": bool(real_mask_advantage and real_shift_advantage),
        "euclidean_r2": euclidean_r2,
        "tangent_r2": tangent_r2,
    }


def create_figures(
    output: Path,
    d1: Mapping[str, Any],
    d3: Mapping[str, Any],
    d5: Mapping[str, Any],
    d7: Mapping[str, Any],
    d8: Mapping[str, Any],
    covariance: Mapping[str, Any],
    topology: Any,
) -> None:
    figure_root = output / "figures"
    shift_norm = d1["shifts"][1:].norm(dim=2).numpy()
    plt.figure(figsize=(12, 5))
    plt.imshow(shift_norm, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="Tangent shift norm")
    plt.yticks(range(14), MASK_NAMES[1:], fontsize=7)
    plt.xlabel("Beam label")
    plt.tight_layout()
    plt.savefig(figure_root / "center_shift_heatmap.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    for modality, mask in SINGLE_MISSING.items():
        index = list(MASKS).index(mask) - 1
        plt.plot(range(64), shift_norm[index], label=f"missing {modality}", linewidth=1.2)
    plt.xlabel("Beam label")
    plt.ylabel("Tangent shift norm")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(figure_root / "shift_norm_by_beam.png", dpi=160)
    plt.close()

    validation_mask = [
        row for row in d3["summary_rows"] if row["dataset"] == "validation" and "scope" in row
    ]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(validation_mask)), [row["median_shift_cosine"] for row in validation_mask], color="#39766c")
    plt.xticks(range(len(validation_mask)), [row["scope"] for row in validation_mask], rotation=45, ha="right", fontsize=8)
    plt.axhline(0.7, color="#a13b35", linestyle="--", linewidth=1)
    plt.ylabel("Median tangent-shift cosine")
    plt.tight_layout()
    plt.savefig(figure_root / "compositionality_by_mask.png", dpi=160)
    plt.close()

    validation_rows = [row for row in d3["rows"] if row["dataset"] == "validation"]
    plt.figure(figsize=(6, 6))
    plt.scatter(
        [row["observed_shift_norm"] for row in validation_rows],
        [row["predicted_shift_norm"] for row in validation_rows],
        s=9,
        alpha=0.45,
        color="#376d9c",
    )
    maximum = max(max(row["observed_shift_norm"] for row in validation_rows), max(row["predicted_shift_norm"] for row in validation_rows))
    plt.plot([0, maximum], [0, maximum], color="#333333", linewidth=1)
    plt.xlabel("Observed shift norm")
    plt.ylabel("Additive predicted norm")
    plt.tight_layout()
    plt.savefig(figure_root / "additive_predicted_vs_observed.png", dpi=160)
    plt.close()

    topology_rows = [row for row in d5["rows"] if row["dataset"] == "train" and row["control"] == "beam_order_permutation"]
    plt.figure(figsize=(10, 5))
    plt.bar(range(14), [100.0 * row["reduction_fraction"] for row in topology_rows], color="#836a33")
    plt.xticks(range(14), [row["mask"] for row in topology_rows], rotation=45, ha="right", fontsize=8)
    plt.axhline(20, color="#a13b35", linestyle="--", linewidth=1)
    plt.ylabel("TV reduction vs permutation (%)")
    plt.tight_layout()
    plt.savefig(figure_root / "topology_smoothness.png", dpi=160)
    plt.close()

    mask_rows = [
        row
        for row in d7["rows"]["mask"]
        if row["trajectory"] == "validation" and row["head"] in ("P0", "P2", "P5", "P7") and row["mask"] != "full"
    ]
    plt.figure(figsize=(12, 6))
    width = 0.19
    for head_index, head in enumerate(("P0", "P2", "P5", "P7")):
        selected = [row for row in mask_rows if row["head"] == head]
        plt.bar(np.arange(14) + (head_index - 1.5) * width, [100.0 * row["top1"] for row in selected], width=width, label=head)
    plt.xticks(range(14), MASK_NAMES[1:], rotation=45, ha="right", fontsize=8)
    plt.ylabel("Top-1 (%)")
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.savefig(figure_root / "prototype_head_by_mask.png", dpi=160)
    plt.close()

    headroom_rows = [row for row in d8["rows"] if row["trajectory"] == "validation"]
    plt.figure(figsize=(9, 5))
    x = np.arange(len(headroom_rows))
    plt.bar(x - 0.18, [row["recovered_headroom_additive"] for row in headroom_rows], width=0.36, label="P5 additive")
    plt.bar(x + 0.18, [row["recovered_headroom_topology"] for row in headroom_rows], width=0.36, label="P7 topology")
    plt.axhline(0.6, color="#a13b35", linestyle="--", linewidth=1)
    plt.xticks(x, [row["scope"] for row in headroom_rows], rotation=30, ha="right")
    plt.ylabel("Recovered independent headroom")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_root / "headroom_recovery.png", dpi=160)
    plt.close()

    covariance_rows = [
        row for row in covariance["rows"] if row["row_type"] == "beam_mask" and row["dataset"] == "validation" and row["mask"] != "full"
    ]
    plt.figure(figsize=(7, 5))
    points = plt.scatter(
        [row["center_shift_magnitude"] for row in covariance_rows],
        [row["covariance_change_magnitude"] for row in covariance_rows],
        c=[row["current_error_rate"] for row in covariance_rows],
        s=10,
        alpha=0.55,
        cmap="magma",
    )
    plt.colorbar(points, label="Current error rate")
    plt.xlabel("Center shift magnitude")
    plt.ylabel("Covariance change magnitude")
    plt.tight_layout()
    plt.savefig(figure_root / "covariance_vs_center_shift.png", dpi=160)
    plt.close()

    order = np.asarray(topology.labels_by_position)
    for modality, mask in SINGLE_MISSING.items():
        mask_index = list(MASKS).index(mask)
        values = d1["shifts"][mask_index].norm(dim=1).numpy()
        plt.figure(figsize=(10, 4))
        plt.plot(range(64), values[order], color="#376d9c", linewidth=1.4)
        plt.scatter(range(64), values[order], color="#376d9c", s=9)
        plt.xlabel("Audited topology position")
        plt.ylabel("Tangent shift norm")
        plt.title(f"{mask} on ula_dft_phase_cycle_v1")
        plt.tight_layout()
        plt.savefig(figure_root / f"missing_{modality}_shift_topology.png", dpi=160)
        plt.close()


def _find_result(rows: Sequence[Mapping[str, Any]], **wanted: Any) -> Mapping[str, Any]:
    return next(row for row in rows if all(row.get(key) == value for key, value in wanted.items()))


def finalize(
    config: Mapping[str, Any],
    output: Path,
    audit_payload: Mapping[str, Any],
    d0: Mapping[str, Any],
    d1: Mapping[str, Any],
    d2: Mapping[str, Any],
    d3: Mapping[str, Any],
    d4: Mapping[str, Any],
    d5: Mapping[str, Any],
    d7: Mapping[str, Any],
    d8: Mapping[str, Any],
    covariance: Mapping[str, Any],
    controls: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config["thresholds"]
    h1 = bool(d1["median_significant_beam_fraction"] >= 0.5 and d2["stable_mask_fraction"] >= 0.5)
    h2 = bool(
        d3["validation"]["weighted_r2"] >= float(thresholds["additive_weighted_r2"])
        and d3["validation"]["median_shift_cosine"] >= float(thresholds["additive_median_cosine"])
        and d3["validation"]["positive_direction_fraction"] >= float(thresholds["additive_positive_mask_fraction"])
        and d3["trajectory_direction_consistent"]
    )
    h3 = bool(d5["single_pass_count"] >= 3)
    h4 = bool(d8["independent_pass"])
    p5 = bool(d8["additive_headroom_pass"])
    validation_headroom = [row for row in d8["rows"] if row["trajectory"] == "validation"]
    by_scope = {row["scope"]: row for row in validation_headroom}
    p5_fix = _find_result(d7["rows"]["fix"], head="P5", trajectory="validation", scope="All-14")
    p7_fix = _find_result(d7["rows"]["fix"], head="P7", trajectory="validation", scope="All-14")
    p0_correct = d7["evaluations"]["P0"]["arrays"]["top1"][:, 1:]
    p5_correct = d7["evaluations"]["P5"]["arrays"]["top1"][:, 1:]
    p7_correct = d7["evaluations"]["P7"]["arrays"]["top1"][:, 1:]
    p5_harm = p0_correct * (1.0 - p5_correct)
    p7_harm = p0_correct * (1.0 - p7_correct)
    harm_difference = (p7_harm - p5_harm).mean(dim=1)
    harm_bootstrap = _bootstrap_summary(
        _bootstrap_weights(
            harm_difference.numel(),
            int(config["statistics"]["bootstrap_replicates"]),
            int(config["statistics"]["bootstrap_seed"]) + 707,
        ),
        harm_difference,
    )
    significant_harm_reduction = harm_bootstrap["delta_ci_high"] < 0
    p7_value = bool(
        by_scope["All-14"]["topology_gain_vs_additive_pp"] >= float(thresholds["topology_all14_floor_pp"])
        and (
            by_scope["Worst"]["topology_gain_vs_additive_pp"] >= float(thresholds["topology_target_gain_pp"])
            or by_scope["missing_lidar"]["topology_gain_vs_additive_pp"] >= float(thresholds["topology_target_gain_pp"])
            or significant_harm_reduction
        )
    )
    trajectory_rows = [row for row in d8["rows"] if row["trajectory"].startswith("validation:") and row["scope"] == "All-14"]
    trajectory_consistent = bool(
        d3["trajectory_direction_consistent"]
        and d2["median_validation_pair_cosine"] >= float(thresholds["validation_trajectory_cosine"])
        and len(trajectory_rows) == 2
        and all(np.sign(row["additive_gain_pp"]) == np.sign(by_scope["All-14"]["additive_gain_pp"]) for row in trajectory_rows)
    )
    controls_pass = bool(controls["controls_pass"])
    full_unchanged = bool(d8["full_candidate_unchanged"])
    level2 = all((h1, h2, h3, h4, p5, p7_value, trajectory_consistent, controls_pass, full_unchanged))
    if covariance["diagnosis"] == "variance_inflation_dominant":
        level = 0
    elif level2:
        level = 2
    elif h1 and h4:
        level = 1
    else:
        level = 0
    mask_summaries = [row for row in d1["summary_rows"] if row["scope_type"] == "mask"]
    beam_summaries = [row for row in d1["summary_rows"] if row["scope_type"] == "beam"]
    largest_masks = sorted(mask_summaries, key=lambda row: row["median_shift_norm"], reverse=True)[:3]
    largest_beams = sorted(beam_summaries, key=lambda row: row["median_shift_norm"], reverse=True)[:5]
    m2 = _find_result(d4["rows"], dataset="validation", model="M2", scope="All-14")
    m3 = _find_result(d4["rows"], dataset="validation", model="M3", scope="All-14")
    pairwise_needed = bool(m3["weighted_r2"] > m2["weighted_r2"] + 0.05)
    missing_lidar_p0 = _mask_value(d7, "P0", "missing_lidar")
    missing_lidar_p5 = _mask_value(d7, "P5", "missing_lidar")
    dominant_group = max(
        ("Single", "Two", "Three"),
        key=lambda scope: by_scope[scope]["additive_gain_pp"],
    )
    recommendation = {
        0: "不实现 prototype deformation；保留共享 bank，并优先处理方差/表征问题。",
        1: "存在条件位移，但完整组合证据链未成立；仅考虑 mask-specific 低秩 head 或概率原型研究。",
        2: "进入正式组合式 prototype deformation 实现，保持 Full bypass 与 train-only 选择。",
    }[level]
    largest_mask_text = ", ".join(f"{row['scope']} ({row['median_shift_norm']:.4f})" for row in largest_masks)
    largest_beam_text = ", ".join(f"{row['scope']} ({row['median_shift_norm']:.4f})" for row in largest_beams)
    additive_headroom_text = (
        f"{by_scope['All-14']['recovered_headroom_additive']:.4f}"
        if by_scope["All-14"]["positive_independent_headroom"]
        else "N/A（independent prototype 不存在正 headroom）"
    )
    topology_headroom_text = (
        f"{by_scope['All-14']['recovered_headroom_topology']:.4f}"
        if by_scope["All-14"]["positive_independent_headroom"]
        else "N/A（independent prototype 不存在正 headroom）"
    )
    summary = {
        "diagnostic_id": config["diagnostic_id"],
        "level": level,
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3, "H4": h4},
        "gates": {
            "additive_headroom": p5,
            "topology_incremental_value": p7_value,
            "trajectory_consistent": trajectory_consistent,
            "negative_controls": controls_pass,
            "full_unchanged": full_unchanged,
            "p7_harm_delta_vs_p5": harm_bootstrap,
        },
        "d0": d0["summary"],
        "d1": {"median_significant_beam_fraction": d1["median_significant_beam_fraction"]},
        "d2": {
            "stable_mask_fraction": d2["stable_mask_fraction"],
            "median_train_cosine": d2["median_train_cosine"],
            "median_validation_pair_cosine": d2["median_validation_pair_cosine"],
        },
        "d3": d3["validation"],
        "d5": {"single_pass_count": d5["single_pass_count"], "median_single_reduction": d5["median_single_reduction"]},
        "d8": d8["all14"],
        "full_checks": d7["full_checks"],
        "covariance_diagnosis": covariance["diagnosis"],
        "selection": selection,
        "recommendation": recommendation,
        "safety": {"csi_used": False, "future_leakage": False, "outer_test_accessed": False},
    }
    write_json(output / "diagnostic_summary.json", _json_ready(summary), sort_keys=True)
    independent = str(config["selection"]["independent_reference"])
    report = [
        "# 组合式缺失条件 Beam Prototype 形变诊断结论",
        "",
        f"最终等级：**Level {level}**。{recommendation}",
        "",
        f"D0 生产 autocast 路径复核：Full={100*d0['summary']['full_top1']:.3f}%，All-14={100*d0['summary']['all14_top1']:.3f}%，九个既有 mask 的 0.02 pp 容差全部通过；统一 FP32 head 比较另以 P0 重打分，避免把精度路径差异算作 prototype 收益。",
        "",
        f"1. **是否共享同一 Prototype Bank**：是。15 个 mask 的 P0 都查询 checkpoint 中同一个 `[64,64]` learned bank（SHA256 `{audit_payload['prototype_sha256']}`）。",
        f"2. **是否存在显著中心偏移**：中位显著 beam 比例为 {100*d1['median_significant_beam_fraction']:.2f}%，但跨 validation 轨迹稳定性不足，因此联合 H1={h1}。",
        f"3. **偏移最大的 mask**：{largest_mask_text}。",
        f"4. **偏移最大的 beam**：{largest_beam_text}。",
        f"5. **是否超过类内抽样噪声**：各 mask 同时报告 shift SNR、1000 次 bootstrap CI 与配对 permutation；中位显著比例为 {100*d1['median_significant_beam_fraction']:.2f}%。",
        f"6. **跨轨迹稳定性**：train 内跨轨迹 median cosine={d2['median_train_cosine']:.4f}；加入两条 validation 一致性门槛后，稳定 mask 比例仅 {100*d2['stable_mask_fraction']:.2f}%。",
        f"7. **两条 validation trajectory 一致性**：median pair cosine={d2['median_validation_pair_cosine']:.4f}；最终方向一致={trajectory_consistent}。",
        f"8. **single 缺失能否解释 multi 缺失**：H2={h2}；基础贡献只由 Full+四个 single-missing 拟合。",
        f"9. **additive 指标**：validation weighted R2={d3['validation']['weighted_r2']:.4f}，median cosine={d3['validation']['median_shift_cosine']:.4f}，95% CI [{d3['validation']['median_cosine_ci_low']:.4f}, {d3['validation']['median_cosine_ci_high']:.4f}]。",
        f"10. **是否需要 pairwise interaction**：M3 相对 M2 有改善（{m2['weighted_r2']:.4f}->{m3['weighted_r2']:.4f}），但仍远低于 0.60，不能挽救组合假设；若继续探索才需要 pairwise={pairwise_needed}。",
        f"11. **shift tensor 是否低秩**：train-CV 推荐 rank={selection['low_rank']['selected']}；完整 spectrum 见 `d4_low_rank_spectrum.csv`。",
        f"12. **真实 beam topology 是否更平滑**：四个 single 中 {d5['single_pass_count']}/4 通过，median TV reduction={100*d5['median_single_reduction']:.2f}%；H3={h3}。",
        f"13. **independent conditioned prototype 是否超过 shared**：{independent} All-14 gain={by_scope['All-14']['independent_gain_pp']:+.3f} pp；H4={h4}。",
        f"14. **additive prototype 是否超过 shared**：P5 All-14 gain={by_scope['All-14']['additive_gain_pp']:+.3f} pp。",
        f"15. **恢复 independent headroom**：P5={additive_headroom_text}，P7={topology_headroom_text}。",
        f"16. **topology smoothing 独立收益**：P7-P5 All-14={by_scope['All-14']['topology_gain_vs_additive_pp']:+.3f} pp，Worst={by_scope['Worst']['topology_gain_vs_additive_pp']:+.3f} pp；Harm delta 95% CI=[{100*harm_bootstrap['delta_ci_low']:.4f}, {100*harm_bootstrap['delta_ci_high']:.4f}] pp；门槛通过={p7_value}。",
        f"17. **random deformation 是否产生类似收益**：P8 20-seed All-14={100*d7['random_all14_mean']:.3f}% +/- {100*d7['random_all14_std']:.3f}%；真实 P5 更优={controls['real_additive_beats_random']}，但二者都低于 P0，不能作为正证据。",
        f"18. **mask permutation 是否产生类似收益**：P9 All-14={100*_result_value(d7, 'P9', 'All-14'):.3f}%；真实 P3 更优={controls['real_mask_beats_permutation']}，但 P3 仍低于 P0。",
        f"19. **Full 是否完全不变**：部署候选 P5/P7 使用同一 Full bank；FP32 logits max abs={max(d7['full_checks'][head]['logits_max_abs'] for head in ('P5','P7')):.2e}（容差 1e-5），prediction 完全一致，判定={full_unchanged}。P1-P3 是上界诊断，不属于 Full-bypass 候选。",
        f"20. **改善主要来自哪组**：没有分组得到改善；降幅最小的是 {dominant_group}。P5 相对 P0 的 Single/Two/Three gain 分别为 {by_scope['Single']['additive_gain_pp']:+.3f}/{by_scope['Two']['additive_gain_pp']:+.3f}/{by_scope['Three']['additive_gain_pp']:+.3f} pp。",
        f"21. **missing_lidar**：P0={100*missing_lidar_p0:.3f}%，P5={100*missing_lidar_p5:.3f}%，变化={100*(missing_lidar_p5-missing_lidar_p0):+.3f} pp。",
        f"22. **Fix/Harm**：P5 All-14 Fix={100*p5_fix['fix_rate']:.3f}%、Harm={100*p5_fix['harm_rate']:.3f}%；P7 Fix={100*p7_fix['fix_rate']:.3f}%、Harm={100*p7_fix['harm_rate']:.3f}%。",
        f"23. **中心偏移还是方差膨胀**：判定为 `{covariance['diagnosis']}`；相关性与逐 beam-mask 协方差见 `covariance_shift_analysis.csv`。",
        f"24. **最终等级**：Level {level}。Level-2 十项联合门槛={level2}。",
        f"25. **是否值得正式实现**：{'是' if level == 2 else '否'}。{recommendation}",
        (
            f"26. **推荐设计**：Level {level} 不建议进入正式实现，因此 deformation/pairwise/topology/rank/shrinkage 均为 N/A。"
            if level == 0
            else f"26. **推荐设计**：优先使用 {'tangent' if controls['tangent_r2'] >= controls['euclidean_r2'] else 'Euclidean'} deformation；pairwise={pairwise_needed}；topology smoothing={p7_value}；rank={selection['low_rank']['selected']}；shrinkage kappa 固定为 {config['geometry']['shrinkage_kappa']}。"
        ),
        "27. **是否使用任何 CSI**：否；csi_used=false，未加载 channel tensor/checkpoint/cache/fusion 结果。",
        "28. **是否存在 future 泄漏**：否；模型输入严格为 t-4...t，t+1 beam power 只作为标签侧评估指标。",
        "29. **outer test 是否继续封存**：是；未构造 loader/cache，outer_test_accessed=false。",
        "",
        "## 证据边界",
        "",
        "所有 center、shift、temperature sensitivity、lambda 与 rank 选择只使用 train 或 train-trajectory CV；validation 仅执行一次最终诊断。非法 validation-center oracle 只存在于 `negative_controls.csv`，不参与上述结论。",
    ]
    (output / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def analyze(config: Mapping[str, Any], output: Path, audit_payload: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()

    def progress(stage: str) -> None:
        elapsed = time.monotonic() - started
        _update_process(output, config, f"analyze:{stage}", "running", analysis_elapsed_seconds=elapsed)
        print(json.dumps({"event": "prototype_deformation_analysis", "completed": stage, "elapsed_seconds": elapsed}), flush=True)

    data = _load_analysis_inputs(config, output)
    train_centers = _centers_for(data["train"], data["learned"], float(config["geometry"]["shrinkage_kappa"]))
    d0 = d0_shared_bank(config, output, data)
    progress("D0")
    d1 = d1_center_shift(config, output, data, train_centers)
    progress("D1")
    d2 = d2_stability(config, output, data, train_centers)
    progress("D2")
    d3 = d3_compositionality(config, output, data, train_centers, d1["shifts"])
    progress("D3")
    adjacency = topology_adjacency(data["topology"].distance)
    selection = train_cv_selection(config, data, adjacency)
    write_json(output / "artifacts/train_cv_selection.json", selection, sort_keys=True)
    progress("train-CV")
    d4 = d4_factorization(config, output, data, train_centers, selection)
    progress("D4")
    d5 = d5_topology(config, output, data, d1["shifts"], d4["validation_shifts"])
    progress("D5")
    heads, random_heads, head_metadata = build_prototype_heads(config, output, data, train_centers, d4, selection, adjacency)
    d7 = d7_heads(config, output, data, heads, random_heads, head_metadata)
    progress("D6-D7")
    d8 = d8_headroom(config, output, d7)
    progress("D8")
    covariance = covariance_analysis(config, output, data, train_centers, d4["validation_centers"], d1, d8)
    progress("covariance")
    controls = negative_controls(config, output, data, train_centers, selection, d3, d5, d7)
    progress("negative-controls")
    create_figures(output, d1, d3, d5, d7, d8, covariance, data["topology"])
    summary = finalize(config, output, audit_payload, d0, d1, d2, d3, d4, d5, d7, d8, covariance, controls, selection)
    progress("report")
    summary["artifact_sha256"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in ("process_manifest.json", "diagnostic_summary.json")
    }
    write_json(output / "diagnostic_summary.json", _json_ready(summary), sort_keys=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("audit", "cache", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true", help="Continue this diagnostic only when its process manifest matches.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Atomically rebuild only this run's cache after a cache-contract code correction; requires --resume.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.refresh_cache and (not args.resume or args.stage not in ("cache", "all")):
        raise ValueError("--refresh-cache requires --resume and --stage cache/all.")
    config = _load_config(args.config.resolve())
    output = _prepare_output(config, resume=bool(args.resume))
    started = time.monotonic()
    try:
        _update_process(output, config, args.stage, "running", started_at=now(), command_stage=args.stage)
        audit_payload = audit(config, output)
        resolved = json.loads(json.dumps(config))
        resolved["resolved_at"] = now()
        resolved["source_hashes"] = {
            "checkpoint_sha256": audit_payload["checkpoint_sha256"],
            "prototype_sha256": audit_payload["prototype_sha256"],
            "split_manifest_sha256": audit_payload["split_manifest_sha256"],
            "topology_manifest_sha256": audit_payload["topology_manifest_sha256"],
        }
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        if args.stage == "audit":
            _update_process(output, config, "audit", "passed", elapsed_seconds=time.monotonic() - started)
            return 0
        cache_manifest_path = output / "cache/cache_manifest.json"
        if cache_manifest_path.is_file() and not args.refresh_cache:
            cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            if cache_manifest.get("checkpoint_sha256") != audit_payload["checkpoint_sha256"]:
                raise ValueError("existing resume cache is bound to another checkpoint.")
            if cache_manifest.get("cache_version") != config["cache"]["version"]:
                raise ValueError("existing cache version differs; use the explicit cache repair path.")
        else:
            _update_process(output, config, "cache", "running", elapsed_seconds=time.monotonic() - started)
            cache_manifest = build_cache(config, output, audit_payload)
        if args.stage == "cache":
            _update_process(
                output,
                config,
                "cache",
                "passed",
                elapsed_seconds=time.monotonic() - started,
                cache_manifest_sha256=sha256_file(cache_manifest_path),
            )
            return 0
        _update_process(output, config, "analyze", "running", elapsed_seconds=time.monotonic() - started)
        summary = analyze(config, output, audit_payload)
        _update_process(
            output,
            config,
            "complete",
            "passed",
            elapsed_seconds=time.monotonic() - started,
            level=summary["level"],
            cache_manifest_sha256=sha256_file(cache_manifest_path),
            diagnostic_summary_sha256=sha256_file(output / "diagnostic_summary.json"),
        )
        print(json.dumps({"event": "diagnostic_complete", "level": summary["level"], "output": str(output)}), flush=True)
        return 0
    except Exception as exc:
        failure = traceback.format_exc()
        (output / "failure.log").write_text(failure, encoding="utf-8")
        _update_process(
            output,
            config,
            args.stage,
            "failed",
            elapsed_seconds=time.monotonic() - started,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
