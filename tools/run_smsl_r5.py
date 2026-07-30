#!/usr/bin/env python3
"""Run the local, train-only SMSL R5 study on the frozen M4 trajectory protocol.

This is deliberately a local experiment runner.  It does not register a new
model, change M4's inference graph, construct the sealed outer-test loader, or
publish a public CLI route.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from kd_sensing.baselines.full_pool_bt_scl import load_audited_topology
from kd_sensing.baselines.full_pool_common import atomic_csv, now, set_seed, sha256_file, sha256_json, write_json
from kd_sensing.baselines.mmw_trajectory import (
    ABTC_CONSISTENCY_WEIGHT,
    ABTC_METHOD,
    TrajectoryBaselineModel,
    availability_balanced_assignment,
    baseline_loss,
    model_contract,
    paired_missing_loss,
    topology_smoothed_consistency_loss,
)
from kd_sensing.baselines.smsl import (
    F2_FEATURE_NAMES,
    SMSL_ARMS,
    directional_margin_distillation,
    legal_f2_features,
    normalized_hard_weights,
    normalized_risk_weights,
    severe_availability,
    shuffled_weights,
    validate_legal_feature_names,
)
from kd_sensing.data.mmw.trajectory_protocol import load_trajectory_protocol
from kd_sensing.diagnostics.paired_geometry import (
    binary_probe_metrics,
    classification_groups,
    fit_logistic_probe,
    predict_logistic_probe,
    validate_train_only_selection,
)
from kd_sensing.diagnostics.prototype_deformation import MASKS, MODALITIES, mask_metadata
from kd_sensing.engine.data_factory import shutdown_dataloader_workers
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices

# These helpers are deliberately reused from the official M4 local runner so
# raw input construction, normalization, optimizer, and autocast match M4.
from run_mmw_trajectory_baselines import _autocast, _batch_ids, _inputs, _labels, _loaders, _optimizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/smsl_r5_v1.yaml"
NONFULL_MASK_NAMES = tuple(MASKS)[1:]
NONFULL_MASK_BITS = tuple(MASKS.values())[1:]
MASK_INDEX = {tuple(bits): index for index, bits in enumerate(MASKS.values())}
NONFULL_INDEX = {tuple(bits): index - 1 for bits, index in MASK_INDEX.items() if index > 0}
RESOURCE_BOUND_PHASES = frozenset(("screening", "phase2", "final-eval"))


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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty SMSL CSV: {path}")
    fields: list[str] = []
    for row in rows:
        fields.extend(name for name in row if name not in fields)
    atomic_csv(path, rows, fields, extrasaction="ignore")


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _scheduled_f2_alpha(completed_steps: int, total_steps: int, *, alpha: float, warmup_fraction: float) -> float:
    if total_steps <= 0 or completed_steps < 0 or not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("SMSL F2 warm-up schedule is invalid.")
    warmup_steps = int(math.ceil(float(warmup_fraction) * total_steps))
    return 0.0 if completed_steps < warmup_steps else float(alpha)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("experiment_id") != "smsl_r5_v1":
        raise ValueError("SMSL requires the smsl_r5_v1 config.")
    if tuple(config["training"]["arms"]) != SMSL_ARMS:
        raise ValueError("SMSL arm order changed from the pre-registered contract.")
    if tuple(config["f2"]["feature_names"]) != F2_FEATURE_NAMES:
        raise ValueError("SMSL F2 config differs from the inference-legal feature contract.")
    if config["training"]["base_method"] != ABTC_METHOD:
        raise ValueError("SMSL must start from the official ABTC M4 method.")
    runtime = config.get("runtime", {})
    expected_environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if int(runtime.get("dataloader_workers", -1)) != 8:
        raise ValueError("SMSL formal runs require the production dataloader worker count of 8.")
    if int(runtime.get("torch_intraop_threads", -1)) != 1 or int(runtime.get("torch_interop_threads", -1)) != 1:
        raise ValueError("SMSL formal runs require one PyTorch host thread per process.")
    if runtime.get("thread_environment") != expected_environment:
        raise ValueError("SMSL host thread environment differs from the frozen resource contract.")
    if tuple(MASKS) != (
        "full",
        "missing_image",
        "missing_radar",
        "missing_gps",
        "missing_lidar",
        "missing_image_radar",
        "missing_image_gps",
        "missing_image_lidar",
        "missing_radar_gps",
        "missing_lidar_radar",
        "missing_lidar_gps",
        "image_only",
        "radar_only",
        "gps_only",
        "lidar_only",
    ):
        raise ValueError("SMSL requires the audited 15-mask order.")
    validate_legal_feature_names()
    return config


def _resource_launch_command(config: Mapping[str, Any], gpu: str | int, argv: Sequence[str]) -> str:
    environment = config["runtime"]["thread_environment"]
    assignments = [f"CUDA_VISIBLE_DEVICES={gpu}"] + [f"{name}={value}" for name, value in environment.items()]
    return " ".join(["env", *assignments, "conda", "run", "--no-capture-output", "-n", "kd_mm_beam", "python", shlex.join(argv)])


def _enforce_resource_contract(config: Mapping[str, Any], *, phase: str, workers: int) -> dict[str, Any]:
    runtime = config["runtime"]
    expected_workers = int(runtime["dataloader_workers"])
    if phase in RESOURCE_BOUND_PHASES and int(workers) != expected_workers:
        raise RuntimeError(f"SMSL {phase} requires --workers {expected_workers}, got {workers}.")
    expected_environment = {str(name): str(value) for name, value in runtime["thread_environment"].items()}
    observed_environment = {name: os.environ.get(name) for name in expected_environment}
    mismatches = {
        name: {"expected": expected, "observed": observed_environment[name]}
        for name, expected in expected_environment.items()
        if observed_environment[name] != expected
    }
    if phase in RESOURCE_BOUND_PHASES and mismatches:
        prefix = " ".join(f"{name}={value}" for name, value in expected_environment.items())
        raise RuntimeError(f"SMSL {phase} host thread contract mismatch: {mismatches}. Launch with: env {prefix}")
    if phase in RESOURCE_BOUND_PHASES:
        torch.set_num_threads(int(runtime["torch_intraop_threads"]))
        try:
            torch.set_num_interop_threads(int(runtime["torch_interop_threads"]))
        except RuntimeError:
            if torch.get_num_interop_threads() != int(runtime["torch_interop_threads"]):
                raise
    return {
        "enforced": phase in RESOURCE_BOUND_PHASES,
        "dataloader_workers": int(workers),
        "thread_environment": observed_environment,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def _output(config: Mapping[str, Any]) -> Path:
    output = _path(config["output"]["root"])
    marker = output / "process_manifest.json"
    if output.exists() and marker.is_file():
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current.get("experiment_id") != config["experiment_id"]:
            raise ValueError("SMSL output root belongs to another experiment.")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"SMSL output root is non-empty without its process manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("artifacts/f2", "smoke", "screening", "phase2", "reports"):
        (output / name).mkdir(parents=True, exist_ok=True)
    return output


def _update_process(output: Path, config: Mapping[str, Any], stage: str, status: str, **extra: Any) -> None:
    path = output / "process_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload.update(
        experiment_id=config["experiment_id"],
        stage=stage,
        status=status,
        pid=os.getpid(),
        updated_at=now(),
        csi_used=False,
        channel_input_used=False,
        f1_used=False,
        outer_test_accessed=False,
    )
    payload.update(_json_ready(extra))
    if status != "failed":
        payload.pop("error", None)
        payload.pop("traceback", None)
    write_json(path, payload, sort_keys=True)


def _load_source_checkpoint(
    config: Mapping[str, Any], device: torch.device, *, frozen: bool
) -> tuple[TrajectoryBaselineModel, dict[str, Any]]:
    checkpoint = _path(config["source"]["checkpoint"])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("method") != ABTC_METHOD:
        raise ValueError("SMSL source checkpoint is not the audited M4 ABTC checkpoint.")
    model = TrajectoryBaselineModel(saved["method"], **saved["model_config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    if frozen:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, saved


def _trajectory_records(dataset: Any) -> list[dict[str, str]]:
    """Return the root dataset order with stable IDs and train trajectory IDs."""
    records: list[dict[str, str]] = []
    for leaf, indices in leaf_datasets_with_indices(dataset):
        rows = getattr(getattr(leaf, "samples", None), "rows", None)
        if not isinstance(rows, list):
            raise ValueError("SMSL needs prepared MMW rows to select trajectory folds.")
        for index in indices:
            row = rows[int(index)]
            sample = str(row.get("sample_id", "")).strip()
            trajectory = str(row.get("trajectory_group_id", "")).strip()
            if not sample or not trajectory:
                raise ValueError("SMSL needs stable sample and trajectory identities.")
            records.append(
                {
                    "sample_id": f"mmw:{leaf.condition}:{leaf.scene_slug}:train:{sample}",
                    "trajectory_id": trajectory,
                }
            )
    if len(records) != len(dataset) or len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("SMSL trajectory records do not align with the train dataset.")
    return records


def _subset_loader(
    loader: DataLoader,
    indices: Sequence[int],
    *,
    workers: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if shuffle:
        if seed is None:
            raise ValueError("SMSL shuffled loader requires an explicit seed.")
        generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        Subset(loader.dataset, [int(index) for index in indices]),
        batch_size=int(loader.batch_size or 64),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=bool(workers),
        persistent_workers=bool(workers),
        prefetch_factor=2 if workers else None,
        collate_fn=loader.collate_fn,
        drop_last=False,
        generator=generator,
    )


def _protocol_audit(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    source = config["source"]
    protocol_path = _path(config["protocol"]["manifest"])
    split_audit_path = _path(config["protocol"]["audit"])
    checkpoint = _path(source["checkpoint"])
    topology_path = _path(source["topology_manifest"])
    production_config = checkpoint.parent / "resolved_config.yaml"
    production_runner = ROOT / "tools/run_mmw_trajectory_baselines.py"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    topology_manifest = json.loads(topology_path.read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, saved = _load_source_checkpoint(config, device, frozen=True)
    legacy_path = ROOT / "outputs/full_missing_hard_sample_geometry/trajectory_v1/artifacts/hardness_probe/feature_manifest.json"
    legacy_features: list[str] = []
    if legacy_path.is_file():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        legacy_features = [str(value) for value in legacy.get("features", {}).get("F2", [])]
    rejected_legacy = "missing_target_rank" in legacy_features
    checks = {
        "checkpoint_sha256": sha256_file(checkpoint) == source["expected_checkpoint_sha256"],
        "split_sha256": sha256_file(protocol_path) == source["expected_split_manifest_sha256"],
        "topology_sha256": sha256_file(topology_path) == source["expected_topology_manifest_sha256"],
        "source_method": saved.get("method") == ABTC_METHOD,
        "fixed_temperature": float(model.prototype_bank.temperature) == 0.1,
        "prototype_shape": tuple(model.prototype_bank.prototypes.shape) == (64, 64),
        "forward_no_channel": model_contract(model)["channel_input_present"] is False,
        "protocol_passed": split_audit.get("status") == "passed",
        "outer_test_unaccessed": protocol.get("outer_test_accessed") is False and split_audit.get("outer_test_accessed") is False,
        "train_count": int(protocol["train_window_count"]) == int(config["protocol"]["expected_train_samples"]),
        "validation_count": int(protocol["validation_window_count"]) == int(config["protocol"]["expected_validation_samples"]),
        "train_trajectory_count": int(protocol["train_group_count"]) == int(config["protocol"]["expected_train_trajectories"]),
        "validation_trajectory_count": int(protocol["validation_group_count"])
        == int(config["protocol"]["expected_validation_trajectories"]),
        "all_split_overlaps_zero": all(
            int(detail["count"]) == 0 for pair in split_audit["pairwise_overlaps"].values() for detail in pair.values()
        ),
        "production_config_exists": production_config.is_file(),
        "production_evaluator_exists": production_runner.is_file(),
        "legacy_f2_rejected": rejected_legacy,
        "legal_f2_contract": tuple(config["f2"]["feature_names"]) == F2_FEATURE_NAMES,
    }
    failures = [name for name, value in checks.items() if not bool(value)]
    payload = {
        "status": "failed" if failures else "passed",
        "checks": checks,
        "failures": failures,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_manifest_sha256": sha256_file(protocol_path),
        "topology_manifest_sha256": sha256_file(topology_path),
        "production": {
            "training_config": str(production_config),
            "training_config_sha256": sha256_file(production_config),
            "training_runner": str(production_runner),
            "training_entry": "train",
            "formal_15_mask_evaluator": f"{production_runner}::evaluate_all_masks",
            "mask_owner": "kd_sensing.diagnostics.prototype_deformation.MASKS",
        },
        "m4_contract": model_contract(model),
        "prototype_scoring": {
            "prototype_bank_shape": [64, 64],
            "temperature": 0.1,
            "beam_logit_definition": "normalize(z) @ normalize(P).T / 0.1",
            "prediction_definition": "argmax over the 64 prototype beam logits",
            "prototype_changed_by_smsl": False,
        },
        "topology_distance": {
            "manifest": str(topology_path),
            "topology_id": topology_manifest["descriptor"]["topology_id"],
            "definition": "64-bin ULA-DFT phase-cycle distance min(|i-j|,64-|i-j|)",
            "endpoint_0_63_distance": 1,
            "claim_boundary": topology_manifest["descriptor"]["claim_boundary"],
        },
        "trajectory_split": {
            "protocol_id": protocol["protocol_id"],
            "train_samples": int(protocol["train_window_count"]),
            "validation_samples": int(protocol["validation_window_count"]),
            "sealed_outer_test_samples": int(protocol["test_window_count"]),
            "train_trajectories": int(protocol["train_group_count"]),
            "validation_trajectories": int(protocol["validation_group_count"]),
            "sealed_outer_test_trajectories": int(protocol["test_group_count"]),
            "train_validation_resource_overlap_count": 0,
            "outer_test_enabled": False,
            "outer_test_accessed": False,
        },
        "group_definitions": {
            "G0": "Full correct and Missing correct",
            "G1": "Full correct and Missing wrong",
            "G2": "Full wrong and Missing wrong",
            "G3": "Full wrong and Missing correct",
            "owner": "kd_sensing.diagnostics.paired_geometry.classification_groups",
        },
        "mask_metadata": mask_metadata(),
        "f2_legacy": {
            "path": str(legacy_path),
            "legacy_feature_count": len(legacy_features),
            "forbidden_feature": "missing_target_rank",
            "reused_for_smsl": False,
            "reason": "requires the true target beam label at inference",
        },
        "f2_legal_features": list(F2_FEATURE_NAMES),
        "f2_current": {
            "training_runner": str(Path(__file__)),
            "feature_owner": str(ROOT / "src/kd_sensing/baselines/smsl.py"),
            "checkpoint": str(output / "artifacts/f2/f2_checkpoint.pt"),
            "audit": str(output / "f2_audit.json"),
            "trajectory_cv": str(output / "f2_train_cv.csv"),
            "normalization": "train-only mean/scale stored in checkpoint state and f2_audit.json",
            "output": "P(G1 | inference-available missing-view features)",
        },
        "selection_contract": {
            "f2_fit_roles": ["train"],
            "screening_validation_access": False,
            "teacher_full_only": True,
            "teacher_frozen": True,
            "outer_test_accessed": False,
        },
        "csi_used": False,
        "channel_input_used": False,
        "f1_used": False,
        "csi_channel_entry_audit": {
            "protocol_channel_identity_role": "resource-overlap audit only",
            "model_channel_input_present": False,
            "pilot_or_csi_loader_constructed": False,
        },
        "outer_test_accessed": False,
    }
    write_json(output / "protocol_audit.json", payload, sort_keys=True)
    lines = [
        "# SMSL R5 协议审计",
        "",
        f"状态：**{payload['status']}**；失败项：{failures or '无'}。",
        "",
        "## 生产 M4 与 evaluator",
        "",
        f"- checkpoint：`{checkpoint}`；SHA256 `{payload['checkpoint_sha256']}`。",
        f"- 正式配置：`{production_config}`；训练入口 `{production_runner}::train`。",
        f"- 正式 15-mask evaluator：`{production_runner}::evaluate_all_masks`；mask 顺序来自现有 `MASKS`。",
        "- prototype bank 为 `[64,64]`，temperature 固定 0.1；beam logits 为 `normalize(z) @ normalize(P).T / 0.1`，Top-1 为 64 beam logits 的 argmax。",
        "",
        "## F2 与 G0--G3",
        "",
        f"- 旧 F2：`{legacy_path}`。虽为 train-only probe，但含 `missing_target_rank` 标签侧特征，已拒绝作为 SMSL scoring checkpoint。",
        f"- 新 F2 runner：`{Path(__file__)}`；特征 owner：`{ROOT / 'src/kd_sensing/baselines/smsl.py'}`；checkpoint：`{output / 'artifacts/f2/f2_checkpoint.pt'}`。",
        "- 新 F2 只含 mask、Missing top1-top2 gap、熵、embedding norm、可用单模态 agreement/disagreement；train-only mean/scale、线性 state、CV 与 score 公式记录在 `f2_audit.json`/`f2_train_cv.csv`。",
        "- `r=P(G1 | inference-available features)`，越高表示同一 subset 更可能 Full 正确而 Missing 错误；F2 冻结、detach，且不进入 M4 推理。",
        "- G0/G1/G2/G3 依次为双方正确、仅 Full 正确、双方错误、仅 Missing 正确；owner 为 `classification_groups`。",
        "",
        "## Topology 与 split",
        "",
        f"- topology：`{topology_path}`；`ula_dft_phase_cycle_v1`，距离 `min(|i-j|,64-|i-j|)`，因此 `d(0,63)=1`。",
        f"- split：`{protocol_path}`；train/validation 为 `{protocol['train_window_count']}/{protocol['validation_window_count']}` 样本、`{protocol['train_group_count']}/{protocol['validation_group_count']}` trajectories，资源级 overlap 全为 0。",
        f"- outer test 为封存的 `{protocol['test_window_count']}` 样本、`{protocol['test_group_count']}` trajectory；未构造 loader、未读取样本。",
        "",
        "## 泄漏确认",
        "",
        "- F2 不使用 Full-Missing drift、future power、future label oracle 或 validation label 训练/调参；development validation 在 screening 阶段未访问。",
        "- A3/C3 teacher 是冻结生产 M4，`eval()`、`torch.inference_mode()`，只接收同一样本的 `x_full`。",
        "- outer test 未访问；CSI/channel/F1 未构造或使用。protocol 中 channel identity 仅用于已有资源重叠审计，不进入模型。",
    ]
    (output / "protocol_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f"SMSL protocol audit failed: {failures}")
    return payload


def _source_cache(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError("SMSL source cache split must be train or validation.")
    path = _path(config["source"][f"{split}_m4_cache"])
    cache = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"sample_id", "trajectory_id", "target", "z_raw", "shared_bank_logits", "current_prediction"}
    if not expected.issubset(cache) or str(cache.get("split")) != split:
        raise ValueError(f"SMSL source cache has an invalid {split} contract.")
    if any(cache.get(flag) is not False for flag in ("csi_used", "channel_input_used", "outer_test_accessed")):
        raise ValueError("SMSL source cache violates the sensing safety contract.")
    if tuple(cache["z_raw"].shape[1:]) != (15, 64) or tuple(cache["shared_bank_logits"].shape[1:]) != (15, 64):
        raise ValueError("SMSL source cache does not preserve the 15-mask M4 outputs.")
    if len(cache["sample_id"]) != len(set(cache["sample_id"])):
        raise ValueError("SMSL source cache has non-unique sample IDs.")
    return cache


def _frozen_unimodal_logits(
    config: Mapping[str, Any],
    split: str,
    cache: Mapping[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Replay only cached encoder tokens to obtain available unimodal predictions."""
    shard_root = _path(config["source"]["geometry_cache_root"]) / "layers"
    shards = sorted(shard_root.glob(f"{split}_*.pt"))
    if not shards:
        raise FileNotFoundError(f"SMSL needs frozen geometry layer shards for {split}: {shard_root}")
    model, _ = _load_source_checkpoint(config, device, frozen=True)
    ids = [str(value) for value in cache["sample_id"]]
    index_by_id = {value: index for index, value in enumerate(ids)}
    result = torch.empty((len(ids), 4, 64), dtype=torch.float16)
    seen = torch.zeros(len(ids), dtype=torch.bool)
    with torch.inference_mode():
        for shard_path in shards:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            shard_ids = [str(value) for value in shard["sample_id"]]
            positions = [index_by_id.get(value, -1) for value in shard_ids]
            if any(value < 0 for value in positions) or len(set(positions)) != len(positions):
                raise ValueError(f"SMSL layer shard does not align with {split} cache: {shard_path}")
            tokens = torch.as_tensor(shard["encoder_tokens"], dtype=torch.float32, device=device)
            if tokens.ndim != 4 or tokens.shape[1:] != (5, 4, 64):
                raise ValueError(f"SMSL geometry shard has invalid encoder-token shape: {shard_path}")
            token_map = {name: tokens[:, :, index] for index, name in enumerate(MODALITIES)}
            output = model.forward_tokens(token_map)
            result[torch.tensor(positions, dtype=torch.long)] = output["unimodal_logits"].detach().cpu().to(torch.float16)
            seen[torch.tensor(positions, dtype=torch.long)] = True
    del model
    if not bool(seen.all()):
        raise ValueError(f"SMSL geometry layer shards do not cover every {split} cache sample.")
    return result


def _make_legal_f2_cache(config: Mapping[str, Any], output: Path, split: str) -> dict[str, Any]:
    if split == "validation" and not (output / "artifacts/f2/f2_checkpoint.pt").is_file():
        raise ValueError("SMSL validation F2 features require a frozen train-only F2 checkpoint.")
    target = output / f"artifacts/f2/{split}_legal_features.pt"
    if target.is_file():
        cached = torch.load(target, map_location="cpu", weights_only=False)
        if tuple(cached.get("feature_names", ())) == F2_FEATURE_NAMES and cached.get("split") == split:
            return cached
        raise ValueError("SMSL legal F2 cache exists with a different contract.")
    source = _source_cache(config, split)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unimodal = _frozen_unimodal_logits(config, split, source, device)
    count = len(source["sample_id"])
    features = torch.empty((count, 14, len(F2_FEATURE_NAMES)), dtype=torch.float32)
    for mask_index, bits in enumerate(NONFULL_MASK_BITS, 1):
        availability = torch.tensor(bits, dtype=torch.bool).expand(count, -1)
        features[:, mask_index - 1] = legal_f2_features(
            availability,
            source["shared_bank_logits"][:, mask_index],
            source["z_raw"][:, mask_index],
            unimodal,
        )
    payload = {
        "schema_version": 1,
        "split": split,
        "sample_id": [str(value) for value in source["sample_id"]],
        "trajectory_id": [str(value) for value in source["trajectory_id"]],
        "features": features,
        "mask_names": list(NONFULL_MASK_NAMES),
        "mask_bits": [list(value) for value in NONFULL_MASK_BITS],
        "feature_names": list(F2_FEATURE_NAMES),
        "feature_contract": "missing_view_only_v1",
        "source_m4_cache": str(_path(config["source"][f"{split}_m4_cache"])),
        "source_m4_cache_sha256": sha256_file(_path(config["source"][f"{split}_m4_cache"])),
        "source_roles": ["train"] if split == "train" else ["validation"],
        "labels_used_only_for_g1_target": split == "train",
        "f2_inference_features_include_labels": False,
        "full_output_used_as_f2_feature": False,
        "future_input_used_as_f2_feature": False,
        "trajectory_metadata_used_as_f2_feature": False,
        "csi_used": False,
        "channel_input_used": False,
        "f1_used": False,
        "outer_test_accessed": False,
    }
    if split == "train":
        target_beam = torch.as_tensor(source["target"], dtype=torch.long)
        prediction = torch.as_tensor(source["current_prediction"], dtype=torch.long)
        payload["g1_train_target"] = (prediction[:, :1].eq(target_beam[:, None]) & prediction[:, 1:].ne(target_beam[:, None])).to(
            torch.uint8
        )
    _torch_save(target, payload)
    return payload


def _fit_state(config: Mapping[str, Any], features: np.ndarray, labels: np.ndarray, *, seed: int) -> dict[str, Any]:
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("SMSL F2 fit needs both G1 classes.")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return fit_logistic_probe(
        features,
        labels,
        l2=float(config["f2"]["l2"]),
        epochs=int(config["f2"]["epochs"]),
        batch_size=int(config["f2"]["batch_size"]),
        seed=int(seed),
        device=device,
    )


def _predict_risk(cache: Mapping[str, Any], state: Mapping[str, Any]) -> torch.Tensor:
    values = torch.as_tensor(cache["features"], dtype=torch.float32).numpy()
    _, score = predict_logistic_probe(values.reshape(-1, values.shape[-1]), state)
    return torch.from_numpy(score.reshape(values.shape[:2]).astype(np.float32, copy=False))


def _trajectory_folds(cache: Mapping[str, Any], count: int) -> list[list[str]]:
    trajectories = sorted(set(str(value) for value in cache["trajectory_id"]))
    if len(trajectories) < int(count):
        raise ValueError("SMSL has fewer train trajectories than F2 CV folds.")
    return [trajectories[index :: int(count)] for index in range(int(count))]


def _f2_state_audit(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    if checkpoint.get("fit_source_roles") != ["train"] or checkpoint.get("feature_contract") != "missing_view_only_v1":
        raise ValueError("SMSL F2 checkpoint is not proven train-only and inference-legal.")
    state = checkpoint["state"]
    mean = np.asarray(state["mean"], dtype=np.float64)
    scale = np.asarray(state["scale"], dtype=np.float64)
    weight = np.asarray(state["weight"], dtype=np.float64)
    bias = float(state["bias"])
    if mean.shape != (len(F2_FEATURE_NAMES),) or scale.shape != mean.shape or not np.isfinite(mean).all():
        raise ValueError("SMSL F2 normalization is malformed.")
    if not np.isfinite(scale).all() or not bool((scale > 0).all()):
        raise ValueError("SMSL F2 normalization scale is invalid.")
    if weight.shape != mean.shape or not np.isfinite(weight).all() or not math.isfinite(bias):
        raise ValueError("SMSL F2 scoring state is invalid.")
    return {
        "normalization": {
            "fit_source_roles": ["train"],
            "definition": "z=(feature-train_mean)/train_scale",
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "sha256": sha256_json({"mean": mean.tolist(), "scale": scale.tolist()}),
        },
        "output_score": {
            "definition": "sigmoid(z @ weight + bias)",
            "semantics": "P(sample belongs to G1=Full-correct-and-Missing-wrong | inference-available features)",
            "higher_means": "higher missing-view insufficiency risk",
            "training_weight_clamp": [0.05, 0.95],
            "weight_sha256": sha256_json(weight.tolist()),
            "bias": bias,
        },
        "frozen_for_smsl_training": True,
        "detached_from_student": True,
        "used_at_m4_inference": False,
        "development_validation_used_for_fit_or_tuning": False,
        "future_power_used": False,
        "full_missing_drift_used": False,
        "future_label_oracle_used": False,
    }


def build_f2(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Fit a replacement F2 strictly from train trajectories and legal features."""
    checkpoint_path = output / "artifacts/f2/f2_checkpoint.pt"
    audit_path = output / "f2_audit.json"
    if checkpoint_path.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        audit.update(_f2_state_audit(checkpoint))
        return audit
    cache = _make_legal_f2_cache(config, output, "train")
    validate_train_only_selection({"source_roles": cache["source_roles"], "validation_leakage_oracle": False})
    features = torch.as_tensor(cache["features"], dtype=torch.float32).numpy()
    labels = torch.as_tensor(cache["g1_train_target"], dtype=torch.uint8).numpy().astype(np.int8)
    trajectories = np.asarray(cache["trajectory_id"], dtype=object)
    folds = _trajectory_folds(cache, int(config["f2"]["trajectory_cv_folds"]))
    cv_rows: list[dict[str, Any]] = []
    fold_states: dict[str, Any] = {}
    for fold_index, holdout_trajectories in enumerate(folds):
        holdout = np.isin(trajectories, holdout_trajectories)
        fit = ~holdout
        state = _fit_state(
            config,
            features[fit].reshape(-1, features.shape[-1]),
            labels[fit].reshape(-1),
            seed=int(config["f2"]["seed"]) + fold_index,
        )
        _, probability = predict_logistic_probe(features[holdout].reshape(-1, features.shape[-1]), state)
        metrics = binary_probe_metrics(labels[holdout].reshape(-1), probability)
        risk = _predict_risk(cache, state)
        _torch_save(
            output / f"artifacts/f2/f2_fold_{fold_index}_risk.pt",
            {
                "fold": fold_index,
                "holdout_trajectories": holdout_trajectories,
                "sample_id": cache["sample_id"],
                "risk": risk,
                "state": state,
                "feature_names": list(F2_FEATURE_NAMES),
                "fit_source_roles": ["train"],
                "validation_accessed": False,
                "outer_test_accessed": False,
            },
        )
        fold_states[str(fold_index)] = {
            "holdout_trajectories": holdout_trajectories,
            "metrics": metrics,
            "risk_sha256": _tensor_sha256(risk),
        }
        cv_rows.append(
            {
                "scope": f"train_trajectory_cv_fold_{fold_index}",
                "fold": fold_index,
                "holdout_trajectories": "+".join(holdout_trajectories),
                "fit_sample_count": int(fit.sum()),
                "holdout_sample_count": int(holdout.sum()),
                **metrics,
            }
        )
    state = _fit_state(
        config,
        features.reshape(-1, features.shape[-1]),
        labels.reshape(-1),
        seed=int(config["f2"]["seed"]) + 100,
    )
    risk = _predict_risk(cache, state)
    checkpoint = {
        "schema_version": 1,
        "state": state,
        "feature_names": list(F2_FEATURE_NAMES),
        "feature_contract": "missing_view_only_v1",
        "fit_source_roles": ["train"],
        "g1_target": "full_correct_missing_wrong",
        "labels_used_only_for_fit": True,
        "f2_inference_features_include_labels": False,
        "full_output_used_as_f2_feature": False,
        "future_input_used_as_f2_feature": False,
        "trajectory_metadata_used_as_f2_feature": False,
        "cache_sha256": sha256_file(output / "artifacts/f2/train_legal_features.pt"),
        "outer_test_accessed": False,
    }
    _torch_save(checkpoint_path, checkpoint)
    _torch_save(
        output / "artifacts/f2/f2_train_risk.pt",
        {
            "sample_id": cache["sample_id"],
            "risk": risk,
            "feature_names": list(F2_FEATURE_NAMES),
            "state_checkpoint_sha256": sha256_file(checkpoint_path),
            "source_roles": ["train"],
            "outer_test_accessed": False,
        },
    )
    _write_csv(output / "f2_train_cv.csv", cv_rows)
    audit = {
        "status": "passed",
        "train_only": True,
        "validation_accessed": False,
        "outer_test_accessed": False,
        "feature_names": list(F2_FEATURE_NAMES),
        "excluded_features": config["f2"]["excluded_features"],
        "train_sample_count": len(cache["sample_id"]),
        "train_trajectory_count": len(set(cache["trajectory_id"])),
        "cv": fold_states,
        "full_state_checkpoint": str(checkpoint_path),
        "full_state_checkpoint_sha256": sha256_file(checkpoint_path),
        "full_train_risk_sha256": _tensor_sha256(risk),
        "legacy_f2_reused": False,
        "legacy_f2_rejection": "missing_target_rank requires the true target label",
        **_f2_state_audit(checkpoint),
    }
    write_json(audit_path, audit, sort_keys=True)
    return audit


def _risk_lookup(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ids = [str(value) for value in payload.get("sample_id", ())]
    risk = torch.as_tensor(payload.get("risk"), dtype=torch.float32)
    if not ids or len(ids) != len(set(ids)) or risk.shape != (len(ids), 14):
        raise ValueError(f"SMSL risk lookup is malformed: {path}")
    if not bool(torch.isfinite(risk).all()) or not bool(((risk >= 0) & (risk <= 1)).all()):
        raise ValueError(f"SMSL risk lookup is not a probability matrix: {path}")
    return {
        "sample_id": ids,
        "index": {value: index for index, value in enumerate(ids)},
        "risk": risk.clamp(0.05, 0.95),
        "path": str(path),
    }


def _risk_for_batch(
    lookup: Mapping[str, Any],
    sample_ids: Sequence[str],
    availability: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    positions = [lookup["index"].get(str(value), -1) for value in sample_ids]
    if any(index < 0 for index in positions):
        raise ValueError("SMSL batch has a sample absent from its train-only F2 lookup.")
    mask_rows = [NONFULL_INDEX.get(tuple(int(value) for value in row), -1) for row in availability.detach().cpu().tolist()]
    if any(index < 0 for index in mask_rows):
        raise ValueError("SMSL risk is defined only for non-full availability masks.")
    return lookup["risk"][torch.tensor(positions), torch.tensor(mask_rows)].to(device=device, non_blocking=True)


def _fixed_mask_risk(lookup: Mapping[str, Any], train_ids: Sequence[str]) -> torch.Tensor:
    positions = [lookup["index"].get(str(value), -1) for value in train_ids]
    if any(index < 0 for index in positions):
        raise ValueError("SMSL fixed-mask control has a train ID absent from F2 risk.")
    return lookup["risk"][torch.tensor(positions)].mean(dim=0)


def _run_directory(output: Path, phase: str, arm: str, *, fold: int | None, seed: int) -> Path:
    if phase == "screening":
        if fold is None:
            raise ValueError("SMSL screening requires a train-trajectory fold.")
        return output / "screening" / f"fold_{fold}" / arm
    if phase == "phase2":
        return output / "phase2" / f"seed_{seed}" / arm
    if phase == "smoke":
        return output / "smoke" / arm
    raise ValueError(f"unknown SMSL run phase: {phase}")


def _training_contract_sha256(config: Mapping[str, Any], output: Path, *, phase: str, arm: str, seed: int, fold: int | None) -> str:
    return sha256_json(
        {
            "config": _json_ready(config),
            "phase": phase,
            "arm": arm,
            "seed": int(seed),
            "fold": fold,
            "runner_sha256": sha256_file(Path(__file__)),
            "smsl_core_sha256": sha256_file(ROOT / "src/kd_sensing/baselines/smsl.py"),
            "source_checkpoint_sha256": sha256_file(_path(config["source"]["checkpoint"])),
            "f2_checkpoint_sha256": sha256_file(output / "artifacts/f2/f2_checkpoint.pt"),
        }
    )


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _weight_for_arm(
    arm: str,
    per_sample_ce: torch.Tensor,
    severe: torch.Tensor,
    risk: torch.Tensor,
    availability: torch.Tensor,
    fixed_mask_risk: torch.Tensor,
    *,
    alpha: float,
    config: Mapping[str, Any],
    shuffle_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, str]:
    weights = torch.ones_like(per_sample_ce)
    if not bool(severe.any()):
        return weights, "none_no_severe_samples"
    limits = config["f2"]["weight_clamp"]
    lower, upper = float(limits[0]), float(limits[1])
    if arm == "a1":
        weights[severe] = normalized_hard_weights(per_sample_ce[severe], minimum=lower, maximum=upper)
        return weights, "label_side_hardness"
    if arm in {"a2", "a3", "c2"}:
        local = normalized_risk_weights(risk[severe], alpha=alpha, minimum=lower, maximum=upper)
        weights[severe] = shuffled_weights(local, generator=shuffle_generator) if arm == "c2" else local
        return weights, "f2_shuffled" if arm == "c2" else "f2_sample_specific"
    if arm == "c1":
        indices = torch.tensor(
            [NONFULL_INDEX[tuple(int(value) for value in row)] for row in availability.detach().cpu().tolist()],
            dtype=torch.long,
            device=availability.device,
        )
        raw = 1.0 + float(alpha) * fixed_mask_risk.to(availability.device)[indices[severe]]
        normalized = raw / raw.mean().clamp_min(torch.finfo(raw.dtype).tiny)
        weights[severe] = normalized.clamp(lower, upper).detach()
        return weights, "fixed_mask_f2_mean"
    return weights, "uniform"


def _adjusted_subset_loss(
    model: TrajectoryBaselineModel,
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, float]]:
    """Preserve ABTC alignment and replace only its subset CE reduction."""
    base, report = baseline_loss(model, output, labels)
    per_sample_ce = F.cross_entropy(output["logits"].float(), labels, reduction="none")
    adjusted_ce = (per_sample_ce * weights).mean()
    return base - per_sample_ce.mean() + adjusted_ce, per_sample_ce, report


def _gradient_norm(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    total = sum(float(gradient.detach().float().square().sum()) for gradient in gradients if gradient is not None)
    return math.sqrt(total)


def _gradient_probe_rows(
    per_sample_ce: torch.Tensor,
    weights: torch.Tensor,
    risk: torch.Tensor,
    availability: torch.Tensor,
    model: TrajectoryBaselineModel,
    *,
    epoch: int,
    seen_masks: set[str],
) -> list[dict[str, Any]]:
    """One per-mask/epoch autograd probe for high-risk CE contribution."""
    rows: list[dict[str, Any]] = []
    severe = severe_availability(availability)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for bits in {tuple(int(value) for value in row) for row in availability.detach().cpu().tolist()}:
        name = next(key for key, value in MASKS.items() if tuple(value) == bits)
        if name in seen_masks or sum(bits) > 2:
            continue
        selected = severe & availability.eq(torch.tensor(bits, device=availability.device)).all(dim=1)
        if int(selected.sum()) < 2:
            continue
        threshold = torch.quantile(risk[selected].detach(), 0.8)
        high = selected & risk.detach().ge(threshold)
        if not bool(high.any()):
            continue
        all_loss = (per_sample_ce[selected] * weights[selected]).mean()
        high_loss = (per_sample_ce[high] * weights[high]).mean()
        all_norm = _gradient_norm(all_loss, parameters)
        high_norm = _gradient_norm(high_loss, parameters)
        rows.append(
            {
                "epoch": epoch,
                "mask": name,
                "available_count": sum(bits),
                "sample_count": int(selected.sum()),
                "high_risk_sample_count": int(high.sum()),
                "risk_threshold_p80": float(threshold),
                "all_severe_ce_gradient_norm": all_norm,
                "high_risk_ce_gradient_norm": high_norm,
                "high_risk_subset_gradient_ratio": high_norm / max(all_norm, 1e-12),
            }
        )
        seen_masks.add(name)
    return rows


def _mask_stats_row(bucket: Mapping[str, float], *, epoch: int, arm: str, phase: str, fold: int | None, seed: int) -> dict[str, Any]:
    count = max(float(bucket.get("count", 0.0)), 1.0)
    margin_count = max(float(bucket.get("margin_sample_count", 0.0)), 1.0)
    teacher_correct = max(float(bucket.get("teacher_correct", 0.0)), 1.0)
    g1_count = max(float(bucket.get("g1_count", 0.0)), 1.0)
    error_count = max(float(bucket.get("near_error", 0.0) + bucket.get("far_error", 0.0)), 1.0)
    return {
        "phase": phase,
        "arm": arm,
        "fold": "" if fold is None else fold,
        "seed": seed,
        "epoch": epoch,
        "mask": str(bucket["mask"]),
        "available_count": int(bucket["available_count"]),
        "sample_count": int(bucket["count"]),
        "task_loss": bucket["task_loss"] / count,
        "weighted_subset_ce": bucket["weighted_subset_ce"] / count,
        "mean_weight": bucket["weight"] / count,
        "f2_risk_mean": bucket["risk"] / count,
        "f2_risk_std": math.sqrt(max(0.0, bucket["risk_sq"] / count - (bucket["risk"] / count) ** 2)),
        "f2_g1_risk_mean": bucket["g1_risk"] / g1_count,
        "f2_g1_risk_std": math.sqrt(max(0.0, bucket["g1_risk_sq"] / g1_count - (bucket["g1_risk"] / g1_count) ** 2)),
        "g1_sample_count": int(bucket["g1_count"]),
        "teacher_sample_count": int(bucket["margin_sample_count"]),
        "teacher_correct_rate": bucket["teacher_correct"] / margin_count,
        "teacher_margin_mean": bucket["teacher_margin"] / teacher_correct,
        "student_margin_mean": bucket["student_margin"] / teacher_correct,
        "margin_violation_mean": bucket["margin_violation"] / teacher_correct,
        "margin_violation_rate": bucket["margin_violation_events"] / teacher_correct,
        "gradient_norm": bucket["gradient_norm"] / count,
        "top1": bucket["top1"] / count,
        "within3": bucket["within3"] / count,
        "mae": bucket["mae"] / count,
        "near_error_rate": bucket["near_error"] / error_count,
        "far_error_rate": bucket["far_error"] / error_count,
        "far_error_frequency": bucket["far_error"] / count,
        "g1_far_error_rate": bucket["g1_far_error"] / g1_count,
        **{f"g{index}_rate": bucket[f"g{index}"] / count for index in range(4)},
    }


def _train_run(
    config: Mapping[str, Any],
    output: Path,
    *,
    phase: str,
    arm: str,
    seed: int,
    fold: int | None,
    workers: int,
) -> tuple[Path, dict[str, Any], list[int]]:
    if arm not in SMSL_ARMS:
        raise ValueError(f"unknown SMSL arm: {arm}")
    run_dir = _run_directory(output, phase, arm, fold=fold, seed=seed)
    status_path = run_dir / "status.json"
    run_contract_sha256 = _training_contract_sha256(config, output, phase=phase, arm=arm, seed=seed, fold=fold)
    if (run_dir / "final_checkpoint.pt").is_file() and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "passed":
            if status.get("run_contract_sha256") != run_contract_sha256:
                raise ValueError(f"SMSL existing run does not match the frozen contract: {run_dir}")
            return run_dir, status, [int(value) for value in status.get("heldout_indices", [])]
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_trajectory_protocol(_path(config["protocol"]["manifest"]))
    normalization_root = _path(config["source"]["normalization_root"])
    loaders, _, normalization = _loaders(normalization_root, protocol, create_normalization=False)
    base_loader = loaders["train"]
    records = _trajectory_records(base_loader.dataset)
    train_cache = torch.load(output / "artifacts/f2/train_legal_features.pt", map_location="cpu", weights_only=False)
    if set(row["sample_id"] for row in records) != set(str(value) for value in train_cache["sample_id"]):
        raise ValueError("SMSL raw train dataset and train-only legal F2 cache do not have the same samples.")
    all_trajectories = sorted({row["trajectory_id"] for row in records})
    if phase == "screening":
        folds = _trajectory_folds(train_cache, int(config["screening"]["folds"]))
        if fold is None or not 0 <= int(fold) < len(folds):
            raise ValueError("SMSL screening fold is outside the pre-registered range.")
        heldout_trajectories = set(folds[int(fold)])
        train_indices = [index for index, row in enumerate(records) if row["trajectory_id"] not in heldout_trajectories]
        heldout_indices = [index for index, row in enumerate(records) if row["trajectory_id"] in heldout_trajectories]
        risk_path = output / f"artifacts/f2/f2_fold_{fold}_risk.pt"
        epochs = int(config["training"]["screening_epochs"])
        max_batches = None
    elif phase == "phase2":
        heldout_trajectories = set()
        train_indices = list(range(len(records)))
        heldout_indices = []
        risk_path = output / "artifacts/f2/f2_train_risk.pt"
        epochs = int(config["training"]["epochs"])
        max_batches = None
    elif phase == "smoke":
        heldout_trajectories = set()
        train_indices = list(range(min(len(records), int(config["training"]["batch_size"]) * int(config["training"]["smoke_batches"]))))
        heldout_indices = []
        risk_path = output / "artifacts/f2/f2_train_risk.pt"
        epochs = 1
        max_batches = int(config["training"]["smoke_batches"])
    else:
        raise ValueError(f"unsupported SMSL training phase: {phase}")
    if not train_indices or (phase == "screening" and not heldout_indices):
        raise ValueError("SMSL selected an empty train or heldout trajectory partition.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(int(seed))
    train_loader = _subset_loader(base_loader, train_indices, workers=workers, shuffle=True, seed=seed)
    steps_per_epoch = min(len(train_loader), max_batches or len(train_loader))
    if steps_per_epoch <= 0:
        raise ValueError("SMSL selected no training steps.")
    total_optimizer_steps = epochs * steps_per_epoch
    f2_warmup_steps = int(math.ceil(float(config["f2"]["warmup_fraction"]) * total_optimizer_steps))
    lookup = _risk_lookup(risk_path)
    train_ids = [records[index]["sample_id"] for index in train_indices]
    fixed_mask_risk = _fixed_mask_risk(lookup, train_ids)
    model, source_saved = _load_source_checkpoint(config, device, frozen=False)
    source_state_hash = _state_hash(model)
    teacher: TrajectoryBaselineModel | None = None
    if arm in {"a3", "c3"}:
        teacher, _ = _load_source_checkpoint(config, device, frozen=True)
        if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
            raise AssertionError("SMSL Full teacher must be frozen and in eval mode.")
    topology = load_audited_topology(_path(config["source"]["topology_manifest"]))
    topology_distance = topology.distance.to(device=device, dtype=torch.float32)
    optimizer, scheduler = _optimizer(model, epochs, steps_per_epoch)
    # Keep student stochasticity identical across arms after optional teacher
    # construction. C2 uses its own generator so permutations do not perturb it.
    set_seed(int(seed))
    shuffle_generator = torch.Generator(device=device).manual_seed(int(seed) + 880301)
    run_started_at = now()
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    command = _resource_launch_command(config, visible_gpu, sys.argv)
    resolved = {
        "phase": phase,
        "arm": arm,
        "seed": int(seed),
        "fold": fold,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "f2_warmup_fraction": float(config["f2"]["warmup_fraction"]),
        "f2_warmup_steps": f2_warmup_steps,
        "batch_size": int(config["training"]["batch_size"]),
        "source_checkpoint": str(_path(config["source"]["checkpoint"])),
        "source_checkpoint_sha256": sha256_file(_path(config["source"]["checkpoint"])),
        "source_state_hash": source_state_hash,
        "normalization_fingerprint": normalization["metadata"]["normalization_fingerprint"],
        "selected_train_trajectories": sorted(set(records[index]["trajectory_id"] for index in train_indices)),
        "heldout_train_trajectories": sorted(heldout_trajectories),
        "checkpoint_selection": config["training"]["checkpoint_selection"],
        "validation_accessed_during_training": False,
        "outer_test_accessed": False,
        "teacher": {
            "enabled": teacher is not None,
            "source": "frozen_full_m4",
            "input": "x_full_only",
            "eval": True,
            "requires_grad": False,
            "inference_mode": True,
        },
        "risk_lookup": lookup["path"],
        "m4_model_contract": model_contract(model),
        "pid": os.getpid(),
        "visible_gpu": visible_gpu,
        "host_runtime": _enforce_resource_contract(config, phase=phase, workers=workers),
        "command": command,
        "started_at": run_started_at,
        "run_contract_sha256": run_contract_sha256,
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(_json_ready(resolved), sort_keys=False), encoding="utf-8")
    curve_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    started = time.monotonic()
    completed = 0
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            assignments = availability_balanced_assignment(train_ids, epoch=epoch, seed=int(seed))
            totals: dict[str, float] = defaultdict(float)
            buckets: dict[str, dict[str, float]] = {}
            gradient_seen: set[str] = set()
            epoch_started = time.monotonic()
            for step, batch in enumerate(train_loader):
                if max_batches is not None and step >= max_batches:
                    break
                optimizer.zero_grad(set_to_none=True)
                labels = _labels(batch, device)
                batch_ids = _batch_ids(batch)
                availability = torch.as_tensor([assignments[value] for value in batch_ids], dtype=torch.bool, device=device)
                with _autocast(device):
                    inputs = _inputs(batch, device)
                    full_output, missing_output = model.forward_paired(inputs, availability)
                    # M4's prototype-alignment path must remain inside the
                    # official bf16 autocast boundary.
                    full_loss, full_report = baseline_loss(model, full_output, labels)
                risk = _risk_for_batch(lookup, batch_ids, availability, device)
                severe = severe_availability(availability)
                missing_prediction = missing_output["logits"].argmax(dim=1)
                full_prediction = full_output["logits"].argmax(dim=1)
                error_distance = topology_distance[labels, missing_prediction]
                top1 = missing_prediction.eq(labels)
                within3 = error_distance.le(int(config["evaluation"]["near_cycle_distance"]))
                near_error = ~top1 & within3
                far_error = ~top1 & ~within3
                groups = classification_groups(full_prediction, missing_prediction, labels).squeeze(1)
                # The CE itself is always FP32.  The rest of M4's ABTC path is
                # retained verbatim, including topology alignment and KL.
                per_sample_ce = F.cross_entropy(missing_output["logits"].float(), labels, reduction="none")
                alpha = _scheduled_f2_alpha(
                    completed,
                    total_optimizer_steps,
                    alpha=float(config["f2"]["alpha"]),
                    warmup_fraction=float(config["f2"]["warmup_fraction"]),
                )
                weights, weighting = _weight_for_arm(
                    arm,
                    per_sample_ce,
                    severe,
                    risk,
                    availability,
                    fixed_mask_risk,
                    alpha=alpha,
                    config=config,
                    shuffle_generator=shuffle_generator,
                )
                with _autocast(device):
                    adjusted_subset, per_sample_ce, subset_report = _adjusted_subset_loss(model, missing_output, labels, weights)
                    consistency = topology_smoothed_consistency_loss(missing_output["logits"], full_output["logits"], topology_distance)
                    task_loss = (
                        0.5 * (full_loss + float(config["training"]["lambda_subset"]) * adjusted_subset)
                        + ABTC_CONSISTENCY_WEIGHT * consistency
                    )
                margin_loss = task_loss * 0.0
                margin_values: Mapping[str, torch.Tensor] | None = None
                if teacher is not None and bool(severe.any()):
                    with torch.inference_mode():
                        teacher_logits = teacher(_inputs(batch, device))["logits"]
                    margin_values = directional_margin_distillation(
                        missing_output["logits"][severe],
                        teacher_logits[severe],
                        labels[severe],
                        weights=weights[severe] if arm == "a3" else None,
                    )
                    margin_loss = margin_values["loss"]
                loss = task_loss + float(config["training"]["lambda_margin"]) * margin_loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"SMSL {arm} produced non-finite loss at epoch {epoch}, step {step}.")
                gradient_rows.extend(
                    _gradient_probe_rows(
                        per_sample_ce,
                        weights,
                        risk,
                        availability,
                        model,
                        epoch=epoch,
                        seen_masks=gradient_seen,
                    )
                )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError(f"SMSL {arm} produced a non-finite gradient norm.")
                optimizer.step()
                scheduler.step()
                count = labels.numel()
                completed += 1
                totals["count"] += count
                totals["loss"] += float(loss.detach()) * count
                totals["task_loss"] += float(task_loss.detach()) * count
                totals["margin_loss"] += float(margin_loss.detach()) * count
                totals["full_ce"] += float(full_report["ce"]) * count
                totals["subset_ce"] += float(subset_report["ce"]) * count
                totals["consistency"] += float(consistency.detach()) * count
                totals["severe_count"] += int(severe.sum())
                totals["f2_weight_active_steps"] += int(alpha > 0.0)
                for bits in {tuple(int(value) for value in row) for row in availability.detach().cpu().tolist()}:
                    name = next(key for key, value in MASKS.items() if tuple(value) == bits)
                    chosen = availability.eq(torch.tensor(bits, device=device)).all(dim=1)
                    bucket = buckets.setdefault(
                        name,
                        defaultdict(float, mask=name, available_count=sum(bits)),
                    )
                    local_count = int(chosen.sum())
                    bucket["count"] += local_count
                    bucket["task_loss"] += float(per_sample_ce[chosen].detach().sum())
                    bucket["weighted_subset_ce"] += float((per_sample_ce[chosen] * weights[chosen]).mean().detach()) * local_count
                    bucket["weight"] += float(weights[chosen].detach().sum())
                    bucket["risk"] += float(risk[chosen].detach().sum())
                    bucket["risk_sq"] += float(risk[chosen].detach().square().sum())
                    bucket["gradient_norm"] += float(gradient_norm.detach()) * local_count
                    bucket["top1"] += float(top1[chosen].sum())
                    bucket["within3"] += float(within3[chosen].sum())
                    bucket["mae"] += float(error_distance[chosen].sum())
                    bucket["near_error"] += float(near_error[chosen].sum())
                    bucket["far_error"] += float(far_error[chosen].sum())
                    for group_index in range(4):
                        bucket[f"g{group_index}"] += float(groups[chosen].eq(group_index).sum())
                    local_g1 = chosen & groups.eq(1)
                    bucket["g1_count"] += int(local_g1.sum())
                    bucket["g1_risk"] += float(risk[local_g1].detach().sum())
                    bucket["g1_risk_sq"] += float(risk[local_g1].detach().square().sum())
                    bucket["g1_far_error"] += float(far_error[local_g1].sum())
                    if margin_values is not None:
                        local_severe = chosen[severe]
                        # ``margin_values`` is indexed only over severe rows.
                        eligible = margin_values["teacher_correct"][local_severe]
                        bucket["margin_sample_count"] += int(local_severe.sum())
                        bucket["teacher_correct"] += float(eligible.sum().detach())
                        bucket["teacher_margin"] += float(margin_values["teacher_margin"][local_severe][eligible].sum().detach())
                        bucket["student_margin"] += float(margin_values["student_margin"][local_severe][eligible].sum().detach())
                        local_violation = margin_values["violation"][local_severe][eligible]
                        bucket["margin_violation"] += float(local_violation.sum().detach())
                        bucket["margin_violation_events"] += int(local_violation.gt(0).sum())
                if completed % 50 == 0 or step + 1 == steps_per_epoch:
                    elapsed = time.monotonic() - started
                    write_json(
                        run_dir / "runtime_status.json",
                        {
                            "status": "running",
                            "phase": phase,
                            "arm": arm,
                            "fold": fold,
                            "seed": seed,
                            "epoch": epoch,
                            "optimizer_step": completed,
                            "latest_train_loss": float(loss.detach()),
                            "weighting": weighting,
                            "f2_alpha": alpha,
                            "f2_warmup_steps": f2_warmup_steps,
                            "elapsed_seconds": elapsed,
                            "estimated_remaining_seconds": elapsed * (epochs * steps_per_epoch / max(completed, 1) - 1),
                            "pid": os.getpid(),
                            "visible_gpu": visible_gpu,
                            "outer_test_accessed": False,
                        },
                    )
            count = max(totals["count"], 1.0)
            row = {
                "phase": phase,
                "arm": arm,
                "fold": "" if fold is None else fold,
                "seed": seed,
                "epoch": epoch,
                "optimizer_steps": completed,
                "train_samples": int(totals["count"]),
                "train_loss": totals["loss"] / count,
                "task_loss": totals["task_loss"] / count,
                "margin_loss": totals["margin_loss"] / count,
                "full_ce": totals["full_ce"] / count,
                "subset_ce": totals["subset_ce"] / count,
                "topology_consistency": totals["consistency"] / count,
                "severe_fraction": totals["severe_count"] / count,
                "configured_f2_alpha": float(config["f2"]["alpha"]),
                "last_step_f2_alpha": alpha,
                "f2_warmup_steps": f2_warmup_steps,
                "f2_weight_active_step_fraction": totals["f2_weight_active_steps"] / max(steps_per_epoch, 1),
                "lr": optimizer.param_groups[-1]["lr"],
                "epoch_seconds": time.monotonic() - epoch_started,
                "validation_accessed": False,
            }
            curve_rows.append(row)
            mask_rows.extend(
                _mask_stats_row(bucket, epoch=epoch, arm=arm, phase=phase, fold=fold, seed=seed) for bucket in buckets.values()
            )
            _write_csv(run_dir / "training_curve.csv", curve_rows)
            if mask_rows:
                _write_csv(run_dir / "per_epoch_mask_stats.csv", mask_rows)
            if gradient_rows:
                _write_csv(run_dir / "gradient_contribution.csv", gradient_rows)
            _torch_save(
                run_dir / "resume_checkpoint.pt",
                {
                    "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "optimizer_steps": completed,
                    "method": ABTC_METHOD,
                    "phase": phase,
                    "arm": arm,
                    "seed": seed,
                    "outer_test_accessed": False,
                    "run_contract_sha256": run_contract_sha256,
                },
            )
            print(json.dumps({"event": "smsl_epoch", **row}), flush=True)
        checkpoint = run_dir / "final_checkpoint.pt"
        torch.save(
            {
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "method": ABTC_METHOD,
                "model_config": source_saved["model_config"],
                "source_checkpoint_sha256": sha256_file(_path(config["source"]["checkpoint"])),
                "source_state_hash": source_state_hash,
                "phase": phase,
                "arm": arm,
                "seed": seed,
                "fold": fold,
                "checkpoint_selection": "final_epoch_train_only",
                "optimizer_steps": completed,
                "outer_test_accessed": False,
                "run_contract_sha256": run_contract_sha256,
            },
            checkpoint,
        )
        reloaded = TrajectoryBaselineModel(ABTC_METHOD, **source_saved["model_config"])
        reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"], strict=True)
        status = {
            "status": "passed",
            "phase": phase,
            "arm": arm,
            "seed": seed,
            "fold": fold,
            "epochs_completed": epochs,
            "optimizer_steps": completed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_selection": "final_epoch_train_only",
            "validation_accessed_during_training": False,
            "heldout_indices": heldout_indices if phase == "screening" else [],
            "pid": os.getpid(),
            "visible_gpu": visible_gpu,
            "command": command,
            "started_at": run_started_at,
            "ended_at": now(),
            "return_code": 0,
            "outer_test_accessed": False,
            "run_contract_sha256": run_contract_sha256,
        }
        write_json(status_path, status, sort_keys=True)
        write_json(
            run_dir / "runtime_status.json",
            {**status, "elapsed_seconds": time.monotonic() - started, "status": "passed"},
            sort_keys=True,
        )
        return run_dir, status, heldout_indices
    except Exception as exc:
        write_json(
            status_path,
            {
                "status": "failed",
                "phase": phase,
                "arm": arm,
                "fold": fold,
                "seed": seed,
                "pid": os.getpid(),
                "visible_gpu": visible_gpu,
                "command": command,
                "started_at": run_started_at,
                "ended_at": now(),
                "return_code": 1,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "outer_test_accessed": False,
                "run_contract_sha256": run_contract_sha256,
            },
            sort_keys=True,
        )
        raise
    finally:
        if teacher is not None:
            del teacher
        del model
        shutdown_dataloader_workers(train_loader)
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validation_risk_lookup(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    target = output / "artifacts/f2/f2_validation_risk.pt"
    if target.is_file():
        saved = torch.load(target, map_location="cpu", weights_only=False)
        if (
            saved.get("source_roles") != ["train"]
            or saved.get("evaluation_role") != "validation_final_only"
            or saved.get("state_checkpoint_sha256") != sha256_file(output / "artifacts/f2/f2_checkpoint.pt")
        ):
            raise ValueError("SMSL validation risk exists with a stale or non-train-only contract.")
        return _risk_lookup(target)
    checkpoint = torch.load(output / "artifacts/f2/f2_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("fit_source_roles") != ["train"]:
        raise ValueError("SMSL validation risk requires a train-only F2 checkpoint.")
    cache = _make_legal_f2_cache(config, output, "validation")
    risk = _predict_risk(cache, checkpoint["state"])
    _torch_save(
        target,
        {
            "sample_id": cache["sample_id"],
            "risk": risk,
            "feature_names": list(F2_FEATURE_NAMES),
            "state_checkpoint_sha256": sha256_file(output / "artifacts/f2/f2_checkpoint.pt"),
            "source_roles": ["train"],
            "evaluation_role": "validation_final_only",
            "outer_test_accessed": False,
        },
    )
    return _risk_lookup(target)


def _load_run_model(run_dir: Path, device: torch.device) -> tuple[TrajectoryBaselineModel, Mapping[str, Any]]:
    checkpoint = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("method") != ABTC_METHOD or checkpoint.get("checkpoint_selection") != "final_epoch_train_only":
        raise ValueError("SMSL run checkpoint violates its M4/final-epoch selection contract.")
    model = TrajectoryBaselineModel(ABTC_METHOD, **checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def _append_record(store: dict[str, dict[str, list[Any]]], name: str, **values: Any) -> None:
    target = store.setdefault(name, defaultdict(list))
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            target[key].append(value.detach().cpu())
        elif isinstance(value, np.ndarray):
            target[key].append(torch.from_numpy(value))
        elif isinstance(value, list):
            target[key].extend(value)
        else:
            target[key].append(value)


def _concat(values: Sequence[Any], *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensors = [torch.as_tensor(value) for value in values]
    result = torch.cat(tensors, dim=0) if tensors else torch.empty(0)
    return result.to(dtype=dtype) if dtype is not None else result


def _mask_metric_row(name: str, values: Mapping[str, list[Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    top1 = _concat(values["top1"], dtype=torch.float32)
    top3 = _concat(values["top3"], dtype=torch.float32)
    within3 = _concat(values["within3"], dtype=torch.float32)
    mae = _concat(values["mae"], dtype=torch.float32)
    near_error = _concat(values["near_error"], dtype=torch.float32)
    far_error = _concat(values["far_error"], dtype=torch.float32)
    groups = _concat(values["group"], dtype=torch.long)
    risk = _concat(values["risk"], dtype=torch.float32)
    margin_violation = _concat(values["margin_violation"], dtype=torch.float32)
    teacher_correct = _concat(values["teacher_correct"], dtype=torch.float32)
    teacher_margin = _concat(values["teacher_margin"], dtype=torch.float32)
    student_margin = _concat(values["student_margin"], dtype=torch.float32)
    bits = MASKS[name]
    count = int(top1.numel())
    if count == 0:
        raise ValueError(f"SMSL evaluation has no samples for {name}.")
    eligible = teacher_correct.bool()
    eligible_count = max(int(eligible.sum()), 1)
    eligible_violation = margin_violation[eligible]
    error_count = max(int(near_error.sum() + far_error.sum()), 1)
    g1 = groups.eq(1)
    g1_count = max(int(g1.sum()), 1)
    row = {
        "mask": name,
        "mask_id": MASK_INDEX[tuple(bits)],
        "available_count": sum(bits),
        "sample_count": count,
        "top1": float(top1.mean()),
        "top3": float(top3.mean()),
        "within3": float(within3.mean()),
        "mae": float(mae.mean()),
        "near_error_rate": float(near_error.sum() / error_count),
        "far_error_rate": float(far_error.sum() / error_count),
        "far_error_frequency": float(far_error.mean()),
        "g1_far_error_rate": float(far_error[g1].sum() / g1_count),
        **{f"g{index}_rate": float(groups.eq(index).float().mean()) for index in range(4)},
        "teacher_correct_rate": float(teacher_correct.mean()),
        "margin_violation_mean": float(eligible_violation.sum() / eligible_count),
    }
    risk_rows: list[dict[str, Any]] = []
    valid = torch.isfinite(risk)
    if bool(valid.any()):
        valid_risk, valid_top1, valid_group = risk[valid], top1[valid], groups[valid]
        order = valid_risk.argsort(descending=True, stable=True)
        high_count = max(1, int(math.ceil(0.2 * len(order))))
        high, low = order[:high_count], order[high_count:]
        for bucket, selected in (("high_risk_top20", high), ("lower_risk_bottom80", low)):
            if selected.numel() == 0:
                continue
            risk_rows.append(
                {
                    "mask": name,
                    "available_count": sum(bits),
                    "risk_bucket": bucket,
                    "sample_count": int(selected.numel()),
                    "risk_mean": float(valid_risk[selected].mean()),
                    "top1": float(valid_top1[selected].mean()),
                    "g1_rate": float(valid_group[selected].eq(1).float().mean()),
                }
            )
        row["f2_high20_top1"] = float(valid_top1[high].mean())
        row["f2_high20_g1_rate"] = float(valid_group[high].eq(1).float().mean())
        row["f2_risk_mean"] = float(valid_risk.mean())
    else:
        row["f2_high20_top1"] = float("nan")
        row["f2_high20_g1_rate"] = float("nan")
        row["f2_risk_mean"] = float("nan")
    margin = {
        "mask": name,
        "available_count": sum(bits),
        "sample_count": count,
        "teacher_correct_count": int(eligible.sum()),
        "teacher_correct_rate": float(teacher_correct.mean()),
        "teacher_margin_mean": float(teacher_margin[eligible].sum() / eligible_count),
        "student_margin_mean": float(student_margin[eligible].sum() / eligible_count),
        "margin_violation_mean": float(eligible_violation.sum() / eligible_count),
        "margin_violation_rate": float(eligible_violation.gt(0).sum() / eligible_count),
    }
    return row, risk_rows, margin


def _aggregate_masks(rows: Sequence[Mapping[str, Any]], names: Sequence[str], scope: str) -> dict[str, Any]:
    chosen = [row for row in rows if row["mask"] in set(names)]
    if not chosen:
        raise ValueError(f"SMSL aggregate scope is empty: {scope}")
    return {
        "scope": scope,
        "mask_count": len(chosen),
        "sample_count_per_mask": int(chosen[0]["sample_count"]),
        "top1_macro": float(np.mean([float(row["top1"]) for row in chosen])),
        "top1_worst": float(min(float(row["top1"]) for row in chosen)),
        "top3_macro": float(np.mean([float(row["top3"]) for row in chosen])),
        "within3_macro": float(np.mean([float(row["within3"]) for row in chosen])),
        "within3_worst": float(min(float(row["within3"]) for row in chosen)),
        "mae_macro": float(np.mean([float(row["mae"]) for row in chosen])),
        "mae_worst": float(max(float(row["mae"]) for row in chosen)),
        "far_error_macro": float(np.mean([float(row["far_error_rate"]) for row in chosen])),
        "far_error_worst": float(max(float(row["far_error_rate"]) for row in chosen)),
        "far_error_frequency_macro": float(np.mean([float(row["far_error_frequency"]) for row in chosen])),
        "g1_far_error_macro": float(np.mean([float(row["g1_far_error_rate"]) for row in chosen])),
        "g1_far_error_worst": float(max(float(row["g1_far_error_rate"]) for row in chosen)),
    }


def _evaluate_run(
    config: Mapping[str, Any],
    output: Path,
    run_dir: Path,
    *,
    role: str,
    heldout_indices: Sequence[int] | None,
    fold: int | None,
    workers: int,
) -> dict[str, Any]:
    summary_path = run_dir / "evaluation_summary.json"
    if summary_path.is_file():
        saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved_summary.get("role") != role or saved_summary.get("checkpoint_sha256") != sha256_file(run_dir / "final_checkpoint.pt"):
            raise ValueError(f"SMSL evaluation summary is stale or has the wrong role: {summary_path}")
        return saved_summary
    if role not in {"train_heldout", "validation"}:
        raise ValueError("SMSL evaluation role must be train_heldout or validation.")
    protocol = load_trajectory_protocol(_path(config["protocol"]["manifest"]))
    loaders, _, _ = _loaders(_path(config["source"]["normalization_root"]), protocol, create_normalization=False)
    if role == "train_heldout":
        if not heldout_indices:
            raise ValueError("SMSL train-heldout evaluation needs explicit screening indices.")
        loader = _subset_loader(loaders["train"], heldout_indices, workers=workers, shuffle=False)
        lookup = _risk_lookup(output / f"artifacts/f2/f2_fold_{fold}_risk.pt")
        validation_accessed = False
    else:
        loader = _subset_loader(loaders["validation"], list(range(len(loaders["validation"].dataset))), workers=workers, shuffle=False)
        lookup = _validation_risk_lookup(config, output)
        validation_accessed = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint = _load_run_model(run_dir, device)
    teacher, _ = _load_source_checkpoint(config, device, frozen=True)
    topology = load_audited_topology(_path(config["source"]["topology_manifest"]))
    distance = topology.distance.to(device=device, dtype=torch.long)
    stores: dict[str, dict[str, list[Any]]] = {}
    started = time.monotonic()
    try:
        model.eval()
        teacher.eval()
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, 1):
                labels = _labels(batch, device)
                sample_ids = _batch_ids(batch)
                trajectories = [str(value) for value in batch["metadata"]["trajectory_group_id"]]
                inputs = _inputs(batch, device)
                with _autocast(device):
                    tokens = model.encode(inputs)
                    teacher_logits = teacher(inputs)["logits"].float()
                outputs = {}
                for name, bits in MASKS.items():
                    with _autocast(device):
                        outputs[name] = model.forward_tokens(
                            tokens,
                            availability=torch.tensor(bits, dtype=torch.bool, device=device).expand(labels.numel(), -1),
                        )["logits"].float()
                full_prediction = outputs["full"].argmax(dim=1)
                for name, bits in MASKS.items():
                    logits = outputs[name]
                    prediction = logits.argmax(dim=1)
                    error_distance = distance[labels, prediction].float()
                    top3 = logits.topk(3, dim=1).indices.eq(labels[:, None]).any(dim=1)
                    group = classification_groups(full_prediction, prediction, labels).squeeze(1)
                    margin = directional_margin_distillation(logits, teacher_logits, labels)
                    if name == "full":
                        risk = torch.full((labels.numel(),), float("nan"), device=device)
                    else:
                        availability = torch.tensor(bits, dtype=torch.bool, device=device).expand(labels.numel(), -1)
                        risk = _risk_for_batch(lookup, sample_ids, availability, device)
                    _append_record(
                        stores,
                        name,
                        trajectory=trajectories,
                        top1=prediction.eq(labels).to(torch.float32),
                        top3=top3.to(torch.float32),
                        within3=error_distance.le(int(config["evaluation"]["near_cycle_distance"])).to(torch.float32),
                        mae=error_distance,
                        near_error=(prediction.ne(labels) & error_distance.le(int(config["evaluation"]["near_cycle_distance"]))).to(
                            torch.float32
                        ),
                        far_error=(prediction.ne(labels) & error_distance.gt(int(config["evaluation"]["near_cycle_distance"]))).to(
                            torch.float32
                        ),
                        group=group,
                        risk=risk,
                        teacher_correct=margin["teacher_correct"].to(torch.float32),
                        teacher_margin=margin["teacher_margin"],
                        student_margin=margin["student_margin"],
                        margin_violation=margin["violation"],
                    )
                if batch_index % 10 == 0 or batch_index == len(loader):
                    elapsed = time.monotonic() - started
                    write_json(
                        run_dir / "evaluation_runtime_status.json",
                        {
                            "status": "running",
                            "role": role,
                            "completed_batches": batch_index,
                            "total_batches": len(loader),
                            "elapsed_seconds": elapsed,
                            "estimated_remaining_seconds": elapsed * (len(loader) / batch_index - 1),
                            "outer_test_accessed": False,
                        },
                    )
        per_mask: list[dict[str, Any]] = []
        risk_rows: list[dict[str, Any]] = []
        margin_rows: list[dict[str, Any]] = []
        for name in MASKS:
            row, local_risk, margin = _mask_metric_row(name, stores[name])
            per_mask.append(row)
            risk_rows.extend(local_risk)
            margin_rows.append(margin)
        scopes = {
            "Full": ("full",),
            "Three": tuple(name for name, bits in MASKS.items() if sum(bits) == 3),
            "Two": tuple(name for name, bits in MASKS.items() if sum(bits) == 2),
            "Single": tuple(name for name, bits in MASKS.items() if sum(bits) == 1),
            "Severe": tuple(name for name, bits in MASKS.items() if 1 <= sum(bits) <= 2),
            "All14": NONFULL_MASK_NAMES,
        }
        aggregate = [_aggregate_masks(per_mask, names, scope) for scope, names in scopes.items()]
        _write_csv(run_dir / "per_mask_metrics.csv", per_mask)
        _write_csv(run_dir / "risk_bucket_metrics.csv", risk_rows)
        _write_csv(run_dir / "margin_recovery.csv", margin_rows)
        _write_csv(run_dir / "aggregate_metrics.csv", aggregate)
        records = {
            "role": role,
            "mask_records": {
                name: {key: value if key == "trajectory" else _concat(value) for key, value in payload.items()}
                for name, payload in stores.items()
            },
            "outer_test_accessed": False,
        }
        _torch_save(run_dir / "evaluation_records.pt", records)
        by_scope = {row["scope"]: row for row in aggregate}
        summary = {
            "status": "passed",
            "role": role,
            "validation_accessed": validation_accessed,
            "outer_test_accessed": False,
            "checkpoint_sha256": sha256_file(run_dir / "final_checkpoint.pt"),
            "checkpoint_selection": checkpoint["checkpoint_selection"],
            "masks": per_mask,
            "aggregate": by_scope,
            "single_worst_top1": by_scope["Single"]["top1_worst"],
            "severe_worst_top1": by_scope["Severe"]["top1_worst"],
            "single_within3_worst": by_scope["Single"]["within3_worst"],
            "severe_within3_worst": by_scope["Severe"]["within3_worst"],
            "single_mae_worst": by_scope["Single"]["mae_worst"],
            "severe_mae_worst": by_scope["Severe"]["mae_worst"],
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(summary_path, summary, sort_keys=True)
        write_json(run_dir / "evaluation_runtime_status.json", {**summary, "status": "passed"}, sort_keys=True)
        return summary
    except Exception as exc:
        write_json(
            run_dir / "evaluation_runtime_status.json",
            {
                "status": "failed",
                "role": role,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "outer_test_accessed": False,
            },
            sort_keys=True,
        )
        raise
    finally:
        del model, teacher
        shutdown_dataloader_workers(loader)
        for candidate in loaders.values():
            shutdown_dataloader_workers(candidate)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_smoke(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Run the required one-batch/overfit/mini-epoch/15-mask smoke checks."""
    smoke_path = output / "smoke/smoke_checks.json"
    if smoke_path.is_file():
        return json.loads(smoke_path.read_text(encoding="utf-8"))
    build_f2(config, output)
    protocol = load_trajectory_protocol(_path(config["protocol"]["manifest"]))
    loaders, _, _ = _loaders(_path(config["source"]["normalization_root"]), protocol, create_normalization=False)
    mini_loader = _subset_loader(
        loaders["train"],
        list(range(int(config["training"]["batch_size"]))),
        workers=0,
        shuffle=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(int(config["training"]["base_seed"]))
    model, _ = _load_source_checkpoint(config, device, frozen=False)
    teacher, _ = _load_source_checkpoint(config, device, frozen=True)
    lookup = _risk_lookup(output / "artifacts/f2/f2_train_risk.pt")
    topology = load_audited_topology(_path(config["source"]["topology_manifest"]))
    topology_distance = topology.distance.to(device=device, dtype=torch.float32)
    try:
        batch = next(iter(mini_loader))
        labels = _labels(batch, device)
        sample_ids = _batch_ids(batch)
        availability = torch.tensor(MASKS["image_only"], dtype=torch.bool, device=device).expand(labels.numel(), -1)
        model.train()
        with _autocast(device):
            inputs = _inputs(batch, device)
            full_output, missing_output = model.forward_paired(inputs, availability)
            full_loss, _ = baseline_loss(model, full_output, labels)
            official_a0_loss, _ = paired_missing_loss(model, full_output, missing_output, labels, topology_distance)
        risk = _risk_for_batch(lookup, sample_ids, availability, device)
        severe = severe_availability(availability)
        per_ce = F.cross_entropy(missing_output["logits"].float(), labels, reduction="none")
        weights, _ = _weight_for_arm(
            "a3",
            per_ce,
            severe,
            risk,
            availability,
            _fixed_mask_risk(lookup, sample_ids),
            alpha=float(config["f2"]["alpha"]),
            config=config,
        )
        shuffled_control, _ = _weight_for_arm(
            "c2",
            per_ce,
            severe,
            risk,
            availability,
            _fixed_mask_risk(lookup, sample_ids),
            alpha=float(config["f2"]["alpha"]),
            config=config,
            shuffle_generator=torch.Generator(device=device).manual_seed(917),
        )
        with _autocast(device):
            adjusted, _, _ = _adjusted_subset_loss(model, missing_output, labels, weights)
            uniform_adjusted, _, _ = _adjusted_subset_loss(model, missing_output, labels, torch.ones_like(weights))
            consistency = topology_smoothed_consistency_loss(missing_output["logits"], full_output["logits"], topology_distance)
            local_a0_loss = 0.5 * (full_loss + uniform_adjusted) + ABTC_CONSISTENCY_WEIGHT * consistency
        with torch.inference_mode():
            teacher_logits = teacher(_inputs(batch, device))["logits"]
        margin = directional_margin_distillation(
            missing_output["logits"][severe], teacher_logits[severe], labels[severe], weights=weights[severe]
        )
        loss = (
            0.5 * (full_loss + adjusted)
            + ABTC_CONSISTENCY_WEIGHT * consistency
            + float(config["training"]["lambda_margin"]) * margin["loss"]
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        finite_gradients = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        overfit_losses: list[float] = []
        for _ in range(int(config["training"]["one_batch_overfit_steps"])):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                full_output, missing_output = model.forward_paired(_inputs(batch, device), availability)
                full_loss, _ = baseline_loss(model, full_output, labels)
                adjusted, _, _ = _adjusted_subset_loss(model, missing_output, labels, weights)
                consistency = topology_smoothed_consistency_loss(missing_output["logits"], full_output["logits"], topology_distance)
            with torch.inference_mode():
                teacher_logits = teacher(_inputs(batch, device))["logits"]
            margin = directional_margin_distillation(
                missing_output["logits"][severe], teacher_logits[severe], labels[severe], weights=weights[severe]
            )
            overfit = (
                0.5 * (full_loss + adjusted)
                + ABTC_CONSISTENCY_WEIGHT * consistency
                + float(config["training"]["lambda_margin"]) * margin["loss"]
            )
            overfit.backward()
            optimizer.step()
            overfit_losses.append(float(overfit.detach()))
        model.eval()
        with torch.inference_mode():
            tokens = model.encode(_inputs(batch, device))
            mask_finite = {
                name: bool(
                    torch.isfinite(
                        model.forward_tokens(
                            tokens,
                            availability=torch.tensor(bits, dtype=torch.bool, device=device).expand(labels.numel(), -1),
                        )["logits"]
                    ).all()
                )
                for name, bits in MASKS.items()
            }
        one_batch = {
            "forward_backward_loss": float(loss.detach()),
            "finite_gradients": finite_gradients,
            "teacher_gradients_absent": all(parameter.grad is None for parameter in teacher.parameters()),
            "f2_risk_detached": not risk.requires_grad and not weights.requires_grad,
            "all_new_losses_fp32": per_ce.dtype == torch.float32 and margin["loss"].dtype == torch.float32,
            "f2_weight_batch_mean": float(weights.mean()),
            "f2_weight_batch_mean_is_one": math.isclose(float(weights.mean()), 1.0, abs_tol=1e-6),
            "shuffled_weight_distribution_exact": bool(torch.equal(weights[severe].sort().values, shuffled_control[severe].sort().values)),
            "a0_matches_official_loss": bool(torch.allclose(local_a0_loss.float(), official_a0_loss.float(), atol=1e-6, rtol=1e-6)),
            "mask_15_finite": mask_finite,
            "overfit_start": overfit_losses[0],
            "overfit_end": overfit_losses[-1],
            "overfit_reduced": overfit_losses[-1] < overfit_losses[0],
        }
    finally:
        del model, teacher
        shutdown_dataloader_workers(mini_loader)
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    run_dir, status, _ = _train_run(
        config,
        output,
        phase="smoke",
        arm="a3",
        seed=int(config["training"]["base_seed"]),
        fold=None,
        workers=0,
    )
    resume = torch.load(run_dir / "resume_checkpoint.pt", map_location="cpu", weights_only=False)
    restored = TrajectoryBaselineModel(
        ABTC_METHOD,
        **torch.load(run_dir / "final_checkpoint.pt", map_location="cpu", weights_only=False)["model_config"],
    ).to(device)
    restored.load_state_dict(resume["state_dict"], strict=True)
    resumed_optimizer, resumed_scheduler = _optimizer(restored, 1, int(config["training"]["smoke_batches"]))
    resumed_optimizer.load_state_dict(resume["optimizer_state_dict"])
    resumed_scheduler.load_state_dict(resume["scheduler_state_dict"])
    restored.train()
    resumed_optimizer.zero_grad(set_to_none=True)
    with _autocast(device):
        resumed_full, resumed_missing = restored.forward_paired(_inputs(batch, device), availability)
        resumed_loss, _ = paired_missing_loss(restored, resumed_full, resumed_missing, labels, topology_distance)
    resumed_loss.backward()
    resumed_gradient_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in restored.parameters()
    )
    resumed_optimizer.step()
    resumed_scheduler.step()
    result = {
        "status": "passed"
        if one_batch["finite_gradients"]
        and one_batch["teacher_gradients_absent"]
        and one_batch["f2_risk_detached"]
        and one_batch["all_new_losses_fp32"]
        and one_batch["f2_weight_batch_mean_is_one"]
        and one_batch["shuffled_weight_distribution_exact"]
        and one_batch["a0_matches_official_loss"]
        and one_batch["overfit_reduced"]
        and all(one_batch["mask_15_finite"].values())
        and resumed_gradient_finite
        else "failed",
        "one_batch": one_batch,
        "one_epoch_smoke": {
            "arm": "a3",
            "configured_batches": int(config["training"]["smoke_batches"]),
            "status": status["status"],
            "optimizer_steps": status["optimizer_steps"],
        },
        "save_load_resume": {
            "resume_epoch": resume["epoch"],
            "resume_optimizer_steps": resume["optimizer_steps"],
            "strict_model_reload": True,
            "optimizer_state_present": bool(resume["optimizer_state_dict"]),
            "scheduler_state_present": bool(resume["scheduler_state_dict"]),
            "continued_one_step": True,
            "continued_loss": float(resumed_loss.detach()),
            "continued_gradients_finite": resumed_gradient_finite,
        },
        "validation_accessed": False,
        "outer_test_accessed": False,
    }
    del restored, resumed_optimizer, resumed_scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    write_json(smoke_path, result, sort_keys=True)
    if result["status"] != "passed":
        raise RuntimeError(f"SMSL smoke failed: {result}")
    return result


def run_screening_arm(config: Mapping[str, Any], output: Path, *, arm: str, fold: int, workers: int) -> dict[str, Any]:
    build_f2(config, output)
    run_dir, status, heldout_indices = _train_run(
        config,
        output,
        phase="screening",
        arm=arm,
        seed=int(config["training"]["base_seed"]),
        fold=fold,
        workers=workers,
    )
    if status["status"] != "passed":
        raise RuntimeError(f"SMSL screening training failed for {arm}/fold {fold}.")
    return _evaluate_run(
        config,
        output,
        run_dir,
        role="train_heldout",
        heldout_indices=heldout_indices,
        fold=fold,
        workers=workers,
    )


def _rejected_run_incidents(output: Path) -> list[dict[str, Any]]:
    incidents: list[dict[str, Any]] = []
    for path in sorted(output.glob("rejected_runs/*/incident.json")):
        incidents.append({"path": str(path), **json.loads(path.read_text(encoding="utf-8"))})
    return incidents


def _write_training_manifest(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for status_path in sorted(output.glob("screening/fold_*/*/status.json")) + sorted(output.glob("phase2/seed_*/*/status.json")):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        run_dir = status_path.parent
        resolved_path = run_dir / "resolved_config.yaml"
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) if resolved_path.is_file() else {}
        arm = str(status.get("arm", run_dir.name))
        phase = str(status.get("phase", "unknown"))
        fold = status.get("fold")
        seed = int(status.get("seed", config["training"]["base_seed"]))
        gpu = status.get("visible_gpu", config["training"]["gpu_map"].get(arm, ""))
        if phase == "screening":
            log = output / "launch_logs" / f"{arm}_fold{fold}.log"
            default_command = _resource_launch_command(
                config,
                gpu,
                ["tools/run_smsl_r5.py", "--phase", "screening", "--arm", arm, "--fold", str(fold), "--workers", "8"],
            )
        else:
            log = output / "launch_logs" / f"{arm}_seed{seed}.log"
            default_command = _resource_launch_command(
                config,
                gpu,
                ["tools/run_smsl_r5.py", "--phase", "phase2", "--arm", arm, "--seed", str(seed), "--workers", "8"],
            )
        started_at = status.get("started_at") or resolved.get("started_at")
        ended_at = status.get("ended_at")
        if ended_at is None:
            ended_at = datetime.fromtimestamp(status_path.stat().st_mtime, tz=timezone.utc).isoformat()
        rows.append(
            {
                "phase": phase,
                "arm": arm,
                "fold": fold,
                "seed": seed,
                "status": status.get("status"),
                "pid": status.get("pid", resolved.get("pid", "")),
                "gpu": gpu,
                "command": status.get("command", resolved.get("command", default_command)),
                "log": str(log),
                "checkpoint": status.get("checkpoint", ""),
                "checkpoint_sha256": status.get("checkpoint_sha256", ""),
                "started_at": started_at or "unavailable_for_fold0_legacy_runtime_record",
                "ended_at": ended_at,
                "return_code": status.get("return_code", 0 if status.get("status") == "passed" else 1),
                "optimizer_steps": status.get("optimizer_steps", 0),
                "outer_test_accessed": False,
            }
        )
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "runs": rows,
        "run_count": len(rows),
        "monitor_interval_seconds": int(config["training"]["monitor_interval_seconds"]),
        "validation_used_for_training_or_selection": False,
        "outer_test_accessed": False,
        "csi_used": False,
        "channel_input_used": False,
        "rejected_runs": _rejected_run_incidents(output),
    }
    write_json(output / "training_manifest.json", manifest, sort_keys=True)
    return manifest


def _aggregate_run_artifacts(output: Path) -> None:
    sources = {
        "per_mask_metrics.csv": "per_mask_metrics.csv",
        "risk_bucket_metrics.csv": "risk_bucket_metrics.csv",
        "margin_recovery.csv": "margin_recovery.csv",
        "gradient_contribution.csv": "gradient_analysis.csv",
    }
    for source_name, target_name in sources.items():
        rows: list[dict[str, Any]] = []
        paths = sorted(output.glob(f"screening/fold_*/*/{source_name}")) + sorted(output.glob(f"phase2/seed_*/*/{source_name}"))
        for path in paths:
            relative = path.relative_to(output).parts
            phase = relative[0]
            scope = relative[1]
            arm = relative[2]
            fold = int(scope.removeprefix("fold_")) if phase == "screening" else ""
            seed = int(scope.removeprefix("seed_")) if phase == "phase2" else 2026
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rows.append({"phase": phase, "arm": arm, "fold": fold, "seed": seed, **row})
        if rows:
            _write_csv(output / target_name, rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mask_and_scope_means(
    summaries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mask_means: dict[str, Any] = {}
    scope_means: dict[str, Any] = {}
    for arm, values in summaries.items():
        mask_names = [str(row["mask"]) for row in values[0]["masks"]]
        mask_means[arm] = {}
        for name in mask_names:
            rows = [next(row for row in summary["masks"] if row["mask"] == name) for summary in values]
            numeric = (
                "top1",
                "top3",
                "within3",
                "mae",
                "near_error_rate",
                "far_error_rate",
                "far_error_frequency",
                "g1_far_error_rate",
                "g1_rate",
            )
            mask_means[arm][name] = {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in numeric}
        scope_means[arm] = {}
        for scope in values[0]["aggregate"]:
            rows = [summary["aggregate"][scope] for summary in values]
            numeric = (
                "top1_macro",
                "top1_worst",
                "within3_macro",
                "within3_worst",
                "mae_macro",
                "mae_worst",
                "far_error_macro",
                "far_error_worst",
                "far_error_frequency_macro",
                "g1_far_error_macro",
                "g1_far_error_worst",
            )
            scope_means[arm][scope] = {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in numeric}
    return mask_means, scope_means


def _phase2_metric_row(summary: Mapping[str, Any]) -> dict[str, float]:
    scopes = summary["aggregate"]
    masks = {str(row["mask"]): row for row in summary["masks"]}
    return {
        "full_top1": float(masks["full"]["top1"]),
        "single_top1_macro": float(scopes["Single"]["top1_macro"]),
        "single_worst_top1": float(scopes["Single"]["top1_worst"]),
        "severe_top1_macro": float(scopes["Severe"]["top1_macro"]),
        "severe_worst_top1": float(scopes["Severe"]["top1_worst"]),
        "all14_top1_macro": float(scopes["All14"]["top1_macro"]),
        "all14_worst_top1": float(scopes["All14"]["top1_worst"]),
        "single_within3_worst": float(scopes["Single"]["within3_worst"]),
        "severe_within3_worst": float(scopes["Severe"]["within3_worst"]),
        "single_mae_worst": float(scopes["Single"]["mae_worst"]),
        "severe_mae_worst": float(scopes["Severe"]["mae_worst"]),
        "severe_far_error_macro": float(scopes["Severe"]["far_error_macro"]),
        "severe_far_error_worst": float(scopes["Severe"]["far_error_worst"]),
        "severe_far_error_frequency_macro": float(scopes["Severe"]["far_error_frequency_macro"]),
        "severe_g1_far_error_macro": float(scopes["Severe"]["g1_far_error_macro"]),
        "severe_g1_far_error_worst": float(scopes["Severe"]["g1_far_error_worst"]),
        "image_only_top1": float(masks["image_only"]["top1"]),
        "radar_only_top1": float(masks["radar_only"]["top1"]),
        "gps_only_top1": float(masks["gps_only"]["top1"]),
        "lidar_only_top1": float(masks["lidar_only"]["top1"]),
        "missing_lidar_top1": float(masks["missing_lidar"]["top1"]),
        "missing_lidar_radar_top1": float(masks["missing_lidar_radar"]["top1"]),
        "missing_lidar_gps_top1": float(masks["missing_lidar_gps"]["top1"]),
    }


def _risk_diagnostics(rows: Sequence[Mapping[str, str]], phase: str, arms: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in arms:
        selected = [row for row in rows if row["phase"] == phase and row["arm"] == arm and int(row["available_count"]) <= 2]
        arm_result: dict[str, Any] = {}
        for bucket in ("high_risk_top20", "lower_risk_bottom80"):
            bucket_rows = [row for row in selected if row["risk_bucket"] == bucket]
            if not bucket_rows:
                raise ValueError(f"SMSL {phase}/{arm} has no {bucket} risk rows.")
            arm_result[bucket] = {
                "top1": float(np.mean([float(row["top1"]) for row in bucket_rows])),
                "g1_rate": float(np.mean([float(row["g1_rate"]) for row in bucket_rows])),
                "row_count": len(bucket_rows),
            }
        masks = sorted({row["mask"] for row in selected})
        positive = 0
        gaps: list[float] = []
        by_mask: dict[str, Any] = {}
        for name in masks:
            by_mask[name] = {}
            for bucket in ("high_risk_top20", "lower_risk_bottom80"):
                mask_rows = [row for row in selected if row["mask"] == name and row["risk_bucket"] == bucket]
                by_mask[name][bucket] = {
                    "top1": float(np.mean([float(row["top1"]) for row in mask_rows])),
                    "g1_rate": float(np.mean([float(row["g1_rate"]) for row in mask_rows])),
                }
            high = by_mask[name]["high_risk_top20"]["g1_rate"]
            low = by_mask[name]["lower_risk_bottom80"]["g1_rate"]
            gap = high - low
            gaps.append(gap)
            positive += int(gap > 0.0)
        arm_result["by_mask"] = by_mask
        arm_result["within_mask_g1_separation"] = {
            "positive_mask_count": positive,
            "mask_count": len(masks),
            "mean_g1_rate_gap": float(np.mean(gaps)),
        }
        result[arm] = arm_result
    return result


def _baseline_g3_cohort_metrics(record_paths: Mapping[str, Sequence[Path]], arms: Sequence[str]) -> dict[str, Any]:
    severe_names = tuple(name for name, bits in MASKS.items() if 1 <= sum(bits) <= 2)
    loaded = {arm: [torch.load(path, map_location="cpu", weights_only=False)["mask_records"] for path in record_paths[arm]] for arm in arms}
    result: dict[str, Any] = {}
    for arm in arms:
        correct = 0.0
        count = 0
        for seed_index, baseline_run in enumerate(loaded["a0"]):
            candidate_run = loaded[arm][seed_index]
            for name in severe_names:
                cohort = torch.as_tensor(baseline_run[name]["group"]).eq(3)
                candidate_top1 = torch.as_tensor(candidate_run[name]["top1"], dtype=torch.float32)
                correct += float(candidate_top1[cohort].sum())
                count += int(cohort.sum())
        result[arm] = {
            "baseline_a0_g3_cohort_count": count,
            "top1": correct / max(count, 1),
        }
    return result


def _screening_failure_report(output: Path, gate: Mapping[str, Any]) -> None:
    mask = gate["arm_mask_means"]
    scope = gate["arm_scope_means"]
    risk_rows = _read_csv(output / "risk_bucket_metrics.csv")

    def risk_top1(arm: str, bucket: str) -> float:
        values = [
            float(row["top1"])
            for row in risk_rows
            if row["phase"] == "screening" and row["arm"] == arm and row["risk_bucket"] == bucket and int(row["available_count"]) <= 2
        ]
        return float(np.mean(values)) if values else float("nan")

    conclusion = "全部失败，停止R5路线"
    a2_controls = {arm: 100.0 * (scope["a2"]["Severe"]["top1_worst"] - scope[arm]["Severe"]["top1_worst"]) for arm in ("a1", "c1", "c2")}
    lines = [
        "# SMSL R5 最终报告",
        "",
        f"最终结论：**{conclusion}**。Phase 2 与 development validation 未运行。",
        "",
        f"1. F2 是否优于普通 loss hard mining：A2 相对 A1 的 Severe Worst 差值为 `{a2_controls['a1']:+.3f} pp`，未通过完整 gate。",
        f"2. F2 是否优于固定 mask 权重：A2 相对 C1 的 Severe Worst 差值为 `{a2_controls['c1']:+.3f} pp`。",
        f"3. 随机打乱后效果是否消失：A2 相对 C2 的 Severe Worst 差值为 `{a2_controls['c2']:+.3f} pp`。",
        f"4. A3 是否优于普通 margin KD：train-CV gate 结果为 `{gate['a3_beats_c3']}`。",
        f"5. 提升是否集中在高风险样本：A2 high20/lower80 Top-1 为 `{risk_top1('a2', 'high_risk_top20'):.6f}` / `{risk_top1('a2', 'lower_risk_bottom80'):.6f}`。",
        f"6. 同一 mask 内是否有效：A2 在 `{sum(mask['a2'][name]['top1'] > mask['a1'][name]['top1'] for name in NONFULL_MASK_NAMES)}/14` 个 mask 上高于 A1，但整体 gate 未通过。",
        f"7. image-only / missing-LiDAR：A2 相对 A0 为 `{100.0 * (mask['a2']['image_only']['top1'] - mask['a0']['image_only']['top1']):+.3f}` / `{100.0 * (mask['a2']['missing_lidar']['top1'] - mask['a0']['missing_lidar']['top1']):+.3f} pp`。",
        f"8. far error：A2 Severe G1 far-error macro 相对 A0 为 `{100.0 * (scope['a2']['Severe']['g1_far_error_macro'] - scope['a0']['Severe']['g1_far_error_macro']):+.3f} pp`。",
        f"9. Full / Within-3 / MAE：A2 Full Top-1、Severe Within-3 worst、Severe MAE worst 相对 A0 为 `{100.0 * (mask['a2']['full']['top1'] - mask['a0']['full']['top1']):+.3f} pp` / `{100.0 * (scope['a2']['Severe']['within3_worst'] - scope['a0']['Severe']['within3_worst']):+.3f} pp` / `{scope['a2']['Severe']['mae_worst'] - scope['a0']['Severe']['mae_worst']:+.4f}`。",
        "10. 第二创新点证据：预注册 train-only gate 未通过，因此证据不足；不得将 R5 包装为第二创新点。",
        "",
        "未访问 outer test；未使用 CSI/channel；未用 development validation 训练、调参或选择 checkpoint。",
    ]
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output / "final_conclusion.json",
        {"conclusion": conclusion, "screening_gate": gate, "outer_test_accessed": False},
        sort_keys=True,
    )


def aggregate_screening(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, Any]]] = {arm: [] for arm in SMSL_ARMS}
    for fold in range(int(config["screening"]["folds"])):
        for arm in SMSL_ARMS:
            path = output / "screening" / f"fold_{fold}" / arm / "evaluation_summary.json"
            if not path.is_file():
                raise FileNotFoundError(f"SMSL screening result is absent: {path}")
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("role") != "train_heldout" or summary.get("validation_accessed") is not False:
                raise ValueError(f"SMSL screening result is not train-only: {path}")
            summaries[arm].append(summary)
            rows.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "single_worst_top1": summary["single_worst_top1"],
                    "severe_worst_top1": summary["severe_worst_top1"],
                    "single_within3_worst": summary["single_within3_worst"],
                    "severe_within3_worst": summary["severe_within3_worst"],
                    "single_mae_worst": summary["single_mae_worst"],
                    "severe_mae_worst": summary["severe_mae_worst"],
                }
            )
    _write_csv(output / "screening_summary.csv", rows)
    _write_training_manifest(config, output)
    _aggregate_run_artifacts(output)
    means = {
        arm: {
            key: float(np.mean([float(summary[key]) for summary in values]))
            for key in (
                "single_worst_top1",
                "severe_worst_top1",
                "single_within3_worst",
                "severe_within3_worst",
                "single_mae_worst",
                "severe_mae_worst",
            )
        }
        for arm, values in summaries.items()
    }
    mask_means, scope_means = _mask_and_scope_means(summaries)
    baseline = means["a0"]

    def better(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
        return left["single_worst_top1"] > right["single_worst_top1"] and left["severe_worst_top1"] > right["severe_worst_top1"]

    def no_material_degradation(candidate: Mapping[str, float]) -> bool:
        within = float(config["screening"]["gate"]["within3_tolerance"])
        mae = float(config["screening"]["gate"]["mae_tolerance"])
        return (
            candidate["single_within3_worst"] >= baseline["single_within3_worst"] - within
            and candidate["severe_within3_worst"] >= baseline["severe_within3_worst"] - within
            and candidate["single_mae_worst"] <= baseline["single_mae_worst"] + mae
            and candidate["severe_mae_worst"] <= baseline["severe_mae_worst"] + mae
        )

    def improves_baseline(candidate: Mapping[str, float]) -> bool:
        return (
            candidate["single_worst_top1"] > baseline["single_worst_top1"] or candidate["severe_worst_top1"] > baseline["severe_worst_top1"]
        )

    a2_wins = [arm for arm in config["screening"]["gate"]["a2_beats_at_least_two_of"] if better(means["a2"], means[arm])]
    a3_wins = better(means["a3"], means[config["screening"]["gate"]["a3_must_beat"]])
    a2_qualifies = len(a2_wins) >= 2 and improves_baseline(means["a2"]) and no_material_degradation(means["a2"])
    a3_qualifies = a3_wins and improves_baseline(means["a3"]) and no_material_degradation(means["a3"])
    strongest_non_f2 = max(
        ("a1", "c1", "c2", "c3"),
        key=lambda arm: (means[arm]["severe_worst_top1"], means[arm]["single_worst_top1"]),
    )
    gate = {
        "status": "passed" if (a2_qualifies or a3_qualifies) else "failed",
        "validation_accessed": False,
        "outer_test_accessed": False,
        "arm_means": means,
        "arm_mask_means": mask_means,
        "arm_scope_means": scope_means,
        "a2_wins_against": a2_wins,
        "a2_qualifies": a2_qualifies,
        "a3_beats_c3": a3_wins,
        "a3_qualifies": a3_qualifies,
        "improves_a0_primary": {"a2": improves_baseline(means["a2"]), "a3": improves_baseline(means["a3"])},
        "no_material_degradation": {"a2": no_material_degradation(means["a2"]), "a3": no_material_degradation(means["a3"])},
        "strongest_non_f2": strongest_non_f2,
        "phase2_arms": ["a0", "a2", "a3", strongest_non_f2] if (a2_qualifies or a3_qualifies) else [],
        "failure_reason": "" if (a2_qualifies or a3_qualifies) else "screening gate not met; stop R5 before validation/full training",
    }
    write_json(output / "screening_gate.json", gate, sort_keys=True)
    if gate["status"] == "failed":
        write_json(
            output / "failure_manifest.json",
            {
                "route": "SMSL_R5",
                "status": "stopped_after_train_only_screening",
                "rejected_runs": _rejected_run_incidents(output),
                **gate,
            },
            sort_keys=True,
        )
        _write_csv(
            output / "full_seed_summary.csv",
            [{"status": "not_run_screening_gate_failed", "seed_count": 0, "outer_test_accessed": False}],
        )
        _screening_failure_report(output, gate)
    else:
        write_json(output / "phase2/selected_arms.json", gate, sort_keys=True)
    return gate


def _selected_phase2_arms(output: Path) -> list[str]:
    path = output / "phase2/selected_arms.json"
    if not path.is_file():
        raise ValueError("SMSL Phase 2 is blocked until the train-only screening gate passes.")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("status") != "passed" or not selection.get("phase2_arms"):
        raise ValueError("SMSL Phase 2 selection is invalid.")
    return [str(value) for value in selection["phase2_arms"]]


def run_phase2_arm(config: Mapping[str, Any], output: Path, *, arm: str, seed: int, workers: int) -> dict[str, Any]:
    if arm not in _selected_phase2_arms(output):
        raise ValueError(f"SMSL Phase 2 arm is not selected by the screening gate: {arm}")
    run_dir, status, _ = _train_run(
        config,
        output,
        phase="phase2",
        arm=arm,
        seed=seed,
        fold=None,
        workers=workers,
    )
    return {"run_dir": str(run_dir), **status}


def run_final_evaluation(config: Mapping[str, Any], output: Path, *, arm: str, seed: int, workers: int) -> dict[str, Any]:
    if arm not in _selected_phase2_arms(output):
        raise ValueError(f"SMSL final evaluation arm is not selected: {arm}")
    run_dir = _run_directory(output, "phase2", arm, fold=None, seed=seed)
    if not (run_dir / "final_checkpoint.pt").is_file():
        raise FileNotFoundError(f"SMSL final checkpoint is missing: {run_dir}")
    return _evaluate_run(
        config,
        output,
        run_dir,
        role="validation",
        heldout_indices=None,
        fold=None,
        workers=workers,
    )


def _scope_bootstrap(
    record_paths: Sequence[Path],
    names: Sequence[str],
    metric: str,
    *,
    aggregation: str,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Cluster-bootstrap trajectories while preserving equal-mask aggregation."""
    if aggregation not in {"macro", "worst"}:
        raise ValueError(f"Unsupported SMSL bootstrap aggregation: {aggregation}")

    def reduce_masks(values: Sequence[float]) -> float:
        if aggregation == "macro":
            return float(np.mean(values))
        if metric in {"top1", "within3"}:
            return float(np.min(values))
        return float(np.max(values))

    runs = [torch.load(path, map_location="cpu", weights_only=False)["mask_records"] for path in record_paths]
    trajectories = sorted({str(value) for value in runs[0][names[0]]["trajectory"]})
    if not trajectories:
        raise ValueError("SMSL bootstrap evaluation has no trajectories.")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    run_means: list[float] = []
    per_run: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []
    for run in runs:
        current: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in names:
            raw_trajectories = np.asarray(run[name]["trajectory"], dtype=object)
            raw_metric = np.asarray(torch.as_tensor(run[name][metric]).cpu(), dtype=np.float64)
            current[name] = raw_trajectories, raw_metric
        per_run.append(current)
        run_means.append(reduce_masks([float(metric_values.mean()) for _, metric_values in current.values()]))
    for index in range(int(replicates)):
        chosen = rng.choice(trajectories, size=len(trajectories), replace=True)
        seed_estimates = []
        for current in per_run:
            mask_estimates = []
            for raw_trajectories, raw_metric in current.values():
                selected = np.concatenate([np.flatnonzero(raw_trajectories == trajectory) for trajectory in chosen])
                mask_estimates.append(float(raw_metric[selected].mean()))
            seed_estimates.append(reduce_masks(mask_estimates))
        estimates[index] = float(np.mean(seed_estimates))
    return {
        "mean": float(np.mean(run_means)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


def _success_checks(candidate: Mapping[str, float], baseline: Mapping[str, float], thresholds: Mapping[str, Any]) -> dict[str, bool]:
    epsilon = 1e-12
    return {
        "single_worst_gain": candidate["single_worst_top1"] - baseline["single_worst_top1"]
        >= float(thresholds["single_worst_gain"]) - epsilon,
        "severe_worst_gain": candidate["severe_worst_top1"] - baseline["severe_worst_top1"]
        >= float(thresholds["severe_worst_gain"]) - epsilon,
        "severe_macro_gain": candidate["severe_top1_macro"] - baseline["severe_top1_macro"]
        >= float(thresholds["severe_macro_gain"]) - epsilon,
        "all14_worst_gain": candidate["all14_worst_top1"] - baseline["all14_worst_top1"] >= float(thresholds["all14_worst_gain"]) - epsilon,
        "single_within3_preserved": candidate["single_within3_worst"]
        >= baseline["single_within3_worst"] - float(thresholds["within3_max_drop"]) - epsilon,
        "severe_within3_preserved": candidate["severe_within3_worst"]
        >= baseline["severe_within3_worst"] - float(thresholds["within3_max_drop"]) - epsilon,
        "single_mae_preserved": candidate["single_mae_worst"]
        <= baseline["single_mae_worst"] + float(thresholds["mae_max_increase"]) + epsilon,
        "severe_mae_preserved": candidate["severe_mae_worst"]
        <= baseline["severe_mae_worst"] + float(thresholds["mae_max_increase"]) + epsilon,
        "full_top1_preserved": candidate["full_top1"] >= baseline["full_top1"] - float(thresholds["full_top1_max_drop"]) - epsilon,
        "far_error_decreased": candidate["severe_g1_far_error_macro"] < baseline["severe_g1_far_error_macro"]
        if bool(thresholds["far_error_requires_decrease"])
        else True,
    }


def _meaningful_primary_gain(candidate: Mapping[str, float], reference: Mapping[str, float], thresholds: Mapping[str, Any]) -> bool:
    epsilon = 1e-12
    single = candidate["single_worst_top1"] - reference["single_worst_top1"]
    severe = candidate["severe_worst_top1"] - reference["severe_worst_top1"]
    return (
        single >= -epsilon
        and severe >= -epsilon
        and (single >= float(thresholds["single_worst_gain"]) - epsilon or severe >= float(thresholds["severe_worst_gain"]) - epsilon)
    )


def aggregate_phase2(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    arms = _selected_phase2_arms(output)
    seeds = [int(value) for value in config["training"]["seeds"]]
    raw_rows: list[dict[str, Any]] = []
    summaries: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    record_paths: dict[str, list[Path]] = {arm: [] for arm in arms}
    for arm in arms:
        for seed in seeds:
            run_dir = _run_directory(output, "phase2", arm, fold=None, seed=seed)
            summary_path = run_dir / "evaluation_summary.json"
            record_path = run_dir / "evaluation_records.pt"
            if not summary_path.is_file() or not record_path.is_file():
                raise FileNotFoundError(f"SMSL final validation result is absent for {arm}/seed {seed}.")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("role") != "validation" or summary.get("validation_accessed") is not True:
                raise ValueError(f"SMSL final result has the wrong evaluation role: {summary_path}")
            summaries[arm].append(summary)
            record_paths[arm].append(record_path)
            raw_rows.append({"arm": arm, "seed": seed, **_phase2_metric_row(summary)})
    _write_csv(output / "phase2/final_seed_metrics.csv", raw_rows)
    metric_names = tuple(name for name in raw_rows[0] if name not in {"arm", "seed"})
    final_rows = []
    means: dict[str, dict[str, float]] = {}
    for arm, values in summaries.items():
        means[arm] = {}
        row: dict[str, Any] = {"arm": arm, "seed_count": len(values)}
        for metric in metric_names:
            samples = np.asarray([_phase2_metric_row(value)[metric] for value in values], dtype=np.float64)
            row[f"{metric}_mean"] = float(samples.mean())
            row[f"{metric}_std"] = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0
            means[arm][metric] = float(samples.mean())
        final_rows.append(row)
    _write_csv(output / "phase2/final_summary.csv", final_rows)
    _write_csv(output / "full_seed_summary.csv", final_rows)
    _write_training_manifest(config, output)
    _aggregate_run_artifacts(output)
    ci_rows: list[dict[str, Any]] = []
    scope_names = {
        "Three": tuple(name for name, bits in MASKS.items() if sum(bits) == 3),
        "Two": tuple(name for name, bits in MASKS.items() if sum(bits) == 2),
        "Single": tuple(name for name, bits in MASKS.items() if sum(bits) == 1),
        "Severe": tuple(name for name, bits in MASKS.items() if 1 <= sum(bits) <= 2),
        "All14": NONFULL_MASK_NAMES,
    }
    bootstrap_specs = (
        ("Full", "top1", "macro"),
        ("Single", "top1", "macro"),
        ("Single", "top1", "worst"),
        ("Severe", "top1", "macro"),
        ("Severe", "top1", "worst"),
        ("All14", "top1", "macro"),
        ("All14", "top1", "worst"),
        ("Single", "within3", "worst"),
        ("Severe", "within3", "worst"),
        ("Single", "mae", "worst"),
        ("Severe", "mae", "worst"),
    )
    for arm in arms:
        for scope, metric, aggregation in bootstrap_specs:
            ci_rows.append(
                {
                    "arm": arm,
                    "scope": scope,
                    "metric": metric,
                    "aggregation": aggregation,
                    **_scope_bootstrap(
                        record_paths[arm],
                        scope_names[scope],
                        metric,
                        aggregation=aggregation,
                        replicates=int(config["evaluation"]["trajectory_bootstrap_replicates"]),
                        seed=_stable_seed(arm, scope, metric, aggregation),
                    ),
                }
            )
    _write_csv(output / "phase2/trajectory_bootstrap_ci.csv", ci_rows)
    gate = json.loads((output / "screening_gate.json").read_text(encoding="utf-8"))
    control = str(gate["strongest_non_f2"])
    baseline = means["a0"]
    a2 = means["a2"]
    a3 = means["a3"]
    control_mean = means[control]
    thresholds = config["success"]
    success_checks = {arm: _success_checks(value, baseline, thresholds) for arm, value in means.items() if arm != "a0"}
    success = {arm: all(checks.values()) for arm, checks in success_checks.items()}
    mask_means, scope_means = _mask_and_scope_means(summaries)
    risk = _risk_diagnostics(_read_csv(output / "risk_bucket_metrics.csv"), "phase2", arms)
    g3_cohort = _baseline_g3_cohort_metrics(record_paths, arms)
    within_mask = risk["a0"]["within_mask_g1_separation"]
    within_mask_effective = within_mask["mean_g1_rate_gap"] > 0.0 and within_mask["positive_mask_count"] >= math.ceil(
        float(thresholds["within_mask_min_positive_fraction"]) * within_mask["mask_count"]
    )

    def high_risk_concentrated(arm: str) -> bool:
        high_gain = risk[arm]["high_risk_top20"]["top1"] - risk["a0"]["high_risk_top20"]["top1"]
        low_gain = risk[arm]["lower_risk_bottom80"]["top1"] - risk["a0"]["lower_risk_bottom80"]["top1"]
        return high_gain > 0.0 and high_gain > low_gain

    a2_all_screening_controls = set(gate["a2_wins_against"]) >= {"a1", "c1", "c2"}
    a2_attribution = _meaningful_primary_gain(a2, control_mean, thresholds)
    a3_attribution = _meaningful_primary_gain(a3, control_mean, thresholds)
    weighting_supported = (
        success.get("a2", False) and a2_all_screening_controls and a2_attribution and high_risk_concentrated("a2") and within_mask_effective
    )
    interaction_supported = (
        success.get("a3", False)
        and bool(gate["a3_qualifies"])
        and a3_attribution
        and high_risk_concentrated("a3")
        and within_mask_effective
    )
    ordinary_margin_supported = (
        not weighting_supported
        and not interaction_supported
        and control == "c3"
        and success.get("c3", False)
        and not _meaningful_primary_gain(a3, means["c3"], thresholds)
    )
    conclusion = (
        "支持样本级模态充分性学习"
        if weighting_supported or interaction_supported
        else "只支持普通margin distillation，不支持充分性创新"
        if ordinary_margin_supported
        else "全部失败，停止R5路线"
    )
    high_risk_concentration = {arm: high_risk_concentrated(arm) for arm in ("a2", "a3")}
    result = {
        "status": "completed",
        "conclusion": conclusion,
        "screening_gate": gate,
        "phase2_means": means,
        "phase2_success_checks": success_checks,
        "phase2_success": success,
        "weighting_supported": weighting_supported,
        "interaction_supported": interaction_supported,
        "ordinary_margin_supported": ordinary_margin_supported,
        "high_risk_concentration": high_risk_concentration,
        "within_mask_effective": within_mask_effective,
        "risk_diagnostics": risk,
        "baseline_defined_g3_cohort": g3_cohort,
        "arm_mask_means": mask_means,
        "arm_scope_means": scope_means,
        "strongest_non_f2": control,
        "outer_test_accessed": False,
        "csi_used": False,
        "channel_input_used": False,
    }
    write_json(output / "final_conclusion.json", result, sort_keys=True)
    write_json(
        output / "failure_manifest.json",
        {
            "status": "no_runtime_failures" if conclusion != "全部失败，停止R5路线" else "research_stop_condition_triggered",
            "conclusion": conclusion,
            "outer_test_accessed": False,
            "rejected_runs": _rejected_run_incidents(output),
        },
        sort_keys=True,
    )
    screen_scope = gate["arm_scope_means"]
    screen_deltas = {
        arm: 100.0 * (screen_scope["a2"]["Severe"]["top1_worst"] - screen_scope[arm]["Severe"]["top1_worst"]) for arm in ("a1", "c1", "c2")
    }
    high_gain = {arm: 100.0 * (risk[arm]["high_risk_top20"]["top1"] - risk["a0"]["high_risk_top20"]["top1"]) for arm in ("a2", "a3")}
    low_gain = {arm: 100.0 * (risk[arm]["lower_risk_bottom80"]["top1"] - risk["a0"]["lower_risk_bottom80"]["top1"]) for arm in ("a2", "a3")}
    same_mask_high_wins = {
        arm: sum(
            risk[arm]["by_mask"][name]["high_risk_top20"]["top1"] > risk["a0"]["by_mask"][name]["high_risk_top20"]["top1"]
            for name in risk["a0"]["by_mask"]
        )
        for arm in ("a2", "a3")
    }

    def metric_cell(arm: str, metric: str) -> str:
        row = next(value for value in final_rows if value["arm"] == arm)
        return f"{100.0 * float(row[f'{metric}_mean']):.3f} +/- {100.0 * float(row[f'{metric}_std']):.3f}"

    table = [
        "| Arm | Single Worst | Severe Worst | Severe Macro | All-14 Worst | Full |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        table.append(
            "| "
            + " | ".join(
                [
                    arm.upper(),
                    metric_cell(arm, "single_worst_top1"),
                    metric_cell(arm, "severe_worst_top1"),
                    metric_cell(arm, "severe_top1_macro"),
                    metric_cell(arm, "all14_worst_top1"),
                    metric_cell(arm, "full_top1"),
                ]
            )
            + " |"
        )
    missing_lidar_names = ("missing_lidar", "missing_lidar_radar", "missing_lidar_gps")
    missing_lidar_delta = {
        arm: [100.0 * (mask_means[arm][name]["top1"] - mask_means["a0"][name]["top1"]) for name in missing_lidar_names]
        for arm in ("a2", "a3")
    }
    final_c3 = (
        f"final C3 Severe Worst 差值 `{100.0 * (a3['severe_worst_top1'] - means['c3']['severe_worst_top1']):+.3f} pp`"
        if "c3" in means
        else "C3 未被 train-CV 选为最强非 F2 对照，因此完整阶段未重复训练 C3"
    )
    report = [
        "# SMSL R5 最终报告",
        "",
        f"最终结论：**{conclusion}**。",
        "",
        *table,
        "",
        f"1. F2 是否优于普通 loss hard mining：train-CV A2-A1 Severe Worst 为 `{screen_deltas['a1']:+.3f} pp`；全对照条件为 `{a2_all_screening_controls}`。",
        f"2. F2 是否优于固定 mask 权重：train-CV A2-C1 Severe Worst 为 `{screen_deltas['c1']:+.3f} pp`。",
        f"3. 随机打乱后效果是否消失：train-CV A2-C2 Severe Worst 为 `{screen_deltas['c2']:+.3f} pp`。",
        f"4. A3 是否优于普通 margin KD：train-CV A3>C3 为 `{gate['a3_beats_c3']}`；{final_c3}。",
        f"5. 提升是否集中在高风险样本：A2 high20/lower80 增益 `{high_gain['a2']:+.3f}/{low_gain['a2']:+.3f} pp`，A3 为 `{high_gain['a3']:+.3f}/{low_gain['a3']:+.3f} pp`。",
        f"6. 同一 mask 内是否仍有效：F2 high20 的 G1 rate 在 `{within_mask['positive_mask_count']}/{within_mask['mask_count']}` 个 Severe mask 高于 lower80，平均差 `{100.0 * within_mask['mean_g1_rate_gap']:+.3f} pp`；A2/A3 分别在 `{same_mask_high_wins['a2']}/{same_mask_high_wins['a3']}` 个 Severe mask 改善高风险 Top-1。",
        f"7. image-only 与 missing-LiDAR：A2 image-only 相对 A0 `{100.0 * (mask_means['a2']['image_only']['top1'] - mask_means['a0']['image_only']['top1']):+.3f} pp`，三个 missing-LiDAR 差值 `{', '.join(f'{value:+.3f}' for value in missing_lidar_delta['a2'])} pp`；A3 对应 `{100.0 * (mask_means['a3']['image_only']['top1'] - mask_means['a0']['image_only']['top1']):+.3f} pp` 与 `{', '.join(f'{value:+.3f}' for value in missing_lidar_delta['a3'])} pp`。",
        f"8. far error：A2/A3 Severe G1 macro 相对 A0 为 `{100.0 * (a2['severe_g1_far_error_macro'] - baseline['severe_g1_far_error_macro']):+.3f}/{100.0 * (a3['severe_g1_far_error_macro'] - baseline['severe_g1_far_error_macro']):+.3f} pp`。",
        f"9. 退化检查：A2 Full/Severe Within-3 worst/Severe MAE worst 差值为 `{100.0 * (a2['full_top1'] - baseline['full_top1']):+.3f} pp`、`{100.0 * (a2['severe_within3_worst'] - baseline['severe_within3_worst']):+.3f} pp`、`{a2['severe_mae_worst'] - baseline['severe_mae_worst']:+.4f}`；A3 为 `{100.0 * (a3['full_top1'] - baseline['full_top1']):+.3f} pp`、`{100.0 * (a3['severe_within3_worst'] - baseline['severe_within3_worst']):+.3f} pp`、`{a3['severe_mae_worst'] - baseline['severe_mae_worst']:+.4f}`。A0 定义的 Severe G3 固定样本上，A2/A3 Top-1 为 `{g3_cohort['a2']['top1']:.6f}/{g3_cohort['a3']['top1']:.6f}`。",
        f"10. 第二创新点证据：weighting_supported=`{weighting_supported}`，interaction_supported=`{interaction_supported}`，ordinary_margin_supported=`{ordinary_margin_supported}`；结论严格按预注册门槛生成。",
        "",
        "多 seed mean/std 见 `full_seed_summary.csv`，trajectory bootstrap 95% CI 见 `phase2/trajectory_bootstrap_ci.csv`。",
        "训练与筛选只使用 trajectory train split；development validation 仅用于冻结配置的最终评估。未访问 outer test，未使用 CSI/channel/F1。",
    ]
    (output / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("audit", "f2", "smoke", "screening", "aggregate-screening", "phase2", "final-eval", "aggregate-phase2", "all"),
        default="audit",
    )
    parser.add_argument("--arm", choices=SMSL_ARMS)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = _load_config(args.config.resolve())
    resource_contract = _enforce_resource_contract(config, phase=args.phase, workers=args.workers)
    output = _output(config)
    started = time.monotonic()
    _update_process(output, config, args.phase, "running", command_phase=args.phase, started_at=now())
    try:
        audit_payload = _protocol_audit(config, output)
        resolved = dict(config)
        resolved["resolved_at"] = now()
        resolved["audit"] = {
            "checkpoint_sha256": audit_payload["checkpoint_sha256"],
            "split_manifest_sha256": audit_payload["split_manifest_sha256"],
            "topology_manifest_sha256": audit_payload["topology_manifest_sha256"],
        }
        resolved["active_host_runtime"] = resource_contract
        (output / "resolved_config.yaml").write_text(yaml.safe_dump(_json_ready(resolved), sort_keys=False), encoding="utf-8")
        if args.phase == "audit":
            result: Any = audit_payload
        elif args.phase == "f2":
            result = build_f2(config, output)
        elif args.phase == "smoke":
            result = run_smoke(config, output)
        elif args.phase == "screening":
            if args.arm is None or args.fold is None:
                raise ValueError("SMSL screening requires --arm and --fold.")
            result = run_screening_arm(config, output, arm=args.arm, fold=args.fold, workers=args.workers)
        elif args.phase == "aggregate-screening":
            result = aggregate_screening(config, output)
        elif args.phase == "phase2":
            if args.arm is None or args.seed is None:
                raise ValueError("SMSL Phase 2 requires --arm and --seed.")
            result = run_phase2_arm(config, output, arm=args.arm, seed=args.seed, workers=args.workers)
        elif args.phase == "final-eval":
            if args.arm is None or args.seed is None:
                raise ValueError("SMSL final evaluation requires --arm and --seed.")
            result = run_final_evaluation(config, output, arm=args.arm, seed=args.seed, workers=args.workers)
        elif args.phase == "aggregate-phase2":
            result = aggregate_phase2(config, output)
        else:
            result = {"audit": audit_payload, "f2": build_f2(config, output), "smoke": run_smoke(config, output)}
        _update_process(
            output,
            config,
            args.phase,
            "passed",
            elapsed_seconds=time.monotonic() - started,
            result_summary=_json_ready(result),
        )
        print(json.dumps(_json_ready(result), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        _update_process(
            output,
            config,
            args.phase,
            "failed",
            elapsed_seconds=time.monotonic() - started,
            error=repr(exc),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
