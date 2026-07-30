#!/usr/bin/env python3
"""Run the frozen paired Full-to-Missing hard-sample geometry diagnostic."""

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
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import mannwhitneyu, spearmanr

from kd_sensing.baselines.full_pool_bt_scl import load_audited_topology
from kd_sensing.baselines.full_pool_common import atomic_csv, now, sha256_file, write_json
from kd_sensing.baselines.full_pool_candidate12 import MODALITIES
from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel, model_contract
from kd_sensing.diagnostics.paired_geometry import (
    GROUP_NAMES,
    binary_probe_metrics,
    classification_groups,
    cosine_knn,
    decision_decomposition,
    fit_logistic_probe,
    fit_pca_directions,
    linear_cka,
    minimal_interpolation_alpha,
    nonempty_subset_utilities,
    predict_logistic_probe,
    predictive_statistics,
    representation_spectrum,
    ridge_probe_fit,
    ridge_probe_predict,
    scatter_ratio,
    signed_cycle_offset,
    validate_pair_alignment,
    validate_safety_contract,
    validate_train_only_selection,
)
from kd_sensing.diagnostics.prototype_deformation import MASKS, benjamini_hochberg, mask_metadata, normalize
from kd_sensing.engine.data_factory import shutdown_dataloader_workers
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices

from run_mmw_trajectory_baselines import _autocast, _fixed_loader, _inputs, _labels, _loaders


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/full_missing_hard_sample_geometry.yaml"
MASK_NAMES = tuple(MASKS)
MISSING_INDICES = tuple(range(1, len(MASKS)))
TEMPORAL_NAMES = (
    "T0_original",
    "T1_last_only",
    "T2_history_mean",
    "T3_fixed_permutation",
    "T4_drop_earliest",
    "T5_drop_last",
)


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


def _torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
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
    validate_safety_contract(config.get("safety", {}))
    if config["safety"].get("sensing_modalities") != ["image", "lidar", "radar", "gps"]:
        raise ValueError("sensing modality order changed.")
    if int(config["statistics"]["bootstrap_replicates"]) < 1000:
        raise ValueError("formal paired bootstrap requires at least 1000 replicates.")
    if tuple(config["cache"]["temporal_ablations"]) != TEMPORAL_NAMES:
        raise ValueError("temporal ablation order changed.")
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
        raise ValueError("15-mask order changed.")
    return config


def _prepare_output(config: Mapping[str, Any], *, resume: bool) -> Path:
    output = _path(config["output"]["root"])
    marker = output / "process_manifest.json"
    if output.exists() and not resume:
        raise FileExistsError(f"diagnostic output exists and will not be overwritten: {output}")
    if output.exists():
        if not marker.is_file():
            raise ValueError("resume target lacks this diagnostic's process manifest.")
        current = json.loads(marker.read_text(encoding="utf-8"))
        if current.get("diagnostic_id") != config["diagnostic_id"]:
            raise ValueError("resume target belongs to another diagnostic.")
    else:
        output.mkdir(parents=True)
    for name in ("cache", "cache/layers", "artifacts", "artifacts/knn_indices", "artifacts/hardness_probe", "diagnostics", "figures"):
        (output / name).mkdir(parents=True, exist_ok=True)
    return output


def _update_process(output: Path, config: Mapping[str, Any], stage: str, status: str, **extra: Any) -> None:
    path = output / "process_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload.update(
        diagnostic_id=config["diagnostic_id"],
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
        payload.pop("error_type", None)
    write_json(path, payload, sort_keys=True)


def _scan_split_contract(protocol: Mapping[str, Any]) -> dict[str, Any]:
    role_ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    role_groups: dict[str, set[str]] = {"train": set(), "validation": set()}
    counts = defaultdict(int)
    chronology_errors = 0
    history_lengths: set[int] = set()
    future_offsets: set[int] = set()
    for domain in protocol["domains"]:
        for role, key in (("train", "train_split"), ("validation", "validation_split")):
            path = domain.get(key)
            if not path:
                continue
            with Path(path).open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    counts[role] += 1
                    role_ids[role].add(f"{domain['id']}:{row['sample_id']}")
                    role_groups[role].add(row["trajectory_group_id"])
                    history = json.loads(row["history_frame_ids_json"])
                    future = json.loads(row["future_frame_ids_json"])
                    history_lengths.add(len(history))
                    if history and future:
                        future_offsets.add(int(future[0]) - int(history[-1]))
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
        "history_lengths": sorted(history_lengths),
        "future_offsets": sorted(future_offsets),
    }


def _load_model(config: Mapping[str, Any], device: torch.device) -> tuple[TrajectoryBaselineModel, dict[str, Any]]:
    saved = torch.load(_path(config["source"]["checkpoint"]), map_location="cpu", weights_only=False)
    model = TrajectoryBaselineModel(saved["method"], **saved["model_config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, saved


def _hook_audit(model: TrajectoryBaselineModel, device: torch.device) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(44011)
    tokens = {name: torch.randn(4, 5, 64, generator=generator).to(device) for name in MODALITIES}
    availability = torch.tensor(list(MASKS.values())[:4], dtype=torch.bool, device=device)
    with torch.inference_mode():
        fp32 = model.forward_tokens(tokens, availability=availability)["logits"].float()
        captured: dict[str, torch.Tensor] = {}
        handles = [
            model.fusion[2].register_forward_hook(
                lambda _module, _inputs, value: captured.__setitem__("fusion_hidden_1024", value.detach())
            ),
            model.fusion[5].register_forward_hook(
                lambda _module, _inputs, value: captured.__setitem__("fusion_hidden_512", value.detach())
            ),
            model.fusion[7].register_forward_hook(
                lambda _module, _inputs, value: captured.__setitem__("final_sensing_embedding", value.detach())
            ),
        ]
        hooked = model.forward_tokens(tokens, availability=availability)["logits"].float()
        for handle in handles:
            handle.remove()
        with _autocast(device):
            mixed = model.forward_tokens(tokens, availability=availability)["logits"].float()
    return {
        "stable_hook_layers": {key: list(value.shape) for key, value in captured.items()},
        "hook_logits_max_abs": float((hooked - fp32).abs().max()),
        "hook_prediction_equal": bool(hooked.argmax(1).eq(fp32.argmax(1)).all()),
        "autocast_fp32_logits_max_abs": float((mixed - fp32).abs().max()),
        "autocast_fp32_prediction_agreement": float(mixed.argmax(1).eq(fp32.argmax(1)).float().mean()),
        "autocast_finite": bool(torch.isfinite(mixed).all()),
    }


def audit(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    source = config["source"]
    checkpoint = _path(source["checkpoint"])
    protocol_path = _path(config["protocol"]["manifest"])
    split_audit_path = _path(config["protocol"]["audit"])
    topology_path = _path(source["topology_manifest"])
    source_manifest_path = _path(source["sensing_cache_manifest"])
    train_cache_path = _path(source["train_sensing_cache"])
    validation_cache_path = _path(source["validation_sensing_cache"])
    checkpoint_hash = sha256_file(checkpoint)
    split_hash = sha256_file(protocol_path)
    topology_hash = sha256_file(topology_path)
    train_cache_hash = sha256_file(train_cache_path)
    validation_cache_hash = sha256_file(validation_cache_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    split_audit = json.loads(split_audit_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    split_scan = _scan_split_contract(protocol)
    topology = load_audited_topology(topology_path)
    train_cache = torch.load(train_cache_path, map_location="cpu", weights_only=False)
    validation_cache = torch.load(validation_cache_path, map_location="cpu", weights_only=False)
    validate_pair_alignment(train_cache["sample_id"], train_cache["z_raw"])
    validate_pair_alignment(validation_cache["sample_id"], validation_cache["z_raw"])
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model, saved = _load_model(config, device)
    prototype = saved["state_dict"]["prototype_bank.prototypes"].detach().float()
    prototype_hash = _tensor_sha256(prototype)
    hook = _hook_audit(model, device)
    random_feature = normalize(torch.randn(8, 64, generator=torch.Generator().manual_seed(44012)))
    direct = random_feature @ normalize(prototype).t() / float(config["geometry"]["temperature"])
    with torch.inference_mode():
        bank_score = model.prototype_bank(random_feature.to(device)).cpu().float()
    scoring_error = float((direct - bank_score).abs().max())
    cache_flags = (
        all(cache.get("csi_used") is False for cache in (train_cache, validation_cache))
        and all(cache.get("channel_input_used") is False for cache in (train_cache, validation_cache))
        and all(cache.get("outer_test_accessed") is False for cache in (train_cache, validation_cache))
    )
    expected_counts = {
        "train": int(config["protocol"]["expected_train_samples"]),
        "validation": int(config["protocol"]["expected_validation_samples"]),
    }
    expected_trajectories = (
        int(config["protocol"]["expected_train_trajectories"]),
        int(config["protocol"]["expected_validation_trajectories"]),
    )
    checks = {
        "01_checkpoint_path_sha256": checkpoint_hash == source["expected_checkpoint_sha256"] and saved.get("method") == ABTC_METHOD,
        "02_prototype_shape_rank_sha256": tuple(prototype.shape) == (64, 64)
        and int(torch.linalg.matrix_rank(prototype)) == 64
        and prototype_hash == source["expected_prototype_sha256"],
        "03_actual_forward_path": model_contract(model)["channel_input_present"] is False,
        "04_mask_names_bits_order": source_manifest.get("mask_order") == list(MASKS)
        and source_manifest.get("masks") == mask_metadata(),
        "05_full_has_four_modalities": MASKS["full"] == (1, 1, 1, 1),
        "06_same_sample_15_mask_alignment": train_cache["z_raw"].shape[1] == 15
        and validation_cache["z_raw"].shape[1] == 15,
        "07_sample_id_unique_stable": len(train_cache["sample_id"]) == len(set(train_cache["sample_id"]))
        and len(validation_cache["sample_id"]) == len(set(validation_cache["sample_id"]))
        and _string_hash(train_cache["sample_id"]) == source["expected_train_sample_sha256"]
        and _string_hash(validation_cache["sample_id"]) == source["expected_validation_sample_sha256"],
        "08_train_validation_sample_overlap_zero": split_scan["sample_overlap"] == 0,
        "09_train_validation_trajectory_overlap_zero": split_scan["trajectory_overlap"] == 0,
        "10_history_is_t_minus_4_to_t": split_scan["history_lengths"] == [5],
        "11_target_is_t_plus_1": split_scan["chronology_errors"] == 0 and split_scan["future_offsets"] == [1],
        "12_no_csi_channel_f1_access": cache_flags and config["safety"]["f1_used"] is False,
        "13_outer_test_not_constructed": protocol.get("outer_test_accessed") is False
        and split_audit.get("outer_test_accessed") is False,
        "14_topology_manifest": topology_hash == source["expected_topology_manifest_sha256"]
        and topology.num_beams == 64
        and int(topology.distance[0, 63]) == 1,
        "15_exact_prototype_scoring": scoring_error <= 1e-5,
        "16_temperature_fixed_0_1": float(model.prototype_bank.temperature) == 0.1
        and "temperature" not in saved["state_dict"],
        "17_stable_hook_layers": hook["stable_hook_layers"]
        == {
            "fusion_hidden_1024": [4, 1024],
            "fusion_hidden_512": [4, 512],
            "final_sensing_embedding": [4, 64],
        },
        "18_hook_output_unchanged": hook["hook_logits_max_abs"] == 0.0 and hook["hook_prediction_equal"],
        "19_autocast_fp32_measured": hook["autocast_finite"],
        "20_future_power_label_side_only": source_manifest.get("future_beam_power_role")
        == "label_side_evaluation_metric_only",
        "source_cache_hashes": train_cache_hash == source["expected_train_cache_sha256"]
        and validation_cache_hash == source["expected_validation_cache_sha256"],
        "split_hash_counts": split_hash == source["expected_split_sha256"]
        and split_scan["counts"] == expected_counts
        and (split_scan["train_trajectories"], split_scan["validation_trajectories"]) == expected_trajectories,
    }
    hard_failures = [name for name, passed in checks.items() if not bool(passed)]
    payload = {
        "status": "failed" if hard_failures else "passed",
        "checks": checks,
        "hard_failures": hard_failures,
        "checkpoint_sha256": checkpoint_hash,
        "prototype_sha256": prototype_hash,
        "prototype_shape": list(prototype.shape),
        "prototype_rank": int(torch.linalg.matrix_rank(prototype)),
        "split_manifest_sha256": split_hash,
        "topology_manifest_sha256": topology_hash,
        "train_source_cache_sha256": train_cache_hash,
        "validation_source_cache_sha256": validation_cache_hash,
        "split_scan": split_scan,
        "mask_metadata": mask_metadata(),
        "forward_path": "encoders -> [B,5,4,64] positioned/masked tokens -> flatten[1280] -> LayerNorm -> Linear1024/GELU -> Linear512/GELU -> Linear64 -> shared prototype bank",
        "semantic_layer_absence": {
            "per_frame_fusion": "not_present_in_frozen_m4",
            "temporal_module": "not_present_in_frozen_m4",
        },
        "scoring_formula": "normalize(z) @ normalize(P).T / 0.1",
        "hook_audit": hook,
        "scoring_max_abs": scoring_error,
        "csi_used": False,
        "channel_input_used": False,
        "f1_used": False,
        "outer_test_accessed": False,
        "future_beam_power_role": "label_side_evaluation_metric_only",
    }
    write_json(output / "audit.json", payload, sort_keys=True)
    lines = [
        "# 完整输入—缺失输入配对困难样本协议审计",
        "",
        f"审计状态：**{payload['status']}**；硬失败：{hard_failures or '无'}。",
        "",
    ]
    descriptions = (
        f"M4 checkpoint：`{checkpoint}`；SHA256 `{checkpoint_hash}`。",
        f"prototype bank：shape `{list(prototype.shape)}`，rank `{payload['prototype_rank']}`，SHA256 `{prototype_hash}`。",
        f"实际 forward：{payload['forward_path']}。冻结 M4 不含 per-frame fusion 或 temporal module，二者均标记 `not_present_in_frozen_m4`。",
        f"15-mask 顺序：{list(MASKS)}；bits：{list(MASKS.values())}。",
        "Full bits 为 `[1,1,1,1]`。",
        "source cache 逐样本保存完整 15-mask 轴，shape 与顺序校验通过。",
        "train/validation sample_id 均唯一，并与既有 cache manifest 的 sample hash 一致。",
        f"train/validation sample overlap={split_scan['sample_overlap']}。",
        f"train/validation trajectory overlap={split_scan['trajectory_overlap']}。",
        f"history 长度集合={split_scan['history_lengths']}，严格对应 t-4...t。",
        f"future offset 集合={split_scan['future_offsets']}，target 严格为 t+1。",
        "本 runner 的正式输入路径不含 CSI/channel/RadioPrototypeExpert/F1；flags 均为 false。",
        "未构造 outer-test loader/cache；outer_test_accessed=false。",
        f"topology manifest SHA256 `{topology_hash}`，0/63 cycle distance=1。",
        f"精确 scoring 为 `{payload['scoring_formula']}`；直接公式最大误差={scoring_error:.3e}。",
        "temperature 是固定 Python float 0.1，不在可学习 state_dict 中。",
        f"稳定 hook 层与 shape：{hook['stable_hook_layers']}。",
        f"注册 hook 前后 logits 最大差={hook['hook_logits_max_abs']:.3e}，prediction_equal={hook['hook_prediction_equal']}。",
        f"固定合成 token 上 autocast/FP32 logits 最大差={hook['autocast_fp32_logits_max_abs']:.6g}，prediction agreement={hook['autocast_fp32_prediction_agreement']:.3f}；正式 cache 保留生产 autocast 口径，几何统计转 FP32/FP64。",
        "future_beam_power 只进入标签侧 gain/ambiguity 指标，不进入 M4/probe 的部署特征组。",
    )
    lines.extend(f"{index}. {description}" for index, description in enumerate(descriptions, 1))
    (output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    del train_cache, validation_cache, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hard_failures:
        raise RuntimeError(f"protocol audit failed: {hard_failures}")
    return payload


def _metadata_values(batch: Mapping[str, Any], key: str, *, default: str = "unavailable") -> list[str]:
    metadata = batch.get("metadata")
    batch_size = int(torch.as_tensor(batch["target_beam"]).shape[0])
    if not isinstance(metadata, Mapping) or key not in metadata:
        return [default] * batch_size
    return [str(value) for value in metadata[key]]


def _enable_label_side_power(loaders: Mapping[str, Any]) -> None:
    if set(loaders) != {"train", "validation"}:
        raise ValueError(f"formal loader roles must be train/validation only, got {sorted(loaders)}")
    for loader in loaders.values():
        for dataset, _ in leaf_datasets_with_indices(loader.dataset):
            if bool(getattr(dataset, "include_channel_ref", False)) or bool(
                getattr(dataset, "include_channel_history_refs", False)
            ):
                raise ValueError("channel reference loading is forbidden.")
            dataset.include_router_utility_targets = True
            dataset.include_router_corruption_metadata = False


def _target_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return predictive_statistics(logits, labels)["target_margin"]


def _temporal_forward(
    model: TrajectoryBaselineModel,
    stacked: torch.Tensor,
    mask_bits: torch.Tensor,
    labels: torch.Tensor,
    permutation: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    variants: list[tuple[torch.Tensor, torch.Tensor | None]] = [(stacked, None)]
    keep_last = torch.zeros(5, dtype=torch.bool, device=stacked.device)
    keep_last[-1] = True
    variants.append((stacked, keep_last))
    variants.append((stacked.mean(dim=1, keepdim=True).expand_as(stacked), None))
    variants.append((stacked[:, list(permutation)], None))
    drop_first = torch.ones(5, dtype=torch.bool, device=stacked.device)
    drop_first[0] = False
    variants.append((stacked, drop_first))
    drop_last = torch.ones(5, dtype=torch.bool, device=stacked.device)
    drop_last[-1] = False
    variants.append((stacked, drop_last))
    predictions = []
    margins = []
    ranks = []
    for tokens, frame_keep in variants:
        positioned = tokens + model.time_embedding[None, :, None, :] + model.modality_embedding[None, None, :, :]
        if frame_keep is not None:
            positioned = positioned * frame_keep[None, :, None, None].to(positioned)
        masked = positioned[:, None] * mask_bits[None, :, None, :, None].to(positioned)
        with _autocast(stacked.device):
            fused = model.fusion(masked.flatten(2).flatten(0, 1))
            logits = model.prototype_bank(fused).float().view(stacked.shape[0], len(MASKS), 64)
        stats = predictive_statistics(logits, labels)
        predictions.append(stats["prediction"].cpu())
        margins.append(stats["target_margin"].cpu())
        ranks.append(stats["target_rank"].cpu())
    return torch.stack(predictions, dim=2), torch.stack(margins, dim=2), torch.stack(ranks, dim=2)


def _synchronize_temporal_t0_payload(cache: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    stats = predictive_statistics(cache["shared_bank_logits"].float(), cache["target"].long())
    expected = {
        "temporal_prediction": stats["prediction"],
        "temporal_target_margin": stats["target_margin"],
        "temporal_target_rank": stats["target_rank"],
    }
    current = {name: cache[name][:, :, 0] for name in expected}
    exact = {name: torch.equal(current[name], value) for name, value in expected.items()}
    existing = cache.get("temporal_t0_contract")
    if all(exact.values()) and isinstance(existing, Mapping):
        if existing.get("authoritative_source") == "source_shared_bank_logits":
            return dict(existing), False

    prediction_mismatch = int((current["temporal_prediction"] != expected["temporal_prediction"]).sum())
    rank_mismatch = int((current["temporal_target_rank"] != expected["temporal_target_rank"]).sum())
    margin_difference = (current["temporal_target_margin"] - expected["temporal_target_margin"]).abs()
    metadata = {
        "authoritative_source": "source_shared_bank_logits",
        "reason": "T0 is the formal frozen production baseline; mixed-precision replay is reserved for T1-T5 ablations.",
        "prediction_mismatch_count_before": prediction_mismatch,
        "target_rank_mismatch_count_before": rank_mismatch,
        "target_margin_nonzero_count_before": int(margin_difference.ne(0).sum()),
        "target_margin_max_abs_before": float(margin_difference.max()),
        "prediction_mismatch_count_after": 0,
        "target_rank_mismatch_count_after": 0,
        "target_margin_exact_after": True,
    }
    for name, value in expected.items():
        cache[name][:, :, 0].copy_(value.to(cache[name]))
    cache["temporal_t0_contract"] = metadata
    if any(not torch.equal(cache[name][:, :, 0], value) for name, value in expected.items()):
        raise ValueError("failed to bind temporal T0 to the formal production predictions.")
    return metadata, True


def _flush_layer_shard(
    output: Path,
    split: str,
    shard_index: int,
    start: int,
    buffers: Mapping[str, list[Any]],
) -> dict[str, Any]:
    sample_ids = [item for part in buffers["sample_id"] for item in part]
    payload = {
        "schema_version": 1,
        "split": split,
        "start": start,
        "stop": start + len(sample_ids),
        "sample_id": sample_ids,
    }
    for key in ("encoder_tokens", "positioned_tokens", "masked_flat_fusion_input", "fusion_hidden_1024", "fusion_hidden_512"):
        payload[key] = torch.cat(buffers[key]).contiguous().half()
    path = output / f"cache/layers/{split}_{shard_index:04d}.pt"
    _torch_save(path, payload)
    return {
        "path": str(path.relative_to(output)),
        "start": payload["start"],
        "stop": payload["stop"],
        "sample_id_sha256": _string_hash(sample_ids),
        "sha256": sha256_file(path),
        "shapes": {key: list(payload[key].shape) for key in payload if torch.is_tensor(payload[key])},
    }


def _cache_split(
    config: Mapping[str, Any],
    output: Path,
    split: str,
    loader: Any,
    model: TrajectoryBaselineModel,
    device: torch.device,
) -> dict[str, Any]:
    source_path = _path(config["source"][f"{split}_sensing_cache"])
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    expected = int(config["protocol"][f"expected_{split}_samples"])
    if len(source["sample_id"]) != expected:
        raise ValueError(f"{split} source cache sample count changed.")
    mask_bits = torch.tensor(list(MASKS.values()), dtype=torch.bool, device=device)
    shard_size = int(config["cache"]["shard_size"])
    sample_ids: list[str] = []
    trajectory_ids: list[str] = []
    conditions: list[str] = []
    scenarios: list[str] = []
    towns: list[str] = []
    source_ids: list[str] = []
    gps_parts: list[torch.Tensor] = []
    temporal_prediction_parts: list[torch.Tensor] = []
    temporal_margin_parts: list[torch.Tensor] = []
    temporal_rank_parts: list[torch.Tensor] = []
    shard_records: list[dict[str, Any]] = []
    buffers: dict[str, list[Any]] = defaultdict(list)
    buffer_count = 0
    shard_start = 0
    shard_index = 0
    final_feature_max_abs = 0.0
    logits_max_abs = 0.0
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            inputs = _inputs(batch, device)
            labels = _labels(batch, device)
            current_ids = _metadata_values(batch, "stable_sample_id")
            stop = len(sample_ids) + len(current_ids)
            if current_ids != source["sample_id"][len(sample_ids) : stop]:
                raise ValueError(f"{split} loader/source cache sample alignment failed at batch {batch_index}.")
            with _autocast(device):
                tokens = model.encode(inputs)
            stacked = torch.stack([tokens[name] for name in MODALITIES], dim=2)
            positioned = stacked + model.time_embedding[None, :, None, :] + model.modality_embedding[None, None, :, :]
            masked_flat = (positioned[:, None] * mask_bits[None, :, None, :, None].to(positioned)).flatten(2)
            expanded = {
                name: value[:, None].expand(-1, len(MASKS), -1, -1).flatten(0, 1) for name, value in tokens.items()
            }
            availability = mask_bits[None].expand(labels.numel(), -1, -1).flatten(0, 1)
            captured: dict[str, torch.Tensor] = {}
            handles = [
                model.fusion[2].register_forward_hook(
                    lambda _module, _inputs, value: captured.__setitem__("fusion_hidden_1024", value.detach())
                ),
                model.fusion[5].register_forward_hook(
                    lambda _module, _inputs, value: captured.__setitem__("fusion_hidden_512", value.detach())
                ),
            ]
            with _autocast(device):
                result = model.forward_tokens(expanded, availability=availability)
            for handle in handles:
                handle.remove()
            rerun_features = result["fused_features"].float().view(labels.numel(), len(MASKS), 64).cpu()
            rerun_logits = result["logits"].float().view(labels.numel(), len(MASKS), 64).cpu()
            final_feature_max_abs = max(
                final_feature_max_abs,
                float((rerun_features - source["z_raw"][len(sample_ids) : stop]).abs().max()),
            )
            logits_max_abs = max(
                logits_max_abs,
                float((rerun_logits - source["shared_bank_logits"][len(sample_ids) : stop]).abs().max()),
            )
            predictions, margins, ranks = _temporal_forward(
                model,
                stacked,
                mask_bits,
                labels,
                config["cache"]["temporal_permutation"],
            )
            gps = torch.as_tensor(batch["gps"], dtype=torch.float32)
            gps_mean = torch.as_tensor(batch["gps_scaler_mean"], dtype=torch.float32)
            gps_scale = torch.as_tensor(batch["gps_scaler_scale"], dtype=torch.float32)
            gps_raw = gps * gps_scale[:, None, :] + gps_mean[:, None, :]
            sample_ids.extend(current_ids)
            trajectory_ids.extend(_metadata_values(batch, "trajectory_group_id"))
            conditions.extend(_metadata_values(batch, "condition"))
            scenarios.extend(_metadata_values(batch, "scenario"))
            towns.extend(_metadata_values(batch, "town"))
            source_ids.extend(_metadata_values(batch, "source_sample_id"))
            gps_parts.append(gps_raw.cpu())
            temporal_prediction_parts.append(predictions)
            temporal_margin_parts.append(margins)
            temporal_rank_parts.append(ranks)
            buffers["sample_id"].append(current_ids)
            buffers["encoder_tokens"].append(stacked.cpu())
            buffers["positioned_tokens"].append(positioned.cpu())
            buffers["masked_flat_fusion_input"].append(masked_flat.cpu())
            buffers["fusion_hidden_1024"].append(
                captured["fusion_hidden_1024"].view(labels.numel(), len(MASKS), 1024).cpu()
            )
            buffers["fusion_hidden_512"].append(
                captured["fusion_hidden_512"].view(labels.numel(), len(MASKS), 512).cpu()
            )
            buffer_count += len(current_ids)
            if buffer_count >= shard_size or batch_index == len(loader):
                shard_records.append(_flush_layer_shard(output, split, shard_index, shard_start, buffers))
                shard_start += buffer_count
                shard_index += 1
                buffers = defaultdict(list)
                buffer_count = 0
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(
                    json.dumps(
                        {
                            "event": "multilayer_cache",
                            "split": split,
                            "batch": batch_index,
                            "batches": len(loader),
                            "samples": len(sample_ids),
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    if sample_ids != source["sample_id"] or trajectory_ids != source["trajectory_id"]:
        raise ValueError(f"{split} completed cache identity differs from source cache.")
    payload = dict(source)
    payload.update(
        schema_version=2,
        cache_version=config["cache"]["version"],
        source_sensing_cache=str(source_path),
        source_sensing_cache_sha256=sha256_file(source_path),
        condition=conditions,
        scenario=scenarios,
        town=towns,
        source_sample_id=source_ids,
        gps_history_relative_polar=torch.cat(gps_parts).float(),
        gps_feature_definition=["bs_distance", "sin_bearing", "cos_bearing"],
        temporal_ablation_names=list(TEMPORAL_NAMES),
        temporal_prediction=torch.cat(temporal_prediction_parts).long(),
        temporal_target_margin=torch.cat(temporal_margin_parts).float(),
        temporal_target_rank=torch.cat(temporal_rank_parts).long(),
        layer_shards=shard_records,
        layer_semantics=config["layers"],
        per_frame_fusion_present=False,
        temporal_module_present=False,
        csi_used=False,
        channel_input_used=False,
        f1_used=False,
        outer_test_accessed=False,
        future_beam_power_role="label_side_evaluation_metric_only",
    )
    temporal_t0_contract, _ = _synchronize_temporal_t0_payload(payload)
    path = output / f"cache/{split}_multilayer_features.pt"
    _torch_save(path, payload)
    return {
        "split": split,
        "file": str(path),
        "sha256": sha256_file(path),
        "sample_count": len(sample_ids),
        "sample_id_sha256": _string_hash(sample_ids),
        "trajectory_count": len(set(trajectory_ids)),
        "trajectory_ids": sorted(set(trajectory_ids)),
        "source_sensing_cache": str(source_path),
        "source_sensing_cache_sha256": sha256_file(source_path),
        "layer_shards": shard_records,
        "rerun_final_feature_max_abs": final_feature_max_abs,
        "rerun_logits_max_abs": logits_max_abs,
        "gps_shape": list(payload["gps_history_relative_polar"].shape),
        "temporal_prediction_shape": list(payload["temporal_prediction"].shape),
        "temporal_t0_contract": temporal_t0_contract,
    }


def build_cache(config: Mapping[str, Any], output: Path, audit_payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol = json.loads(_path(config["protocol"]["manifest"]).read_text(encoding="utf-8"))
    baseline_root = _path(config["source"]["baseline_root"])
    loaders, _, normalization = _loaders(baseline_root, protocol, create_normalization=False)
    _enable_label_side_power(loaders)
    workers = int(config["runtime"]["dataloader_workers"])
    fixed = {role: _fixed_loader(loaders[role], workers=workers) for role in ("train", "validation")}
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model, _ = _load_model(config, device)
    records: dict[str, Any] = {}
    try:
        for role in ("train", "validation"):
            target = output / f"cache/{role}_multilayer_features.pt"
            if target.is_file():
                existing = torch.load(target, map_location="cpu", weights_only=False)
                if existing.get("cache_version") != config["cache"]["version"]:
                    raise ValueError(f"existing {role} cache version changed.")
                records[role] = {
                    "split": role,
                    "file": str(target),
                    "sha256": sha256_file(target),
                    "sample_count": len(existing["sample_id"]),
                    "sample_id_sha256": _string_hash(existing["sample_id"]),
                    "trajectory_count": len(set(existing["trajectory_id"])),
                    "trajectory_ids": sorted(set(existing["trajectory_id"])),
                    "source_sensing_cache": existing["source_sensing_cache"],
                    "source_sensing_cache_sha256": existing["source_sensing_cache_sha256"],
                    "layer_shards": existing["layer_shards"],
                    "resume_reused": True,
                }
            else:
                records[role] = _cache_split(config, output, role, fixed[role], model, device)
    finally:
        for loader in fixed.values():
            shutdown_dataloader_workers(loader)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest = {
        "schema_version": 1,
        "cache_version": config["cache"]["version"],
        "checkpoint_sha256": audit_payload["checkpoint_sha256"],
        "prototype_sha256": audit_payload["prototype_sha256"],
        "split_manifest_sha256": audit_payload["split_manifest_sha256"],
        "topology_manifest_sha256": audit_payload["topology_manifest_sha256"],
        "source_cache_revalidated": True,
        "normalization_source_split": normalization.get("metadata", {}).get("source_split"),
        "mask_order": list(MASKS),
        "mask_metadata": mask_metadata(),
        "layer_semantics": config["layers"],
        "hidden_dtype": config["cache"]["hidden_dtype"],
        "final_dtype": config["cache"]["final_dtype"],
        "splits": records,
        "csi_used": False,
        "channel_input_used": False,
        "f1_used": False,
        "outer_test_accessed": False,
        "outer_test_cache_created": False,
        "future_beam_power_role": "label_side_evaluation_metric_only",
    }
    write_json(output / "cache/cache_manifest.json", manifest, sort_keys=True)
    return manifest


def _synchronize_existing_temporal_t0(output: Path) -> dict[str, Any]:
    manifest_path = output / "cache/cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contracts: dict[str, Any] = {}
    manifest_changed = False
    for role in ("train", "validation"):
        path = output / f"cache/{role}_multilayer_features.pt"
        cache = torch.load(path, map_location="cpu", weights_only=False)
        contract, cache_changed = _synchronize_temporal_t0_payload(cache)
        if cache_changed:
            _torch_save(path, cache)
        file_hash = sha256_file(path)
        split_record = manifest["splits"][role]
        if split_record.get("sha256") != file_hash or split_record.get("temporal_t0_contract") != contract:
            split_record["sha256"] = file_hash
            split_record["temporal_t0_contract"] = contract
            manifest_changed = True
        contracts[role] = contract
    if manifest.get("temporal_t0_contract") != contracts:
        manifest["temporal_t0_contract"] = contracts
        manifest_changed = True
    if manifest_changed:
        write_json(manifest_path, manifest, sort_keys=True)
    return manifest


def _load_analysis_inputs(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    train = torch.load(output / "cache/train_multilayer_features.pt", map_location="cpu", weights_only=False)
    validation = torch.load(output / "cache/validation_multilayer_features.pt", map_location="cpu", weights_only=False)
    for cache in (train, validation):
        validate_pair_alignment(cache["sample_id"], cache["z_raw"])
        validate_safety_contract(cache)
        if cache.get("mask_metadata") != mask_metadata():
            raise ValueError(f"{cache['split']} mask metadata changed.")
    if set(train["sample_id"]) & set(validation["sample_id"]):
        raise ValueError("analysis cache train/validation sample overlap is non-zero.")
    if set(train["trajectory_id"]) & set(validation["trajectory_id"]):
        raise ValueError("analysis cache train/validation trajectory overlap is non-zero.")
    saved = torch.load(_path(config["source"]["checkpoint"]), map_location="cpu", weights_only=False)
    prototypes = saved["state_dict"]["prototype_bank.prototypes"].detach().float()
    topology = load_audited_topology(_path(config["source"]["topology_manifest"]))
    return {"train": train, "validation": validation, "prototypes": prototypes, "topology": topology}


def _ece(logits: torch.Tensor, labels: torch.Tensor, *, bins: int = 15) -> float:
    probability = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=-1)
    confidence, prediction = probability.max(dim=-1)
    target = torch.as_tensor(labels, dtype=torch.long)
    correct = prediction.eq(target).float()
    result = 0.0
    boundaries = torch.linspace(0.0, 1.0, int(bins) + 1)
    for index in range(int(bins)):
        chosen = confidence.gt(boundaries[index]) & confidence.le(boundaries[index + 1])
        if bool(chosen.any()):
            result += float(chosen.float().mean() * (confidence[chosen].mean() - correct[chosen].mean()).abs())
    return result


def _finite_summary(values: torch.Tensor | np.ndarray, prefix: str = "") -> dict[str, float | int]:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {f"{prefix}count": 0}
    return {
        f"{prefix}count": int(array.size),
        f"{prefix}mean": float(array.mean()),
        f"{prefix}std": float(array.std()),
        f"{prefix}median": float(np.median(array)),
        f"{prefix}q25": float(np.quantile(array, 0.25)),
        f"{prefix}q75": float(np.quantile(array, 0.75)),
    }


def _prepare_cache_geometry(
    cache: Mapping[str, Any],
    prototypes: torch.Tensor,
    distance: torch.Tensor,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    labels = cache["target"].long()
    logits = cache["shared_bank_logits"].float()
    stats = predictive_statistics(logits, labels)
    groups = classification_groups(stats["prediction"][:, 0], stats["prediction"][:, 1:], labels)
    cycle_error = distance[labels[:, None], stats["prediction"]]
    g1 = groups.eq(1)
    g1_near = g1 & cycle_error[:, 1:].le(int(thresholds["near_cycle_distance"]))
    g1_far = g1 & ~g1_near
    g1_rank = g1 & stats["target_rank"][:, 1:].le(3)
    g1_collapse = g1 & stats["target_rank"][:, 1:].gt(int(thresholds["collapse_rank_threshold"]))
    full_margin = stats["target_margin"][:, 0]
    low = full_margin.le(float(thresholds["low_margin_threshold"]))[:, None]
    high = full_margin.ge(float(thresholds["high_margin_threshold"]))[:, None]
    bank = normalize(prototypes)
    full = normalize(cache["z_raw"][:, 0].float())
    missing = normalize(cache["z_raw"][:, 1:].float())
    delta = missing - full[:, None]
    confuser = stats["prediction"][:, 1:].clone()
    target_expanded = labels[:, None].expand_as(confuser)
    correct_confuser = confuser.eq(target_expanded)
    if bool(correct_confuser.any()):
        alternative = logits[:, 1:].clone()
        alternative.scatter_(2, labels[:, None, None].expand(-1, 14, 1), -torch.inf)
        confuser[correct_confuser] = alternative.argmax(dim=2)[correct_confuser]
    difference = bank[labels][:, None] - bank[confuser]
    direction = normalize(difference)
    signed = (delta * direction).sum(dim=2)
    parallel = signed[:, :, None] * direction
    orthogonal = delta - parallel
    energy = delta.square().sum(dim=2)
    geometry = {
        "cosine": (full[:, None] * missing).sum(dim=2).clamp(-1, 1),
        "angular_distance": torch.acos((full[:, None] * missing).sum(dim=2).clamp(-1, 1)),
        "euclidean_distance": delta.norm(dim=2),
        "raw_norm_change": cache["z_raw"][:, 1:].float().norm(dim=2) - cache["z_raw"][:, 0].float().norm(dim=1)[:, None],
        "delta_target": (missing * bank[labels][:, None]).sum(dim=2) - (full * bank[labels]).sum(dim=1)[:, None],
        "delta_confuser": (missing * bank[confuser]).sum(dim=2) - (full[:, None] * bank[confuser]).sum(dim=2),
        "delta_margin": stats["target_margin"][:, 1:] - stats["target_margin"][:, 0, None],
        "delta_rank": stats["target_rank"][:, 1:] - stats["target_rank"][:, 0, None],
        "signed_parallel": signed,
        "absolute_parallel": signed.abs(),
        "orthogonal_norm": orthogonal.norm(dim=2),
        "parallel_energy_ratio": parallel.square().sum(dim=2) / energy.clamp_min(1e-12),
        "delta_direction_cosine": signed / delta.norm(dim=2).clamp_min(1e-12),
        "confuser": confuser,
        "parallel_vector": parallel,
        "orthogonal_vector": orthogonal,
    }
    return {
        "stats": stats,
        "groups": groups,
        "cycle_error": cycle_error,
        "g1_near": g1_near,
        "g1_far": g1_far,
        "g1_rank": g1_rank,
        "g1_collapse": g1_collapse,
        "g1_low_margin": g1 & low,
        "g1_high_margin_break": g1 & high,
        "geometry": geometry,
    }


def _selection_manifest(config: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    train_stats = predictive_statistics(data["train"]["shared_bank_logits"], data["train"]["target"])
    full_margin = train_stats["target_margin"][:, 0].numpy()
    selection = {
        "source_roles": ["train", "split_independent"],
        "validation_leakage_oracle": False,
        "low_margin_threshold": float(np.quantile(full_margin, config["geometry"]["low_margin_quantile"])),
        "high_margin_threshold": float(np.quantile(full_margin, config["geometry"]["high_margin_quantile"])),
        "near_cycle_distance": int(config["geometry"]["near_cycle_distance"]),
        "collapse_rank_threshold": int(config["geometry"]["collapse_rank_threshold"]),
        "pca_components": list(config["geometry"]["pca_components"]),
        "knn_k": list(config["geometry"]["knn_k"]),
        "matrix_sample_cap": int(config["statistics"]["matrix_sample_cap"]),
        "probe_l2": float(config["probe"]["l2"]),
        "probe_epochs": int(config["probe"]["epochs"]),
        "selection_note": "All thresholds and hyperparameters are train-derived or pre-registered; validation is final evaluation only.",
    }
    validate_train_only_selection(selection)
    return selection


def _metric_row(
    cache: Mapping[str, Any],
    prepared: Mapping[str, Any],
    distance: torch.Tensor,
    indices: torch.Tensor,
    mask_index: int,
    dataset: str,
    scope: str,
) -> dict[str, Any]:
    labels = cache["target"][indices]
    logits = cache["shared_bank_logits"][indices, mask_index].float()
    stats = predictive_statistics(logits, labels)
    prediction = stats["prediction"]
    topology_error = distance[labels, prediction]
    powers = cache["future_beam_power"][indices].float()
    selected_power = powers.gather(1, prediction[:, None]).squeeze(1)
    oracle_power = powers.max(dim=1).values.clamp_min(1e-12)
    gain = selected_power / oracle_power
    top = logits.topk(5, dim=1).indices
    return {
        "dataset": dataset,
        "scope": scope,
        "mask": MASK_NAMES[mask_index],
        "mask_id": mask_index,
        "sample_count": int(indices.numel()),
        "top1": float(stats["correct"].float().mean()),
        "top3": float(top[:, :3].eq(labels[:, None]).any(dim=1).float().mean()),
        "top5": float(top.eq(labels[:, None]).any(dim=1).float().mean()),
        "within1": float(topology_error.le(1).float().mean()),
        "within3": float(topology_error.le(3).float().mean()),
        "within5": float(topology_error.le(5).float().mean()),
        "cycle_mae": float(topology_error.float().mean()),
        "normalized_beamforming_gain": float(gain.mean()),
        "beam_loss_db": float((-10.0 * gain.clamp_min(1e-12).log10()).mean()),
        "mean_target_rank": float(stats["target_rank"].float().mean()),
        "mean_target_margin": float(stats["target_margin"].mean()),
        "mean_entropy": float(stats["entropy"].mean()),
        "ece": _ece(logits, labels),
        "nll": float(stats["nll"].mean()),
        "g1_rate": float(prepared["groups"][indices, mask_index - 1].eq(1).float().mean()) if mask_index else 0.0,
    }


def _group_rows(
    cache: Mapping[str, Any],
    prepared: Mapping[str, Any],
    dataset: str,
) -> dict[str, list[dict[str, Any]]]:
    groups = prepared["groups"]
    labels = cache["target"]
    outputs: dict[str, list[dict[str, Any]]] = {"mask": [], "beam": [], "trajectory": [], "domain": []}
    for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
        current = groups[:, mask_index - 1]
        for group_id, group in enumerate(GROUP_NAMES):
            chosen = current.eq(group_id)
            row = {
                "dataset": dataset,
                "mask": mask,
                "group": group,
                "count": int(chosen.sum()),
                "rate": float(chosen.float().mean()),
            }
            if group == "G1":
                denominator = chosen.sum().clamp_min(1)
                row.update(
                    near_fraction=float(prepared["g1_near"][:, mask_index - 1].sum() / denominator),
                    far_fraction=float(prepared["g1_far"][:, mask_index - 1].sum() / denominator),
                    top3_contains_target_fraction=float(prepared["g1_rank"][:, mask_index - 1].sum() / denominator),
                    rank_collapse_fraction=float(prepared["g1_collapse"][:, mask_index - 1].sum() / denominator),
                    low_full_margin_fraction=float(prepared["g1_low_margin"][:, mask_index - 1].sum() / denominator),
                    high_full_margin_break_fraction=float(
                        prepared["g1_high_margin_break"][:, mask_index - 1].sum() / denominator
                    ),
                )
            outputs["mask"].append(row)
        for beam in range(64):
            scope = labels.eq(beam)
            count = int(scope.sum())
            for group_id, group in enumerate(GROUP_NAMES):
                chosen = scope & current.eq(group_id)
                outputs["beam"].append(
                    {
                        "dataset": dataset,
                        "mask": mask,
                        "target_beam": beam,
                        "class_count": count,
                        "group": group,
                        "count": int(chosen.sum()),
                        "class_frequency_normalized_rate": float(chosen.sum() / max(1, count)),
                    }
                )
        for trajectory in sorted(set(cache["trajectory_id"])):
            scope = torch.tensor([value == trajectory for value in cache["trajectory_id"]])
            for group_id, group in enumerate(GROUP_NAMES):
                chosen = scope & current.eq(group_id)
                outputs["trajectory"].append(
                    {
                        "dataset": dataset,
                        "mask": mask,
                        "trajectory_id": trajectory,
                        "group": group,
                        "count": int(chosen.sum()),
                        "rate": float(chosen.sum() / scope.sum().clamp_min(1)),
                    }
                )
        domains = [f"{condition}/{scenario}" for condition, scenario in zip(cache["condition"], cache["scenario"])]
        for domain in sorted(set(domains)):
            scope = torch.tensor([value == domain for value in domains])
            for group_id, group in enumerate(GROUP_NAMES):
                chosen = scope & current.eq(group_id)
                outputs["domain"].append(
                    {
                        "dataset": dataset,
                        "mask": mask,
                        "domain": domain,
                        "weather": domain.split("/", 1)[0],
                        "scene": domain.split("/", 1)[1],
                        "group": group,
                        "count": int(chosen.sum()),
                        "rate": float(chosen.sum() / scope.sum().clamp_min(1)),
                    }
                )
    return outputs


def d0_groups_baseline(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    distance = data["topology"].distance.cpu()
    all_group_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in ("mask", "beam", "trajectory", "domain")}
    metrics: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for dataset in ("train", "validation"):
        cache = data[dataset]
        current = prepared[dataset]
        indices = torch.arange(len(cache["sample_id"]))
        for mask_index in range(15):
            metrics.append(_metric_row(cache, current, distance, indices, mask_index, dataset, "all"))
            if dataset == "validation":
                for trajectory in sorted(set(cache["trajectory_id"])):
                    selected = torch.tensor([value == trajectory for value in cache["trajectory_id"]]).nonzero().flatten()
                    metrics.append(
                        _metric_row(cache, current, distance, selected, mask_index, dataset, f"trajectory:{trajectory}")
                    )
            prediction = current["stats"]["prediction"][:, mask_index]
            target_confusion = torch.bincount(cache["target"] * 64 + prediction, minlength=4096).view(64, 64)
            for target_beam, predicted_beam in target_confusion.nonzero().tolist():
                transitions.append(
                    {
                        "dataset": dataset,
                        "mask": MASK_NAMES[mask_index],
                        "transition_type": "target_to_prediction",
                        "source_beam": target_beam,
                        "destination_beam": predicted_beam,
                        "count": int(target_confusion[target_beam, predicted_beam]),
                    }
                )
            cycle_error = distance[cache["target"], prediction].long()
            for cycle_distance, count in enumerate(torch.bincount(cycle_error, minlength=33).tolist()):
                if count:
                    transitions.append(
                        {
                            "dataset": dataset,
                            "mask": MASK_NAMES[mask_index],
                            "transition_type": "cycle_distance_error_histogram",
                            "cycle_distance": cycle_distance,
                            "count": int(count),
                        }
                    )
            rank_delta = current["stats"]["target_rank"][:, mask_index] - current["stats"]["target_rank"][:, 0]
            for value in rank_delta.unique(sorted=True).tolist():
                transitions.append(
                    {
                        "dataset": dataset,
                        "mask": MASK_NAMES[mask_index],
                        "transition_type": "target_rank_degradation_histogram",
                        "target_rank_delta": int(value),
                        "count": int(rank_delta.eq(value).sum()),
                    }
                )
            if mask_index:
                full_prediction = current["stats"]["prediction"][:, 0]
                missing_prediction = prediction
                encoded = full_prediction * 64 + missing_prediction
                counts = torch.bincount(encoded, minlength=4096).view(64, 64)
                for source_beam, target_beam in counts.nonzero().tolist():
                    transitions.append(
                        {
                            "dataset": dataset,
                            "mask": MASK_NAMES[mask_index],
                            "transition_type": "full_prediction_to_missing_prediction",
                            "source_beam": source_beam,
                            "destination_beam": target_beam,
                            "full_prediction": source_beam,
                            "missing_prediction": target_beam,
                            "count": int(counts[source_beam, target_beam]),
                        }
                    )
                g1 = current["groups"][:, mask_index - 1].eq(1)
                g1_counts = torch.bincount(
                    cache["target"][g1] * 64 + missing_prediction[g1], minlength=4096
                ).view(64, 64)
                for target_beam, predicted_beam in g1_counts.nonzero().tolist():
                    transitions.append(
                        {
                            "dataset": dataset,
                            "mask": MASK_NAMES[mask_index],
                            "transition_type": "g1_target_to_missing_prediction",
                            "source_beam": target_beam,
                            "destination_beam": predicted_beam,
                            "count": int(g1_counts[target_beam, predicted_beam]),
                        }
                    )
        rows = _group_rows(cache, current, dataset)
        for key in all_group_rows:
            all_group_rows[key].extend(rows[key])
    _write_csv(output / "diagnostics/group_counts_by_mask.csv", all_group_rows["mask"])
    _write_csv(output / "diagnostics/group_counts_by_beam.csv", all_group_rows["beam"])
    _write_csv(output / "diagnostics/group_counts_by_trajectory.csv", all_group_rows["trajectory"])
    _write_csv(output / "diagnostics/group_counts_by_domain.csv", all_group_rows["domain"])
    _write_csv(output / "diagnostics/d0_baseline_metrics.csv", metrics)
    _write_csv(output / "diagnostics/d0_prediction_transition.csv", transitions)
    validation_metrics = [row for row in metrics if row["dataset"] == "validation" and row["scope"] == "all"]
    return {
        "metrics": metrics,
        "validation": validation_metrics,
        "group_rows": all_group_rows,
        "full_top1": validation_metrics[0]["top1"],
        "all14_top1": float(np.mean([row["top1"] for row in validation_metrics[1:]])),
        "worst_mask": min(validation_metrics[1:], key=lambda row: row["top1"])["mask"],
        "worst_top1": min(row["top1"] for row in validation_metrics[1:]),
    }


def _aggregate_geometry_scope(
    cache: Mapping[str, Any],
    prepared: Mapping[str, Any],
    mask_index: int,
    chosen: torch.Tensor,
) -> dict[str, Any]:
    geometry = prepared["geometry"]
    result: dict[str, Any] = {"sample_count": int(chosen.sum())}
    metrics = (
        "cosine",
        "angular_distance",
        "euclidean_distance",
        "raw_norm_change",
        "delta_target",
        "delta_confuser",
        "delta_margin",
        "delta_rank",
    )
    for metric in metrics:
        result.update(_finite_summary(geometry[metric][chosen, mask_index - 1], f"{metric}_"))
    result["full_target_rank_mean"] = float(prepared["stats"]["target_rank"][chosen, 0].float().mean())
    result["missing_target_rank_mean"] = float(prepared["stats"]["target_rank"][chosen, mask_index].float().mean())
    result["full_margin_mean"] = float(prepared["stats"]["target_margin"][chosen, 0].mean())
    result["missing_margin_mean"] = float(prepared["stats"]["target_margin"][chosen, mask_index].mean())
    result["full_entropy_mean"] = float(prepared["stats"]["entropy"][chosen, 0].mean())
    result["missing_entropy_mean"] = float(prepared["stats"]["entropy"][chosen, mask_index].mean())
    result["prototype_span_rank"] = 64
    result["prototype_span_note"] = "full_rank_bank_spans_entire_64d_embedding; orthogonal residual is numerically zero"
    return result


def d1_final_geometry(
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    by_group: list[dict[str, Any]] = []
    by_mask: list[dict[str, Any]] = []
    by_beam: list[dict[str, Any]] = []
    for dataset in ("train", "validation"):
        cache = data[dataset]
        current = prepared[dataset]
        for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
            all_chosen = torch.ones(len(cache["sample_id"]), dtype=torch.bool)
            by_mask.append(
                {
                    "dataset": dataset,
                    "mask": mask,
                    "scope_type": "all",
                    "scope_value": "all",
                    **_aggregate_geometry_scope(cache, current, mask_index, all_chosen),
                }
            )
            for group_id, group in enumerate(GROUP_NAMES):
                chosen = current["groups"][:, mask_index - 1].eq(group_id)
                if bool(chosen.any()):
                    by_group.append(
                        {
                            "dataset": dataset,
                            "mask": mask,
                            "group": group,
                            "scope_type": "all",
                            "scope_value": "all",
                            **_aggregate_geometry_scope(cache, current, mask_index, chosen),
                        }
                    )
            for subgroup, chosen in (
                ("G1-near", current["g1_near"][:, mask_index - 1]),
                ("G1-far", current["g1_far"][:, mask_index - 1]),
                ("G1-rank", current["g1_rank"][:, mask_index - 1]),
                ("G1-collapse", current["g1_collapse"][:, mask_index - 1]),
                ("G1-low-margin", current["g1_low_margin"][:, mask_index - 1]),
                ("G1-high-margin-break", current["g1_high_margin_break"][:, mask_index - 1]),
            ):
                if bool(chosen.any()):
                    by_group.append(
                        {
                            "dataset": dataset,
                            "mask": mask,
                            "group": subgroup,
                            "scope_type": "all",
                            "scope_value": "all",
                            **_aggregate_geometry_scope(cache, current, mask_index, chosen),
                        }
                    )
            scope_dimensions = {
                "trajectory": cache["trajectory_id"],
                "domain": [
                    f"{condition}/{scenario}"
                    for condition, scenario in zip(cache["condition"], cache["scenario"])
                ],
                "scene": cache["scenario"],
                "weather": cache["condition"],
            }
            for scope_type, scope_values in scope_dimensions.items():
                for scope_value in sorted(set(scope_values)):
                    scope = torch.tensor([value == scope_value for value in scope_values])
                    for group_id, group in enumerate(GROUP_NAMES):
                        chosen = scope & current["groups"][:, mask_index - 1].eq(group_id)
                        if bool(chosen.any()):
                            by_group.append(
                                {
                                    "dataset": dataset,
                                    "mask": mask,
                                    "group": group,
                                    "scope_type": scope_type,
                                    "scope_value": scope_value,
                                    **_aggregate_geometry_scope(cache, current, mask_index, chosen),
                                }
                            )
            for beam in range(64):
                chosen = cache["target"].eq(beam)
                if bool(chosen.any()):
                    by_beam.append(
                        {
                            "dataset": dataset,
                            "mask": mask,
                            "target_beam": beam,
                            "scope_type": "target_beam",
                            "scope_value": beam,
                            **_aggregate_geometry_scope(cache, current, mask_index, chosen),
                        }
                    )
    _write_csv(output / "diagnostics/d1_final_geometry_by_group.csv", by_group)
    _write_csv(output / "diagnostics/d1_final_geometry_by_mask.csv", by_mask)
    _write_csv(output / "diagnostics/d1_final_geometry_by_beam.csv", by_beam)
    validation_g1 = [
        row
        for row in by_group
        if row["dataset"] == "validation"
        and row["group"] == "G1"
        and row["scope_type"] == "all"
    ]
    return {"by_group": by_group, "by_mask": by_mask, "validation_g1": validation_g1}


def _decision_topology_geometry(
    cache: Mapping[str, Any],
    prepared: Mapping[str, Any],
    prototypes: torch.Tensor,
    labels_by_position: tuple[int, ...],
    mask_index: int,
) -> dict[str, torch.Tensor]:
    target = cache["target"].long()
    prediction = prepared["stats"]["prediction"][:, mask_index]
    offset = signed_cycle_offset(target, prediction, labels_by_position)
    position = torch.empty(len(labels_by_position), dtype=torch.long)
    position[torch.tensor(labels_by_position)] = torch.arange(len(labels_by_position))
    target_position = position[target]
    clockwise = torch.tensor(labels_by_position)[(target_position + 1) % len(labels_by_position)]
    counterclockwise = torch.tensor(labels_by_position)[(target_position - 1) % len(labels_by_position)]
    bank = normalize(prototypes.float())
    full = normalize(cache["z_raw"][:, 0].float())
    missing = normalize(cache["z_raw"][:, mask_index].float())
    delta = missing - full
    clockwise_direction = normalize(bank[clockwise] - bank[target])
    counterclockwise_direction = normalize(bank[counterclockwise] - bank[target])
    clockwise_projection = (delta * clockwise_direction).sum(dim=1)
    counterclockwise_projection = (delta * counterclockwise_direction).sum(dim=1)
    error_direction_projection = torch.where(
        offset.gt(0),
        clockwise_projection,
        torch.where(offset.lt(0), counterclockwise_projection, torch.zeros_like(clockwise_projection)),
    )
    return {
        "signed_cycle_error_steps": offset,
        "clockwise_error": offset.gt(0),
        "counterclockwise_error": offset.lt(0),
        "adjacent_cycle_error": offset.abs().eq(1),
        "clockwise_tangent_projection": clockwise_projection,
        "counterclockwise_tangent_projection": counterclockwise_projection,
        "predicted_cycle_direction_tangent_projection": error_direction_projection,
    }


def _decision_row(
    prepared: Mapping[str, Any],
    decomposition: Mapping[str, torch.Tensor],
    mask_index: int,
    chosen: torch.Tensor,
    **identity: Any,
) -> dict[str, Any]:
    geometry = prepared["geometry"]
    result = {**identity, "sample_count": int(chosen.sum())}
    for metric in (
        "signed_parallel",
        "absolute_parallel",
        "orthogonal_norm",
        "parallel_energy_ratio",
        "delta_direction_cosine",
    ):
        result.update(_finite_summary(geometry[metric][chosen, mask_index - 1], f"{metric}_"))
    result.update(
        _finite_summary(decomposition["identity_absolute_error"][chosen], "fp32_identity_absolute_error_")
    )
    for metric in (
        "signed_cycle_error_steps",
        "clockwise_tangent_projection",
        "counterclockwise_tangent_projection",
        "predicted_cycle_direction_tangent_projection",
    ):
        result.update(_finite_summary(decomposition[metric][chosen], f"{metric}_"))
    result.update(
        clockwise_error_fraction=float(decomposition["clockwise_error"][chosen].float().mean()),
        counterclockwise_error_fraction=float(decomposition["counterclockwise_error"][chosen].float().mean()),
        adjacent_cycle_error_fraction=float(decomposition["adjacent_cycle_error"][chosen].float().mean()),
        cycle_direction_convention="clockwise is the next label in audited ascending phase-coordinate order",
    )
    result.update(
        _finite_summary(
            decomposition["production_identity_absolute_error"][chosen],
            "production_identity_absolute_error_",
        )
    )
    margin_drop = -geometry["delta_margin"][chosen, mask_index - 1]
    signed_evidence = -geometry["signed_parallel"][chosen, mask_index - 1]
    if margin_drop.numel() > 1 and float(margin_drop.std()) > 0 and float(signed_evidence.std()) > 0:
        result["margin_drop_parallel_pearson"] = float(torch.corrcoef(torch.stack((margin_drop, signed_evidence)))[0, 1])
    else:
        result["margin_drop_parallel_pearson"] = 0.0
    return result


def d2_decision_directions(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    max_identity_error = 0.0
    max_production_identity_error = 0.0
    production_identity_error_sum = 0.0
    production_identity_error_count = 0
    for dataset in ("train", "validation"):
        cache = data[dataset]
        current = prepared[dataset]
        for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
            decomposition = decision_decomposition(
                cache["z_raw"][:, 0],
                cache["z_raw"][:, mask_index],
                cache["shared_bank_logits"][:, 0],
                cache["shared_bank_logits"][:, mask_index],
                cache["target"],
                data["prototypes"],
                temperature=float(config["geometry"]["temperature"]),
            )
            decomposition.update(
                _decision_topology_geometry(
                    cache,
                    current,
                    data["prototypes"],
                    data["topology"].labels_by_position,
                    mask_index,
                )
            )
            max_identity_error = max(max_identity_error, float(decomposition["identity_absolute_error"].max()))
            production_error = decomposition["production_identity_absolute_error"]
            max_production_identity_error = max(max_production_identity_error, float(production_error.max()))
            production_identity_error_sum += float(production_error.double().sum())
            production_identity_error_count += production_error.numel()
            for group_id, group in enumerate(GROUP_NAMES):
                chosen = current["groups"][:, mask_index - 1].eq(group_id)
                if bool(chosen.any()):
                    rows.append(
                        _decision_row(
                            current,
                            decomposition,
                            mask_index,
                            chosen,
                            dataset=dataset,
                            mask=mask,
                            group=group,
                            scope="all",
                        )
                    )
            for subgroup, chosen in (
                ("G1-near", current["g1_near"][:, mask_index - 1]),
                ("G1-far", current["g1_far"][:, mask_index - 1]),
                ("G1-low-margin", current["g1_low_margin"][:, mask_index - 1]),
                ("G1-high-margin-break", current["g1_high_margin_break"][:, mask_index - 1]),
            ):
                if bool(chosen.any()):
                    rows.append(
                        _decision_row(
                            current,
                            decomposition,
                            mask_index,
                            chosen,
                            dataset=dataset,
                            mask=mask,
                            group=subgroup,
                            scope="all",
                        )
                    )
            if dataset == "validation":
                for trajectory in sorted(set(cache["trajectory_id"])):
                    trajectory_scope = torch.tensor([value == trajectory for value in cache["trajectory_id"]])
                    for group_id, group in enumerate(("G0", "G1")):
                        chosen = trajectory_scope & current["groups"][:, mask_index - 1].eq(group_id)
                        if bool(chosen.any()):
                            rows.append(
                                _decision_row(
                                    current,
                                    decomposition,
                                    mask_index,
                                    chosen,
                                    dataset=dataset,
                                    mask=mask,
                                    group=group,
                                    scope=f"trajectory:{trajectory}",
                                )
                            )
    _write_csv(output / "diagnostics/d2_decision_direction_decomposition.csv", rows)
    validation_g1 = [
        row for row in rows if row["dataset"] == "validation" and row["scope"] == "all" and row["group"] == "G1"
    ]
    return {
        "rows": rows,
        "validation_g1": validation_g1,
        "max_identity_error": max_identity_error,
        "max_production_identity_error": max_production_identity_error,
        "mean_production_identity_error": production_identity_error_sum / production_identity_error_count,
    }


def _balanced_sample_indices(cache: Mapping[str, Any], cap: int, *, seed: int) -> torch.Tensor:
    trajectories = sorted(set(cache["trajectory_id"]))
    quota = max(1, int(cap) // len(trajectories))
    selected: list[int] = []
    for trajectory in trajectories:
        candidates = [index for index, value in enumerate(cache["trajectory_id"]) if value == trajectory]
        ordered = sorted(
            candidates,
            key=lambda index: hashlib.sha256(f"{seed}:{cache['sample_id'][index]}".encode()).digest(),
        )
        selected.extend(ordered[:quota])
    if len(selected) < int(cap):
        remaining = sorted(
            set(range(len(cache["sample_id"]))) - set(selected),
            key=lambda index: hashlib.sha256(f"{seed}:remainder:{cache['sample_id'][index]}".encode()).digest(),
        )
        selected.extend(remaining[: int(cap) - len(selected)])
    return torch.tensor(sorted(selected[: int(cap)]), dtype=torch.long)


def _stratified_matrix_indices(
    cache: Mapping[str, Any],
    groups: torch.Tensor,
    cap: int,
    *,
    seed: int,
) -> torch.Tensor:
    selected = set(_balanced_sample_indices(cache, cap, seed=seed).tolist())
    trajectories = np.asarray(cache["trajectory_id"])
    for mask_index in range(groups.shape[1]):
        for group_id in (0, 1):
            candidates = groups[:, mask_index].eq(group_id).nonzero().flatten().tolist()
            if not candidates:
                continue
            by_trajectory = sorted(set(trajectories[candidates].tolist()))
            quota = max(1, int(cap) // max(1, len(by_trajectory)))
            scoped: list[int] = []
            for trajectory in by_trajectory:
                current = [index for index in candidates if trajectories[index] == trajectory]
                current.sort(
                    key=lambda index: hashlib.sha256(
                        f"{seed}:{mask_index}:{group_id}:{cache['sample_id'][index]}".encode()
                    ).digest()
                )
                scoped.extend(current[:quota])
            selected.update(scoped[: int(cap)])
    return torch.tensor(sorted(selected), dtype=torch.long)


def _cap_selected_scope(
    chosen: torch.Tensor,
    selected_indices: torch.Tensor,
    cache: Mapping[str, Any],
    cap: int,
    *,
    seed: int,
) -> torch.Tensor:
    positions = chosen.nonzero().flatten().tolist()
    if len(positions) <= int(cap):
        return chosen
    positions.sort(
        key=lambda position: hashlib.sha256(
            f"{seed}:{cache['sample_id'][int(selected_indices[position])]}".encode()
        ).digest()
    )
    result = torch.zeros_like(chosen)
    result[torch.tensor(positions[: int(cap)])] = True
    return result


def _load_layer_rows(
    output: Path,
    cache: Mapping[str, Any],
    layer: str,
    indices: torch.Tensor,
) -> torch.Tensor:
    selected = torch.as_tensor(indices, dtype=torch.long).sort().values
    if layer == "L5":
        return cache["z_raw"][selected].float()
    if layer == "L6":
        return cache["shared_bank_logits"][selected].float()
    if layer not in {"L0", "L1", "L2", "L3", "L4"}:
        raise ValueError(f"unsupported cached layer: {layer}")
    key = {
        "L0": "encoder_tokens",
        "L1": "positioned_tokens",
        "L2": "masked_flat_fusion_input",
        "L3": "fusion_hidden_1024",
        "L4": "fusion_hidden_512",
    }[layer]
    parts: list[torch.Tensor] = []
    for record in cache["layer_shards"]:
        start, stop = int(record["start"]), int(record["stop"])
        chosen = selected[(selected >= start) & (selected < stop)]
        if not chosen.numel():
            continue
        shard = torch.load(output / record["path"], map_location="cpu", weights_only=False)
        values = shard[key][chosen - start].float()
        if layer in {"L0", "L1"}:
            bits = torch.tensor(list(MASKS.values()), dtype=values.dtype)
            values = (values[:, None] * bits[None, :, None, :, None]).flatten(2)
        parts.append(values)
    if not parts:
        raise ValueError(f"no layer rows selected for {cache['split']} {layer}.")
    result = torch.cat(parts)
    if result.shape[0] != selected.numel():
        raise ValueError(f"layer shard selection misaligned for {cache['split']} {layer}.")
    return result


def _project_for_probe(features: torch.Tensor, dimension: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.shape[1] <= int(dimension):
        return values, None
    generator = torch.Generator().manual_seed(int(seed))
    projection = torch.randn(values.shape[1], int(dimension), generator=generator) / math.sqrt(int(dimension))
    return values @ projection, projection


def _neighborhood_summary(features: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    values = torch.as_tensor(features, dtype=torch.float32)
    target = torch.as_tensor(labels, dtype=torch.long)
    if values.shape[0] < 4:
        return {"knn_label_purity": 0.0, "local_density": 0.0}
    k = min(10, values.shape[0] - 2)
    similarity, indices = cosine_knn(values, values, k=k, exclude_self=True)
    return {
        "knn_label_purity": float(target[indices].eq(target[:, None]).float().mean()),
        "local_density": float(similarity.mean()),
    }


def _neighborhood_preservation(full: torch.Tensor, missing: torch.Tensor) -> float:
    if full.shape[0] < 4:
        return 0.0
    k = min(10, full.shape[0] - 2)
    _, full_index = cosine_knn(full, full, k=k, exclude_self=True)
    _, missing_index = cosine_knn(missing, missing, k=k, exclude_self=True)
    values = []
    for left, right in zip(full_index.tolist(), missing_index.tolist()):
        values.append(len(set(left) & set(right)) / k)
    return float(np.mean(values))


def _layer_group_row(
    layer: str,
    layer_name: str,
    dataset: str,
    mask: str,
    group: str,
    full: torch.Tensor,
    missing: torch.Tensor,
    labels: torch.Tensor,
    probe_logits: torch.Tensor | None,
    scope: str = "all",
) -> dict[str, Any]:
    cosine = F.cosine_similarity(full, missing, dim=1)
    euclidean = (missing - full).norm(dim=1)
    relative_norm = (missing.norm(dim=1) - full.norm(dim=1)) / full.norm(dim=1).clamp_min(1e-12)
    full_spectrum = representation_spectrum(full)
    missing_spectrum = representation_spectrum(missing)
    full_scatter = scatter_ratio(full, labels)
    missing_scatter = scatter_ratio(missing, labels)
    neighborhood = _neighborhood_summary(missing, labels)
    return {
        "dataset": dataset,
        "layer": layer,
        "layer_name": layer_name,
        "mask": mask,
        "group": group,
        "scope": scope,
        "sample_count": int(full.shape[0]),
        **_finite_summary(cosine, "paired_cosine_"),
        **_finite_summary(euclidean, "paired_euclidean_"),
        **_finite_summary(relative_norm, "relative_norm_change_"),
        "cka": linear_cka(full, missing),
        "linear_cka": linear_cka(full, missing),
        "centered_kernel_alignment": linear_cka(full, missing),
        "full_effective_rank": full_spectrum["effective_rank"],
        "missing_effective_rank": missing_spectrum["effective_rank"],
        "effective_rank_delta": missing_spectrum["effective_rank"] - full_spectrum["effective_rank"],
        "full_stable_rank": full_spectrum["stable_rank"],
        "missing_stable_rank": missing_spectrum["stable_rank"],
        "full_participation_ratio": full_spectrum["participation_ratio"],
        "missing_participation_ratio": missing_spectrum["participation_ratio"],
        "full_top1_eigenvalue_energy": full_spectrum["top1_energy"],
        "missing_top1_eigenvalue_energy": missing_spectrum["top1_energy"],
        "full_top5_eigenvalue_energy": full_spectrum["top5_energy"],
        "missing_top5_eigenvalue_energy": missing_spectrum["top5_energy"],
        "full_top10_eigenvalue_energy": full_spectrum["top10_energy"],
        "missing_top10_eigenvalue_energy": missing_spectrum["top10_energy"],
        "full_dead_dimension_ratio": full_spectrum["dead_dimension_ratio"],
        "missing_dead_dimension_ratio": missing_spectrum["dead_dimension_ratio"],
        "full_feature_variance_mean": full_spectrum["feature_variance_mean"],
        "missing_feature_variance_mean": missing_spectrum["feature_variance_mean"],
        "full_feature_variance_std": full_spectrum["feature_variance_std"],
        "missing_feature_variance_std": missing_spectrum["feature_variance_std"],
        "full_mean_pairwise_cosine": full_spectrum["mean_pairwise_cosine"],
        "missing_mean_pairwise_cosine": missing_spectrum["mean_pairwise_cosine"],
        "full_within_class_scatter": full_scatter["within_class_scatter"],
        "missing_within_class_scatter": missing_scatter["within_class_scatter"],
        "full_between_class_scatter": full_scatter["between_class_scatter"],
        "missing_between_class_scatter": missing_scatter["between_class_scatter"],
        "full_centroid_separation": full_scatter["centroid_separation"],
        "missing_centroid_separation": missing_scatter["centroid_separation"],
        "full_fisher_ratio": full_scatter["fisher_ratio"],
        "missing_fisher_ratio": missing_scatter["fisher_ratio"],
        "fisher_ratio_delta": missing_scatter["fisher_ratio"] - full_scatter["fisher_ratio"],
        "knn_label_purity": neighborhood["knn_label_purity"],
        "local_density": neighborhood["local_density"],
        "full_missing_neighborhood_preservation": _neighborhood_preservation(full, missing),
        "target_linear_probe_accuracy": (
            float(probe_logits.argmax(dim=1).eq(labels).float().mean()) if probe_logits is not None else ""
        ),
        "matrix_metric_scope": "pre_registered_trajectory_balanced_cap",
        "_full_covariance_eigenvalues": full_spectrum["eigenvalues"],
        "_missing_covariance_eigenvalues": missing_spectrum["eigenvalues"],
        "_feature_dimension": int(full.shape[1]),
    }


def d3_layerwise_geometry(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    cap = int(config["statistics"]["matrix_sample_cap"])
    seed = int(config["statistics"]["sample_selection_seed"])
    selected = {
        role: _stratified_matrix_indices(
            data[role],
            prepared[role]["groups"],
            min(cap, len(data[role]["sample_id"])),
            seed=seed + offset,
        )
        for offset, role in enumerate(("train", "validation"))
    }
    write_json(
        output / "artifacts/layer_matrix_sample_selection.json",
        {
            "source_roles": ["train", "split_independent"],
            "validation_leakage_oracle": False,
            "cap": cap,
            "loaded_union_sample_count": {role: int(values.numel()) for role, values in selected.items()},
            "train_sample_id_sha256": _string_hash([data["train"]["sample_id"][i] for i in selected["train"]]),
            "validation_sample_id_sha256": _string_hash(
                [data["validation"]["sample_id"][i] for i in selected["validation"]]
            ),
            "selection_rule": "fixed SHA256 ordering with equal trajectory quota",
        },
        sort_keys=True,
    )
    layer_names = {
        "L0": "available_encoder_tokens",
        "L1": "available_positioned_tokens",
        "L2": "masked_flat_fusion_input",
        "L3": "fusion_hidden_1024",
        "L4": "fusion_hidden_512",
        "L5": "final_sensing_embedding",
        "L6": "prototype_logits",
    }
    rows: list[dict[str, Any]] = []
    probe_states: dict[str, Any] = {}
    for layer_index, (layer, layer_name) in enumerate(layer_names.items()):
        train_values = _load_layer_rows(output, data["train"], layer, selected["train"])
        validation_values = _load_layer_rows(output, data["validation"], layer, selected["validation"])
        train_labels = data["train"]["target"][selected["train"]]
        validation_labels = data["validation"]["target"][selected["validation"]]
        validation_trajectory = np.asarray(data["validation"]["trajectory_id"])[selected["validation"].numpy()]
        train_groups = prepared["train"]["groups"][selected["train"]]
        validation_groups = prepared["validation"]["groups"][selected["validation"]]
        for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
            projected_train, projection = _project_for_probe(
                train_values[:, mask_index],
                int(config["statistics"]["probe_projection_dimension"]),
                seed=seed + 1000 * layer_index + mask_index,
            )
            state = ridge_probe_fit(projected_train, train_labels, l2=float(config["probe"]["l2"]))
            projected_validation = (
                validation_values[:, mask_index]
                if projection is None
                else validation_values[:, mask_index] @ projection
            )
            validation_probe_logits = ridge_probe_predict(projected_validation, state)
            probe_states[f"{layer}:{mask}"] = {
                "projection": projection,
                "state": state,
                "source_roles": ["train"],
                "validation_used_for_selection": False,
            }
            for dataset, values, labels, groups, probe_logits in (
                ("train", train_values, train_labels, train_groups, None),
                ("validation", validation_values, validation_labels, validation_groups, validation_probe_logits),
            ):
                for group_id, group in enumerate(("G0", "G1")):
                    chosen = groups[:, mask_index - 1].eq(group_id)
                    chosen = _cap_selected_scope(
                        chosen,
                        selected[dataset],
                        data[dataset],
                        cap,
                        seed=seed + 10000 * layer_index + 100 * mask_index + group_id,
                    )
                    if int(chosen.sum()) < 4:
                        continue
                    rows.append(
                        _layer_group_row(
                            layer,
                            layer_name,
                            dataset,
                            mask,
                            group,
                            values[chosen, 0],
                            values[chosen, mask_index],
                            labels[chosen],
                            probe_logits[chosen] if probe_logits is not None else None,
                        )
                    )
                    if dataset == "validation":
                        for trajectory in sorted(set(validation_trajectory.tolist())):
                            trajectory_chosen = chosen & torch.from_numpy(validation_trajectory == trajectory)
                            if int(trajectory_chosen.sum()) < 4:
                                continue
                            rows.append(
                                _layer_group_row(
                                    layer,
                                    layer_name,
                                    dataset,
                                    mask,
                                    group,
                                    values[trajectory_chosen, 0],
                                    values[trajectory_chosen, mask_index],
                                    labels[trajectory_chosen],
                                    probe_logits[trajectory_chosen] if probe_logits is not None else None,
                                    scope=f"trajectory:{trajectory}",
                                )
                            )
        del train_values, validation_values
    _torch_save(output / "artifacts/layer_target_probe_states.pt", probe_states)
    first_break_rows: list[dict[str, Any]] = []
    threshold = float(config["success"]["layer_break_standardized_min"])
    for mask in MASK_NAMES[1:]:
        layer_scores: list[tuple[str, float, float, float]] = []
        for layer in layer_names:
            g0 = next(
                row
                for row in rows
                if row["dataset"] == "train" and row["mask"] == mask and row["layer"] == layer and row["group"] == "G0"
                and row["scope"] == "all"
            )
            g1 = next(
                (
                row
                for row in rows
                if row["dataset"] == "train" and row["mask"] == mask and row["layer"] == layer and row["group"] == "G1"
                and row["scope"] == "all"
                ),
                None,
            )
            if g1 is None:
                layer_scores = []
                break
            pooled = math.sqrt(
                0.5 * (float(g0["paired_euclidean_std"]) ** 2 + float(g1["paired_euclidean_std"]) ** 2)
            )
            standardized = (
                float(g1["paired_euclidean_mean"]) - float(g0["paired_euclidean_mean"])
            ) / max(pooled, 1e-12)
            cosine_gap = float(g0["paired_cosine_mean"]) - float(g1["paired_cosine_mean"])
            neighborhood_gap = float(g0["full_missing_neighborhood_preservation"]) - float(
                g1["full_missing_neighborhood_preservation"]
            )
            layer_scores.append((layer, standardized, cosine_gap, neighborhood_gap))
        if not layer_scores:
            first_break_rows.append(
                {
                    "mask": mask,
                    "selection_dataset": "train",
                    "first_break_layer": "not_estimable_no_train_g1",
                    "most_severe_break_layer": "not_estimable_no_train_g1",
                    "most_severe_standardized_break": 0.0,
                    "most_severe_cosine_gap": 0.0,
                    "most_severe_neighborhood_gap": 0.0,
                    "threshold": threshold,
                    "per_frame_fusion": "not_present_in_frozen_m4",
                    "temporal_module": "not_present_in_frozen_m4",
                    "validation_used_for_selection": False,
                }
            )
            continue
        qualifying = [item for item in layer_scores if item[1] >= threshold]
        first = qualifying[0][0] if qualifying else "none"
        severe = max(layer_scores, key=lambda item: item[1])
        first_break_rows.append(
            {
                "mask": mask,
                "selection_dataset": "train",
                "first_break_layer": first,
                "most_severe_break_layer": severe[0],
                "most_severe_standardized_break": severe[1],
                "most_severe_cosine_gap": severe[2],
                "most_severe_neighborhood_gap": severe[3],
                "threshold": threshold,
                "per_frame_fusion": "not_present_in_frozen_m4",
                "temporal_module": "not_present_in_frozen_m4",
                "validation_used_for_selection": False,
            }
        )
    _write_csv(
        output / "diagnostics/d3_layerwise_geometry.csv",
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows],
    )
    _write_csv(output / "diagnostics/d3_first_break_layer.csv", first_break_rows)
    return {"rows": rows, "first_break": first_break_rows, "sample_indices": selected}


def d4_representation_collapse(output: Path, d3: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in d3["rows"]:
        rows.append(
            {
                "dataset": row["dataset"],
                "layer": row["layer"],
                "layer_name": row["layer_name"],
                "mask": row["mask"],
                "group": row["group"],
                "scope": row["scope"],
                "sample_count": row["sample_count"],
                "full_effective_rank": row["full_effective_rank"],
                "missing_effective_rank": row["missing_effective_rank"],
                "effective_rank_delta": row["effective_rank_delta"],
                "full_stable_rank": row["full_stable_rank"],
                "missing_stable_rank": row["missing_stable_rank"],
                "full_participation_ratio": row["full_participation_ratio"],
                "missing_participation_ratio": row["missing_participation_ratio"],
                "full_top1_eigenvalue_energy": row["full_top1_eigenvalue_energy"],
                "missing_top1_eigenvalue_energy": row["missing_top1_eigenvalue_energy"],
                "full_top5_eigenvalue_energy": row["full_top5_eigenvalue_energy"],
                "missing_top5_eigenvalue_energy": row["missing_top5_eigenvalue_energy"],
                "full_top10_eigenvalue_energy": row["full_top10_eigenvalue_energy"],
                "missing_top10_eigenvalue_energy": row["missing_top10_eigenvalue_energy"],
                "full_dead_dimension_ratio": row["full_dead_dimension_ratio"],
                "missing_dead_dimension_ratio": row["missing_dead_dimension_ratio"],
                "full_feature_variance_mean": row["full_feature_variance_mean"],
                "missing_feature_variance_mean": row["missing_feature_variance_mean"],
                "full_feature_variance_std": row["full_feature_variance_std"],
                "missing_feature_variance_std": row["missing_feature_variance_std"],
                "full_mean_pairwise_cosine": row["full_mean_pairwise_cosine"],
                "missing_mean_pairwise_cosine": row["missing_mean_pairwise_cosine"],
                "full_within_class_scatter": row["full_within_class_scatter"],
                "missing_within_class_scatter": row["missing_within_class_scatter"],
                "full_between_class_scatter": row["full_between_class_scatter"],
                "missing_between_class_scatter": row["missing_between_class_scatter"],
                "full_centroid_separation": row["full_centroid_separation"],
                "missing_centroid_separation": row["missing_centroid_separation"],
                "full_fisher_ratio": row["full_fisher_ratio"],
                "missing_fisher_ratio": row["missing_fisher_ratio"],
                "fisher_ratio_delta": row["fisher_ratio_delta"],
                "mean_pairwise_cosine_proxy": row["missing_mean_pairwise_cosine"],
                "full_covariance_eigenvalues_json": json.dumps(
                    row["_full_covariance_eigenvalues"].tolist(), separators=(",", ":")
                ),
                "missing_covariance_eigenvalues_json": json.dumps(
                    row["_missing_covariance_eigenvalues"].tolist(), separators=(",", ":")
                ),
                "implicit_zero_eigenvalue_count": max(
                    0, int(row["_feature_dimension"]) - int(row["_missing_covariance_eigenvalues"].numel())
                ),
                "neural_collapse_status": "not_reported; stratified scopes do not reliably support every class",
                "g0_resampling_control": "independent pre-registered equal-cap G0/G1 scopes",
                "beam_trajectory_matching_control": "reported in negative_controls.csv",
                "dimension_permutation_control": "covered by stronger orthogonal-rotation rank control",
                "matrix_metric_scope": row["matrix_metric_scope"],
            }
        )
    _write_csv(output / "diagnostics/d4_representation_collapse.csv", rows)
    validation_final = [
        row
        for row in rows
        if row["dataset"] == "validation"
        and row["layer"] == "L5"
        and row["group"] in {"G0", "G1"}
        and row["scope"] == "all"
    ]
    return {"rows": rows, "validation_final": validation_final}


def _utility_arrays(cache: Mapping[str, Any], prepared: Mapping[str, Any], distance: torch.Tensor) -> dict[str, torch.Tensor]:
    stats = prepared["stats"]
    prediction = stats["prediction"]
    powers = cache["future_beam_power"].float()
    oracle = powers.max(dim=1).values.clamp_min(1e-12)
    gain = torch.stack(
        [powers.gather(1, prediction[:, index, None]).squeeze(1) / oracle for index in range(len(MASKS))],
        dim=1,
    )
    return {
        "margin": stats["target_margin"].float(),
        "top1": stats["correct"].float(),
        "rank": -stats["target_rank"].float(),
        "topology": -distance[cache["target"][:, None], prediction].float(),
        "normalized_gain": gain,
    }


def d5_modality_utility(
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    mask_bits = torch.tensor(list(MASKS.values()), dtype=torch.bool)
    marginal_contract, interaction_contract = nonempty_subset_utilities(torch.zeros(1, len(MASKS)), mask_bits)
    marginal_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    features: dict[str, Any] = {}
    pair_names = list(combinations(range(4), 2))
    for dataset in ("train", "validation"):
        cache = data[dataset]
        arrays = _utility_arrays(cache, prepared[dataset], data["topology"].distance.cpu())
        per_modality: dict[str, list[list[torch.Tensor]]] = {
            metric: [[] for _ in range(4)] for metric in arrays
        }
        per_interaction: dict[str, list[list[torch.Tensor]]] = {
            metric: [[] for _ in pair_names] for metric in arrays
        }
        for relation in marginal_contract:
            base = int(relation["base"])
            union = int(relation["union"])
            modality = int(relation["modality"])
            base_group = prepared[dataset]["groups"][:, base - 1]
            repaired = base_group.eq(1) & prepared[dataset]["stats"]["correct"][:, union]
            harmed = base_group.eq(0) & ~prepared[dataset]["stats"]["correct"][:, union]
            for metric, values in arrays.items():
                delta = values[:, union] - values[:, base]
                per_modality[metric][modality].append(delta)
                marginal_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "modality": MODALITIES[modality],
                        "base_mask": MASK_NAMES[base],
                        "union_mask": MASK_NAMES[union],
                        **_finite_summary(delta, "utility_"),
                        "positive_fraction": float(delta.gt(0).float().mean()),
                        "negative_fraction": float(delta.lt(0).float().mean()),
                        "g1_repair_rate": float(repaired.sum() / base_group.eq(1).sum().clamp_min(1)),
                        "g0_harm_rate": float(harmed.sum() / base_group.eq(0).sum().clamp_min(1)),
                        "empty_subset_used": False,
                    }
                )
        for relation in interaction_contract:
            pair = (int(relation["modality_left"]), int(relation["modality_right"]))
            pair_index = pair_names.index(pair)
            for metric, values in arrays.items():
                interaction = (
                    values[:, relation["both"]]
                    - values[:, relation["left"]]
                    - values[:, relation["right"]]
                    + values[:, relation["base"]]
                )
                per_interaction[metric][pair_index].append(interaction)
                interaction_rows.append(
                    {
                        "dataset": dataset,
                        "metric": metric,
                        "modality_left": MODALITIES[pair[0]],
                        "modality_right": MODALITIES[pair[1]],
                        "base_mask": MASK_NAMES[relation["base"]],
                        **_finite_summary(interaction, "interaction_"),
                        "positive_fraction": float(interaction.gt(0).float().mean()),
                        "negative_fraction": float(interaction.lt(0).float().mean()),
                        "empty_subset_used": False,
                    }
                )
        modality_margin = torch.stack(
            [torch.stack(per_modality["margin"][index], dim=1).mean(dim=1) for index in range(4)], dim=1
        )
        modality_negative = torch.stack(
            [torch.stack(per_modality["margin"][index], dim=1).lt(0).float().mean(dim=1) for index in range(4)],
            dim=1,
        )
        interaction_margin = torch.stack(
            [torch.stack(per_interaction["margin"][index], dim=1).mean(dim=1) for index in range(len(pair_names))],
            dim=1,
        )
        features[dataset] = {
            "modality_margin_utility": modality_margin,
            "modality_negative_fraction": modality_negative,
            "pairwise_margin_interaction": interaction_margin,
            "pair_names": [f"{MODALITIES[left]}+{MODALITIES[right]}" for left, right in pair_names],
        }
        for index, sample_id in enumerate(cache["sample_id"]):
            row: dict[str, Any] = {"dataset": dataset, "sample_id": sample_id}
            for modality_index, modality in enumerate(MODALITIES):
                row[f"{modality}_mean_margin_utility"] = float(modality_margin[index, modality_index])
                row[f"{modality}_negative_utility_fraction"] = float(modality_negative[index, modality_index])
            for pair_index, name in enumerate(features[dataset]["pair_names"]):
                row[f"{name}_mean_margin_interaction"] = float(interaction_margin[index, pair_index])
            row["exact_shapley_status"] = "not_formal_empty_subset_absent"
            sample_rows.append(row)
    _write_csv(output / "diagnostics/d5_modality_marginal_utility.csv", marginal_rows)
    _write_csv(output / "diagnostics/d5_pairwise_interactions.csv", interaction_rows)
    _write_csv(output / "diagnostics/d5_sample_utility_features.csv", sample_rows)
    return {
        "marginal_rows": marginal_rows,
        "interaction_rows": interaction_rows,
        "features": features,
        "marginal_contract_count": len(marginal_contract),
        "interaction_contract_count": len(interaction_contract),
        "empty_subset_used": False,
    }


def _motion_attributes(gps: torch.Tensor) -> dict[str, np.ndarray]:
    values = torch.as_tensor(gps, dtype=torch.float64).numpy()
    radius = values[:, :, 0]
    bearing = np.unwrap(np.arctan2(values[:, :, 1], values[:, :, 2]), axis=1)
    xy = np.stack((radius * np.cos(bearing), radius * np.sin(bearing)), axis=2)
    velocity = np.diff(xy, axis=1)
    speed = np.linalg.norm(velocity, axis=2)
    acceleration = np.diff(velocity, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=2)
    angular_velocity = np.diff(bearing, axis=1)
    angular_acceleration = np.diff(angular_velocity, axis=1)
    cross = np.abs(velocity[:, 1:, 0] * acceleration[:, :, 1] - velocity[:, 1:, 1] * acceleration[:, :, 0])
    curvature = cross / np.maximum(np.linalg.norm(velocity[:, 1:], axis=2) ** 3, 1e-9)
    heading = np.unwrap(np.arctan2(velocity[:, :, 1], velocity[:, :, 0]), axis=1)
    return {
        "bs_distance_last": radius[:, -1],
        "bearing_last": bearing[:, -1],
        "speed_mean": speed.mean(axis=1),
        "speed_last": speed[:, -1],
        "acceleration_mean": acceleration_norm.mean(axis=1),
        "angular_velocity_mean": np.abs(angular_velocity).mean(axis=1),
        "angular_velocity_last": np.abs(angular_velocity[:, -1]),
        "angular_acceleration_mean": np.abs(angular_acceleration).mean(axis=1),
        "trajectory_curvature_mean": curvature.mean(axis=1),
        "five_frame_displacement": np.linalg.norm(xy[:, -1] - xy[:, 0], axis=1),
        "heading_change": np.abs(heading[:, -1] - heading[:, 0]),
        "frame_to_frame_gps_variance": np.var(xy, axis=1).sum(axis=1),
    }


def _power_attributes(power: torch.Tensor) -> dict[str, np.ndarray]:
    values = torch.as_tensor(power, dtype=torch.float64)
    ordered = values.sort(dim=1, descending=True).values
    peak = ordered[:, 0].clamp_min(1e-12)
    probability = values / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1) / math.log(64)
    return {
        "future_power_peak": peak.numpy(),
        "future_power_top1_top2_gap": ((ordered[:, 0] - ordered[:, 1]) / peak).numpy(),
        "future_power_entropy": entropy.numpy(),
        "future_power_effective_candidates": values.ge(0.5 * peak[:, None]).sum(dim=1).numpy(),
        "normalized_oracle_gain_gap": (1.0 - ordered[:, 1] / peak).numpy(),
    }


def _protocol_label_metadata(config: Mapping[str, Any], split: str) -> dict[str, dict[str, int]]:
    protocol = json.loads(_path(config["protocol"]["manifest"]).read_text(encoding="utf-8"))
    result: dict[str, dict[str, int]] = {}
    for domain in protocol["domains"]:
        path = domain.get("train_split" if split == "train" else "validation_split")
        if not path:
            continue
        condition, scenario = domain["id"].split("/", 1)
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                stable = f"mmw:{condition}:{scenario}:{split}:{row['sample_id']}"
                result[stable] = {
                    "current_beam": int(row["beam_label"]),
                    "end_frame": int(row["end_frame"]),
                }
    return result


def _trajectory_positions(cache: Mapping[str, Any]) -> np.ndarray:
    result = np.zeros(len(cache["sample_id"]), dtype=np.float64)
    for trajectory in sorted(set(cache["trajectory_id"])):
        indices = np.asarray([index for index, value in enumerate(cache["trajectory_id"]) if value == trajectory])
        if indices.size > 1:
            result[indices] = np.arange(indices.size) / (indices.size - 1)
    return result


def _build_attribute_frame(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
    d3: Mapping[str, Any],
    d4: Mapping[str, Any],
    d5: Mapping[str, Any],
    dataset: str,
    train_class_frequency: np.ndarray,
    motion_thresholds: Mapping[str, float],
) -> pd.DataFrame:
    cache = data[dataset]
    current = prepared[dataset]
    count = len(cache["sample_id"])
    motion = _motion_attributes(cache["gps_history_relative_polar"])
    power = _power_attributes(cache["future_beam_power"])
    labels = cache["target"].numpy()
    metadata = _protocol_label_metadata(config, dataset)
    current_beam = np.asarray([metadata[value]["current_beam"] for value in cache["sample_id"]])
    topology = data["topology"].distance.cpu().numpy()
    beam_transition = topology[current_beam, labels]
    positions = _trajectory_positions(cache)
    single_indices = [11, 14, 12, 13]
    single_names = ["image", "lidar", "radar", "gps"]
    single_margin = current["stats"]["target_margin"][:, single_indices].numpy()
    single_prediction = current["stats"]["prediction"][:, single_indices].numpy()
    agreement = np.asarray(
        [max(np.bincount(row, minlength=64)) / 4.0 for row in single_prediction], dtype=np.float64
    )
    disagreements = np.asarray(
        [sum(left != right for left, right in combinations(row.tolist(), 2)) / 6.0 for row in single_prediction],
        dtype=np.float64,
    )
    best_single = np.asarray(single_names)[single_margin.argmax(axis=1)]
    worst_single = np.asarray(single_names)[single_margin.argmin(axis=1)]
    break_scores = {row["mask"]: float(row["most_severe_standardized_break"]) for row in d3["first_break"]}
    rank_delta: dict[tuple[str, str], float] = {}
    for row in d4["rows"]:
        if row["dataset"] == dataset and row["layer"] == "L5":
            rank_delta[(row["mask"], row["group"])] = float(row["effective_rank_delta"])
    frames: list[pd.DataFrame] = []
    neighbor_frequency = np.asarray(
        [train_class_frequency[(beam - 1) % 64] + train_class_frequency[(beam + 1) % 64] for beam in labels]
    )
    for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
        groups = current["groups"][:, mask_index - 1].numpy()
        geometry = current["geometry"]
        frame = pd.DataFrame(
            {
                "dataset": dataset,
                "sample_id": cache["sample_id"],
                "trajectory_id": cache["trajectory_id"],
                "domain_id": [f"{a}/{b}" for a, b in zip(cache["condition"], cache["scenario"])],
                "weather": cache["condition"],
                "scene": cache["scenario"],
                "town": cache["town"],
                "road_type": "unavailable",
                "mask_id": mask_index,
                "mask": mask,
                "available_modalities": "+".join(mask_metadata()[mask_index]["available_modalities"]),
                "missing_modalities": "+".join(mask_metadata()[mask_index]["missing_modalities"]),
                "missing_count": mask_metadata()[mask_index]["missing_count"],
                "group": np.asarray(GROUP_NAMES)[groups],
                "target_hard": (groups == 1).astype(np.int8),
                "g1_near": current["g1_near"][:, mask_index - 1].numpy(),
                "g1_far": current["g1_far"][:, mask_index - 1].numpy(),
                "g1_rank": current["g1_rank"][:, mask_index - 1].numpy(),
                "g1_collapse": current["g1_collapse"][:, mask_index - 1].numpy(),
                "target_beam": labels,
                "current_beam_label_side": current_beam,
                "beam_transition_cycle_distance": beam_transition,
                "beam_class_frequency_train": train_class_frequency[labels],
                "neighbor_beam_frequency_train": neighbor_frequency,
                "sequence_position": positions,
                "trajectory_edge": ((positions <= 0.05) | (positions >= 0.95)).astype(np.int8),
                "full_prediction": current["stats"]["prediction"][:, 0].numpy(),
                "missing_prediction": current["stats"]["prediction"][:, mask_index].numpy(),
                "full_target_rank": current["stats"]["target_rank"][:, 0].numpy(),
                "missing_target_rank": current["stats"]["target_rank"][:, mask_index].numpy(),
                "full_margin": current["stats"]["target_margin"][:, 0].numpy(),
                "missing_margin": current["stats"]["target_margin"][:, mask_index].numpy(),
                "full_entropy": current["stats"]["entropy"][:, 0].numpy(),
                "missing_entropy": current["stats"]["entropy"][:, mask_index].numpy(),
                "missing_embedding_norm": cache["z_raw"][:, mask_index].norm(dim=1).numpy(),
                "full_missing_cosine": geometry["cosine"][:, mask_index - 1].numpy(),
                "full_missing_euclidean": geometry["euclidean_distance"][:, mask_index - 1].numpy(),
                "margin_delta": geometry["delta_margin"][:, mask_index - 1].numpy(),
                "rank_delta": geometry["delta_rank"][:, mask_index - 1].numpy(),
                "parallel_drift": geometry["signed_parallel"][:, mask_index - 1].numpy(),
                "parallel_energy_ratio": geometry["parallel_energy_ratio"][:, mask_index - 1].numpy(),
                "orthogonal_drift": geometry["orthogonal_norm"][:, mask_index - 1].numpy(),
                "layer_break_score_train": break_scores[mask],
                "effective_rank_delta_group": [rank_delta.get((mask, GROUP_NAMES[value]), 0.0) for value in groups],
                "single_modality_agreement": agreement,
                "single_modality_pairwise_disagreement": disagreements,
                "best_single_modality": best_single,
                "worst_single_modality": worst_single,
                "fast_turn_train_threshold": (motion["angular_velocity_mean"] >= motion_thresholds["angular_velocity_q75"]).astype(np.int8),
                "high_speed_train_threshold": (motion["speed_mean"] >= motion_thresholds["speed_q75"]).astype(np.int8),
                "future_beam_power_role": "label_side_evaluation_metric_only",
                "historical_beam_role": "label_side_current_t_only",
                "historical_beam_change": "unavailable",
                "time_of_day": "unavailable",
            }
        )
        for key, values in motion.items():
            frame[key] = values
        for key, values in power.items():
            frame[key] = values
        for single_index, name in enumerate(single_names):
            frame[f"{name}_only_margin"] = single_margin[:, single_index]
            frame[f"{name}_mean_margin_utility"] = d5["features"][dataset]["modality_margin_utility"][:, single_index].numpy()
            frame[f"{name}_negative_utility_fraction"] = d5["features"][dataset]["modality_negative_fraction"][:, single_index].numpy()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
    cap: int = 1024,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.size > cap:
        first = first[rng.choice(first.size, cap, replace=False)]
    if second.size > cap:
        second = second[rng.choice(second.size, cap, replace=False)]
    estimates = np.empty(int(replicates), dtype=np.float64)
    for start in range(0, int(replicates), 100):
        stop = min(start + 100, int(replicates))
        left_index = rng.integers(0, first.size, size=(stop - start, first.size))
        right_index = rng.integers(0, second.size, size=(stop - start, second.size))
        estimates[start:stop] = first[left_index].mean(axis=1) - second[right_index].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _matched_effect(
    frame: pd.DataFrame,
    attribute: str,
    left_group: str = "G1",
    right_group: str = "G0",
) -> tuple[float, int]:
    right_group_frame = frame[frame["group"] == right_group]
    left_group_frame = frame[frame["group"] == left_group]
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for key, hard in left_group_frame.groupby(["trajectory_id", "target_beam"], sort=True):
        easy = right_group_frame[
            (right_group_frame["trajectory_id"] == key[0])
            & (right_group_frame["target_beam"] == key[1])
        ]
        count = min(len(hard), len(easy))
        if count:
            left.append(hard[attribute].to_numpy(dtype=np.float64)[:count])
            right.append(easy[attribute].to_numpy(dtype=np.float64)[:count])
    if not left:
        return 0.0, 0
    first = np.concatenate(left)
    second = np.concatenate(right)
    pooled = math.sqrt(0.5 * (first.var() + second.var()))
    return float((first.mean() - second.mean()) / max(pooled, 1e-12)), int(first.size)


def d6_hard_sample_attributes(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
    d3: Mapping[str, Any],
    d4: Mapping[str, Any],
    d5: Mapping[str, Any],
) -> dict[str, Any]:
    train_counts = torch.bincount(data["train"]["target"], minlength=64).numpy()
    train_motion = _motion_attributes(data["train"]["gps_history_relative_polar"])
    motion_thresholds = {
        "speed_q75": float(np.quantile(train_motion["speed_mean"], 0.75)),
        "angular_velocity_q75": float(np.quantile(train_motion["angular_velocity_mean"], 0.75)),
    }
    write_json(
        output / "artifacts/train_attribute_thresholds.json",
        {
            **motion_thresholds,
            "source_roles": ["train"],
            "validation_leakage_oracle": False,
        },
        sort_keys=True,
    )
    frames = {
        dataset: _build_attribute_frame(
            config,
            data,
            prepared,
            d3,
            d4,
            d5,
            dataset,
            train_counts,
            motion_thresholds,
        )
        for dataset in ("train", "validation")
    }
    combined = pd.concat((frames["train"], frames["validation"]), ignore_index=True)
    combined.to_parquet(output / "diagnostics/d6_hard_sample_attributes.parquet", index=False, compression="zstd")
    numeric_attributes = (
        "beam_class_frequency_train",
        "neighbor_beam_frequency_train",
        "beam_transition_cycle_distance",
        "future_power_top1_top2_gap",
        "future_power_entropy",
        "future_power_effective_candidates",
        "bs_distance_last",
        "speed_mean",
        "acceleration_mean",
        "angular_velocity_mean",
        "trajectory_curvature_mean",
        "five_frame_displacement",
        "heading_change",
        "sequence_position",
        "full_margin",
        "missing_margin",
        "missing_entropy",
        "single_modality_pairwise_disagreement",
        "full_missing_euclidean",
        "parallel_energy_ratio",
        "orthogonal_drift",
    )
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    tested_row_indices: list[int] = []
    replicates = int(config["statistics"]["bootstrap_replicates"])
    seed = int(config["statistics"]["bootstrap_seed"])
    contrasts = (("G1", "G0"), ("G2", "G0"), ("G3", "G0"))
    for dataset in ("train", "validation"):
        for mask_index, mask in enumerate(MASK_NAMES[1:]):
            scoped = frames[dataset][frames[dataset]["mask"] == mask]
            for attribute_index, attribute in enumerate(numeric_attributes):
                for contrast_index, (left_group, right_group) in enumerate(contrasts):
                    left = scoped.loc[scoped["group"] == left_group, attribute].to_numpy(dtype=np.float64)
                    right = scoped.loc[scoped["group"] == right_group, attribute].to_numpy(dtype=np.float64)
                    identity = {
                        "dataset": dataset,
                        "mask": mask,
                        "attribute": attribute,
                        "attribute_type": "numeric",
                        "contrast": f"{left_group}_vs_{right_group}",
                        "left_group": left_group,
                        "right_group": right_group,
                        "left_count": len(left),
                        "right_count": len(right),
                        "g1_count": len(left) if left_group == "G1" else "",
                        "g0_count": len(right) if right_group == "G0" else "",
                    }
                    if not left.size or not right.size:
                        rows.append(
                            {
                                **identity,
                                "status": "not_estimable_missing_contrast_group",
                                "bh_fdr_q": "",
                                "fdr_significant_0_05": False,
                            }
                        )
                        continue
                    pooled = math.sqrt(0.5 * (left.var() + right.var()))
                    effect = float((left.mean() - right.mean()) / max(pooled, 1e-12))
                    ci_low, ci_high = _bootstrap_difference(
                        left,
                        right,
                        replicates=replicates,
                        seed=(
                            seed
                            + 10000 * (dataset == "validation")
                            + 1000 * contrast_index
                            + 50 * mask_index
                            + attribute_index
                        ),
                    )
                    p_value = float(mannwhitneyu(left, right, alternative="two-sided").pvalue)
                    matched_effect, matched_count = _matched_effect(
                        scoped,
                        attribute,
                        left_group,
                        right_group,
                    )
                    rows.append(
                        {
                            **identity,
                            "left_mean": float(left.mean()),
                            "right_mean": float(right.mean()),
                            "standardized_effect_left_minus_right": effect,
                            "g1_mean": float(left.mean()) if left_group == "G1" else "",
                            "g0_mean": float(right.mean()) if right_group == "G0" else "",
                            "standardized_effect_g1_minus_g0": effect if left_group == "G1" else "",
                            "mean_difference_ci_low": ci_low,
                            "mean_difference_ci_high": ci_high,
                            "mann_whitney_p": p_value,
                            "matched_beam_trajectory_effect": matched_effect,
                            "matched_count_per_group": matched_count,
                            "bootstrap_replicates": replicates,
                            "status": "estimated",
                        }
                    )
                    p_values.append(p_value)
                    tested_row_indices.append(len(rows) - 1)
    q_values = benjamini_hochberg(np.asarray(p_values))
    for row_index, q_value in zip(tested_row_indices, q_values):
        row = rows[row_index]
        row["bh_fdr_q"] = float(q_value)
        row["fdr_significant_0_05"] = bool(q_value <= float(config["statistics"]["fdr_q"]))
    _write_csv(output / "diagnostics/d6_group_attribute_statistics.csv", rows)
    return {
        "frames": frames,
        "statistics": rows,
        "motion_thresholds": motion_thresholds,
        "parquet_rows": len(combined),
        "future_beam_power_role": "label_side_evaluation_metric_only",
    }


def _one_hot(values: np.ndarray, categories: Sequence[Any]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(categories)}
    result = np.zeros((len(values), len(categories)), dtype=np.float32)
    for row, value in enumerate(values):
        if value in lookup:
            result[row, lookup[value]] = 1.0
    return result


def _probe_feature_sets(
    frame: pd.DataFrame,
    cache: Mapping[str, Any],
    categories: Mapping[str, Sequence[Any]],
) -> dict[str, tuple[np.ndarray, list[str]]]:
    mask = _one_hot(frame["mask"].to_numpy(), categories["mask"])
    weather = _one_hot(frame["weather"].to_numpy(), categories["weather"])
    scene = _one_hot(frame["scene"].to_numpy(), categories["scene"])
    target = _one_hot(frame["target_beam"].to_numpy(), categories["target_beam"])
    mask_names = [f"mask={value}" for value in categories["mask"]]
    weather_names = [f"weather={value}" for value in categories["weather"]]
    scene_names = [f"scene={value}" for value in categories["scene"]]
    target_names = [f"target_beam={value}" for value in categories["target_beam"]]
    f0_numeric = ["missing_count"]
    f1_numeric = ["beam_class_frequency_train", "neighbor_beam_frequency_train"]
    f2_numeric = [
        "missing_count",
        "bs_distance_last",
        "bearing_last",
        "speed_mean",
        "speed_last",
        "acceleration_mean",
        "angular_velocity_mean",
        "angular_velocity_last",
        "angular_acceleration_mean",
        "trajectory_curvature_mean",
        "five_frame_displacement",
        "heading_change",
        "frame_to_frame_gps_variance",
        "trajectory_edge",
        "missing_margin",
        "missing_entropy",
        "missing_target_rank",
        "missing_embedding_norm",
        "single_modality_agreement",
        "single_modality_pairwise_disagreement",
        "layer_break_score_train",
    ]
    f3_numeric = [
        "full_margin",
        "full_entropy",
        "full_target_rank",
        "full_missing_cosine",
        "full_missing_euclidean",
        "margin_delta",
        "rank_delta",
        "parallel_drift",
        "parallel_energy_ratio",
        "orthogonal_drift",
        "future_power_peak",
        "future_power_top1_top2_gap",
        "future_power_entropy",
        "future_power_effective_candidates",
        "normalized_oracle_gain_gap",
    ]
    repeated_index = np.tile(np.arange(len(cache["sample_id"])), len(MISSING_INDICES))
    full_embedding = cache["z_raw"][:, 0].numpy()[repeated_index]
    sets = {
        "F0": (
            np.concatenate((mask, frame[f0_numeric].to_numpy(dtype=np.float32)), axis=1),
            mask_names + f0_numeric,
        ),
        "F1": (
            np.concatenate((mask, target, weather, scene, frame[f1_numeric].to_numpy(dtype=np.float32)), axis=1),
            mask_names + target_names + weather_names + scene_names + f1_numeric,
        ),
        "F2": (
            np.concatenate((mask, weather, scene, frame[f2_numeric].to_numpy(dtype=np.float32)), axis=1),
            mask_names + weather_names + scene_names + f2_numeric,
        ),
        "F3": (
            np.concatenate(
                (
                    mask,
                    target,
                    weather,
                    scene,
                    frame[f2_numeric + f3_numeric].to_numpy(dtype=np.float32),
                    full_embedding.astype(np.float32),
                ),
                axis=1,
            ),
            mask_names
            + target_names
            + weather_names
            + scene_names
            + f2_numeric
            + f3_numeric
            + [f"full_embedding_{index}" for index in range(64)],
        ),
    }
    for name, (values, _) in sets.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} probe features contain non-finite values.")
    return sets


def _calibrate_probe_state(features: np.ndarray, labels: np.ndarray, state: Mapping[str, Any]) -> dict[str, Any]:
    calibrated = dict(state)
    logits, _ = predict_logistic_probe(features, calibrated)
    prevalence = float(np.mean(labels))
    lower, upper = -20.0, 20.0
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits + middle, -50, 50)))
        if float(probability.mean()) < prevalence:
            lower = middle
        else:
            upper = middle
    calibrated["bias"] = float(calibrated["bias"]) + 0.5 * (lower + upper)
    calibrated["calibration"] = "train_only_intercept_prevalence_matching"
    calibrated["calibration_source_role"] = "train"
    return calibrated


def _fit_probe(
    config: Mapping[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    epochs: int | None = None,
) -> dict[str, Any]:
    device = config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    state = fit_logistic_probe(
        features,
        labels,
        l2=float(config["probe"]["l2"]),
        epochs=int(epochs or config["probe"]["epochs"]),
        batch_size=int(config["probe"]["batch_size"]),
        seed=int(seed),
        device=device,
    )
    return _calibrate_probe_state(features, labels, state)


def _probe_metric_rows(
    feature_set: str,
    scope: str,
    labels: np.ndarray,
    probability: np.ndarray,
    trajectories: np.ndarray,
) -> list[dict[str, Any]]:
    rows = [
        {
            "feature_set": feature_set,
            "model": "logistic_regression",
            "scope": scope,
            "trajectory_id": "all",
            "status": "estimated",
            **binary_probe_metrics(labels, probability),
        }
    ]
    for trajectory in sorted(set(trajectories.tolist())):
        chosen = trajectories == trajectory
        if set(np.unique(labels[chosen])) == {0, 1}:
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": "logistic_regression",
                    "scope": scope,
                    "trajectory_id": trajectory,
                    "status": "estimated",
                    **binary_probe_metrics(labels[chosen], probability[chosen]),
                }
            )
    return rows


def d7_hardness_probes(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    d6: Mapping[str, Any],
) -> dict[str, Any]:
    train_frame = d6["frames"]["train"]
    validation_frame = d6["frames"]["validation"]
    categories = {
        "mask": list(MASK_NAMES[1:]),
        "weather": sorted(train_frame["weather"].unique().tolist()),
        "scene": sorted(train_frame["scene"].unique().tolist()),
        "target_beam": list(range(64)),
    }
    train_sets = _probe_feature_sets(train_frame, data["train"], categories)
    validation_sets = _probe_feature_sets(validation_frame, data["validation"], categories)
    train_labels = train_frame["target_hard"].to_numpy(dtype=np.int8)
    validation_labels = validation_frame["target_hard"].to_numpy(dtype=np.int8)
    validation_trajectories = validation_frame["trajectory_id"].to_numpy()
    summary_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    states: dict[str, Any] = {}
    seed = int(config["statistics"]["probe_seed"])
    validation_probability: dict[str, np.ndarray] = {}
    for feature_index, feature_set in enumerate(("F0", "F1", "F2", "F3")):
        train_x, names = train_sets[feature_set]
        validation_x, _ = validation_sets[feature_set]
        state = _fit_probe(config, train_x, train_labels, seed=seed + feature_index)
        _, probability = predict_logistic_probe(validation_x, state)
        validation_probability[feature_set] = probability
        summary_rows.extend(
            _probe_metric_rows(feature_set, "validation_unified", validation_labels, probability, validation_trajectories)
        )
        states[f"unified:{feature_set}"] = state
        for name, coefficient in zip(names, np.asarray(state["weight"])):
            importance_rows.append(
                {
                    "scope": "unified",
                    "feature_set": feature_set,
                    "feature": name,
                    "importance_type": "absolute_standardized_coefficient",
                    "importance": float(abs(coefficient)),
                    "signed_coefficient": float(coefficient),
                    "source_role": "train",
                }
            )
    for mask_index, mask in enumerate(MASK_NAMES[1:]):
        train_chosen = train_frame["mask"].to_numpy() == mask
        validation_chosen = validation_frame["mask"].to_numpy() == mask
        for feature_index, feature_set in enumerate(("F0", "F1", "F2", "F3")):
            train_x, names = train_sets[feature_set]
            validation_x, _ = validation_sets[feature_set]
            if set(np.unique(train_labels[train_chosen])) != {0, 1}:
                summary_rows.append(
                    {
                        "feature_set": feature_set,
                        "model": "logistic_regression",
                        "scope": f"validation_mask:{mask}",
                        "trajectory_id": "all",
                        "status": "not_estimable_no_train_positive_or_negative",
                        "train_sample_count": int(train_chosen.sum()),
                        "train_prevalence": float(train_labels[train_chosen].mean()),
                        "validation_evaluated": False,
                    }
                )
                continue
            state = _fit_probe(
                config,
                train_x[train_chosen],
                train_labels[train_chosen],
                seed=seed + 100 + 10 * mask_index + feature_index,
            )
            _, probability = predict_logistic_probe(validation_x[validation_chosen], state)
            summary_rows.extend(
                _probe_metric_rows(
                    feature_set,
                    f"validation_mask:{mask}",
                    validation_labels[validation_chosen],
                    probability,
                    validation_trajectories[validation_chosen],
                )
            )
            states[f"mask:{mask}:{feature_set}"] = state
            for name, coefficient in zip(names, np.asarray(state["weight"])):
                importance_rows.append(
                    {
                        "scope": f"mask:{mask}",
                        "feature_set": feature_set,
                        "feature": name,
                        "importance_type": "absolute_standardized_coefficient",
                        "importance": float(abs(coefficient)),
                        "signed_coefficient": float(coefficient),
                        "source_role": "train",
                    }
                )
    train_trajectories = train_frame["trajectory_id"].to_numpy()
    f2_train, f2_names = train_sets["F2"]
    unique_train_trajectories = sorted(set(train_trajectories.tolist()))
    stability_epochs = int(config["probe"]["stability_epochs"])
    folds = [unique_train_trajectories[index:: int(config["probe"]["trajectory_cv_folds"])] for index in range(4)]
    permutation_importance: dict[str, list[float]] = defaultdict(list)
    for fold_index, holdout_trajectories in enumerate(folds):
        holdout = np.isin(train_trajectories, holdout_trajectories)
        fit = ~holdout
        state = _fit_probe(
            config,
            f2_train[fit],
            train_labels[fit],
            seed=seed + 1000 + fold_index,
            epochs=stability_epochs,
        )
        _, probability = predict_logistic_probe(f2_train[holdout], state)
        metrics = binary_probe_metrics(train_labels[holdout], probability)
        summary_rows.append(
            {
                "feature_set": "F2",
                "model": "logistic_regression",
                "scope": f"train_4fold_cv:{fold_index}",
                "trajectory_id": "+".join(holdout_trajectories),
                **metrics,
            }
        )
        baseline_pr = metrics["pr_auc"]
        rng = np.random.default_rng(seed + 2000 + fold_index)
        holdout_x = f2_train[holdout]
        for feature_index, name in enumerate(f2_names):
            shuffled = holdout_x.copy()
            shuffled[:, feature_index] = shuffled[rng.permutation(len(shuffled)), feature_index]
            _, shuffled_probability = predict_logistic_probe(shuffled, state)
            permutation_importance[name].append(
                baseline_pr - binary_probe_metrics(train_labels[holdout], shuffled_probability)["pr_auc"]
            )
    for name, values in permutation_importance.items():
        importance_rows.append(
            {
                "scope": "train_4fold_cv",
                "feature_set": "F2",
                "feature": name,
                "importance_type": "permutation_pr_auc_drop",
                "importance": float(np.mean(values)),
                "signed_coefficient": "",
                "source_role": "train_holdout",
            }
        )
    loto_rows: list[dict[str, Any]] = []
    loto_top_feature_sets: list[set[str]] = []
    for trajectory_index, trajectory in enumerate(unique_train_trajectories):
        holdout = train_trajectories == trajectory
        fit = ~holdout
        state = _fit_probe(
            config,
            f2_train[fit],
            train_labels[fit],
            seed=seed + 3000 + trajectory_index,
            epochs=stability_epochs,
        )
        _, probability = predict_logistic_probe(f2_train[holdout], state)
        metrics = binary_probe_metrics(train_labels[holdout], probability)
        top_count = min(20, len(f2_names))
        top_indices = np.argsort(np.abs(np.asarray(state["weight"])).reshape(-1))[-top_count:]
        loto_top_feature_sets.append({f2_names[index] for index in top_indices})
        row = {
            "feature_set": "F2",
            "model": "logistic_regression",
            "scope": "train_loto",
            "trajectory_id": trajectory,
            "status": "estimated",
            **metrics,
        }
        summary_rows.append(row)
        loto_rows.append(row)
    loto_jaccards = [
        len(left & right) / max(1, len(left | right))
        for left, right in combinations(loto_top_feature_sets, 2)
    ]
    loto_top_feature_jaccard = float(np.mean(loto_jaccards))
    _torch_save(output / "artifacts/hardness_probe/probe_states.pt", states)
    write_json(
        output / "artifacts/hardness_probe/feature_manifest.json",
        {
            "categories": categories,
            "features": {key: names for key, (_, names) in train_sets.items()},
            "selection_source": "train_only",
            "validation_evaluations": 1,
            "f2_train_loto_top20_feature_jaccard": loto_top_feature_jaccard,
            "hist_gradient_boosting_status": "not_installed; no new modeling dependency added",
            "random_forest_status": "not_run; logistic regression is the pre-registered primary probe",
        },
        sort_keys=True,
    )
    _write_csv(output / "diagnostics/d7_hardness_probe_summary.csv", summary_rows)
    _write_csv(output / "diagnostics/d7_feature_importance.csv", importance_rows)
    unified = {
        row["feature_set"]: row
        for row in summary_rows
        if row["scope"] == "validation_unified" and row["trajectory_id"] == "all"
    }
    return {
        "summary_rows": summary_rows,
        "importance_rows": importance_rows,
        "unified": unified,
        "loto": loto_rows,
        "loto_top_feature_jaccard": loto_top_feature_jaccard,
        "validation_probability": validation_probability,
        "validation_labels": validation_labels,
        "validation_trajectories": validation_trajectories,
        "validation_evaluations": 1,
    }


def _repair_summary(
    alpha: torch.Tensor,
    start_correct: torch.Tensor,
    end_correct: torch.Tensor,
    chosen: torch.Tensor,
    **identity: Any,
) -> dict[str, Any]:
    selected_alpha = alpha[chosen]
    finite = torch.isfinite(selected_alpha)
    repaired = (~start_correct[chosen]) & end_correct[chosen]
    harmed = start_correct[chosen] & ~end_correct[chosen]
    return {
        **identity,
        "sample_count": int(chosen.sum()),
        "recoverable_fraction": float(finite.float().mean()),
        "alpha_star_mean_recovered": float(selected_alpha[finite].mean()) if bool(finite.any()) else "",
        "alpha_star_median_recovered": float(selected_alpha[finite].median()) if bool(finite.any()) else "",
        "alpha_star_q25_recovered": float(selected_alpha[finite].quantile(0.25)) if bool(finite.any()) else "",
        "alpha_star_q75_recovered": float(selected_alpha[finite].quantile(0.75)) if bool(finite.any()) else "",
        "unrecovered_count": int((~finite).sum()),
        "fix_rate": float(repaired.float().mean()),
        "harm_rate": float(harmed.float().mean()),
    }


def d8_counterfactual_repair(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    _set_seed(int(config["statistics"]["sample_selection_seed"]) + 800)
    train_full = normalize(data["train"]["z_raw"][:, 0].float())
    train_missing = normalize(data["train"]["z_raw"][:, 1:].float())
    train_repair = train_full[:, None] - train_missing
    train_g1 = prepared["train"]["groups"].eq(1)
    pca = fit_pca_directions(train_repair[train_g1], max_components=max(config["geometry"]["pca_components"]))
    pca_artifact = {
        **pca,
        "source_roles": ["train"],
        "validation_leakage_oracle": False,
        "fit_sample_mask_pairs": int(train_g1.sum()),
        "definition": "PCA of normalized Full-minus-Missing repair vectors on train G1 pairs",
    }
    _torch_save(output / "artifacts/train_pca_directions.pt", pca_artifact)
    cache = data["validation"]
    current = prepared["validation"]
    full = normalize(cache["z_raw"][:, 0].float())
    missing = normalize(cache["z_raw"][:, 1:].float())
    labels = cache["target"].long()
    bank = normalize(data["prototypes"])
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    basis = pca["basis"].to(device)
    rows: list[dict[str, Any]] = []
    sample_artifact: dict[str, Any] = {
        "sample_id": cache["sample_id"],
        "mask_order": list(MASK_NAMES[1:]),
        "source_roles": ["train", "validation_final"],
        "validation_used_for_selection": False,
        "methods": {},
    }
    random_recovery_by_mask: dict[str, float] = {}
    for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
        source = missing[:, mask_index - 1].to(device)
        destination_full = full.to(device)
        target = labels.to(device)
        g1 = current["groups"][:, mask_index - 1].eq(1)
        if not bool(g1.any()):
            continue
        repair = destination_full - source
        parallel_destination = source - current["geometry"]["parallel_vector"][:, mask_index - 1].to(device)
        orthogonal_destination = source - current["geometry"]["orthogonal_vector"][:, mask_index - 1].to(device)
        methods: dict[str, torch.Tensor] = {
            "CF1_full": destination_full,
            "CF2_parallel": parallel_destination,
            "CF3_orthogonal": orthogonal_destination,
        }
        for components in config["geometry"]["pca_components"]:
            projected = repair @ basis[:, : int(components)] @ basis[:, : int(components)].t()
            methods[f"CF4_pca_k{components}"] = source + projected
        start_prediction = source @ bank.to(device).t()
        start_correct = start_prediction.argmax(dim=1).eq(target).cpu()
        sample_artifact["methods"][mask] = {}
        for method, destination in methods.items():
            alpha = minimal_interpolation_alpha(
                source[g1.to(device)],
                destination[g1.to(device)],
                target[g1.to(device)],
                bank.to(device),
                steps=int(config["geometry"]["counterfactual_steps"]),
            ).cpu()
            full_alpha = torch.full((len(cache["sample_id"]),), torch.inf)
            full_alpha[g1] = alpha
            end_prediction = normalize(destination) @ bank.to(device).t()
            end_correct = end_prediction.argmax(dim=1).eq(target).cpu()
            rows.append(
                _repair_summary(
                    full_alpha,
                    start_correct,
                    end_correct,
                    g1,
                    dataset="validation",
                    mask=mask,
                    method=method,
                    scope="all_g1",
                    random_seed="",
                )
            )
            for scope_name, scope_mask in (
                ("G1-near", current["g1_near"][:, mask_index - 1]),
                ("G1-far", current["g1_far"][:, mask_index - 1]),
            ):
                if bool(scope_mask.any()):
                    rows.append(
                        _repair_summary(
                            full_alpha,
                            start_correct,
                            end_correct,
                            scope_mask,
                            dataset="validation",
                            mask=mask,
                            method=method,
                            scope=scope_name,
                            random_seed="",
                        )
                    )
            for beam in labels.unique(sorted=True).tolist():
                beam_scope = labels.eq(beam) & g1
                if bool(beam_scope.any()):
                    rows.append(
                        _repair_summary(
                            full_alpha,
                            start_correct,
                            end_correct,
                            beam_scope,
                            dataset="validation",
                            mask=mask,
                            method=method,
                            scope=f"target_beam:{beam}",
                            random_seed="",
                        )
                    )
            for trajectory in sorted(set(cache["trajectory_id"])):
                trajectory_scope = torch.tensor([value == trajectory for value in cache["trajectory_id"]]) & g1
                if bool(trajectory_scope.any()):
                    rows.append(
                        _repair_summary(
                            full_alpha,
                            start_correct,
                            end_correct,
                            trajectory_scope,
                            dataset="validation",
                            mask=mask,
                            method=method,
                            scope=f"trajectory:{trajectory}",
                            random_seed="",
                        )
                    )
            sample_artifact["methods"][mask][method] = full_alpha
        random_rates = []
        repair_norm = repair.norm(dim=1, keepdim=True)
        for random_seed in config["statistics"]["random_counterfactual_seeds"]:
            generator = torch.Generator(device=device).manual_seed(int(random_seed) + mask_index)
            random_direction = normalize(torch.randn(source.shape, generator=generator, device=device)) * repair_norm
            random_destination = source + random_direction
            random_end = (normalize(random_destination) @ bank.to(device).t()).argmax(dim=1).eq(target).cpu()
            recovery = float(random_end[g1].float().mean())
            random_rates.append(recovery)
            rows.append(
                {
                    "dataset": "validation",
                    "mask": mask,
                    "method": "CF5_random_same_energy",
                    "scope": "all_g1",
                    "random_seed": int(random_seed),
                    "sample_count": int(g1.sum()),
                    "recoverable_fraction": recovery,
                    "alpha_star_mean_recovered": "",
                    "alpha_star_median_recovered": "",
                    "alpha_star_q25_recovered": "",
                    "alpha_star_q75_recovered": "",
                    "unrecovered_count": int((~random_end[g1]).sum()),
                    "fix_rate": recovery,
                    "harm_rate": float((start_correct & ~random_end)[~g1].float().mean()) if bool((~g1).any()) else 0.0,
                }
            )
        random_recovery_by_mask[mask] = float(np.mean(random_rates))
    _torch_save(output / "artifacts/counterfactual_sample_results.pt", sample_artifact)
    _write_csv(output / "diagnostics/d8_counterfactual_repair.csv", rows)
    primary = [
        row
        for row in rows
        if row["scope"] == "all_g1" and row["random_seed"] == "" and row["dataset"] == "validation"
    ]
    return {
        "rows": rows,
        "primary": primary,
        "random_recovery_by_mask": random_recovery_by_mask,
        "pca_fit_pairs": int(train_g1.sum()),
        "validation_used_for_direction_selection": False,
    }


def _target_manifold_similarity(
    query: torch.Tensor,
    query_labels: torch.Tensor,
    index: torch.Tensor,
    index_labels: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    result = torch.zeros(query.shape[0], dtype=torch.float32)
    normalized_index = normalize(index.float())
    normalized_query = normalize(query.float())
    for beam in range(64):
        query_rows = query_labels.eq(beam).nonzero().flatten()
        index_rows = index_labels.eq(beam).nonzero().flatten()
        if not query_rows.numel() or not index_rows.numel():
            continue
        database = normalized_index[index_rows].to(device)
        for start in range(0, query_rows.numel(), 256):
            rows = query_rows[start : start + 256]
            result[rows] = normalized_query[rows].to(device).matmul(database.t()).max(dim=1).values.cpu()
    return result


def _local_intrinsic_dimension(similarity: torch.Tensor, k: int) -> torch.Tensor:
    distance = (2.0 * (1.0 - similarity[:, :k]).clamp_min(1e-12)).sqrt()
    radius = distance[:, -1:].clamp_min(1e-12)
    denominator = torch.log(distance.clamp_min(1e-12) / radius).sum(dim=1)
    return -float(k) / denominator.clamp_max(-1e-12)


def _knn_scope_row(
    cache: Mapping[str, Any],
    prepared: Mapping[str, Any],
    train_labels: torch.Tensor,
    full_indices: torch.Tensor,
    missing_indices: torch.Tensor,
    missing_similarity: torch.Tensor,
    target_manifold_similarity: torch.Tensor,
    prototypes: torch.Tensor,
    mask_index: int,
    k: int,
    chosen: torch.Tensor,
    **identity: Any,
) -> dict[str, Any]:
    labels = cache["target"]
    neighbor_labels = train_labels[missing_indices[:, :k]]
    purity = neighbor_labels.eq(labels[:, None]).float().mean(dim=1)
    missing_prediction = prepared["stats"]["prediction"][:, mask_index]
    wrong_cluster = neighbor_labels.eq(missing_prediction[:, None]).float().mean(dim=1)
    preservation = torch.tensor(
        [
            len(set(left[:k]) & set(right[:k])) / k
            for left, right in zip(full_indices.tolist(), missing_indices.tolist())
        ],
        dtype=torch.float32,
    )
    intrinsic = _local_intrinsic_dimension(missing_similarity, k)
    geometry = prepared["geometry"]
    boundary = prepared["stats"]["target_margin"][:, mask_index].abs()
    normalized_missing = normalize(cache["z_raw"][:, mask_index].float())
    bank = normalize(prototypes.float())
    target_similarity = (normalized_missing * bank[labels]).sum(dim=1)
    confuser = geometry["confuser"][:, mask_index - 1]
    confuser_similarity = (normalized_missing * bank[confuser]).sum(dim=1)
    return {
        **identity,
        "sample_count": int(chosen.sum()),
        "k": k,
        "target_neighbor_purity": float(purity[chosen].mean()),
        "full_missing_neighbor_preservation": float(preservation[chosen].mean()),
        "wrong_prediction_cluster_fraction": float(wrong_cluster[chosen].mean()),
        "local_density": float(missing_similarity[chosen, :k].mean()),
        "target_prototype_cosine": float(target_similarity[chosen].mean()),
        "target_prototype_cosine_distance": float((1.0 - target_similarity[chosen]).mean()),
        "confuser_prototype_cosine": float(confuser_similarity[chosen].mean()),
        "target_prototype_similarity_delta": float(geometry["delta_target"][chosen, mask_index - 1].mean()),
        "confuser_prototype_similarity_delta": float(geometry["delta_confuser"][chosen, mask_index - 1].mean()),
        "target_empirical_manifold_cosine": float(target_manifold_similarity[chosen].mean()),
        "local_intrinsic_dimension": float(intrinsic[chosen].mean()),
        "nearest_decision_boundary_margin_proxy": float(boundary[chosen].mean()),
    }


def d9_local_neighborhood(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    train = data["train"]
    validation = data["validation"]
    device = torch.device(config["runtime"]["knn_device"] if torch.cuda.is_available() else "cpu")
    full_similarity, full_indices = cosine_knn(
        validation["z_raw"][:, 0],
        train["z_raw"][:, 0],
        k=max(config["geometry"]["knn_k"]),
        device=device,
        chunk_size=256,
    )
    rows: list[dict[str, Any]] = []
    for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
        similarity, indices = cosine_knn(
            validation["z_raw"][:, mask_index],
            train["z_raw"][:, mask_index],
            k=max(config["geometry"]["knn_k"]),
            device=device,
            chunk_size=256,
        )
        manifold = _target_manifold_similarity(
            validation["z_raw"][:, mask_index],
            validation["target"],
            train["z_raw"][:, mask_index],
            train["target"],
            device=device,
        )
        _torch_save(
            output / f"artifacts/knn_indices/{mask}.pt",
            {
                "mask": mask,
                "query_split": "validation",
                "index_split": "train",
                "query_sample_id_sha256": _string_hash(validation["sample_id"]),
                "index_sample_id_sha256": _string_hash(train["sample_id"]),
                "k": int(max(config["geometry"]["knn_k"])),
                "cosine_similarity": similarity,
                "indices": indices,
                "exact_brute_force": True,
            },
        )
        for k in config["geometry"]["knn_k"]:
            for group_id, group in enumerate(("G0", "G1")):
                chosen = prepared["validation"]["groups"][:, mask_index - 1].eq(group_id)
                rows.append(
                    _knn_scope_row(
                        validation,
                        prepared["validation"],
                        train["target"],
                        full_indices,
                        indices,
                        similarity,
                        manifold,
                        data["prototypes"],
                        mask_index,
                        int(k),
                        chosen,
                        dataset="validation",
                        mask=mask,
                        group=group,
                        scope="all",
                    )
                )
            for trajectory in sorted(set(validation["trajectory_id"])):
                trajectory_scope = torch.tensor([value == trajectory for value in validation["trajectory_id"]])
                for group_id, group in enumerate(("G0", "G1")):
                    chosen = trajectory_scope & prepared["validation"]["groups"][:, mask_index - 1].eq(group_id)
                    rows.append(
                        _knn_scope_row(
                            validation,
                            prepared["validation"],
                            train["target"],
                            full_indices,
                            indices,
                            similarity,
                            manifold,
                            data["prototypes"],
                            mask_index,
                            int(k),
                            chosen,
                            dataset="validation",
                            mask=mask,
                            group=group,
                            scope=f"trajectory:{trajectory}",
                        )
                    )
    _write_csv(output / "diagnostics/d9_local_neighborhood.csv", rows)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rows": rows, "exact_train_index": True, "k": list(config["geometry"]["knn_k"])}


def _load_raw_encoder_tokens(output: Path, cache: Mapping[str, Any]) -> torch.Tensor:
    parts = []
    for record in cache["layer_shards"]:
        shard = torch.load(output / record["path"], map_location="cpu", weights_only=False)
        parts.append(shard["encoder_tokens"].float())
    result = torch.cat(parts)
    if result.shape[0] != len(cache["sample_id"]):
        raise ValueError(f"{cache['split']} raw token shards are misaligned.")
    return result


def d10_temporal_difficulty(
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
    d6: Mapping[str, Any],
) -> dict[str, Any]:
    cache = data["validation"]
    current = prepared["validation"]
    if not cache["temporal_prediction"][:, :, 0].eq(current["stats"]["prediction"]).all():
        raise ValueError("T0 temporal cache does not match the original frozen prediction.")
    if not torch.equal(cache["temporal_target_margin"][:, :, 0], current["stats"]["target_margin"]):
        raise ValueError("T0 temporal cache does not match the original frozen target margin.")
    if not cache["temporal_target_rank"][:, :, 0].eq(current["stats"]["target_rank"]).all():
        raise ValueError("T0 temporal cache does not match the original frozen target rank.")
    tokens = _load_raw_encoder_tokens(output, cache)
    bits = torch.tensor(list(MASKS.values()), dtype=tokens.dtype)
    available = tokens[:, None] * bits[None, :, None, :, None]
    flattened_frames = available.flatten(3)
    normalized_frames = normalize(flattened_frames)
    frame_cosine = (normalized_frames[:, :, 1:] * normalized_frames[:, :, :-1]).sum(dim=3).mean(dim=2)
    temporal_variance = flattened_frames.var(dim=2).mean(dim=2)
    last_history_difference = (flattened_frames[:, :, -1] - flattened_frames[:, :, :-1].mean(dim=2)).norm(dim=2)
    full_frames = available[:, 0]
    frame_missing_distance = (available[:, 1:] - full_frames[:, None]).flatten(3).norm(dim=3)
    attribute_frame = d6["frames"]["validation"]
    beam_switch = torch.from_numpy(
        attribute_frame[attribute_frame["mask"] == MASK_NAMES[1]]["beam_transition_cycle_distance"]
        .to_numpy(copy=True)
    )
    rows: list[dict[str, Any]] = []
    for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
        for group_id, group in enumerate(("G0", "G1")):
            chosen = current["groups"][:, mask_index - 1].eq(group_id)
            frame_drift = frame_missing_distance[chosen, mask_index - 1]
            impact = frame_drift.mean(dim=0)
            rows.append(
                {
                    "dataset": "validation",
                    "mask": mask,
                    "group": group,
                    "scope": "all",
                    "analysis": "token_temporal_geometry",
                    "ablation": "not_applicable",
                    "sample_count": int(chosen.sum()),
                    "frame_to_frame_cosine": float(frame_cosine[chosen, mask_index].mean()),
                    "temporal_variance": float(temporal_variance[chosen, mask_index].mean()),
                    "last_vs_history_mean_distance": float(last_history_difference[chosen, mask_index].mean()),
                    "missing_frame_0_distance": float(impact[0]),
                    "missing_frame_1_distance": float(impact[1]),
                    "missing_frame_2_distance": float(impact[2]),
                    "missing_frame_3_distance": float(impact[3]),
                    "missing_frame_4_distance": float(impact[4]),
                    "largest_missing_impact_frame": int(impact.argmax()),
                    "beam_transition_cycle_distance": float(beam_switch[chosen].float().mean()),
                    "temporal_module": "not_present_in_frozen_m4",
                    "attention_or_gru_state": "not_present_in_frozen_m4",
                }
            )
            for temporal_index, ablation in enumerate(TEMPORAL_NAMES):
                prediction = cache["temporal_prediction"][:, mask_index, temporal_index]
                rows.append(
                    {
                        "dataset": "validation",
                        "mask": mask,
                        "group": group,
                        "scope": "all",
                        "analysis": "frozen_fusion_temporal_ablation",
                        "ablation": ablation,
                        "sample_count": int(chosen.sum()),
                        "top1": float(prediction[chosen].eq(cache["target"][chosen]).float().mean()),
                        "top1_delta_vs_t0": float(
                            prediction[chosen].eq(cache["target"][chosen]).float().mean()
                            - cache["temporal_prediction"][chosen, mask_index, 0]
                            .eq(cache["target"][chosen])
                            .float()
                            .mean()
                        ),
                        "target_margin": float(cache["temporal_target_margin"][chosen, mask_index, temporal_index].mean()),
                        "target_rank": float(cache["temporal_target_rank"][chosen, mask_index, temporal_index].float().mean()),
                        "beam_transition_cycle_distance": float(beam_switch[chosen].float().mean()),
                        "temporal_module": "not_present_in_frozen_m4",
                        "attention_or_gru_state": "not_present_in_frozen_m4",
                    }
                )
        for trajectory in sorted(set(cache["trajectory_id"])):
            trajectory_scope = torch.tensor([value == trajectory for value in cache["trajectory_id"]])
            for group_id, group in enumerate(("G0", "G1")):
                chosen = trajectory_scope & current["groups"][:, mask_index - 1].eq(group_id)
                if not bool(chosen.any()):
                    continue
                for temporal_index, ablation in enumerate(TEMPORAL_NAMES):
                    prediction = cache["temporal_prediction"][:, mask_index, temporal_index]
                    rows.append(
                        {
                            "dataset": "validation",
                            "mask": mask,
                            "group": group,
                            "scope": f"trajectory:{trajectory}",
                            "analysis": "frozen_fusion_temporal_ablation",
                            "ablation": ablation,
                            "sample_count": int(chosen.sum()),
                            "top1": float(prediction[chosen].eq(cache["target"][chosen]).float().mean()),
                            "top1_delta_vs_t0": float(
                                prediction[chosen].eq(cache["target"][chosen]).float().mean()
                                - cache["temporal_prediction"][chosen, mask_index, 0]
                                .eq(cache["target"][chosen])
                                .float()
                                .mean()
                            ),
                            "target_margin": float(
                                cache["temporal_target_margin"][chosen, mask_index, temporal_index].mean()
                            ),
                            "target_rank": float(
                                cache["temporal_target_rank"][chosen, mask_index, temporal_index].float().mean()
                            ),
                            "beam_transition_cycle_distance": float(beam_switch[chosen].float().mean()),
                            "temporal_module": "not_present_in_frozen_m4",
                            "attention_or_gru_state": "not_present_in_frozen_m4",
                        }
                    )
    _write_csv(output / "diagnostics/d10_temporal_difficulty.csv", rows)
    return {"rows": rows, "temporal_module_present": False, "ablation_names": list(TEMPORAL_NAMES)}


def stability_analysis(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
    d2: Mapping[str, Any],
    d3: Mapping[str, Any],
    d5: Mapping[str, Any],
    d6: Mapping[str, Any],
    d7: Mapping[str, Any],
    d8: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    validation = data["validation"]
    trajectories = sorted(set(validation["trajectory_id"]))
    bootstrap_replicates = int(config["statistics"]["bootstrap_replicates"])
    seed = int(config["statistics"]["bootstrap_seed"]) + 9000
    parallel_effects: dict[str, list[float]] = {}
    rank_effects: dict[str, list[float]] = {}
    layer_agreements = []
    for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
        geometry = prepared["validation"]["geometry"]["signed_parallel"][:, mask_index - 1].numpy()
        groups = prepared["validation"]["groups"][:, mask_index - 1].numpy()
        g1, g0 = geometry[groups == 1], geometry[groups == 0]
        ci_low, ci_high = _bootstrap_difference(
            g1,
            g0,
            replicates=bootstrap_replicates,
            seed=seed + mask_index,
        )
        effects = []
        for trajectory in trajectories:
            scope = np.asarray(validation["trajectory_id"]) == trajectory
            effects.append(float(geometry[scope & (groups == 1)].mean() - geometry[scope & (groups == 0)].mean()))
        parallel_effects[mask] = effects
        rows.append(
            {
                "metric": "parallel_drift_g1_minus_g0",
                "mask": mask,
                "scope": "validation",
                "value": float(g1.mean() - g0.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "trajectory_1_value": effects[0],
                "trajectory_2_value": effects[1],
                "trajectory_direction_agreement": bool(np.sign(effects[0]) == np.sign(effects[1])),
                "bootstrap_replicates": bootstrap_replicates,
            }
        )
        severe_by_trajectory = []
        rank_by_trajectory = []
        for trajectory in trajectories:
            scope_name = f"trajectory:{trajectory}"
            scores = []
            for layer in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
                g0_row = next(
                    row
                    for row in d3["rows"]
                    if row["dataset"] == "validation"
                    and row["scope"] == scope_name
                    and row["mask"] == mask
                    and row["layer"] == layer
                    and row["group"] == "G0"
                )
                g1_row = next(
                    row
                    for row in d3["rows"]
                    if row["dataset"] == "validation"
                    and row["scope"] == scope_name
                    and row["mask"] == mask
                    and row["layer"] == layer
                    and row["group"] == "G1"
                )
                pooled = math.sqrt(
                    0.5
                    * (
                        float(g0_row["paired_euclidean_std"]) ** 2
                        + float(g1_row["paired_euclidean_std"]) ** 2
                    )
                )
                scores.append(
                    (
                        layer,
                        (float(g1_row["paired_euclidean_mean"]) - float(g0_row["paired_euclidean_mean"]))
                        / max(pooled, 1e-12),
                    )
                )
            severe_by_trajectory.append(max(scores, key=lambda item: item[1])[0])
            g0_rank = next(
                row
                for row in d3["rows"]
                if row["dataset"] == "validation"
                and row["scope"] == scope_name
                and row["mask"] == mask
                and row["layer"] == "L5"
                and row["group"] == "G0"
            )["missing_effective_rank"]
            g1_rank = next(
                row
                for row in d3["rows"]
                if row["dataset"] == "validation"
                and row["scope"] == scope_name
                and row["mask"] == mask
                and row["layer"] == "L5"
                and row["group"] == "G1"
            )["missing_effective_rank"]
            rank_by_trajectory.append(float(g1_rank) - float(g0_rank))
        rank_effects[mask] = rank_by_trajectory
        layer_agreement = severe_by_trajectory[0] == severe_by_trajectory[1]
        layer_agreements.append(layer_agreement)
        rows.append(
            {
                "metric": "most_severe_layer",
                "mask": mask,
                "scope": "validation_trajectories",
                "value": d3["first_break"][mask_index - 1]["most_severe_break_layer"],
                "ci_low": "",
                "ci_high": "",
                "trajectory_1_value": severe_by_trajectory[0],
                "trajectory_2_value": severe_by_trajectory[1],
                "trajectory_direction_agreement": layer_agreement,
                "bootstrap_replicates": 0,
            }
        )
        rows.append(
            {
                "metric": "effective_rank_g1_minus_g0",
                "mask": mask,
                "scope": "validation_trajectories",
                "value": float(np.mean(rank_by_trajectory)),
                "ci_low": "",
                "ci_high": "",
                "trajectory_1_value": rank_by_trajectory[0],
                "trajectory_2_value": rank_by_trajectory[1],
                "trajectory_direction_agreement": bool(np.sign(rank_by_trajectory[0]) == np.sign(rank_by_trajectory[1])),
                "bootstrap_replicates": 0,
            }
        )
    utility = d5["features"]["validation"]["modality_margin_utility"].numpy()
    interaction = d5["features"]["validation"]["pairwise_margin_interaction"].numpy()
    trajectory_values = np.asarray(validation["trajectory_id"])
    for modality_index, modality in enumerate(MODALITIES):
        effects = [float(utility[trajectory_values == trajectory, modality_index].mean()) for trajectory in trajectories]
        rows.append(
            {
                "metric": "modality_margin_utility",
                "mask": modality,
                "scope": "validation_trajectories",
                "value": float(utility[:, modality_index].mean()),
                "ci_low": "",
                "ci_high": "",
                "trajectory_1_value": effects[0],
                "trajectory_2_value": effects[1],
                "trajectory_direction_agreement": bool(np.sign(effects[0]) == np.sign(effects[1])),
                "bootstrap_replicates": 0,
            }
        )
    for pair_index, pair in enumerate(d5["features"]["validation"]["pair_names"]):
        effects = [float(interaction[trajectory_values == trajectory, pair_index].mean()) for trajectory in trajectories]
        rows.append(
            {
                "metric": "pairwise_margin_interaction",
                "mask": pair,
                "scope": "validation_trajectories",
                "value": float(interaction[:, pair_index].mean()),
                "ci_low": "",
                "ci_high": "",
                "trajectory_1_value": effects[0],
                "trajectory_2_value": effects[1],
                "trajectory_direction_agreement": bool(np.sign(effects[0]) == np.sign(effects[1])),
                "bootstrap_replicates": 0,
            }
        )
    f2_trajectory = [
        row
        for row in d7["summary_rows"]
        if row["feature_set"] == "F2"
        and row["scope"] == "validation_unified"
        and row["trajectory_id"] != "all"
    ]
    f2_trajectory_agreement = bool(
        f2_trajectory[0]["pr_auc"] > f2_trajectory[0]["prevalence"]
        and f2_trajectory[1]["pr_auc"] > f2_trajectory[1]["prevalence"]
    )
    rows.append(
        {
            "metric": "f2_probe_pr_auc_lift",
            "mask": "all14",
            "scope": "validation_trajectories",
            "value": d7["unified"]["F2"]["pr_auc"] - d7["unified"]["F2"]["prevalence"],
            "ci_low": "",
            "ci_high": "",
            "trajectory_1_value": f2_trajectory[0]["pr_auc"] - f2_trajectory[0]["prevalence"],
            "trajectory_2_value": f2_trajectory[1]["pr_auc"] - f2_trajectory[1]["prevalence"],
            "trajectory_direction_agreement": f2_trajectory_agreement,
            "bootstrap_replicates": 0,
        }
    )
    rows.append(
        {
            "metric": "f2_probe_train_loto_top20_feature_jaccard",
            "mask": "all14",
            "scope": "train_loto",
            "value": d7["loto_top_feature_jaccard"],
            "ci_low": "",
            "ci_high": "",
            "trajectory_1_value": "",
            "trajectory_2_value": "",
            "trajectory_direction_agreement": d7["loto_top_feature_jaccard"]
            >= float(config["stability"]["top_feature_jaccard"]),
            "bootstrap_replicates": 0,
        }
    )
    alpha_agreements = []
    alpha_threshold = float(config["success"]["alpha_star_median_max"])
    for mask in MASK_NAMES[1:]:
        alpha_rows = [
            row
            for row in d8["rows"]
            if row["mask"] == mask
            and row["method"] == "CF1_full"
            and str(row["scope"]).startswith("trajectory:")
        ]
        if len(alpha_rows) != len(trajectories):
            continue
        values = [float(row["alpha_star_median_recovered"]) for row in alpha_rows]
        agreement = (values[0] <= alpha_threshold) == (values[1] <= alpha_threshold)
        alpha_agreements.append(agreement)
        rows.append(
            {
                "metric": "cf1_alpha_star_median",
                "mask": mask,
                "scope": "validation_trajectories",
                "value": float(np.median(values)),
                "ci_low": "",
                "ci_high": "",
                "trajectory_1_value": values[0],
                "trajectory_2_value": values[1],
                "trajectory_direction_agreement": agreement,
                "bootstrap_replicates": 0,
                "pre_registered_small_alpha_threshold": alpha_threshold,
            }
        )
    for row in d7["loto"]:
        rows.append(
            {
                "metric": "f2_probe_train_loto_pr_auc_lift",
                "mask": "all14",
                "scope": row["trajectory_id"],
                "value": row["pr_auc"] - row["prevalence"],
                "ci_low": "",
                "ci_high": "",
                "trajectory_1_value": "",
                "trajectory_2_value": "",
                "trajectory_direction_agreement": row["pr_auc"] > row["prevalence"],
                "bootstrap_replicates": 0,
            }
        )
    parallel_first = np.asarray([parallel_effects[mask][0] for mask in MASK_NAMES[1:]])
    parallel_second = np.asarray([parallel_effects[mask][1] for mask in MASK_NAMES[1:]])
    rank_first = np.asarray([rank_effects[mask][0] for mask in MASK_NAMES[1:]])
    rank_second = np.asarray([rank_effects[mask][1] for mask in MASK_NAMES[1:]])
    parallel_spearman = float(spearmanr(parallel_first, parallel_second).statistic)
    rank_spearman = float(spearmanr(rank_first, rank_second).statistic)
    _write_csv(output / "diagnostics/stability_summary.csv", rows)
    return {
        "rows": rows,
        "parallel_direction_fraction": float(np.mean(np.sign(parallel_first) == np.sign(parallel_second))),
        "parallel_rank_spearman": parallel_spearman,
        "rank_direction_fraction": float(np.mean(np.sign(rank_first) == np.sign(rank_second))),
        "rank_spearman": rank_spearman,
        "layer_agreement_fraction": float(np.mean(layer_agreements)),
        "f2_trajectory_agreement": f2_trajectory_agreement,
        "f2_loto_top_feature_jaccard": d7["loto_top_feature_jaccard"],
        "alpha_threshold_agreement_fraction": float(np.mean(alpha_agreements)) if alpha_agreements else 0.0,
        "train_loto_positive_fraction": float(
            np.mean([row["pr_auc"] > row["prevalence"] for row in d7["loto"]])
        ),
        "paired_bootstrap_replicates": bootstrap_replicates,
    }


def negative_controls(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
    d2: Mapping[str, Any],
    d5: Mapping[str, Any],
    d6: Mapping[str, Any],
    d7: Mapping[str, Any],
    d8: Mapping[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["statistics"]["negative_control_seed"]))
    cache = data["validation"]
    rows: list[dict[str, Any]] = []
    full = normalize(cache["z_raw"][:, 0].float())
    missing = normalize(cache["z_raw"][:, 1:].float())
    true_cosine = float((full[:, None] * missing).sum(dim=2).mean())
    permutation = torch.from_numpy(rng.permutation(len(cache["sample_id"]))).long()
    shuffled_cosine = float((full[permutation, None] * missing).sum(dim=2).mean())
    rows.append(
        {
            "control": "sample_id_full_missing_pair_shuffle",
            "formal": True,
            "real_metric": true_cosine,
            "control_metric": shuffled_cosine,
            "passed": true_cosine > shuffled_cosine,
            "interpretation": "paired cosine must exceed random sample pairing",
        }
    )
    f2 = d7["unified"]["F2"]
    shuffled_labels = d7["validation_labels"][rng.permutation(len(d7["validation_labels"]))]
    label_control = binary_probe_metrics(shuffled_labels, d7["validation_probability"]["F2"])
    rows.append(
        {
            "control": "beam_label_permutation",
            "formal": True,
            "real_metric": f2["roc_auc"],
            "control_metric": label_control["roc_auc"],
            "passed": f2["roc_auc"] > label_control["roc_auc"],
            "interpretation": "F2 hardness signal must exceed permuted labels",
        }
    )
    arrays = _utility_arrays(cache, prepared["validation"], data["topology"].distance.cpu())
    contracts, _ = nonempty_subset_utilities(torch.zeros(1, 15), torch.tensor(list(MASKS.values())))
    real_utility = np.mean(
        [float((arrays["margin"][:, row["union"]] - arrays["margin"][:, row["base"]]).mean()) for row in contracts]
    )
    mask_permutation = torch.tensor([0, *(rng.permutation(14) + 1).tolist()])
    permuted_margin = arrays["margin"][:, mask_permutation]
    permuted_utility = np.mean(
        [float((permuted_margin[:, row["union"]] - permuted_margin[:, row["base"]]).mean()) for row in contracts]
    )
    rows.append(
        {
            "control": "mask_identity_permutation",
            "formal": True,
            "real_metric": real_utility,
            "control_metric": permuted_utility,
            "passed": real_utility > permuted_utility,
            "interpretation": "registered subset relations should have larger marginal utility than permuted mask identities",
        }
    )
    validation_frame = d6["frames"]["validation"]
    base_frame = validation_frame[validation_frame["mask"] == "missing_lidar"].copy()
    real_trajectory_rates = base_frame.groupby("trajectory_id")["target_hard"].mean().to_numpy()
    shuffled_trajectory = base_frame["trajectory_id"].to_numpy().copy()
    rng.shuffle(shuffled_trajectory)
    shuffled_frame = base_frame.assign(trajectory_permuted=shuffled_trajectory)
    shuffled_rates = shuffled_frame.groupby("trajectory_permuted")["target_hard"].mean().to_numpy()
    rows.append(
        {
            "control": "trajectory_identity_permutation",
            "formal": True,
            "real_metric": float(np.std(real_trajectory_rates)),
            "control_metric": float(np.std(shuffled_rates)),
            "passed": float(np.std(real_trajectory_rates)) >= float(np.std(shuffled_rates)),
            "interpretation": "trajectory-specific hardness variation compared with permuted identity",
        }
    )
    geometry_stats = [
        row
        for row in d6["statistics"]
        if row["dataset"] == "validation"
        and row["contrast"] == "G1_vs_G0"
        and row["attribute"] == "full_missing_euclidean"
    ]
    unmatched = float(np.mean([row["standardized_effect_g1_minus_g0"] for row in geometry_stats]))
    matched = float(np.mean([row["matched_beam_trajectory_effect"] for row in geometry_stats]))
    rows.extend(
        [
            {
                "control": "g0_g1_beam_frequency_matching",
                "formal": True,
                "real_metric": unmatched,
                "control_metric": matched,
                "passed": np.sign(unmatched) == np.sign(matched),
                "interpretation": "geometry effect direction after target-beam matching",
            },
            {
                "control": "g0_g1_trajectory_matching",
                "formal": True,
                "real_metric": unmatched,
                "control_metric": matched,
                "passed": np.sign(unmatched) == np.sign(matched),
                "interpretation": "geometry effect direction after trajectory matching",
            },
        ]
    )
    random_bank = normalize(torch.randn(64, 64, generator=torch.Generator().manual_seed(45501)))
    random_identity = decision_decomposition(
        cache["z_raw"][:, 0],
        cache["z_raw"][:, 1],
        cache["shared_bank_logits"][:, 0],
        cache["shared_bank_logits"][:, 1],
        cache["target"],
        random_bank,
    )["production_identity_absolute_error"].mean()
    rows.append(
        {
            "control": "random_prototype_bank",
            "formal": True,
            "real_metric": d2["mean_production_identity_error"],
            "control_metric": float(random_identity),
            "passed": d2["mean_production_identity_error"] < float(random_identity),
            "interpretation": "production-logit residual is smaller for the frozen shared bank than a random bank",
        }
    )
    g1 = prepared["validation"]["groups"].eq(1)
    margin_drop = -prepared["validation"]["geometry"]["delta_margin"][g1].numpy()
    true_projection = -prepared["validation"]["geometry"]["signed_parallel"][g1].numpy()
    delta = (missing - full[:, None])[g1]
    random_direction = normalize(torch.randn(delta.shape, generator=torch.Generator().manual_seed(45502)))
    random_projection = (delta * random_direction).sum(dim=1).numpy()
    true_correlation = float(np.corrcoef(margin_drop, true_projection)[0, 1])
    random_correlation = float(np.corrcoef(margin_drop, random_projection)[0, 1])
    rows.append(
        {
            "control": "random_decision_direction",
            "formal": True,
            "real_metric": abs(true_correlation),
            "control_metric": abs(random_correlation),
            "passed": abs(true_correlation) > abs(random_correlation),
            "interpretation": "prototype decision direction should explain margin drop better than random directions",
        }
    )
    selected = _balanced_sample_indices(cache, 512, seed=45503)
    original_spectrum = representation_spectrum(cache["z_raw"][selected, 1].float())["effective_rank"]
    rotation, _ = torch.linalg.qr(torch.randn(64, 64, generator=torch.Generator().manual_seed(45503)))
    rotated_spectrum = representation_spectrum(cache["z_raw"][selected, 1].float() @ rotation)["effective_rank"]
    rows.append(
        {
            "control": "random_orthogonal_rotation",
            "formal": True,
            "real_metric": original_spectrum,
            "control_metric": rotated_spectrum,
            "passed": abs(original_spectrum - rotated_spectrum) <= 1e-3,
            "interpretation": "rank metric must be invariant to orthogonal coordinates",
        }
    )
    parallel_rows = [row for row in d8["primary"] if row["method"] == "CF2_parallel"]
    parallel_recovery = float(np.mean([row["recoverable_fraction"] for row in parallel_rows]))
    random_recovery = float(np.mean(list(d8["random_recovery_by_mask"].values())))
    rows.append(
        {
            "control": "random_counterfactual_direction",
            "formal": True,
            "real_metric": parallel_recovery,
            "control_metric": random_recovery,
            "passed": parallel_recovery > random_recovery,
            "interpretation": "parallel-only repair compared with 20 random same-energy directions",
        }
    )
    validation_repair = (full[:, None] - missing)[g1]
    oracle_pca = fit_pca_directions(validation_repair, max_components=16)
    oracle_projected = validation_repair @ oracle_pca["basis"] @ oracle_pca["basis"].t()
    oracle_source = missing[g1]
    oracle_labels = cache["target"][:, None].expand(-1, 14)[g1]
    oracle_recovery = float(
        (normalize(oracle_source + oracle_projected) @ normalize(data["prototypes"]).t())
        .argmax(dim=1)
        .eq(oracle_labels)
        .float()
        .mean()
    )
    rows.append(
        {
            "control": "validation_selected_direction_illegal_oracle",
            "formal": False,
            "validation_leakage_oracle": True,
            "real_metric": oracle_recovery,
            "control_metric": "excluded",
            "passed": True,
            "interpretation": "sanity-check only; never enters route selection or formal conclusions",
        }
    )
    _write_csv(output / "diagnostics/negative_controls.csv", rows)
    return {
        "rows": rows,
        "formal_pass_fraction": float(np.mean([bool(row["passed"]) for row in rows if row["formal"]])),
        "all_formal_passed": all(bool(row["passed"]) for row in rows if row["formal"]),
        "validation_leakage_oracle_excluded": True,
    }


def select_route(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
    d0: Mapping[str, Any],
    d2: Mapping[str, Any],
    d3: Mapping[str, Any],
    d4: Mapping[str, Any],
    d5: Mapping[str, Any],
    d6: Mapping[str, Any],
    d7: Mapping[str, Any],
    d8: Mapping[str, Any],
    d10: Mapping[str, Any],
    stability: Mapping[str, Any],
) -> dict[str, Any]:
    validation_geometry = prepared["validation"]["geometry"]
    groups = prepared["validation"]["groups"]
    g1 = groups.eq(1)
    g0 = groups.eq(0)
    energy_difference = float(
        validation_geometry["parallel_energy_ratio"][g1].mean()
        - validation_geometry["parallel_energy_ratio"][g0].mean()
    )
    parallel_correlation = float(
        np.mean([abs(float(row["margin_drop_parallel_pearson"])) for row in d2["validation_g1"]])
    )
    parallel_recovery = float(
        np.mean([row["recoverable_fraction"] for row in d8["primary"] if row["method"] == "CF2_parallel"])
    )
    random_recovery = float(np.mean(list(d8["random_recovery_by_mask"].values())))
    full_alpha = [
        float(row["alpha_star_median_recovered"])
        for row in d8["primary"]
        if row["method"] == "CF1_full" and row["alpha_star_median_recovered"] != ""
    ]
    alpha_median = float(np.median(full_alpha))
    valid_layer_ids = {f"L{index}" for index in range(7)}
    first_layers = Counter(
        row["first_break_layer"]
        for row in d3["first_break"]
        if row["first_break_layer"] in valid_layer_ids
    )
    dominant_first_layer, dominant_first_count = first_layers.most_common(1)[0] if first_layers else ("none", 0)
    validation_rank = d4["validation_final"]
    rank_lower_masks = 0
    for mask in MASK_NAMES[1:]:
        g0_row = next(row for row in validation_rank if row["mask"] == mask and row["group"] == "G0")
        g1_row = next(row for row in validation_rank if row["mask"] == mask and row["group"] == "G1")
        rank_lower_masks += int(float(g1_row["missing_effective_rank"]) < float(g0_row["missing_effective_rank"]))
    best_pca_k = int(config["success"]["route_pca_k"])
    selected_pca_rows = [row for row in d8["primary"] if row["method"] == f"CF4_pca_k{best_pca_k}"]
    selected_pca_recovery = float(np.mean([row["recoverable_fraction"] for row in selected_pca_rows]))
    validation_stats = [
        row
        for row in d6["statistics"]
        if row["dataset"] == "validation" and row["contrast"] == "G1_vs_G0"
    ]
    motion_rows = [
        row
        for row in validation_stats
        if row["attribute"] in {"angular_velocity_mean", "beam_transition_cycle_distance"}
    ]
    motion_positive_masks = sum(
        row["standardized_effect_g1_minus_g0"] > 0 and row["fdr_significant_0_05"] for row in motion_rows
    )
    temporal_g1 = [
        row
        for row in d10["rows"]
        if row["analysis"] == "frozen_fusion_temporal_ablation"
        and row["scope"] == "all"
        and row["group"] == "G1"
        and row["ablation"] != "T0_original"
    ]
    temporal_support = float(np.mean([row["top1_delta_vs_t0"] for row in temporal_g1])) < -0.01
    utility_rows = [
        row
        for row in d5["marginal_rows"]
        if row["dataset"] == "validation" and row["metric"] == "margin"
    ]
    validation_trajectory = np.asarray(data["validation"]["trajectory_id"])
    trajectory_ids = sorted(set(validation_trajectory.tolist()))
    modality_utility = d5["features"]["validation"]["modality_margin_utility"].numpy()
    stable_negative_modalities = [
        modality
        for modality_index, modality in enumerate(MODALITIES)
        if all(
            float(modality_utility[validation_trajectory == trajectory, modality_index].mean()) < 0
            for trajectory in trajectory_ids
        )
    ]
    stable_positive_modalities = [
        modality
        for modality_index, modality in enumerate(MODALITIES)
        if all(
            float(modality_utility[validation_trajectory == trajectory, modality_index].mean()) > 0
            for trajectory in trajectory_ids
        )
    ]
    negative_utility_structure = bool(stable_negative_modalities and stable_positive_modalities) and any(
        row["negative_fraction"] > 0.5 for row in utility_rows
    )
    f2 = d7["unified"]["F2"]
    f2_lift = float(f2["pr_auc"] - f2["prevalence"])
    f2_recall_lift = float(f2["recall_at_20pct"] - 0.2)
    ambiguity_rows = [
        row for row in validation_stats if row["attribute"] == "future_power_top1_top2_gap"
    ]
    ambiguity_masks = sum(
        row["standardized_effect_g1_minus_g0"] < 0 and row["fdr_significant_0_05"] for row in ambiguity_rows
    )
    validation_group_rows = [
        row
        for row in d0["group_rows"]["mask"]
        if row["dataset"] == "validation" and row["group"] == "G2"
    ]
    mean_g2_rate = float(np.mean([row["rate"] for row in validation_group_rows]))
    chain_a = {
        "parallel_energy_g1_gt_g0": energy_difference > 0,
        "parallel_explains_margin": parallel_correlation >= 0.5,
        "parallel_recovery_beats_random": parallel_recovery >= random_recovery + 0.05,
        "alpha_median_le_0_25": alpha_median <= float(config["success"]["alpha_star_median_max"]),
        "two_validation_trajectories_agree": stability["parallel_direction_fraction"]
        >= float(config["stability"]["trajectory_direction_fraction"]),
    }
    chain_b = {
        "same_first_break_layer": dominant_first_layer != "none",
        "stable_in_at_least_10_masks": dominant_first_count >= int(config["success"]["fusion_mask_count_min"]),
        "negative_modality_utility": negative_utility_structure,
        "two_validation_trajectories_agree": stability["layer_agreement_fraction"]
        >= float(config["stability"]["trajectory_direction_fraction"]),
    }
    chain_c = {
        "g1_rank_lower_than_g0": rank_lower_masks >= 10,
        "topk_beats_random": selected_pca_recovery >= random_recovery + 0.05,
        "k_le_16": best_pca_k <= int(config["success"]["low_rank_k_max"]),
        "directions_stable": stability["rank_direction_fraction"]
        >= float(config["stability"]["trajectory_direction_fraction"]),
        "worst_mask_in_low_rank_set": d0["worst_mask"]
        in {
            row["mask"]
            for row in validation_rank
            if row["group"] == "G1" and float(row["effective_rank_delta"]) < 0
        },
    }
    chain_d = {
        "high_motion_enrichment": motion_positive_masks >= 10,
        "effect_stable": stability["parallel_direction_fraction"]
        >= float(config["stability"]["trajectory_direction_fraction"]),
        "temporal_ablation_support": temporal_support,
        "beam_topology_motion_support": sum(
            row["standardized_effect_g1_minus_g0"] > 0
            for row in motion_rows
            if row["attribute"] == "beam_transition_cycle_distance"
        )
        >= 10,
        "two_validation_trajectories_agree": stability["parallel_direction_fraction"]
        >= float(config["stability"]["trajectory_direction_fraction"]),
    }
    chain_e = {
        "stable_positive_negative_utility": negative_utility_structure,
        "f2_pr_auc_lift": f2_lift >= float(config["success"]["f2_pr_auc_lift_min"]),
        "top20_recall_lift": f2_recall_lift >= float(config["success"]["top20_recall_lift_min"]),
        "two_validation_trajectories_agree": bool(stability["f2_trajectory_agreement"]),
        "no_full_or_future_features": True,
    }
    chains = {"A": chain_a, "B": chain_b, "C": chain_c, "D": chain_d, "E": chain_e}
    passed = {name: all(values.values()) for name, values in chains.items()}
    intrinsic_ambiguity = ambiguity_masks >= 10 and mean_g2_rate >= 0.2 and alpha_median > 0.25
    if passed["A"]:
        route = "R1"
    elif passed["B"]:
        route = "R2"
    elif passed["C"]:
        route = "R3"
    elif passed["D"]:
        route = "R4"
    elif passed["E"]:
        route = "R5"
    elif intrinsic_ambiguity:
        route = "R6"
    else:
        route = "R0"
    recommendations = {
        "R0": "停止设计复杂缺失模块，优先复核数据、标签可预测性与任务协议。",
        "R1": "Full-guided Topology-Aware Hard-Boundary Distillation",
        "R2": "Hard-Sample Fusion Deinterference",
        "R3": "Prototype-Subspace Information Preservation",
        "R4": "Motion-Conditioned Topological Hard-Sample Learning",
        "R5": "Sample-Specific Modality Sufficiency Learning",
        "R6": "困难主要来自标签模糊；如重定义问题，只考虑 top-k soft target 或 beam-gain 优化。",
    }
    return {
        "route": route,
        "recommendation": recommendations[route],
        "chains": chains,
        "chain_passed": passed,
        "any_complete_chain": any(passed.values()),
        "intrinsic_ambiguity_rule": intrinsic_ambiguity,
        "diagnostics": {
            "parallel_energy_difference": energy_difference,
            "parallel_margin_correlation": parallel_correlation,
            "parallel_recovery": parallel_recovery,
            "random_recovery": random_recovery,
            "full_alpha_median": alpha_median,
            "dominant_first_layer": dominant_first_layer,
            "dominant_first_layer_mask_count": dominant_first_count,
            "rank_lower_mask_count": rank_lower_masks,
            "best_pca_k": best_pca_k,
            "best_pca_recovery": selected_pca_recovery,
            "pca_k_selection": "pre_registered train-only K=16 for route; other K are sensitivity only",
            "motion_positive_mask_attribute_count": motion_positive_masks,
            "f2_pr_auc_lift": f2_lift,
            "f2_recall20_lift": f2_recall_lift,
            "ambiguity_mask_count": ambiguity_masks,
            "mean_g2_rate": mean_g2_rate,
            "stable_negative_modalities": stable_negative_modalities,
            "stable_positive_modalities": stable_positive_modalities,
        },
    }


def _save_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def create_figures(
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
    d0: Mapping[str, Any],
    d2: Mapping[str, Any],
    d3: Mapping[str, Any],
    d4: Mapping[str, Any],
    d5: Mapping[str, Any],
    d7: Mapping[str, Any],
    d8: Mapping[str, Any],
    d10: Mapping[str, Any],
) -> None:
    validation = data["validation"]
    group_rows = [row for row in d0["group_rows"]["mask"] if row["dataset"] == "validation"]
    figure, axis = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(14)
    for group in GROUP_NAMES:
        values = np.asarray([row["rate"] for row in group_rows if row["group"] == group])
        axis.bar(np.arange(14), values, bottom=bottom, label=group)
        bottom += values
    axis.set_xticks(np.arange(14), MASK_NAMES[1:], rotation=60, ha="right")
    axis.set_ylabel("pair fraction")
    axis.legend(ncol=4)
    _save_figure(figure, output / "figures/group_distribution_by_mask.png")
    figure, axes = plt.subplots(3, 5, figsize=(16, 9), sharex=True, sharey=True)
    distance = prepared["validation"]["cycle_error"]
    for mask_index, (axis, mask) in enumerate(zip(axes.flat, MASK_NAMES), 0):
        axis.hist(distance[:, mask_index].numpy(), bins=np.arange(34) - 0.5, color="#35618f")
        axis.set_title(mask, fontsize=8)
    _save_figure(figure, output / "figures/topology_error_histograms.png")
    validation_d2 = [
        row
        for row in d2["rows"]
        if row["dataset"] == "validation" and row["scope"] == "all" and row["group"] in {"G0", "G1"}
    ]
    figure, axis = plt.subplots(figsize=(12, 5))
    x = np.arange(14)
    for offset, group in ((-0.18, "G0"), (0.18, "G1")):
        values = [row["signed_parallel_mean"] for row in validation_d2 if row["group"] == group]
        orthogonal = [row["orthogonal_norm_mean"] for row in validation_d2 if row["group"] == group]
        axis.bar(x + offset, values, width=0.34, label=f"{group} parallel")
        axis.plot(x + offset, orthogonal, marker="o", linestyle="none", label=f"{group} orthogonal")
    axis.set_xticks(x, MASK_NAMES[1:], rotation=60, ha="right")
    axis.legend(ncol=2)
    _save_figure(figure, output / "figures/parallel_vs_orthogonal_by_group.png")
    figure, axis = plt.subplots(figsize=(12, 5))
    for group in ("G0", "G1"):
        values = []
        for layer in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
            selected = [
                row["paired_cosine_mean"]
                for row in d3["rows"]
                if row["dataset"] == "validation"
                and row["scope"] == "all"
                and row["layer"] == layer
                and row["group"] == group
            ]
            values.append(np.mean(selected))
        axis.plot(range(7), values, marker="o", label=group)
    axis.set_xticks(range(7), ("L0", "L1", "L2", "L3", "L4", "L5", "L6"))
    axis.set_ylabel("paired cosine")
    axis.legend()
    _save_figure(figure, output / "figures/layerwise_g0_g1_cosine.png")
    figure, axis = plt.subplots(figsize=(12, 5))
    for group in ("G0", "G1"):
        values = []
        for layer in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
            selected = [
                row["missing_effective_rank"]
                for row in d4["rows"]
                if row["dataset"] == "validation"
                and row["scope"] == "all"
                and row["layer"] == layer
                and row["group"] == group
            ]
            values.append(np.mean(selected))
        axis.plot(range(7), values, marker="o", label=group)
    axis.set_xticks(range(7), ("L0", "L1", "L2", "L3", "L4", "L5", "L6"))
    axis.set_ylabel("effective rank")
    axis.legend()
    _save_figure(figure, output / "figures/layerwise_rank.png")
    figure, axis = plt.subplots(figsize=(12, 5))
    for group in ("G0", "G1"):
        values = []
        for layer in ("L0", "L1", "L2", "L3", "L4", "L5", "L6"):
            selected = [
                row["full_missing_neighborhood_preservation"]
                for row in d3["rows"]
                if row["dataset"] == "validation"
                and row["scope"] == "all"
                and row["layer"] == layer
                and row["group"] == group
            ]
            values.append(np.mean(selected))
        axis.plot(range(7), values, marker="o", label=group)
    axis.set_xticks(range(7), ("L0", "L1", "L2", "L3", "L4", "L5", "L6"))
    axis.set_ylabel("neighborhood preservation")
    axis.legend()
    _save_figure(figure, output / "figures/layerwise_neighborhood_preservation.png")
    utility = np.zeros((4, 15), dtype=np.float64)
    for modality_index, modality in enumerate(MODALITIES):
        rows = [
            row
            for row in d5["marginal_rows"]
            if row["dataset"] == "validation" and row["metric"] == "margin" and row["modality"] == modality
        ]
        for row in rows:
            utility[modality_index, MASK_NAMES.index(row["base_mask"])] = row["utility_mean"]
    figure, axis = plt.subplots(figsize=(12, 3))
    image = axis.imshow(utility, aspect="auto", cmap="coolwarm")
    axis.set_yticks(range(4), MODALITIES)
    axis.set_xticks(range(15), MASK_NAMES, rotation=60, ha="right")
    figure.colorbar(image, ax=axis, label="margin utility")
    _save_figure(figure, output / "figures/modality_utility_heatmap.png")
    interaction_matrix = np.zeros((4, 4), dtype=np.float64)
    for left, right in combinations(range(4), 2):
        values = [
            row["interaction_mean"]
            for row in d5["interaction_rows"]
            if row["dataset"] == "validation"
            and row["metric"] == "margin"
            and row["modality_left"] == MODALITIES[left]
            and row["modality_right"] == MODALITIES[right]
        ]
        interaction_matrix[left, right] = interaction_matrix[right, left] = np.mean(values)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(interaction_matrix, cmap="coolwarm")
    axis.set_xticks(range(4), MODALITIES, rotation=30)
    axis.set_yticks(range(4), MODALITIES)
    figure.colorbar(image, ax=axis, label="pair interaction")
    _save_figure(figure, output / "figures/pairwise_interaction_heatmap.png")
    full_rows = [row for row in d8["primary"] if row["method"] == "CF1_full"]
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.bar(range(14), [row["alpha_star_median_recovered"] for row in full_rows])
    axis.set_xticks(range(14), MASK_NAMES[1:], rotation=60, ha="right")
    axis.set_ylabel("median alpha star")
    _save_figure(figure, output / "figures/alpha_star_by_mask.png")
    parallel_rows = [row for row in d8["primary"] if row["method"] == "CF2_parallel"]
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.bar(range(14), [row["recoverable_fraction"] for row in parallel_rows])
    axis.set_xticks(range(14), MASK_NAMES[1:], rotation=60, ha="right")
    axis.set_ylabel("parallel recovery")
    _save_figure(figure, output / "figures/parallel_repair_rate.png")
    pca_methods = [f"CF4_pca_k{k}" for k in (1, 2, 4, 8, 16, 32)]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(
        (1, 2, 4, 8, 16, 32),
        [np.mean([row["recoverable_fraction"] for row in d8["primary"] if row["method"] == method]) for method in pca_methods],
        marker="o",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks((1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32))
    axis.set_ylabel("top-K recovery")
    _save_figure(figure, output / "figures/topk_repair_rate.png")
    validation_rank = [
        row
        for row in d4["validation_final"]
        if row["layer"] == "L5" and row["scope"] == "all"
    ]
    figure, axis = plt.subplots(figsize=(12, 4))
    for offset, group in ((-0.18, "G0"), (0.18, "G1")):
        axis.bar(
            np.arange(14) + offset,
            [row["missing_effective_rank"] for row in validation_rank if row["group"] == group],
            width=0.34,
            label=group,
        )
    axis.set_xticks(range(14), MASK_NAMES[1:], rotation=60, ha="right")
    axis.legend()
    _save_figure(figure, output / "figures/effective_rank_by_mask.png")
    labels = d7["validation_labels"]
    figure, axis = plt.subplots(figsize=(6, 5))
    for feature_set in ("F0", "F1", "F2", "F3"):
        probability = d7["validation_probability"][feature_set]
        order = np.argsort(-probability)
        recall = np.cumsum(labels[order]) / labels.sum()
        precision = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
        axis.plot(recall, precision, label=feature_set)
    axis.axhline(labels.mean(), color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.legend()
    _save_figure(figure, output / "figures/hardness_probe_pr_curve.png")
    temporal = [
        row
        for row in d10["rows"]
        if row["analysis"] == "frozen_fusion_temporal_ablation"
        and row["scope"] == "all"
        and row["group"] == "G1"
    ]
    figure, axis = plt.subplots(figsize=(8, 4))
    values = [np.mean([row["top1_delta_vs_t0"] for row in temporal if row["ablation"] == name]) for name in TEMPORAL_NAMES]
    axis.bar(range(6), values)
    axis.set_xticks(range(6), TEMPORAL_NAMES, rotation=30, ha="right")
    axis.set_ylabel("G1 top1 delta vs T0")
    _save_figure(figure, output / "figures/temporal_difficulty.png")
    sample = _balanced_sample_indices(validation, 5000, seed=45601)
    g1_sample = prepared["validation"]["groups"][sample].eq(1)
    margin_drop = -prepared["validation"]["geometry"]["delta_margin"][sample][g1_sample]
    projection = -prepared["validation"]["geometry"]["signed_parallel"][sample][g1_sample]
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(projection.numpy(), margin_drop.numpy(), s=3, alpha=0.2)
    axis.set_xlabel("decision-parallel repair magnitude")
    axis.set_ylabel("margin drop")
    _save_figure(figure, output / "figures/margin_drop_vs_parallel_projection.png")
    figure, axis = plt.subplots(figsize=(12, 4))
    margin_g0 = [
        np.mean(
            prepared["validation"]["geometry"]["delta_margin"][:, index][
                prepared["validation"]["groups"][:, index].eq(group_id)
            ].numpy()
        )
        for group_id in (0, 1)
        for index in range(14)
    ]
    axis.plot(range(14), margin_g0[:14], marker="o", label="G0")
    axis.plot(range(14), margin_g0[14:], marker="o", label="G1")
    axis.set_xticks(range(14), MASK_NAMES[1:], rotation=60, ha="right")
    axis.set_ylabel("margin delta")
    axis.legend()
    _save_figure(figure, output / "figures/margin_drop_by_group.png")
    for mask_index, mask in enumerate(MASK_NAMES):
        prediction = prepared["validation"]["stats"]["prediction"][:, mask_index]
        encoded = validation["target"] * 64 + prediction
        confusion = torch.bincount(encoded, minlength=4096).view(64, 64).numpy()
        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(np.log1p(confusion), cmap="viridis", aspect="auto")
        axis.set_xlabel("prediction")
        axis.set_ylabel("target")
        axis.set_title(mask)
        figure.colorbar(image, ax=axis, label="log(1+count)")
        _save_figure(figure, output / f"figures/confusion_{mask}.png")
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(prepared["validation"]["cycle_error"][:, mask_index].numpy(), bins=np.arange(34) - 0.5)
        axis.set_xlabel("cycle distance")
        axis.set_ylabel("count")
        axis.set_title(mask)
        _save_figure(figure, output / f"figures/topology_error_hist_{mask}.png")
    selected = d3["sample_indices"]["validation"]
    final_values = validation["z_raw"][selected]
    selected_groups = prepared["validation"]["groups"][selected]
    for mask_index, mask in enumerate(MASK_NAMES[1:], 1):
        figure, axis = plt.subplots(figsize=(6, 4))
        for group_id, group in enumerate(("G0", "G1")):
            chosen = selected_groups[:, mask_index - 1].eq(group_id)
            spectrum = representation_spectrum(final_values[chosen, mask_index])["eigenvalues"].numpy()
            axis.plot(np.arange(1, min(65, len(spectrum) + 1)), spectrum[:64], label=group)
        axis.set_yscale("log")
        axis.set_xlabel("component")
        axis.set_ylabel("covariance eigenvalue")
        axis.legend()
        _save_figure(figure, output / f"figures/eigenspectrum_g0_g1_{mask}.png")


def _attribute_effect(d6: Mapping[str, Any], attribute: str) -> tuple[float, int]:
    rows = [
        row
        for row in d6["statistics"]
        if row["dataset"] == "validation"
        and row["contrast"] == "G1_vs_G0"
        and row["attribute"] == attribute
    ]
    return float(np.mean([row["standardized_effect_g1_minus_g0"] for row in rows])), sum(
        bool(row["fdr_significant_0_05"]) for row in rows
    )


def finalize(
    config: Mapping[str, Any],
    output: Path,
    data: Mapping[str, Any],
    prepared: Mapping[str, Any],
    selection: Mapping[str, Any],
    d0: Mapping[str, Any],
    d2: Mapping[str, Any],
    d3: Mapping[str, Any],
    d4: Mapping[str, Any],
    d5: Mapping[str, Any],
    d6: Mapping[str, Any],
    d7: Mapping[str, Any],
    d8: Mapping[str, Any],
    d10: Mapping[str, Any],
    stability: Mapping[str, Any],
    controls: Mapping[str, Any],
    route: Mapping[str, Any],
) -> dict[str, Any]:
    validation_groups = [
        row for row in d0["group_rows"]["mask"] if row["dataset"] == "validation"
    ]
    group_by_mask = {
        mask: {
            group: next(
                row for row in validation_groups if row["mask"] == mask and row["group"] == group
            )
            for group in GROUP_NAMES
        }
        for mask in MASK_NAMES[1:]
    }
    highest_g1 = max(MASK_NAMES[1:], key=lambda mask: group_by_mask[mask]["G1"]["rate"])
    total_g1 = sum(group_by_mask[mask]["G1"]["count"] for mask in MASK_NAMES[1:])
    near_fraction = sum(
        group_by_mask[mask]["G1"]["count"] * group_by_mask[mask]["G1"]["near_fraction"]
        for mask in MASK_NAMES[1:]
    ) / max(1, total_g1)
    full_margin_g1 = prepared["validation"]["stats"]["target_margin"][:, 0, None].expand(-1, 14)[
        prepared["validation"]["groups"].eq(1)
    ]
    geometry_mask = prepared["validation"]["groups"].eq(1)
    g1_geometry = {
        metric: float(values[geometry_mask].mean())
        for metric, values in prepared["validation"]["geometry"].items()
        if torch.is_tensor(values) and values.ndim == 2 and values.dtype.is_floating_point
    }
    d2_g1 = d2["validation_g1"]
    valid_layer_ids = {f"L{index}" for index in range(7)}
    estimable_break_rows = [
        row
        for row in d3["first_break"]
        if row["first_break_layer"] != "not_estimable_no_train_g1"
    ]
    detected_break_rows = [
        row for row in estimable_break_rows if row["first_break_layer"] in valid_layer_ids
    ]
    first_counter = Counter(row["first_break_layer"] for row in detected_break_rows)
    severe_counter = Counter(
        row["most_severe_break_layer"]
        for row in estimable_break_rows
        if row["most_severe_break_layer"] in valid_layer_ids
    )
    first_layer = first_counter.most_common(1)[0] if first_counter else ("none", 0)
    severe_layer = severe_counter.most_common(1)[0] if severe_counter else ("none", 0)
    unavailable_break_count = len(d3["first_break"]) - len(estimable_break_rows)
    no_break_count = len(estimable_break_rows) - len(detected_break_rows)
    validation_utility = [
        row
        for row in d5["marginal_rows"]
        if row["dataset"] == "validation" and row["metric"] == "margin"
    ]
    modality_means = {
        modality: float(
            np.mean([row["utility_mean"] for row in validation_utility if row["modality"] == modality])
        )
        for modality in MODALITIES
    }
    modality_negative = {
        modality: float(
            np.mean([row["negative_fraction"] for row in validation_utility if row["modality"] == modality])
        )
        for modality in MODALITIES
    }
    interaction_means: dict[str, float] = {}
    for left, right in combinations(MODALITIES, 2):
        values = [
            row["interaction_mean"]
            for row in d5["interaction_rows"]
            if row["dataset"] == "validation"
            and row["metric"] == "margin"
            and row["modality_left"] == left
            and row["modality_right"] == right
        ]
        interaction_means[f"{left}+{right}"] = float(np.mean(values))
    effects = {
        name: _attribute_effect(d6, attribute)
        for name, attribute in {
            "beam_frequency": "beam_class_frequency_train",
            "speed": "speed_mean",
            "angular_velocity": "angular_velocity_mean",
            "curvature": "trajectory_curvature_mean",
            "beam_transition": "beam_transition_cycle_distance",
            "ambiguity_gap": "future_power_top1_top2_gap",
        }.items()
    }
    domain_rows = [
        row
        for row in d0["group_rows"]["domain"]
        if row["dataset"] == "validation" and row["group"] == "G1"
    ]
    domain_range = min(row["rate"] for row in domain_rows), max(row["rate"] for row in domain_rows)
    f2, f3 = d7["unified"]["F2"], d7["unified"]["F3"]
    repairs = {
        method: [row for row in d8["primary"] if row["method"] == method]
        for method in ("CF1_full", "CF2_parallel", "CF4_pca_k16")
    }
    neighborhood = {
        layer: float(
            np.mean(
                [
                    row["full_missing_neighborhood_preservation"]
                    for row in d3["rows"]
                    if row["dataset"] == "validation"
                    and row["scope"] == "all"
                    and row["layer"] == layer
                    and row["group"] == "G1"
                ]
            )
        )
        for layer in ("L0", "L1", "L2", "L3", "L4", "L5", "L6")
    }
    pairs = list(zip(tuple(neighborhood)[:-1], tuple(neighborhood)[1:]))
    largest_neighbor_drop = max(pairs, key=lambda pair: neighborhood[pair[0]] - neighborhood[pair[1]])
    temporal_rows = [
        row
        for row in d10["rows"]
        if row["analysis"] == "frozen_fusion_temporal_ablation"
        and row["scope"] == "all"
        and row["group"] == "G1"
        and row["ablation"] != "T0_original"
    ]
    temporal_effects = {
        name: float(np.mean([row["top1_delta_vs_t0"] for row in temporal_rows if row["ablation"] == name]))
        for name in TEMPORAL_NAMES[1:]
    }
    worst_temporal = max(temporal_effects, key=lambda name: abs(temporal_effects[name]))
    rank_rows = [row for row in d4["validation_final"] if row["group"] == "G1"]
    most_collapsed = min(rank_rows, key=lambda row: row["effective_rank_delta"])["mask"]
    summary = {
        "diagnostic_id": config["diagnostic_id"],
        "status": "passed",
        "route": route,
        "baseline": {
            "full_top1": d0["full_top1"],
            "all14_top1": d0["all14_top1"],
            "worst_mask": d0["worst_mask"],
            "worst_top1": d0["worst_top1"],
        },
        "group_by_mask": group_by_mask,
        "g1": {
            "highest_mask": highest_g1,
            "near_fraction": near_fraction,
            "full_margin": _finite_summary(full_margin_g1),
            "geometry": g1_geometry,
        },
        "layers": {
            "dominant_first_break": first_layer,
            "dominant_most_severe": severe_layer,
            "estimable_mask_count": len(estimable_break_rows),
            "no_detected_break_mask_count": no_break_count,
            "not_estimable_no_train_g1_mask_count": unavailable_break_count,
            "rank_lower_mask_count": route["diagnostics"]["rank_lower_mask_count"],
            "per_frame_fusion": "not_present_in_frozen_m4",
            "temporal_module": "not_present_in_frozen_m4",
        },
        "modality_mean_utility": modality_means,
        "modality_negative_fraction": modality_negative,
        "pairwise_interaction": interaction_means,
        "probe": {"F2": f2, "F3": f3},
        "stability": stability,
        "negative_controls": controls,
        "selection": selection,
        "safety": {
            "csi_used": False,
            "channel_input_used": False,
            "f1_used": False,
            "future_leakage": False,
            "outer_test_accessed": False,
        },
    }
    context = {
        "group_by_mask": group_by_mask,
        "highest_g1": highest_g1,
        "near_fraction": near_fraction,
        "full_margin_g1": full_margin_g1,
        "g1_geometry": g1_geometry,
        "d2_g1": d2_g1,
        "first_layer": first_layer,
        "severe_layer": severe_layer,
        "estimable_break_count": len(estimable_break_rows),
        "no_break_count": no_break_count,
        "unavailable_break_count": unavailable_break_count,
        "most_collapsed": most_collapsed,
        "modality_means": modality_means,
        "modality_negative": modality_negative,
        "interaction_means": interaction_means,
        "effects": effects,
        "domain_range": domain_range,
        "f2": f2,
        "f3": f3,
        "repairs": repairs,
        "neighborhood": neighborhood,
        "largest_neighbor_drop": largest_neighbor_drop,
        "temporal_effects": temporal_effects,
        "worst_temporal": worst_temporal,
    }
    _write_final_report(output, d0, d2, d4, d8, stability, controls, route, selection, context)
    return summary


def _write_final_report(
    output: Path,
    d0: Mapping[str, Any],
    d2: Mapping[str, Any],
    d4: Mapping[str, Any],
    d8: Mapping[str, Any],
    stability: Mapping[str, Any],
    controls: Mapping[str, Any],
    route: Mapping[str, Any],
    selection: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    groups = context["group_by_mask"]
    table = ["| Mask | G0 | G1 | G2 | G3 |", "|---|---:|---:|---:|---:|"]
    for mask in MASK_NAMES[1:]:
        values = " | ".join(f"{100 * groups[mask][group]['rate']:.2f}%" for group in GROUP_NAMES)
        table.append(f"| {mask} | {values} |")
    d2_g1 = context["d2_g1"]
    repairs = context["repairs"]
    full_alpha = np.median(
        [row["alpha_star_median_recovered"] for row in repairs["CF1_full"] if row["alpha_star_median_recovered"] != ""]
    )
    parallel_recovery = np.mean([row["recoverable_fraction"] for row in repairs["CF2_parallel"]])
    pca_recovery = np.mean([row["recoverable_fraction"] for row in repairs["CF4_pca_k16"]])
    random_recovery = np.mean(list(d8["random_recovery_by_mask"].values()))
    modality_means = context["modality_means"]
    modality_negative = context["modality_negative"]
    interactions = context["interaction_means"]
    effects = context["effects"]
    f2, f3 = context["f2"], context["f3"]
    first_layer, severe_layer = context["first_layer"], context["severe_layer"]
    neighbor_pair = context["largest_neighbor_drop"]
    neighborhood = context["neighborhood"]
    worst_temporal = context["worst_temporal"]
    temporal_effect = context["temporal_effects"][worst_temporal]
    answers = [
        "每个 mask 的 G0/G1/G2/G3 比例见下表；分母均为该 mask 的全部 validation paired samples。",
        f"G1 比例最高的是 `{context['highest_g1']}`，比例为 {100*groups[context['highest_g1']]['G1']['rate']:.2f}%。",
        f"全部 G1 中 near（cycle distance <=3）占 {100*context['near_fraction']:.2f}%，far 占 {100*(1-context['near_fraction']):.2f}%。",
        f"G1 的 Full margin 均值/中位数为 {float(context['full_margin_g1'].mean()):.4f}/{float(context['full_margin_g1'].median()):.4f}；train-only low/high 阈值为 {selection['low_margin_threshold']:.4f}/{selection['high_margin_threshold']:.4f}。",
        f"G1 normalized Euclidean drift 均值为 {context['g1_geometry']['euclidean_distance']:.4f}，Full 插值 alpha 中位数跨 mask 为 {full_alpha:.3f}；完整链 A={'通过' if route['chain_passed']['A'] else '未通过'}，因此不凭位移均值单独声称小幅边界漂移。",
        f"G1 平均 |parallel|={np.mean([row['absolute_parallel_mean'] for row in d2_g1]):.4f}，orthogonal norm={np.mean([row['orthogonal_norm_mean'] for row in d2_g1]):.4f}，parallel energy ratio={np.mean([row['parallel_energy_ratio_mean'] for row in d2_g1]):.4f}。",
        f"parallel 与 margin drop 的跨 mask |Pearson| 均值为 {np.mean([abs(row['margin_drop_parallel_pearson']) for row in d2_g1]):.4f}；统一 FP32 scoring 的决策恒等式最大误差为 {d2['max_identity_error']:.3e}，生产 autocast logits 相对该几何的最大残差为 {d2['max_production_identity_error']:.3e}。",
        f"train-only 首次 break layer 众数是 `{first_layer[0]}`（{first_layer[1]}/{context['estimable_break_count']} 个可估计 mask）；其中 {context['no_break_count']} 个未越过预注册 break 阈值，另有 {context['unavailable_break_count']} 个因 train G1=0 而不可估计。冻结 M4 不存在独立 per-frame fusion 或 temporal module。",
        f"train-only 最严重 break layer 众数是 `{severe_layer[0]}`（{severe_layer[1]}/{context['estimable_break_count']} 个可估计 mask）；不可估计项不进入众数或路线 B。",
        f"validation L5 中 G1 effective rank 低于 G0 的 mask 为 {route['diagnostics']['rank_lower_mask_count']}/14。",
        f"低秩坍缩链 C={'通过' if route['chain_passed']['C'] else '未通过'}；只有 rank、K16 repair、随机对照和双轨迹稳定同时通过才称为坍缩。",
        f"按 validation L5 G1 effective-rank delta，最明显的 mask 是 `{context['most_collapsed']}`；全部 layer/mask/group 见 `d4_representation_collapse.csv`。",
        f"平均 margin 边际价值最高的模态是 `{max(modality_means, key=modality_means.get)}`，utility={max(modality_means.values()):+.4f}。",
        f"负贡献最频繁的模态是 `{max(modality_negative, key=modality_negative.get)}`，负贡献比例={100*max(modality_negative.values()):.2f}%。",
        f"平均协同最大的组合是 `{max(interactions, key=interactions.get)}`（{max(interactions.values()):+.4f}）；是否跨轨迹稳定以 `stability_summary.csv` 为准。",
        f"平均干扰最大的组合是 `{min(interactions, key=interactions.get)}`（{min(interactions.values()):+.4f}）；单一合并均值不作为稳定机制证据。",
        "困难性随 target beam 改变；逐 beam 的类别频率归一化 G1 rate 已输出，结论同时受 beam-frequency matching control 约束。",
        f"train class-frequency 的 G1-G0 standardized effect={effects['beam_frequency'][0]:+.3f}，FDR 显著 mask={effects['beam_frequency'][1]}/14。",
        f"validation domain 条件 G1 rate 范围为 {100*context['domain_range'][0]:.2f}%–{100*context['domain_range'][1]:.2f}%；仅两条轨迹，因此只作条件描述。",
        f"速度/角速度/曲率 standardized effect 为 {effects['speed'][0]:+.3f}/{effects['angular_velocity'][0]:+.3f}/{effects['curvature'][0]:+.3f}，FDR 显著 mask 为 {effects['speed'][1]}/{effects['angular_velocity'][1]}/{effects['curvature'][1]}。",
        f"合法 t→t+1 beam cycle transition effect={effects['beam_transition'][0]:+.3f}，FDR 显著 mask={effects['beam_transition'][1]}/14；未读取 channel history。",
        f"future top1-top2 power gap effect={effects['ambiguity_gap'][0]:+.3f}，FDR 显著 mask={effects['ambiguity_gap'][1]}/14；它只属于标签侧 oracle。",
        f"F2 ROC-AUC={f2['roc_auc']:.3f}，PR-AUC={f2['pr_auc']:.3f}（prevalence={f2['prevalence']:.3f}），balanced accuracy={f2['balanced_accuracy']:.3f}，recall@20%={f2['recall_at_20pct']:.3f}。",
        f"F3 oracle ROC-AUC={f3['roc_auc']:.3f}，PR-AUC={f3['pr_auc']:.3f}；F3 含 Full/drift/future power，禁止部署。",
        f"推理时识别困难性的链 E={'通过' if route['chain_passed']['E'] else '未通过'}；只依据 F2 lift、top20 recall 与双轨迹一致性判断。",
        f"Full 插值 alpha_star 的跨 mask 中位数中位数为 {full_alpha:.3f}；不可恢复样本以 unrecovered count 报告。",
        f"parallel-only 平均恢复 {100*parallel_recovery:.2f}% G1；20-seed random 为 {100*random_recovery:.2f}%。",
        f"预注册 K=16 PCA 方向平均恢复 {100*pca_recovery:.2f}% G1；其他 K 仅作 sensitivity，不按 validation 选择。",
        f"K16 相对随机方向变化为 {100*(pca_recovery-random_recovery):+.2f} pp；正式判断还要求链 C 的 rank 与双轨迹条件。",
        f"G1 邻域保持最大相邻层下降为 `{neighbor_pair[0]}→{neighbor_pair[1]}`（{neighborhood[neighbor_pair[0]]:.3f}→{neighborhood[neighbor_pair[1]]:.3f}）。",
        f"绝对影响最大的冻结融合输入消融是 `{worst_temporal}`，G1 Top-1 相对 T0 平均变化 {100*temporal_effect:+.2f} pp；模型本身没有 temporal/GRU/attention module，且链 D 的时间支持={'通过' if route['chains']['D']['temporal_ablation_support'] else '未通过'}。",
        f"两条 validation trajectory 的 parallel/rank/最严重层一致比例为 {100*stability['parallel_direction_fraction']:.1f}%/{100*stability['rank_direction_fraction']:.1f}%/{100*stability['layer_agreement_fraction']:.1f}%。",
        f"正式负对照{'全部通过' if controls['all_formal_passed'] else '未全部通过'}，通过比例 {100*controls['formal_pass_fraction']:.1f}%；validation-leakage oracle 已标记并排除。",
        f"最终自动匹配路线 `{route['route']}`，固定优先级为 A→B→C→D→E→R6→R0。",
        f"完整预注册证据链{'存在' if route['any_complete_chain'] else '不存在'}；A-E={route['chain_passed']}。",
        f"下一阶段建议：{route['recommendation']}",
        "不应凭单图继续 prototype deformation、probabilistic prototype、uncertainty network、自由动态门控、全局 feature-MSE 或 CSI 融合；未通过证据链对应的方法也不得实现。",
        "本轮未使用 CSI/channel/RadioPrototypeExpert/F1；`csi_used=false`、`channel_input_used=false`、`f1_used=false`。",
        "不存在 future leakage；M4 与 F2 严格使用 t-4...t，t+1 power 只进入标签侧 D0/D6/F3 oracle 指标。",
        "outer test 继续封存；未构造 loader/cache，`outer_test_accessed=false`。",
    ]
    if len(answers) != 40:
        raise AssertionError(f"final report must contain exactly 40 answers, got {len(answers)}")
    lines = [
        "# 完整输入—缺失输入配对困难样本表征几何诊断",
        "",
        f"冻结生产 M4 重算：Full Top-1={100*d0['full_top1']:.3f}%，All-14={100*d0['all14_top1']:.3f}%，Worst `{d0['worst_mask']}`={100*d0['worst_top1']:.3f}%。自动路线为 **{route['route']}**。",
        "",
        "## 每个 mask 的四组比例",
        "",
        *table,
        "",
        "## 40 项必答结论",
        "",
    ]
    lines.extend(f"{index}. {answer}" for index, answer in enumerate(answers, 1))
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "所有 margin 阈值、PCA、probe、层判定和路线门槛来自 train、train trajectory CV 或预注册常量。Validation 固定评估一次并按两条 trajectory 分报；非法 validation-selected direction 仅作 sanity check。",
        ]
    )
    (output / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_outputs(output: Path) -> dict[str, Any]:
    required = [
        "audit.md",
        "resolved_config.yaml",
        "process_manifest.json",
        "cache/train_multilayer_features.pt",
        "cache/validation_multilayer_features.pt",
        "cache/cache_manifest.json",
        "diagnostics/group_counts_by_mask.csv",
        "diagnostics/group_counts_by_beam.csv",
        "diagnostics/group_counts_by_trajectory.csv",
        "diagnostics/group_counts_by_domain.csv",
        "diagnostics/d0_baseline_metrics.csv",
        "diagnostics/d0_prediction_transition.csv",
        "diagnostics/d1_final_geometry_by_group.csv",
        "diagnostics/d1_final_geometry_by_mask.csv",
        "diagnostics/d1_final_geometry_by_beam.csv",
        "diagnostics/d2_decision_direction_decomposition.csv",
        "diagnostics/d3_layerwise_geometry.csv",
        "diagnostics/d3_first_break_layer.csv",
        "diagnostics/d4_representation_collapse.csv",
        "diagnostics/d5_modality_marginal_utility.csv",
        "diagnostics/d5_pairwise_interactions.csv",
        "diagnostics/d5_sample_utility_features.csv",
        "diagnostics/d6_hard_sample_attributes.parquet",
        "diagnostics/d6_group_attribute_statistics.csv",
        "diagnostics/d7_hardness_probe_summary.csv",
        "diagnostics/d7_feature_importance.csv",
        "diagnostics/d8_counterfactual_repair.csv",
        "diagnostics/d9_local_neighborhood.csv",
        "diagnostics/d10_temporal_difficulty.csv",
        "diagnostics/stability_summary.csv",
        "diagnostics/negative_controls.csv",
        "artifacts/train_pca_directions.pt",
        "final_report.md",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required diagnostic outputs missing: {missing}")
    bad_cells = []
    for path in sorted((output / "diagnostics").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row_index, row in enumerate(csv.reader(handle), 1):
                if any(value.strip().lower() in {"nan", "inf", "+inf", "-inf"} for value in row):
                    bad_cells.append(f"{path.name}:{row_index}")
    if bad_cells:
        raise ValueError(f"formal CSV contains non-finite literals: {bad_cells[:10]}")
    figures = sorted((output / "figures").glob("*.png"))
    bad_figures = []
    for path in figures:
        image = plt.imread(path)
        if path.stat().st_size < 1000 or not np.isfinite(image).all() or float(image.var()) <= 1e-8:
            bad_figures.append(path.name)
    if bad_figures:
        raise ValueError(f"blank or invalid figures: {bad_figures}")
    return {
        "required_file_count": len(required),
        "figure_count": len(figures),
        "bad_csv_cells": 0,
        "bad_figures": 0,
    }


def analyze(config: Mapping[str, Any], output: Path, audit_payload: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()

    def progress(stage: str) -> None:
        elapsed = time.monotonic() - started
        _update_process(output, config, f"analyze:{stage}", "running", analysis_elapsed_seconds=elapsed)
        print(
            json.dumps({"event": "full_missing_geometry", "completed": stage, "elapsed_seconds": elapsed}),
            flush=True,
        )

    data = _load_analysis_inputs(config, output)
    selection = _selection_manifest(config, data)
    write_json(output / "artifacts/train_only_selection.json", selection, sort_keys=True)
    prepared = {
        role: _prepare_cache_geometry(
            data[role], data["prototypes"], data["topology"].distance.cpu(), selection
        )
        for role in ("train", "validation")
    }
    d0 = d0_groups_baseline(config, output, data, prepared)
    progress("D0-groups")
    d1_final_geometry(output, data, prepared)
    progress("D1")
    d2 = d2_decision_directions(config, output, data, prepared)
    progress("D2")
    d3 = d3_layerwise_geometry(config, output, data, prepared)
    progress("D3")
    d4 = d4_representation_collapse(output, d3)
    progress("D4")
    d5 = d5_modality_utility(output, data, prepared)
    progress("D5")
    d6 = d6_hard_sample_attributes(config, output, data, prepared, d3, d4, d5)
    progress("D6")
    d7 = d7_hardness_probes(config, output, data, d6)
    progress("D7")
    d8 = d8_counterfactual_repair(config, output, data, prepared)
    progress("D8")
    d9_local_neighborhood(config, output, data, prepared)
    progress("D9")
    d10 = d10_temporal_difficulty(output, data, prepared, d6)
    progress("D10")
    stability = stability_analysis(config, output, data, prepared, d2, d3, d5, d6, d7, d8)
    progress("stability")
    controls = negative_controls(config, output, data, prepared, d2, d5, d6, d7, d8)
    progress("negative-controls")
    route = select_route(config, data, prepared, d0, d2, d3, d4, d5, d6, d7, d8, d10, stability)
    create_figures(output, data, prepared, d0, d2, d3, d4, d5, d7, d8, d10)
    progress("figures")
    summary = finalize(
        config,
        output,
        data,
        prepared,
        selection,
        d0,
        d2,
        d3,
        d4,
        d5,
        d6,
        d7,
        d8,
        d10,
        stability,
        controls,
        route,
    )
    summary["audit_status"] = audit_payload["status"]
    summary["output_validation"] = _validate_outputs(output)
    summary["artifact_sha256"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"process_manifest.json", "diagnostic_summary.json"}
    }
    write_json(output / "diagnostic_summary.json", _json_ready(summary), sort_keys=True)
    progress("report")
    del data, prepared, d6, d7
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("audit", "cache", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main(*, stage_override: str | None = None) -> int:
    args = _parse_args()
    stage = stage_override or args.stage
    config = _load_config(args.config.resolve())
    output = _prepare_output(config, resume=bool(args.resume))
    started = time.monotonic()
    try:
        _update_process(output, config, stage, "running", started_at=now(), command_stage=stage)
        audit_payload = audit(config, output)
        resolved = json.loads(json.dumps(config))
        resolved["resolved_at"] = now()
        resolved["source_hashes"] = {
            "checkpoint_sha256": audit_payload["checkpoint_sha256"],
            "prototype_sha256": audit_payload["prototype_sha256"],
            "split_manifest_sha256": audit_payload["split_manifest_sha256"],
            "topology_manifest_sha256": audit_payload["topology_manifest_sha256"],
            "train_source_cache_sha256": audit_payload["train_source_cache_sha256"],
            "validation_source_cache_sha256": audit_payload["validation_source_cache_sha256"],
        }
        (output / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        if stage == "audit":
            _update_process(output, config, "audit", "passed", elapsed_seconds=time.monotonic() - started)
            return 0
        cache_manifest_path = output / "cache/cache_manifest.json"
        if cache_manifest_path.is_file():
            cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            if cache_manifest.get("cache_version") != config["cache"]["version"]:
                raise ValueError("resume cache version changed.")
        else:
            _update_process(output, config, "cache", "running", elapsed_seconds=time.monotonic() - started)
            build_cache(config, output, audit_payload)
        cache_manifest = _synchronize_existing_temporal_t0(output)
        if stage == "cache":
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
            route=summary["route"]["route"],
            cache_manifest_sha256=sha256_file(cache_manifest_path),
            diagnostic_summary_sha256=sha256_file(output / "diagnostic_summary.json"),
        )
        print(
            json.dumps(
                {"event": "diagnostic_complete", "route": summary["route"]["route"], "output": str(output)}
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        _update_process(
            output,
            config,
            stage,
            "failed",
            elapsed_seconds=time.monotonic() - started,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
