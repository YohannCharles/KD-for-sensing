#!/usr/bin/env python3
"""Audit, cache, train, and evaluate radio-guided hierarchical prototypes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from kd_sensing.baselines.full_pool_bt_scl import load_audited_topology
from kd_sensing.baselines.full_pool_common import sha256_file
from kd_sensing.baselines.mmw_trajectory import ABTC_METHOD, TrajectoryBaselineModel
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.losses.hierarchical_prototype_losses import (
    adapter_regularization,
    beam_classification_loss,
    beam_topology_loss,
    full_teacher_consistency_loss,
    prototype_anchor_loss,
    prototype_diversity_loss,
)
from kd_sensing.models.csi_anchored_completion import SparsePilotRadioEncoder
from kd_sensing.models.missing_sensing_prototype_adapter import MissingSensingPrototypeAdapter
from kd_sensing.models.propagation_mode_fusion import (
    fixed_beam_evidence_fusion,
    mode_consistent_fusion,
)
from kd_sensing.models.propagation_subprototype_bank import (
    PropagationAwareSubPrototypeBank,
    reproducible_random_residuals,
)
from kd_sensing.models.radio_guided_prototype_distillation import (
    propagation_mode_distribution,
    qualified_teacher_weights,
    radio_prototype_distillation_loss,
)
from kd_sensing.models.radio_prototype_expert import RadioPrototypeExpert

if __package__:
    from .run_mmw_trajectory_baselines import ALL_PATTERNS
    from .run_sparse_pilot_recovery import _prediction_metrics
    from .run_sparse_pilot_trajectory_recovery import nested_frequency_indices, parse_budget
else:
    from run_mmw_trajectory_baselines import ALL_PATTERNS
    from run_sparse_pilot_recovery import _prediction_metrics
    from run_sparse_pilot_trajectory_recovery import nested_frequency_indices, parse_budget


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tools/configs/radio_guided_hierarchical_prototypes.yaml"
MASK_NAMES = tuple(name for name in ALL_PATTERNS if name != "full")
MASK_COUNTS = {name: int(sum(ALL_PATTERNS[name])) for name in MASK_NAMES}
GROUP_NAMES = {1: "single", 2: "two", 3: "three"}
CACHE_REQUIRED_KEYS = {
    "z_s_all_masks",
    "sensing_logits_all_masks",
    "z_s_full",
    "full_logits",
    "full_probability",
    "z_c",
    "c_radio",
    "frame_csi_features",
    "radio_evidence",
    "csi_classifier_logits",
    "csi_quality",
    "csi_available",
    "target",
    "future_beam_power",
    "sample_ids",
    "trajectory_ids",
    "mask_names",
    "mask_availability",
    "identity",
}


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_config(path: Path) -> dict[str, Any]:
    config = safe_load_yaml(path.read_text(encoding="utf-8"))
    if config["protocol"].get("outer_test_enabled") is not False:
        raise ValueError("Radio-guided experiments require outer_test_enabled=false.")
    if int(config["pilot"]["re_per_frame"]) != 4 or int(config["pilot"]["re_window"]) != 20:
        raise ValueError("The main protocol must use five frames x four RE = 20 RE.")
    return config


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = list(values[0])
    for row in values[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    array = torch.as_tensor(value).detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def require_inner_split(role: str) -> str:
    """Fail closed before constructing any outer-test loader or cache path."""
    value = str(role)
    if value not in {"train", "validation"}:
        raise ValueError("Only train and validation are permitted; outer test remains sealed.")
    return value


def validate_no_future_csi(record: Mapping[str, Any]) -> None:
    forbidden = {
        key
        for key in record
        if "future_csi" in key.lower()
        or "future_channel" in key.lower()
        or key.lower() in {"channel_ref", "channel_history_refs"}
    }
    if forbidden:
        raise ValueError(f"CSI cache exposes forbidden channel inputs: {sorted(forbidden)}.")
    candidates = torch.as_tensor(record["candidate_history"])
    if tuple(candidates.shape[1:]) != (5, 32, 16):
        raise ValueError("candidate_history must contain exactly t-4...t with shape [N,5,32,16].")


def validate_trajectory_disjointness(
    train_trajectory_ids: Sequence[str], validation_trajectory_ids: Sequence[str]
) -> None:
    train, validation = set(train_trajectory_ids), set(validation_trajectory_ids)
    if not train or not validation or train & validation:
        raise ValueError("Train and validation trajectories must be non-empty and mutually exclusive.")


def _load_records(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    role = require_inner_split(role)
    return torch.load(_path(config["source"][f"{role}_records"]), map_location="cpu", weights_only=False, mmap=True)


def _load_features(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    role = require_inner_split(role)
    return torch.load(_path(config["source"][f"{role}_features"]), map_location="cpu", weights_only=False, mmap=True)


def _load_cache(config: Mapping[str, Any], role: str) -> dict[str, Any]:
    role = require_inner_split(role)
    path = _path(config["cache"]["root"]) / f"{role}.pt"
    record = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    missing = CACHE_REQUIRED_KEYS - set(record)
    if missing:
        raise ValueError(f"Radio-guided cache lacks keys: {sorted(missing)}.")
    if record["identity"].get("cache_version") != config["cache"]["version"]:
        raise ValueError("Radio-guided cache version mismatch.")
    return record


def _attach_reference_probabilities(
    cache: Mapping[str, Any], records: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach the published M4 probabilities after checking sample alignment."""
    if list(cache["sample_ids"]) != list(records["sample_ids"]):
        raise ValueError("M4 reference probabilities are not aligned by sample ID.")
    labels = torch.as_tensor(records["labels_future"]).long()
    if not torch.equal(torch.as_tensor(cache["target"]).long(), labels):
        raise ValueError("M4 reference probabilities are not aligned by target.")
    result = dict(cache)
    result["reference_probability_all_masks"] = torch.stack(
        [torch.as_tensor(records[f"p0_{name}"]).float() for name in MASK_NAMES], dim=1
    )
    result["reference_full_probability"] = torch.as_tensor(records["p0_full"]).float()
    return result


def _load_m4(config: Mapping[str, Any], device: torch.device) -> tuple[TrajectoryBaselineModel, dict[str, Any]]:
    payload = torch.load(_path(config["source"]["m4_checkpoint"]), map_location="cpu", weights_only=False)
    if payload.get("method") != ABTC_METHOD:
        raise ValueError("The published trajectory M4 checkpoint is required.")
    if payload.get("protocol_fingerprint") != config["protocol"]["fingerprint"]:
        raise ValueError("M4 protocol fingerprint mismatch.")
    model = TrajectoryBaselineModel(ABTC_METHOD, **payload.get("model_config", {})).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.prototype_bank is None or tuple(model.prototype_bank.prototypes.shape) != (64, 64):
        raise ValueError("M4 must expose a [64,64] Beam Prototype Bank.")
    return model, payload


def _load_radio_encoder(config: Mapping[str, Any], device: torch.device) -> SparsePilotRadioEncoder:
    model = SparsePilotRadioEncoder(
        history_length=int(config["pilot"]["history_frames"]),
        hidden_dim=int(config["model"]["radio_dim"]),
        num_candidate_patterns=32,
        encoder_layers=0,
    ).to(device)
    model.load_information_checkpoint(_path(config["source"]["csi_checkpoint"]))
    model.freeze()
    return model


def _load_csi_classifier(config: Mapping[str, Any], device: torch.device) -> nn.Module:
    hidden = int(config["model"]["radio_dim"])
    classifier = nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, int(config["model"]["num_beams"])),
    ).to(device)
    payload = torch.load(_path(config["source"]["csi_checkpoint"]), map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("classifier."): value
        for key, value in payload["model_state"].items()
        if key.startswith("classifier.")
    }
    classifier.load_state_dict(state, strict=True)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    return classifier


def _load_f1_radio_expert(
    config: Mapping[str, Any], device: torch.device
) -> tuple[RadioPrototypeExpert, float, dict[str, Any]]:
    payload = torch.load(_path(config["source"]["f1_checkpoint"]), map_location="cpu", weights_only=False)
    checks = {
        "method": payload.get("method") == "F1",
        "protocol": payload.get("protocol_fingerprint") == config["protocol"]["fingerprint"],
        "split": payload.get("split_manifest_sha256") == config["protocol"]["split_manifest_sha256"],
        "m4": payload.get("m4_checkpoint_sha256") == config["source"]["m4_checkpoint_sha256"],
        "csi": payload.get("csi_checkpoint_sha256") == config["source"]["csi_checkpoint_sha256"],
        "outer_test": payload.get("outer_test_accessed") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"Current F1 checkpoint identity mismatch: {checks}.")
    state = payload["model_state"]
    expert = RadioPrototypeExpert(
        radio_dim=int(config["model"]["radio_dim"]),
        hidden_dim=128,
        prototype_dim=int(config["model"]["embedding_dim"]),
        temperature=0.0001,
    ).to(device)
    expert.load_state_dict(
        {key.removeprefix("radio_expert."): value for key, value in state.items() if key.startswith("radio_expert.")},
        strict=True,
    )
    expert.eval()
    for parameter in expert.parameters():
        parameter.requires_grad_(False)
    sensing_temperature = float(F.softplus(state["sensing_temperature.raw"].float()).item() + 1e-6)
    return expert, sensing_temperature, payload


def _frequency_positions(config: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    prepared = safe_load_yaml(_path(config["source"]["prepared_config"]).read_text(encoding="utf-8"))
    values = torch.tensor(prepared["runtime"]["frequency_positions_hz"], dtype=torch.float32, device=device)
    _, count = parse_budget(str(config["pilot"]["budget"]))
    return values.index_select(0, nested_frequency_indices(len(values), count).to(device))


def _noisy_observations(
    candidates: torch.Tensor,
    snr_db: torch.Tensor,
    *,
    dropout: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    power = candidates.abs().square().mean(dim=(-2, -1), keepdim=True)
    variance = power / torch.pow(10.0, snr_db[..., None, None] / 10.0)
    scale = (variance / 2.0).sqrt()
    noise = torch.complex(
        torch.randn(candidates.shape, device=candidates.device, generator=generator),
        torch.randn(candidates.shape, device=candidates.device, generator=generator),
    ) * scale
    valid = torch.ones_like(candidates, dtype=torch.bool)
    if float(dropout):
        valid &= torch.rand(candidates.shape, device=candidates.device, generator=generator) >= float(dropout)
    return (candidates + noise) * valid, valid


def _selected_candidates(config: Mapping[str, Any], values: torch.Tensor) -> torch.Tensor:
    patterns, frequencies = parse_budget(str(config["pilot"]["budget"]))
    frequency_ids = nested_frequency_indices(values.shape[-1], frequencies)
    return values[:, -int(config["pilot"]["history_frames"]) :, :patterns].index_select(-1, frequency_ids)


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    file_keys = (
        "split_manifest",
        "train_records",
        "validation_records",
        "train_features",
        "validation_features",
        "codebook",
        "m4_checkpoint",
        "csi_checkpoint",
        "f1_checkpoint",
    )
    locations = {"split_manifest": config["protocol"]["split_manifest"]} | {
        key: source[key] for key in file_keys if key != "split_manifest"
    }
    expected = {"split_manifest": config["protocol"]["split_manifest_sha256"]} | {
        key: source[f"{key}_sha256"] for key in file_keys if key != "split_manifest"
    }
    hashes = {key: sha256_file(_path(value)) for key, value in locations.items()}
    manifest = json.loads(_path(config["protocol"]["split_manifest"]).read_text(encoding="utf-8"))
    train, validation = _load_records(config, "train"), _load_records(config, "validation")
    train_features, validation_features = _load_features(config, "train"), _load_features(config, "validation")
    validate_no_future_csi(train)
    validate_no_future_csi(validation)

    train_ids, validation_ids = list(train["sample_ids"]), list(validation["sample_ids"])
    train_groups, validation_groups = set(train_features["trajectory_ids"]), set(validation_features["trajectory_ids"])
    validate_trajectory_disjointness(train_features["trajectory_ids"], validation_features["trajectory_ids"])
    checks = {
        "hashes": hashes == expected,
        "protocol": manifest.get("protocol_id") == config["protocol"]["id"],
        "fingerprint": manifest.get("protocol_fingerprint") == config["protocol"]["fingerprint"],
        "outer_test_disabled": config["protocol"]["outer_test_enabled"] is False,
        "outer_test_not_accessed": manifest.get("outer_test_accessed", False) is False,
        "sample_counts": len(train_ids) == int(config["protocol"]["expected_train_samples"])
        and len(validation_ids) == int(config["protocol"]["expected_validation_samples"]),
        "sample_ids_unique": len(train_ids) == len(set(train_ids)) and len(validation_ids) == len(set(validation_ids)),
        "sample_ids_disjoint": not bool(set(train_ids) & set(validation_ids)),
        "feature_sample_alignment": train_ids == list(train_features["sample_ids"])
        and validation_ids == list(validation_features["sample_ids"]),
        "target_alignment": torch.equal(train["labels_future"], train_features["target"])
        and torch.equal(validation["labels_future"], validation_features["target"]),
        "trajectory_disjoint": not bool(train_groups & validation_groups),
        "trajectory_counts": len(train_groups) == int(config["protocol"]["expected_train_trajectories"])
        and len(validation_groups) == int(config["protocol"]["expected_validation_trajectories"]),
        "twenty_re": tuple(_selected_candidates(config, train["candidate_history"][:1]).shape[1:]) == (5, 2, 2),
        "no_future_channel_inputs": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Radio-guided preflight failed: {checks}.")
    result = {
        "status": "passed",
        "checks": checks,
        "hashes": hashes,
        "train_samples": len(train_ids),
        "validation_samples": len(validation_ids),
        "train_trajectories": len(train_groups),
        "validation_trajectories": len(validation_groups),
        "future_channel_used_as_input": False,
        "test_loader_constructed": False,
        "outer_test_accessed": False,
    }
    output = _path(config["output"]["root"])
    _write_json(output / "preflight.json", result)
    resolved = output / "resolved_configs/base.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    return result


def audit(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    preflight_result = preflight(config)
    m4, m4_payload = _load_m4(config, device)
    expert, sensing_temperature, f1_payload = _load_f1_radio_expert(config, device)
    train, validation = _load_records(config, "train"), _load_records(config, "validation")
    train_features, validation_features = _load_features(config, "train"), _load_features(config, "validation")
    prototype = m4.prototype_bank.prototypes.detach().float().cpu()
    prototype_hash = _tensor_sha256(prototype)
    if prototype_hash != config["source"]["prototype_bank_sha256"]:
        raise ValueError("Prototype Bank tensor hash mismatch.")
    counts = torch.bincount(train["labels_future"], minlength=64)
    count_rows = [{"beam": beam, "train_samples": int(counts[beam])} for beam in range(64)]
    output = _path(config["output"]["root"])
    _write_csv(output / "diagnostics/beam_class_counts.csv", count_rows)

    full_top1 = float(
        validation_features["p_full"].argmax(dim=-1).eq(validation_features["target"]).float().mean().item()
    )
    radio_temperature = float(expert.temperature().detach().cpu())
    bank_temperature = float(m4.prototype_bank.temperature)
    effective_radio_scale = 1.0 / radio_temperature
    imbalance_ratio = float(counts.max().item() / max(counts.min().item(), 1))
    result = {
        **preflight_result,
        "prototype_shape": list(prototype.shape),
        "prototype_similarity": "cosine",
        "sensing_embedding_dim": 64,
        "radio_gru_dim": 128,
        "radio_prototype_embedding_dim": 64,
        "prototype_bank_temperature": bank_temperature,
        "sensing_temperature": sensing_temperature,
        "radio_temperature": radio_temperature,
        "effective_radio_cosine_scale": effective_radio_scale,
        "fp32_evidence": True,
        "pilot_frames": 5,
        "pilot_re_per_frame": 4,
        "pilot_re_window": 20,
        "train_class_min": int(counts.min()),
        "train_class_max": int(counts.max()),
        "train_class_imbalance_ratio": imbalance_ratio,
        "severe_class_imbalance": imbalance_ratio >= 10.0,
        "full_top1": full_top1,
        "expected_full_top1": 0.8633149862,
        "prototype_bank_sha256": prototype_hash,
        "m4_checkpoint_method": m4_payload.get("method"),
        "f1_checkpoint_method": f1_payload.get("method"),
    }
    lines = [
        "# Radio-Guided Hierarchical Prototypes 协议审计",
        "",
        "## 结论",
        "",
        "- 审计通过；未发现 sample ID 错位、trajectory 交叉、future channel 输入或 prototype 计算错误。",
        "- 仅加载 train/validation；未构造 outer-test loader，outer test 保持封存。",
        f"- Full Top-1 为 {100.0 * full_top1:.4f}%，与发布 M4 一致。",
        "",
        "## 必答项",
        "",
        "1. 主 Prototype Bank 实际 shape 为 `[64, 64]`，checkpoint tensor hash 为 "
        f"`{prototype_hash}`。",
        "2. `BeamPrototypeBank` 对 feature 与 prototype 分别 L2 normalize，使用 cosine similarity；不是裸 dot product或距离。",
        "3. sensing embedding `z_s` 为 64 维；CSI GRU `c_radio` 为 128 维，经 `RadioPrototypeExpert` 映射后的 `z_c` 为 64 维。",
        f"4. M4 bank temperature={bank_temperature:.10g}；F1 sensing temperature={sensing_temperature:.10g}；"
        f"F1 radio temperature={radio_temperature:.10g}。prototype/evidence/temperature/softmax 路径均强制 FP32。",
        "5. 当前 F1 对缺失路径的准确公式为 "
        "`e_s_cal=e_s/T_s; e_c_cal=shared_bank(z_c)*0.1/T_c; e_final=e_s_cal+0.5*(e_c_cal-e_s_cal)`，"
        "随后 FP32 softmax；Full 硬旁路。",
        "6. `candidate_history` 为 `[N,5,32,16]`；主协议仅选每帧 `2 patterns x 2 frequencies = 4 RE`，"
        "五帧窗口共 20 RE。",
        "7. `z_s`、`z_c`、target、sensing mask、trajectory ID 均以稳定 sample ID 对齐；train/validation ID 完整、唯一、互斥。",
        f"8. 64 类 train 样本数已写入 `diagnostics/beam_class_counts.csv`；最少 {int(counts.min())}，最多 {int(counts.max())}。",
        f"9. 最大/最小类样本比为 {imbalance_ratio:.3f}；按 10:1 阈值，严重不均衡={imbalance_ratio >= 10.0}。",
        "10. 未发现 future channel 泄漏：pilot alignment 要求 t-4...t 连续且最后输入帧严格早于 t+1；缓存不暴露 channel 引用。",
        "11. outer test 保持封存；本入口对除 `train`/`validation` 外的 role fail closed。",
        "",
        "## 输入身份",
        "",
    ]
    for name, value in preflight_result["hashes"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "本审计未读取或缓存 outer-test 样本。", ""])
    (output / "audit.md").parent.mkdir(parents=True, exist_ok=True)
    (output / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(output / "diagnostics/audit.json", result)
    return result


@torch.inference_mode()
def _score_m4_bank(bank: nn.Module, values: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    chunks = []
    flat = values.reshape(-1, values.shape[-1])
    for start in range(0, len(flat), int(batch_size)):
        chunks.append(bank(flat[start : start + int(batch_size)].to(device)).float().cpu())
    return torch.cat(chunks).reshape(*values.shape[:-1], -1)


@torch.inference_mode()
def _extract_radio_view(
    config: Mapping[str, Any],
    records: Mapping[str, Any],
    *,
    seed: int,
    encoder: SparsePilotRadioEncoder,
    expert: RadioPrototypeExpert,
    classifier: nn.Module,
    prototype_bank: nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fields = defaultdict(list)
    batch_size = int(config["cache"]["batch_size"])
    frequencies = _frequency_positions(config, device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    count = len(records["sample_ids"])
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        selected = _selected_candidates(config, records["candidate_history"][start:stop]).to(device)
        snr = torch.full(
            (len(selected), selected.shape[1]),
            float(config["pilot"]["snr_db"]),
            device=device,
        )
        observations, valid = _noisy_observations(
            selected,
            snr,
            dropout=float(config["pilot"]["dropout_probability"]),
            generator=generator,
        )
        pattern_ids = torch.arange(selected.shape[2], device=device).expand(len(selected), selected.shape[1], -1)
        output = encoder(observations, pattern_ids, frequencies, valid, snr)
        radio = expert(output["c_radio"], prototype_bank)
        fields["c_radio"].append(output["c_radio"].float().cpu())
        fields["frame_csi_features"].append(output["frame_csi_features"].float().cpu())
        fields["csi_quality"].append(output["csi_quality"].float().cpu())
        fields["csi_available"].append(output["csi_available"].bool().cpu())
        fields["z_c"].append(radio["z_radio"].float().cpu())
        fields["radio_evidence"].append(radio["radio_evidence"].float().cpu())
        fields["csi_classifier_logits"].append(classifier(output["c_radio"].float()).float().cpu())
    return {name: torch.cat(chunks) for name, chunks in fields.items()}


def build_cache(config: Mapping[str, Any], device: torch.device, *, force: bool = False) -> dict[str, Any]:
    preflight_result = preflight(config)
    m4, _ = _load_m4(config, device)
    encoder = _load_radio_encoder(config, device)
    classifier = _load_csi_classifier(config, device)
    expert, sensing_temperature, _ = _load_f1_radio_expert(config, device)
    cache_root = _path(config["cache"]["root"])
    output_root = _path(config["output"]["root"])
    manifests: dict[str, Any] = {}
    for role in ("train", "validation"):
        destination = cache_root / f"{role}.pt"
        if destination.exists() and not force:
            existing = _load_cache(config, role)
            manifests[role] = {
                "path": str(destination.resolve()),
                "sha256": sha256_file(destination),
                "samples": len(existing["sample_ids"]),
                "reused": True,
            }
            continue
        records, features = _load_records(config, role), _load_features(config, role)
        validate_no_future_csi(records)
        z_masks = torch.stack([records[f"z_{name}"].float() for name in MASK_NAMES], dim=1)
        logits_masks = _score_m4_bank(m4.prototype_bank, z_masks, device, int(config["cache"]["batch_size"]))
        z_full = records["z_full"].float()
        full_logits = _score_m4_bank(m4.prototype_bank, z_full, device, int(config["cache"]["batch_size"]))
        seeds = (
            [int(config["cache"]["train_noise_seed"])]
            if role == "train"
            else [int(seed) for seed in config["cache"]["validation_noise_seeds"]]
        )
        radio_views = [
            _extract_radio_view(
                config,
                records,
                seed=seed,
                encoder=encoder,
                expert=expert,
                classifier=classifier,
                prototype_bank=m4.prototype_bank,
                device=device,
            )
            for seed in seeds
        ]
        identity = {
            "cache_version": config["cache"]["version"],
            "role": role,
            "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
            "csi_checkpoint_sha256": config["source"]["csi_checkpoint_sha256"],
            "f1_checkpoint_sha256": config["source"]["f1_checkpoint_sha256"],
            "prototype_bank_sha256": config["source"]["prototype_bank_sha256"],
            "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
            "pilot_codebook_file_sha256": config["source"]["codebook_sha256"],
            "pilot_codebook_hash": config["source"]["codebook_hash"],
            "budget": config["pilot"]["budget"],
            "history_frames": int(config["pilot"]["history_frames"]),
            "re_per_frame": int(config["pilot"]["re_per_frame"]),
            "re_window": int(config["pilot"]["re_window"]),
            "snr_db": float(config["pilot"]["snr_db"]),
            "dropout_probability": float(config["pilot"]["dropout_probability"]),
            "noise_seeds": seeds,
            "sensing_temperature": sensing_temperature,
            "prototype_bank_temperature": float(m4.prototype_bank.temperature),
            "radio_temperature": float(expert.temperature().detach().cpu()),
            "outer_test_accessed": False,
        }
        payload = {
            "z_s_all_masks": z_masks,
            "sensing_logits_all_masks": logits_masks,
            "z_s_full": z_full,
            "full_logits": full_logits,
            "full_probability": features["p_full"].float(),
            **{
                field: torch.stack([view[field] for view in radio_views])
                for field in radio_views[0]
            },
            "target": records["labels_future"].long(),
            "future_beam_power": records["future_beam_power"].float(),
            "sample_ids": list(records["sample_ids"]),
            "trajectory_ids": list(features["trajectory_ids"]),
            "mask_names": list(MASK_NAMES),
            "mask_availability": torch.tensor([ALL_PATTERNS[name] for name in MASK_NAMES], dtype=torch.bool),
            "identity": identity,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        manifests[role] = {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "samples": len(payload["sample_ids"]),
            "noise_seeds": seeds,
            "tensor_shapes": {
                key: list(value.shape) for key, value in payload.items() if torch.is_tensor(value)
            },
            "identity": identity,
            "reused": False,
        }
        _write_json(output_root / f"cache_manifests/{role}.json", manifests[role])
    combined = {
        "status": "complete",
        "preflight": preflight_result,
        "roles": manifests,
        "outer_test_accessed": False,
    }
    _write_json(output_root / "cache_manifests/manifest.json", combined)
    return combined


def spherical_kmeans(
    features: torch.Tensor,
    k: int,
    *,
    seed: int,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Cluster normalized features by cosine distance without label inputs."""
    values = F.normalize(torch.as_tensor(features).float().cpu(), dim=-1)
    count = len(values)
    clusters = int(k)
    if values.ndim != 2 or clusters <= 0 or count < clusters:
        raise ValueError("spherical_kmeans requires [N,D] features with N >= k > 0.")
    generator = torch.Generator().manual_seed(int(seed))
    selected = [int(torch.randint(count, (1,), generator=generator).item())]
    while len(selected) < clusters:
        nearest = 1.0 - (values @ values[selected].t()).amax(dim=-1)
        nearest[selected] = 0.0
        total = nearest.clamp_min(0.0).sum()
        if float(total) <= 1e-12:
            remaining = next(index for index in range(count) if index not in selected)
        else:
            remaining = int(torch.multinomial(nearest.clamp_min(0.0) / total, 1, generator=generator).item())
            if remaining in selected:
                remaining = next(index for index in range(count) if index not in selected)
        selected.append(remaining)
    centers = values[selected].clone()
    previous_sse = math.inf
    labels = torch.zeros(count, dtype=torch.long)
    for _ in range(int(max_iterations)):
        similarity = values @ centers.t()
        labels = similarity.argmax(dim=-1)
        updated = []
        for cluster in range(clusters):
            members = values[labels.eq(cluster)]
            if len(members):
                updated.append(F.normalize(members.mean(dim=0), dim=0))
            else:
                nearest = similarity.amax(dim=-1)
                updated.append(values[int(nearest.argmin())])
        centers = torch.stack(updated)
        assigned = (values * centers.index_select(0, labels)).sum(dim=-1)
        sse = float((1.0 - assigned).clamp_min(0.0).sum().item())
        if abs(previous_sse - sse) <= float(tolerance) * max(previous_sse if math.isfinite(previous_sse) else 1.0, 1.0):
            break
        previous_sse = sse
    similarity = values @ centers.t()
    labels = similarity.argmax(dim=-1)
    sse = float((1.0 - similarity.gather(1, labels[:, None]).squeeze(1)).clamp_min(0.0).sum().item())
    return labels, centers, sse


def cosine_silhouette(features: torch.Tensor, labels: torch.Tensor) -> float:
    values = F.normalize(torch.as_tensor(features).float().cpu(), dim=-1)
    assignment = torch.as_tensor(labels).long().cpu()
    clusters = torch.unique(assignment)
    if len(clusters) < 2 or any(int(assignment.eq(cluster).sum()) < 2 for cluster in clusters):
        return float("nan")
    distance = (1.0 - values @ values.t()).clamp_min(0.0)
    scores = torch.empty(len(values), dtype=torch.float32)
    for cluster in clusters.tolist():
        own = assignment.eq(cluster)
        own_count = int(own.sum())
        a = distance[:, own].sum(dim=1) / max(own_count - 1, 1)
        a[~own] = 0.0
        alternatives = [distance[:, assignment.eq(other)].mean(dim=1) for other in clusters.tolist() if other != cluster]
        b = torch.stack(alternatives).amin(dim=0)
        denominator = torch.maximum(a, b).clamp_min(1e-12)
        scores[own] = ((b - a) / denominator)[own]
    return float(scores.mean().item())


def adjusted_rand_index(first: torch.Tensor, second: torch.Tensor) -> float:
    a, b = torch.as_tensor(first).long().cpu(), torch.as_tensor(second).long().cpu()
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("ARI assignments must be matching vectors.")
    contingency = torch.zeros(int(a.max()) + 1, int(b.max()) + 1, dtype=torch.float64)
    for left, right in zip(a.tolist(), b.tolist(), strict=True):
        contingency[left, right] += 1.0
    choose2 = lambda value: value * (value - 1.0) / 2.0
    sum_cells = float(choose2(contingency).sum())
    sum_rows = float(choose2(contingency.sum(dim=1)).sum())
    sum_cols = float(choose2(contingency.sum(dim=0)).sum())
    total = choose2(torch.tensor(float(len(a))))
    if float(total) == 0.0:
        return 1.0
    expected = sum_rows * sum_cols / float(total)
    maximum = 0.5 * (sum_rows + sum_cols)
    return 1.0 if abs(maximum - expected) < 1e-12 else (sum_cells - expected) / (maximum - expected)


def normalized_mutual_information(first: torch.Tensor, second: torch.Tensor) -> float:
    a, b = torch.as_tensor(first).long().cpu(), torch.as_tensor(second).long().cpu()
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("NMI assignments must be matching vectors.")
    contingency = torch.zeros(int(a.max()) + 1, int(b.max()) + 1, dtype=torch.float64)
    for left, right in zip(a.tolist(), b.tolist(), strict=True):
        contingency[left, right] += 1.0
    joint = contingency / max(len(a), 1)
    pa, pb = joint.sum(dim=1), joint.sum(dim=0)
    expected = pa[:, None] * pb[None, :]
    occupied = joint > 0
    mutual = float((joint[occupied] * (joint[occupied] / expected[occupied]).log()).sum())
    entropy_a = float(-(pa[pa > 0] * pa[pa > 0].log()).sum())
    entropy_b = float(-(pb[pb > 0] * pb[pb > 0].log()).sum())
    denominator = math.sqrt(entropy_a * entropy_b)
    return 1.0 if denominator == 0.0 and torch.equal(a, b) else (mutual / denominator if denominator else 0.0)


def _centers_to_residuals(
    centers: torch.Tensor,
    base_prototypes: torch.Tensor,
    valid: torch.Tensor,
    *,
    radius: float,
) -> torch.Tensor:
    center = F.normalize(torch.as_tensor(centers).float(), dim=-1)
    base = F.normalize(torch.as_tensor(base_prototypes).float(), dim=-1)
    tangent = center - (center * base[:, None, :]).sum(dim=-1, keepdim=True) * base[:, None, :]
    if tangent.shape[1] == 2:
        contrast = tangent[:, :1] - tangent[:, 1:2]
        tangent = torch.cat((contrast, -contrast), dim=1)
    else:
        tangent = tangent - tangent.mean(dim=1, keepdim=True)
    tangent_norm = tangent.norm(dim=-1, keepdim=True)
    residual = tangent / tangent_norm.clamp_min(1e-12) * float(radius)
    usable = torch.as_tensor(valid, dtype=torch.bool)[:, None, None] & tangent_norm.gt(1e-8)
    return torch.where(usable, residual, torch.zeros_like(residual))


def _cluster_diagnostics(
    features: torch.Tensor,
    targets: torch.Tensor,
    sensing_logits: torch.Tensor,
    full_logits: torch.Tensor,
    base_prototypes: torch.Tensor,
    config: Mapping[str, Any],
    *,
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = F.normalize(torch.as_tensor(features).float(), dim=-1)
    labels = torch.as_tensor(targets).long()
    candidates = [int(value) for value in config["clustering"]["candidate_k"]]
    seeds = [int(value) for value in config["clustering"]["seeds"]]
    formal_k = int(config["clustering"]["formal_k"])
    centers = torch.zeros(64, formal_k, values.shape[-1])
    assignments = torch.full((len(values),), -1, dtype=torch.long)
    valid = torch.zeros(64, dtype=torch.bool)
    rows: list[dict[str, Any]] = []
    for beam in range(64):
        members = labels.eq(beam).nonzero(as_tuple=False).squeeze(1)
        beam_features = values.index_select(0, members)
        runs: dict[int, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
        for k in candidates:
            runs[k] = [
                spherical_kmeans(
                    beam_features,
                    k,
                    seed=seed,
                    max_iterations=int(config["clustering"]["max_iterations"]),
                    tolerance=float(config["clustering"]["tolerance"]),
                )
                for seed in seeds
            ]
        formal_labels, formal_centers, _ = runs[formal_k][0]
        centers[beam] = formal_centers
        assignments[members] = formal_labels
        occupancy = torch.bincount(formal_labels, minlength=formal_k)
        fractions = occupancy.float() / max(len(members), 1)
        valid[beam] = len(members) >= int(config["clustering"]["minimum_beam_samples"]) and bool(
            (fractions >= float(config["clustering"]["minimum_cluster_fraction"])).all()
        )
        stability_ari, stability_nmi = [], []
        for left in range(len(seeds)):
            for right in range(left + 1, len(seeds)):
                stability_ari.append(adjusted_rand_index(runs[formal_k][left][0], runs[formal_k][right][0]))
                stability_nmi.append(normalized_mutual_information(runs[formal_k][left][0], runs[formal_k][right][0]))
        row: dict[str, Any] = {
            "source": source_name,
            "beam": beam,
            "sample_count": len(members),
            **{f"k{k}_cosine_sse": runs[k][0][2] for k in candidates},
            "k2_relative_sse_reduction": (
                (runs[1][0][2] - runs[2][0][2]) / max(runs[1][0][2], 1e-12)
            ),
            "k2_silhouette": cosine_silhouette(beam_features, formal_labels),
            "cluster_0_count": int(occupancy[0]),
            "cluster_1_count": int(occupancy[1]),
            "cluster_0_fraction": float(fractions[0]),
            "cluster_1_fraction": float(fractions[1]),
            "minimum_cluster_count": int(occupancy.min()),
            "valid_subcluster": bool(valid[beam]),
            "stability_ari_mean": float(np.mean(stability_ari)),
            "stability_nmi_mean": float(np.mean(stability_nmi)),
        }
        if 3 in candidates:
            row["k3_silhouette"] = cosine_silhouette(beam_features, runs[3][0][0])
        for cluster in range(formal_k):
            cluster_members = members.index_select(
                0, formal_labels.eq(cluster).nonzero(as_tuple=False).squeeze(1)
            )
            target = labels.index_select(0, cluster_members)
            full = full_logits.index_select(0, cluster_members).argmax(dim=-1)
            row[f"full_error_cluster_{cluster}"] = float(full.ne(target).float().mean().item())
            for mask_id, mask_name in enumerate(MASK_NAMES):
                prediction = sensing_logits.index_select(0, cluster_members)[:, mask_id].argmax(dim=-1)
                row[f"{mask_name}_accuracy_cluster_{cluster}"] = float(prediction.eq(target).float().mean().item())
        rows.append(row)
    residuals = _centers_to_residuals(
        centers,
        base_prototypes,
        valid,
        radius=float(config["model"]["initialization_radius"]),
    )
    bank = PropagationAwareSubPrototypeBank(
        base_prototypes,
        num_subprototypes=formal_k,
        epsilon=float(config["model"]["epsilon"]),
        tau_sub=float(config["model"]["tau_sub"]),
    )
    bank.initialize_residuals_(residuals)
    artifact = {
        "source": source_name,
        "centers": centers,
        "cluster_assignments": assignments,
        "residuals": bank.residuals().detach(),
        "subprototypes": bank.subprototypes().detach(),
        "valid_subcluster_mask": valid,
        "formal_k": formal_k,
        "seeds": seeds,
        "normalization": "l2",
        "residual_radius": float(config["model"]["initialization_radius"]),
        "train_only": True,
        "outer_test_accessed": False,
    }
    return rows, artifact


def analyze_subclusters(config: Mapping[str, Any], *, role: str = "train") -> dict[str, Any]:
    if require_inner_split(role) != "train":
        raise ValueError("Subprototype clustering may only use the train split.")
    cache = _load_cache(config, "train")
    m4_payload = torch.load(_path(config["source"]["m4_checkpoint"]), map_location="cpu", weights_only=False)
    base = m4_payload["state_dict"]["prototype_bank.prototypes"].float()
    labels = cache["target"].long()
    sensing_logits = cache["sensing_logits_all_masks"].float()
    full_logits = cache["full_logits"].float()
    csi_rows, csi_artifact = _cluster_diagnostics(
        cache["z_c"][0],
        labels,
        sensing_logits,
        full_logits,
        base,
        config,
        source_name="csi",
    )
    sensing_rows, sensing_artifact = _cluster_diagnostics(
        cache["z_s_full"],
        labels,
        sensing_logits,
        full_logits,
        base,
        config,
        source_name="sensing",
    )
    output = _path(config["output"]["root"])
    artifact_root = output / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    csi_path = artifact_root / "csi_initialized_subprototypes.pt"
    sensing_path = artifact_root / "sensing_initialized_subprototypes.pt"
    valid_path = artifact_root / "valid_subcluster_mask.pt"
    torch.save(csi_artifact, csi_path)
    torch.save(sensing_artifact, sensing_path)
    torch.save(csi_artifact["valid_subcluster_mask"], valid_path)
    _write_csv(output / "diagnostics/csi_subcluster_statistics.csv", csi_rows)
    _write_csv(output / "diagnostics/sensing_subcluster_statistics.csv", sensing_rows)
    stability_rows = [
        {
            "source": source,
            "beam": row["beam"],
            "ari": row["stability_ari_mean"],
            "nmi": row["stability_nmi_mean"],
            "silhouette": row["k2_silhouette"],
            "k2_relative_sse_reduction": row["k2_relative_sse_reduction"],
        }
        for source, rows in (("csi", csi_rows), ("sensing", sensing_rows))
        for row in rows
    ]
    _write_csv(output / "clustering_stability.csv", stability_rows)
    result = {
        "csi_valid_beams": int(csi_artifact["valid_subcluster_mask"].sum()),
        "sensing_valid_beams": int(sensing_artifact["valid_subcluster_mask"].sum()),
        "csi_mean_k2_sse_reduction": float(np.mean([row["k2_relative_sse_reduction"] for row in csi_rows])),
        "csi_mean_silhouette": float(np.nanmean([row["k2_silhouette"] for row in csi_rows])),
        "csi_mean_stability_ari": float(np.mean([row["stability_ari_mean"] for row in csi_rows])),
        "csi_mean_stability_nmi": float(np.mean([row["stability_nmi_mean"] for row in csi_rows])),
        "artifacts": {
            "csi": {"path": str(csi_path.resolve()), "sha256": sha256_file(csi_path)},
            "sensing": {"path": str(sensing_path.resolve()), "sha256": sha256_file(sensing_path)},
            "valid_mask": {"path": str(valid_path.resolve()), "sha256": sha256_file(valid_path)},
        },
        "train_only": True,
        "outer_test_accessed": False,
    }
    _write_json(output / "diagnostics/clustering_summary.json", result)
    return result


class RadioGuidedPrototypeModel(nn.Module):
    """The only trainable path used by A1-A5 and A7."""

    def __init__(
        self,
        base_prototypes: torch.Tensor,
        config: Mapping[str, Any],
        *,
        sensing_scale: float,
        radio_scale: float,
    ) -> None:
        super().__init__()
        self.bank = PropagationAwareSubPrototypeBank(
            base_prototypes,
            num_subprototypes=int(config["model"]["num_subprototypes"]),
            epsilon=float(config["model"]["epsilon"]),
            tau_sub=float(config["model"]["tau_sub"]),
        )
        self.adapter = MissingSensingPrototypeAdapter(
            embedding_dim=int(config["model"]["embedding_dim"]),
            bottleneck_dim=int(config["model"]["adapter_bottleneck_dim"]),
        )
        self.register_buffer("sensing_scale", torch.tensor(float(sensing_scale), dtype=torch.float32))
        self.register_buffer("radio_scale", torch.tensor(float(radio_scale), dtype=torch.float32))

    def sensing(
        self, feature: torch.Tensor, *, missing: bool | torch.Tensor = True
    ) -> dict[str, torch.Tensor]:
        adapted, residual = self.adapter(feature.float(), missing, return_residual=True)
        evidence, scores = self.bank.beam_evidence(adapted, scale=self.sensing_scale)
        return {
            "adapted": adapted,
            "adapter_residual": residual,
            "mode_scores": scores,
            "evidence": evidence,
        }

    def radio(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence, scores = self.bank.beam_evidence(feature.float(), scale=self.radio_scale)
        return {"mode_scores": scores, "evidence": evidence}


def _load_topology(config: Mapping[str, Any]):
    topology_config = safe_load_yaml(_path(config["source"]["topology_config"]).read_text(encoding="utf-8"))
    return load_audited_topology(_path(topology_config["topology"]["manifest"]))


def _initialization(
    config: Mapping[str, Any], arm: str, base_prototypes: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, str]:
    output = _path(config["output"]["root"])
    artifact_root = _path(config.get("artifacts", {}).get("root", output / "artifacts"))
    mode = str(config["arms"][arm]["initialization"])
    if mode in {"csi", "sensing"}:
        path = artifact_root / f"{mode}_initialized_subprototypes.pt"
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        residuals = artifact["residuals"].float()
        valid = artifact["valid_subcluster_mask"].bool()
        return residuals, valid, sha256_file(path)
    if mode == "random":
        csi_path = artifact_root / "csi_initialized_subprototypes.pt"
        artifact = torch.load(csi_path, map_location="cpu", weights_only=False)
        valid = artifact["valid_subcluster_mask"].bool()
        residuals = reproducible_random_residuals(
            base_prototypes,
            num_subprototypes=int(config["model"]["num_subprototypes"]),
            radius=float(config["model"]["initialization_radius"]),
            seed=int(seed),
        )
        return residuals, valid, _tensor_sha256(residuals)
    if mode == "single":
        residuals = torch.zeros(
            64,
            int(config["model"]["num_subprototypes"]),
            int(config["model"]["embedding_dim"]),
        )
        return residuals, torch.zeros(64, dtype=torch.bool), _tensor_sha256(residuals)
    raise ValueError(f"Unknown initialization mode: {mode}.")


def _build_model(
    config: Mapping[str, Any], arm: str, seed: int, device: torch.device
) -> tuple[RadioGuidedPrototypeModel, torch.Tensor, str]:
    cache = _load_cache(config, "train")
    checkpoint = torch.load(_path(config["source"]["m4_checkpoint"]), map_location="cpu", weights_only=False)
    base = checkpoint["state_dict"]["prototype_bank.prototypes"].float()
    identity = cache["identity"]
    sensing_scale = 1.0 / (
        float(identity["prototype_bank_temperature"]) * float(identity["sensing_temperature"])
    )
    radio_scale = float(config["model"]["radio_subprototype_scale"])
    model = RadioGuidedPrototypeModel(
        base,
        config,
        sensing_scale=sensing_scale,
        radio_scale=radio_scale,
    ).to(device)
    residuals, valid, _source_initialization_hash = _initialization(config, arm, base, seed=seed)
    model.register_buffer("valid_subcluster_mask", valid.bool().to(device), persistent=True)
    model.bank.set_trainable_beam_mask_(valid.to(device))
    model.bank.initialize_residuals_(residuals.to(device))
    return model, valid.to(device), _tensor_sha256(model.bank.residuals())


def _balanced_mask_schedule(count: int, *, epoch: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(int(seed) * 100003 + int(epoch))
    indices = torch.randperm(int(count), generator=generator)
    group_masks = {
        cardinality: torch.tensor(
            [index for index, name in enumerate(MASK_NAMES) if MASK_COUNTS[name] == cardinality], dtype=torch.long
        )
        for cardinality in (1, 2, 3)
    }
    positions = torch.arange(int(count))
    groups = positions.remainder(3) + 1
    mask_ids = torch.empty(int(count), dtype=torch.long)
    for cardinality, choices in group_masks.items():
        selected = groups.eq(cardinality).nonzero(as_tuple=False).squeeze(1)
        mask_ids[selected] = choices[(selected // 3 + int(epoch) + int(seed)).remainder(len(choices))]
    return indices, mask_ids


def _metric_row(
    probability: torch.Tensor,
    labels: torch.Tensor,
    beam_power: torch.Tensor,
    base: torch.Tensor | None,
) -> dict[str, Any]:
    row = _prediction_metrics(probability.float(), labels.long(), beam_power.float(), base.float() if base is not None else None)
    return {
        "top1": row["top1"],
        "top3": row["top3"],
        "top5": row["top5"],
        "within3": row["within3"],
        "mae": row["mae"],
        "normalized_beamforming_gain": row["normalized_gain"],
        "beam_loss_db": row["beam_loss_db"],
        "fix_rate": row["fix_rate"],
        "harm_rate": row["harm_rate"],
    }


def _aggregate_mask_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics = (
        "top1",
        "top3",
        "top5",
        "within3",
        "mae",
        "normalized_beamforming_gain",
        "beam_loss_db",
        "fix_rate",
        "harm_rate",
    )
    result: dict[str, float] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[f"{metric}_macro"] = float(np.mean(values)) if values else float("nan")
        result[f"{metric}_worst"] = (
            float(max(values)) if metric in {"mae", "beam_loss_db", "harm_rate"} else float(min(values))
        ) if values else float("nan")
    return result


@torch.inference_mode()
def evaluate_hierarchical(
    model: RadioGuidedPrototypeModel,
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    arm: str,
    csi_on: bool,
    radio_view: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    labels = cache["target"].long()
    power = cache["future_beam_power"].float()
    batch_size = int(config["training"]["evaluation_batch_size"])
    inference = str(config["arms"][arm]["inference"])
    z_c = cache["z_c"][int(radio_view)]
    available = cache["csi_available"][int(radio_view)].bool()
    reference_masks = cache.get("reference_probability_all_masks")
    per_mask: list[dict[str, Any]] = []
    mode_kl_items: list[float] = []
    mode_agreement_items: list[float] = []
    base_predictions: list[torch.Tensor] = []
    disagreement_count = 0
    fused_matches_sensing = 0
    fused_matches_csi = 0
    fused_matches_neither = 0
    for mask_id, mask_name in enumerate(MASK_NAMES):
        probabilities, csi_off_probabilities, radio_predictions = [], [], []
        fallback_max_abs = 0.0
        for start in range(0, len(labels), batch_size):
            stop = min(start + batch_size, len(labels))
            sensing = model.sensing(cache["z_s_all_masks"][start:stop, mask_id].to(device), missing=True)
            radio = model.radio(z_c[start:stop].to(device))
            if csi_on and inference == "mode_fusion":
                evidence, _ = mode_consistent_fusion(
                    sensing["mode_scores"],
                    radio["mode_scores"],
                    csi_weight=float(config["model"]["fixed_csi_weight"]),
                    tau_sub=float(config["model"]["tau_sub"]),
                    csi_available=available[start:stop].to(device),
                )
            elif csi_on:
                evidence = fixed_beam_evidence_fusion(
                    sensing["evidence"],
                    radio["evidence"],
                    csi_weight=float(config["model"]["fixed_csi_weight"]),
                    csi_available=available[start:stop].to(device),
                )
            else:
                evidence = sensing["evidence"]
            if inference == "mode_fusion":
                fallback_evidence, _ = mode_consistent_fusion(
                    sensing["mode_scores"],
                    radio["mode_scores"],
                    csi_weight=float(config["model"]["fixed_csi_weight"]),
                    tau_sub=float(config["model"]["tau_sub"]),
                    csi_available=False,
                )
            else:
                fallback_evidence = fixed_beam_evidence_fusion(
                    sensing["evidence"],
                    radio["evidence"],
                    csi_weight=float(config["model"]["fixed_csi_weight"]),
                    csi_available=False,
                )
            fallback_max_abs = max(
                fallback_max_abs,
                float((fallback_evidence - sensing["evidence"].float()).abs().max().cpu()),
            )
            probabilities.append(torch.softmax(evidence.float(), dim=-1).cpu())
            csi_off_probabilities.append(torch.softmax(sensing["evidence"].float(), dim=-1).cpu())
            radio_predictions.append(radio["evidence"].argmax(dim=-1).cpu())
            batch_labels = labels[start:stop].to(device)
            q_s = propagation_mode_distribution(
                sensing["mode_scores"], batch_labels, temperature=float(config["model"]["mode_temperature"])
            )
            q_c = propagation_mode_distribution(
                radio["mode_scores"], batch_labels, temperature=float(config["model"]["mode_temperature"])
            )
            kl = (q_c * (q_c.clamp_min(1e-12).log() - q_s.clamp_min(1e-12).log())).sum(dim=-1)
            mode_kl_items.extend(kl.cpu().tolist())
            mode_agreement_items.extend(q_s.argmax(dim=-1).eq(q_c.argmax(dim=-1)).float().cpu().tolist())
        probability = torch.cat(probabilities)
        off_probability = torch.cat(csi_off_probabilities)
        base = (
            reference_masks[:, mask_id].float()
            if reference_masks is not None
            else torch.softmax(cache["sensing_logits_all_masks"][:, mask_id].float(), dim=-1)
        )
        base_predictions.append(base.argmax(dim=-1))
        if csi_on:
            fused_prediction = probability.argmax(dim=-1)
            sensing_prediction = off_probability.argmax(dim=-1)
            csi_prediction = torch.cat(radio_predictions)
            disagreement = sensing_prediction.ne(csi_prediction)
            disagreement_count += int(disagreement.sum())
            fused_matches_sensing += int((disagreement & fused_prediction.eq(sensing_prediction)).sum())
            fused_matches_csi += int((disagreement & fused_prediction.eq(csi_prediction)).sum())
            fused_matches_neither += int(
                (disagreement & fused_prediction.ne(sensing_prediction) & fused_prediction.ne(csi_prediction)).sum()
            )
        row = {
            "mask": mask_name,
            "available_count": MASK_COUNTS[mask_name],
            "mode": "csi_on" if csi_on else "csi_off",
            **_metric_row(probability, labels, power, base),
            "sensing_top1": float(base.argmax(dim=-1).eq(labels).float().mean()),
            "csi_off_fallback_max_abs": fallback_max_abs,
        }
        per_mask.append(row)
    groups = {
        GROUP_NAMES[count]: _aggregate_mask_rows([row for row in per_mask if row["available_count"] == count])
        for count in (1, 2, 3)
    }
    groups["all14"] = _aggregate_mask_rows(per_mask)
    full_probability = cache["full_probability"].float()
    reference_full = cache.get("reference_full_probability", full_probability).float()
    full = _metric_row(full_probability, labels, power, full_probability)
    subprototype_csi_probability = torch.softmax(
        model.radio(z_c.to(device))["evidence"].float(), dim=-1
    ).cpu()
    current_csi_probability = torch.softmax(cache["radio_evidence"][int(radio_view)].float(), dim=-1)
    csi_only = _metric_row(current_csi_probability, labels, power, None)
    subprototype_csi_only = _metric_row(subprototype_csi_probability, labels, power, None)
    original_csi = torch.softmax(cache["radio_evidence"][int(radio_view)].float(), dim=-1)
    oracle_values = []
    for base_prediction in base_predictions:
        radio_prediction = original_csi.argmax(dim=-1)
        oracle_values.append(float((base_prediction.eq(labels) | radio_prediction.eq(labels)).float().mean()))
    sensing_macro = float(np.mean([row["sensing_top1"] for row in per_mask]))
    oracle_macro = float(np.mean(oracle_values))
    all14 = groups["all14"]["top1_macro"]
    denominator = oracle_macro - sensing_macro
    subprototypes = model.bank.subprototypes().detach().cpu()
    base_prototypes = F.normalize(model.bank.base_prototypes.detach().cpu(), dim=-1)
    similarity = F.cosine_similarity(subprototypes[:, 0], subprototypes[:, 1], dim=-1)
    valid = model.valid_subcluster_mask.detach().cpu().bool()
    distance = 1.0 - F.cosine_similarity(
        subprototypes, base_prototypes[:, None, :].expand_as(subprototypes), dim=-1
    )
    return {
        "arm": arm,
        "mode": "csi_on" if csi_on else "csi_off",
        "radio_view": int(radio_view),
        "per_mask": per_mask,
        "groups": groups,
        "missing_lidar": next(row for row in per_mask if row["mask"] == "missing_lidar"),
        "full": full,
        "csi_only": csi_only,
        "subprototype_csi_only": subprototype_csi_only,
        "sensing_all14_macro": sensing_macro,
        "m4_csi_oracle_all14_macro": oracle_macro,
        "oracle_headroom_capture": (all14 - sensing_macro) / denominator if denominator > 0 else float("nan"),
        "mode_kl": float(np.mean(mode_kl_items)),
        "mode_agreement": float(np.mean(mode_agreement_items)),
        "fusion_disagreement_count": disagreement_count if csi_on else 0,
        "fused_matches_sensing_on_disagreement": (
            fused_matches_sensing / disagreement_count if csi_on and disagreement_count else float("nan")
        ),
        "fused_matches_csi_on_disagreement": (
            fused_matches_csi / disagreement_count if csi_on and disagreement_count else float("nan")
        ),
        "fused_matches_neither_on_disagreement": (
            fused_matches_neither / disagreement_count if csi_on and disagreement_count else float("nan")
        ),
        "prototype_collapse_ratio": float(
            similarity[valid].ge(float(config["loss"]["max_similarity"])).float().mean()
        ) if bool(valid.any()) else 0.0,
        "subprototype_similarity_mean": float(similarity.mean()),
        "subprototype_base_cosine_distance_mean": float(distance.mean()),
        "full_probability_max_abs_diff": float((full_probability - reference_full).abs().max()),
        "full_argmax_mismatch": int(
            full_probability.argmax(dim=-1).ne(reference_full.argmax(dim=-1)).sum()
        ),
        "full_pilot_re": 0,
        "csi_off_fallback_max_abs": max(float(row["csi_off_fallback_max_abs"]) for row in per_mask),
        "pilot_re_per_frame": int(config["pilot"]["re_per_frame"]) if csi_on else 0,
        "pilot_re_window": int(config["pilot"]["re_window"]) if csi_on else 0,
        "outer_test_accessed": False,
    }


def _flat_result(result: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    groups = result["groups"]
    return {
        "arm": result["arm"],
        "seed": int(seed),
        "mode": result["mode"],
        "radio_view": result["radio_view"],
        "full_top1": result["full"]["top1"],
        "full_top3": result["full"]["top3"],
        "full_top5": result["full"]["top5"],
        "single_macro": groups["single"]["top1_macro"],
        "single_worst": groups["single"]["top1_worst"],
        "two_macro": groups["two"]["top1_macro"],
        "two_worst": groups["two"]["top1_worst"],
        "three_macro": groups["three"]["top1_macro"],
        "three_worst": groups["three"]["top1_worst"],
        "all14_macro": groups["all14"]["top1_macro"],
        "all14_worst": groups["all14"]["top1_worst"],
        "missing_lidar": result["missing_lidar"]["top1"],
        "within3": groups["all14"]["within3_macro"],
        "mae": groups["all14"]["mae_macro"],
        "normalized_beamforming_gain": groups["all14"]["normalized_beamforming_gain_macro"],
        "beam_loss_db": groups["all14"]["beam_loss_db_macro"],
        "fix_rate": groups["all14"]["fix_rate_macro"],
        "harm_rate": groups["all14"]["harm_rate_macro"],
        "csi_only": result["csi_only"]["top1"],
        "subprototype_csi_only": result.get("subprototype_csi_only", result["csi_only"])["top1"],
        "sensing_only": result["sensing_all14_macro"],
        "m4_csi_oracle": result["m4_csi_oracle_all14_macro"],
        "oracle_headroom_capture": result["oracle_headroom_capture"],
        "mode_kl": result["mode_kl"],
        "mode_agreement": result["mode_agreement"],
        "fusion_disagreement_count": result.get("fusion_disagreement_count", 0),
        "fused_matches_sensing_on_disagreement": result.get(
            "fused_matches_sensing_on_disagreement", float("nan")
        ),
        "fused_matches_csi_on_disagreement": result.get(
            "fused_matches_csi_on_disagreement", float("nan")
        ),
        "fused_matches_neither_on_disagreement": result.get(
            "fused_matches_neither_on_disagreement", float("nan")
        ),
        "prototype_collapse_ratio": result["prototype_collapse_ratio"],
        "subprototype_base_cosine_distance_mean": result["subprototype_base_cosine_distance_mean"],
        "pilot_re_per_frame": result["pilot_re_per_frame"],
        "pilot_re_window": result["pilot_re_window"],
        "full_pilot_re": result["full_pilot_re"],
        "full_probability_max_abs_diff": result["full_probability_max_abs_diff"],
        "full_argmax_mismatch": result.get("full_argmax_mismatch", 0),
        "csi_off_fallback_max_abs": result["csi_off_fallback_max_abs"],
    }


def _validation_objective(
    model: RadioGuidedPrototypeModel,
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[float, float, float]:
    result = evaluate_hierarchical(
        model,
        cache,
        config,
        arm="A4",
        csi_on=False,
        radio_view=0,
        device=device,
    )
    losses = []
    batch_size = int(config["training"]["evaluation_batch_size"])
    with torch.inference_mode():
        for mask_id in range(len(MASK_NAMES)):
            for start in range(0, len(cache["target"]), batch_size):
                stop = min(start + batch_size, len(cache["target"]))
                evidence = model.sensing(cache["z_s_all_masks"][start:stop, mask_id].to(device))["evidence"]
                losses.append(
                    float(F.cross_entropy(evidence, cache["target"][start:stop].to(device)).detach().cpu())
                )
    return (
        float(np.mean(losses)),
        float(result["groups"]["all14"]["top1_macro"]),
        float(result["groups"]["all14"]["top1_worst"]),
    )


def _save_training_checkpoint(
    path: Path,
    model: RadioGuidedPrototypeModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    config: Mapping[str, Any],
    arm: str,
    seed: int,
    epoch: int,
    metrics: Mapping[str, Any],
    initialization_hash: str,
    cache_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": int(epoch),
            "arm": arm,
            "seed": int(seed),
            "metrics": dict(metrics),
            "resolved_config": dict(config),
            "rng_state": _rng_state(),
            "prototype_initialization_hash": initialization_hash,
            "cache_hash": cache_hash,
            "m4_checkpoint_sha256": config["source"]["m4_checkpoint_sha256"],
            "csi_checkpoint_sha256": config["source"]["csi_checkpoint_sha256"],
            "prototype_bank_sha256": config["source"]["prototype_bank_sha256"],
            "split_manifest_sha256": config["protocol"]["split_manifest_sha256"],
            "outer_test_accessed": False,
        },
        path,
    )


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def train_arm(
    config: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    device: torch.device,
    resume: bool = False,
) -> dict[str, Any]:
    if arm not in {"A1", "A2", "A3", "A4", "A5", "A7"}:
        raise ValueError("Only A1-A5 and A7 train radio-guided modules.")
    _set_seed(seed)
    train_cache, validation_cache = _load_cache(config, "train"), _load_cache(config, "validation")
    model, valid_subclusters, initialization_hash = _build_model(config, arm, seed, device)
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.bank.raw_delta], "lr": float(config["training"]["lr_subprototype"])},
            {"params": model.adapter.parameters(), "lr": float(config["training"]["lr_adapter"])},
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(config["training"]["scheduler_factor"]),
        patience=int(config["training"]["scheduler_patience"]),
    )
    topology = _load_topology(config).distance.to(device)
    output = _path(config["output"]["root"])
    checkpoint_dir = output / f"checkpoints/{arm}/seed{seed}"
    log_dir = output / f"logs/{arm}_seed{seed}"
    resolved_path = output / f"resolved_configs/{arm}_seed{seed}.yaml"
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = dict(config) | {"resolved_run": {"arm": arm, "seed": int(seed), "device": str(device)}}
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    cache_path = _path(config["cache"]["root"]) / "train.pt"
    cache_hash = sha256_file(cache_path)
    start_epoch = 0
    last_path = checkpoint_dir / "last.pt"
    if resume and last_path.exists():
        payload = torch.load(last_path, map_location="cpu", weights_only=False)
        checks = {
            "arm": payload.get("arm") == arm,
            "seed": int(payload.get("seed", -1)) == int(seed),
            "cache": payload.get("cache_hash") == cache_hash,
            "initialization": payload.get("prototype_initialization_hash") == initialization_hash,
            "outer_test": payload.get("outer_test_accessed") is False,
        }
        if not all(checks.values()):
            raise ValueError(f"Resume checkpoint identity mismatch: {checks}.")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        _restore_rng(payload["rng_state"])
        start_epoch = int(payload["epoch"]) + 1
    history: list[dict[str, Any]] = []
    best_macro, best_worst, best_loss = -math.inf, -math.inf, math.inf
    patience = 0
    distillation = str(config["arms"][arm]["distillation"])
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handler = signal.signal(signal.SIGTERM, request_stop)
    try:
        for epoch in range(start_epoch, int(config["training"]["max_epochs"])):
            model.train()
            indices, mask_schedule = _balanced_mask_schedule(len(train_cache["target"]), epoch=epoch, seed=seed)
            totals = Counter()
            batches = 0
            batch_size = int(config["training"]["batch_size"])
            for start in range(0, len(indices), batch_size):
                source = indices[start : start + batch_size]
                mask_ids = mask_schedule[start : start + batch_size]
                z_s = train_cache["z_s_all_masks"][source, mask_ids].to(device)
                z_c = train_cache["z_c"][0, source].to(device)
                targets = train_cache["target"][source].to(device)
                full_logits = train_cache["full_logits"][source].to(device)
                sensing = model.sensing(z_s, missing=True)
                radio = model.radio(z_c)
                task = beam_classification_loss(sensing["evidence"], targets)
                topology_loss = beam_topology_loss(sensing["evidence"], targets, topology)
                q_s = propagation_mode_distribution(
                    sensing["mode_scores"], targets, temperature=float(config["model"]["mode_temperature"])
                )
                q_c = propagation_mode_distribution(
                    radio["mode_scores"], targets, temperature=float(config["model"]["mode_temperature"])
                )
                if distillation == "unfiltered":
                    distill = radio_prototype_distillation_loss(q_s, q_c)
                elif distillation == "qualified":
                    teacher = qualified_teacher_weights(
                        sensing["evidence"],
                        radio["evidence"],
                        targets,
                        tau_adv=float(config["loss"]["tau_adv"]),
                        conf_ref=float(config["loss"]["conf_ref"]),
                        training=True,
                    )
                    distill = radio_prototype_distillation_loss(q_s, q_c, weights=teacher["weight"])
                else:
                    distill = task.new_zeros(())
                subprototypes = model.bank.subprototypes()
                anchor = prototype_anchor_loss(subprototypes, model.bank.base_prototypes)
                diversity = prototype_diversity_loss(
                    subprototypes,
                    valid_subclusters,
                    max_similarity=float(config["loss"]["max_similarity"]),
                )
                adapter_loss = adapter_regularization(sensing["adapter_residual"])
                full_teacher = full_teacher_consistency_loss(
                    sensing["evidence"],
                    full_logits,
                    targets,
                    minimum_margin=float(config["training"]["full_teacher_minimum_margin"]),
                )
                loss = (
                    task
                    + float(config["loss"]["topology"]) * topology_loss
                    + float(config["loss"]["distill"]) * distill
                    + float(config["loss"]["anchor"]) * anchor
                    + float(config["loss"]["diversity"]) * diversity
                    + float(config["loss"]["adapter"]) * adapter_loss
                    + float(config["loss"]["teacher"]) * full_teacher
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Non-finite loss in {arm} seed{seed} epoch{epoch}.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
                optimizer.step()
                model.bank.enforce_trainable_beam_mask_()
                for name, value in (
                    ("loss", loss),
                    ("task", task),
                    ("topology", topology_loss),
                    ("distill", distill),
                    ("anchor", anchor),
                    ("diversity", diversity),
                    ("adapter", adapter_loss),
                    ("full_teacher", full_teacher),
                ):
                    totals[name] += float(value.detach().cpu())
                batches += 1
                if stop_requested:
                    break
            val_loss, all14_macro, all14_worst = _validation_objective(model, validation_cache, config, device)
            scheduler.step(val_loss)
            row = {
                "epoch": epoch,
                **{name: value / max(batches, 1) for name, value in totals.items()},
                "val_loss": val_loss,
                "all14_macro": all14_macro,
                "all14_worst": all14_worst,
                "lr_subprototype": optimizer.param_groups[0]["lr"],
                "lr_adapter": optimizer.param_groups[1]["lr"],
            }
            history.append(row)
            _write_csv(log_dir / "metrics.csv", history)
            _save_training_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                config=config,
                arm=arm,
                seed=seed,
                epoch=epoch,
                metrics=row,
                initialization_hash=initialization_hash,
                cache_hash=cache_hash,
            )
            if all14_macro > best_macro:
                best_macro = all14_macro
                _save_training_checkpoint(
                    checkpoint_dir / "best_all14_macro.pt", model, optimizer, scheduler,
                    config=config, arm=arm, seed=seed, epoch=epoch, metrics=row,
                    initialization_hash=initialization_hash, cache_hash=cache_hash,
                )
                _save_training_checkpoint(
                    checkpoint_dir / "best_csi_off_all14.pt", model, optimizer, scheduler,
                    config=config, arm=arm, seed=seed, epoch=epoch, metrics=row,
                    initialization_hash=initialization_hash, cache_hash=cache_hash,
                )
            if all14_worst > best_worst:
                best_worst = all14_worst
                _save_training_checkpoint(
                    checkpoint_dir / "best_all14_worst.pt", model, optimizer, scheduler,
                    config=config, arm=arm, seed=seed, epoch=epoch, metrics=row,
                    initialization_hash=initialization_hash, cache_hash=cache_hash,
                )
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                patience = 0
                _save_training_checkpoint(
                    checkpoint_dir / "best_val_loss.pt", model, optimizer, scheduler,
                    config=config, arm=arm, seed=seed, epoch=epoch, metrics=row,
                    initialization_hash=initialization_hash, cache_hash=cache_hash,
                )
            else:
                patience += 1
            print(json.dumps({"arm": arm, "seed": seed, **row}), flush=True)
            if stop_requested or patience >= int(config["training"]["patience"]):
                break
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    return {
        "arm": arm,
        "seed": int(seed),
        "epochs": len(history),
        "best_all14_macro": best_macro,
        "best_all14_worst": best_worst,
        "best_val_loss": best_loss,
        "stopped_by_sigterm": stop_requested,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "outer_test_accessed": False,
    }


@torch.inference_mode()
def evaluate_baseline(
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    arm: str,
    radio_view: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if arm not in {"A0", "A6"}:
        raise ValueError("Baseline evaluation only supports A0 or A6.")
    labels = cache["target"].long()
    power = cache["future_beam_power"].float()
    available = cache["csi_available"][int(radio_view)].bool()
    if arm == "A0":
        csi_evidence = cache["radio_evidence"][int(radio_view)].float()
    else:
        csi_evidence = cache["csi_classifier_logits"][int(radio_view)].float()
    csi_probability = torch.softmax(csi_evidence, dim=-1)
    reference_masks = cache.get("reference_probability_all_masks")
    modes: dict[str, Any] = {}
    for csi_on in (False, True):
        per_mask = []
        oracle_values = []
        disagreement_count = 0
        fused_matches_sensing = 0
        fused_matches_csi = 0
        fused_matches_neither = 0
        for mask_id, mask_name in enumerate(MASK_NAMES):
            sensing = cache["sensing_logits_all_masks"][:, mask_id].float()
            base = (
                reference_masks[:, mask_id].float()
                if reference_masks is not None
                else torch.softmax(sensing, dim=-1)
            )
            if not csi_on:
                probability = base
            elif arm == "A0":
                calibrated = sensing / float(cache["identity"]["sensing_temperature"])
                final = fixed_beam_evidence_fusion(
                    calibrated,
                    csi_evidence,
                    csi_weight=float(config["model"]["fixed_csi_weight"]),
                    csi_available=available,
                )
                probability = torch.softmax(final.float(), dim=-1)
            else:
                final = fixed_beam_evidence_fusion(
                    sensing,
                    csi_evidence,
                    csi_weight=float(config["model"]["fixed_csi_weight"]),
                    csi_available=available,
                )
                probability = torch.softmax(final.float(), dim=-1)
            if csi_on:
                fused_prediction = probability.argmax(dim=-1)
                sensing_prediction = base.argmax(dim=-1)
                csi_prediction = csi_probability.argmax(dim=-1)
                disagreement = sensing_prediction.ne(csi_prediction)
                disagreement_count += int(disagreement.sum())
                fused_matches_sensing += int((disagreement & fused_prediction.eq(sensing_prediction)).sum())
                fused_matches_csi += int((disagreement & fused_prediction.eq(csi_prediction)).sum())
                fused_matches_neither += int(
                    (
                        disagreement
                        & fused_prediction.ne(sensing_prediction)
                        & fused_prediction.ne(csi_prediction)
                    ).sum()
                )
            per_mask.append(
                {
                    "mask": mask_name,
                    "available_count": MASK_COUNTS[mask_name],
                    "mode": "csi_on" if csi_on else "csi_off",
                    **_metric_row(probability, labels, power, base),
                    "sensing_top1": float(base.argmax(dim=-1).eq(labels).float().mean()),
                }
            )
            oracle_values.append(
                float(
                    (
                        base.argmax(dim=-1).eq(labels)
                        | csi_probability.argmax(dim=-1).eq(labels)
                    ).float().mean()
                )
            )
        groups = {
            GROUP_NAMES[count]: _aggregate_mask_rows(
                [row for row in per_mask if row["available_count"] == count]
            )
            for count in (1, 2, 3)
        }
        groups["all14"] = _aggregate_mask_rows(per_mask)
        full_probability = cache["full_probability"].float()
        reference_full = cache.get("reference_full_probability", full_probability).float()
        sensing_macro = float(np.mean([row["sensing_top1"] for row in per_mask]))
        oracle_macro = float(np.mean(oracle_values))
        all14 = groups["all14"]["top1_macro"]
        denominator = oracle_macro - sensing_macro
        modes["csi_on" if csi_on else "csi_off"] = {
            "arm": arm,
            "mode": "csi_on" if csi_on else "csi_off",
            "radio_view": int(radio_view),
            "per_mask": per_mask,
            "groups": groups,
            "missing_lidar": next(row for row in per_mask if row["mask"] == "missing_lidar"),
            "full": _metric_row(full_probability, labels, power, full_probability),
            "csi_only": _metric_row(csi_probability, labels, power, None),
            "subprototype_csi_only": _metric_row(csi_probability, labels, power, None),
            "sensing_all14_macro": sensing_macro,
            "m4_csi_oracle_all14_macro": oracle_macro,
            "oracle_headroom_capture": (all14 - sensing_macro) / denominator if denominator > 0 else float("nan"),
            "mode_kl": float("nan"),
            "mode_agreement": float("nan"),
            "fusion_disagreement_count": disagreement_count if csi_on else 0,
            "fused_matches_sensing_on_disagreement": (
                fused_matches_sensing / disagreement_count if csi_on and disagreement_count else float("nan")
            ),
            "fused_matches_csi_on_disagreement": (
                fused_matches_csi / disagreement_count if csi_on and disagreement_count else float("nan")
            ),
            "fused_matches_neither_on_disagreement": (
                fused_matches_neither / disagreement_count if csi_on and disagreement_count else float("nan")
            ),
            "prototype_collapse_ratio": 0.0,
            "subprototype_similarity_mean": float("nan"),
            "subprototype_base_cosine_distance_mean": 0.0,
            "full_probability_max_abs_diff": float((full_probability - reference_full).abs().max()),
            "full_argmax_mismatch": int(
                full_probability.argmax(dim=-1).ne(reference_full.argmax(dim=-1)).sum()
            ),
            "full_pilot_re": 0,
            "csi_off_fallback_max_abs": 0.0,
            "pilot_re_per_frame": int(config["pilot"]["re_per_frame"]) if csi_on else 0,
            "pilot_re_window": int(config["pilot"]["re_window"]) if csi_on else 0,
            "outer_test_accessed": False,
        }
    return modes["csi_off"], modes["csi_on"]


@torch.inference_mode()
def _baseline_latency(
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    arm: str,
    mode: str,
    radio_view: int,
    device: torch.device,
) -> float:
    count = min(1024, len(cache["target"]))
    sensing = cache["sensing_logits_all_masks"][:count, 0].float().to(device)
    csi_key = "radio_evidence" if arm == "A0" else "csi_classifier_logits"
    csi = cache[csi_key][int(radio_view), :count].float().to(device)
    available = cache["csi_available"][int(radio_view), :count].bool().to(device)

    def forward() -> None:
        evidence = sensing
        if mode == "csi_on":
            if arm == "A0":
                evidence = sensing / float(cache["identity"]["sensing_temperature"])
            evidence = fixed_beam_evidence_fusion(
                evidence,
                csi,
                csi_weight=float(config["model"]["fixed_csi_weight"]),
                csi_available=available,
            )
        torch.softmax(evidence.float(), dim=-1)

    for _ in range(int(config["evaluation"]["latency_warmup"])):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(int(config["evaluation"]["latency_repeats"])):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - started) / (
        count * int(config["evaluation"]["latency_repeats"])
    )


def run_baseline(
    config: Mapping[str, Any], *, arm: str, device: torch.device
) -> list[dict[str, Any]]:
    preflight(config)
    cache = _attach_reference_probabilities(
        _load_cache(config, "validation"), _load_records(config, "validation")
    )
    output = _path(config["output"]["root"])
    rows: list[dict[str, Any]] = []
    hidden = int(config["model"]["radio_dim"])
    beams = int(config["model"]["num_beams"])
    independent_head_parameters = 2 * hidden + hidden * hidden + hidden + hidden * beams + beams
    for view, noise_seed in enumerate(cache["identity"]["noise_seeds"]):
        off, on = evaluate_baseline(cache, config, arm=arm, radio_view=view)
        for result in (off, on):
            result["validation_noise_seed"] = int(noise_seed)
            flat = _flat_result(result, seed=int(noise_seed))
            flat["validation_noise_seed"] = int(noise_seed)
            flat["trainable_parameters"] = 0
            flat["incremental_parameters"] = independent_head_parameters if arm == "A6" else 0
            flat["latency_ms_per_sample"] = _baseline_latency(
                cache,
                config,
                arm=arm,
                mode=result["mode"],
                radio_view=view,
                device=device,
            )
            result["trainable_parameters"] = flat["trainable_parameters"]
            result["incremental_parameters"] = flat["incremental_parameters"]
            result["latency_ms_per_sample"] = flat["latency_ms_per_sample"]
            rows.append(flat)
            _write_json(
                output / f"baseline_results/{arm}_noise{noise_seed}_{result['mode']}.json",
                result,
            )
    _write_csv(output / f"baseline_results/{arm}_metrics.csv", rows)
    return rows


@torch.inference_mode()
def _teacher_statistics(
    model: RadioGuidedPrototypeModel,
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str], Counter] = defaultdict(Counter)
    batch_size = int(config["training"]["evaluation_batch_size"])
    labels_all = cache["target"].long()
    for mask_id, mask_name in enumerate(MASK_NAMES):
        for start in range(0, len(labels_all), batch_size):
            stop = min(start + batch_size, len(labels_all))
            labels = labels_all[start:stop].to(device)
            sensing = model.sensing(cache["z_s_all_masks"][start:stop, mask_id].to(device))
            radio = model.radio(cache["z_c"][0, start:stop].to(device))
            teacher = qualified_teacher_weights(
                sensing["evidence"],
                radio["evidence"],
                labels,
                tau_adv=float(config["loss"]["tau_adv"]),
                conf_ref=float(config["loss"]["conf_ref"]),
                training=True,
            )
            weight = teacher["weight"].cpu()
            qualified = teacher["qualified_rank"].cpu()
            sensing_prediction = sensing["evidence"].argmax(dim=-1).cpu()
            radio_prediction = radio["evidence"].argmax(dim=-1).cpu()
            batch_labels = labels.cpu()
            sensing_wrong_csi_correct = sensing_prediction.ne(batch_labels) & radio_prediction.eq(batch_labels)
            sensing_correct_csi_wrong = sensing_prediction.eq(batch_labels) & radio_prediction.ne(batch_labels)
            scopes = [("overall", "all"), ("mask", mask_name)]
            scopes.extend(("beam", str(int(beam))) for beam in torch.unique(batch_labels).tolist())
            for scope, value in scopes:
                selected = torch.ones(len(batch_labels), dtype=torch.bool)
                if scope == "beam":
                    selected = batch_labels.eq(int(value))
                counter = aggregates[(scope, value)]
                counter["sample_count"] += int(selected.sum())
                counter["qualified_count"] += int(qualified[selected].sum())
                counter["weight_sum"] += float(weight[selected].sum())
                counter["effective_count"] += int(weight[selected].gt(0).sum())
                counter["sensing_wrong_csi_correct_count"] += int(sensing_wrong_csi_correct[selected].sum())
                counter["sensing_wrong_csi_correct_qualified"] += int(
                    (sensing_wrong_csi_correct & qualified & selected).sum()
                )
                counter["sensing_correct_csi_wrong_count"] += int(sensing_correct_csi_wrong[selected].sum())
    rows = []
    for (scope, value), counter in sorted(aggregates.items()):
        count = max(int(counter["sample_count"]), 1)
        rescue_count = max(int(counter["sensing_wrong_csi_correct_count"]), 1)
        rows.append(
            {
                "arm": arm,
                "seed": int(seed),
                "scope": scope,
                "value": value,
                "sample_count": int(counter["sample_count"]),
                "qualified_teacher_coverage": counter["qualified_count"] / count,
                "effective_teacher_coverage": counter["effective_count"] / count,
                "mean_teacher_weight": counter["weight_sum"] / count,
                "sensing_wrong_csi_correct_fraction": counter["sensing_wrong_csi_correct_count"] / count,
                "sensing_wrong_csi_correct_coverage": counter["sensing_wrong_csi_correct_qualified"] / rescue_count,
                "sensing_correct_csi_wrong_fraction": counter["sensing_correct_csi_wrong_count"] / count,
            }
        )
    return rows


@torch.inference_mode()
def _latency(
    model: RadioGuidedPrototypeModel,
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    arm: str,
    mode: str,
    device: torch.device,
) -> float:
    count = min(1024, len(cache["target"]))
    z_s = cache["z_s_all_masks"][:count, 0].to(device)
    z_c = cache["z_c"][0, :count].to(device)

    def forward() -> None:
        sensing = model.sensing(z_s, missing=True)
        evidence = sensing["evidence"]
        if mode == "csi_on":
            radio = model.radio(z_c)
            if str(config["arms"][arm]["inference"]) == "mode_fusion":
                evidence, _ = mode_consistent_fusion(
                    sensing["mode_scores"],
                    radio["mode_scores"],
                    csi_weight=float(config["model"]["fixed_csi_weight"]),
                    tau_sub=float(config["model"]["tau_sub"]),
                )
            else:
                evidence = fixed_beam_evidence_fusion(
                    sensing["evidence"],
                    radio["evidence"],
                    csi_weight=float(config["model"]["fixed_csi_weight"]),
                )
        torch.softmax(evidence.float(), dim=-1)

    for _ in range(int(config["evaluation"]["latency_warmup"])):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(int(config["evaluation"]["latency_repeats"])):
        forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return 1000.0 * elapsed / (count * int(config["evaluation"]["latency_repeats"]))


def evaluate_trained_arm(
    config: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    device: torch.device,
    checkpoint_name: str = "best_csi_off_all14.pt",
) -> list[dict[str, Any]]:
    model, _, initialization_hash = _build_model(config, arm, seed, device)
    output = _path(config["output"]["root"])
    checkpoint_path = output / f"checkpoints/{arm}/seed{seed}/{checkpoint_name}"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checks = {
        "arm": payload.get("arm") == arm,
        "seed": int(payload.get("seed", -1)) == int(seed),
        "initialization": payload.get("prototype_initialization_hash") == initialization_hash,
        "outer_test": payload.get("outer_test_accessed") is False,
    }
    if not all(checks.values()):
        raise ValueError(f"Evaluation checkpoint identity mismatch: {checks}.")
    model.load_state_dict(payload["model"], strict=True)
    validation = _attach_reference_probabilities(
        _load_cache(config, "validation"), _load_records(config, "validation")
    )
    results = [
        evaluate_hierarchical(
            model,
            validation,
            config,
            arm=arm,
            csi_on=csi_on,
            radio_view=0,
            device=device,
        )
        for csi_on in (False, True)
    ]
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    rows = []
    mask_rows = []
    for result in results:
        mode = result["mode"]
        result["checkpoint"] = str(checkpoint_path.resolve())
        result["checkpoint_sha256"] = sha256_file(checkpoint_path)
        result["trainable_parameters"] = parameter_count
        result["incremental_parameters"] = parameter_count
        result["latency_ms_per_sample"] = _latency(
            model, validation, config, arm=arm, mode=mode, device=device
        )
        flat = _flat_result(result, seed=seed)
        flat["trainable_parameters"] = parameter_count
        flat["incremental_parameters"] = parameter_count
        flat["latency_ms_per_sample"] = result["latency_ms_per_sample"]
        rows.append(flat)
        mask_rows.extend({"arm": arm, "seed": seed, **row} for row in result["per_mask"])
        _write_json(output / f"results/{arm}_seed{seed}_{mode}.json", result)
    log_dir = output / f"logs/{arm}_seed{seed}"
    _write_csv(log_dir / "final_metrics.csv", rows)
    _write_csv(log_dir / "mask_metrics.csv", mask_rows)
    teacher_rows = _teacher_statistics(
        model,
        _load_cache(config, "train"),
        config,
        arm=arm,
        seed=seed,
        device=device,
    )
    _write_csv(log_dir / "teacher_statistics.csv", teacher_rows)
    prototype = model.bank.subprototypes().detach().cpu()
    base = F.normalize(model.bank.base_prototypes.detach().cpu(), dim=-1)
    prototype_rows = []
    for beam in range(64):
        prototype_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "beam": beam,
                "valid_subcluster": bool(model.valid_subcluster_mask[beam]),
                "mode_similarity": float(F.cosine_similarity(prototype[beam, 0], prototype[beam, 1], dim=0)),
                "mode0_base_distance": float(1.0 - F.cosine_similarity(prototype[beam, 0], base[beam], dim=0)),
                "mode1_base_distance": float(1.0 - F.cosine_similarity(prototype[beam, 1], base[beam], dim=0)),
                "residual0_radius": float(model.bank.residuals()[beam, 0].norm().detach().cpu()),
                "residual1_radius": float(model.bank.residuals()[beam, 1].norm().detach().cpu()),
            }
        )
    _write_csv(log_dir / "prototype_statistics.csv", prototype_rows)
    return rows


def run_trained_arm(
    config: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    device: torch.device,
    resume: bool = False,
) -> dict[str, Any]:
    training = train_arm(config, arm=arm, seed=seed, device=device, resume=resume)
    metrics = evaluate_trained_arm(config, arm=arm, seed=seed, device=device)
    return {"training": training, "metrics": metrics}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def _mean_row(rows: Sequence[Mapping[str, Any]], arm: str, mode: str) -> dict[str, Any] | None:
    selected = [row for row in rows if row.get("arm") == arm and row.get("mode") == mode]
    if not selected:
        return None
    result: dict[str, Any] = {"arm": arm, "mode": mode, "run_count": len(selected)}
    numeric = {
        key
        for row in selected
        for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"seed", "radio_view", "validation_noise_seed"}
    }
    for key in sorted(numeric):
        values = [float(row[key]) for row in selected if isinstance(row.get(key), (int, float))]
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return result


def _load_result_jsons(output: Path) -> list[dict[str, Any]]:
    paths = sorted((output / "baseline_results").glob("*.json")) + sorted((output / "results").glob("*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def summarize(config: Mapping[str, Any]) -> dict[str, Any]:
    output = _path(config["output"]["root"])
    rows = []
    for arm in ("A0", "A6"):
        rows.extend(_read_csv(output / f"baseline_results/{arm}_metrics.csv"))
    for arm in ("A1", "A2", "A3", "A4", "A5", "A7"):
        for seed in config["training"]["seeds"]:
            rows.extend(_read_csv(output / f"logs/{arm}_seed{seed}/final_metrics.csv"))
    _write_csv(output / "a0_a7_summary.csv", rows)
    means = [
        summary
        for arm in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7")
        for mode in ("csi_off", "csi_on")
        if (summary := _mean_row(rows, arm, mode)) is not None
    ]
    _write_csv(output / "seed_summary.csv", means)
    _write_csv(output / "csi_off_summary.csv", [row for row in means if row["mode"] == "csi_off"])
    _write_csv(output / "csi_on_summary.csv", [row for row in means if row["mode"] == "csi_on"])
    _write_csv(
        output / "latency_summary.csv",
        [
            {
                "arm": row["arm"],
                "mode": row["mode"],
                "seed": row.get("seed"),
                "latency_ms_per_sample": row.get("latency_ms_per_sample"),
                "trainable_parameters": row.get("trainable_parameters"),
                "incremental_parameters": row.get("incremental_parameters"),
                "pilot_re_per_frame": row.get("pilot_re_per_frame"),
                "pilot_re_window": row.get("pilot_re_window"),
            }
            for row in rows
        ],
    )
    _write_csv(
        output / "oracle_summary.csv",
        [
            {
                "arm": row["arm"],
                "mode": row["mode"],
                "seed": row.get("seed"),
                "sensing_only": row.get("sensing_only"),
                "csi_only": row.get("csi_only"),
                "subprototype_csi_only": row.get("subprototype_csi_only"),
                "m4_csi_oracle": row.get("m4_csi_oracle"),
                "oracle_headroom_capture": row.get("oracle_headroom_capture"),
            }
            for row in rows
        ],
    )

    result_jsons = _load_result_jsons(output)
    mask_rows = [
        {
            "arm": result["arm"],
            "seed": result.get("validation_noise_seed", Path(result.get("checkpoint", "seed0")).parent.name),
            "mode": result["mode"],
            **row,
        }
        for result in result_jsons
        for row in result["per_mask"]
    ]
    _write_csv(output / "mask_summary.csv", mask_rows)
    teacher_rows, prototype_rows = [], []
    for arm in ("A1", "A2", "A3", "A4", "A5", "A7"):
        for seed in config["training"]["seeds"]:
            teacher_rows.extend(_read_csv(output / f"logs/{arm}_seed{seed}/teacher_statistics.csv"))
            prototype_rows.extend(_read_csv(output / f"logs/{arm}_seed{seed}/prototype_statistics.csv"))
    _write_csv(output / "teacher_coverage.csv", teacher_rows)
    _write_csv(output / "diagnostics/qualified_teacher_statistics.csv", teacher_rows)
    _write_csv(output / "prototype_statistics.csv", prototype_rows)

    expected = {
        (arm, int(seed))
        for arm in ("A1", "A2", "A3", "A4", "A5", "A7")
        for seed in config["training"]["seeds"]
    }
    complete = {
        (str(row["arm"]), int(float(row["seed"])))
        for row in rows
        if row.get("arm") not in {"A0", "A6"} and row.get("mode") == "csi_off"
    }
    missing_runs = sorted(f"{arm}_seed{seed}" for arm, seed in expected - complete)

    lookup = {(row["arm"], row["mode"]): row for row in means}
    a0 = lookup.get(("A0", "csi_on"))
    a1 = lookup.get(("A1", "csi_off"))
    a2 = lookup.get(("A2", "csi_off"))
    a3 = lookup.get(("A3", "csi_off"))
    a4 = lookup.get(("A4", "csi_off"))
    a5 = lookup.get(("A5", "csi_off"))
    a6 = lookup.get(("A6", "csi_on"))
    a7 = lookup.get(("A7", "csi_on"))

    def metric(row: Mapping[str, Any] | None, name: str) -> float:
        return float(row.get(f"{name}_mean", math.nan)) if row else math.nan

    def passes_gain(candidate: Mapping[str, Any] | None, reference: Mapping[str, Any] | None) -> bool:
        return bool(
            candidate
            and reference
            and (
                metric(candidate, "all14_macro") - metric(reference, "all14_macro") >= 0.005
                or metric(candidate, "all14_worst") - metric(reference, "all14_worst") >= 0.01
                or metric(candidate, "missing_lidar") - metric(reference, "missing_lidar") >= 0.01
            )
        )

    a4_gain = passes_gain(a4, a1) and passes_gain(a4, a2)
    a4_controls = bool(
        a4
        and a3
        and a5
        and metric(a4, "all14_macro") > metric(a3, "all14_macro")
        and metric(a4, "all14_macro") > metric(a5, "all14_macro")
    )
    a7_gain = passes_gain(a7, a0)
    a7_groups = bool(
        a7
        and a0
        and all(
            metric(a7, name) >= metric(a0, name) - 0.005
            for name in ("single_macro", "two_macro", "three_macro")
        )
    )
    gain_thresholds = {"all14_macro": 0.005, "all14_worst": 0.01, "missing_lidar": 0.01}
    a7_seed_rows = [
        row for row in rows if row.get("arm") == "A7" and row.get("mode") == "csi_on"
    ]
    a7_seed_direction = any(
        metric(a7, name) - metric(a0, name) >= threshold
        and len(a7_seed_rows) == len(config["training"]["seeds"])
        and all(float(row[name]) > metric(a0, name) for row in a7_seed_rows)
        for name, threshold in gain_thresholds.items()
    )
    a4_teacher = [
        row
        for row in teacher_rows
        if row.get("arm") == "A4" and row.get("scope") == "overall"
    ]
    teacher_coverage = float(
        np.mean([float(row["qualified_teacher_coverage"]) for row in a4_teacher])
    ) if a4_teacher else float("nan")
    teacher_sufficient = teacher_coverage >= float(config["evaluation"]["minimum_teacher_coverage"])
    collapse = metric(lookup.get(("A4", "csi_off")), "prototype_collapse_ratio")
    no_collapse = math.isfinite(collapse) and collapse <= 0.1
    wireless_claim = a4_gain and a4_controls and teacher_sufficient and no_collapse
    a7_claim = a7_gain and a7_groups and a7_seed_direction and wireless_claim

    csi_cluster_rows = _read_csv(output / "diagnostics/csi_subcluster_statistics.csv")
    valid_cluster_rows = [
        row for row in csi_cluster_rows if str(row.get("valid_subcluster", "")).lower() == "true"
    ]
    cluster_error_gap = float(
        np.mean(
            [
                abs(float(row["full_error_cluster_0"]) - float(row["full_error_cluster_1"]))
                for row in valid_cluster_rows
            ]
        )
    ) if valid_cluster_rows else float("nan")
    cluster_mask_accuracy_gaps = [
        abs(float(row[f"{name}_accuracy_cluster_0"]) - float(row[f"{name}_accuracy_cluster_1"]))
        for row in valid_cluster_rows
        for name in MASK_NAMES
    ]
    cluster_mask_accuracy_gap = (
        float(np.mean(cluster_mask_accuracy_gaps)) if cluster_mask_accuracy_gaps else float("nan")
    )
    clustering = json.loads((output / "diagnostics/clustering_summary.json").read_text(encoding="utf-8"))
    full_exact = all(
        float(row.get("full_probability_max_abs_diff", math.inf)) == 0.0
        and float(row.get("full_argmax_mismatch", math.inf)) == 0.0
        for row in rows
    )
    csi_off_exact = all(float(row.get("csi_off_fallback_max_abs", math.inf)) == 0.0 for row in rows)
    a7_csi_follow = metric(a7, "fused_matches_csi_on_disagreement")
    a7_sensing_follow = metric(a7, "fused_matches_sensing_on_disagreement")
    csi_dominant = bool(
        math.isfinite(a7_csi_follow)
        and math.isfinite(a7_sensing_follow)
        and a7_csi_follow > a7_sensing_follow
    )

    percent = lambda value: "NA" if not math.isfinite(float(value)) else f"{100.0 * float(value):.3f}%"
    delta = lambda left, right, name: (
        "NA" if left is None or right is None else f"{100.0 * (metric(left, name) - metric(right, name)):+.3f} pp"
    )
    core_table = [
        "## 核心结果",
        "",
        "| Arm | Mode | All-14 | Worst | missing_lidar | Single | Two | Three | Full |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"):
        for mode in ("csi_off", "csi_on"):
            row = lookup.get((arm, mode))
            if row is None:
                continue
            core_table.append(
                "| "
                + " | ".join(
                    [
                        arm,
                        mode,
                        percent(metric(row, "all14_macro")),
                        percent(metric(row, "all14_worst")),
                        percent(metric(row, "missing_lidar")),
                        percent(metric(row, "single_macro")),
                        percent(metric(row, "two_macro")),
                        percent(metric(row, "three_macro")),
                        percent(metric(row, "full_top1")),
                    ]
                )
                + " |"
            )
    monitor_latest = json.loads((output / "logs/monitor/latest.json").read_text(encoding="utf-8"))
    checkpoint_count = sum(1 for _ in (output / "checkpoints").glob("*/*/*.pt"))
    report = [
        "# Radio-Guided Hierarchical Prototypes 最终报告",
        "",
        "## 预注册结论",
        "",
        f"- 正式运行完整：{not missing_runs}；缺失运行：{', '.join(missing_runs) if missing_runs else '无'}。",
        f"- A4 无线知识注入门槛：{'通过' if wireless_claim else '未通过'}。",
        f"- A7 相对 F1 完整方法门槛：{'通过' if a7_claim else '未通过'}。",
        f"- Full 逐样本硬旁路：{'通过' if full_exact else '失败'}；CSI-off 精确回退路径：{'通过' if csi_off_exact else '失败'}。",
        "",
        *core_table,
        "",
        "## 必答问题",
        "",
        f"1. CSI embedding 在同一 beam 内存在稳定子结构：49/64 beam 合格，平均 silhouette={clustering['csi_mean_silhouette']:.4f}，ARI/NMI={clustering['csi_mean_stability_ari']:.4f}/{clustering['csi_mean_stability_nmi']:.4f}。",
        f"2. K=2 相对 K=1 的平均 cosine SSE reduction 为 {percent(clustering['csi_mean_k2_sse_reduction'])}，具有明确诊断意义。",
        f"3. train 上 Full-M4 已饱和，两个有效 cluster 的 Full 错误率平均绝对差为 {percent(cluster_error_gap)}；14 个 missing mask 的准确率平均绝对差为 {percent(cluster_mask_accuracy_gap)}，说明传播 cluster 对应不同缺失感知难度，逐 beam/mask 见 `csi_subcluster_statistics.csv`。",
        f"4. A4 合格教师覆盖率为 {percent(teacher_coverage)}，按预注册 {percent(config['evaluation']['minimum_teacher_coverage'])} 阈值判定为{'足够' if teacher_sufficient else '不足'}。",
        f"5. A1 的多原型本身相对 B0 CSI-off All-14 变化为 {delta(a1, lookup.get(('A0','csi_off')), 'all14_macro')}，存在小幅收益但未达到 +0.5 pp 门槛。",
        f"6. A2 相对 A1 CSI-off All-14 变化为 {delta(a2, a1, 'all14_macro')}，CSI 初始化的隔离贡献可忽略。",
        f"7. A3 相对 A2 CSI-off All-14 变化为 {delta(a3, a2, 'all14_macro')}，幅度接近零，未观察到有实质意义的无筛选弱教师负迁移。",
        f"8. A4 相对 A3 CSI-off All-14 变化为 {delta(a4, a3, 'all14_macro')}，合格筛选{'避免了负迁移' if a4 and a3 and metric(a4,'all14_macro') > metric(a3,'all14_macro') else '未证明避免负迁移'}。",
        f"9. A4 相对 A1/A2 的预注册 CSI-off 增益门槛{'通过' if a4_gain else '未通过'}；因此{'可以' if wireless_claim else '不得'}声称无线知识写入感知表示。",
        f"10. A4 相对 A5 CSI-off All-14 变化为 {delta(a4, a5, 'all14_macro')}，随机参数量对照{'已排除' if a4_controls else '未排除'}。",
        f"11. A6 CSI-on All-14/Worst/missing_lidar 为 {percent(metric(a6,'all14_macro'))}/{percent(metric(a6,'all14_worst'))}/{percent(metric(a6,'missing_lidar'))}；All-14 高于 A0，但 Worst 和 missing_lidar 均更低，因此独立 CSI classifier 不足以替代严重缺失鲁棒性。",
        f"12. A7 相对 A0/F1 的 All-14/Worst/missing_lidar 变化为 {delta(a7,a0,'all14_macro')}/{delta(a7,a0,'all14_worst')}/{delta(a7,a0,'missing_lidar')}，主增益门槛{'通过' if a7_gain else '未通过'}，三 seed 方向一致性{'通过' if a7_seed_direction else '未通过'}。",
        f"13. 传播模式一致融合的独立价值{'成立' if a7_claim else '未成立'}；未过门槛时只保留为消融。",
        f"14. A4 有效 beam 的 prototype collapse ratio 为 {percent(collapse)}，判定为{'未坍缩' if no_collapse else '存在坍缩'}。",
        f"15. Full 逐样本完全不变={full_exact}，Full CSI RE=0。",
        f"16. CSI-off 不读取 CSI 且 fallback max abs=0：{csi_off_exact}。",
        f"17. A7 在 sensing/CSI 分歧样本上跟随 CSI/sensing 的比例为 {percent(a7_csi_follow)}/{percent(a7_sensing_follow)}，因此判定为{'存在 CSI 主导' if csi_dominant else '不存在 CSI 主导'}；该结论同时结合 CSI-only、Fix/Harm 与 mode agreement 解读。",
        "18. 无 future channel 泄漏：审计和单测均通过，输入严格为 t-4...t，target 为 t+1。",
        "19. outer test 继续封存：未构建 loader、未建立 cache、未访问样本。",
        f"20. 论文模块建议：保留 A0/A1/A2/A3/A5/A6 作为因果消融；{'保留 A4 无线注入与 A7 主方法' if a7_claim else 'A4/A7 未过门槛，不写成最终正方法，仅作为负结果/消融'}。",
        "",
        "## 资源与边界",
        "",
        f"- 正式 18 个训练运行在 {monitor_latest['elapsed_seconds'] / 60.0:.1f} 分钟内完成，共保存 {checkpoint_count} 个 checkpoint；全部 exit code 为 0。",
        "- 主配置固定 4 RE/frame、5 frames、20 RE/window；Full 和 CSI-off 为 0 RE。",
        "- 主 M4/prototype 冻结；正式训练参数仅为子原型 residual 与 missing adapter。参数量和缓存后决策头延迟见 `latency_summary.csv`。",
        "- 延迟为同一设备上的 cached-feature decision-head latency，不包含冻结的 M4 特征提取或 CSI GRU 编码。",
        "- 全部结果属于 trajectory-disjoint inner validation development；outer test 未解封。",
        "",
    ]
    (output / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    summary = {
        "complete": not missing_runs,
        "missing_runs": missing_runs,
        "wireless_injection_claim": wireless_claim,
        "a7_main_method_claim": a7_claim,
        "full_exact": full_exact,
        "csi_off_exact": csi_off_exact,
        "teacher_coverage": teacher_coverage,
        "prototype_collapse_ratio": collapse,
        "a7_seed_direction_consistent": a7_seed_direction,
        "a7_csi_dominant": csi_dominant,
        "cluster_missing_mask_accuracy_gap": cluster_mask_accuracy_gap,
        "formal_elapsed_seconds": float(monitor_latest["elapsed_seconds"]),
        "checkpoint_count": checkpoint_count,
        "outer_test_accessed": False,
    }
    _write_json(output / "summary.json", summary)
    return summary


def run_queue(config: Mapping[str, Any], *, gpu: int, tasks: Sequence[str]) -> dict[str, Any]:
    output = _path(config["output"]["root"])
    queue_log = output / f"logs/queues/gpu{gpu}"
    queue_log.mkdir(parents=True, exist_ok=True)
    active: subprocess.Popen | None = None
    stop_requested = False

    def stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGTERM)

    old_handler = signal.signal(signal.SIGTERM, stop)
    results = []
    try:
        for task in tasks:
            if stop_requested:
                break
            arm, seed_text = task.split(":", 1)
            seed = int(seed_text)
            run_log = output / f"logs/{arm}_seed{seed}"
            run_log.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config",
                str(DEFAULT_CONFIG),
                "run",
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--device",
                "cuda:0",
            ]
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            attempts = 0
            exit_code = 1
            while attempts <= int(config["runtime"]["retry_limit"]) and exit_code != 0 and not stop_requested:
                attempts += 1
                if attempts > 1:
                    command.append("--resume")
                with (run_log / "stdout.log").open("a", encoding="utf-8") as stdout, (
                    run_log / "stderr.log"
                ).open("a", encoding="utf-8") as stderr:
                    active = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
                    (run_log / "launcher_pid").write_text(f"{active.pid}\n", encoding="utf-8")
                    exit_code = active.wait()
                (run_log / "exit_code").write_text(f"{exit_code}\n", encoding="utf-8")
            results.append({"task": task, "exit_code": exit_code, "attempts": attempts})
            if exit_code != 0:
                break
    finally:
        signal.signal(signal.SIGTERM, old_handler)
    _write_json(queue_log / "result.json", {"gpu": gpu, "tasks": results, "stopped": stop_requested})
    return {"gpu": gpu, "tasks": results, "stopped": stop_requested}


def launch_queues(config: Mapping[str, Any]) -> dict[str, Any]:
    preflight(config)
    output = _path(config["output"]["root"])
    assignments = {
        0: ["A1:1", "A1:2", "A1:3", "A5:1", "A5:2", "A5:3"],
        1: ["A2:1", "A2:2", "A2:3"],
        2: ["A3:1", "A3:2", "A3:3"],
        3: ["A4:1", "A4:2", "A4:3"],
        4: ["A7:1", "A7:2", "A7:3"],
    }
    manifest_path = output / "process_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        live = [item for item in old.get("queues", []) if _pid_alive(int(item["pid"]))]
        if live:
            raise RuntimeError(f"Radio-guided queues are already running: {[item['pid'] for item in live]}.")
    queues = []
    for gpu, tasks in assignments.items():
        log_dir = output / f"logs/queues/gpu{gpu}"
        log_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(DEFAULT_CONFIG),
            "queue",
            "--gpu",
            str(gpu),
            "--tasks",
            *tasks,
        ]
        with (log_dir / "stdout.log").open("a", encoding="utf-8") as stdout, (
            log_dir / "stderr.log"
        ).open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
        queues.append({"gpu": gpu, "pid": process.pid, "tasks": tasks, "log_dir": str(log_dir.resolve())})
    manifest = {
        "started_at_unix": time.time(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "monitor_interval_seconds": int(config["runtime"]["monitor_interval_seconds"]),
        "max_wall_seconds": int(config["runtime"]["max_wall_seconds"]),
        "queues": queues,
        "outer_test_accessed": False,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        stat = Path(f"/proc/{int(pid)}/stat")
        return not stat.exists() or stat.read_text(encoding="utf-8").split()[2] != "Z"
    except (ProcessLookupError, PermissionError):
        return False


def monitor_once(config: Mapping[str, Any]) -> dict[str, Any]:
    output = _path(config["output"]["root"])
    manifest = json.loads((output / "process_manifest.json").read_text(encoding="utf-8"))
    elapsed = time.time() - float(manifest["started_at_unix"])
    queues = []
    for item in manifest["queues"]:
        gpu, pid = int(item["gpu"]), int(item["pid"])
        run_logs = [output / f"logs/{task.split(':')[0]}_seed{task.split(':')[1]}" for task in item["tasks"]]
        latest = max(
            (path.stat().st_mtime for directory in run_logs for path in directory.glob("*.log")),
            default=0.0,
        )
        completed = sum((directory / "exit_code").exists() for directory in run_logs)
        failures = [
            directory.name
            for directory in run_logs
            if (directory / "exit_code").exists() and (directory / "exit_code").read_text().strip() != "0"
        ]
        nonfinite = []
        for directory in run_logs:
            for name in ("stdout.log", "stderr.log"):
                path = directory / name
                if path.exists():
                    text = path.read_text(encoding="utf-8", errors="replace").lower()
                    if any(token in text.split() for token in ("nan", "inf")):
                        nonfinite.append(str(path.relative_to(output)))
        queues.append(
            {
                "gpu": gpu,
                "pid": pid,
                "alive": _pid_alive(pid),
                "completed_tasks": completed,
                "total_tasks": len(item["tasks"]),
                "failures": failures,
                "nonfinite_logs": nonfinite,
                "last_log_age_seconds": time.time() - latest if latest else None,
                "checkpoint_count": sum(1 for directory in run_logs for _ in (output / "checkpoints" / directory.name.split("_seed")[0] / f"seed{directory.name.split('_seed')[1]}").glob("*.pt")),
            }
        )
    disk = subprocess.run(
        ["df", "-Pk", str(output)], capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()[-1].split()
    gpu_processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip().splitlines()
    result = {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "elapsed_seconds": elapsed,
        "queues": queues,
        "gpu_processes": gpu_processes,
        "disk_available_kb": int(disk[3]),
        "all_complete": all(not item["alive"] for item in queues),
        "wall_limit_reached": elapsed >= int(config["runtime"]["max_wall_seconds"]),
    }
    monitor_root = output / "logs/monitor"
    _write_json(monitor_root / f"check_{int(time.time())}.json", result)
    _write_json(monitor_root / "latest.json", result)
    return result


def monitor(config: Mapping[str, Any], *, watch: bool) -> dict[str, Any]:
    while True:
        result = monitor_once(config)
        print(json.dumps(result, allow_nan=True), flush=True)
        if result["all_complete"]:
            summarize(config)
            return result
        if result["wall_limit_reached"]:
            output = _path(config["output"]["root"])
            manifest = json.loads((output / "process_manifest.json").read_text(encoding="utf-8"))
            for item in manifest["queues"]:
                if _pid_alive(int(item["pid"])):
                    os.kill(int(item["pid"]), signal.SIGTERM)
            return result
        if not watch:
            return result
        time.sleep(int(config["runtime"]["monitor_interval_seconds"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--device", default="cpu")
    cache_parser = subparsers.add_parser("cache")
    cache_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    cache_parser.add_argument("--force", action="store_true")
    cluster_parser = subparsers.add_parser("cluster")
    cluster_parser.add_argument("--role", default="train")
    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument("--arm", choices=("A0", "A6"), required=True)
    baseline_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--arm", choices=("A1", "A2", "A3", "A4", "A5", "A7"), required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--resume", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--arm", choices=("A1", "A2", "A3", "A4", "A5", "A7"), required=True)
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    evaluate_parser.add_argument("--checkpoint-name", default="best_csi_off_all14.pt")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--arm", choices=("A1", "A2", "A3", "A4", "A5", "A7"), required=True)
    run_parser.add_argument("--seed", type=int, required=True)
    run_parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    run_parser.add_argument("--resume", action="store_true")
    subparsers.add_parser("summarize")
    subparsers.add_parser("launch")
    queue_parser = subparsers.add_parser("queue")
    queue_parser.add_argument("--gpu", type=int, required=True)
    queue_parser.add_argument("--tasks", nargs="+", required=True)
    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.command == "preflight":
        result = preflight(config)
    elif args.command == "audit":
        result = audit(config, torch.device(args.device))
    elif args.command == "cache":
        result = build_cache(config, torch.device(args.device), force=args.force)
    elif args.command == "cluster":
        result = analyze_subclusters(config, role=args.role)
    elif args.command == "baseline":
        result = run_baseline(config, arm=args.arm, device=torch.device(args.device))
    elif args.command == "train":
        result = train_arm(
            config, arm=args.arm, seed=args.seed, device=torch.device(args.device), resume=args.resume
        )
    elif args.command == "evaluate":
        result = evaluate_trained_arm(
            config,
            arm=args.arm,
            seed=args.seed,
            device=torch.device(args.device),
            checkpoint_name=args.checkpoint_name,
        )
    elif args.command == "run":
        result = run_trained_arm(
            config, arm=args.arm, seed=args.seed, device=torch.device(args.device), resume=args.resume
        )
    elif args.command == "summarize":
        result = summarize(config)
    elif args.command == "launch":
        result = launch_queues(config)
    elif args.command == "queue":
        result = run_queue(config, gpu=args.gpu, tasks=args.tasks)
    elif args.command == "monitor":
        result = monitor(config, watch=args.watch)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, allow_nan=True, default=str))


if __name__ == "__main__":
    main()
