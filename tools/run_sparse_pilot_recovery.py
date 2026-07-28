#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from kd_sensing.baselines.prototype_decision_adapter import (
    MASKS,
    checkpoint_normalization_overrides,
    load_frozen_u0,
    load_u0_artifact_config,
    preflight,
)
from kd_sensing.baselines.sparse_pilot_transition import (
    SparsePilotInformationClassifier,
    SparsePilotTransitionModel,
)
from kd_sensing.channel.pilot_cache import PilotCache, PilotCacheSpec
from kd_sensing.channel.probe_codebook import generate_probe_codebook
from kd_sensing.channel.sparse_pilot_simulator import (
    frequency_offsets_hz,
    load_path_channel,
    pilot_subcarrier_indices,
    simulate_candidate_pilots,
)
from kd_sensing.config.io import dump_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.engine.batch import prepare_fusion_inputs
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers


TASKS = {
    "I1": {"input_time": "t", "target_time": "t", "history_length": 1, "concat": False},
    "I2": {"input_time": "t", "target_time": "t+1", "history_length": 1, "concat": False},
    "I3": {"input_time": "t-4:t", "target_time": "t+1", "history_length": 5, "concat": False},
    "I4": {"input_time": "U0+CSI_t", "target_time": "t+1", "history_length": 1, "concat": True},
    "I5": {"input_time": "U0+CSI_t-4:t", "target_time": "t+1", "history_length": 5, "concat": True},
}
SEVERE_MASKS = ("image_only", "radar_only", "gps_only", "lidar_only")
EXTRACT_MASKS = (*SEVERE_MASKS, "full")


def balanced_subset_indices(dataset: object, count: int) -> list[int]:
    requested = int(count)
    if requested <= 0 or requested > len(dataset):  # type: ignore[arg-type]
        raise ValueError("Balanced subset count must be positive and no larger than the dataset.")
    if not isinstance(dataset, ConcatDataset):
        return [((2 * index + 1) * len(dataset)) // (2 * requested) for index in range(requested)]
    components = list(dataset.datasets)
    if requested < len(components):
        raise ValueError("Balanced pooled subset requires at least one sample per domain.")
    quotas = [requested // len(components)] * len(components)
    for index in range(requested % len(components)):
        quotas[index] += 1
    if any(quota > len(component) for quota, component in zip(quotas, components, strict=True)):
        raise ValueError("Balanced pooled subset quota exceeds a domain size.")
    indices: list[int] = []
    offset = 0
    for quota, component in zip(quotas, components, strict=True):
        indices.extend(offset + ((2 * index + 1) * len(component)) // (2 * quota) for index in range(quota))
        offset += len(component)
    return indices


def subset_audit(dataset: object, indices: list[int]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "count": len(indices),
        "index_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
        "domain_counts": {},
    }
    if isinstance(dataset, ConcatDataset):
        start = 0
        for position, component in enumerate(dataset.datasets):
            stop = start + len(component)
            name = str(getattr(component, "domain_id", position))
            result["domain_counts"][name] = sum(start <= index < stop for index in indices)
            start = stop
    return result


def _mask_lookup() -> dict[str, tuple[bool, bool, bool, bool]]:
    return {key: tuple(bool(value) for value in pattern) for key, _, pattern in MASKS}


def _fixed_loader(loader: DataLoader, indices: list[int], batch_size: int) -> DataLoader:
    return DataLoader(
        Subset(loader.dataset, indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=loader.collate_fn,
    )


def _simulate_history(
    batch_history_refs: list[tuple[str, ...]],
    *,
    codebook,
    frequencies: np.ndarray,
    cache: PilotCache,
    cache_spec: PilotCacheSpec,
) -> torch.Tensor:
    batch_size = len(batch_history_refs[0])
    rows: list[np.ndarray] = []
    for sample_index in range(batch_size):
        frames: list[np.ndarray] = []
        for time_refs in batch_history_refs:
            path = Path(time_refs[sample_index])

            def compute(path: Path = path):
                matrices, delays = load_path_channel(path)
                return simulate_candidate_pilots(
                    matrices[None, None, :, None, :, :, None],
                    delays[None, None, None, :],
                    codebook,
                    frequencies,
                )

            frames.append(cache.get_or_compute(path, cache_spec, compute))
        rows.append(np.stack(frames))
    return torch.from_numpy(np.stack(rows))


@torch.no_grad()
def _extract_role(
    loader: DataLoader,
    transition: SparsePilotTransitionModel,
    *,
    cfg: Mapping[str, Any],
    device: torch.device,
    codebook,
    frequencies: np.ndarray,
    cache: PilotCache,
    cache_spec: PilotCacheSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensor_chunks: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "labels_current",
            "labels_future",
            "current_beam_power",
            "future_beam_power",
            "candidate_history",
            *(f"z_{mask}" for mask in EXTRACT_MASKS),
            *(f"p0_{mask}" for mask in EXTRACT_MASKS),
        )
    }
    sample_ids: list[str] = []
    current_label_mismatches = 0
    prepared_alias_mismatches = 0
    masks = _mask_lookup()
    for batch in loader:
        inputs = prepare_fusion_inputs(dict(batch), seq_length=int(cfg["model"]["seq_length"]), device=device)
        batch_size = len(batch["channel_ref"])
        for mask_name in EXTRACT_MASKS:
            pattern = torch.tensor(masks[mask_name], device=device, dtype=torch.bool).expand(batch_size, -1)
            sensing = transition.sensing_forward(inputs, missing_mask=pattern)
            tensor_chunks[f"z_{mask_name}"].append(sensing["z_sensing"].detach().cpu())
            tensor_chunks[f"p0_{mask_name}"].append(sensing["p0"].detach().cpu())
        current_labels = torch.as_tensor(batch["current_beam"]).reshape(-1).long()
        current_power = torch.as_tensor(batch["current_beam_power"]).float()
        current_label_mismatches += int(current_labels.ne(current_power.argmax(dim=-1)).sum().item())
        prepared_alias_mismatches += int(
            torch.as_tensor(batch["prepared_beam_label"]).reshape(-1).long().ne(current_labels).sum().item()
        )
        tensor_chunks["labels_current"].append(current_labels)
        tensor_chunks["labels_future"].append(torch.as_tensor(batch["target_beam"]).reshape(-1).long())
        tensor_chunks["current_beam_power"].append(current_power)
        tensor_chunks["future_beam_power"].append(torch.as_tensor(batch["future_beam_power"]).float())
        tensor_chunks["candidate_history"].append(
            _simulate_history(
                batch["channel_history_refs"],
                codebook=codebook,
                frequencies=frequencies,
                cache=cache,
                cache_spec=cache_spec,
            )
        )
        sample_ids.extend(str(value) for value in batch["sample_id"])
    if current_label_mismatches:
        raise ValueError(f"beam_label disagrees with argmax(beam5) for {current_label_mismatches} samples.")
    records: dict[str, Any] = {key: torch.cat(value, dim=0) for key, value in tensor_chunks.items()}
    records["sample_ids"] = sample_ids
    audit = {
        "sample_count": len(sample_ids),
        "sample_id_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "beam_t_label_power_mismatch_count": current_label_mismatches,
        "prepared_beam_label_alias_mismatch_count": prepared_alias_mismatches,
        "current_class_counts": _class_counts(records["labels_current"]),
        "future_class_counts": _class_counts(records["labels_future"]),
        "u0": _u0_statistics(records),
    }
    return records, audit


def _class_counts(labels: torch.Tensor) -> dict[str, int]:
    counts = torch.bincount(labels, minlength=64)
    return {str(index): int(value) for index, value in enumerate(counts.tolist()) if value}


def _u0_statistics(records: Mapping[str, Any]) -> dict[str, Any]:
    labels = records["labels_future"]
    result: dict[str, Any] = {}
    for mask in EXTRACT_MASKS:
        prediction = records[f"p0_{mask}"].argmax(dim=-1)
        distance = (prediction - labels).abs()
        distance = torch.minimum(distance, 64 - distance)
        result[mask] = {
            "correct": int(prediction.eq(labels).sum().item()),
            "wrong": int(prediction.ne(labels).sum().item()),
            "top1": float(prediction.eq(labels).float().mean().item()),
            "distance_counts": {str(key): int(value) for key, value in Counter(distance.tolist()).items()},
        }
    return result


def _initial_transition_audit(
    transition: SparsePilotTransitionModel,
    records: Mapping[str, Any],
    frequencies: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    count = min(256, len(records["labels_future"]))
    selected = records["candidate_history"][:count, -1].to(device)
    snr = torch.full((count,), 10.0, device=device)
    generator = torch.Generator(device=device).manual_seed(2026)
    observed, valid = _noisy_observations(selected, snr, generator=generator)
    pattern_ids = torch.arange(4, device=device).expand(count, -1)
    transition.eval()
    with torch.no_grad():
        output = transition.forward_selected(
            {
                "z_sensing": records["z_full"][:count].to(device),
                "p0": records["p0_full"][:count].to(device),
                "sensing_availability": torch.ones(count, device=device),
            },
            observed,
            pattern_ids=pattern_ids,
            frequency_positions=frequencies,
            pilot_mask=valid,
            snr_db=snr,
        )
    alpha = output["alpha"].cpu()
    return {
        "explicit_alpha_parameter_initialization": False,
        "implementation": "sigmoid(default-initialized reliability MLP) * quality_confidence",
        "sample_count": count,
        "mean": float(alpha.mean().item()),
        "std": float(alpha.std(unbiased=False).item()),
        "p10": float(torch.quantile(alpha, 0.1).item()),
        "p50": float(torch.quantile(alpha, 0.5).item()),
        "p90": float(torch.quantile(alpha, 0.9).item()),
    }


def prepare(args: argparse.Namespace, config: dict[str, Any]) -> None:
    root = args.output_root
    record_dir = root / "records"
    if record_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Recovery records already exist: {record_dir}")
    root.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    _, preflight_audit = preflight(args.u0_config, args.checkpoint, args.expected_sha256)
    cfg = load_u0_artifact_config(args.u0_config)
    cfg["data"]["dataset"].update(
        include_channel_ref=True,
        include_channel_history_refs=True,
        pilot_time_mode="last_input",
        include_router_utility_targets=True,
    )
    cfg["data"]["dataloader"].update(num_workers=4, persistent_workers=True, pin_memory=True)
    loaders = build_dataloaders(cfg, normalization_overrides=checkpoint_normalization_overrides(args.checkpoint))
    training = config["training"]
    indices = {
        "train": balanced_subset_indices(loaders["train"].dataset, int(training["train_samples"])),
        "validation": balanced_subset_indices(
            loaders["validation"].dataset, int(training["validation_samples"])
        ),
    }
    fixed = {
        role: _fixed_loader(loaders[role], indices[role], int(training["batch_size"]))
        for role in ("train", "validation")
    }
    device = torch.device(args.device)
    sensing_model, model_audit = load_frozen_u0(cfg, args.checkpoint, device)
    transition = SparsePilotTransitionModel(sensing_model, topology_positions=torch.arange(64)).to(device)
    pilot_cfg = config["pilot"]
    codebook_path = root / "pilot_codebook.npz"
    codebook = generate_probe_codebook(
        64,
        16,
        num_patterns=int(pilot_cfg["num_patterns"]),
        seed=int(pilot_cfg["codebook_seed"]),
        method=str(pilot_cfg["codebook_method"]),
    )
    codebook.save(codebook_path)
    subcarrier_indices = pilot_subcarrier_indices(
        int(pilot_cfg["num_subcarriers"]), int(pilot_cfg["num_pilot_subcarriers"])
    )
    frequencies_np = frequency_offsets_hz(
        subcarrier_indices,
        num_subcarriers=int(pilot_cfg["num_subcarriers"]),
        subcarrier_spacing_hz=float(pilot_cfg["subcarrier_spacing_hz"]),
        mode=str(pilot_cfg["frequency_index_mode"]),
    )
    cache = PilotCache(config["output"]["cache_root"])
    cache_spec = PilotCacheSpec(
        codebook.hash,
        tuple(frequencies_np),
        float(pilot_cfg["subcarrier_spacing_hz"]),
        str(pilot_cfg["frequency_index_mode"]),
        64,
        16,
    )
    role_audits: dict[str, Any] = {}
    role_records: dict[str, dict[str, Any]] = {}
    for role in ("train", "validation"):
        records, role_audit = _extract_role(
            fixed[role],
            transition,
            cfg=cfg,
            device=device,
            codebook=codebook,
            frequencies=frequencies_np,
            cache=cache,
            cache_spec=cache_spec,
        )
        torch.save(records, record_dir / f"{role}.pt")
        role_records[role] = records
        role_audits[role] = role_audit | {"subset": subset_audit(loaders[role].dataset, indices[role])}
    initial_alpha = _initial_transition_audit(
        transition,
        role_records["validation"],
        torch.from_numpy(frequencies_np).to(device=device, dtype=torch.float32),
        device,
    )
    resolved = dict(config)
    resolved["runtime"] = {
        "u0_checkpoint": str(args.checkpoint.resolve()),
        "u0_sha256": args.expected_sha256,
        "codebook_hash": codebook.hash,
        "subcarrier_indices": subcarrier_indices.tolist(),
        "frequency_positions_hz": frequencies_np.tolist(),
        "outer_test_accessed": False,
        "future_channel_used_as_input": False,
    }
    dump_config(resolved, root / "resolved_configs" / "prepare.yaml")
    sampling = {
        "preflight": preflight_audit,
        "u0_model": model_audit,
        "roles": role_audits,
        "initial_alpha": initial_alpha,
        "outer_test_accessed": False,
    }
    (root / "sampling_statistics.json").write_text(
        json.dumps(sampling, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_audit(root / "audit.md", sampling, config)
    for loader in (*fixed.values(), *loaders.values()):
        shutdown_dataloader_workers(loader)
    print(json.dumps({"status": "prepared", "roles": role_audits}, indent=2, sort_keys=True))


def _write_audit(path: Path, sampling: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    train = sampling["roles"]["train"]
    validation = sampling["roles"]["validation"]
    lines = [
        "# Sparse-Pilot Recovery 实现审计",
        "",
        "本审计只使用 Full-pool development train/inner validation；outer test 未访问。",
        "",
        "1. pilot 输入来自 `csi5`，即最后感知输入帧 t；recovery 历史输入逐帧读取 `csi1..csi5`，即 t-4..t。",
        "2. 正式未来 target 来自 `future_beam_label1`，即 t+1；D1 当前 target 直接取既有 `beam5` 功率向量 argmax。",
        "3. 可以取得五帧历史 channel；解析器逐帧校验 path 文件名与 `history_frame_ids_json` 一致并要求严格连续。",
        "4. 旧 cache 每个 sample 只保存单个时刻 t 的 candidate pilot；recovery cache 按 channel identity 独立缓存，因此覆盖五个历史时刻但不保存 future channel。",
        "5. `sensing_forward` 对 U0 logits 调用 softmax，所以 p0 是概率，不是 logits。",
        "6. q_local 使用 `argmax(p0)` 为中心构造循环半径 3 邻域；不是 proto_id，也不是 p0 Top-K 联合邻域。",
        f"7. alpha 没有显式常数初始化；默认 MLP 初始化下 256 条 validation 的统计为 `{json.dumps(sampling['initial_alpha'], sort_keys=True)}`。",
        "8. 当前 preserve loss 对所有 mask 的 U0-correct 样本生效，没有 `available_modalities>=3` 限制；权重另乘 `(1-alpha.detach())`。",
        "9. 既有 40-epoch history 已记录 selector/CSI encoder/transition 非零梯度；recovery runner 继续逐 epoch记录 encoder/head 梯度，gate/route 不适用。",
        "10. U0 输出在 no_grad 后 detach 是预期冻结边界；SparsePilotEncoder 与 transition 之间没有 detach，反向测试覆盖两者梯度。",
        f"11. train/validation 通过 15-domain 均衡、域内等距无重复抽取 2,000/1,000；索引 SHA 分别为 `{train['subset']['index_sha256']}` / `{validation['subset']['index_sha256']}`。",
        "12. 类别与 U0 错误统计见 `sampling_statistics.json`；validation future label 仅覆盖 "
        f"{len(validation['future_class_counts'])}/64 类，Full U0 wrong={validation['u0']['full']['wrong']}，"
        f"四个 single mask wrong={[validation['u0'][mask]['wrong'] for mask in SEVERE_MASKS]}，错误样本充足但类别明显不均衡。",
        "",
        "## 固定诊断协议",
        "",
        f"- pilot: M={config['pilot']['num_patterns']}, Kp={config['pilot']['num_pilot_subcarriers']}, QPSK fixed across split/seed",
        f"- train SNR: [{config['training']['snr_db_min']}, {config['training']['snr_db_max']}] dB",
        f"- validation: {config['training']['validation_snr_db']} dB, dropout=0",
        f"- Full-pool `beam_label` 是未来标签别名：train 中有 {train['prepared_beam_label_alias_mismatch_count']}/{train['sample_count']} 条不等于 beam_t；该字段未作为 D1 target。",
        "- 未发现时间泄漏、beam5 当前标签错位或 CSI encoder 梯度中断；因此允许进入 I0--I5 信息诊断。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_records(records: Mapping[str, Any], task: str) -> dict[str, torch.Tensor]:
    spec = TASKS[task]
    candidates = records["candidate_history"][:, -int(spec["history_length"]) :]
    labels = records["labels_current"] if spec["target_time"] == "t" else records["labels_future"]
    power = records["current_beam_power"] if spec["target_time"] == "t" else records["future_beam_power"]
    if not spec["concat"]:
        output = {"candidate_history": candidates, "labels": labels, "beam_power": power}
        if spec["target_time"] == "t+1":
            output["p0"] = records["p0_full"]
        return output
    return {
        "candidate_history": candidates.repeat(len(SEVERE_MASKS), 1, 1, 1),
        "labels": labels.repeat(len(SEVERE_MASKS)),
        "beam_power": power.repeat(len(SEVERE_MASKS), 1),
        "sensing_feature": torch.cat([records[f"z_{mask}"] for mask in SEVERE_MASKS]),
        "p0": torch.cat([records[f"p0_{mask}"] for mask in SEVERE_MASKS]),
        "physical_sample_count": torch.tensor(len(labels)),
    }


def _full_concat_records(records: Mapping[str, Any], task: str) -> dict[str, torch.Tensor]:
    spec = TASKS[task]
    if not spec["concat"]:
        raise ValueError("Full concat records are only defined for I4/I5.")
    return {
        "candidate_history": records["candidate_history"][:, -int(spec["history_length"]) :],
        "labels": records["labels_future"],
        "beam_power": records["future_beam_power"],
        "sensing_feature": records["z_full"],
        "p0": records["p0_full"],
    }


def _noisy_observations(
    selected: torch.Tensor, snr_db: torch.Tensor, *, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    power = selected.abs().square().mean(dim=(-2, -1), keepdim=True)
    variance = power / torch.pow(10.0, snr_db[..., None, None] / 10.0)
    scale = (variance / 2.0).sqrt()
    noise = torch.complex(
        torch.randn(selected.shape, device=selected.device, generator=generator),
        torch.randn(selected.shape, device=selected.device, generator=generator),
    ) * scale
    return selected + noise, torch.ones_like(selected, dtype=torch.bool)


def _batch_forward(
    model: SparsePilotInformationClassifier,
    batch: Mapping[str, torch.Tensor],
    *,
    frequencies: torch.Tensor,
    snr: torch.Tensor,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    candidates = batch["candidate_history"]
    expanded_snr = snr[:, None].expand(-1, candidates.shape[1])
    observed, valid = _noisy_observations(candidates, expanded_snr, generator=generator)
    pattern_ids = torch.arange(4, device=candidates.device).expand(
        candidates.shape[0], candidates.shape[1], -1
    )
    return model(
        observed,
        pattern_ids,
        frequencies,
        valid,
        expanded_snr,
        sensing_feature=batch.get("sensing_feature"),
    )


def _gradient_norm(module: nn.Module | None) -> float | None:
    if module is None:
        return None
    square_sum = sum(
        float(parameter.grad.detach().float().square().sum().item())
        for parameter in module.parameters()
        if parameter.grad is not None
    )
    return square_sum**0.5


def _prediction_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    beam_power: torch.Tensor,
    base: torch.Tensor | None = None,
) -> dict[str, float | None]:
    prediction = probabilities.argmax(dim=-1)
    distance = (prediction - labels).abs()
    distance = torch.minimum(distance, 64 - distance)
    top = probabilities.topk(5, dim=-1).indices
    row = torch.arange(len(labels))
    ratio = beam_power[row, prediction] / beam_power.amax(dim=-1).clamp_min(1e-12)
    result: dict[str, float | None] = {
        "top1": float(prediction.eq(labels).float().mean().item()),
        "top3": float(top[:, :3].eq(labels[:, None]).any(dim=1).float().mean().item()),
        "top5": float(top.eq(labels[:, None]).any(dim=1).float().mean().item()),
        "within3": float(distance.le(3).float().mean().item()),
        "mae": float(distance.float().mean().item()),
        "normalized_gain": float(ratio.mean().item()),
        "beam_loss_db": float((-10.0 * ratio.clamp_min(1e-12).log10()).mean().item()),
        "fix_rate": None,
        "harm_rate": None,
        "p_final_p0_kl": None,
    }
    if base is not None:
        base_correct = base.argmax(dim=-1).eq(labels)
        correct = prediction.eq(labels)
        result["fix_rate"] = (
            float(correct[~base_correct].float().mean().item()) if bool((~base_correct).any()) else 0.0
        )
        result["harm_rate"] = (
            float((~correct[base_correct]).float().mean().item()) if bool(base_correct.any()) else 0.0
        )
        result["p_final_p0_kl"] = float(
            (probabilities * (probabilities.clamp_min(1e-12).log() - base.clamp_min(1e-12).log()))
            .sum(dim=-1)
            .mean()
            .item()
        )
    return result


@torch.no_grad()
def evaluate(
    model: SparsePilotInformationClassifier,
    records: Mapping[str, torch.Tensor],
    *,
    frequencies: torch.Tensor,
    snr_db: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    model.eval()
    probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    generator = torch.Generator(device=device).manual_seed(100_000 + int(seed))
    for start in range(0, len(records["labels"]), int(batch_size)):
        stop = start + int(batch_size)
        batch = {
            key: value[start:stop].to(device)
            for key, value in records.items()
            if torch.is_tensor(value) and value.ndim > 0
        }
        snr = torch.full((len(batch["labels"]),), float(snr_db), device=device)
        output = _batch_forward(model, batch, frequencies=frequencies, snr=snr, generator=generator)
        probabilities.append(output["logits"].softmax(dim=-1).cpu())
        entropies.append(output["q_entropy"].cpu())
    probs = torch.cat(probabilities)
    entropy = torch.cat(entropies)
    labels = records["labels"]
    power = records["beam_power"]
    base = records.get("p0")
    loss = float(F.nll_loss(probs.clamp_min(1e-12).log(), labels).item())
    physical_count = int(records.get("physical_sample_count", torch.tensor(0)).item())
    if not physical_count:
        metrics = _prediction_metrics(probs, labels, power, base)
        return metrics | {
            "validation_loss": loss,
            "single_macro": metrics["top1"],
            "single_worst": metrics["top1"],
            "all14_macro": None,
            "all14_worst": None,
            "full_top1": None,
            "q_transition_entropy": float(entropy.mean().item()),
        }
    per_mask = []
    for mask_index, mask in enumerate(SEVERE_MASKS):
        subset = slice(mask_index * physical_count, (mask_index + 1) * physical_count)
        per_mask.append(
            {"mask": mask}
            | _prediction_metrics(
                probs[subset], labels[subset], power[subset], base[subset] if base is not None else None
            )
        )
    aggregate = {
        key: float(np.mean([float(row[key]) for row in per_mask if row[key] is not None]))
        for key in (
            "top1",
            "top3",
            "top5",
            "within3",
            "mae",
            "normalized_gain",
            "beam_loss_db",
            "fix_rate",
            "harm_rate",
            "p_final_p0_kl",
        )
    }
    return aggregate | {
        "validation_loss": loss,
        "single_macro": aggregate["top1"],
        "single_worst": min(float(row["top1"]) for row in per_mask),
        "all14_macro": None,
        "all14_worst": None,
        "full_top1": None,
        "q_transition_entropy": float(entropy.mean().item()),
        "per_mask": per_mask,
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    resolved_config: Mapping[str, Any],
    selection_metric: str,
    selection_value: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": int(epoch),
            "resolved_config": dict(resolved_config),
            "rng_state": _rng_state(),
            "selection_metric": selection_metric,
            "selection_value": selection_value,
            "outer_test_accessed": False,
        },
        temporary,
    )
    temporary.replace(path)


def train(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.task not in TASKS or args.seed is None:
        raise ValueError("train mode requires --task I1..I5 and --seed.")
    task, seed = args.task, int(args.seed)
    result_path = args.output_root / "results" / f"seed{seed}_{task}.json"
    if result_path.exists() and not args.overwrite:
        raise FileExistsError(f"Recovery result already exists: {result_path}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_raw = torch.load(args.output_root / "records" / "train.pt", map_location="cpu", weights_only=False)
    validation_raw = torch.load(
        args.output_root / "records" / "validation.pt", map_location="cpu", weights_only=False
    )
    train_records = _task_records(train_raw, task)
    validation_records = _task_records(validation_raw, task)
    spec = TASKS[task]
    full_validation_records = _full_concat_records(validation_raw, task) if spec["concat"] else None
    device = torch.device(args.device)
    model = SparsePilotInformationClassifier(
        history_length=int(spec["history_length"]),
        sensing_dim=64 if spec["concat"] else 0,
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["max_epochs"]))
    prepared = safe_load_yaml(
        (args.output_root / "resolved_configs" / "prepare.yaml").read_text(encoding="utf-8")
    )
    resolved = prepared | {"run": {"task": task, "seed": seed, "device": str(device), **spec}}
    resolved_path = args.output_root / "resolved_configs" / f"seed{seed}_{task}.yaml"
    dump_config(resolved, resolved_path)
    frequencies = torch.tensor(prepared["runtime"]["frequency_positions_hz"], device=device)
    history_rows: list[dict[str, Any]] = []
    checkpoint_dir = args.output_root / "checkpoints" / f"seed{seed}" / task
    best = {"val_loss": float("inf"), "single_macro": float("-inf"), "fix_rate": float("-inf")}
    patience_count = 0
    stop_reason = "max_epochs"
    train_generator = torch.Generator(device=device).manual_seed(10_000 + seed)
    order_generator = torch.Generator().manual_seed(20_000 + seed)
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_records["labels"]), generator=order_generator)
        loss_sum = grad_encoder = grad_head = entropy_sum = 0.0
        seen = batches = 0
        for start in range(0, len(order), int(training["batch_size"])):
            selected_indices = order[start : start + int(training["batch_size"])]
            batch = {
                key: value[selected_indices].to(device)
                for key, value in train_records.items()
                if torch.is_tensor(value) and value.ndim > 0
            }
            snr = torch.empty(len(batch["labels"]), device=device).uniform_(
                float(training["snr_db_min"]),
                float(training["snr_db_max"]),
                generator=train_generator,
            )
            output = _batch_forward(model, batch, frequencies=frequencies, snr=snr, generator=train_generator)
            loss = F.cross_entropy(output["logits"], batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            grad_encoder += float(_gradient_norm(model.csi_encoder) or 0.0)
            grad_head += float(_gradient_norm(model.classifier) or 0.0)
            optimizer.step()
            count = len(batch["labels"])
            seen += count
            batches += 1
            loss_sum += float(loss.detach().item()) * count
            entropy_sum += float(output["q_entropy"].detach().mean().item()) * count
        metrics = evaluate(
            model,
            validation_records,
            frequencies=frequencies,
            snr_db=float(training["validation_snr_db"]),
            batch_size=int(training["batch_size"]),
            device=device,
            seed=seed,
        )
        if full_validation_records is not None:
            full_metrics = evaluate(
                model,
                full_validation_records,
                frequencies=frequencies,
                snr_db=float(training["validation_snr_db"]),
                batch_size=int(training["batch_size"]),
                device=device,
                seed=seed,
            )
            metrics["full_top1"] = full_metrics["top1"]
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_total_loss": loss_sum / seen,
            "train_final_ce": loss_sum / seen,
            "train_topology_loss": 0.0,
            "train_route_loss": None,
            "train_preserve_loss": None,
            "train_selector_loss": None,
            "validation_loss": metrics["validation_loss"],
            "validation_top1": metrics["top1"],
            "validation_top3": metrics["top3"],
            "validation_top5": metrics["top5"],
            "validation_within3": metrics["within3"],
            "validation_mae": metrics["mae"],
            "single_macro": metrics["single_macro"],
            "single_worst": metrics["single_worst"],
            "all14_macro": metrics["all14_macro"],
            "all14_worst": metrics["all14_worst"],
            "full_top1": metrics["full_top1"],
            "fix_rate": metrics["fix_rate"],
            "harm_rate": metrics["harm_rate"],
            "alpha_mean": None,
            "alpha_std": None,
            "alpha_p10": None,
            "alpha_p25": None,
            "alpha_p50": None,
            "alpha_p75": None,
            "alpha_p90": None,
            "r_global_mean": None,
            "r_global_std": None,
            "csi_encoder_gradient_norm": grad_encoder / batches,
            "transition_gradient_norm": grad_head / batches,
            "gate_gradient_norm": None,
            "p_final_p0_kl": metrics["p_final_p0_kl"],
            "q_transition_entropy": metrics["q_transition_entropy"],
            "learning_rate": current_lr,
        }
        history_rows.append(row)
        _write_csv(args.output_root / "training_logs" / f"seed{seed}_{task}.csv", history_rows)
        scheduler.step()
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            resolved_config=resolved,
            selection_metric="last",
            selection_value=None,
        )
        selections = {
            "best_val_loss.pt": ("val_loss", float(metrics["validation_loss"]), True),
            "best_single_macro.pt": ("single_macro", float(metrics["single_macro"]), False),
            "best_fix_rate.pt": (
                "fix_rate",
                None if metrics["fix_rate"] is None else float(metrics["fix_rate"]),
                False,
            ),
        }
        val_improved = False
        for filename, (name, value, minimize) in selections.items():
            improved = epoch == 1 or (
                value is not None and (value < best[name] if minimize else value > best[name])
            )
            if improved:
                if value is not None:
                    best[name] = value
                _save_checkpoint(
                    checkpoint_dir / filename,
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    resolved_config=resolved,
                    selection_metric=name,
                    selection_value=value,
                )
                if name == "val_loss":
                    val_improved = True
        patience_count = 0 if val_improved else patience_count + 1
        if patience_count >= int(training["patience"]):
            stop_reason = "early_stopping_patience"
            break
    best_payload = torch.load(checkpoint_dir / "best_val_loss.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state"])
    final_metrics = evaluate(
        model,
        validation_records,
        frequencies=frequencies,
        snr_db=float(training["validation_snr_db"]),
        batch_size=int(training["batch_size"]),
        device=device,
        seed=seed,
    )
    if full_validation_records is not None:
        full_metrics = evaluate(
            model,
            full_validation_records,
            frequencies=frequencies,
            snr_db=float(training["validation_snr_db"]),
            batch_size=int(training["batch_size"]),
            device=device,
            seed=seed,
        )
        final_metrics["full_top1"] = full_metrics["top1"]
    result = {
        "task": task,
        "seed": seed,
        "input_time": spec["input_time"],
        "target_time": spec["target_time"],
        "CSI_history_length": spec["history_length"],
        "mask_scope": "single_macro" if spec["concat"] else "csi_only",
        "selected_epoch": int(best_payload["epoch"]),
        "epochs_ran": len(history_rows),
        "stop_reason": stop_reason,
        "outer_test_accessed": False,
        **final_metrics,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def _baseline_rows(validation: Mapping[str, Any], seeds: list[int]) -> list[dict[str, Any]]:
    labels = validation["labels_future"]
    power = validation["future_beam_power"]
    per_mask = [
        _prediction_metrics(validation[f"p0_{mask}"], labels, power, validation[f"p0_{mask}"])
        for mask in SEVERE_MASKS
    ]
    mean = {
        key: float(np.mean([float(row[key]) for row in per_mask if row[key] is not None]))
        for key in ("top1", "top3", "top5", "within3", "mae", "normalized_gain", "beam_loss_db")
    }
    worst = min(float(row["top1"]) for row in per_mask)
    return [
        {
            "task": "I0",
            "seed": seed,
            "input_time": "Frozen U0 p0",
            "target_time": "t+1",
            "CSI_history_length": 0,
            "mask_scope": "single_macro",
            "selected_epoch": 0,
            "epochs_ran": 0,
            "stop_reason": "frozen_baseline",
            "outer_test_accessed": False,
            **mean,
            "validation_loss": float(
                np.mean(
                    [
                        F.nll_loss(validation[f"p0_{mask}"].clamp_min(1e-12).log(), labels).item()
                        for mask in SEVERE_MASKS
                    ]
                )
            ),
            "single_macro": mean["top1"],
            "single_worst": worst,
            "all14_macro": None,
            "all14_worst": None,
            "full_top1": float(validation["p0_full"].argmax(dim=-1).eq(labels).float().mean().item()),
            "fix_rate": 0.0,
            "harm_rate": 0.0,
            "p_final_p0_kl": 0.0,
            "q_transition_entropy": float(
                -(validation["p0_full"] * validation["p0_full"].clamp_min(1e-12).log())
                .sum(dim=-1)
                .mean()
                .item()
            ),
        }
        for seed in seeds
    ]


def summarize(args: argparse.Namespace, config: dict[str, Any]) -> None:
    seeds = [int(value) for value in config["training"]["seeds"]]
    validation = torch.load(
        args.output_root / "records" / "validation.pt", map_location="cpu", weights_only=False
    )
    rows = _baseline_rows(validation, seeds)
    missing: list[str] = []
    for seed in seeds:
        for task in TASKS:
            path = args.output_root / "results" / f"seed{seed}_{task}.json"
            if not path.is_file():
                missing.append(str(path))
            else:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    if missing:
        raise FileNotFoundError("Missing recovery results:\n" + "\n".join(missing))
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        table_rows.append(
            {
                "row_type": "seed",
                "task": row["task"],
                "seed": row["seed"],
                "input_time": row["input_time"],
                "target_time": row["target_time"],
                "CSI_history_length": row["CSI_history_length"],
                "mask_scope": row["mask_scope"],
                "Top1": row["top1"],
                "Top3": row["top3"],
                "Top5": row["top5"],
                "Within3": row["within3"],
                "MAE": row["mae"],
                "normalized_gain": row["normalized_gain"],
                "beam_loss_db": row["beam_loss_db"],
                "Fix_Rate": row["fix_rate"],
                "Harm_Rate": row["harm_rate"],
                "Single_Macro": row["single_macro"],
                "Single_Worst": row["single_worst"],
                "Full_Top1": row["full_top1"],
                "selected_epoch": row["selected_epoch"],
                "epochs_ran": row["epochs_ran"],
                "stop_reason": row["stop_reason"],
            }
        )
    metric_names = (
        "Top1",
        "Top3",
        "Top5",
        "Within3",
        "MAE",
        "normalized_gain",
        "beam_loss_db",
        "Fix_Rate",
        "Harm_Rate",
        "Single_Macro",
        "Single_Worst",
        "Full_Top1",
    )
    for task in ("I0", *TASKS):
        task_rows = [row for row in table_rows if row["task"] == task and row["row_type"] == "seed"]
        aggregate = dict(task_rows[0])
        aggregate.update(row_type="mean", seed="mean", selected_epoch="", epochs_ran="", stop_reason="")
        for metric in metric_names:
            values = [float(row[metric]) for row in task_rows if row[metric] is not None]
            aggregate[metric] = float(np.mean(values)) if values else None
        table_rows.append(aggregate)
    _write_csv(args.output_root / "information_diagnostics.csv", table_rows)
    by_task_seed = {(row["task"], int(row["seed"])): row for row in table_rows if row["row_type"] == "seed"}
    deltas = {
        task: [
            float(by_task_seed[(task, seed)]["Single_Macro"])
            - float(by_task_seed[("I0", seed)]["Single_Macro"])
            for seed in seeds
        ]
        for task in ("I4", "I5")
    }
    train = torch.load(args.output_root / "records" / "train.pt", map_location="cpu", weights_only=False)
    majority_class = int(torch.bincount(train["labels_current"], minlength=64).argmax().item())
    majority_top1 = float(validation["labels_current"].eq(majority_class).float().mean().item())
    d1_mean = float(next(row for row in table_rows if row["row_type"] == "mean" and row["task"] == "I1")["Top1"])
    best_concat = max(deltas, key=lambda task: float(np.mean(deltas[task])))
    recommend_a1 = d1_mean > majority_top1 and all(delta > 0.0 for delta in deltas[best_concat])
    decision = {
        "D1_mean_top1": d1_mean,
        "D1_majority_baseline_top1": majority_top1,
        "concat_single_macro_deltas": deltas,
        "preferred_csi_input": "history" if best_concat == "I5" else "single_frame",
        "recommend_stage_a1": recommend_a1,
        "outer_test_accessed": False,
    }
    (args.output_root / "diagnostic_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_final_report(args.output_root / "final_report.md", table_rows, decision)
    print(json.dumps(decision, indent=2, sort_keys=True))


def _write_final_report(path: Path, rows: list[dict[str, Any]], decision: Mapping[str, Any]) -> None:
    means = {row["task"]: row for row in rows if row["row_type"] == "mean"}
    lines = [
        "# Sparse-Pilot CSI 信息恢复诊断",
        "",
        "三 seed、2,000/1,000 development；固定 4x8 QPSK pilot，train SNR [-10,30] dB，validation 10 dB 且无 dropout；outer test 未访问。",
        "",
        "| Task | 输入 -> 目标 | Scope | Top-1 | Top-3 | Top-5 | Within-3 | MAE | Full Top-1 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in ("I0", *TASKS):
        row = means[task]
        full_top1 = "-" if row["Full_Top1"] is None else f"{row['Full_Top1']:.4f}"
        lines.append(
            f"| {task} | {row['input_time']} -> {row['target_time']} | {row['mask_scope']} | "
            f"{row['Top1']:.4f} | {row['Top3']:.4f} | {row['Top5']:.4f} | {row['Within3']:.4f} | {row['MAE']:.3f} | "
            f"{full_top1} |"
        )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            f"1. 当前 CSI 能否预测 beam_t：I1 Top-1={means['I1']['Top1']:.4f}，train-majority validation baseline={decision['D1_majority_baseline_top1']:.4f}。",
            f"2. 单帧 CSI 能否预测 beam_t+1：I2 Top-1={means['I2']['Top1']:.4f}。",
            f"3. 历史 CSI 是否更有效：I3-I2 Top-1={means['I3']['Top1'] - means['I2']['Top1']:+.4f}。",
            f"4. 原有失败是否仅由 alpha 坍缩导致：信息诊断不含 alpha；需 forced transition 成功后才能归因，不能提前肯定。",
            f"5. forced transition 是否产生真实 Fix：尚未运行；recommend_stage_a1={decision['recommend_stage_a1']}。",
            "6. reliability gate 是否实现兜底：本阶段按协议禁用 gate，尚无新证据。",
            "7. selector 是否值得保留：本阶段固定 pattern，selector 继续暂缓。",
            "8. 泄漏/主导/Full 风险：未读取 future channel 或 outer test；concat 主结果为 single masks，Full 仅作只读约束。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-pilot CSI information recovery diagnostics.")
    parser.add_argument("--mode", choices=("prepare", "train", "summarize"), required=True)
    parser.add_argument("--config", type=Path, default=Path("tools/configs/sparse_pilot_recovery.yaml"))
    parser.add_argument(
        "--u0-config", type=Path, default=Path("outputs/full_pool_capacity/protocol/u0_seed1_config.yaml")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/full_pool_capacity/u0_seed1/checkpoints/last.pth")
    )
    parser.add_argument(
        "--expected-sha256", default="ed909406a37ec4ccd2b08bd1fb65ab66fc437cec226a526fdaf7ada1407ba8cf"
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/sparse_pilot_recovery"))
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = safe_load_yaml(args.config.read_text(encoding="utf-8"))
    if args.mode == "prepare":
        prepare(args, config)
    elif args.mode == "train":
        train(args, config)
    else:
        summarize(args, config)


if __name__ == "__main__":
    main()
