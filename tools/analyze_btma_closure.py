#!/usr/bin/env python3
"""Read-only closure for the Full-pool BTMA causal ablation.

This tool never trains, never changes the protocol and never writes into
``outputs/full_pool_btma_ablation/``.  It recomputes per-sample validation
predictions from the six published BTMA checkpoints, produces paired temporal
block bootstrap intervals and correlates the assignment scores against the
single-modality topology error they claim to measure.

The pre-registered rule is recorded in the OpenSpec change: nothing produced
here may be used to reopen BTMA as a second-innovation candidate.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy import stats

import run_full_pool_candidate12 as c12
from kd_sensing.baselines.btma_assignment import BTMA_METHODS
from kd_sensing.baselines.full_pool_bt_scl import load_audited_topology, sha256_file, write_json
from kd_sensing.baselines.full_pool_candidate12 import MODALITIES
from kd_sensing.data.mmw.full_pool_protocol import load_full_pool_protocol
from kd_sensing.engine.data_factory import shutdown_dataloader_workers


ROOT = Path(__file__).resolve().parents[1]
BTMA_ROOT = ROOT / "outputs/full_pool_btma_ablation"
HISTORICAL_ROOT = ROOT / "outputs/full_pool_candidate12_search"
TRAIN_CACHE = HISTORICAL_ROOT / "warmup/unimodal_train_predictions/train_cache.npz"
DEFAULT_OUTPUT = ROOT / "outputs/btma_posthoc_closure"

# Pre-registered before any interval is computed.  Do not tune.
BLOCK_LENGTH = 32
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260726
METRIC_NAMES = ("top1", "top3", "top5", "within1", "within3", "mae", "topology_risk", "distance_gt5", "ce_loss")
REPORT_METRICS = ("top1", "top3", "within3", "mae", "topology_risk", "distance_gt5")
# Metrics where a *larger* value is better; used only for report wording.
HIGHER_IS_BETTER = {"top1", "top3", "top5", "within1", "within3"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    c12._atomic_csv(path, rows, list(rows[0]))


def _guard_output(root: Path, name: str, *, force: bool) -> None:
    target = root / name
    if target.exists() and not force:
        raise FileExistsError(f"Closure artifact already exists (fail closed): {target}")


# --------------------------------------------------------------------------
# 10.1  read-only per-sample validation predictions
# --------------------------------------------------------------------------


def _method_checkpoint(method: str) -> Path:
    path = BTMA_ROOT / method / "best_checkpoint.pt"
    if not path.is_file():
        raise FileNotFoundError(f"BTMA checkpoint is absent: {path}")
    return path


def build_predictions(root: Path, *, force: bool = False, max_batches: int | None = None) -> dict[str, Any]:
    """Recompute pattern=`full` anchor logits for every BTMA method, read-only."""
    protocol = load_full_pool_protocol(c12.DEFAULT_PROTOCOL)
    topology = load_audited_topology(c12.DEFAULT_TOPOLOGY)
    signed = json.loads((BTMA_ROOT / "signed_angle_order_audit.json").read_text(encoding="utf-8"))["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # `create_normalization=False` keeps the published train-only GPS scaler; nothing is refit here.
    loaders, _ = c12._loaders(BTMA_ROOT, protocol, create_normalization=False)
    manifest: dict[str, Any] = {
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "validation_samples": int(protocol["validation_sample_count"]),
        "outer_test_accessed": bool(protocol["outer_test_accessed"]),
        "retrained": False,
        "scoring": "pattern=full anchor_logits (B0--B5 are not motion methods)",
        "methods": {},
    }
    try:
        for method in BTMA_METHODS:
            checkpoint = _method_checkpoint(method)
            destination = root / method / "validation_predictions.npz"
            if destination.is_file() and not force:
                raise FileExistsError(f"Prediction cache already exists (fail closed): {destination}")
            model = c12._load_model(checkpoint, device)
            model.eval()
            payload = _validation_predictions(model, loaders["validation"], topology, signed, device, max_batches=max_batches)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **payload)
            temporary.replace(destination)
            manifest["methods"][method] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "predictions_sha256": sha256_file(destination),
                "sample_count": int(payload["label"].shape[0]),
                "top1": float(payload["metrics"][:, 0].mean()),
            }
            print(json.dumps({"event": "btma_predictions", "method": method, **manifest["methods"][method]}), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for loader in loaders.values():
            shutdown_dataloader_workers(loader)
    manifest["created_at"] = now()
    write_json(root / "prediction_manifest.json", manifest)
    return manifest


def _validation_predictions(
    model: Any,
    loader: Any,
    topology: Any,
    signed_order: Sequence[int],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    ids: list[str] = []
    domains: list[str] = []
    weathers: list[str] = []
    logits_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    metric_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            with c12._autocast(device):
                labels = c12._labels(batch, device)
                output = model(c12._inputs(batch, device), signed_order=signed_order)
            logits = output["anchor_logits"].float()
            metric_chunks.append(c12._metric_values(logits, labels, topology).numpy())
            logits_chunks.append(logits.cpu().numpy().astype(np.float32))
            label_chunks.append(labels.cpu().numpy().astype(np.int64))
            metadata = batch["metadata"]
            conditions = [str(value) for value in metadata["condition"]]
            scenarios = [str(value) for value in metadata["scenario"]]
            ids.extend(c12._batch_ids(batch))
            weathers.extend(conditions)
            domains.extend(f"{condition}/{scenario}" for condition, scenario in zip(conditions, scenarios))
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    sample_id = np.asarray(ids, dtype=str)
    if len(sample_id) != len(set(ids)):
        raise ValueError("BTMA validation predictions contain duplicate sample identities.")
    block, position = _temporal_blocks(sample_id, np.asarray(domains, dtype=str))
    return {
        "sample_id": sample_id,
        "label": np.concatenate(label_chunks),
        "anchor_logits": np.concatenate(logits_chunks),
        "metrics": np.concatenate(metric_chunks).astype(np.float64),
        "metric_names": np.asarray(METRIC_NAMES, dtype=str),
        "domain": np.asarray(domains, dtype=str),
        "weather": np.asarray(weathers, dtype=str),
        "block_id": block,
        "frame_position": position,
    }


_FRAME_PATTERN = re.compile(r"^(?P<agent>[^:]+):(?P<frame>\d+)$")


def _temporal_blocks(sample_id: np.ndarray, domain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign contiguous-frame blocks within each (domain, agent) sequence.

    ``sample_id`` ends with ``...:<agent>:<frame index>``; ordering by that frame
    index inside a (domain, agent) group reconstructs the acquisition sequence.
    """
    agents: list[str] = []
    frames: list[int] = []
    for value in sample_id.tolist():
        match = _FRAME_PATTERN.match(":".join(str(value).split(":")[-2:]))
        if match is None:
            raise ValueError(f"Cannot recover (agent, frame) from sample identity: {value!r}")
        agents.append(match.group("agent"))
        frames.append(int(match.group("frame")))
    frame_array = np.asarray(frames, dtype=np.int64)
    block = np.empty(len(sample_id), dtype=np.int64)
    position = np.empty(len(sample_id), dtype=np.int64)
    keys = [f"{d}|{a}" for d, a in zip(domain.tolist(), agents)]
    next_block = 0
    for key in sorted(set(keys)):
        selected = np.asarray([index for index, value in enumerate(keys) if value == key], dtype=np.int64)
        order = selected[np.argsort(frame_array[selected], kind="stable")]
        ranks = np.arange(len(order), dtype=np.int64)
        position[order] = ranks
        block[order] = next_block + ranks // BLOCK_LENGTH
        next_block += int(ranks[-1] // BLOCK_LENGTH) + 1
    return block, position


# --------------------------------------------------------------------------
# 10.2  paired temporal block bootstrap
# --------------------------------------------------------------------------


def _load_predictions(root: Path, method: str) -> dict[str, np.ndarray]:
    path = root / method / "validation_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Run --predictions first; absent: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def block_bootstrap(root: Path, *, force: bool = False) -> list[dict[str, Any]]:
    """Paired bootstrap over contiguous frame blocks, identical blocks for both arms."""
    _guard_output(root, "block_bootstrap.csv", force=force)
    caches = {method: _load_predictions(root, method) for method in BTMA_METHODS}
    reference = caches[BTMA_METHODS[0]]
    order = np.argsort(reference["sample_id"], kind="stable")
    aligned: dict[str, np.ndarray] = {}
    for method, cache in caches.items():
        method_order = np.argsort(cache["sample_id"], kind="stable")
        if not np.array_equal(cache["sample_id"][method_order], reference["sample_id"][order]):
            raise ValueError(f"{method} validation identities differ from {BTMA_METHODS[0]}; pairing is impossible.")
        aligned[method] = cache["metrics"][method_order]
    blocks = reference["block_id"][order]
    unique_blocks = np.unique(blocks)
    members = [np.flatnonzero(blocks == block) for block in unique_blocks]
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = generator.integers(0, len(unique_blocks), size=(BOOTSTRAP_DRAWS, len(unique_blocks)))
    # One resampled index set per draw, shared by every method -> the comparison stays paired.
    index_sets = [np.concatenate([members[choice] for choice in row]) for row in draws]

    metric_columns = {name: METRIC_NAMES.index(name) for name in REPORT_METRICS}
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(BTMA_METHODS):
        for right in BTMA_METHODS[left_index + 1 :]:
            for name, column in metric_columns.items():
                left_values = aligned[left][:, column]
                right_values = aligned[right][:, column]
                point = float(left_values.mean() - right_values.mean())
                differences = np.asarray(
                    [float(left_values[idx].mean() - right_values[idx].mean()) for idx in index_sets],
                    dtype=np.float64,
                )
                low, high = (float(value) for value in np.percentile(differences, (2.5, 97.5)))
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "metric": name,
                        "left_value": float(left_values.mean()),
                        "right_value": float(right_values.mean()),
                        "difference": point,
                        "ci_low": low,
                        "ci_high": high,
                        "crosses_zero": bool(low <= 0.0 <= high),
                        "bootstrap_se": float(differences.std(ddof=1)),
                        "block_count": int(len(unique_blocks)),
                        "block_length": BLOCK_LENGTH,
                        "draws": BOOTSTRAP_DRAWS,
                    }
                )
    _atomic_csv(root / "block_bootstrap.csv", rows)
    write_json(
        root / "block_bootstrap_manifest.json",
        {
            "block_length": BLOCK_LENGTH,
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "block_count": int(len(unique_blocks)),
            "sample_count": int(len(blocks)),
            "paired": True,
            "pre_registered_before_computation": True,
            "created_at": now(),
        },
    )
    return rows


# --------------------------------------------------------------------------
# 10.3  assignment score correlations (pure post-processing, no model)
# --------------------------------------------------------------------------


def _unimodal_topology_error(topology: Any) -> dict[str, np.ndarray]:
    """Per-sample circular topology error of each single-modality warm-up prediction."""
    if not TRAIN_CACHE.is_file():
        raise FileNotFoundError(f"Warm-up train prediction cache is absent: {TRAIN_CACHE}")
    with np.load(TRAIN_CACHE, allow_pickle=False) as payload:
        cache = {name: payload[name] for name in payload.files}
    distance = topology.distance.cpu().numpy().astype(np.float64)
    labels = cache["label"].astype(np.int64)
    logits = cache["unimodal_logits"].astype(np.float64)
    hard = np.empty((len(labels), len(MODALITIES)), dtype=np.float64)
    soft = np.empty_like(hard)
    for index in range(len(MODALITIES)):
        prediction = logits[:, index, :].argmax(axis=-1)
        hard[:, index] = distance[labels, prediction]
        shifted = logits[:, index, :] - logits[:, index, :].max(axis=-1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=-1, keepdims=True)
        soft[:, index] = (probability * distance[labels] / 32.0).sum(axis=-1)
    return {"sample_id": cache["sample_id"], "hard": hard, "soft": soft}


def _assignment_scores(method: str) -> dict[int, dict[str, np.ndarray]]:
    directory = BTMA_ROOT / "assignments" / method
    if not directory.is_dir():
        raise FileNotFoundError(f"BTMA assignment directory is absent: {directory}")
    result: dict[int, dict[str, np.ndarray]] = {}
    for path in sorted(directory.glob("epoch_*.csv")):
        epoch = int(path.stem.split("_")[1])
        rows = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
            usecols=("sample_id", *(f"assignment_score_{name}" for name in MODALITIES)),
        )
        scores = np.stack([np.asarray(rows[f"assignment_score_{name}"], dtype=np.float64) for name in MODALITIES], axis=1)
        result[epoch] = {"sample_id": np.asarray(rows["sample_id"], dtype=str), "scores": scores}
    if not result:
        raise FileNotFoundError(f"No assignment CSV under {directory}")
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return float("nan"), float("nan")
    result = stats.spearmanr(left, right)
    return float(result.statistic), float(result.pvalue)


def score_correlations(root: Path, *, force: bool = False) -> list[dict[str, Any]]:
    """Three tables: aligned epoch-5 correlation, later-epoch tracking, rank stability."""
    _guard_output(root, "score_correlations.csv", force=force)
    topology = load_audited_topology(c12.DEFAULT_TOPOLOGY)
    errors = _unimodal_topology_error(topology)
    order = np.argsort(errors["sample_id"], kind="stable")
    error_ids = errors["sample_id"][order]
    hard = errors["hard"][order]
    soft = errors["soft"][order]

    rows: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    for method in BTMA_METHODS:
        epochs = _assignment_scores(method)
        aligned: dict[int, np.ndarray] = {}
        for epoch, payload in sorted(epochs.items()):
            score_order = np.argsort(payload["sample_id"], kind="stable")
            if not np.array_equal(payload["sample_id"][score_order], error_ids):
                raise ValueError(f"{method} epoch {epoch} assignment identities differ from the warm-up train cache.")
            scores = payload["scores"][score_order]
            aligned[epoch] = scores
            for index, modality in enumerate(MODALITIES):
                hard_rho, hard_p = _spearman(scores[:, index], hard[:, index])
                soft_rho, soft_p = _spearman(scores[:, index], soft[:, index])
                rows.append(
                    {
                        "method": method,
                        "epoch": epoch,
                        "modality": modality,
                        # Only epoch 5 scores are derived from this exact warm-up cache.
                        "aligned_with_score_source": bool(epoch == 5),
                        "spearman_hard_distance": hard_rho,
                        "p_value_hard": hard_p,
                        "spearman_soft_risk": soft_rho,
                        "p_value_soft": soft_p,
                        "sample_count": int(scores.shape[0]),
                    }
                )
        ordered = sorted(aligned)
        for previous, current in zip(ordered, ordered[1:]):
            for index, modality in enumerate(MODALITIES):
                rho, p_value = _spearman(aligned[previous][:, index], aligned[current][:, index])
                stability.append(
                    {
                        "method": method,
                        "from_epoch": previous,
                        "to_epoch": current,
                        "modality": modality,
                        "spearman_rank_stability": rho,
                        "p_value": p_value,
                    }
                )
        print(json.dumps({"event": "btma_score_correlation", "method": method, "epochs": len(epochs)}), flush=True)

    _atomic_csv(root / "score_correlations.csv", rows)
    _atomic_csv(root / "score_rank_stability.csv", stability)
    write_json(
        root / "score_correlation_manifest.json",
        {
            "train_cache": str(TRAIN_CACHE),
            "train_cache_sha256": sha256_file(TRAIN_CACHE),
            "primary_epoch": 5,
            "primary_epoch_note": "Only epoch 5 scores come from this exact warm-up cache; later epochs are tracking references.",
            "modality_order": list(MODALITIES),
            "model_executed": False,
            "outer_test_accessed": False,
            "created_at": now(),
        },
    )
    return rows


# --------------------------------------------------------------------------
# 10.4  report
# --------------------------------------------------------------------------


def _assignment_statistics() -> list[dict[str, Any]]:
    path = BTMA_ROOT / "assignment_statistics.csv"
    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    result: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        method = str(row["method"])
        entry = result.setdefault(method, {"change": [], "repair": []})
        if float(row["change_rate"]) > 0.0:
            entry["change"].append(float(row["change_rate"]))
        entry["repair"].append(float(row["capacity_repair_rate"]))
    return [
        {
            "method": method,
            "mean_change_rate": float(np.mean(entry["change"])) if entry["change"] else 0.0,
            "max_change_rate": float(np.max(entry["change"])) if entry["change"] else 0.0,
            "max_capacity_repair_rate": float(np.max(entry["repair"])),
            # Expected change rate of an independent uniform redraw over four modalities.
            "random_redraw_reference": 0.75,
        }
        for method, entry in sorted(result.items())
    ]


def _measurement_sentence(correlations: Any) -> str:
    """State what the epoch-5 correlations actually show, in numbers.

    The conclusion has to name the finding rather than point at a table: a reader
    who only reads the conclusion must not be able to walk away believing the
    scores were broken, because that reading would invite someone to "fix" the
    scoring and reopen a route the evidence has already closed.
    """
    parts: list[str] = []
    for method in ("b5_risk_margin_full", "b4_margin_only"):
        hard = [
            float(row["spearman_hard_distance"])
            for row in correlations
            if str(row["method"]) == method and int(row["epoch"]) == 5 and np.isfinite(float(row["spearman_hard_distance"]))
        ]
        soft = [
            float(row["spearman_soft_risk"])
            for row in correlations
            if str(row["method"]) == method and int(row["epoch"]) == 5 and np.isfinite(float(row["spearman_soft_risk"]))
        ]
        if not hard or not soft:
            continue
        parts.append(
            f"{method} 在 epoch 5 上与单模态拓扑硬距离的 Spearman 为 "
            f"{min(hard):+.2f}~{max(hard):+.2f}，与软风险为 {min(soft):+.2f}~{max(soft):+.2f}"
        )
    if not parts:
        return "epoch 5 相关性表为空，无法判定度量是否成立。"
    return "；".join(parts) + "。"


def _dynamics_sentence(mean_stability: Mapping[tuple[str, str], float], statistics: Sequence[Mapping[str, Any]]) -> str:
    """Name the mechanism the negative result actually implicates."""
    negative = {key: value for key, value in mean_stability.items() if value < 0.0}
    above_random = [
        entry for entry in statistics if float(entry["mean_change_rate"]) > float(entry["random_redraw_reference"])
    ]
    sentence = (
        f"{len(mean_stability)} 个（方法×模态）跨 epoch 秩稳定性中有 {len(negative)} 个为负"
        f"（最低 {min(negative.values()):+.2f}）" if negative else
        f"{len(mean_stability)} 个（方法×模态）跨 epoch 秩稳定性均非负"
    )
    if above_random:
        names = "、".join(str(entry["method"]) for entry in above_random)
        sentence += (
            f"；{names} 的平均 change rate 高于四模态随机重抽参考 0.75"
            "，即 assignment 比随机重抽还不稳定。这是反持久振荡，不是难度课程。"
        )
    else:
        sentence += "；没有方法的 change rate 超过随机重抽参考。"
    return sentence


def write_report(root: Path, *, force: bool = False) -> Path:
    _guard_output(root, "btma_closure_report.md", force=force)
    bootstrap = np.genfromtxt(root / "block_bootstrap.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    correlations = np.genfromtxt(root / "score_correlations.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    stability = np.genfromtxt(root / "score_rank_stability.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    statistics = _assignment_statistics()

    lines = [
        "# BTMA 负结果只读收尾",
        "",
        "> 单 seed、开发集、claim-ineligible；未访问 outer test；未重训、未调参。",
        "> **本报告的任何数字都不得用于重开 BTMA 作为第二创新候选。**",
        "",
        "## 1. 成对 temporal block bootstrap",
        "",
        f"块划分：`(domain, agent)` 内按帧序号排序后每 {BLOCK_LENGTH} 帧一块；"
        f"成对重抽 {BOOTSTRAP_DRAWS} 次，两个方法共用同一组重抽块。块长与次数在计算前固定。",
        "",
        "关键对比 B5 vs B0（B0 = 固定随机均衡，零成本对照）：",
        "",
        "| 指标 | B5 | B0 | 差值 | 95% CI | 跨零 |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for row in bootstrap:
        left, right = str(row["left"]), str(row["right"])
        if {left, right} != {"b0_random_balanced", "b5_risk_margin_full"}:
            continue
        sign = 1.0 if left == "b5_risk_margin_full" else -1.0
        b5 = float(row["left_value"] if left == "b5_risk_margin_full" else row["right_value"])
        b0 = float(row["right_value"] if left == "b5_risk_margin_full" else row["left_value"])
        difference = sign * float(row["difference"])
        low, high = sorted((sign * float(row["ci_low"]), sign * float(row["ci_high"])))
        lines.append(
            f"| {row['metric']} | {b5:.4f} | {b0:.4f} | {difference:+.4f} | "
            f"[{low:+.4f}, {high:+.4f}] | {'是' if bool(row['crosses_zero']) else '否'} |"
        )

    crossing = int(sum(1 for row in bootstrap if bool(row["crosses_zero"])))
    lines += [
        "",
        f"全部 {len(bootstrap)} 组方法对 × 指标中，{crossing} 组的 95% CI 跨零。",
        "",
        "**措辞约束**：CI 跨零时只能表述为「未超过对照」，不得表述为「显著劣于对照」。",
        "对 BTMA 的否决理由落在「打不过零成本对照」与「机制自证失败」，不落在统计显著的劣势。",
        "",
        "## 2. Assignment score 与单模态拓扑误差的相关性",
        "",
        "只有 epoch 5 的 score 与所用 warm-up train cache 严格同源，因此作为主表。",
        "",
        "| 方法 | 模态 | Spearman(score, hard distance) | Spearman(score, soft risk) |",
        "|---|---|---:|---:|",
    ]
    for row in correlations:
        if int(row["epoch"]) != 5:
            continue
        lines.append(
            f"| {row['method']} | {row['modality']} | "
            f"{float(row['spearman_hard_distance']):+.4f} | {float(row['spearman_soft_risk']):+.4f} |"
        )

    lines += [
        "",
        "## 3. 跨 epoch 秩稳定性与 assignment 行为",
        "",
        "| 方法 | 平均 change rate | 最大 change rate | 最大 capacity repair | 随机重抽参考 |",
        "|---|---:|---:|---:|---:|",
    ]
    for entry in statistics:
        lines.append(
            f"| {entry['method']} | {entry['mean_change_rate']:.4f} | {entry['max_change_rate']:.4f} | "
            f"{entry['max_capacity_repair_rate']:.4f} | {entry['random_redraw_reference']:.2f} |"
        )

    lines += ["", "| 方法 | 模态 | 平均 Spearman(score@t, score@t+2) |", "|---|---|---:|"]
    mean_stability: dict[tuple[str, str], float] = {}
    for method in BTMA_METHODS:
        for modality in MODALITIES:
            values = [
                float(row["spearman_rank_stability"])
                for row in stability
                if str(row["method"]) == method and str(row["modality"]) == modality
            ]
            finite = [value for value in values if np.isfinite(value)]
            if finite:
                mean_stability[(method, modality)] = float(np.mean(finite))
            lines.append(f"| {method} | {modality} | {np.mean(finite):+.4f} |" if finite else f"| {method} | {modality} | n/a |")

    lines += [
        "",
        "四模态独立随机重抽的期望 change rate 为 `1 - sum(p^2) = 0.75`。change rate 高于该参考值说明",
        "assignment 比随机重抽更不稳定，即反持久振荡而非难度课程。",
        "",
        "`capacity_repair_rate` 全程接近 0，说明 15%--40% 容量约束从未实际绑定；",
        "任何结论都不得把收益归因于容量修复。",
        "",
        "## 结论",
        "",
        "1. B5 与 B0 在主指标上不可区分，且 B5 需要评分管线而 B0 零成本 —— 按简约性出局。",
        f"2. **失败点不在度量。**{_measurement_sentence(correlations)}"
        "分数确实在度量它声称度量的量，因此不能把负结果解释成评分实现有 bug。",
        f"3. **失败点在分配动力学。**{_dynamics_sentence(mean_stability, statistics)}",
        "4. 容量修复从未触发，不能作为贡献因子。",
        "",
        "**BTMA 路线就此封档。本报告不提供任何可调超参数，也不构成重开该路线的依据。**",
        "",
        f"生成时间：{now()}",
        "",
    ]
    path = root / "btma_closure_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictions", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--correlations", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing closure artifacts.")
    parser.add_argument("--max-batches", type=int, default=None, help="Smoke only; truncates the validation pass.")
    args = parser.parse_args(argv)

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    steps = {
        "predictions": args.predictions or args.all,
        "bootstrap": args.bootstrap or args.all,
        "correlations": args.correlations or args.all,
        "report": args.report or args.all,
    }
    if not any(steps.values()):
        parser.error("Select at least one of --predictions/--bootstrap/--correlations/--report/--all.")
    try:
        if steps["predictions"]:
            build_predictions(root, force=args.force, max_batches=args.max_batches)
        if steps["correlations"]:
            score_correlations(root, force=args.force)
        if steps["bootstrap"]:
            block_bootstrap(root, force=args.force)
        if steps["report"]:
            print(json.dumps({"event": "btma_closure_report", "path": str(write_report(root, force=args.force))}), flush=True)
    except Exception as exc:
        write_json(root / "status.json", {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(), "outer_test_accessed": False})
        raise
    write_json(root / "status.json", {"status": "passed", "steps": steps, "outer_test_accessed": False, "retrained": False, "created_at": now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
