from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DEFAULT_BUCKET_FEATURES = (
    "mmwave_entropy",
    "mmwave_top1_prob",
    "mmwave_top1_top2_margin",
    "mmwave_peak_sharpness",
    "mmwave_total_power",
    "mmwave_peak_drift",
    "range_to_bs",
    "delta_range",
    "delta_bearing",
    "angular_velocity",
    "gps_jump_magnitude",
    "beam_transition",
)


def compute_mmwave_state_features(mmwave: torch.Tensor | np.ndarray) -> pd.DataFrame:
    tensor = _as_float_tensor(mmwave)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"mmwave must have shape [B,T,C], got {tuple(tensor.shape)}.")
    last = tensor[:, -1, :]
    prev = tensor[:, -2, :] if tensor.shape[1] > 1 else last
    probs = F.softmax(last, dim=-1)
    top_probs, top_idx = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
    top1_prob = top_probs[:, 0]
    top2_prob = top_probs[:, 1] if top_probs.shape[1] > 1 else torch.zeros_like(top1_prob)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    peak_sharpness = top1_prob * float(probs.shape[-1])
    current_peak = torch.argmax(last, dim=-1)
    previous_peak = torch.argmax(prev, dim=-1)
    return pd.DataFrame(
        {
            "mmwave_entropy": entropy.cpu().numpy(),
            "mmwave_top1_prob": top1_prob.cpu().numpy(),
            "mmwave_top1_top2_margin": (top1_prob - top2_prob).cpu().numpy(),
            "mmwave_peak_sharpness": peak_sharpness.cpu().numpy(),
            "mmwave_total_power": last.sum(dim=-1).cpu().numpy(),
            "mmwave_peak_drift": torch.abs(current_peak - previous_peak).cpu().numpy(),
        }
    )


def compute_gps_state_features(gps: torch.Tensor | np.ndarray) -> pd.DataFrame:
    tensor = _as_float_tensor(gps)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-1] < 3:
        raise ValueError(f"gps must have shape [B,T,3+], got {tuple(tensor.shape)}.")
    last = tensor[:, -1, :]
    prev = tensor[:, -2, :] if tensor.shape[1] > 1 else last
    range_to_bs = last[:, 0]
    prev_range = prev[:, 0]
    bearing = torch.atan2(last[:, 1], last[:, 2])
    prev_bearing = torch.atan2(prev[:, 1], prev[:, 2])
    delta_range = range_to_bs - prev_range
    delta_bearing = _wrap_angle(bearing - prev_bearing)
    gps_jump = torch.sqrt(delta_range.square() + delta_bearing.square())
    return pd.DataFrame(
        {
            "range_to_bs": range_to_bs.cpu().numpy(),
            "bearing": bearing.cpu().numpy(),
            "delta_range": delta_range.cpu().numpy(),
            "delta_bearing": delta_bearing.cpu().numpy(),
            "angular_velocity": delta_bearing.cpu().numpy(),
            "gps_jump_magnitude": gps_jump.cpu().numpy(),
        }
    )


def compute_beam_transition_features(
    input_beam: torch.Tensor | np.ndarray,
    target_beam: torch.Tensor | np.ndarray,
    *,
    horizon_names: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    history = _as_long_tensor(input_beam)
    future = _as_long_tensor(target_beam)
    if history.ndim == 1:
        history = history.unsqueeze(0)
    if future.ndim == 1:
        future = future.unsqueeze(0)
    if history.ndim != 2 or future.ndim != 2:
        raise ValueError(
            f"input_beam and target_beam must have shape [B,T], got {tuple(history.shape)} and {tuple(future.shape)}."
        )
    names = list(horizon_names or [f"t+{idx + 1}" for idx in range(future.shape[1])])
    rows = []
    for batch_idx in range(future.shape[0]):
        previous = int(history[batch_idx, -1].item())
        for horizon_idx in range(future.shape[1]):
            current = int(future[batch_idx, horizon_idx].item())
            rows.append(
                {
                    "row_idx": int(batch_idx),
                    "horizon_idx": int(horizon_idx),
                    "horizon_name": names[horizon_idx],
                    "beam_transition": int(current != previous),
                    "beam_delta": int(abs(current - previous)),
                }
            )
            previous = current
    return pd.DataFrame(rows)


def communication_state_feature_records(
    batch: dict[str, Any],
    *,
    labels: torch.Tensor | np.ndarray | None = None,
    metadata: Any | None = None,
    horizon_names: list[str] | tuple[str, ...] | None = None,
    dataset_index_offset: int = 0,
) -> list[dict[str, Any]]:
    base: pd.DataFrame | None = None
    if "mmwave" in batch:
        base = _concat_feature_frame(base, compute_mmwave_state_features(batch["mmwave"]))
    if "gps" in batch:
        base = _concat_feature_frame(base, compute_gps_state_features(batch["gps"]))
    if base is None:
        first_tensor = next((value for value in batch.values() if torch.is_tensor(value)), None)
        batch_size = int(first_tensor.shape[0]) if first_tensor is not None and first_tensor.ndim > 0 else 0
        base = pd.DataFrame(index=range(batch_size))
    batch_size = len(base)
    if labels is None:
        labels = batch.get("target_beam")
    if "input_beam" in batch and labels is not None:
        transitions = compute_beam_transition_features(batch["input_beam"], labels, horizon_names=horizon_names)
    else:
        horizon_count = len(horizon_names or ["t+1"])
        transitions = pd.DataFrame(
            [
                {"row_idx": idx, "horizon_idx": horizon_idx, "horizon_name": f"t+{horizon_idx + 1}"}
                for idx in range(batch_size)
                for horizon_idx in range(horizon_count)
            ]
        )
    metadata_rows = _metadata_rows(metadata, batch_size, dataset_index_offset=dataset_index_offset)
    rows = []
    for _, transition in transitions.iterrows():
        row_idx = int(transition["row_idx"])
        row = dict(metadata_rows[row_idx])
        row.update(base.iloc[row_idx].to_dict())
        row.update({key: transition[key] for key in transitions.columns if key != "row_idx"})
        rows.append(_json_ready(row))
    return rows


def fit_bucket_thresholds(
    frame: pd.DataFrame,
    feature_names: Iterable[str] = DEFAULT_BUCKET_FEATURES,
    *,
    quantiles: list[float] | tuple[float, ...] = (1.0 / 3.0, 2.0 / 3.0),
    bucket_names: list[str] | tuple[str, ...] = ("low", "mid", "high"),
    binary_features: Iterable[str] = ("beam_transition",),
) -> dict[str, dict[str, Any]]:
    thresholds: dict[str, dict[str, Any]] = {}
    binary = set(binary_features)
    for feature in feature_names:
        if feature not in frame.columns:
            continue
        if feature in binary:
            thresholds[feature] = {
                "type": "binary",
                "names": ["stable", "transition"],
            }
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        if values.empty:
            continue
        qs = [float(q) for q in quantiles]
        names = list(bucket_names)
        if len(names) != len(qs) + 1:
            names = ["low", "high"] if len(qs) == 1 else ["low", "mid", "high"]
        thresholds[feature] = {
            "type": "quantile",
            "quantiles": qs,
            "thresholds": [float(values.quantile(q)) for q in qs],
            "names": names,
        }
    return thresholds


def assign_buckets(frame: pd.DataFrame, thresholds: dict[str, dict[str, Any]]) -> pd.DataFrame:
    result = frame.copy()
    for feature, spec in thresholds.items():
        if feature not in result.columns:
            continue
        result[f"{feature}_bucket"] = [
            assign_bucket(value, spec)
            for value in result[feature].tolist()
        ]
    return result


def assign_bucket(value: Any, spec: dict[str, Any]) -> str | None:
    if pd.isna(value):
        return None
    if spec.get("type") == "binary":
        return "transition" if int(value) != 0 else "stable"
    thresholds = [float(item) for item in spec.get("thresholds", [])]
    names = list(spec.get("names") or [])
    if not names:
        names = ["low", "high"] if len(thresholds) == 1 else ["low", "mid", "high"]
    numeric = float(value)
    bucket_idx = 0
    for threshold in thresholds:
        if numeric > threshold:
            bucket_idx += 1
    bucket_idx = min(bucket_idx, len(names) - 1)
    return str(names[bucket_idx])


def _as_float_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().float().cpu()
    return torch.as_tensor(value, dtype=torch.float32)


def _as_long_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.detach().long().cpu()
    return torch.as_tensor(value, dtype=torch.long)


def _wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def _concat_feature_frame(base: pd.DataFrame | None, other: pd.DataFrame) -> pd.DataFrame:
    if base is None:
        return other.reset_index(drop=True)
    return pd.concat([base.reset_index(drop=True), other.reset_index(drop=True)], axis=1)


def _metadata_rows(metadata: Any | None, batch_size: int, *, dataset_index_offset: int) -> list[dict[str, Any]]:
    if metadata is None:
        return [
            {"dataset_index": dataset_index_offset + idx, "sample_id": f"sample_{dataset_index_offset + idx}"}
            for idx in range(batch_size)
        ]
    if isinstance(metadata, list) and len(metadata) == batch_size and all(isinstance(item, dict) for item in metadata):
        return [_json_ready(item) for item in metadata]
    if isinstance(metadata, dict):
        rows = []
        for idx in range(batch_size):
            row = {}
            for key, value in metadata.items():
                item = _metadata_value_at(value, idx)
                if item is not None:
                    row[str(key)] = item
            row.setdefault("dataset_index", dataset_index_offset + idx)
            row.setdefault("sample_id", f"sample_{row['dataset_index']}")
            rows.append(_json_ready(row))
        return rows
    return [
        {"dataset_index": dataset_index_offset + idx, "sample_id": f"sample_{dataset_index_offset + idx}"}
        for idx in range(batch_size)
    ]


def _metadata_value_at(value: Any, idx: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        if idx < value.shape[0]:
            item = value[idx]
            return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if idx < value.shape[0]:
            item = value[idx]
            return item.item() if np.asarray(item).ndim == 0 else np.asarray(item).tolist()
        return None
    if isinstance(value, (list, tuple)):
        if idx < len(value):
            return value[idx]
        return None
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


__all__ = [
    "DEFAULT_BUCKET_FEATURES",
    "assign_bucket",
    "assign_buckets",
    "communication_state_feature_records",
    "compute_beam_transition_features",
    "compute_gps_state_features",
    "compute_mmwave_state_features",
    "fit_bucket_thresholds",
]
