from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tools.visualization.viewer_constants import DISTRIBUTION_MODALITIES, MODALITIES
from tools.visualization.viewer_manifest_io import _natural_key, _numeric_score_items, safe_get

def single_modality_confidence_dataframe(sample: dict[str, Any] | None) -> pd.DataFrame:
    return dict_to_dataframe(single_modality_t1_confidence(sample), "confidence")


def single_modality_t1_confidence(sample: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(sample, dict):
        return {}
    scores: dict[str, float] = {}
    for modality in DISTRIBUTION_MODALITIES:
        value = _t1_confidence_for_modality(sample, modality)
        if value is not None:
            scores[modality] = value
    for modality in _extra_confidence_modalities(sample):
        if modality in scores:
            continue
        value = _t1_confidence_for_modality(sample, modality)
        if value is not None:
            scores[modality] = value
    return scores



def get_future_beams(sample: dict[str, Any] | None) -> list[int]:
    return _beam_label_values(safe_get(sample, "label.future_beams") if sample else None)


def get_horizon_choices(sample: dict[str, Any] | None) -> list[str]:
    return [f"t+{index + 1}" for index, _ in enumerate(get_future_beams(sample))]


def parse_horizon_label(horizon_label: str | None) -> int:
    text = str(horizon_label or "").strip().lower().replace(" ", "")
    if not text:
        return 0
    if text.startswith("t+"):
        text = text[2:]
    elif text.startswith("+"):
        text = text[1:]
    try:
        value = int(text)
    except ValueError:
        return 0
    return max(0, value - 1)


def extract_beam_distribution(sample: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(sample, dict):
        return {}
    for source in (sample.get("beam_distribution"), safe_get(sample, "prediction.beam_distribution")):
        if isinstance(source, dict):
            return source
    return {}


def get_distribution_for_modality(
    sample: dict[str, Any] | None,
    modality: str,
    horizon_index: int,
    view_type: str,
) -> list[float] | None:
    distribution = extract_beam_distribution(sample)
    entry = distribution.get(modality)
    values = _distribution_values(entry, _view_distribution_key(view_type), horizon_index)
    if values is None:
        return None
    return values.astype(float).tolist()


def get_probability_for_modality(
    sample: dict[str, Any] | None,
    modality: str,
    horizon_index: int,
) -> list[float] | None:
    distribution = extract_beam_distribution(sample)
    values = _distribution_values(distribution.get(modality), "prob", horizon_index)
    if values is None:
        return None
    return values.astype(float).tolist()


def compute_rank(values: Any, target_index: int | None) -> int | None:
    if target_index is None:
        return None
    array = _numeric_array(values)
    if array is None:
        return None
    array = array.reshape(-1)
    if target_index < 0 or target_index >= array.size:
        return None
    target_value = float(array[target_index])
    return int(np.sum(array > target_value) + 1)


def compute_entropy(prob_values: Any) -> float | None:
    array = _numeric_array(prob_values)
    if array is None:
        return None
    probs = np.clip(array.reshape(-1).astype(np.float64), 0.0, None)
    total = float(np.sum(probs))
    if total <= 1e-12:
        return None
    if not np.isclose(total, 1.0, rtol=1e-3, atol=1e-6):
        probs = probs / total
    eps = 1e-12
    return float(-np.sum(probs * np.log(np.clip(probs, eps, None))))


def compute_top1_top2_margin(values: Any) -> float | None:
    array = _numeric_array(values)
    if array is None:
        return None
    flat = array.reshape(-1)
    if flat.size < 2:
        return None
    top2 = np.partition(flat, -2)[-2:]
    top2.sort()
    return float(top2[-1] - top2[-2])


def make_future_distribution_summary(
    sample: dict[str, Any] | None,
    horizon_label: str | None,
    view_type: str,
    show_fusion: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    horizon, horizon_index, gt_beam = _resolved_horizon(sample, horizon_label)
    for modality in _distribution_modalities(show_fusion):
        row = _distribution_summary_row(sample, modality, horizon, horizon_index, gt_beam, view_type)
        if row is not None:
            rows.append(row)
            continue
        legacy_row = _legacy_prediction_summary_row(sample, modality, horizon, horizon_index, gt_beam, view_type)
        if legacy_row is not None:
            rows.append(legacy_row)
    return pd.DataFrame(rows, columns=_future_distribution_summary_columns())



def build_future_distribution_detail(
    sample: dict[str, Any] | None,
    horizon_label: str | None,
    view_type: str,
    show_fusion: bool = True,
) -> dict[str, Any]:
    horizon, horizon_index, gt_beam = _resolved_horizon(sample, horizon_label)
    detail: dict[str, Any] = {
        "horizon": horizon,
        "horizon_index": int(horizon_index),
        "gt_beam": _native_int(gt_beam),
        "view_type": str(view_type or "probability"),
        "modalities": {},
    }
    _, _, warnings = _distribution_matrix(sample, horizon_index, view_type, show_fusion)
    if warnings:
        detail["warnings"] = warnings

    for modality in _distribution_modalities(show_fusion):
        row = _distribution_summary_row(sample, modality, horizon, horizon_index, gt_beam, view_type)
        if row is None:
            row = _legacy_prediction_summary_row(sample, modality, horizon, horizon_index, gt_beam, view_type)
        if row is None:
            continue
        detail["modalities"][modality] = {
            "top1_beam": _native_int(row.get("top1_beam")),
            "top1_value": _native_float(row.get("top1_value")),
            "gt_value": _native_float(row.get("gt_value")),
            "gt_rank": _native_int(row.get("gt_rank")),
            "entropy": _native_float(row.get("entropy")),
            "is_correct": row.get("is_correct"),
            "distance_to_gt": _native_int(row.get("distance_to_gt")),
        }
    if not detail["modalities"]:
        detail["message"] = "Future beam distribution not available"
    return detail


def dict_to_dataframe(score_dict: dict[str, Any] | None, value_name: str) -> pd.DataFrame:
    items = _numeric_score_items(score_dict)
    return pd.DataFrame(items, columns=["modality", value_name])


def build_info(sample: dict[str, Any] | None) -> dict[str, Any]:
    if not sample:
        return {"message": "No samples found"}
    keys = ("sample_id", "scene_id", "scene_slug", "split", "sequence_id", "time_index", "timestamp")
    info = {key: sample.get(key) for key in keys if key in sample}
    for key in ("label", "prediction", "extra"):
        if key in sample:
            info[key] = sample.get(key)
    return info


def _same_sequence(sample: dict[str, Any], current_sample: dict[str, Any]) -> bool:
    return (
        sample.get("scene_id", sample.get("scene_slug")) == current_sample.get("scene_id", current_sample.get("scene_slug"))
        and sample.get("split") == current_sample.get("split")
        and sample.get("sequence_id") == current_sample.get("sequence_id")
    )


def _first_future_beam(sample: dict[str, Any]) -> int | None:
    values = _beam_label_values(safe_get(sample, "label.future_beams"))
    return int(values[0]) if values else None


def _sample_time_sort_key(sample: dict[str, Any]) -> tuple[int, int]:
    return (_sortable_int(sample.get("time_index")), _sortable_int(sample.get("_manifest_index")))


def _current_sample_position(samples: list[dict[str, Any]], current_sample: dict[str, Any]) -> int | None:
    current_identity = (
        current_sample.get("_manifest_dir"),
        current_sample.get("sample_id"),
        current_sample.get("_manifest_index"),
    )
    for index, sample in enumerate(samples):
        identity = (sample.get("_manifest_dir"), sample.get("sample_id"), sample.get("_manifest_index"))
        if identity == current_identity:
            return index
    return None


def _sortable_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0



def _extra_confidence_modalities(sample: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for source in (
        sample.get("confidence"),
        sample.get("confidence_curves"),
        sample.get("beam_distribution"),
        sample.get("modality_prediction"),
        sample.get("modality_predictions"),
        safe_get(sample, "prediction.modalities"),
    ):
        if isinstance(source, dict):
            keys.extend(str(key) for key in source.keys())
    order = {modality: index for index, modality in enumerate(DISTRIBUTION_MODALITIES)}
    return sorted(set(keys), key=lambda item: (order.get(item, len(order)), _natural_key(item)))


def _t1_confidence_for_modality(sample: dict[str, Any], modality: str) -> float | None:
    prob_values = get_probability_for_modality(sample, modality, 0)
    if prob_values:
        return _round_float(float(np.max(np.asarray(prob_values, dtype=np.float64))))

    for source in (
        safe_get(sample, f"confidence_curves.{modality}"),
        safe_get(sample, f"prediction.confidence_curves.{modality}"),
        safe_get(sample, f"prediction.per_label_confidence.{modality}"),
        safe_get(sample, f"prediction.probabilities.{modality}"),
        safe_get(sample, f"prediction.probs.{modality}"),
        safe_get(sample, f"prediction.scores.{modality}"),
    ):
        value = _first_curve_max(source)
        if value is not None:
            return _round_float(value)

    for source in (
        safe_get(sample, f"modality_prediction.{modality}"),
        safe_get(sample, f"modality_predictions.{modality}"),
        safe_get(sample, f"prediction.modalities.{modality}"),
    ):
        if not isinstance(source, dict):
            continue
        value = _float_at(
            source.get("top1_confidence", source.get("top1_prob", source.get("top1_probability"))),
            0,
        )
        if value is not None:
            return _round_float(value)

    value = sample.get("confidence", {}).get(modality) if isinstance(sample.get("confidence"), dict) else None
    if isinstance(value, (list, tuple, np.ndarray)):
        curve_value = _first_curve_max(value)
        if curve_value is not None:
            return _round_float(curve_value)
    native = _native_float(value)
    return _round_float(native)


def _first_curve_max(values: Any) -> float | None:
    array = _numeric_array(values)
    if array is None:
        return None
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        return float(np.max(array))
    first = np.asarray(array[0], dtype=np.float64).reshape(-1)
    if first.size == 0:
        return None
    return float(np.max(first))


def _future_distribution_summary_columns() -> list[str]:
    return [
        "modality",
        "horizon",
        "gt_beam",
        "top1_beam",
        "top1_value",
        "gt_value",
        "gt_rank",
        "top1_minus_gt",
        "top1_top2_margin",
        "entropy",
        "is_correct",
        "distance_to_gt",
    ]


def _distribution_modalities(show_fusion: bool) -> tuple[str, ...]:
    if show_fusion:
        return DISTRIBUTION_MODALITIES
    return MODALITIES


def _resolved_horizon(sample: dict[str, Any] | None, horizon_label: str | None) -> tuple[str, int, int | None]:
    future_beams = get_future_beams(sample)
    horizon_index = parse_horizon_label(horizon_label)
    if not future_beams:
        return "t+1", 0, None
    if horizon_index >= len(future_beams):
        horizon_index = 0
    return f"t+{horizon_index + 1}", horizon_index, int(future_beams[horizon_index])


def _view_distribution_key(view_type: str | None) -> str:
    text = str(view_type or "probability").strip().lower()
    return "logit" if text == "logit" else "prob"


def _distribution_values(entry: Any, key: str, horizon_index: int) -> np.ndarray | None:
    if entry is None:
        return None
    raw: Any
    if isinstance(entry, dict):
        raw = None
        aliases = {
            "prob": ("prob", "probs", "probability", "probabilities"),
            "logit": ("logit", "logits"),
        }[key]
        for alias in aliases:
            if alias in entry:
                raw = entry[alias]
                break
        if raw is None:
            return None
    else:
        if key != "prob":
            return None
        raw = entry

    array = _numeric_array(raw)
    if array is None:
        return None
    if array.ndim == 1:
        if horizon_index > 0:
            return None
        return array.reshape(-1)
    if horizon_index < 0 or horizon_index >= array.shape[0]:
        return None
    return np.asarray(array[horizon_index], dtype=np.float64).reshape(-1)


def _available_distribution_items(
    sample: dict[str, Any] | None,
    horizon_index: int,
    view_type: str,
    show_fusion: bool,
) -> list[tuple[str, list[float]]]:
    items = []
    for modality in _distribution_modalities(show_fusion):
        values = get_distribution_for_modality(sample, modality, horizon_index, view_type)
        if values:
            items.append((modality, values))
    return items


def _distribution_matrix(
    sample: dict[str, Any] | None,
    horizon_index: int,
    view_type: str,
    show_fusion: bool,
) -> tuple[list[str], list[list[float]], list[str]]:
    rows: list[str] = []
    values: list[list[float]] = []
    warnings: list[str] = []
    expected_size: int | None = None
    for modality in _distribution_modalities(show_fusion):
        modality_values = get_distribution_for_modality(sample, modality, horizon_index, view_type)
        if not modality_values:
            continue
        if expected_size is None:
            expected_size = len(modality_values)
        elif len(modality_values) != expected_size:
            warnings.append(
                f"Skipped {modality}: num_beams={len(modality_values)} differs from {expected_size}"
            )
            continue
        rows.append(modality)
        values.append([float(value) for value in modality_values])
    return rows, values, warnings


def _missing_distribution_title(sample: dict[str, Any] | None, view_type: str) -> str:
    if _view_distribution_key(view_type) == "logit" and extract_beam_distribution(sample):
        return "Logits not available"
    return "Future Beam Distribution Not Available"


def _distribution_summary_row(
    sample: dict[str, Any] | None,
    modality: str,
    horizon: str,
    horizon_index: int,
    gt_beam: int | None,
    view_type: str,
) -> dict[str, Any] | None:
    values = get_distribution_for_modality(sample, modality, horizon_index, view_type)
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    top1_beam = int(np.argmax(array))
    top1_value = float(array[top1_beam])
    gt_value = _value_at(array, gt_beam)
    prob_values = get_probability_for_modality(sample, modality, horizon_index)
    rank_source = prob_values if prob_values is not None else values
    gt_rank = compute_rank(rank_source, gt_beam)
    entropy = compute_entropy(prob_values)
    margin = compute_top1_top2_margin(values)
    return _summary_row(
        modality=modality,
        horizon=horizon,
        gt_beam=gt_beam,
        top1_beam=top1_beam,
        top1_value=top1_value,
        gt_value=gt_value,
        gt_rank=gt_rank,
        top1_top2_margin=margin,
        entropy=entropy,
    )


def _legacy_prediction_summary_row(
    sample: dict[str, Any] | None,
    modality: str,
    horizon: str,
    horizon_index: int,
    gt_beam: int | None,
    view_type: str,
) -> dict[str, Any] | None:
    if not isinstance(sample, dict):
        return None
    source = None
    for candidate in (
        safe_get(sample, f"modality_prediction.{modality}"),
        safe_get(sample, f"modality_predictions.{modality}"),
        safe_get(sample, f"prediction.modalities.{modality}"),
    ):
        if isinstance(candidate, dict):
            source = candidate
            break
    if source is None:
        return None

    top1_beam = _int_at(source.get("top1", source.get("top1_beam")), horizon_index)
    if top1_beam is None:
        return None
    top1_value = None
    gt_value = None
    if _view_distribution_key(view_type) == "prob":
        top1_value = _float_at(
            source.get("top1_confidence", source.get("top1_prob", source.get("top1_probability"))),
            horizon_index,
        )
        gt_value = _float_at(
            source.get(
                "future_label_confidence",
                source.get("gt_confidence", source.get("gt_probability", source.get("gt_prob"))),
            ),
            horizon_index,
        )
    gt_rank = _int_at(source.get("future_label_rank", source.get("gt_rank")), horizon_index)
    return _summary_row(
        modality=modality,
        horizon=horizon,
        gt_beam=gt_beam,
        top1_beam=top1_beam,
        top1_value=top1_value,
        gt_value=gt_value,
        gt_rank=gt_rank,
        top1_top2_margin=None,
        entropy=None,
    )


def _summary_row(
    *,
    modality: str,
    horizon: str,
    gt_beam: int | None,
    top1_beam: int | None,
    top1_value: float | None,
    gt_value: float | None,
    gt_rank: int | None,
    top1_top2_margin: float | None,
    entropy: float | None,
) -> dict[str, Any]:
    top1_minus_gt = None
    if top1_value is not None and gt_value is not None:
        top1_minus_gt = float(top1_value) - float(gt_value)
    is_correct = None
    distance_to_gt = None
    if top1_beam is not None and gt_beam is not None:
        is_correct = bool(int(top1_beam) == int(gt_beam))
        distance_to_gt = abs(int(top1_beam) - int(gt_beam))
    return {
        "modality": modality,
        "horizon": horizon,
        "gt_beam": _native_int(gt_beam),
        "top1_beam": _native_int(top1_beam),
        "top1_value": _round_float(top1_value),
        "gt_value": _round_float(gt_value),
        "gt_rank": _native_int(gt_rank),
        "top1_minus_gt": _round_float(top1_minus_gt),
        "top1_top2_margin": _round_float(top1_top2_margin),
        "entropy": _round_float(entropy),
        "is_correct": is_correct,
        "distance_to_gt": _native_int(distance_to_gt),
    }


def _value_at(values: Any, index: int | None) -> float | None:
    if index is None:
        return None
    array = _numeric_array(values)
    if array is None:
        return None
    flat = array.reshape(-1)
    if index < 0 or index >= flat.size:
        return None
    return float(flat[index])


def _int_at(values: Any, index: int) -> int | None:
    value = _value_from_sequence(values, index)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def _float_at(values: Any, index: int) -> float | None:
    value = _value_from_sequence(values, index)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _value_from_sequence(values: Any, index: int) -> Any:
    if values is None:
        return None
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return values.item() if index == 0 else None
        values = values.reshape(-1).tolist()
    if isinstance(values, (list, tuple, np.ndarray)):
        if len(values) == 0:
            return None
        if len(values) == 1 and index > 0:
            return values[0]
        if index < len(values):
            return values[index]
        return None
    return values if index == 0 else None


def _round_float(value: Any) -> float | None:
    number = _native_float(value)
    if number is None:
        return None
    return round(number, 4)


def _native_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _native_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def _display_value(value: Any) -> str:
    native = _native_int(value)
    return "N/A" if native is None else str(native)


def _display_float(value: Any) -> str:
    native = _native_float(value)
    return "N/A" if native is None else f"{native:.4f}"


def _beam_confidence_curves(sample: dict[str, Any]) -> list[tuple[str, list[int], list[float]]]:
    sources = [
        sample.get("beam_distribution"),
        safe_get(sample, "prediction.per_label_confidence"),
        safe_get(sample, "prediction.confidence_curves"),
        safe_get(sample, "prediction.label_confidence"),
        safe_get(sample, "prediction.modality_confidence"),
        safe_get(sample, "prediction.probabilities"),
        safe_get(sample, "prediction.probs"),
        safe_get(sample, "prediction.scores"),
        safe_get(sample, "prediction.logits"),
        sample.get("confidence_curves"),
        sample.get("label_confidence"),
        sample.get("confidence"),
    ]
    curves: list[tuple[str, list[int], list[float]]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for source in sources:
        for name, x_values, y_values in _extract_curve_items(source, "confidence"):
            key = (name, tuple(x_values))
            if key in seen:
                continue
            seen.add(key)
            curves.append((name, x_values, y_values))
    return curves


def _extract_curve_items(data: Any, default_name: str) -> list[tuple[str, list[int], list[float]]]:
    if data is None:
        return []
    if isinstance(data, dict):
        if isinstance(data.get("modalities"), dict):
            return _extract_curve_items(data["modalities"], default_name)
        for key in ("prob", "probability", "probabilities", "probs", "scores", "confidences", "confidence", "values", "logit", "logits"):
            if key in data:
                labels = data.get("labels", data.get("beam_labels"))
                return _curve_entries(default_name, data[key], labels=labels, logits=key in {"logit", "logits"})
        if _dict_has_numeric_labels(data):
            x_values = [int(float(key)) for key in data.keys()]
            y_values = []
            for value in data.values():
                try:
                    y_values.append(float(value))
                except (TypeError, ValueError):
                    return []
            pairs = sorted(zip(x_values, y_values), key=lambda item: item[0])
            return [(default_name, [item[0] for item in pairs], [item[1] for item in pairs])]
        curves: list[tuple[str, list[int], list[float]]] = []
        for key, value in data.items():
            if key in {"labels", "beam_labels"}:
                continue
            curves.extend(_extract_curve_items(value, str(key)))
        return curves
    return _curve_entries(default_name, data)


def _curve_entries(
    name: str,
    values: Any,
    *,
    labels: Any = None,
    logits: bool = False,
) -> list[tuple[str, list[int], list[float]]]:
    array = _numeric_array(values)
    if array is None or array.size <= 1:
        return []
    if logits:
        array = _softmax(array, axis=-1)
    if array.ndim == 1:
        x_values = _label_axis(labels, array.size)
        return [(name, x_values, array.astype(float).tolist())]
    if array.ndim == 2:
        x_values = _label_axis(labels, array.shape[1])
        return [(f"{name} t+1", x_values, array[0].astype(float).tolist())]
    if array.ndim == 3:
        flat = array.reshape((-1, array.shape[-1]))
        x_values = _label_axis(labels, array.shape[-1])
        return [(f"{name} t+1", x_values, flat[0].astype(float).tolist())]
    return []


def _label_axis(labels: Any, size: int) -> list[int]:
    raw = _numeric_array(labels)
    if raw is not None and raw.size == size:
        return raw.reshape(-1).astype(int).tolist()
    return list(range(size))


def _beam_label_values(values: Any) -> list[int]:
    array = _numeric_array(values)
    if array is None:
        return []
    return [int(value) for value in array.reshape(-1).tolist() if np.isfinite(value)]


def _dict_has_numeric_labels(data: dict[str, Any]) -> bool:
    if not data:
        return False
    for key, value in data.items():
        try:
            float(key)
            float(value)
        except (TypeError, ValueError):
            return False
    return True


def _softmax(array: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=axis, keepdims=True)
    return exp / np.clip(denom, 1e-12, None)



def _numeric_list(values: Any) -> list[float]:
    array = _numeric_array(values)
    if array is None:
        return []
    return array.reshape(-1).astype(float).tolist()


def _numeric_array(values: Any) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return array




__all__ = [
    "build_future_distribution_detail",
    "build_info",
    "compute_entropy",
    "compute_rank",
    "compute_top1_top2_margin",
    "dict_to_dataframe",
    "extract_beam_distribution",
    "get_distribution_for_modality",
    "get_future_beams",
    "get_horizon_choices",
    "get_probability_for_modality",
    "make_future_distribution_summary",
    "parse_horizon_label",
    "single_modality_confidence_dataframe",
    "single_modality_t1_confidence",
]
