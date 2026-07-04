import json
import math
from typing import Any, Mapping

import numpy as np

from kd_sensing.evaluation.metrics import dba_from_circular_distances, dba_zero_ratio


def summary_from_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    protocol: str,
    ablation: str,
    target: Any,
    source_scenes: list[str],
    label_space: str,
    mapping: Any,
    protocol_note: str,
    support_info: Mapping[str, Any],
    scaler_metadata: Mapping[str, Any],
    adapter_fit: Any,
    dba_delta: float,
    num_beams: int,
) -> dict[str, Any]:
    metrics = metrics_from_prediction_rows(rows, num_beams=num_beams, dba_delta=dba_delta)
    return {
        "protocol": protocol,
        "ablation": ablation,
        "scene": target.slug,
        "scene_name": target.name,
        "target_scene": target.slug,
        "source_scenes": json.dumps(source_scenes),
        "label_space": label_space,
        "beam_label_space": mapping.label_space,
        "beam_label_mapping_fingerprint": mapping.fingerprint,
        "protocol_note": protocol_note,
        "support_count": int(support_info.get("support_count", 0)),
        "query_count": int(support_info.get("query_count", 0)),
        "support_mode": support_info.get("selection_mode", "none"),
        "strict_eligibility": protocol != "within_scene_train",
        "upper_bound_protocol": protocol == "within_scene_train",
        "adapter_fit": json.dumps(adapter_fit.to_dict()),
        "scaler_metadata": json.dumps(dict(scaler_metadata)),
        **metrics,
    }


def metrics_from_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    num_beams: int,
    dba_delta: float,
) -> dict[str, float | int]:
    sample_count = len(rows)
    distances = np.asarray([float(row.get("circular_error", 0.0)) for row in rows], dtype=np.float64)
    if distances.size == 0:
        return {
            "sample_count": 0,
            "valid_label_count": 0,
            "DBA": 0.0,
            "DBA_zero_ratio": 0.0,
            "mean_circular_error": 0.0,
            "median_circular_error": 0.0,
            "exact_acc": 0.0,
            "pm1_acc": 0.0,
            "pm2_acc": 0.0,
            "pm4_acc": 0.0,
            "top1": 0.0,
            "top3": 0.0,
            "top5": 0.0,
        }
    target = [int(row["true_beam"]) for row in rows]
    topk = [_json_list_int(row.get("topk_predictions")) for row in rows]

    def top_hit(k: int) -> float:
        hits = 0
        for truth, preds in zip(target, topk):
            if int(truth) in [int(item) % int(num_beams) for item in preds[:k]]:
                hits += 1
        return float(hits / max(len(target), 1))

    return {
        "sample_count": sample_count,
        "valid_label_count": sample_count,
        "DBA": dba_from_circular_distances(distances, delta=dba_delta),
        "DBA_zero_ratio": dba_zero_ratio(distances),
        "mean_circular_error": float(np.mean(distances)),
        "median_circular_error": float(np.median(distances)),
        "exact_acc": float(np.mean(distances == 0)),
        "pm1_acc": float(np.mean(distances <= 1)),
        "pm2_acc": float(np.mean(distances <= 2)),
        "pm4_acc": float(np.mean(distances <= 4)),
        "top1": top_hit(1),
        "top3": top_hit(3),
        "top5": top_hit(5),
    }


def overall_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["protocol"]), str(row["ablation"]), str(row["label_space"])), []).append(row)
    result = []
    metric_keys = [
        "DBA",
        "DBA_zero_ratio",
        "mean_circular_error",
        "median_circular_error",
        "exact_acc",
        "pm1_acc",
        "pm2_acc",
        "pm4_acc",
        "top1",
        "top3",
        "top5",
    ]
    for (protocol, ablation, label_space), rows in sorted(grouped.items()):
        payload: dict[str, Any] = {
            "protocol": protocol,
            "ablation": ablation,
            "label_space": label_space,
            "scene_count": len(rows),
            "valid_label_count": sum(int(row.get("valid_label_count", 0)) for row in rows),
        }
        for key in metric_keys:
            payload[key] = float(np.mean([float(row.get(key, 0.0)) for row in rows])) if rows else 0.0
        result.append(payload)
    return result


def residual_by_theta_rows(rows: list[dict[str, Any]], *, bins: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        bin_id = int(math.floor((float(row["theta_degrees"]) % 360.0) / 360.0 * int(bins))) % int(bins)
        key = (row["protocol"], row["ablation"], row["scene"], row["label_space"], bin_id)
        grouped.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(grouped.items()):
        result.append(
            {
                "protocol": key[0],
                "ablation": key[1],
                "scene": key[2],
                "label_space": key[3],
                "theta_bin": key[4],
                "count": len(values),
                "mean_circular_error": float(np.mean([float(row["circular_error"]) for row in values])),
                "mean_signed_residual": float(np.mean([float(row["signed_residual"]) for row in values])),
            }
        )
    return result


def residual_by_branch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["protocol"], row["ablation"], row["scene"], row["label_space"], row["branch_id"])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(grouped.items()):
        result.append(
            {
                "protocol": key[0],
                "ablation": key[1],
                "scene": key[2],
                "label_space": key[3],
                "branch_id": key[4],
                "count": len(values),
                "mean_circular_error": float(np.mean([float(row["circular_error"]) for row in values])),
                "mean_signed_residual": float(np.mean([float(row["signed_residual"]) for row in values])),
                "branch_source": values[0].get("branch_source", ""),
            }
        )
    return result


def support_manifest_rows(
    support: list[Any],
    query: list[Any],
    *,
    protocol: str,
    target_scene: str,
    label_space: str,
    support_info: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for role, samples in (("support", support), ("query", query)):
        for sample in samples:
            rows.append(
                {
                    "protocol": protocol,
                    "target_scene": target_scene,
                    "label_space": label_space,
                    "beam_label_space": _sample_metadata(sample).get("beam_label_space", ""),
                    "beam_label_mapping_fingerprint": _sample_attr(sample, "mapping_fingerprint"),
                    "role": role,
                    "sample_id": _sample_attr(sample, "sample_id"),
                    "scene": _sample_attr(sample, "scene"),
                    "split": _sample_attr(sample, "split"),
                    "target_label": _sample_attr(sample, "label"),
                    "order_key": _sample_attr(sample, "order_key"),
                    "selection_mode": support_info.get("selection_mode", "none"),
                    "seed": support_info.get("seed", ""),
                }
            )
    return rows


def _json_list_int(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value in {None, ""}:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [int(item) for item in payload] if isinstance(payload, list) else []


def _sample_metadata(sample: Any) -> Mapping[str, Any]:
    value = _sample_attr(sample, "metadata", {})
    return value if isinstance(value, Mapping) else {}


def _sample_attr(sample: Any, name: str, default: Any = "") -> Any:
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return getattr(sample, name, default)


__all__ = [
    "metrics_from_prediction_rows",
    "overall_rows",
    "residual_by_branch_rows",
    "residual_by_theta_rows",
    "summary_from_prediction_rows",
    "support_manifest_rows",
]
