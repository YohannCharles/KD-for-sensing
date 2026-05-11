from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


WEAK_MODALITIES = ("image", "radar", "lidar")
DEFAULT_ORACLE_CANDIDATES = (
    "strong_only",
    "strong_plus_image",
    "strong_plus_radar",
    "strong_plus_lidar",
    "all",
)


def records_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    subset_name: str,
    modalities: Iterable[str],
    metadata: Any | None = None,
    dba_delta: float = 5.0,
    horizon_names: list[str] | tuple[str, ...] | None = None,
    max_topk: int = 5,
    dataset_index_offset: int = 0,
) -> list[dict[str, Any]]:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B,H,C], got {tuple(logits.shape)}.")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [B,H], got {tuple(labels.shape)}.")
    if logits.shape[:2] != labels.shape:
        raise ValueError(f"logits/labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}.")

    batch_size, horizon_count, num_classes = logits.shape
    names = list(horizon_names or [f"t+{idx + 1}" for idx in range(horizon_count)])
    if len(names) < horizon_count:
        raise ValueError(f"horizon_names must contain at least {horizon_count} names.")
    k = min(int(max_topk), int(num_classes))
    probs = F.softmax(logits.detach().float().cpu(), dim=-1)
    top_probs, top_idx = torch.topk(probs, k=k, dim=-1)
    labels_cpu = labels.detach().cpu().to(torch.long)
    metadata_rows = _metadata_rows(metadata, batch_size, dataset_index_offset=dataset_index_offset)
    modality_text = ",".join(str(name) for name in modalities)

    rows: list[dict[str, Any]] = []
    for batch_idx in range(batch_size):
        base = dict(metadata_rows[batch_idx])
        dataset_index = int(base.get("dataset_index", dataset_index_offset + batch_idx))
        sample_id = str(base.get("sample_id", f"sample_{dataset_index}"))
        for horizon_idx in range(horizon_count):
            gt = int(labels_cpu[batch_idx, horizon_idx].item())
            valid = gt >= 0 and gt < num_classes
            pred_values = top_idx[batch_idx, horizon_idx].tolist()
            prob_values = top_probs[batch_idx, horizon_idx].tolist()
            gt_prob = float(probs[batch_idx, horizon_idx, gt].item()) if valid else float("nan")
            ce = float(-np.log(max(gt_prob, 1e-12))) if valid else float("nan")
            top1_hit = bool(valid and gt in pred_values[:1])
            top3_hit = bool(valid and gt in pred_values[: min(3, k)])
            top5_hit = bool(valid and gt in pred_values[: min(5, k)])
            beam_distance = int(abs(int(pred_values[0]) - gt)) if valid and pred_values else None
            dba_score = _dba_contribution(pred_values[: min(3, k)], gt, dba_delta) if valid else float("nan")
            row = {
                **base,
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                "horizon_idx": int(horizon_idx),
                "horizon_name": names[horizon_idx],
                "gt_beam": gt if valid else None,
                "valid": bool(valid),
                "subset_name": str(subset_name),
                "modalities": modality_text,
                "gt_prob": gt_prob,
                "ce": ce,
                "top1_hit": int(top1_hit),
                "top3_hit": int(top3_hit),
                "top5_hit": int(top5_hit),
                "beam_distance_top1": beam_distance,
                "dba_score": dba_score,
            }
            for rank in range(1, int(max_topk) + 1):
                if rank <= k:
                    row[f"pred_top{rank}"] = int(pred_values[rank - 1])
                    row[f"top{rank}_prob"] = float(prob_values[rank - 1])
                else:
                    row[f"pred_top{rank}"] = None
                    row[f"top{rank}_prob"] = float("nan")
            rows.append(row)
    return rows


def write_table(
    records_or_frame: list[dict[str, Any]] | pd.DataFrame,
    output_dir: str | Path,
    stem: str,
) -> dict[str, Any]:
    frame = _to_dataframe(records_or_frame)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = target_dir / f"{stem}.parquet"
    try:
        frame.to_parquet(parquet_path, index=False)
        return {
            "stem": stem,
            "format": "parquet",
            "path": str(parquet_path),
            "num_rows": int(len(frame)),
        }
    except Exception as exc:  # pragma: no cover - depends on optional parquet engine
        csv_path = target_dir / f"{stem}.csv.gz"
        frame.to_csv(csv_path, index=False, compression="gzip")
        return {
            "stem": stem,
            "format": "csv.gz",
            "path": str(csv_path),
            "num_rows": int(len(frame)),
            "parquet_error": str(exc),
        }


def read_table(path_or_dir: str | Path, stem: str | None = None) -> pd.DataFrame:
    path = Path(path_or_dir)
    if stem is not None:
        parquet_path = path / f"{stem}.parquet"
        csv_path = path / f"{stem}.csv.gz"
        if parquet_path.exists():
            path = parquet_path
        elif csv_path.exists():
            path = csv_path
        else:
            raise FileNotFoundError(f"Could not find {stem}.parquet or {stem}.csv.gz in {path}.")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.name.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def subset_prediction_records(
    logits_by_subset: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    subset_modalities: dict[str, Iterable[str]],
    metadata: Any | None = None,
    dba_delta: float = 5.0,
    horizon_names: list[str] | tuple[str, ...] | None = None,
    dataset_index_offset: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset_name, logits in logits_by_subset.items():
        rows.extend(
            records_from_logits(
                logits,
                labels,
                subset_name=subset_name,
                modalities=subset_modalities[subset_name],
                metadata=metadata,
                dba_delta=dba_delta,
                horizon_names=horizon_names,
                dataset_index_offset=dataset_index_offset,
            )
        )
    return rows


def aggregate_subset_metrics(records_or_frame: list[dict[str, Any]] | pd.DataFrame) -> dict[str, dict[str, Any]]:
    frame = _to_dataframe(records_or_frame)
    if frame.empty:
        return {}
    frame = frame[_valid_mask(frame)].copy()
    horizon_order = _horizon_order(frame)
    results: dict[str, dict[str, Any]] = {}
    for subset_name, subset_frame in frame.groupby("subset_name", sort=False):
        metrics = {
            "loss": float(subset_frame["ce"].mean()) if len(subset_frame) else 0.0,
            "topk": {},
            "dba": [],
            "total": [],
            "modalities": _first_text(subset_frame.get("modalities")),
        }
        for k in (1, 3, 5):
            values = []
            for horizon_idx in horizon_order:
                horizon_frame = subset_frame[subset_frame["horizon_idx"] == horizon_idx]
                values.append(float(horizon_frame[f"top{k}_hit"].mean()) if len(horizon_frame) else 0.0)
            metrics["topk"][str(k)] = values
        for horizon_idx in horizon_order:
            horizon_frame = subset_frame[subset_frame["horizon_idx"] == horizon_idx]
            metrics["dba"].append(float(horizon_frame["dba_score"].mean()) if len(horizon_frame) else 0.0)
            metrics["total"].append(int(len(horizon_frame)))
        results[str(subset_name)] = metrics
    return results


def compute_marginal_deltas(
    subset_predictions: list[dict[str, Any]] | pd.DataFrame,
    *,
    baseline_subset: str = "strong_only",
    weak_modalities: Iterable[str] = WEAK_MODALITIES,
) -> pd.DataFrame:
    frame = _to_dataframe(subset_predictions)
    if frame.empty:
        return pd.DataFrame()
    keys = ["sample_id", "dataset_index", "horizon_idx", "horizon_name"]
    base = frame[(frame["subset_name"] == baseline_subset) & _valid_mask(frame)].copy()
    if base.empty:
        return pd.DataFrame()
    base_cols = keys + ["gt_beam", "ce", "top1_hit", "top3_hit", "dba_score"]
    base = base[base_cols].rename(
        columns={
            "ce": "ce_strong_only",
            "top1_hit": "strong_only_top1",
            "top3_hit": "strong_only_top3",
            "dba_score": "strong_only_dba",
        }
    )
    rows = []
    for weak in weak_modalities:
        subset_name = f"strong_plus_{weak}"
        plus = frame[(frame["subset_name"] == subset_name) & _valid_mask(frame)].copy()
        if plus.empty:
            continue
        plus = plus[base_cols].rename(
            columns={
                "ce": "ce_strong_plus",
                "top1_hit": "strong_plus_top1",
                "top3_hit": "strong_plus_top3",
                "dba_score": "strong_plus_dba",
            }
        )
        merged = base.merge(plus, on=keys + ["gt_beam"], how="inner")
        if merged.empty:
            continue
        merged["weak_modality"] = weak
        merged["strong_plus_subset"] = subset_name
        merged["delta_ce"] = merged["ce_strong_only"] - merged["ce_strong_plus"]
        merged["delta_top1"] = merged["strong_plus_top1"] - merged["strong_only_top1"]
        merged["delta_top3"] = merged["strong_plus_top3"] - merged["strong_only_top3"]
        merged["delta_dba"] = merged["strong_plus_dba"] - merged["strong_only_dba"]
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    ordered = [
        "sample_id",
        "dataset_index",
        "horizon_idx",
        "horizon_name",
        "weak_modality",
        "strong_plus_subset",
        "gt_beam",
        "ce_strong_only",
        "ce_strong_plus",
        "delta_ce",
        "strong_only_top1",
        "strong_plus_top1",
        "delta_top1",
        "strong_only_top3",
        "strong_plus_top3",
        "delta_top3",
        "strong_only_dba",
        "strong_plus_dba",
        "delta_dba",
    ]
    return pd.concat(rows, ignore_index=True)[ordered]


def compute_subset_oracle(
    subset_predictions: list[dict[str, Any]] | pd.DataFrame,
    *,
    candidates: Iterable[str] = DEFAULT_ORACLE_CANDIDATES,
    baseline_subset: str = "strong_only",
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = _to_dataframe(subset_predictions)
    if frame.empty:
        return _empty_oracle_summary(), pd.DataFrame()
    candidate_list = list(candidates)
    candidate_rank = {name: idx for idx, name in enumerate(candidate_list)}
    work = frame[frame["subset_name"].isin(candidate_list) & _valid_mask(frame)].copy()
    if work.empty:
        return _empty_oracle_summary(), pd.DataFrame()
    work["_candidate_rank"] = work["subset_name"].map(candidate_rank).fillna(len(candidate_rank)).astype(int)
    sort_cols = ["sample_id", "dataset_index", "horizon_idx", "ce", "_candidate_rank"]
    best = (
        work.sort_values(sort_cols)
        .groupby(["sample_id", "dataset_index", "horizon_idx", "horizon_name"], as_index=False)
        .first()
        .rename(columns={"subset_name": "oracle_subset"})
    )
    oracle_rows = best[
        [
            "sample_id",
            "dataset_index",
            "horizon_idx",
            "horizon_name",
            "oracle_subset",
            "ce",
            "top1_hit",
            "top3_hit",
            "top5_hit",
            "dba_score",
        ]
    ].copy()
    summary = {
        "metrics": _metrics_from_prediction_frame(oracle_rows),
        "oracle_choice_distribution": _value_distribution(oracle_rows["oracle_subset"]),
        "oracle_choice_distribution_by_horizon": _choice_distribution_by_horizon(oracle_rows),
        "oracle_gain_vs_strong_only": _oracle_gain_vs_baseline(oracle_rows, frame, baseline_subset),
    }
    return _json_ready(summary), oracle_rows


def compute_teacher_complementarity(
    teacher_predictions: list[dict[str, Any]] | pd.DataFrame,
    subset_predictions: list[dict[str, Any]] | pd.DataFrame,
    *,
    weak_modalities: Iterable[str] = WEAK_MODALITIES,
    baseline_subset: str = "strong_only",
) -> tuple[dict[str, Any], pd.DataFrame]:
    teacher = _to_dataframe(teacher_predictions)
    subset = _to_dataframe(subset_predictions)
    if teacher.empty or subset.empty:
        return {}, pd.DataFrame()
    if "teacher_modality" not in teacher.columns:
        teacher = teacher.copy()
        teacher["teacher_modality"] = teacher["subset_name"]
    keys = ["sample_id", "dataset_index", "horizon_idx", "horizon_name"]
    strong = subset[(subset["subset_name"] == baseline_subset) & _valid_mask(subset)]
    strong = strong[keys + ["top1_hit", "gt_prob", "ce"]].rename(
        columns={
            "top1_hit": "strong_only_top1",
            "gt_prob": "strong_only_gt_prob",
            "ce": "strong_only_ce",
        }
    )
    summary: dict[str, Any] = {}
    rescue_rows = []
    for weak in weak_modalities:
        teacher_modality = teacher[(teacher["teacher_modality"] == weak) & _valid_mask(teacher)].copy()
        if teacher_modality.empty:
            continue
        teacher_modality = teacher_modality[keys + ["top1_hit", "gt_prob", "ce"]].rename(
            columns={
                "top1_hit": "teacher_top1",
                "gt_prob": "teacher_gt_prob",
                "ce": "teacher_ce",
            }
        )
        merged = strong.merge(teacher_modality, on=keys, how="inner")
        if merged.empty:
            continue
        merged["weak_modality"] = weak
        merged["teacher_rescue_top1"] = (
            (merged["strong_only_top1"].astype(int) == 0) & (merged["teacher_top1"].astype(int) == 1)
        ).astype(int)
        merged["teacher_gt_prob_advantage"] = (
            merged["teacher_gt_prob"] > merged["strong_only_gt_prob"]
        ).astype(int)
        merged["teacher_ce_better_than_strong"] = (merged["teacher_ce"] < merged["strong_only_ce"]).astype(int)
        strong_wrong = merged[merged["strong_only_top1"].astype(int) == 0]
        summary[weak] = {
            "rescue_rate_given_strong_top1_wrong": _safe_mean(strong_wrong["teacher_rescue_top1"]),
            "teacher_gt_prob_advantage_rate": _safe_mean(merged["teacher_gt_prob_advantage"]),
            "teacher_ce_better_than_strong_rate": _safe_mean(merged["teacher_ce_better_than_strong"]),
            "num_samples": int(len(merged)),
            "num_strong_top1_wrong": int(len(strong_wrong)),
            "by_horizon": _teacher_summary_by_horizon(merged),
        }
        rescue_rows.append(
            merged[
                keys
                + [
                    "weak_modality",
                    "teacher_rescue_top1",
                    "teacher_gt_prob_advantage",
                    "teacher_ce_better_than_strong",
                ]
            ]
        )
    rescue_frame = pd.concat(rescue_rows, ignore_index=True) if rescue_rows else pd.DataFrame()
    return _json_ready(summary), rescue_frame


def compute_bucket_summary(
    deltas: list[dict[str, Any]] | pd.DataFrame,
    bucketed_features: list[dict[str, Any]] | pd.DataFrame,
    *,
    oracle_choices: pd.DataFrame | None = None,
    teacher_rescue: pd.DataFrame | None = None,
    min_samples: int = 1,
) -> pd.DataFrame:
    delta_frame = _to_dataframe(deltas)
    feature_frame = _to_dataframe(bucketed_features)
    if delta_frame.empty or feature_frame.empty:
        return pd.DataFrame()
    bucket_columns = [col for col in feature_frame.columns if col.endswith("_bucket")]
    if not bucket_columns:
        return pd.DataFrame()
    keys = ["sample_id", "dataset_index", "horizon_idx", "horizon_name"]
    feature_columns = keys + bucket_columns
    merged = delta_frame.merge(feature_frame[feature_columns].drop_duplicates(), on=keys, how="left")
    if oracle_choices is not None and not oracle_choices.empty:
        merged = merged.merge(
            oracle_choices[keys + ["oracle_subset"]],
            on=keys,
            how="left",
        )
        merged["oracle_chose_strong_plus"] = (
            merged["oracle_subset"] == merged["strong_plus_subset"]
        ).astype(float)
    else:
        merged["oracle_chose_strong_plus"] = np.nan
    if teacher_rescue is not None and not teacher_rescue.empty:
        merged = merged.merge(
            teacher_rescue[keys + ["weak_modality", "teacher_rescue_top1"]],
            on=keys + ["weak_modality"],
            how="left",
        )
    else:
        merged["teacher_rescue_top1"] = np.nan

    rows: list[dict[str, Any]] = []
    for bucket_col in bucket_columns:
        bucket_feature = bucket_col[: -len("_bucket")]
        valid = merged[merged[bucket_col].notna()]
        if valid.empty:
            continue
        for (bucket_name, weak, horizon_name), group in valid.groupby(
            [bucket_col, "weak_modality", "horizon_name"], sort=True
        ):
            num_samples = int(len(group))
            if num_samples < int(min_samples):
                continue
            rows.append(
                {
                    "bucket_feature": bucket_feature,
                    "bucket_name": str(bucket_name),
                    "weak_modality": str(weak),
                    "horizon_name": str(horizon_name),
                    "num_samples": num_samples,
                    "strong_only_top1": _safe_mean(group["strong_only_top1"]),
                    "strong_plus_top1": _safe_mean(group["strong_plus_top1"]),
                    "delta_top1": _safe_mean(group["delta_top1"]),
                    "strong_only_top3": _safe_mean(group["strong_only_top3"]),
                    "strong_plus_top3": _safe_mean(group["strong_plus_top3"]),
                    "delta_top3": _safe_mean(group["delta_top3"]),
                    "strong_only_dba": _safe_mean(group["strong_only_dba"]),
                    "strong_plus_dba": _safe_mean(group["strong_plus_dba"]),
                    "delta_dba": _safe_mean(group["delta_dba"]),
                    "mean_delta_ce": _safe_mean(group["delta_ce"]),
                    "positive_delta_ce_rate": _safe_mean((group["delta_ce"] > 0).astype(float)),
                    "oracle_choice_rate": _safe_mean(group["oracle_chose_strong_plus"]),
                    "teacher_rescue_rate": _safe_mean(group["teacher_rescue_top1"]),
                }
            )
    return pd.DataFrame(rows)


def build_conditional_utility_summary(
    *,
    run_name: str,
    scene: str | int | None,
    num_samples: int,
    horizons: list[str],
    aggregate_metrics: dict[str, Any],
    deltas: list[dict[str, Any]] | pd.DataFrame,
    oracle_summary: dict[str, Any],
    teacher_summary: dict[str, Any] | None,
    bucket_summary: list[dict[str, Any]] | pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    diagnosis_thresholds: dict[str, Any] | None = None,
    bootstrap_confidence: list[dict[str, Any]] | pd.DataFrame | None = None,
) -> dict[str, Any]:
    delta_frame = _to_dataframe(deltas)
    bucket_frame = _to_dataframe(bucket_summary)
    thresholds = {
        "global_delta_dba": 0.0,
        "global_delta_ce": 0.0,
        "conditional_delta_dba": 0.02,
        "conditional_delta_ce": 0.02,
        "teacher_rescue_rate": 0.10,
        "oracle_gain_dba": 0.02,
        "min_bucket_samples": 10,
    }
    thresholds.update(diagnosis_thresholds or {})
    marginal = _marginal_summary(delta_frame)
    by_horizon = _marginal_summary_by_horizon(delta_frame)
    summary = {
        "run_name": run_name,
        "scene": scene,
        "num_samples": int(num_samples),
        "horizons": list(horizons),
        "aggregate_metrics": aggregate_metrics,
        "marginal_utility_vs_strong_only": marginal,
        "marginal_utility_by_horizon": by_horizon,
        "oracle_subset": oracle_summary,
        "teacher_complementarity": teacher_summary or {},
        "bucket_highlights": _bucket_highlights(bucket_frame, thresholds),
        "metadata": metadata or {},
        "thresholds": thresholds,
        "diagnosis": diagnose_modalities(
            marginal,
            bucket_frame,
            teacher_summary or {},
            thresholds=thresholds,
            bootstrap_confidence=bootstrap_confidence,
        ),
    }
    return _json_ready(summary)


def diagnose_modalities(
    marginal_summary: dict[str, Any],
    bucket_summary: pd.DataFrame,
    teacher_summary: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    bootstrap_confidence: list[dict[str, Any]] | pd.DataFrame | None = None,
) -> dict[str, Any]:
    diagnosis: dict[str, Any] = {}
    ci_frame = _to_dataframe(bootstrap_confidence) if bootstrap_confidence is not None else pd.DataFrame()
    for weak in WEAK_MODALITIES:
        overall = marginal_summary.get(weak, {})
        mean_delta_dba = float(overall.get("delta_dba", 0.0) or 0.0)
        mean_delta_ce = float(overall.get("delta_ce", 0.0) or 0.0)
        teacher = teacher_summary.get(weak, {}) if isinstance(teacher_summary, dict) else {}
        rescue_rate = float(teacher.get("rescue_rate_given_strong_top1_wrong", 0.0) or 0.0)
        trigger = {
            "overall_delta_dba": mean_delta_dba,
            "overall_delta_ce": mean_delta_ce,
            "teacher_rescue_rate": rescue_rate,
        }
        global_candidate = _global_useful_candidate(weak, mean_delta_dba, mean_delta_ce, thresholds, ci_frame)
        if global_candidate["passes"]:
            diagnosis[weak] = {"label": "globally_useful", "evidence": {**trigger, **global_candidate}}
            continue
        if global_candidate["blocked_by_ci"]:
            diagnosis[weak] = {"label": "not_significant", "evidence": {**trigger, **global_candidate}}
            continue
        if rescue_rate >= float(thresholds.get("teacher_rescue_rate", 0.10)):
            diagnosis[weak] = {
                "label": "representation_exists_but_not_exploited",
                "evidence": trigger,
            }
            continue
        bucket_trigger = _conditional_bucket_trigger(bucket_summary, weak, thresholds, bootstrap_confidence=ci_frame)
        if bucket_trigger is not None:
            diagnosis[weak] = {
                "label": "conditionally_useful",
                "evidence": {**trigger, "bucket": bucket_trigger},
            }
            continue
        diagnosis[weak] = {"label": "currently_low_utility", "evidence": trigger}
    return diagnosis


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(data), f, indent=2)
    return target


def _dba_contribution(preds: list[int], gt: int, delta: float) -> float:
    if not preds:
        return 0.0
    norm_dists = [min(abs(int(pred) - int(gt)) / float(delta), 1.0) for pred in preds]
    return float(1.0 - min(norm_dists))


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


def _to_dataframe(records_or_frame: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records_or_frame, pd.DataFrame):
        return records_or_frame.copy()
    return pd.DataFrame(records_or_frame)


def _valid_mask(frame: pd.DataFrame) -> pd.Series:
    if "valid" not in frame.columns:
        return pd.Series(True, index=frame.index)
    return frame["valid"].astype(bool)


def _horizon_order(frame: pd.DataFrame) -> list[int]:
    return sorted(int(value) for value in frame["horizon_idx"].dropna().unique().tolist())


def _first_text(series: pd.Series | None) -> list[str]:
    if series is None or len(series) == 0:
        return []
    value = next((item for item in series.tolist() if isinstance(item, str) and item), "")
    return [part for part in str(value).split(",") if part]


def _safe_mean(values: Any) -> float:
    series = pd.Series(values).dropna()
    if series.empty:
        return 0.0
    return float(series.astype(float).mean())


def _metrics_from_prediction_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"ce": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0, "dba": 0.0, "by_horizon": {}}
    metrics = {
        "ce": _safe_mean(frame["ce"]),
        "top1": _safe_mean(frame["top1_hit"]),
        "top3": _safe_mean(frame["top3_hit"]),
        "top5": _safe_mean(frame["top5_hit"]),
        "dba": _safe_mean(frame["dba_score"]),
        "by_horizon": {},
    }
    for horizon_name, group in frame.groupby("horizon_name", sort=True):
        metrics["by_horizon"][str(horizon_name)] = {
            "ce": _safe_mean(group["ce"]),
            "top1": _safe_mean(group["top1_hit"]),
            "top3": _safe_mean(group["top3_hit"]),
            "top5": _safe_mean(group["top5_hit"]),
            "dba": _safe_mean(group["dba_score"]),
            "num_samples": int(len(group)),
        }
    return metrics


def _value_distribution(values: pd.Series) -> dict[str, float]:
    counts = values.value_counts(dropna=True)
    total = float(counts.sum())
    if total <= 0:
        return {}
    return {str(key): float(value / total) for key, value in counts.items()}


def _choice_distribution_by_horizon(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(horizon): _value_distribution(group["oracle_subset"])
        for horizon, group in frame.groupby("horizon_name", sort=True)
    }


def _oracle_gain_vs_baseline(oracle_rows: pd.DataFrame, predictions: pd.DataFrame, baseline_subset: str) -> dict[str, Any]:
    keys = ["sample_id", "dataset_index", "horizon_idx", "horizon_name"]
    baseline = predictions[predictions["subset_name"] == baseline_subset][
        keys + ["ce", "top1_hit", "top3_hit", "dba_score"]
    ].rename(
        columns={
            "ce": "baseline_ce",
            "top1_hit": "baseline_top1",
            "top3_hit": "baseline_top3",
            "dba_score": "baseline_dba",
        }
    )
    merged = oracle_rows.merge(baseline, on=keys, how="inner")
    if merged.empty:
        return {}
    merged["delta_ce"] = merged["baseline_ce"] - merged["ce"]
    merged["delta_top1"] = merged["top1_hit"] - merged["baseline_top1"]
    merged["delta_top3"] = merged["top3_hit"] - merged["baseline_top3"]
    merged["delta_dba"] = merged["dba_score"] - merged["baseline_dba"]
    result = {
        "delta_ce": _safe_mean(merged["delta_ce"]),
        "delta_top1": _safe_mean(merged["delta_top1"]),
        "delta_top3": _safe_mean(merged["delta_top3"]),
        "delta_dba": _safe_mean(merged["delta_dba"]),
        "by_horizon": {},
    }
    for horizon_name, group in merged.groupby("horizon_name", sort=True):
        result["by_horizon"][str(horizon_name)] = {
            "delta_ce": _safe_mean(group["delta_ce"]),
            "delta_top1": _safe_mean(group["delta_top1"]),
            "delta_top3": _safe_mean(group["delta_top3"]),
            "delta_dba": _safe_mean(group["delta_dba"]),
        }
    return result


def _empty_oracle_summary() -> dict[str, Any]:
    return {
        "metrics": {},
        "oracle_choice_distribution": {},
        "oracle_choice_distribution_by_horizon": {},
        "oracle_gain_vs_strong_only": {},
    }


def _teacher_summary_by_horizon(frame: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for horizon_name, group in frame.groupby("horizon_name", sort=True):
        strong_wrong = group[group["strong_only_top1"].astype(int) == 0]
        result[str(horizon_name)] = {
            "rescue_rate_given_strong_top1_wrong": _safe_mean(strong_wrong["teacher_rescue_top1"]),
            "teacher_gt_prob_advantage_rate": _safe_mean(group["teacher_gt_prob_advantage"]),
            "teacher_ce_better_than_strong_rate": _safe_mean(group["teacher_ce_better_than_strong"]),
            "num_samples": int(len(group)),
            "num_strong_top1_wrong": int(len(strong_wrong)),
        }
    return result


def _marginal_summary(delta_frame: pd.DataFrame) -> dict[str, Any]:
    if delta_frame.empty:
        return {}
    summary = {}
    for weak, group in delta_frame.groupby("weak_modality", sort=True):
        summary[str(weak)] = {
            "delta_ce": _safe_mean(group["delta_ce"]),
            "delta_top1": _safe_mean(group["delta_top1"]),
            "delta_top3": _safe_mean(group["delta_top3"]),
            "delta_dba": _safe_mean(group["delta_dba"]),
            "positive_delta_ce_rate": _safe_mean((group["delta_ce"] > 0).astype(float)),
            "num_samples": int(len(group)),
        }
    return summary


def _marginal_summary_by_horizon(delta_frame: pd.DataFrame) -> dict[str, Any]:
    if delta_frame.empty:
        return {}
    summary: dict[str, Any] = {}
    for (weak, horizon_name), group in delta_frame.groupby(["weak_modality", "horizon_name"], sort=True):
        summary.setdefault(str(weak), {})[str(horizon_name)] = {
            "delta_ce": _safe_mean(group["delta_ce"]),
            "delta_top1": _safe_mean(group["delta_top1"]),
            "delta_top3": _safe_mean(group["delta_top3"]),
            "delta_dba": _safe_mean(group["delta_dba"]),
            "positive_delta_ce_rate": _safe_mean((group["delta_ce"] > 0).astype(float)),
            "num_samples": int(len(group)),
        }
    return summary


def _bucket_highlights(bucket_frame: pd.DataFrame, thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    if bucket_frame.empty:
        return []
    min_samples = int(thresholds.get("min_bucket_samples", 10))
    valid = bucket_frame[bucket_frame["num_samples"] >= min_samples].copy()
    if valid.empty:
        return []
    valid["_score"] = valid[["delta_dba", "mean_delta_ce"]].max(axis=1)
    valid = valid.sort_values("_score", ascending=False).head(12).drop(columns=["_score"])
    return [_json_ready(row) for row in valid.to_dict(orient="records")]


def _global_useful_candidate(
    weak: str,
    mean_delta_dba: float,
    mean_delta_ce: float,
    thresholds: dict[str, Any],
    bootstrap_confidence: pd.DataFrame,
) -> dict[str, Any]:
    candidates = [
        ("delta_dba", mean_delta_dba, float(thresholds.get("global_delta_dba", 0.0))),
        ("delta_ce", mean_delta_ce, float(thresholds.get("global_delta_ce", 0.0))),
    ]
    for metric, value, threshold in candidates:
        if value < threshold or (threshold == 0.0 and value <= 0.0):
            continue
        ci = _bootstrap_ci_for(bootstrap_confidence, weak, metric, "overall")
        if ci is None:
            return {
                "passes": True,
                "blocked_by_ci": False,
                "metric": metric,
                "threshold": threshold,
                "bootstrap_ci": None,
            }
        ci_lower = float(ci.get("ci_lower", 0.0) or 0.0)
        evidence = {
            "passes": ci_lower > 0.0,
            "blocked_by_ci": ci_lower <= 0.0,
            "metric": metric,
            "threshold": threshold,
            "bootstrap_ci": ci,
        }
        return evidence
    return {
        "passes": False,
        "blocked_by_ci": False,
        "metric": None,
        "threshold": None,
        "bootstrap_ci": None,
    }


def _conditional_bucket_trigger(
    bucket_frame: pd.DataFrame,
    weak: str,
    thresholds: dict[str, Any],
    *,
    bootstrap_confidence: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    if bucket_frame.empty:
        return None
    min_samples = int(thresholds.get("min_bucket_samples", 10))
    delta_dba_threshold = float(thresholds.get("conditional_delta_dba", 0.02))
    delta_ce_threshold = float(thresholds.get("conditional_delta_ce", 0.02))
    candidates = bucket_frame[
        (bucket_frame["weak_modality"] == weak)
        & (bucket_frame["num_samples"] >= min_samples)
        & (
            (bucket_frame["delta_dba"] >= delta_dba_threshold)
            | (bucket_frame["mean_delta_ce"] >= delta_ce_threshold)
        )
    ].copy()
    if candidates.empty:
        return None
    if bootstrap_confidence is not None and not bootstrap_confidence.empty:
        keep_rows = []
        ci_rows = []
        for _, row in candidates.iterrows():
            metric = "delta_dba" if float(row.get("delta_dba", 0.0) or 0.0) >= delta_dba_threshold else "delta_ce"
            ci = _bootstrap_ci_for(bootstrap_confidence, weak, metric, str(row.get("horizon_name", "overall")))
            if ci is not None and float(ci.get("ci_lower", 0.0) or 0.0) <= 0.0:
                continue
            keep_rows.append(row)
            ci_rows.append(ci)
        if not keep_rows:
            return None
        candidates = pd.DataFrame(keep_rows)
        candidates["_bootstrap_ci"] = ci_rows
    row = candidates.sort_values(["delta_dba", "mean_delta_ce"], ascending=False).iloc[0].to_dict()
    return _json_ready(row)


def _bootstrap_ci_for(
    frame: pd.DataFrame,
    weak: str,
    metric: str,
    horizon_name: str,
) -> dict[str, Any] | None:
    if frame.empty:
        return None
    required = {"weak_modality", "metric", "horizon_name", "ci_lower", "ci_upper", "mean_delta"}
    if not required.issubset(frame.columns):
        return None
    rows = frame[
        (frame["weak_modality"].astype(str) == str(weak))
        & (frame["metric"].astype(str) == str(metric))
        & (frame["horizon_name"].astype(str) == str(horizon_name))
    ]
    if rows.empty and str(horizon_name) != "overall":
        rows = frame[
            (frame["weak_modality"].astype(str) == str(weak))
            & (frame["metric"].astype(str) == str(metric))
            & (frame["horizon_name"].astype(str) == "overall")
        ]
    if rows.empty:
        return None
    return _json_ready(rows.iloc[0].to_dict())


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Series):
        return _json_ready(value.to_dict())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


__all__ = [
    "DEFAULT_ORACLE_CANDIDATES",
    "WEAK_MODALITIES",
    "aggregate_subset_metrics",
    "build_conditional_utility_summary",
    "compute_bucket_summary",
    "compute_marginal_deltas",
    "compute_subset_oracle",
    "compute_teacher_complementarity",
    "diagnose_modalities",
    "read_table",
    "records_from_logits",
    "subset_prediction_records",
    "write_json",
    "write_table",
]
