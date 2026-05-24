from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from kd_sensing.diagnostics.complementarity_constants import (
    CASE_ALL_CORRECT,
    CASE_ALL_WRONG,
    CASE_NEGATIVE_TRANSFER,
    CASE_OTHER,
    CASE_RESCUE,
    CASE_STRONG_WRONG_FUSION_CORRECT,
    CASE_UNUSED_COMPLEMENTARY,
    COMMUNICATION_BUCKET_FEATURES,
    DEFAULT_CASE_FILTERS,
    DELTA_COLUMNS,
    KEY_COLUMNS,
    METADATA_COLUMNS,
    PATH_COLUMNS,
    WEAK_MODALITIES,
)
from kd_sensing.diagnostics.complementarity_schema import (
    _clean_name,
    canonical_subset_name,
    normalize_schema,
)

def build_case_table(
    subset_predictions: pd.DataFrame,
    *,
    teacher_predictions: pd.DataFrame | None = None,
    per_sample_delta: pd.DataFrame | None = None,
    communication_state_features: pd.DataFrame | None = None,
    strong_subset: str = "strong_only",
    strong_modalities: Iterable[str] | None = None,
    weak_modalities: Iterable[str] = WEAK_MODALITIES,
    fusion_subsets: dict[str, str] | None = None,
    pair_fusion_subsets: dict[str, str] | None = None,
    horizons: Iterable[str | int] | None = None,
    scene: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset, subset_meta = normalize_schema(subset_predictions, table_name="subset_predictions")
    teacher, teacher_meta = normalize_schema(
        teacher_predictions if teacher_predictions is not None else pd.DataFrame(),
        table_name="teacher_predictions",
    )
    subset = subset[subset["valid"].astype(bool)].copy()
    teacher = teacher[teacher["valid"].astype(bool)].copy() if not teacher.empty else teacher
    subset = _filter_horizons(subset, horizons)
    teacher = _filter_horizons(teacher, horizons)

    requested_weaks = _canonical_modalities(weak_modalities)
    if strong_modalities is not None:
        cases, pair_metadata = _build_pair_mode_cases(
            subset,
            teacher,
            strong_modalities=strong_modalities,
            weak_modalities=requested_weaks,
            pair_fusion_subsets=pair_fusion_subsets,
            per_sample_delta=per_sample_delta,
            communication_state_features=communication_state_features,
            scene=scene,
        )
        probability_available = _probability_available(cases)
        metadata = _metadata_payload(
            subset_meta=subset_meta,
            teacher_meta=teacher_meta if teacher_predictions is not None and not teacher.empty else {},
            subset_predictions=subset_predictions,
            teacher_predictions=teacher_predictions,
            per_sample_delta=per_sample_delta,
            communication_state_features=communication_state_features,
            probability_available=probability_available,
            cases=cases,
            extra=pair_metadata,
        )
        return cases, metadata

    strong_key = canonical_subset_name(strong_subset)
    strong_rows = subset[subset["subset_key"] == strong_key].copy()
    if strong_rows.empty:
        available = sorted(subset["subset_name"].dropna().astype(str).unique().tolist())
        raise ValueError(f"Strong subset '{strong_subset}' resolved to '{strong_key}' but was not found. Available: {available}")

    rows: list[pd.DataFrame] = []
    weak_sources: dict[str, dict[str, Any]] = {}
    unmatched: dict[str, Any] = {}
    warnings: list[str] = []
    fusion_mapping: dict[str, str] = {}

    for weak in requested_weaks:
        fusion_key = canonical_subset_name((fusion_subsets or {}).get(weak, f"strong_plus_{weak}"))
        fusion_mapping[weak] = fusion_key
        fusion_rows = subset[subset["subset_key"] == fusion_key].copy()
        if fusion_rows.empty:
            warnings.append(f"Missing fusion subset for weak modality '{weak}' (resolved '{fusion_key}').")
            unmatched[weak] = {"fusion_rows": 0, "case_rows": 0}
            continue

        weak_rows, weak_source = _select_weak_predictions(subset, teacher, weak)
        weak_sources[weak] = weak_source
        if weak_rows.empty:
            warnings.append(f"Missing weak prediction source for '{weak}'; weak correctness metrics are disabled.")

        strong_view = _prediction_view(strong_rows, "strong", include_metadata=True)
        fusion_view = _prediction_view(fusion_rows, "fusion", include_metadata=False)
        merged = strong_view.merge(fusion_view, on=KEY_COLUMNS, how="inner")
        if not weak_rows.empty:
            weak_view = _prediction_view(weak_rows, "weak", include_metadata=False)
            merged = merged.merge(weak_view, on=KEY_COLUMNS, how="left")
        else:
            merged = _add_missing_prediction_columns(merged, "weak")
        if merged.empty:
            unmatched[weak] = {
                "strong_rows": int(len(strong_rows)),
                "fusion_rows": int(len(fusion_rows)),
                "weak_rows": int(len(weak_rows)),
                "case_rows": 0,
            }
            continue
        merged["weak_modality"] = weak
        merged["strong_subset"] = strong_key
        merged["fusion_subset"] = fusion_key
        merged["weak_prediction_source"] = weak_source["source"]
        merged["weak_prediction_available"] = bool(not weak_rows.empty)
        rows.append(merged)
        unmatched[weak] = {
            "strong_rows": int(len(strong_rows)),
            "fusion_rows": int(len(fusion_rows)),
            "weak_rows": int(len(weak_rows)),
            "case_rows": int(len(merged)),
            "strong_fusion_unmatched": int(max(0, len(strong_rows) - len(merged))),
            "weak_missing_after_join": int(merged["weak_pred"].isna().sum()),
        }

    if rows:
        cases = pd.concat(rows, ignore_index=True)
    else:
        cases = pd.DataFrame(columns=_case_columns())
    if not cases.empty:
        cases = _finalize_cases(cases, scene=scene)
        cases = _merge_deltas(cases, per_sample_delta)
        cases = _merge_bucket_columns(cases, communication_state_features)

    probability_available = _probability_available(cases)
    metadata = _metadata_payload(
        subset_meta=subset_meta,
        teacher_meta=teacher_meta if teacher_predictions is not None and not teacher.empty else {},
        subset_predictions=subset_predictions,
        teacher_predictions=teacher_predictions,
        per_sample_delta=per_sample_delta,
        communication_state_features=communication_state_features,
        probability_available=probability_available,
        cases=cases,
        extra={
            "analysis_mode": "strong_subset",
            "strong_subset": strong_key,
            "weak_modalities": requested_weaks,
            "fusion_subset_mapping": fusion_mapping,
            "weak_prediction_sources": weak_sources,
            "unmatched": unmatched,
            "warnings": warnings,
        },
    )
    return cases, metadata



def _filter_horizons(frame: pd.DataFrame, horizons: Iterable[str | int] | None) -> pd.DataFrame:
    if frame.empty or horizons is None:
        return frame
    requested = [str(item).strip() for item in horizons if str(item).strip()]
    if not requested:
        return frame
    names = set(requested)
    indices: set[int] = set()
    for item in requested:
        lowered = item.lower()
        if lowered.startswith("t+"):
            try:
                indices.add(int(lowered[2:]) - 1)
            except ValueError:
                pass
        else:
            try:
                indices.add(int(lowered))
            except ValueError:
                pass
    mask = frame["horizon_name"].astype(str).isin(names) | frame["horizon_idx"].astype(int).isin(indices)
    return frame[mask].copy()


def _canonical_modalities(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    result: list[str] = []
    for value in values:
        key = canonical_subset_name(value)
        if key and key not in result:
            result.append(key)
    return result


def _metadata_payload(
    *,
    subset_meta: dict[str, Any],
    teacher_meta: dict[str, Any],
    subset_predictions: pd.DataFrame,
    teacher_predictions: pd.DataFrame | None,
    per_sample_delta: pd.DataFrame | None,
    communication_state_features: pd.DataFrame | None,
    probability_available: bool,
    cases: pd.DataFrame,
    extra: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "schema": {
            "subset_predictions": subset_meta,
            "teacher_predictions": teacher_meta,
        },
        "probability_metrics_available": probability_available,
        "probability_metrics_unavailable_reason": None if probability_available else _probability_unavailable_reason(cases),
        "input_row_counts": {
            "subset_predictions": int(len(subset_predictions)),
            "teacher_predictions": int(len(teacher_predictions)) if teacher_predictions is not None else 0,
            "per_sample_delta": int(len(per_sample_delta)) if per_sample_delta is not None else 0,
            "communication_state_features": int(len(communication_state_features)) if communication_state_features is not None else 0,
        },
        "output_rows": int(len(cases)),
    }
    metadata.update(extra)
    return metadata


def _build_pair_mode_cases(
    subset: pd.DataFrame,
    teacher: pd.DataFrame,
    *,
    strong_modalities: Iterable[str],
    weak_modalities: Iterable[str],
    pair_fusion_subsets: dict[str, str] | None,
    per_sample_delta: pd.DataFrame | None,
    communication_state_features: pd.DataFrame | None,
    scene: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    requested_strongs = _canonical_modalities(strong_modalities)
    requested_weaks = _canonical_modalities(weak_modalities)
    pair_fusion_mapping = _normalize_pair_mapping(pair_fusion_subsets)
    strong_sources: dict[str, dict[str, Any]] = {}
    weak_sources: dict[str, dict[str, Any]] = {}
    fusion_availability: dict[str, dict[str, Any]] = {}
    unmatched: dict[str, Any] = {}
    warnings: list[str] = []
    warned_missing_weaks: set[str] = set()

    for strong in requested_strongs:
        strong_rows, strong_source = _select_modality_predictions(subset, teacher, strong, role="strong")
        strong_sources[strong] = strong_source
        if strong_rows.empty:
            warnings.append(f"Missing strong prediction source for '{strong}'; skipping this strong modality.")
            unmatched[strong] = {"strong_rows": 0, "case_rows": 0, "reason": strong_source.get("reason")}
            continue
        strong_view = _prediction_view(strong_rows, "strong", include_metadata=True)
        for weak in requested_weaks:
            pair = _pair_key(strong, weak)
            weak_rows, weak_source = _select_weak_predictions(subset, teacher, weak)
            weak_sources.setdefault(weak, weak_source)
            if weak_rows.empty and weak not in warned_missing_weaks:
                warnings.append(f"Missing weak prediction source for '{weak}'; weak correctness metrics are disabled.")
                warned_missing_weaks.add(weak)

            fusion_rows, fusion_source = _select_pair_fusion_predictions(
                subset,
                strong=strong,
                weak=weak,
                pair_fusion_subsets=pair_fusion_mapping,
            )
            fusion_available = not fusion_rows.empty
            fusion_availability[pair] = fusion_source

            merged = strong_view.copy()
            if not weak_rows.empty:
                weak_view = _prediction_view(weak_rows, "weak", include_metadata=False)
                merged = merged.merge(weak_view, on=KEY_COLUMNS, how="left")
            else:
                merged = _add_missing_prediction_columns(merged, "weak")
            if fusion_available:
                fusion_view = _prediction_view(fusion_rows, "fusion", include_metadata=False)
                merged = merged.merge(fusion_view, on=KEY_COLUMNS, how="left")
            else:
                merged = _add_missing_prediction_columns(merged, "fusion")
                warnings.append(f"Missing fusion subset for strong/weak pair '{pair}'.")

            if merged.empty:
                unmatched[pair] = {
                    "strong_rows": int(len(strong_rows)),
                    "weak_rows": int(len(weak_rows)),
                    "fusion_rows": int(len(fusion_rows)),
                    "case_rows": 0,
                }
                continue
            row_fusion_available = bool(fusion_available) & merged["fusion_pred"].notna()
            merged["strong_modality"] = strong
            merged["weak_modality"] = weak
            merged["strong_weak_pair"] = pair
            merged["strong_prediction_source"] = strong_source["source"]
            merged["weak_prediction_source"] = weak_source["source"]
            merged["weak_prediction_available"] = bool(not weak_rows.empty)
            merged["fusion_prediction_available"] = row_fusion_available
            merged["strong_subset"] = strong_source.get("resolved_subset_name")
            merged["fusion_subset"] = fusion_source.get("resolved_subset_name")
            merged["strong_plus_weak_subset"] = fusion_source.get("resolved_subset_name")
            rows.append(merged)
            unmatched[pair] = {
                "strong_rows": int(len(strong_rows)),
                "weak_rows": int(len(weak_rows)),
                "fusion_rows": int(len(fusion_rows)),
                "case_rows": int(len(merged)),
                "weak_missing_after_join": int(merged["weak_pred"].isna().sum()),
                "fusion_missing_after_join": int(merged["fusion_pred"].isna().sum()),
            }

    if rows:
        cases = pd.concat(rows, ignore_index=True)
    else:
        cases = pd.DataFrame(columns=_case_columns())
    if not cases.empty:
        cases = _finalize_cases(cases, scene=scene)
        cases = _merge_deltas(cases, per_sample_delta)
        cases = _merge_bucket_columns(cases, communication_state_features)
    return cases, {
        "analysis_mode": "strong_modality_pair",
        "strong_modalities": requested_strongs,
        "strong_modalities_default_source": "cli_or_caller",
        "weak_modalities": requested_weaks,
        "strong_prediction_sources": strong_sources,
        "weak_prediction_sources": weak_sources,
        "pair_fusion_subset_mapping": pair_fusion_mapping,
        "fusion_subset_availability": fusion_availability,
        "unmatched": unmatched,
        "warnings": warnings,
    }


def _select_modality_predictions(
    subset: pd.DataFrame,
    teacher: pd.DataFrame,
    modality: str,
    *,
    role: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    modality_key = canonical_subset_name(modality)
    subset_rows = subset[subset["subset_key"] == modality_key].copy() if not subset.empty else pd.DataFrame()
    if not subset_rows.empty:
        names = sorted(subset_rows["subset_name"].dropna().astype(str).unique().tolist())
        return subset_rows, {
            "source": "subset_predictions",
            f"{role}_prediction_available": True,
            "reason": None,
            "rows": int(len(subset_rows)),
            "resolved_modality": modality_key,
            "resolved_subset_name": names[0] if names else modality_key,
            "resolved_subset_names": names,
        }

    teacher_rows = pd.DataFrame()
    if not teacher.empty:
        teacher_rows = teacher[
            (teacher["teacher_modality_key"] == modality_key) | (teacher["subset_key"] == modality_key)
        ].copy()
    if not teacher_rows.empty:
        names = sorted(teacher_rows["subset_name"].dropna().astype(str).unique().tolist())
        return teacher_rows, {
            "source": "teacher_predictions",
            f"{role}_prediction_available": True,
            "reason": None,
            "rows": int(len(teacher_rows)),
            "resolved_modality": modality_key,
            "resolved_subset_name": names[0] if names else modality_key,
            "resolved_subset_names": names,
        }

    return pd.DataFrame(), {
        "source": "missing",
        f"{role}_prediction_available": False,
        "reason": f"No single-modality subset or teacher prediction rows matched modality '{modality_key}'.",
        "rows": 0,
        "resolved_modality": modality_key,
        "resolved_subset_name": None,
        "resolved_subset_names": [],
    }


def _select_weak_predictions(
    subset: pd.DataFrame,
    teacher: pd.DataFrame,
    weak: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _select_modality_predictions(subset, teacher, weak, role="weak")


def _select_pair_fusion_predictions(
    subset: pd.DataFrame,
    *,
    strong: str,
    weak: str,
    pair_fusion_subsets: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pair = _pair_key(strong, weak)
    explicit = pair_fusion_subsets.get(pair)
    candidates = [explicit] if explicit else [
        f"{strong}+{weak}",
        f"{weak}+{strong}",
        f"{strong}_{weak}",
        f"{weak}_{strong}",
        f"{strong}_plus_{weak}",
        f"{weak}_plus_{strong}",
        f"strong_plus_{strong}_{weak}",
        f"strong_plus_{weak}_{strong}",
    ]
    candidate_keys = _candidate_subset_keys(candidates)
    if subset.empty or not candidate_keys:
        return pd.DataFrame(), _fusion_source_payload(pair, explicit, [], 0, False)

    subset_names = subset["subset_name"].map(_clean_name)
    rows = subset[subset["subset_key"].isin(candidate_keys) | subset_names.isin(candidate_keys)].copy()
    if rows.empty:
        return pd.DataFrame(), _fusion_source_payload(pair, explicit, sorted(candidate_keys), 0, False)
    names = sorted(rows["subset_name"].dropna().astype(str).unique().tolist())
    payload = _fusion_source_payload(pair, explicit, sorted(candidate_keys), int(len(rows)), True)
    payload["resolved_subset_name"] = names[0] if names else None
    payload["resolved_subset_names"] = names
    return rows, payload


def _fusion_source_payload(
    pair: str,
    explicit: str | None,
    candidates: list[str],
    rows: int,
    available: bool,
) -> dict[str, Any]:
    return {
        "source": "subset_predictions" if available else "missing",
        "fusion_prediction_available": bool(available),
        "pair": pair,
        "explicit_subset": explicit,
        "candidate_subset_keys": candidates,
        "rows": int(rows),
        "resolved_subset_name": None,
        "resolved_subset_names": [],
        "reason": None if available else "No pair fusion subset matched this strong/weak pair.",
    }


def _normalize_pair_mapping(mapping: dict[str, str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (mapping or {}).items():
        pair = _normalize_pair_key(key)
        if pair and str(value).strip():
            result[pair] = str(value).strip()
    return result


def _normalize_pair_key(value: Any) -> str:
    text = _clean_name(value)
    if not text:
        return ""
    normalized = text.replace("_plus_", "+").replace("_", "+")
    parts = [canonical_subset_name(part) for part in normalized.split("+") if part]
    if len(parts) != 2:
        return ""
    return _pair_key(parts[0], parts[1])


def _pair_key(strong: str, weak: str) -> str:
    return f"{canonical_subset_name(strong)}+{canonical_subset_name(weak)}"


def _candidate_subset_keys(values: Iterable[str | None]) -> set[str]:
    keys: set[str] = set()
    for value in values:
        if value is None:
            continue
        cleaned = _clean_name(value)
        canonical = canonical_subset_name(value)
        if cleaned:
            keys.add(cleaned)
        if canonical:
            keys.add(canonical)
    return keys


def _prediction_view(frame: pd.DataFrame, prefix: str, *, include_metadata: bool) -> pd.DataFrame:
    columns = [*KEY_COLUMNS, "y_true", "pred_top1", "pred_top2", "p_true", "top1_prob", "top2_prob", "margin", "subset_name", "_source_row_index"]
    if include_metadata:
        columns.extend([column for column in METADATA_COLUMNS if column in frame.columns])
    available = [column for column in columns if column in frame.columns]
    view = frame[available].drop_duplicates(KEY_COLUMNS).copy()
    rename = {
        "pred_top1": f"{prefix}_pred",
        "pred_top2": f"{prefix}_pred_top2",
        "p_true": f"p_true_{prefix}",
        "top1_prob": f"{prefix}_top1_prob",
        "top2_prob": f"{prefix}_top2_prob",
        "margin": f"{prefix}_margin",
        "subset_name": f"{prefix}_subset_name",
        "_source_row_index": f"{prefix}_source_row_index",
    }
    if prefix != "strong":
        rename["y_true"] = f"y_true_{prefix}"
    return view.rename(columns=rename)


def _add_missing_prediction_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    for column in [
        f"{prefix}_pred",
        f"{prefix}_pred_top2",
        f"p_true_{prefix}",
        f"{prefix}_top1_prob",
        f"{prefix}_top2_prob",
        f"{prefix}_margin",
        f"{prefix}_subset_name",
        f"{prefix}_source_row_index",
    ]:
        result[column] = np.nan
    return result


def _finalize_cases(cases: pd.DataFrame, *, scene: str | None) -> pd.DataFrame:
    result = cases.copy()
    if "scene" not in result.columns or result["scene"].replace("", np.nan).isna().all():
        result["scene"] = scene or result.get("scene_slug", "")
    if scene:
        result["scene"] = result["scene"].replace("", scene).fillna(scene)
    if "fusion_prediction_available" not in result.columns:
        result["fusion_prediction_available"] = True
    if "weak_prediction_available" not in result.columns:
        result["weak_prediction_available"] = True
    for prefix in ("strong", "weak", "fusion"):
        pred_col = f"{prefix}_pred"
        result[f"{prefix}_correct"] = _correctness(result.get(pred_col), result["y_true"])
    weak_available = _availability_series(result, "weak_prediction_available", default=True)
    fusion_available = _availability_series(result, "fusion_prediction_available", default=True)
    result.loc[~weak_available, "weak_correct"] = pd.NA
    result.loc[result["weak_pred"].isna(), "weak_correct"] = pd.NA
    result.loc[~fusion_available, "fusion_correct"] = pd.NA
    result["weak_gt_gain"] = result["p_true_weak"] - result["p_true_strong"]
    result["fusion_gt_gain"] = result["p_true_fusion"] - result["p_true_strong"]

    strong_correct = result["strong_correct"].fillna(False).astype(bool)
    weak_correct = result["weak_correct"].fillna(False).astype(bool)
    fusion_correct = result["fusion_correct"].fillna(False).astype(bool)
    weak_available = _availability_series(result, "weak_prediction_available", default=False)

    conditions = [
        fusion_available & (~strong_correct) & weak_correct & fusion_correct,
        fusion_available & (~strong_correct) & weak_correct & (~fusion_correct),
        fusion_available & strong_correct & (~fusion_correct),
        fusion_available & (~strong_correct) & fusion_correct,
        (~fusion_available) & (~strong_correct) & weak_correct,
        fusion_available & strong_correct & fusion_correct & (weak_correct | ~weak_available),
        fusion_available & (~strong_correct) & (~fusion_correct) & ((~weak_correct) | ~weak_available),
    ]
    choices = [
        CASE_RESCUE,
        CASE_UNUSED_COMPLEMENTARY,
        CASE_NEGATIVE_TRANSFER,
        CASE_STRONG_WRONG_FUSION_CORRECT,
        "strong_wrong_weak_correct",
        CASE_ALL_CORRECT,
        CASE_ALL_WRONG,
    ]
    result["case_type"] = np.select(conditions, choices, default=CASE_OTHER)
    result["research_tags"] = [
        _research_tags(bool(sc), bool(wc), bool(fc), bool(wa), bool(fa))
        for sc, wc, fc, wa, fa in zip(strong_correct, weak_correct, fusion_correct, weak_available, fusion_available)
    ]
    return result[_ordered_case_columns(result)]


def _correctness(pred: Any, truth: pd.Series) -> pd.Series:
    pred_series = pd.to_numeric(pred, errors="coerce") if pred is not None else pd.Series(np.nan, index=truth.index)
    truth_series = pd.to_numeric(truth, errors="coerce")
    return pred_series.notna() & truth_series.notna() & (pred_series.astype("Int64") == truth_series.astype("Int64"))


def _availability_series(frame: pd.DataFrame, column: str, *, default: bool) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(int(default)).astype(bool)
    return values.fillna(str(default)).astype(str).str.lower().isin({"true", "1", "yes"})


def _research_tags(
    strong_correct: bool,
    weak_correct: bool,
    fusion_correct: bool,
    weak_available: bool,
    fusion_available: bool,
) -> str:
    tags: list[str] = []
    if (not strong_correct) and weak_available and weak_correct:
        tags.append("strong_wrong_weak_correct")
        if fusion_available and fusion_correct:
            tags.append("rescue")
        elif fusion_available:
            tags.append("unused_complementary")
    if fusion_available and (not strong_correct) and fusion_correct:
        tags.append("strong_wrong_fusion_correct")
    if fusion_available and strong_correct and (not fusion_correct):
        tags.append("negative_transfer")
    return "|".join(tags) if tags else "none"


def _merge_deltas(cases: pd.DataFrame, per_sample_delta: pd.DataFrame | None) -> pd.DataFrame:
    if per_sample_delta is None or per_sample_delta.empty:
        for column in DELTA_COLUMNS:
            cases[column] = np.nan
        return cases
    delta = per_sample_delta.copy()
    for column in KEY_COLUMNS:
        if column not in delta.columns:
            return cases
    keep = [*KEY_COLUMNS, "weak_modality", *[column for column in DELTA_COLUMNS if column in delta.columns]]
    if "weak_modality" not in delta.columns:
        return cases
    merged = cases.merge(delta[keep].drop_duplicates(KEY_COLUMNS + ["weak_modality"]), on=KEY_COLUMNS + ["weak_modality"], how="left")
    for column in DELTA_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged


def _merge_bucket_columns(cases: pd.DataFrame, communication_state_features: pd.DataFrame | None) -> pd.DataFrame:
    if communication_state_features is None or communication_state_features.empty:
        return cases
    features = _prepare_bucket_features(communication_state_features)
    if features.empty:
        return cases
    bucket_cols = _bucket_columns(features)
    if not bucket_cols:
        return cases
    return cases.merge(features[[*KEY_COLUMNS, *bucket_cols]].drop_duplicates(KEY_COLUMNS), on=KEY_COLUMNS, how="left")


def _prepare_bucket_features(features: pd.DataFrame) -> pd.DataFrame:
    work = features.copy()
    if "horizon_name" not in work.columns and "horizon_idx" in work.columns:
        work["horizon_name"] = work["horizon_idx"].apply(lambda value: f"t+{int(value) + 1}" if pd.notna(value) else "t+1")
    if "horizon_idx" not in work.columns:
        work["horizon_idx"] = 0
    for column in ("sample_id", "dataset_index"):
        if column not in work.columns:
            return pd.DataFrame()
    existing = _bucket_columns(work)
    if existing:
        return work
    for column in COMMUNICATION_BUCKET_FEATURES:
        if column not in work.columns:
            continue
        bucket = _bucketize_series(work[column])
        if bucket is not None:
            work[f"{column}_bucket"] = bucket
    return work


def _bucketize_series(values: pd.Series) -> pd.Series | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    unique = sorted(numeric.dropna().unique().tolist())
    if len(unique) <= 3:
        return numeric.map(lambda value: f"value_{int(value)}" if pd.notna(value) and float(value).is_integer() else str(value))
    median = float(numeric.median())
    return numeric.map(lambda value: np.nan if pd.isna(value) else ("low" if float(value) <= median else "high"))


def _bucket_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.endswith("_bucket")]


def _probability_available(cases: pd.DataFrame) -> bool:
    if cases.empty:
        return False
    required = ["p_true_strong", "p_true_weak", "p_true_fusion", "strong_margin", "weak_margin", "fusion_margin"]
    return all(column in cases.columns and cases[column].notna().any() for column in required)


def _probability_unavailable_reason(cases: pd.DataFrame) -> str:
    if cases.empty:
        return "No case rows were generated."
    missing = [
        column
        for column in ["p_true_strong", "p_true_weak", "p_true_fusion", "strong_margin", "weak_margin", "fusion_margin"]
        if column not in cases.columns or not cases[column].notna().any()
    ]
    return "Missing probability columns or values: " + ", ".join(missing)


def _case_columns() -> list[str]:
    return [
        "sample_id",
        "dataset_index",
        "scene",
        "horizon_idx",
        "horizon_name",
        "strong_modality",
        "weak_modality",
        "strong_weak_pair",
        "y_true",
        "strong_pred",
        "weak_pred",
        "fusion_pred",
        "strong_correct",
        "weak_correct",
        "fusion_correct",
        "case_type",
        "research_tags",
        "strong_prediction_source",
        "weak_prediction_source",
        "fusion_prediction_available",
    ]


def _ordered_case_columns(frame: pd.DataFrame) -> list[str]:
    primary = [
        "sample_id",
        "dataset_index",
        "scene",
        "scene_id",
        "scene_slug",
        "split",
        "horizon_idx",
        "horizon_name",
        "strong_modality",
        "weak_modality",
        "strong_weak_pair",
        "y_true",
        "strong_pred",
        "weak_pred",
        "fusion_pred",
        "strong_correct",
        "weak_correct",
        "fusion_correct",
        "case_type",
        "research_tags",
        "strong_prediction_source",
        "weak_prediction_source",
        "weak_prediction_available",
        "fusion_prediction_available",
        "strong_subset",
        "weak_subset_name",
        "fusion_subset",
        "strong_plus_weak_subset",
        "p_true_strong",
        "p_true_weak",
        "p_true_fusion",
        "weak_gt_gain",
        "fusion_gt_gain",
        "strong_margin",
        "weak_margin",
        "fusion_margin",
        *DELTA_COLUMNS,
        *PATH_COLUMNS,
    ]
    existing_primary = [column for column in primary if column in frame.columns]
    remaining = [column for column in frame.columns if column not in existing_primary]
    return [*existing_primary, *remaining]




__all__ = ["build_case_table"]
