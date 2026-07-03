import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name


STATISTICAL_SUMMARY_SCHEMA_VERSION = "missing_modality_statistical_summary.v1"
DEFAULT_PRIMARY_METRIC = "top1"
DEFAULT_BOOTSTRAP_ITERATIONS = 1000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_PAIRING_KEYS = ("seed", "split", "pattern", "metric_profile", "label_space")
STRICT_COMPARABILITY_VALUES = {"strict", "complete", "strict_comparable", "eligible"}
METRIC_ALIASES = {
    "top1_acc": "top1",
    "acc": "top1",
    "accuracy": "top1",
    "top3_acc": "top3",
    "top5_acc": "top5",
    "DBA": "adba",
    "dba": "adba",
    "top3_dba": "adba",
    "mean_beam_distance": "beam_distance",
    "mean_circular_error": "mae",
    "mean_error": "mae",
}
KNOWN_METRIC_COLUMNS = (
    "top1",
    "top3",
    "top5",
    "within_3",
    "adba",
    "mae",
    "loss",
    "ece",
    "beam_distance",
    "mean_available_modality_reliability",
)


@dataclass(frozen=True)
class StatisticalWarning:
    code: str
    message: str
    method: str | None = None
    metric: str | None = None
    pattern: str | None = None
    source_artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "message": self.message,
            "method": self.method,
            "metric": self.metric,
            "pattern": self.pattern,
            "source_artifact": self.source_artifact,
        }
        return {key: value for key, value in payload.items() if value not in (None, "")}


def read_metric_rows(paths: str | Path | Iterable[str | Path]) -> list[dict[str, Any]]:
    """Read Scene31 missing-pattern, fresh-eval summary, or generic metric rows."""

    if isinstance(paths, (str, Path)):
        candidates = [paths]
    else:
        candidates = list(paths)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        path = Path(candidate)
        if path.suffix.lower() == ".csv":
            raw_rows = _read_csv(path)
        elif path.suffix.lower() == ".json":
            raw_rows = _read_json_rows(path)
        else:
            raise ValueError(f"Unsupported metrics input '{path}'. Expected CSV or JSON.")
        for raw in raw_rows:
            normalized = dict(raw)
            normalized.setdefault("source_artifact", str(path))
            rows.append(normalized)
    return rows


def build_statistical_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    baseline_method: str | None = None,
    candidate_method: str | None = None,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = 0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    min_seed_count: int = 2,
    min_effect: float = 0.0,
) -> dict[str, Any]:
    warnings: list[StatisticalWarning] = []
    observations = _normalize_observations(rows, primary_metric=primary_metric, warnings=warnings)
    summary_rows = _summary_rows(
        observations,
        warnings=warnings,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        min_seed_count=min_seed_count,
    )
    paired = _paired_comparison(
        observations,
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        primary_metric=primary_metric,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        warnings=warnings,
    )
    claim_gate = _claim_gate(
        observations,
        summary_rows,
        paired,
        candidate_method=candidate_method,
        primary_metric=primary_metric,
        min_seed_count=min_seed_count,
        min_effect=min_effect,
        warnings=warnings,
    )
    return {
        "schema_version": STATISTICAL_SUMMARY_SCHEMA_VERSION,
        "primary_metric": _canonical_metric(primary_metric),
        "summary_rows": summary_rows,
        "paired_comparison": paired,
        "claim_gate": claim_gate,
        "warnings": [warning.to_dict() for warning in warnings],
    }


def summarize_metric_files(
    paths: str | Path | Iterable[str | Path],
    **kwargs: Any,
) -> dict[str, Any]:
    return build_statistical_summary(read_metric_rows(paths), **kwargs)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, Mapping)]
    if not isinstance(data, Mapping):
        return []
    for key in ("rows", "results", "metrics", "condition_metrics", "summary_rows"):
        value = data.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(data.get("conditions"), list):
        return [dict(item) for item in data["conditions"] if isinstance(item, Mapping)]
    return [dict(data)]


def _normalize_observations(
    rows: Sequence[Mapping[str, Any]],
    *,
    primary_metric: str,
    warnings: list[StatisticalWarning],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row in rows:
        items = _observations_from_row(row, primary_metric=primary_metric)
        if not items:
            warnings.append(
                StatisticalWarning(
                    code="metric_missing",
                    message="Metric row did not contain a numeric metric value.",
                    method=_method(row),
                    source_artifact=str(row.get("source_artifact") or ""),
                )
            )
        observations.extend(items)
    return observations


def _observations_from_row(row: Mapping[str, Any], *, primary_metric: str) -> list[dict[str, Any]]:
    if _has_value_metric_row(row):
        value = _float(row.get("value", row.get("score", row.get("mean"))))
        if _isnum(value):
            return [_observation(row, metric=row.get("metric", primary_metric), value=value)]
        return []

    observations: list[dict[str, Any]] = []
    pattern = _pattern(row)
    for raw_metric in KNOWN_METRIC_COLUMNS:
        metric = _canonical_metric(raw_metric)
        value = _float(row.get(raw_metric, row.get(_metric_alias_source(metric))))
        if _isnum(value):
            observations.append(_observation(row, metric=metric, value=value, pattern=pattern))
    if observations:
        return observations

    for key, raw_value in row.items():
        pattern_name = _wide_summary_pattern(key)
        if pattern_name is None:
            continue
        value = _float(raw_value)
        if _isnum(value):
            observations.append(_observation(row, metric=primary_metric, value=value, pattern=pattern_name))
    return observations


def _observation(
    row: Mapping[str, Any],
    *,
    metric: Any,
    value: float,
    pattern: str | None = None,
) -> dict[str, Any]:
    method = _method(row)
    run_name = str(row.get("run_name") or row.get("exp_name") or method)
    seed = _seed(row, run_name=run_name)
    return {
        "method": method,
        "run_name": run_name,
        "seed": seed,
        "pattern": pattern or _pattern(row),
        "metric": _canonical_metric(str(metric)),
        "value": float(value),
        "family": str(row.get("family") or row.get("group") or ""),
        "split": str(row.get("split") or row.get("dataset_split") or ""),
        "metric_profile": str(row.get("metric_profile") or ""),
        "label_space": str(row.get("label_space") or ""),
        "comparability_status": _comparability_status(row),
        "source_artifact": str(row.get("source_artifact") or row.get("metrics_path") or ""),
    }


def _summary_rows(
    observations: Sequence[Mapping[str, Any]],
    *,
    warnings: list[StatisticalWarning],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence_level: float,
    min_seed_count: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[(str(observation["method"]), str(observation["metric"]), str(observation.get("family") or ""))].append(observation)

    rows: list[dict[str, Any]] = []
    for (method, metric, family), items in sorted(grouped.items()):
        values = np.asarray([float(item["value"]) for item in items], dtype=np.float64)
        seeds = sorted({str(item.get("seed")) for item in items if item.get("seed") not in (None, "")})
        patterns = sorted({str(item.get("pattern")) for item in items if item.get("pattern") not in (None, "")})
        sources = sorted({str(item.get("source_artifact")) for item in items if item.get("source_artifact")})
        std_available = values.size > 1
        ci = _bootstrap_ci(
            values,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        )
        if len(seeds) < int(min_seed_count):
            warnings.append(
                StatisticalWarning(
                    code="insufficient_seed_count",
                    message=f"Method '{method}' has {len(seeds)} seed(s); statistical claim gate requires {min_seed_count}.",
                    method=method,
                    metric=metric,
                )
            )
        if ci is None:
            warnings.append(
                StatisticalWarning(
                    code="ci_unavailable",
                    message=f"Method '{method}' metric '{metric}' has insufficient samples for bootstrap CI.",
                    method=method,
                    metric=metric,
                )
            )
        rows.append(
            {
                "method": method,
                "metric": metric,
                "family": family,
                "count": int(values.size),
                "seed_count": int(len(seeds)),
                "pattern_count": int(len(patterns)),
                "mean": float(np.mean(values)) if values.size else math.nan,
                "std": float(np.std(values, ddof=1)) if std_available else None,
                "std_status": "available" if std_available else "unavailable",
                "min": float(np.min(values)) if values.size else math.nan,
                "max": float(np.max(values)) if values.size else math.nan,
                "ci_low": ci[0] if ci else None,
                "ci_high": ci[1] if ci else None,
                "ci_status": "available" if ci else "unavailable",
                "bootstrap_seed": int(bootstrap_seed),
                "bootstrap_iterations": int(bootstrap_iterations),
                "confidence_level": float(confidence_level),
                "source_artifact_paths": sources,
            }
        )
    return rows


def _paired_comparison(
    observations: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str | None,
    candidate_method: str | None,
    primary_metric: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    confidence_level: float,
    warnings: list[StatisticalWarning],
) -> dict[str, Any]:
    metric = _canonical_metric(primary_metric)
    if not baseline_method or not candidate_method:
        return {
            "status": "unavailable",
            "reason": "baseline_method and candidate_method are required for paired comparison.",
            "pairing_keys": list(DEFAULT_PAIRING_KEYS),
        }
    baseline = {
        _pair_key(item): item
        for item in observations
        if item.get("method") == baseline_method and item.get("metric") == metric
    }
    candidate = {
        _pair_key(item): item
        for item in observations
        if item.get("method") == candidate_method and item.get("metric") == metric
    }
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        missing = sorted(set(candidate) ^ set(baseline))
        warnings.append(
            StatisticalWarning(
                code="paired_keys_unavailable",
                message="Candidate and baseline rows could not be paired by seed/split/pattern/metric profile/label space.",
                method=candidate_method,
                metric=metric,
            )
        )
        return {
            "status": "unavailable",
            "baseline_method": baseline_method,
            "candidate_method": candidate_method,
            "metric": metric,
            "pairing_keys": list(DEFAULT_PAIRING_KEYS),
            "paired_count": 0,
            "unpaired_key_count": len(missing),
        }

    deltas: list[float] = []
    per_pattern: list[dict[str, Any]] = []
    for key in shared:
        delta = float(candidate[key]["value"]) - float(baseline[key]["value"])
        deltas.append(delta)
        per_pattern.append(
            {
                "seed": key[0],
                "split": key[1],
                "pattern": key[2],
                "metric_profile": key[3],
                "label_space": key[4],
                "delta": delta,
            }
        )
    values = np.asarray(deltas, dtype=np.float64)
    ci = _bootstrap_ci(values, iterations=bootstrap_iterations, seed=bootstrap_seed, confidence_level=confidence_level)
    return {
        "status": "available",
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "metric": metric,
        "pairing_keys": list(DEFAULT_PAIRING_KEYS),
        "paired_count": int(values.size),
        "paired_delta_mean": float(np.mean(values)),
        "paired_delta_min": float(np.min(values)),
        "paired_delta_max": float(np.max(values)),
        "paired_delta_ci_low": ci[0] if ci else None,
        "paired_delta_ci_high": ci[1] if ci else None,
        "win_count": int(np.sum(values > 0)),
        "loss_count": int(np.sum(values < 0)),
        "tie_count": int(np.sum(values == 0)),
        "per_pattern_delta": per_pattern,
    }


def _claim_gate(
    observations: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    paired: Mapping[str, Any],
    *,
    candidate_method: str | None,
    primary_metric: str,
    min_seed_count: int,
    min_effect: float,
    warnings: list[StatisticalWarning],
) -> dict[str, Any]:
    metric = _canonical_metric(primary_metric)
    reasons: list[str] = []
    next_actions: list[str] = []
    candidate_rows = [
        row
        for row in summary_rows
        if row.get("method") == candidate_method and row.get("metric") == metric
    ]
    seed_count = max((int(row.get("seed_count") or 0) for row in candidate_rows), default=0)
    if seed_count < int(min_seed_count):
        reasons.append("insufficient_seed_count")
        next_actions.append("补齐更多 seed 后重新汇总。")
    if paired.get("status") != "available":
        reasons.append("paired_evidence_unavailable")
        next_actions.append("补齐同 seed/split/pattern/metric_profile/label_space 的 baseline 与 candidate rows。")
    else:
        ci_low = paired.get("paired_delta_ci_low")
        delta = paired.get("paired_delta_mean")
        if not _isnum(delta) or float(delta) <= float(min_effect):
            reasons.append("effect_below_threshold")
            next_actions.append("扩大样本或改进候选方法，使 primary metric delta 超过最小效果阈值。")
        if not _isnum(ci_low) or float(ci_low) <= float(min_effect):
            reasons.append("delta_ci_crosses_threshold")
            next_actions.append("补 seed 或 fresh eval，使 paired delta CI 不跨过阈值。")
    if not _strict_comparable(observations, candidate_method=candidate_method, metric=metric):
        reasons.append("comparability_not_strict")
        next_actions.append("补齐 strict comparability fields 后再进入 claim draft。")
    ready = not reasons
    if not ready:
        warnings.append(
            StatisticalWarning(
                code="claim_gate_not_ready",
                message="Statistical claim gate is not ready: " + ", ".join(reasons),
                method=candidate_method,
                metric=metric,
            )
        )
    return {
        "statistical_claim_ready": bool(ready),
        "candidate_method": candidate_method or "",
        "primary_metric": metric,
        "min_seed_count": int(min_seed_count),
        "min_effect": float(min_effect),
        "reasons": reasons,
        "next_actions": next_actions,
    }


def _pair_key(item: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return tuple(str(item.get(key) or "") for key in DEFAULT_PAIRING_KEYS)  # type: ignore[return-value]


def _bootstrap_ci(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
    confidence_level: float,
) -> tuple[float, float] | None:
    if values.size < 2 or int(iterations) <= 0:
        return None
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(iterations), dtype=np.float64)
    for index in range(int(iterations)):
        sample = rng.choice(values, size=values.size, replace=True)
        means[index] = float(np.mean(sample))
    alpha = (1.0 - float(confidence_level)) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def _method(row: Mapping[str, Any]) -> str:
    for key in ("method", "model", "model_group", "group", "run_name", "exp_name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    source = str(row.get("source_artifact") or row.get("metrics_path") or "unknown")
    return Path(source).stem.removesuffix("_missing_patterns") if source else "unknown"


def _seed(row: Mapping[str, Any], *, run_name: str) -> int | str:
    value = row.get("seed")
    if value not in (None, ""):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return str(value)
    match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", run_name)
    return int(match.group(1)) if match else ""


def _pattern(row: Mapping[str, Any]) -> str:
    for key in ("pattern", "pattern_name", "condition_id", "condition"):
        value = row.get(key)
        if value not in (None, ""):
            try:
                return canonical_missing_pattern_name(str(value))
            except ValueError:
                return str(value)
    return "overall"


def _comparability_status(row: Mapping[str, Any]) -> str:
    for key in ("comparability_status", "strict_comparability_status", "claim_status", "status"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _strict_comparable(
    observations: Sequence[Mapping[str, Any]],
    *,
    candidate_method: str | None,
    metric: str,
) -> bool:
    rows = [row for row in observations if row.get("method") == candidate_method and row.get("metric") == metric]
    if not rows:
        return False
    statuses = {str(row.get("comparability_status") or "").lower() for row in rows}
    return bool(statuses) and statuses <= STRICT_COMPARABILITY_VALUES


def _canonical_metric(metric: Any) -> str:
    text = str(metric or "").strip()
    return METRIC_ALIASES.get(text, text.lower())


def _metric_alias_source(metric: str) -> str:
    for raw, canonical in METRIC_ALIASES.items():
        if canonical == metric:
            return raw
    return metric


def _has_value_metric_row(row: Mapping[str, Any]) -> bool:
    return row.get("metric") not in (None, "") and any(row.get(key) not in (None, "") for key in ("value", "score", "mean"))


def _wide_summary_pattern(key: str) -> str | None:
    name = str(key)
    if name.endswith("_mean"):
        name = name[: -len("_mean")]
    if name in {"full", "avg_missing", "overall_mean", "balanced", "non_gps_only"}:
        return name
    if name.startswith("missing_") or name.endswith("_only"):
        try:
            return canonical_missing_pattern_name(name)
        except ValueError:
            return name
    return None


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _isnum(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating)) and math.isfinite(float(value))


__all__ = [
    "DEFAULT_PRIMARY_METRIC",
    "STATISTICAL_SUMMARY_SCHEMA_VERSION",
    "StatisticalWarning",
    "build_statistical_summary",
    "read_metric_rows",
    "summarize_metric_files",
]
