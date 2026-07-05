#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from kd_sensing.eval.missing_patterns import DEFAULT_MODALITIES, get_missing_pattern_mask


CORE_PATTERNS = ("full", "missing_gps", "radar_only", "lidar_only")
CORE_METRICS = ("top1", "top3", "top5", "within_3", "within@3", "mae", "loss")
SUMMARY_METRICS = (
    "full_top1",
    "missing_gps_top1",
    "radar_only_top1",
    "lidar_only_top1",
    "miss1_top1",
    "miss2_top1",
    "miss3_top1",
    "avg_missing_top1",
    "overall_mean_top1",
    "avg_missing_within@3",
    "avg_missing_MAE",
    "balanced",
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    eval_dir = Path(args.eval_dir)
    metrics_path = eval_dir / "apples_to_apples_metrics.csv"
    pattern_path = eval_dir / "pattern_metrics.csv"
    summary_path = eval_dir / "run_summary.csv"
    rows = _read_csv(metrics_path)
    maskfix_eval = bool(args.maskfix_eval or "fresh_eval_maskfix" in eval_dir.parts)
    suspect, reason, details = _mask_suspect(rows, maskfix_eval=maskfix_eval)
    summary_values = _summary_values(rows)

    decorated = _decorate_rows(rows, maskfix_eval=maskfix_eval, suspect=suspect, summary_values=summary_values)
    _write_csv(metrics_path, decorated)
    _write_csv(pattern_path, _pattern_rows(decorated))
    if summary_path.exists():
        _write_csv(summary_path, _decorate_rows(_read_csv(summary_path), maskfix_eval=maskfix_eval, suspect=suspect, summary_values=summary_values))

    run_name = _first_value(rows, "run_name") or eval_dir.name
    payload = {
        "run_name": run_name,
        "maskfix_eval": bool(maskfix_eval),
        "mask_suspect": bool(suspect),
        "reason": reason,
        "checked_patterns": list(CORE_PATTERNS),
        "identical_metric_groups": details["identical_metric_groups"],
        "logits_full_vs_missing_equal": False,
        "logits_check": "unavailable_from_metrics_artifact",
        "excluded_from_official_ranking": bool(suspect),
        "mask_applied_false_patterns": details["mask_applied_false_patterns"],
        "missing_count_mismatches": details["missing_count_mismatches"],
        "metrics_path": str(metrics_path),
    }
    (eval_dir / "mask_suspect.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("mask_suspect=true" if suspect else "mask_suspect=false")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mark Scene31 fresh eval rows as mask_suspect and add maskfix metadata.")
    parser.add_argument("eval_dir")
    parser.add_argument("--maskfix-eval", action="store_true", help="Mark this artifact as formal maskfix eval.")
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _mask_suspect(rows: list[dict[str, str]], *, maskfix_eval: bool) -> tuple[bool, str, dict[str, Any]]:
    by_pattern = _rows_by_pattern(rows)
    identical_groups = _identical_core_groups(by_pattern)
    false_mask = []
    mismatches = []
    if maskfix_eval:
        for pattern, row in by_pattern.items():
            if pattern == "avg_missing":
                continue
            applied = _mask_applied(pattern, maskfix_eval=maskfix_eval)
            if not applied:
                false_mask.append(pattern)
            expected = _expected_missing_count(pattern)
            observed = _float(row.get("missing_count"))
            if expected != "" and (not _isnum(observed) or int(observed) != int(expected)):
                mismatches.append({"pattern": pattern, "expected": expected, "observed": row.get("missing_count", "")})
    reasons: list[str] = []
    if identical_groups:
        reasons.append("identical_core_metrics")
    if false_mask:
        reasons.append("mask_applied_false")
    if mismatches:
        reasons.append("missing_count_mismatch")
    if maskfix_eval and not rows:
        reasons.append("missing_metrics")
    suspect = bool(reasons)
    if not reasons:
        reason = "" if maskfix_eval else "missing patterns differ from full"
    else:
        reason = ";".join(reasons)
    return suspect, reason, {
        "identical_metric_groups": identical_groups,
        "mask_applied_false_patterns": false_mask,
        "missing_count_mismatches": mismatches,
    }


def _rows_by_pattern(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        if str(row.get("status") or "ok") not in {"", "ok"}:
            continue
        pattern = str(row.get("pattern") or "")
        if pattern and pattern not in out:
            out[pattern] = row
    return out


def _identical_core_groups(by_pattern: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    if not all(pattern in by_pattern for pattern in CORE_PATTERNS):
        return []
    full = by_pattern["full"]
    comparable = [by_pattern[pattern] for pattern in CORE_PATTERNS[1:]]
    metrics = [
        metric for metric in CORE_METRICS
        if full.get(metric, "") not in ("", None)
        and all(row.get(metric, "") not in ("", None) for row in comparable)
    ]
    if not metrics:
        return []
    if all(_same(full.get(metric), row.get(metric)) for metric in metrics for row in comparable):
        return [{"patterns": list(CORE_PATTERNS), "metrics": metrics}]
    return []


def _summary_values(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_pattern = _rows_by_pattern(rows)
    values: dict[str, Any] = {
        "full_top1": _pattern_metric(by_pattern, "full", "top1"),
        "missing_gps_top1": _pattern_metric(by_pattern, "missing_gps", "top1"),
        "radar_only_top1": _pattern_metric(by_pattern, "radar_only", "top1"),
        "lidar_only_top1": _pattern_metric(by_pattern, "lidar_only", "top1"),
        "avg_missing_top1": _pattern_metric(by_pattern, "avg_missing", "top1"),
        "avg_missing_within@3": _pattern_metric(by_pattern, "avg_missing", "within_3"),
        "avg_missing_MAE": _pattern_metric(by_pattern, "avg_missing", "mae"),
    }
    for count in (1, 2, 3):
        values[f"miss{count}_top1"] = _bucket_mean(by_pattern, count, "top1")
    if not _isnum(values["avg_missing_top1"]):
        values["avg_missing_top1"] = _avg_missing(by_pattern, "top1")
    if not _isnum(values["avg_missing_within@3"]):
        values["avg_missing_within@3"] = _avg_missing(by_pattern, "within_3")
    if not _isnum(values["avg_missing_MAE"]):
        values["avg_missing_MAE"] = _avg_missing(by_pattern, "mae")
    values["overall_mean_top1"] = _overall_mean(by_pattern)
    values["balanced"] = _balanced(values)
    return {key: _format(value) if isinstance(value, float) else value for key, value in values.items()}


def _decorate_rows(
    rows: list[dict[str, str]],
    *,
    maskfix_eval: bool,
    suspect: bool,
    summary_values: dict[str, Any],
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        item = dict(row)
        run_name = str(item.get("run_name") or "")
        pattern = str(item.get("pattern") or "")
        item.setdefault("method", _method_name(run_name))
        item["maskfix_eval"] = str(bool(maskfix_eval)).lower()
        item["mask_suspect"] = str(bool(suspect)).lower()
        item["mask_applied"] = str(_mask_applied(pattern, maskfix_eval=maskfix_eval)).lower()
        item["available_modalities"] = ",".join(_modalities(pattern, available=True))
        item["missing_modalities"] = ",".join(_modalities(pattern, available=False))
        if pattern and "missing_count" not in item:
            item["missing_count"] = str(_expected_missing_count(pattern))
        for key in SUMMARY_METRICS:
            item[key] = str(summary_values.get(key, ""))
        out.append(item)
    return out


def _pattern_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    fields = {
        "pattern",
        "top1",
        "top3",
        "top5",
        "within@3",
        "mae",
        "missing_count",
        "available_modalities",
        "missing_modalities",
        "mask_applied",
        "mask_suspect",
    }
    out = []
    for row in rows:
        item = dict(row)
        if not item.get("within@3"):
            item["within@3"] = item.get("within_3", "")
        out.append({key: item.get(key, "") for key in [*fields, *[k for k in item if k not in fields]]})
    return out


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _method_name(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)


def _mask_applied(pattern: str, *, maskfix_eval: bool) -> bool:
    if pattern in {"", "avg_missing"}:
        return bool(maskfix_eval)
    return _expected_missing_count(pattern) != "" or bool(maskfix_eval)


def _modalities(pattern: str, *, available: bool) -> list[str]:
    try:
        mask = get_missing_pattern_mask(pattern, DEFAULT_MODALITIES)
    except ValueError:
        return []
    return [mod for mod, keep in zip(DEFAULT_MODALITIES, mask, strict=False) if bool(keep) is available]


def _expected_missing_count(pattern: str) -> int | str:
    try:
        mask = get_missing_pattern_mask(pattern, DEFAULT_MODALITIES)
    except ValueError:
        return ""
    return int(len(mask) - sum(int(value) for value in mask))


def _pattern_metric(by_pattern: dict[str, dict[str, str]], pattern: str, metric: str) -> float:
    return _float(by_pattern.get(pattern, {}).get(metric))


def _bucket_mean(by_pattern: dict[str, dict[str, str]], count: int, metric: str) -> float:
    values = [
        _float(row.get(metric))
        for pattern, row in by_pattern.items()
        if pattern not in {"full", "avg_missing"} and _expected_missing_count(pattern) == count
    ]
    values = [value for value in values if _isnum(value)]
    return sum(values) / len(values) if values else float("nan")


def _avg_missing(by_pattern: dict[str, dict[str, str]], metric: str) -> float:
    values = [
        _float(row.get(metric))
        for pattern, row in by_pattern.items()
        if pattern not in {"full", "avg_missing"} and _isnum(_float(row.get(metric)))
    ]
    return sum(values) / len(values) if values else float("nan")


def _overall_mean(by_pattern: dict[str, dict[str, str]]) -> float:
    values = [_pattern_metric(by_pattern, pattern, "top1") for pattern in CORE_PATTERNS]
    return sum(values) / len(values) if all(_isnum(value) for value in values) else float("nan")


def _balanced(values: dict[str, Any]) -> float:
    avg = _float(values.get("avg_missing_top1"))
    radar = _float(values.get("radar_only_top1"))
    lidar = _float(values.get("lidar_only_top1"))
    if not _isnum(avg):
        return float("nan")
    return avg + 0.25 * (radar if _isnum(radar) else 0.0) + 0.25 * (lidar if _isnum(lidar) else 0.0)


def _same(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _first_value(rows: list[dict[str, str]], key: str) -> str:
    for row in rows:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _format(value: Any) -> str:
    value = _float(value)
    return f"{value:.8g}" if math.isfinite(value) else ""


if __name__ == "__main__":
    raise SystemExit(main())
