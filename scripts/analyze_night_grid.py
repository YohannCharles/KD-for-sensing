#!/usr/bin/env python3

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


KEY_PATTERNS = ("full", "avg_missing", "missing_gps", "missing_radar", "radar_only", "lidar_only")


def main() -> int:
    args = _parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = _read_csv(Path(args.metrics))
    manifest = {row["run_name"]: row for row in _read_csv(Path(args.manifest))}
    by_run = _summary_by_run(metrics, manifest)
    proto = _proto_baseline(by_run, baseline_method=args.baseline_method)
    for row in by_run:
        row["balanced_score"] = _balanced_score(row, proto, args)
    by_group = _summary_by_group(by_run)
    by_method = _mean_std_by_method(by_run)
    delta = _delta_vs_proto(by_run, proto)

    _write_csv(out_dir / "night_grid_summary_by_run.csv", by_run)
    _write_csv(out_dir / "night_grid_summary_by_group.csv", by_group)
    _write_csv(out_dir / "night_grid_mean_std_by_method.csv", by_method)
    _write_csv(out_dir / "night_grid_delta_vs_proto.csv", delta)
    _write_top_candidates(out_dir / "night_grid_top_candidates.md", by_run, by_method, proto)
    _write_observations(out_dir / "night_grid_paper_observations.md", by_run, proto)
    print(f"Wrote night-grid analysis to {out_dir}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze night-grid fresh evaluation metrics.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline_method", "--baseline-method", default="proto")
    parser.add_argument("--out_dir", "--out-dir", default="outputs/scene31/analysis/night_grid")
    parser.add_argument("--radar_weight", type=float, default=0.25)
    parser.add_argument("--lidar_weight", type=float, default=0.25)
    parser.add_argument("--missing_gps_penalty", type=float, default=0.5)
    parser.add_argument("--missing_radar_penalty", type=float, default=0.5)
    parser.add_argument("--full_penalty", type=float, default=0.25)
    return parser


def _summary_by_run(metrics: list[dict[str, str]], manifest: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in metrics:
        if row.get("status") not in {"", "ok"}:
            continue
        grouped[row["run_name"]][row["pattern"]] = row
    rows = []
    for run_name, patterns in grouped.items():
        meta = manifest.get(run_name, {})
        row: dict[str, Any] = {
            "run_name": run_name,
            "method": _method_name(run_name, meta),
            "group": meta.get("group", ""),
            "seed": meta.get("seed", ""),
        }
        for pattern in KEY_PATTERNS:
            row[f"{pattern}_top1"] = _float(patterns.get(pattern, {}).get("top1"))
            row[f"{pattern}_adba"] = _float(patterns.get(pattern, {}).get("adba"))
        pattern_top1 = [_float(value.get("top1")) for name, value in patterns.items() if name not in {"full", "avg_missing"}]
        pattern_top1 = [value for value in pattern_top1 if _isnum(value)]
        row["worst_pattern_top1"] = min(pattern_top1) if pattern_top1 else float("nan")
        rows.append(row)
    return sorted(rows, key=lambda item: (item.get("group", ""), item.get("run_name", "")))


def _proto_baseline(rows: list[dict[str, Any]], *, baseline_method: str) -> dict[str, float]:
    candidates = [
        row for row in rows
        if row.get("method") == baseline_method or row.get("method") == "proto" or str(row.get("run_name", "")).startswith("main_v3_strong_reliability_proto")
    ]
    out: dict[str, float] = {}
    for key in [f"{pattern}_top1" for pattern in KEY_PATTERNS]:
        values = [row[key] for row in candidates if _isnum(row.get(key))]
        out[key] = mean(values) if values else float("nan")
    return out


def _balanced_score(row: dict[str, Any], proto: dict[str, float], args: argparse.Namespace) -> float:
    avg_missing = row.get("avg_missing_top1", float("nan"))
    if not _isnum(avg_missing):
        return float("nan")
    score = avg_missing
    score += args.radar_weight * _zero_nan(row.get("radar_only_top1"))
    score += args.lidar_weight * _zero_nan(row.get("lidar_only_top1"))
    score -= args.missing_gps_penalty * max(0.0, _zero_nan(proto.get("missing_gps_top1")) - _zero_nan(row.get("missing_gps_top1")))
    score -= args.missing_radar_penalty * max(0.0, _zero_nan(proto.get("missing_radar_top1")) - _zero_nan(row.get("missing_radar_top1")))
    score -= args.full_penalty * max(0.0, _zero_nan(proto.get("full_top1")) - _zero_nan(row.get("full_top1")))
    return score


def _summary_by_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group", ""))].append(row)
    out = []
    for group, items in groups.items():
        out.append({"group": group, "runs": len(items), "mean_balanced_score": _mean_value(items, "balanced_score"), "mean_avg_missing_top1": _mean_value(items, "avg_missing_top1")})
    return out


def _mean_std_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        methods[str(row["method"])].append(row)
    out = []
    for method, items in methods.items():
        row: dict[str, Any] = {"method": method, "n": len(items), "group": items[0].get("group", "")}
        for key in ("full_top1", "avg_missing_top1", "missing_gps_top1", "missing_radar_top1", "radar_only_top1", "lidar_only_top1", "balanced_score"):
            values = [item[key] for item in items if _isnum(item.get(key))]
            row[f"{key}_mean"] = mean(values) if values else float("nan")
            row[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else float("nan")
        out.append(row)
    return sorted(out, key=lambda item: _zero_nan(item.get("balanced_score_mean")), reverse=True)


def _delta_vs_proto(rows: list[dict[str, Any]], proto: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        for key, proto_value in proto.items():
            current = row.get(key)
            out.append({"run_name": row["run_name"], "group": row.get("group", ""), "metric": key, "value": current, "proto_mean": proto_value, "delta": current - proto_value if _isnum(current) and _isnum(proto_value) else float("nan")})
    return out


def _write_top_candidates(path: Path, rows: list[dict[str, Any]], by_method: list[dict[str, Any]], proto: dict[str, float]) -> None:
    lines = ["# Night Grid Top Candidates", ""]
    checks = [
        ("best overall avg_missing", "avg_missing_top1", None),
        ("best radar_only", "radar_only_top1", None),
        ("best lidar_only", "lidar_only_top1", None),
        ("best balanced_score", "balanced_score", None),
        ("best without hurting missing_gps", "balanced_score", lambda row: _zero_nan(row.get("missing_gps_top1")) >= _zero_nan(proto.get("missing_gps_top1"))),
        ("best without hurting missing_radar", "balanced_score", lambda row: _zero_nan(row.get("missing_radar_top1")) >= _zero_nan(proto.get("missing_radar_top1"))),
    ]
    for title, metric, pred in checks:
        row = _best(rows, metric, pred)
        lines.append(f"- {title}: {_label(row, metric)}")
    lines.extend(["", "## Recommended Top 3 For Seed3 / 40 Epoch", ""])
    for row in by_method[:3]:
        caution = _std_caution(row)
        lines.append(f"- {row['method']}: balanced={_fmt(row.get('balanced_score_mean'))}, avg_missing={_fmt(row.get('avg_missing_top1_mean'))}{caution}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_observations(path: Path, rows: list[dict[str, Any]], proto: dict[str, float]) -> None:
    best = _best(rows, "balanced_score", None)
    lines = ["# Night Grid Paper Observations", ""]
    if best:
        lines.append(
            f"当前粗筛 balanced_score 最优为 `{best['run_name']}`，avg_missing top1={_fmt(best.get('avg_missing_top1'))}，"
            f"radar_only top1={_fmt(best.get('radar_only_top1'))}，lidar_only top1={_fmt(best.get('lidar_only_top1'))}。"
        )
        lines.append("")
        lines.append("若该提升小于 proto seed std，后续必须以 seed3/40 epoch 复核后再写入主结论。")
    else:
        lines.append("暂无可用候选；需要先完成 fresh eval。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best(rows: list[dict[str, Any]], metric: str, pred) -> dict[str, Any] | None:
    valid = [row for row in rows if _isnum(row.get(metric)) and (pred is None or pred(row))]
    return max(valid, key=lambda row: row[metric]) if valid else None


def _label(row: dict[str, Any] | None, metric: str) -> str:
    return "unavailable" if row is None else f"`{row['run_name']}` {metric}={_fmt(row.get(metric))}"


def _std_caution(row: dict[str, Any]) -> str:
    delta = _zero_nan(row.get("avg_missing_top1_mean"))
    std = _zero_nan(row.get("avg_missing_top1_std"))
    return " (谨慎：提升可能小于 seed std)" if std > 0 and abs(delta) < std else ""


def _method_name(run_name: str, meta: dict[str, str]) -> str:
    tags = str(meta.get("method_tags", ""))
    if meta.get("group") == "baseline" and tags:
        return tags
    return re.sub(r"_seed\d+$", "", run_name)


def _mean_value(rows: list[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if _isnum(row.get(key))]
    return mean(values) if values else float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()} for row in rows])


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _isnum(value: Any) -> bool:
    number = _float(value)
    return math.isfinite(number)


def _zero_nan(value: Any) -> float:
    number = _float(value)
    return number if math.isfinite(number) else 0.0


def _fmt(value: Any) -> str:
    number = _float(value)
    return f"{number:.8g}" if math.isfinite(number) else ""


if __name__ == "__main__":
    raise SystemExit(main())
