#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from kd_sensing.diagnostics.scene31_summary import bc_next as bc
from kd_sensing.eval.missing_buckets import missing_bucket_mapping_from_rows
from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name


REFERENCE_METHOD = "scenes31_34_proto_randomdrop_subset_es40"
CORE_PATTERNS = ("full", "missing_gps", "missing_radar", "radar_only", "lidar_only")
FIRST_FIELDS = [
    "scene",
    "run_name",
    "method",
    "seed",
    "status",
    "full_top1",
    "miss1_top1",
    "miss2_top1",
    "miss3_top1",
    "radar_only_top1",
    "lidar_only_top1",
    "avg_missing_top1",
    "overall_mean_top1",
    "avg_missing_within@3",
    "avg_missing_MAE",
    "balanced",
    "metrics_path",
]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summarize(Path(args.root), Path(args.out))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31-34 subset reliability fresh eval metrics.")
    parser.add_argument("--root", default="outputs/scenes31_34_subset_reliability_lmdb")
    parser.add_argument("--out", default="outputs/scenes31_34_subset_reliability_lmdb/summary")
    return parser


def summarize(root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(root)
    bucket_mapping, bucket_warnings = missing_bucket_mapping_from_rows(rows)
    per_run = _per_run_rows(rows, bucket_mapping)
    pooled = _method_rows([row for row in per_run if not row.get("scene")])
    per_scene = _per_scene_method_rows([row for row in per_run if row.get("scene")])
    delta_rows = _delta_rows(pooled)
    stability = _scene_stability_rows(per_scene)
    availability = _availability(root)
    conclusion = _conclusion_lines(
        availability=availability,
        pooled=pooled,
        per_run=per_run,
        stability=stability,
        warnings=bucket_warnings,
    )

    _write_csv(out_dir / "per_run.csv", per_run, _fields(per_run, FIRST_FIELDS))
    _write_csv(out_dir / "pooled_method_mean_std.csv", pooled, _fields(pooled, ["method", "n"]))
    _write_csv(out_dir / "per_scene_method_mean_std.csv", per_scene, _fields(per_scene, ["scene", "method", "n"]))
    _write_csv(out_dir / "delta_vs_scenes31_34_randomdrop_subset.csv", delta_rows, _fields(delta_rows, ["method", "n"]))
    _write_rank(out_dir / "rank_by_avg_missing_top1.md", pooled)
    _write_stability_rank(out_dir / "rank_by_scene_stability.md", stability)
    (out_dir / "scenes31_34_conclusion.txt").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
    print(f"Wrote Scene31-34 subset reliability summary to {out_dir}.")
    return {
        "per_run": per_run,
        "pooled": pooled,
        "per_scene": per_scene,
        "delta_rows": delta_rows,
        "stability": stability,
        "conclusion": conclusion,
    }


def _load_rows(root: Path) -> list[dict[str, Any]]:
    args = argparse.Namespace(root=[str(root)], metrics=[], manifest="", name_prefix="")
    return bc._load_metric_rows(args)


def _per_run_rows(rows: list[dict[str, Any]], bucket_mapping: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    statuses: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        run_name = str(row.get("run_name") or "")
        if not run_name:
            continue
        scene = _scene(row)
        key = (scene, run_name)
        statuses[key].add(str(row.get("status") or "ok"))
        if row.get("status") not in {"", "ok", None}:
            meta.setdefault(key, row)
            continue
        pattern = canonical_missing_pattern_name(str(row.get("pattern") or ""))
        for metric in ("top1", "within_3", "mae"):
            value = _float(row.get(metric))
            if _isnum(value):
                grouped[key][pattern][metric].append(value)
        meta.setdefault(key, row)
    out: list[dict[str, Any]] = []
    for key in sorted(set(grouped) | set(meta)):
        scene, run_name = key
        patterns = grouped.get(key, {})
        top1 = {pattern: _mean(values.get("top1", [])) for pattern, values in patterns.items()}
        within = {pattern: _mean(values.get("within_3", [])) for pattern, values in patterns.items()}
        mae = {pattern: _mean(values.get("mae", [])) for pattern, values in patterns.items()}
        row = {
            "scene": scene,
            "run_name": run_name,
            "method": _method_name(run_name),
            "seed": _seed(run_name, meta.get(key, {})),
            "status": _status(statuses.get(key, set())),
            "metrics_path": meta.get(key, {}).get("metrics_path", ""),
            "full_top1": top1.get("full", float("nan")),
            "miss1_top1": _bucket_mean(top1, bucket_mapping, 1),
            "miss2_top1": _bucket_mean(top1, bucket_mapping, 2),
            "miss3_top1": _bucket_mean(top1, bucket_mapping, 3),
            "radar_only_top1": top1.get("radar_only", float("nan")),
            "lidar_only_top1": top1.get("lidar_only", float("nan")),
            "avg_missing_top1": top1.get("avg_missing", _avg_missing(top1)),
            "overall_mean_top1": _overall_mean(top1),
            "avg_missing_within@3": within.get("avg_missing", _avg_missing(within)),
            "avg_missing_MAE": mae.get("avg_missing", _avg_missing(mae)),
        }
        row["balanced"] = _balanced(row)
        out.append(row)
    return out


def _method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") in {"ok", ""}:
            grouped[str(row.get("method") or "")].append(row)
    out: list[dict[str, Any]] = []
    for method, items in grouped.items():
        item: dict[str, Any] = {"method": method, "n": len(items)}
        for metric in _numeric_fields(items):
            if metric == "seed":
                continue
            values = [_float(row.get(metric)) for row in items if _isnum(row.get(metric))]
            item[f"{metric}_mean"] = mean(values) if values else float("nan")
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else float("nan")
        out.append(item)
    return sorted(out, key=lambda row: _zero(row.get("avg_missing_top1_mean")), reverse=True)


def _per_scene_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("scene") or ""), str(row.get("method") or ""))].append(row)
    out: list[dict[str, Any]] = []
    for (scene, method), items in grouped.items():
        for item in _method_rows(items):
            item["scene"] = scene
            item["method"] = method
            out.append(item)
    return sorted(out, key=lambda row: (str(row.get("scene", "")), -_zero(row.get("avg_missing_top1_mean"))))


def _delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = next((row for row in rows if row.get("method") == REFERENCE_METHOD), None)
    metrics = ("full_top1", "miss1_top1", "miss2_top1", "miss3_top1", "avg_missing_top1", "overall_mean_top1", "avg_missing_MAE", "balanced")
    out = []
    for row in rows:
        item = {"method": row.get("method", ""), "n": row.get("n", "")}
        for metric in metrics:
            value = _method_value(row, metric)
            base = _method_value(reference, metric)
            item[metric] = value
            item[f"delta_{metric}_vs_randomdrop_subset"] = _delta(value, base)
        out.append(item)
    return out


def _scene_stability_rows(per_scene: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_scene:
        grouped[str(row.get("method") or "")].append(row)
    out: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        avg_values = [_float(row.get("avg_missing_top1_mean")) for row in rows if _isnum(row.get("avg_missing_top1_mean"))]
        mae_values = [_float(row.get("avg_missing_MAE_mean")) for row in rows if _isnum(row.get("avg_missing_MAE_mean"))]
        out.append(
            {
                "method": method,
                "scene_count": len(rows),
                "avg_missing_top1_mean_over_scenes": mean(avg_values) if avg_values else float("nan"),
                "avg_missing_top1_std_over_scenes": stdev(avg_values) if len(avg_values) > 1 else 0.0 if avg_values else float("nan"),
                "avg_missing_MAE_mean_over_scenes": mean(mae_values) if mae_values else float("nan"),
            }
        )
    return sorted(
        out,
        key=lambda row: (
            _zero(row.get("avg_missing_top1_mean_over_scenes")),
            -_large(row.get("avg_missing_top1_std_over_scenes")),
            -_large(row.get("avg_missing_MAE_mean_over_scenes")),
        ),
        reverse=True,
    )


def _write_rank(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Rank By Avg Missing Top1", ""]
    columns = ["method", "n", "full_top1", "miss1_top1", "miss2_top1", "miss3_top1", "avg_missing_top1", "overall_mean_top1", "avg_missing_MAE"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in sorted(rows, key=lambda item: _zero(item.get("avg_missing_top1_mean")), reverse=True):
        lines.append(
            "| {method} | {n} | {full} | {miss1} | {miss2} | {miss3} | {avg} | {overall} | {mae} |".format(
                method=row.get("method", ""),
                n=row.get("n", ""),
                full=_mean_std(row, "full_top1"),
                miss1=_mean_std(row, "miss1_top1"),
                miss2=_mean_std(row, "miss2_top1"),
                miss3=_mean_std(row, "miss3_top1"),
                avg=_mean_std(row, "avg_missing_top1"),
                overall=_mean_std(row, "overall_mean_top1"),
                mae=_mean_std(row, "avg_missing_MAE"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stability_rank(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Rank By Scene Stability", ""]
    columns = ["method", "scene_count", "avg_missing_top1_mean_over_scenes", "avg_missing_top1_std_over_scenes", "avg_missing_MAE_mean_over_scenes"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) if column != "method" else str(row.get(column, "")) for column in columns) + " |")
    if not rows:
        lines.extend(["", "No per-scene metrics available yet."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _conclusion_lines(
    *,
    availability: dict[str, Any],
    pooled: list[dict[str, Any]],
    per_run: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    warnings: list[str],
) -> list[str]:
    by_method = {str(row.get("method")): row for row in pooled}
    reference = by_method.get(REFERENCE_METHOD)
    reliability = by_method.get("scenes31_34_proto_randomdrop_subset_reliability_fusion_es40")
    winner = pooled[0].get("method", "unavailable") if pooled else "unavailable"
    delta = _delta(_method_value(reliability, "avg_missing_top1"), _method_value(reference, "avg_missing_top1"))
    completed = [row for row in per_run if row.get("status") in {"ok", ""}]
    failed = [row for row in per_run if row.get("status") not in {"ok", ""}]
    scene_rows = availability.get("scenes", [])
    lines = [
        "Scene31-34 quick validation:",
        "- data/config availability: "
        + ", ".join(f"Scene{row.get('scene')}={'ok' if row.get('available') else 'missing'}" for row in scene_rows),
        f"- completed runs: {len(completed)}",
        "- missing/eval failures: " + (", ".join(str(row.get("run_name")) for row in failed) if failed else "none"),
        f"- current pooled winner: {winner}",
        "- per-scene stability: " + (stability[0].get("method", "unavailable") if stability else "unavailable"),
        f"- whether reliability_fusion improves over randomdrop_subset: {'yes' if _isnum(delta) and delta > 0 else 'no' if _isnum(delta) else 'unavailable'}",
        f"- whether to expand to seed2/3: {'yes' if _isnum(delta) and delta > 0 else 'not_yet'}",
    ]
    if warnings:
        lines.extend(["", "Warnings:", *[f"- {warning}" for warning in warnings]])
    return lines


def _availability(root: Path) -> dict[str, Any]:
    path = root / "scene_availability.json"
    if not path.exists():
        return {"scenes": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"scenes": []}
    return data if isinstance(data, dict) else {"scenes": []}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def _fields(rows: list[dict[str, Any]], first: list[str]) -> list[str]:
    fields = list(first)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _scene(row: dict[str, Any]) -> str:
    value = row.get("scene")
    if value not in (None, ""):
        return f"Scene{int(float(value))}" if _isnum(value) else str(value)
    path = Path(str(row.get("metrics_path") or ""))
    for part in path.parts:
        match = re.fullmatch(r"[Ss]cene(\d+)", part)
        if match:
            return f"Scene{match.group(1)}"
    return ""


def _method_name(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", run_name)


def _seed(run_name: str, row: dict[str, Any]) -> Any:
    if row.get("seed") not in (None, ""):
        return row.get("seed")
    match = re.search(r"_seed(\d+)$", run_name)
    return int(match.group(1)) if match else ""


def _status(values: set[str]) -> str:
    cleaned = {value for value in values if value}
    if not cleaned or cleaned == {"ok"}:
        return "ok"
    return ";".join(sorted(cleaned))


def _bucket_mean(values: dict[str, float], mapping: dict[str, dict[str, Any]], count: int) -> float:
    nums = [
        value for pattern, value in values.items()
        if pattern not in {"full", "avg_missing"} and _missing_count(pattern, mapping) == count and _isnum(value)
    ]
    return _mean(nums)


def _missing_count(pattern: str, mapping: dict[str, dict[str, Any]]) -> int | None:
    raw = mapping.get(pattern, {}).get("missing_count")
    if _isnum(raw):
        return int(float(raw))
    return None


def _avg_missing(values: dict[str, float]) -> float:
    nums = [value for pattern, value in values.items() if pattern not in {"full", "avg_missing"} and _isnum(value)]
    return _mean(nums)


def _overall_mean(values: dict[str, float]) -> float:
    nums = [_float(values.get(pattern)) for pattern in CORE_PATTERNS]
    return mean(nums) if all(_isnum(value) for value in nums) else float("nan")


def _balanced(row: dict[str, Any]) -> float:
    avg = _float(row.get("avg_missing_top1"))
    radar = _float(row.get("radar_only_top1"))
    lidar = _float(row.get("lidar_only_top1"))
    if not _isnum(avg):
        return float("nan")
    return avg + 0.25 * (radar if _isnum(radar) else 0.0) + 0.25 * (lidar if _isnum(lidar) else 0.0)


def _method_value(row: dict[str, Any] | None, metric: str) -> float:
    if row is None:
        return float("nan")
    return _float(row.get(f"{metric}_mean", row.get(metric)))


def _mean_std(row: dict[str, Any], metric: str) -> str:
    value = _method_value(row, metric)
    std = _float(row.get(f"{metric}_std"))
    if not _isnum(value):
        return ""
    return f"{_fmt(value)}+-{_fmt(std)}" if _isnum(std) else _fmt(value)


def _numeric_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and _isnum(value):
                fields.append(key)
    return fields


def _delta(value: Any, base: Any) -> float:
    left = _float(value)
    right = _float(base)
    return left - right if _isnum(left) and _isnum(right) else float("nan")


def _mean(values: list[float]) -> float:
    nums = [value for value in values if _isnum(value)]
    return mean(nums) if nums else float("nan")


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _zero(value: Any) -> float:
    value = _float(value)
    return value if math.isfinite(value) else -math.inf


def _large(value: Any) -> float:
    value = _float(value)
    return value if math.isfinite(value) else math.inf


def _fmt(value: Any) -> str:
    value = _float(value)
    return f"{value:.5f}" if math.isfinite(value) else ""


if __name__ == "__main__":
    raise SystemExit(main())
