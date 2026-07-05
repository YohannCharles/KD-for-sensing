#!/usr/bin/env python3

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REFERENCE_METHOD = "scenes31_34_proto_randomdrop_subset_es40"
SCENE_ORDER = ("Scene31", "Scene32", "Scene33", "Scene34")
CORE_PATTERNS = ("full", "missing_gps", "missing_radar", "radar_only", "lidar_only")
PER_RUN_FIELDS = [
    "method",
    "run_name",
    "seed",
    "scene",
    "full_top1",
    "miss1_top1",
    "miss2_top1",
    "miss3_top1",
    "avg_missing_top1",
    "overall_mean_top1",
    "avg_missing_within@3",
    "avg_missing_MAE",
    "balanced",
    "num_samples",
    "num_patterns",
]
METHOD_FIELDS = [
    "method",
    "n",
    "scene",
    "full_top1_mean",
    "full_top1_std",
    "miss1_top1_mean",
    "miss1_top1_std",
    "miss2_top1_mean",
    "miss2_top1_std",
    "miss3_top1_mean",
    "miss3_top1_std",
    "avg_missing_top1_mean",
    "avg_missing_top1_std",
    "overall_mean_top1_mean",
    "overall_mean_top1_std",
    "avg_missing_within@3_mean",
    "avg_missing_within@3_std",
    "avg_missing_MAE_mean",
    "avg_missing_MAE_std",
    "balanced_mean",
    "balanced_std",
]
MEAN_OVER_SCENES_FIELDS = [
    "method",
    "n",
    "avg_missing_top1_mean_over_scenes",
    "avg_missing_top1_std_over_scenes",
    "full_top1_mean_over_scenes",
    "full_top1_std_over_scenes",
    "miss1_top1_mean_over_scenes",
    "miss1_top1_std_over_scenes",
    "miss2_top1_mean_over_scenes",
    "miss2_top1_std_over_scenes",
    "miss3_top1_mean_over_scenes",
    "miss3_top1_std_over_scenes",
    "avg_missing_within@3_mean_over_scenes",
    "avg_missing_MAE_mean_over_scenes",
    "balanced_mean_over_scenes",
]
DELTA_FIELDS = [
    "scene",
    "method",
    "reference_method",
    "delta_full_top1",
    "delta_miss1_top1",
    "delta_miss2_top1",
    "delta_miss3_top1",
    "delta_avg_missing_top1",
    "delta_overall_mean_top1",
    "delta_avg_missing_within@3",
    "delta_avg_missing_MAE",
    "delta_balanced",
]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summarize(Path(args.root), Path(args.out))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Scene31-34 per-scene prediction metrics.")
    parser.add_argument("--root", default="outputs/scenes31_34_subset_reliability_lmdb")
    parser.add_argument("--out", default="outputs/scenes31_34_subset_reliability_lmdb/per_scene_summary")
    return parser


def summarize(root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows, prediction_rows = _load_prediction_rows(root)
    per_run = _per_scene_per_run(prediction_rows)
    method_rows = _per_scene_method_rows(per_run)
    mean_rows = _mean_over_scenes(method_rows)
    delta_rows, delta_warnings = _delta_rows(method_rows)
    conclusion = _conclusion_lines(per_run, method_rows, mean_rows, delta_warnings)

    _write_csv(out_dir / "per_sample_predictions_index.csv", index_rows, _fields(index_rows, ["run_name", "method", "seed", "path"]))
    _write_csv(out_dir / "per_scene_per_run.csv", per_run, PER_RUN_FIELDS)
    _write_csv(out_dir / "per_scene_method_mean_std.csv", method_rows, METHOD_FIELDS)
    _write_csv(out_dir / "mean_over_scenes.csv", mean_rows, MEAN_OVER_SCENES_FIELDS)
    _write_csv(out_dir / "delta_vs_randomdrop_subset_per_scene.csv", delta_rows, DELTA_FIELDS)
    _write_avg_missing_rank(out_dir / "rank_by_avg_missing_per_scene.md", method_rows)
    _write_stability_rank(out_dir / "rank_by_scene_stability.md", mean_rows)
    (out_dir / "scenes31_34_per_scene_conclusion.txt").write_text("\n".join(conclusion) + "\n", encoding="utf-8")
    print(f"Wrote Scene31-34 per-scene summary to {out_dir}.")
    return {
        "index": index_rows,
        "per_run": per_run,
        "method_rows": method_rows,
        "mean_over_scenes": mean_rows,
        "delta_rows": delta_rows,
        "conclusion": conclusion,
    }


def _load_prediction_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = sorted((root / "fresh_eval_with_scene").glob("*/predictions_by_pattern.csv"))
    if not files:
        files = sorted(root.glob("fresh_eval*/**/predictions_by_pattern.csv"))
    index_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in files:
        file_rows = _read_csv(path)
        run_name = _first(file_rows, "run_name") or path.parent.name
        method = _first(file_rows, "method") or _method_name(run_name)
        seed = _first(file_rows, "seed") or _seed(run_name)
        scenes = sorted({_scene_label(row.get("scene")) for row in file_rows if _scene_label(row.get("scene"))})
        index_rows.append(
            {
                "run_name": run_name,
                "method": method,
                "seed": seed,
                "path": str(path),
                "rows": len(file_rows),
                "scenes": ",".join(scenes),
                "scene_count": len(scenes),
            }
        )
        rows.extend({**row, "predictions_path": str(path)} for row in file_rows)
    return index_rows, rows


def _per_scene_per_run(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        run_name = str(row.get("run_name") or "")
        scene = _scene_label(row.get("scene"))
        pattern = str(row.get("pattern") or "")
        if run_name and scene and pattern:
            grouped[(run_name, scene, pattern)].append(row)

    pattern_metrics: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for (run_name, scene, pattern), items in grouped.items():
        key = (run_name, scene)
        meta.setdefault(key, items[0])
        pattern_metrics[key][pattern] = {
            "top1": _mean([_float(row.get("top1_correct")) for row in items]),
            "within3": _mean([_float(row.get("within3_correct")) for row in items]),
            "mae": _mean([_float(row.get("abs_error")) for row in items]),
            "missing_count": _missing_count(items[0]),
            "num_samples": len({str(row.get("sample_id") or index) for index, row in enumerate(items)}),
        }

    out: list[dict[str, Any]] = []
    for (run_name, scene), patterns in sorted(pattern_metrics.items(), key=lambda item: (_scene_sort(item[0][1]), item[0][0])):
        top1 = {pattern: values["top1"] for pattern, values in patterns.items()}
        within3 = {pattern: values["within3"] for pattern, values in patterns.items()}
        mae = {pattern: values["mae"] for pattern, values in patterns.items()}
        row_meta = meta[(run_name, scene)]
        item = {
            "method": row_meta.get("method") or _method_name(run_name),
            "run_name": run_name,
            "seed": row_meta.get("seed") or _seed(run_name),
            "scene": scene,
            "full_top1": top1.get("full", math.nan),
            "miss1_top1": _bucket_mean(top1, patterns, 1),
            "miss2_top1": _bucket_mean(top1, patterns, 2),
            "miss3_top1": _bucket_mean(top1, patterns, 3),
            "radar_only_top1": top1.get("radar_only", math.nan),
            "lidar_only_top1": top1.get("lidar_only", math.nan),
            "avg_missing_top1": _avg_missing(top1),
            "overall_mean_top1": _overall_mean(top1),
            "avg_missing_within@3": _avg_missing(within3),
            "avg_missing_MAE": _avg_missing(mae),
            "num_samples": max((int(values.get("num_samples", 0) or 0) for values in patterns.values()), default=0),
            "num_patterns": len(patterns),
        }
        item["balanced"] = _balanced(item)
        out.append(item)
    return out


def _per_scene_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("scene") or ""), str(row.get("method") or ""))].append(row)
    out: list[dict[str, Any]] = []
    metrics = [
        "full_top1",
        "miss1_top1",
        "miss2_top1",
        "miss3_top1",
        "avg_missing_top1",
        "overall_mean_top1",
        "avg_missing_within@3",
        "avg_missing_MAE",
        "balanced",
    ]
    for (scene, method), items in grouped.items():
        item = {"method": method, "n": len(items), "scene": scene}
        for metric in metrics:
            values = [_float(row.get(metric)) for row in items if _isnum(row.get(metric))]
            item[f"{metric}_mean"] = mean(values) if values else math.nan
            item[f"{metric}_std"] = stdev(values) if len(values) > 1 else math.nan
        out.append(item)
    return sorted(out, key=lambda row: (_scene_sort(str(row.get("scene") or "")), -_zero(row.get("avg_missing_top1_mean"))))


def _mean_over_scenes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method") or "")].append(row)
    out: list[dict[str, Any]] = []
    for method, items in grouped.items():
        item = {"method": method, "n": len(items)}
        for metric in ("avg_missing_top1", "full_top1", "miss1_top1", "miss2_top1", "miss3_top1"):
            values = [_float(row.get(f"{metric}_mean")) for row in items if _isnum(row.get(f"{metric}_mean"))]
            item[f"{metric}_mean_over_scenes"] = mean(values) if values else math.nan
            item[f"{metric}_std_over_scenes"] = stdev(values) if len(values) > 1 else math.nan
        for metric in ("avg_missing_within@3", "avg_missing_MAE", "balanced"):
            values = [_float(row.get(f"{metric}_mean")) for row in items if _isnum(row.get(f"{metric}_mean"))]
            item[f"{metric}_mean_over_scenes"] = mean(values) if values else math.nan
        out.append(item)
    return sorted(out, key=lambda row: (_zero(row.get("avg_missing_top1_mean_over_scenes")), -_large(row.get("avg_missing_top1_std_over_scenes")), -_large(row.get("avg_missing_MAE_mean_over_scenes"))), reverse=True)


def _delta_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    by_scene_method = {(str(row.get("scene")), str(row.get("method"))): row for row in rows}
    warnings: list[str] = []
    out: list[dict[str, Any]] = []
    for row in rows:
        scene = str(row.get("scene") or "")
        reference = by_scene_method.get((scene, REFERENCE_METHOD))
        if reference is None:
            warnings.append(f"missing reference {REFERENCE_METHOD} for {scene}")
        item = {"scene": scene, "method": row.get("method", ""), "reference_method": REFERENCE_METHOD}
        for metric in (
            "full_top1",
            "miss1_top1",
            "miss2_top1",
            "miss3_top1",
            "avg_missing_top1",
            "overall_mean_top1",
            "avg_missing_within@3",
            "avg_missing_MAE",
            "balanced",
        ):
            item[f"delta_{metric}"] = _delta(row.get(f"{metric}_mean"), reference.get(f"{metric}_mean") if reference else math.nan)
        out.append(item)
    return out, sorted(set(warnings))


def _conclusion_lines(
    per_run: list[dict[str, Any]],
    per_scene: list[dict[str, Any]],
    mean_rows: list[dict[str, Any]],
    warnings: list[str],
) -> list[str]:
    scenes_present = sorted({str(row.get("scene")) for row in per_run if row.get("scene")}, key=_scene_sort)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_scene:
        by_scene[str(row.get("scene") or "")].append(row)
    winners = {
        scene: max(rows, key=lambda row: _zero(row.get("avg_missing_top1_mean"))).get("method", "")
        for scene, rows in by_scene.items()
        if rows
    }
    stability_winner = mean_rows[0].get("method", "unavailable") if mean_rows else "unavailable"
    subset = next((row for row in mean_rows if row.get("method") == REFERENCE_METHOD), None)
    support = bool(subset and mean_rows and mean_rows[0].get("method") == REFERENCE_METHOD)
    lines = [
        "Scene31-34 per-scene conclusion:",
        "- scenes present: " + (", ".join(scenes_present) if scenes_present else "none"),
        "- no scene silently dropped: " + ("yes" if set(SCENE_ORDER) <= set(scenes_present) else "warning"),
        f"- scene stability winner: {stability_winner}",
        f"- randomdrop_subset stable by ranking: {'yes' if support else 'mixed_or_no'}",
        "- per-scene winners: " + (", ".join(f"{scene}={method}" for scene, method in sorted(winners.items(), key=lambda item: _scene_sort(item[0]))) if winners else "unavailable"),
        "- seed std warning: Scene31-34 quick validation currently has n=1 per method; seed std is NaN by design.",
    ]
    if warnings:
        lines.extend(["", "Warnings:", *[f"- {warning}" for warning in warnings]])
    return lines


def _write_avg_missing_rank(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Rank By Avg Missing Per Scene", ""]
    columns = ["scene", "method", "n", "avg_missing_top1", "full_top1", "miss3_top1", "avg_missing_MAE"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append(
            "| {scene} | {method} | {n} | {avg} | {full} | {miss3} | {mae} |".format(
                scene=row.get("scene", ""),
                method=row.get("method", ""),
                n=row.get("n", ""),
                avg=_fmt(row.get("avg_missing_top1_mean")),
                full=_fmt(row.get("full_top1_mean")),
                miss3=_fmt(row.get("miss3_top1_mean")),
                mae=_fmt(row.get("avg_missing_MAE_mean")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stability_rank(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Rank By Scene Stability", ""]
    columns = ["method", "n", "avg_missing_top1_mean_over_scenes", "avg_missing_top1_std_over_scenes", "avg_missing_MAE_mean_over_scenes"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) if column in {"method", "n"} else _fmt(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _fields(rows: list[dict[str, Any]], first: list[str]) -> list[str]:
    fields = list(first)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _first(rows: list[dict[str, str]], key: str) -> str:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _bucket_mean(values: dict[str, float], pattern_metrics: dict[str, dict[str, Any]], count: int) -> float:
    return _mean(
        [
            value
            for pattern, value in values.items()
            if pattern != "full" and int(pattern_metrics.get(pattern, {}).get("missing_count", -1)) == count
        ]
    )


def _avg_missing(values: dict[str, float]) -> float:
    return _mean([value for pattern, value in values.items() if pattern != "full"])


def _overall_mean(values: dict[str, float]) -> float:
    nums = [_float(values.get(pattern)) for pattern in CORE_PATTERNS]
    return mean(nums) if all(_isnum(value) for value in nums) else math.nan


def _balanced(row: dict[str, Any]) -> float:
    avg = _float(row.get("avg_missing_top1"))
    radar = _float(row.get("radar_only_top1"))
    lidar = _float(row.get("lidar_only_top1"))
    if not _isnum(avg):
        return math.nan
    return avg + 0.25 * (radar if _isnum(radar) else 0.0) + 0.25 * (lidar if _isnum(lidar) else 0.0)


def _missing_count(row: dict[str, Any]) -> int:
    value = _float(row.get("missing_count"))
    if _isnum(value):
        return int(value)
    missing = [item for item in str(row.get("missing_modalities") or "").split(",") if item]
    return len(missing)


def _method_name(run_name: str) -> str:
    return re.sub(r"_seed\d+$", "", str(run_name or ""))


def _seed(run_name: str) -> str:
    match = re.search(r"_seed(\d+)$", str(run_name or ""))
    return match.group(1) if match else ""


def _scene_label(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    match = re.search(r"(\d+)", text)
    return f"Scene{int(match.group(1))}" if match else text


def _scene_sort(scene: str) -> int:
    match = re.search(r"(\d+)", str(scene))
    return int(match.group(1)) if match else 10**9


def _mean(values: list[float]) -> float:
    nums = [value for value in values if _isnum(value)]
    return mean(nums) if nums else math.nan


def _delta(value: Any, base: Any) -> float:
    left = _float(value)
    right = _float(base)
    return left - right if _isnum(left) and _isnum(right) else math.nan


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


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
    return f"{value:.5f}" if math.isfinite(value) else "NaN"


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return _fmt(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
