#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from typing import Any


POOLED_METHODS = [
    "scenes31_34_proto_natural_es40",
    "scenes31_34_proto_sampler_uniform_es40",
    "scenes31_34_proto_randomdrop_subset_es40",
    "scenes31_34_proto_randomdrop_subset_reliability_fusion_es40",
]
POOLED_COLUMNS = ["Method", "Full", "Miss-1", "Miss-2", "Miss-3", "Avg-Missing", "Within@3", "MAE", "Overall", "Main read"]
PER_SCENE_COLUMNS = [
    "Scene",
    "Natural Avg-Missing",
    "Uniform Avg-Missing",
    "Subset Avg-Missing",
    "Subset+Reliability Avg-Missing",
    "Best",
    "Subset MAE",
    "Subset+Reliability MAE",
]
STABILITY_COLUMNS = ["Method", "Mean Avg-Missing over Scenes", "Std over Scenes", "Mean MAE over Scenes", "Main read"]
LABELS = {
    "scenes31_34_proto_natural_es40": "Natural",
    "scenes31_34_proto_sampler_uniform_es40": "Uniform",
    "scenes31_34_proto_randomdrop_subset_es40": "Subset",
    "scenes31_34_proto_randomdrop_subset_reliability_fusion_es40": "Subset+Reliability",
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    export(Path(args.summary_root), Path(args.per_scene_root), Path(args.out))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Scene31-34 paper-ready result tables.")
    parser.add_argument("--summary-root", default="outputs/scenes31_34_subset_reliability_lmdb/summary")
    parser.add_argument("--per-scene-root", default="outputs/scenes31_34_subset_reliability_lmdb/per_scene_summary")
    parser.add_argument("--out", default="outputs/paper_tables/scenes31_34")
    return parser


def export(summary_root: Path, per_scene_root: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled_rows = _read_csv(summary_root / "pooled_method_mean_std.csv")
    per_scene_rows = _read_csv(per_scene_root / "per_scene_method_mean_std.csv")
    stability_rows = _read_csv(per_scene_root / "mean_over_scenes.csv")

    pooled_table = _pooled_table(pooled_rows)
    per_scene_table = _per_scene_table(per_scene_rows)
    stability_table = _stability_table(stability_rows)

    _write_csv(out_dir / "table_scenes31_34_pooled.csv", pooled_table, POOLED_COLUMNS)
    _write_md(out_dir / "table_scenes31_34_pooled.md", pooled_table, POOLED_COLUMNS)
    _write_csv(out_dir / "table_scenes31_34_per_scene.csv", per_scene_table, PER_SCENE_COLUMNS)
    _write_md(out_dir / "table_scenes31_34_per_scene.md", per_scene_table, PER_SCENE_COLUMNS)
    _write_csv(out_dir / "table_scenes31_34_scene_stability.csv", stability_table, STABILITY_COLUMNS)
    _write_md(out_dir / "table_scenes31_34_scene_stability.md", stability_table, STABILITY_COLUMNS)
    _write_notes(out_dir / "scenes31_34_paper_table_notes.txt", pooled_table, per_scene_table, stability_table)
    print(f"Wrote Scene31-34 paper tables to {out_dir}.")
    return {"pooled": pooled_table, "per_scene": per_scene_table, "stability": stability_table}


def _pooled_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_method = {row.get("method", ""): row for row in rows}
    selected = [by_method[name] for name in POOLED_METHODS if name in by_method]
    best_avg = max((_float(_metric(row, "avg_missing_top1")) for row in selected), default=math.nan)
    out: list[dict[str, str]] = []
    for row in selected:
        method = str(row.get("method", ""))
        avg = _float(_metric(row, "avg_missing_top1"))
        best = _isnum(avg) and _isnum(best_avg) and abs(avg - best_avg) < 1e-12
        out.append(
            {
                "Method": LABELS.get(method, method),
                "Full": _pct(_metric(row, "full_top1")),
                "Miss-1": _pct(_metric(row, "miss1_top1")),
                "Miss-2": _pct(_metric(row, "miss2_top1")),
                "Miss-3": _pct(_metric(row, "miss3_top1")),
                "Avg-Missing": _pct(avg),
                "Within@3": _pct(_metric(row, "avg_missing_within@3", "avg_missing_within_3")),
                "MAE": _num(_metric(row, "avg_missing_MAE")),
                "Overall": _pct(_metric(row, "overall_mean_top1")),
                "Main read": _main_read(method, best=best),
            }
        )
    return out


def _per_scene_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_scene_method = {(row.get("scene", ""), row.get("method", "")): row for row in rows}
    scenes = sorted({row.get("scene", "") for row in rows if row.get("scene")}, key=_scene_sort)
    out: list[dict[str, str]] = []
    for scene in scenes:
        values = {
            method: _float(_metric(by_scene_method.get((scene, method), {}), "avg_missing_top1"))
            for method in POOLED_METHODS
        }
        best_method = max(values, key=lambda method: values[method] if _isnum(values[method]) else -math.inf)
        subset = by_scene_method.get((scene, "scenes31_34_proto_randomdrop_subset_es40"), {})
        reliability = by_scene_method.get((scene, "scenes31_34_proto_randomdrop_subset_reliability_fusion_es40"), {})
        out.append(
            {
                "Scene": scene,
                "Natural Avg-Missing": _pct(values["scenes31_34_proto_natural_es40"]),
                "Uniform Avg-Missing": _pct(values["scenes31_34_proto_sampler_uniform_es40"]),
                "Subset Avg-Missing": _pct(values["scenes31_34_proto_randomdrop_subset_es40"]),
                "Subset+Reliability Avg-Missing": _pct(values["scenes31_34_proto_randomdrop_subset_reliability_fusion_es40"]),
                "Best": LABELS.get(best_method, best_method),
                "Subset MAE": _num(_metric(subset, "avg_missing_MAE")),
                "Subset+Reliability MAE": _num(_metric(reliability, "avg_missing_MAE")),
            }
        )
    return out


def _stability_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("method") in POOLED_METHODS]
    selected.sort(
        key=lambda row: (
            _float(row.get("avg_missing_top1_mean_over_scenes")) if _isnum(row.get("avg_missing_top1_mean_over_scenes")) else -math.inf,
            -(_float(row.get("avg_missing_top1_std_over_scenes")) if _isnum(row.get("avg_missing_top1_std_over_scenes")) else math.inf),
            -(_float(row.get("avg_missing_MAE_mean_over_scenes")) if _isnum(row.get("avg_missing_MAE_mean_over_scenes")) else math.inf),
        ),
        reverse=True,
    )
    out: list[dict[str, str]] = []
    for row in selected:
        method = str(row.get("method", ""))
        out.append(
            {
                "Method": LABELS.get(method, method),
                "Mean Avg-Missing over Scenes": _pct(row.get("avg_missing_top1_mean_over_scenes")),
                "Std over Scenes": _pct(row.get("avg_missing_top1_std_over_scenes")),
                "Mean MAE over Scenes": _num(row.get("avg_missing_MAE_mean_over_scenes")),
                "Main read": _main_read(method, best=(method == selected[0].get("method") if selected else False)),
            }
        )
    return out


def _main_read(method: str, *, best: bool) -> str:
    parts = []
    if best:
        parts.append("best Avg-Missing")
    if method == "scenes31_34_proto_randomdrop_subset_es40":
        parts.append("pooled reference")
    elif method == "scenes31_34_proto_randomdrop_subset_reliability_fusion_es40":
        parts.append("not promoted: lower Avg-Missing and higher MAE")
    elif method == "scenes31_34_proto_sampler_uniform_es40":
        parts.append("ablation")
    elif method == "scenes31_34_proto_natural_es40":
        parts.append("natural baseline")
    return "; ".join(parts)


def _write_notes(path: Path, pooled: list[dict[str, str]], per_scene: list[dict[str, str]], stability: list[dict[str, str]]) -> None:
    subset_scene_wins = sum(1 for row in per_scene if row.get("Best") == "Subset")
    lines = [
        "Scene31-34 pooled quick validation uses seed1 only.",
        "Reference = scenes31_34_proto_randomdrop_subset_es40.",
        "Reliability fusion is not promoted when Avg-Missing or MAE is worse than subset.",
        f"Subset wins {subset_scene_wins}/{len(per_scene)} individual scenes by Avg-Missing.",
        "Scene stability ranking sorts by mean Avg-Missing over scenes desc, std over scenes asc, mean MAE asc.",
    ]
    if stability:
        lines.append(f"Top stability row: {stability[0].get('Method')}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        item = dict(row)
        for key in ("Avg-Missing", "Mean Avg-Missing over Scenes"):
            if key in item and "best Avg-Missing" in item.get("Main read", ""):
                item[key] = f"**{item[key]}**"
        lines.append("| " + " | ".join(item.get(column, "") for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(row: dict[str, str], *names: str) -> float:
    for name in names:
        for key in (f"{name}_mean", name):
            if key in row and row.get(key) not in (None, ""):
                return _float(row.get(key))
    return math.nan


def _pct(value: Any) -> str:
    value = _float(value)
    return f"{value * 100:.2f}%" if _isnum(value) else ""


def _num(value: Any) -> str:
    value = _float(value)
    return f"{value:.2f}" if _isnum(value) else ""


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _scene_sort(scene: str) -> int:
    digits = "".join(ch for ch in str(scene) if ch.isdigit())
    return int(digits) if digits else 10**9


if __name__ == "__main__":
    raise SystemExit(main())
