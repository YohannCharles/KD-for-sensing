#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_ROOT = "outputs/bprr_reliability_router_v1"
DEFAULT_BASELINE_ROOT = "outputs/pcpg_radar_balance_v1"
SUMMARY_FIELDS = [
    "experiment",
    "seed",
    "full",
    "drop1",
    "drop2",
    "drop3",
    "drop1_3_mean",
    "avg_missing",
    "image_only",
    "lidar_only",
    "radar_only",
    "missing_image",
    "within3",
    "MAE",
    "selection_metric",
    "best_epoch",
    "clean_val_acc",
    "gate_entropy",
    "mean_gate_image",
    "mean_gate_lidar",
    "mean_gate_radar",
    "mean_gate_gps",
    "radar_gate_when_available",
    "radar_gate_missing_image",
    "radar_gate_drop3",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize BPRR reliability-router v1 local outputs.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--baseline_root", "--baseline-root", default=DEFAULT_BASELINE_ROOT)
    args = parser.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    pattern_rows = collect_pattern_rows(root)
    gate_rows = collect_gate_rows(root)
    summary_rows = summarize_runs(pattern_rows, gate_rows, root)
    drop_rows = drop_count_summary(pattern_rows)
    oracle_rows = [row for row in pattern_rows if _truthy(row.get("oracle_gate")) or "oracle" in str(row.get("source_path", ""))]

    baseline_rows = summarize_runs(collect_pattern_rows(Path(args.baseline_root)), collect_gate_rows(Path(args.baseline_root)), Path(args.baseline_root))
    _write_csv(root / "summary.csv", summary_rows, fields=SUMMARY_FIELDS)
    _write_csv(root / "drop_count_summary.csv", drop_rows)
    _write_csv(root / "gate_diagnostics.csv", gate_rows)
    _write_csv(root / "oracle_summary.csv", oracle_rows)
    (root / "summary.md").write_text(render_markdown(summary_rows, baseline_rows), encoding="utf-8")
    print(f"Wrote BPRR reliability-router summary to {root}")
    return 0


def collect_pattern_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    wanted = {"eval_matrix.csv", "oracle_eval_matrix.csv", "pattern_metrics.csv"}
    for path in sorted(root.rglob("*.csv")):
        if path.name not in wanted and not path.name.endswith("_missing_patterns.csv"):
            continue
        for row in _read_csv(path):
            run_name = str(row.get("run_name") or _run_name_from_path(path, root))
            experiment, seed = _experiment_seed(run_name, path, root)
            if not seed and row.get("seed"):
                seed = str(row.get("seed"))
            rows.append(
                {
                    **row,
                    "run_name": run_name,
                    "experiment": experiment,
                    "seed": seed,
                    "source_path": str(path),
                    "oracle_gate": row.get("oracle_gate", "true" if "oracle" in path.name else "false"),
                }
            )
    return rows


def collect_gate_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    wanted = {"reliability_weights_epoch.csv", "pcpg_gate_diagnostics.csv", "gate_diagnostics.csv"}
    for path in sorted(root.rglob("*.csv")):
        if path.name not in wanted:
            continue
        run_name = _run_name_from_path(path, root)
        experiment, seed = _experiment_seed(run_name, path, root)
        for row in _read_csv(path):
            rows.append({**row, "run_name": run_name, "experiment": experiment, "seed": seed, "source_path": str(path)})
    return rows


def summarize_runs(pattern_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    by_run: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pattern_rows:
        by_run[(str(row.get("experiment")), str(row.get("seed")))].append(row)
    gate_summary = summarize_gate_rows(gate_rows)
    rows: list[dict[str, Any]] = []
    for (experiment, seed), items in sorted(by_run.items()):
        values = {str(row.get("pattern") or row.get("pattern_name")): row for row in items}
        drop1 = _mean_patterns(values, missing_count=1)
        drop2 = _mean_patterns(values, missing_count=2)
        drop3 = _mean_patterns(values, missing_count=3)
        avg_missing = _top1(values.get("avg_missing"))
        if not _isnum(avg_missing):
            avg_missing = _mean([drop1, drop2, drop3])
        info = selection_info(root, experiment, seed)
        gate = gate_summary.get((experiment, seed), {})
        rows.append(
            {
                "experiment": experiment,
                "seed": seed,
                "full": _top1(values.get("full")),
                "drop1": drop1,
                "drop2": drop2,
                "drop3": drop3,
                "drop1_3_mean": _mean([drop1, drop2, drop3]),
                "avg_missing": avg_missing,
                "image_only": _top1(values.get("image_only")),
                "lidar_only": _top1(values.get("lidar_only")),
                "radar_only": _top1(values.get("radar_only")),
                "missing_image": _top1(values.get("missing_image")),
                "within3": _metric(values.get("avg_missing"), "within_3", "within3"),
                "MAE": _metric(values.get("avg_missing"), "mae", "MAE"),
                "selection_metric": info.get("selection_metric", ""),
                "best_epoch": info.get("best_epoch", ""),
                "clean_val_acc": info.get("clean_val_acc", ""),
                **gate,
            }
        )
    return rows


def summarize_gate_rows(gate_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in gate_rows:
        grouped[(str(row.get("experiment")), str(row.get("seed")))].append(row)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        result: dict[str, Any] = {}
        mean_gates: dict[str, float] = {}
        for modality in ("image", "lidar", "radar", "gps"):
            values = [_float(row.get("mean_weight")) for row in rows if str(row.get("modality")) == modality]
            mean_gates[modality] = _mean(values)
            result[f"mean_gate_{modality}"] = mean_gates[modality]
        result["gate_entropy"] = _mean([_float(row.get("gate_entropy")) for row in rows])
        if not _isnum(result["gate_entropy"]):
            vals = [value for value in mean_gates.values() if _isnum(value) and value > 0]
            total = sum(vals)
            probs = [value / total for value in vals] if total > 0 else []
            result["gate_entropy"] = -sum(p * math.log(max(p, 1e-8)) for p in probs) if probs else math.nan
        radar_rows = [row for row in rows if str(row.get("modality")) == "radar"]
        result["radar_gate_when_available"] = _mean(
            [_float(row.get("mean_weight")) for row in radar_rows if _float(row.get("available_rate")) > 0]
        )
        result["radar_gate_missing_image"] = _mean(
            [_float(row.get("mean_weight")) for row in radar_rows if str(row.get("pattern")) == "missing_image"]
        )
        result["radar_gate_drop3"] = _mean(
            [_float(row.get("mean_weight")) for row in radar_rows if _missing_count(str(row.get("pattern")), None, []) == 3]
        )
        out[key] = result
    return out


def selection_info(root: Path, experiment: str, seed: str) -> dict[str, Any]:
    run_dir = root / experiment / f"seed{seed}"
    sidecars = [
        run_dir / "checkpoints" / "best_avg_missing_top1.pth.json",
        run_dir / "checkpoints" / "best_top1.pth.json",
        run_dir / "checkpoints" / "best.pth.json",
    ]
    for path in sidecars:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "selection_metric": payload.get("selection_metric", ""),
            "best_epoch": payload.get("selected_epoch", payload.get("epoch", "")),
            "clean_val_acc": (payload.get("task_metrics") or {}).get("val_acc", ""),
        }
    metrics = run_dir / "metrics.json"
    if metrics.exists():
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        latest = payload.get("latest", {}) if isinstance(payload, dict) else {}
        return {
            "selection_metric": latest.get("checkpoint_selection_metric", ""),
            "best_epoch": latest.get("best_early_stopping_epoch", ""),
            "clean_val_acc": latest.get("val_acc", ""),
        }
    return {}


def drop_count_summary(pattern_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pattern_rows:
        count = _missing_count(str(row.get("pattern") or row.get("pattern_name")), row.get("mask"), _modalities(row))
        if count is None:
            continue
        grouped[(str(row.get("experiment")), str(row.get("seed")), int(count))].append(row)
    out = []
    for (experiment, seed, count), rows in sorted(grouped.items()):
        out.append(
            {
                "experiment": experiment,
                "seed": seed,
                "missing_count": count,
                "missing_ratio": count / max(len(_modalities(rows[0])), 1),
                "top1": _mean([_top1(row) for row in rows]),
                "within3": _mean([_metric(row, "within_3", "within3") for row in rows]),
                "MAE": _mean([_metric(row, "mae", "MAE") for row in rows]),
                "num_patterns": len(rows),
                "num_samples": sum(int(float(row.get("num_samples", row.get("sample_count", 0)) or 0)) for row in rows),
            }
        )
    return out


def render_markdown(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> str:
    baseline = {str(row.get("experiment")): row for row in baseline_rows}
    e5 = _first_baseline(baseline, "e5")
    e6 = _first_baseline(baseline, "e6")
    oracle = next((row for row in rows if str(row.get("experiment")) == "e3_oracle_gate_eval"), None)
    lines = [
        "# BPRR Reliability Router V1 Summary",
        "",
        "| experiment | seed | full | avg_missing | drop3 | radar_gate | delta_vs_e5 | delta_vs_e6 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        avg = _float(row.get("avg_missing"))
        lines.append(
            f"| {row.get('experiment')} | {row.get('seed')} | {_fmt(row.get('full'))} | {_fmt(avg)} | "
            f"{_fmt(row.get('drop3'))} | {_fmt(row.get('mean_gate_radar'))} | "
            f"{_fmt(avg - _float(e5.get('avg_missing')) if e5 else math.nan)} | "
            f"{_fmt(avg - _float(e6.get('avg_missing')) if e6 else math.nan)} |"
        )
    lines.extend(["", "## Checks", ""])
    lines.append(f"- e5 baseline delta source: {'available' if e5 else 'missing'}.")
    lines.append(f"- e6 robustness-first delta source: {'available' if e6 else 'missing'}.")
    raw = next((row for row in rows if row.get("experiment") == "e7_raw_confidence_gate"), None)
    bprr = next((row for row in rows if row.get("experiment") == "e8_bprr_calibrated_router"), None)
    lines.append(f"- raw_conf_gate vs BPRR avg_missing delta: {_fmt(_float((bprr or {}).get('avg_missing')) - _float((raw or {}).get('avg_missing')))}.")
    lines.append(f"- oracle upper bound source: {'available' if oracle else 'missing'}.")
    lines.append("- Gate collapse check uses mean_gate_* and radar_gate_when_available columns.")
    lines.append("- Hard subset / JEPA split: compare e9/e10/e11/e12 after runs complete.")
    return "\n".join(lines) + "\n"


def _first_baseline(rows: dict[str, dict[str, Any]], prefix: str) -> dict[str, Any] | None:
    for key, row in rows.items():
        if key.startswith(prefix):
            return row
    return None


def _mean_patterns(values: dict[str, dict[str, Any]], *, missing_count: int) -> float:
    return _mean([_top1(row) for name, row in values.items() if _missing_count(name, row.get("mask"), _modalities(row)) == missing_count])


def _missing_count(pattern: str, mask: Any, modalities: list[str]) -> int | None:
    if isinstance(mask, str) and mask and mask not in {"aggregate"} and not mask.startswith("random_"):
        values = [item.strip() for item in mask.split(",") if item.strip()]
        if values and all(item in {"0", "1"} for item in values):
            return values.count("0")
    name = str(pattern)
    if name == "full":
        return 0
    if name == "avg_missing" or name.startswith("random_"):
        return None
    if name.endswith("_only"):
        return max(len(modalities), 4) - 1
    if name == "non_gps_only":
        return 1
    if name.startswith("missing_"):
        return len([item for item in name.removeprefix("missing_").split("_") if item])
    if name == "miss3":
        return 3
    return None


def _top1(row: dict[str, Any] | None) -> float:
    return _metric(row, "top1", "full_top1", "avg_missing_top1")


def _metric(row: dict[str, Any] | None, *keys: str) -> float:
    if row is None:
        return math.nan
    for key in keys:
        value = _float(row.get(key))
        if _isnum(value):
            return value
    return math.nan


def _modalities(row: dict[str, Any]) -> list[str]:
    raw = row.get("modalities")
    if isinstance(raw, str) and raw:
        return [item for item in raw.replace(",", "|").split("|") if item]
    return ["image", "radar", "gps", "lidar"]


def _run_name_from_path(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    if len(parts) >= 2 and parts[1].startswith("seed"):
        return f"{parts[0]}/{parts[1]}"
    if path.parent.name.startswith("seed") and path.parent.parent != root:
        return f"{path.parent.parent.name}/{path.parent.name}"
    return parts[0] if parts else path.parent.name


def _experiment_seed(run_name: str, path: Path, root: Path) -> tuple[str, str]:
    parts = str(run_name).split("/")
    if len(parts) >= 2 and parts[1].startswith("seed"):
        return parts[0], parts[1].removeprefix("seed")
    rel = path.relative_to(root).parts
    if len(rel) >= 2 and rel[1].startswith("seed"):
        return rel[0], rel[1].removeprefix("seed")
    marker = "_seed"
    if marker in run_name:
        exp, seed = run_name.rsplit(marker, 1)
        return exp, seed
    return run_name, ""


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fieldnames = fields or sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def _isnum(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: list[Any]) -> float:
    parsed = [_float(value) for value in values]
    valid = [value for value in parsed if _isnum(value)]
    return mean(valid) if valid else math.nan


def _fmt(value: Any) -> str:
    numeric = _float(value)
    return "n/a" if not _isnum(numeric) else f"{numeric:.4f}"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    raise SystemExit(main())
