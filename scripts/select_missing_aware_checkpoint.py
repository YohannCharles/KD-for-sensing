#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from kd_sensing.eval.missing_buckets import BUCKET_COUNTS, bucket_metric_mean, missing_bucket_mapping_from_rows, write_missing_bucket_mapping
from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name

RULES = ("best_full_val", "best_avg_missing_val", "best_mixed_val", "best_bucket_balanced_val")
DEFAULT_PATTERNS = (
    "full",
    "missing_gps",
    "missing_radar",
    "missing_lidar",
    "missing_image",
    "non_gps_only",
    "gps_only",
    "image_only",
    "radar_only",
    "lidar_only",
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out or root / "checkpoint_selection")
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if args.split == "test":
        warnings.append("selection split is test; this is not allowed for final selection")

    explicit_metrics = _read_metric_files([Path(item) for item in args.metrics])
    per_epoch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for run in args.runs:
        run_rows, run_warnings = _run_selection(root, out_dir, run, args, explicit_metrics)
        warnings.extend(run_warnings)
        per_epoch_rows.extend(run_rows)
        summary_rows.extend(_selection_summary(out_dir, run, run_rows, args))

    mapping, mapping_warnings = missing_bucket_mapping_from_rows(per_epoch_rows)
    warnings.extend(mapping_warnings)
    _write_csv(out_dir / "checkpoint_selection_per_epoch.csv", per_epoch_rows, _per_epoch_fields())
    _write_csv(out_dir / "checkpoint_selection_summary.csv", summary_rows, _summary_fields())
    write_missing_bucket_mapping(out_dir / "missing_bucket_mapping.json", mapping)
    if warnings:
        (out_dir / "checkpoint_selection_warnings.txt").write_text("\n".join(sorted(dict.fromkeys(warnings))) + "\n", encoding="utf-8")
    print(f"Wrote checkpoint selection summary to {out_dir}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select Scene31 checkpoints with missing-aware val metrics.")
    parser.add_argument("--root", default="outputs/scene31_funnel_lmdb")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--metrics", action="append", default=[], help="Per-checkpoint metrics CSV. Can be repeated.")
    parser.add_argument("--rules", nargs="*", default=list(RULES), choices=RULES)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--split", default="val", choices=("val", "validation", "test"))
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--evaluate", action="store_true", help="Run lightweight val eval for checkpoints missing metrics.")
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink selected checkpoints.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _run_selection(
    root: Path,
    out_dir: Path,
    run: str,
    args: argparse.Namespace,
    explicit_metrics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    checkpoints = _checkpoint_candidates(root, run)
    if not checkpoints:
        return [], [f"{run}: no checkpoint candidates found"]
    if len(checkpoints) == 1:
        warnings.append(f"{run}: only one checkpoint candidate found; selection is degenerate")

    metric_rows = [row for row in explicit_metrics if row.get("run_name") in {run, ""}]
    metric_rows.extend(_read_metric_files(_default_metric_paths(root, out_dir, run)))
    out_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        epoch = _checkpoint_epoch(checkpoint)
        rows = _rows_for_epoch(metric_rows, run, epoch)
        if not rows and args.evaluate:
            eval_dir = out_dir / "eval_cache" / run / f"epoch_{epoch or checkpoint.stem}"
            rows = _evaluate_checkpoint(root, run, checkpoint, eval_dir, args)
            metric_rows.extend(rows)
        metrics = _metrics_from_rows(rows)
        if not rows:
            metrics["full_top1"] = _metadata_metric(checkpoint)
        item = {
            "run": run,
            "checkpoint": str(checkpoint),
            "selected_epoch": epoch if epoch is not None else "",
            "metric_source": _metric_source(rows),
            "split": args.split,
            **metrics,
        }
        for rule in args.rules:
            item[f"score_{rule}"] = _score(item, rule, args.alpha)
        out_rows.append(item)
    return out_rows, warnings


def _selection_summary(out_dir: Path, run: str, rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rule in args.rules:
        valid = [row for row in rows if _isnum(row.get(f"score_{rule}"))]
        if not valid:
            out.append({"run": run, "rule": rule, "selected_epoch": "", "score": "", "warning": "no finite score"})
            continue
        selected = max(valid, key=lambda row: (_float(row.get(f"score_{rule}")), _float(row.get("selected_epoch"))))
        _link_or_copy(Path(selected["checkpoint"]), out_dir / run / "selected_checkpoints" / rule / "best.ckpt", copy=bool(args.copy))
        out.append(
            {
                "run": run,
                "rule": rule,
                "selected_epoch": selected.get("selected_epoch", ""),
                "checkpoint": selected.get("checkpoint", ""),
                "full_top1": selected.get("full_top1", ""),
                "miss1_top1": selected.get("miss1_top1", ""),
                "miss2_top1": selected.get("miss2_top1", ""),
                "miss3_top1": selected.get("miss3_top1", ""),
                "avg_missing_top1": selected.get("avg_missing_top1", ""),
                "score": selected.get(f"score_{rule}", ""),
                "warning": "",
            }
        )
    return out


def _checkpoint_candidates(root: Path, run: str) -> list[Path]:
    paths: list[Path] = []
    for run_dir in (root / run, root / "scene31" / run):
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            paths.extend(sorted(ckpt_dir.glob("*.pth")))
            paths.extend(sorted(ckpt_dir.glob("*.pt")))
            paths.extend(sorted(ckpt_dir.glob("*.ckpt")))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen and path.exists():
            seen.add(resolved)
            out.append(path)
    return sorted(out, key=lambda path: (_checkpoint_epoch(path) if _checkpoint_epoch(path) is not None else 10**9, path.name))


def _default_metric_paths(root: Path, out_dir: Path, run: str) -> list[Path]:
    return [
        root / run / "checkpoint_selection_per_epoch.csv",
        root / run / "checkpoint_selection" / "checkpoint_selection_per_epoch.csv",
        out_dir / run / "checkpoint_selection_per_epoch.csv",
    ]


def _read_metric_files(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["metrics_path"] = str(path)
                rows.append(row)
    return rows


def _rows_for_epoch(rows: list[dict[str, Any]], run: str, epoch: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("run_name") not in {run, ""} and row.get("run") not in {run, ""}:
            continue
        row_epoch = _int_value(row.get("checkpoint_epoch") or row.get("selected_epoch") or row.get("epoch"))
        if epoch is None or row_epoch == epoch:
            out.append(row)
    return out


def _evaluate_checkpoint(root: Path, run: str, checkpoint: Path, out_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "reevaluate_apples_to_apples.py"),
        "--root",
        str(root),
        "--runs",
        run,
        "--checkpoint-policy",
        "manual_path",
        "--manual-checkpoint",
        str(checkpoint),
        "--out-dir",
        str(out_dir),
        "--split",
        args.split,
        "--max-batches",
        str(args.max_batches),
        "--eval-patterns",
        *DEFAULT_PATTERNS,
    ]
    subprocess.run(cmd, check=False)
    return _read_metric_files([out_dir / "apples_to_apples_metrics.csv"])


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_pattern: dict[str, float] = {}
    for row in rows:
        if row.get("status") not in {"", "ok", None}:
            continue
        pattern = canonical_missing_pattern_name(str(row.get("pattern") or ""))
        value = _float(row.get("top1") or row.get("top1_val") or row.get("value"))
        if _isnum(value):
            by_pattern[pattern] = value
    mapping, _ = missing_bucket_mapping_from_rows(rows)
    metrics: dict[str, Any] = {
        "full_top1": by_pattern.get("full", float("nan")),
        "avg_missing_top1": by_pattern.get("avg_missing", _avg_missing(by_pattern)),
    }
    for count in BUCKET_COUNTS:
        metrics[f"miss{count}_top1"] = bucket_metric_mean(by_pattern, mapping, count)
    return metrics


def _score(row: dict[str, Any], rule: str, alpha: float) -> float:
    full = _float(row.get("full_top1"))
    avg = _float(row.get("avg_missing_top1"))
    if rule == "best_full_val":
        return full
    if rule == "best_avg_missing_val":
        return avg
    if rule == "best_mixed_val":
        return float(alpha) * full + (1.0 - float(alpha)) * avg if _isnum(full) and _isnum(avg) else float("nan")
    values = [_float(row.get(key)) for key in ("full_top1", "miss1_top1", "miss2_top1", "miss3_top1")]
    values = [value for value in values if _isnum(value)]
    return sum(values) / len(values) if values else float("nan")


def _link_or_copy(source: Path, target: Path, *, copy: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if copy:
        shutil.copy2(source, target)
    else:
        rel_source = os.path.relpath(source.resolve(), target.parent.resolve())
        try:
            target.symlink_to(rel_source)
        except OSError:
            shutil.copy2(source, target)


def _checkpoint_epoch(path: Path) -> int | None:
    metadata = _metadata(path)
    for key in ("epoch", "selected_epoch", "best_top1_epoch", "best_early_stopping_epoch"):
        value = _int_value(metadata.get(key))
        if value is not None:
            return value
    match = re.search(r"(?:epoch|ep|ckpt)[_-]?(\d+)", path.stem)
    if match:
        return int(match.group(1))
    return _int_value(path.stem)


def _metadata_metric(path: Path) -> float:
    metadata = _metadata(path)
    for key in ("metric_value", "best_metric", "best_val_top1", "val_top1", "primary_acc"):
        value = _float(metadata.get(key))
        if _isnum(value):
            return value
    return float("nan")


def _metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _metric_source(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "checkpoint_metadata"
    return str(rows[0].get("metrics_path") or "metrics_rows")


def _avg_missing(values: dict[str, float]) -> float:
    candidates = [value for pattern, value in values.items() if pattern not in {"full", "avg_missing"} and _isnum(value)]
    return sum(candidates) / len(candidates) if candidates else float("nan")


def _per_epoch_fields() -> list[str]:
    return [
        "run",
        "checkpoint",
        "selected_epoch",
        "metric_source",
        "split",
        "full_top1",
        "miss1_top1",
        "miss2_top1",
        "miss3_top1",
        "avg_missing_top1",
        *[f"score_{rule}" for rule in RULES],
    ]


def _summary_fields() -> list[str]:
    return [
        "run",
        "rule",
        "selected_epoch",
        "checkpoint",
        "full_top1",
        "miss1_top1",
        "miss2_top1",
        "miss3_top1",
        "avg_missing_top1",
        "score",
        "warning",
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()})


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_value(value: Any) -> int | None:
    number = _float(value)
    return int(number) if math.isfinite(number) else None


def _isnum(value: Any) -> bool:
    return math.isfinite(_float(value))


def _fmt(value: Any) -> str:
    number = _float(value)
    return f"{number:.8g}" if math.isfinite(number) else ""


if __name__ == "__main__":
    raise SystemExit(main())
