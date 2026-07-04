#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reevaluate_apples_to_apples import DEFAULT_PATTERNS, _canonical_patterns, _evaluate_run


def main() -> int:
    args = _parser().parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = _read_manifest(Path(args.manifest))
    completed = _completed_runs(root)
    requested = _canonical_patterns(list(DEFAULT_PATTERNS))

    rows: list[dict[str, Any]] = []
    checkpoint_manifest: dict[str, Any] = {"root": str(root), "checkpoint_policy": args.checkpoint_policy, "runs": {}}
    for manifest_row in manifest_rows:
        run_name = manifest_row["run_name"]
        if completed and run_name not in completed:
            continue
        run_rows, run_manifest = _evaluate_run(
            root,
            run_name,
            requested,
            checkpoint_policy=args.checkpoint_policy,
            manual_checkpoint=None,
            split=args.split,
            max_batches=args.max_batches,
            device_override=args.device,
        )
        run_rows = _with_avg_missing(run_rows)
        for row in run_rows:
            rows.append(
                {
                    "run_name": run_name,
                    "group": manifest_row.get("group", ""),
                    "seed": manifest_row.get("seed") or row.get("seed", ""),
                    "pattern": row.get("pattern", ""),
                    "top1": row.get("top1", ""),
                    "top3": row.get("top3", ""),
                    "top5": row.get("top5", ""),
                    "adba": row.get("adba", ""),
                    "mae": row.get("mae", ""),
                    "loss": row.get("loss", ""),
                    "count": row.get("count", ""),
                    "checkpoint_path": row.get("checkpoint_path", ""),
                    "checkpoint_epoch": row.get("checkpoint_epoch", ""),
                    "status": row.get("status", ""),
                }
            )
        checkpoint_manifest["runs"][run_name] = {**run_manifest, "group": manifest_row.get("group", "")}

    _write_csv(out_dir / "night_grid_metrics.csv", rows)
    _write_markdown(out_dir / "night_grid_metrics.md", rows)
    (out_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} metric rows to {out_dir}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fresh-evaluate completed Scene31 night-grid runs.")
    parser.add_argument("--root", default="outputs/scene31")
    parser.add_argument("--manifest", default="configs/scene31/night_grid/experiment_manifest.csv")
    parser.add_argument("--checkpoint_policy", "--checkpoint-policy", default="best_val_top1")
    parser.add_argument("--out_dir", "--out-dir", default="outputs/scene31/analysis/night_grid/fresh_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_batches", "--max-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _completed_runs(root: Path) -> set[str]:
    path = root / "analysis" / "night_grid" / "completed_runs.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _with_avg_missing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pattern = {row.get("pattern"): row for row in rows}
    avg = by_pattern.get("avg_missing")
    candidates = [
        row for row in rows
        if row.get("pattern") not in {"full", "avg_missing"} and _isnum(row.get("top1"))
    ]
    if avg is not None and candidates:
        for metric in ("top1", "top3", "top5", "adba", "mae", "loss"):
            values = [_float(row.get(metric)) for row in candidates if _isnum(row.get(metric))]
            avg[metric] = f"{sum(values) / len(values):.8g}" if values else ""
        counts = [_float(row.get("count")) for row in candidates if _isnum(row.get("count"))]
        avg["count"] = int(sum(counts)) if counts else ""
        avg["status"] = "ok"
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["run_name", "group", "seed", "pattern", "top1", "top3", "top5", "adba", "mae", "loss", "count", "checkpoint_path", "checkpoint_epoch", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["run_name", "group", "seed", "pattern", "top1", "adba", "status"]
    lines = ["# Night Grid Metrics", "", "| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _isnum(value: Any) -> bool:
    number = _float(value)
    return number == number


if __name__ == "__main__":
    raise SystemExit(main())
