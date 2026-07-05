#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_EVAL_PATTERNS = {
    "full",
    "avg_missing",
    "missing_gps",
    "missing_radar",
    "radar_only",
    "lidar_only",
    "missing_gps_image",
    "missing_gps_radar",
    "missing_gps_lidar",
    "missing_image_radar",
    "missing_image_lidar",
    "missing_radar_lidar",
}
REQUIRED_EVAL_METRICS = ("top1", "top3", "top5", "within_3", "mae")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shared checks for Scene31 local/manual runners.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest-value")
    manifest_parser.add_argument("manifest")
    manifest_parser.add_argument("run_name")
    manifest_parser.add_argument("key")

    train_parser = subparsers.add_parser("train-complete")
    train_parser.add_argument("root")
    train_parser.add_argument("run_name")
    train_parser.add_argument("--strict-status-checkpoint", action="store_true")

    eval_parser = subparsers.add_parser("eval-complete")
    eval_parser.add_argument("run_dir")
    eval_parser.add_argument("--require-manifest", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "manifest-value":
        value = manifest_value(Path(args.manifest), args.run_name, args.key)
        if value is None:
            return 1
        print(value)
        return 0
    if args.command == "train-complete":
        return 0 if train_complete(Path(args.root), args.run_name, strict_status_checkpoint=args.strict_status_checkpoint) else 1
    if args.command == "eval-complete":
        return 0 if eval_complete(Path(args.run_dir), require_manifest=args.require_manifest) else 1
    raise AssertionError(args.command)


def manifest_value(manifest: Path, run_name: str, key: str) -> str | None:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_name") == run_name:
                return row.get(key, "")
    return None


def train_complete(root: Path, run_name: str, *, strict_status_checkpoint: bool = False) -> bool:
    run_dirs = [root / run_name, root / "scene31" / run_name]
    run_dirs.extend(sorted(root.glob(f"scenegroup_*/{run_name}")))
    for run_dir in run_dirs:
        status_complete = False
        status_path = run_dir / "run_status.json"
        if status_path.exists():
            try:
                if json.loads(status_path.read_text(encoding="utf-8")).get("state") == "complete":
                    status_complete = True
            except json.JSONDecodeError:
                pass
        checkpoint_dir = run_dir / "checkpoints"
        has_checkpoint = checkpoint_dir.exists() and any(
            any(checkpoint_dir.glob(pattern)) for pattern in ("*.pth", "*.pt", "*.ckpt")
        )
        has_checkpoint = has_checkpoint or any((run_dir / name).exists() for name in ("best.pth", "best_top1.pth", "last.pth"))
        if strict_status_checkpoint and status_complete and has_checkpoint:
            return True
        if not strict_status_checkpoint and (status_complete or has_checkpoint):
            return True
    return False


def eval_complete(run_dir: Path, *, require_manifest: bool = False) -> bool:
    metrics = run_dir / "apples_to_apples_metrics.csv"
    manifest = run_dir / "checkpoint_manifest.json"
    if not metrics.exists():
        return False
    if require_manifest and not manifest.exists():
        return False
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if data.get("max_batches") not in (None, ""):
            return False

    with metrics.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_pattern = {row.get("pattern"): row for row in rows}
    if not REQUIRED_EVAL_PATTERNS <= set(by_pattern):
        return False
    for pattern in REQUIRED_EVAL_PATTERNS:
        row = by_pattern[pattern]
        if row.get("status") not in ("", "ok"):
            return False
        for metric in REQUIRED_EVAL_METRICS:
            try:
                value = float(row.get(metric, "nan"))
            except ValueError:
                return False
            if not math.isfinite(value):
                return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
