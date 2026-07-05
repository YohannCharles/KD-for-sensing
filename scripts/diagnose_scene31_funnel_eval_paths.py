#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Any

from scene31_eval_resolution import complete_run_names, diagnostics_row


FIELDS = (
    "run_name",
    "searched_paths",
    "actual_run_dir",
    "status_json_exists",
    "status_state",
    "config_exists",
    "config_path",
    "best_ckpt_exists",
    "best_ckpt_path",
    "last_ckpt_exists",
    "last_ckpt_path",
    "checkpoint_used",
    "checkpoint_path",
    "best_epoch",
    "best_val_acc",
    "warnings",
    "diagnosis",
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root)
    runs = args.runs or complete_run_names(root)
    rows = [diagnostics_row(root, run_name, experiment_group=args.group) for run_name in runs]
    out_path = Path(args.out) if args.out else root / "eval_path_diagnostics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_path, rows)
    counts = _counts(rows)
    print(f"wrote {out_path}")
    print(
        " ".join(
            f"{name}={counts.get(name, 0)}"
            for name in ("ok", "missing_run_dir", "missing_config", "missing_checkpoint", "last_checkpoint_fallback")
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose Scene31 funnel fresh-eval run/config/checkpoint paths.")
    parser.add_argument("--root", default="outputs/scene31_funnel_lmdb")
    parser.add_argument("--out", default="")
    parser.add_argument("--group", default=None)
    parser.add_argument("--runs", nargs="*", default=[])
    return parser


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("diagnosis") or "")
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
