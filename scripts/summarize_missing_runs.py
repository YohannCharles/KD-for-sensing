#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize rbma/weighted-sum missing-modality runs.")
    parser.add_argument("--root", default="outputs/scene31", help="Scene output root.")
    parser.add_argument("--target-epochs", type=int, default=40)
    parser.add_argument("--expected_epochs", type=int, default=None)
    parser.add_argument("--strict_completed", default="false", choices=("false", "true"))
    args = parser.parse_args(argv)
    root = Path(args.root)
    expected_epochs = int(args.expected_epochs or args.target_epochs)
    runs = _run_dirs(root)
    overall = [_overall_row(root, run, expected_epochs, args.strict_completed == "true") for run in runs]
    missing = [row for run in runs for row in _missing_rows(root, run.name)]
    timing = [_timing_row(root, run.name) for run in runs]
    _write_csv(root / "summary_overall.csv", overall)
    _write_csv(root / "summary_missing_patterns.csv", missing)
    _write_csv(root / "summary_timing.csv", timing)
    return 0


def _run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and ((path / "metrics.json").exists() or (path / "metrics.csv").exists()))


def _overall_row(root: Path, run: Path, expected_epochs: int, strict_completed: bool) -> dict:
    metrics = _read_json(run / "metrics.json") or {}
    latest = metrics.get("latest", {}) if isinstance(metrics, dict) else {}
    epoch_logs = metrics.get("epoch_logs", []) if isinstance(metrics, dict) else []
    csv_rows = _read_csv(run / "metrics.csv") if (run / "metrics.csv").exists() else []
    completed_epochs = max(_max_epoch(csv_rows), _max_epoch(epoch_logs), int(latest.get("epoch") or 0))
    best_checkpoint = _best_checkpoint(root, run.name)
    final_eval = root / "eval" / f"{run.name}_missing_patterns.csv"
    log_status = _log_status(root, run.name)
    checkpoint_epoch = _checkpoint_epoch(root, run.name)
    status = _completed_status(
        completed_epochs=completed_epochs,
        expected_epochs=expected_epochs,
        final_eval_exists=final_eval.exists(),
        log_status=log_status,
        checkpoint_epoch=checkpoint_epoch,
        has_checkpoint=best_checkpoint is not None,
        strict_completed=strict_completed,
    )
    return {
        "exp_name": run.name,
        "best_epoch": _best_epoch(epoch_logs),
        "best_val_acc": _best_value(epoch_logs, "val_acc", max),
        "best_val_adba": _best_value(epoch_logs, "val_adba", max),
        "best_val_loss": _best_value(epoch_logs, "val_loss", min),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
        "completed_epochs": completed_epochs,
        "checkpoint_epoch": checkpoint_epoch or "",
        "final_eval": str(final_eval) if final_eval.exists() else "",
        "log_status": log_status,
        "status": status,
    }


def _missing_rows(root: Path, exp_name: str) -> list[dict]:
    path = root / "eval" / f"{exp_name}_missing_patterns.csv"
    if not path.exists():
        return []
    rows = []
    for row in _read_csv(path):
        rows.append(
            {
                "exp_name": exp_name,
                "pattern": row.get("pattern", ""),
                "top1": row.get("top1", ""),
                "top3": row.get("top3", ""),
                "top5": row.get("top5", ""),
                "adba": row.get("adba", ""),
                "loss": row.get("loss", ""),
                "mae": row.get("mae", ""),
                "count": row.get("count") or row.get("sample_count") or row.get("num_samples", ""),
            }
        )
    return rows


def _timing_row(root: Path, exp_name: str) -> dict:
    path = root / "logs" / f"{exp_name}_timing.csv"
    rows = _read_csv(path) if path.exists() else []
    data = [_float(row.get("data_time")) for row in rows]
    step = [_float(row.get("step_time")) for row in rows]
    return {
        "exp_name": exp_name,
        "avg_data_time": _avg(data),
        "p95_data_time": _p95(data),
        "max_data_time": _max(data),
        "avg_step_time": _avg(step),
        "p95_step_time": _p95(step),
        "max_step_time": _max(step),
        "max_gpu_mem_reserved_mb": _max(_float(row.get("gpu_mem_reserved_mb")) for row in rows),
        "max_cpu_rss_mb": _max(_float(row.get("cpu_rss_mb")) for row in rows),
        "slow_batch_count": sum(1 for row in rows if str(row.get("slow_batch", "")).lower() == "true"),
    }


def _best_checkpoint(root: Path, exp_name: str) -> Path | None:
    local = root / exp_name / "checkpoints" / "best_top1.pth"
    if local.exists():
        return local
    registry = sorted((root / "best_checkpoints").glob(f"{exp_name}*.pth")) if (root / "best_checkpoints").exists() else []
    return registry[-1] if registry else None


def _completed_status(
    *,
    completed_epochs: int,
    expected_epochs: int,
    final_eval_exists: bool,
    log_status: str,
    checkpoint_epoch: int,
    has_checkpoint: bool,
    strict_completed: bool,
) -> str:
    if log_status == "killed_or_failed" and completed_epochs < expected_epochs:
        return "killed_or_failed"
    if completed_epochs >= expected_epochs:
        if strict_completed and not final_eval_exists:
            return "completed_missing_final_eval"
        return "completed" if final_eval_exists or not strict_completed else "completed_missing_final_eval"
    if final_eval_exists or log_status == "exit0" or checkpoint_epoch >= expected_epochs:
        return "completed"
    if has_checkpoint:
        return "incomplete_has_checkpoint"
    if log_status == "killed_or_failed":
        return "killed_or_failed"
    return "killed_or_incomplete"


def _max_epoch(rows: list[dict]) -> int:
    values = [int(_float(row.get("epoch"))) for row in rows if _float(row.get("epoch")) == _float(row.get("epoch"))]
    return max(values) if values else 0


def _log_status(root: Path, exp_name: str) -> str:
    paths = list((root / "logs").glob(f"{exp_name}*")) if (root / "logs").exists() else []
    paths += list(Path("logs").glob(f"**/*{exp_name}*"))
    text = "\n".join(_safe_read(path)[-4000:] for path in paths if path.is_file())
    lowered = text.lower()
    if re.search(r"(sigkill|killed|exit code[:= ]+[1-9]|exit[:= ]+[1-9])", lowered):
        return "killed_or_failed"
    if re.search(r"(exit code[:= ]+0|exit[:= ]+0)", lowered):
        return "exit0"
    return ""


def _checkpoint_epoch(root: Path, exp_name: str) -> int:
    candidates = list((root / exp_name / "checkpoints").glob("*.pth"))
    candidates += list((root / "best_checkpoints").glob(f"{exp_name}*.pth")) if (root / "best_checkpoints").exists() else []
    epochs = []
    for path in candidates:
        epochs.extend(int(match) for match in re.findall(r"(?:epoch|ep)[_-]?(\d+)", path.name.lower()))
        sidecar = path.with_suffix(path.suffix + ".json")
        data = _read_json(sidecar) or {}
        for key in ("epoch", "selected_epoch"):
            value = _float(data.get(key))
            if value == value:
                epochs.append(int(value))
    return max(epochs) if epochs else 0


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _best_epoch(epoch_logs: list[dict]) -> str:
    if not epoch_logs:
        return ""
    best = max(epoch_logs, key=lambda row: _float(row.get("val_acc")))
    return str(best.get("epoch", ""))


def _best_value(epoch_logs: list[dict], key: str, fn) -> str:
    values = [_float(row.get(key)) for row in epoch_logs]
    values = [value for value in values if value == value]
    return "" if not values else f"{fn(values):.8g}"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _avg(values) -> str:
    values = [value for value in values if value == value]
    return "" if not values else f"{mean(values):.8g}"


def _max(values) -> str:
    values = [value for value in values if value == value]
    return "" if not values else f"{max(values):.8g}"


def _p95(values) -> str:
    values = sorted(value for value in values if value == value)
    if not values:
        return ""
    index = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
    return f"{values[index]:.8g}"


if __name__ == "__main__":
    raise SystemExit(main())
