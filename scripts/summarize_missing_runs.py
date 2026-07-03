#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean

from kd_sensing.config.io import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.eval.missing_patterns import canonical_missing_pattern_name
from kd_sensing.utils.checkpoint_resolver import resolve_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize rbma/weighted-sum missing-modality runs.")
    parser.add_argument("--root", default="outputs/scene31", help="Scene output root.")
    parser.add_argument("--target-epochs", type=int, default=40)
    parser.add_argument("--expected_epochs", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--strict_completed", default="false", choices=("false", "true"))
    args = parser.parse_args(argv)
    root = Path(args.root)
    expected_epochs = int(args.expected_epochs or args.target_epochs)
    manifest_rows = _read_csv(Path(args.manifest)) if args.manifest else []
    if manifest_rows:
        overall = [_manifest_overall_row(root, row, expected_epochs, args.strict_completed == "true") for row in manifest_rows]
        runs = [root / row["run_name"] for row in manifest_rows if (root / row["run_name"]).exists()]
    else:
        runs = _run_dirs(root)
        overall = [_overall_row(root, run, expected_epochs, args.strict_completed == "true") for run in runs]
    missing = [row for run in runs for row in _missing_rows(root, run.name)]
    timing = [_timing_row(root, run.name) for run in runs]
    _write_csv(root / "summary_overall.csv", overall)
    _write_csv(root / "summary_missing_patterns.csv", missing)
    _write_csv(root / "summary_timing.csv", timing)
    return 0


def _manifest_overall_row(root: Path, manifest_row: dict, expected_epochs: int, strict_completed: bool) -> dict:
    run_name = manifest_row.get("run_name", "")
    run = root / run_name
    if not run.exists():
        return {
            "run_name": run_name,
            "group": manifest_row.get("group", ""),
            "status": "missing",
            "best_epoch": "",
            "final_epoch": "",
            "best_val_acc": "",
            "best_val_adba": "",
            "best_checkpoint": "",
            "log_path": str(_log_path(root, run_name) or ""),
            "exit_code": _exit_code(root, run_name),
            "expected_epochs": manifest_row.get("expected_epochs") or expected_epochs,
        }
    row = _overall_row(
        root,
        run,
        int(manifest_row.get("expected_epochs") or expected_epochs),
        strict_completed,
    )
    row["group"] = manifest_row.get("group", "")
    return row


def _run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and ((path / "metrics.json").exists() or (path / "metrics.csv").exists()))


def _overall_row(root: Path, run: Path, expected_epochs: int, strict_completed: bool) -> dict:
    run_expected_epochs = _expected_epochs_for_run(root, run, expected_epochs)
    metrics = _read_json(run / "metrics.json") or {}
    latest = metrics.get("latest", {}) if isinstance(metrics, dict) else {}
    epoch_logs = metrics.get("epoch_logs", []) if isinstance(metrics, dict) else []
    csv_rows = _read_csv(run / "metrics.csv") if (run / "metrics.csv").exists() else []
    all_metric_rows = [*epoch_logs, *csv_rows]
    completed_epochs = max(_max_epoch(csv_rows), _max_epoch(epoch_logs), int(latest.get("epoch") or 0))
    checkpoint_resolution = resolve_checkpoint(root, run.name, "best_val_top1")
    best_checkpoint = checkpoint_resolution.path
    final_eval = root / "eval" / f"{run.name}_missing_patterns.csv"
    log_status = _log_status(root, run.name)
    checkpoint_epoch = checkpoint_resolution.epoch or _checkpoint_epoch(root, run.name)
    early_stop = _early_stop_info(root, run, metrics, epoch_logs, csv_rows, final_eval_exists=final_eval.exists(), log_status=log_status)
    status = _completed_status(
        completed_epochs=completed_epochs,
        expected_epochs=run_expected_epochs,
        final_eval_exists=final_eval.exists(),
        log_status=log_status,
        checkpoint_epoch=checkpoint_epoch,
        has_checkpoint=best_checkpoint is not None,
        early_stopped=early_stop["early_stopped"],
        strict_completed=strict_completed,
    )
    log_path = _log_path(root, run.name)
    return {
        "run_name": run.name,
        "exp_name": run.name,
        "best_epoch": checkpoint_epoch or _best_epoch(all_metric_rows),
        "best_val_acc": _best_value(all_metric_rows, "val_acc", max),
        "best_val_adba": _best_value(all_metric_rows, "val_adba", max),
        "best_val_loss": _best_value(all_metric_rows, "val_loss", min),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else "",
        "checkpoint_resolver_source": checkpoint_resolution.source,
        "checkpoint_resolver_warnings": "; ".join(checkpoint_resolution.warnings),
        "completed_epochs": completed_epochs,
        "final_epoch": completed_epochs,
        "checkpoint_epoch": checkpoint_epoch or "",
        "final_eval": str(final_eval) if final_eval.exists() else "",
        "log_path": str(log_path) if log_path else "",
        "log_status": log_status,
        "exit_code": _exit_code(root, run.name),
        "status": status,
        "early_stopped": str(bool(early_stop["early_stopped"])).lower(),
        "early_stop_epoch": early_stop["early_stop_epoch"],
        "early_stop_metric": early_stop["early_stop_metric"],
        "expected_epochs": run_expected_epochs,
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
                "pattern": canonical_missing_pattern_name(row.get("pattern", "")),
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
    return resolve_checkpoint(root, exp_name, "best_val_top1").path


def _completed_status(
    *,
    completed_epochs: int,
    expected_epochs: int,
    final_eval_exists: bool,
    log_status: str,
    checkpoint_epoch: int,
    has_checkpoint: bool,
    early_stopped: bool,
    strict_completed: bool,
) -> str:
    if log_status == "killed_or_failed":
        return "killed_or_failed"
    if completed_epochs < expected_epochs and early_stopped:
        return "completed_early_stopped"
    if completed_epochs >= expected_epochs:
        if strict_completed and not final_eval_exists:
            return "completed_missing_final_eval"
        return "completed" if final_eval_exists or not strict_completed else "completed_missing_final_eval"
    if final_eval_exists or log_status == "exit0" or checkpoint_epoch >= expected_epochs:
        return "completed"
    if has_checkpoint:
        return "incomplete_has_checkpoint"
    return "killed_or_failed"


def _early_stop_info(
    root: Path,
    run: Path,
    metrics: dict,
    epoch_logs: list[dict],
    csv_rows: list[dict],
    *,
    final_eval_exists: bool,
    log_status: str,
) -> dict[str, str | bool]:
    rows = [*epoch_logs, *csv_rows]
    for row in reversed(rows):
        if _truthy(row.get("early_stopped")) or row.get("early_stop_epoch") not in (None, ""):
            return {
                "early_stopped": True,
                "early_stop_epoch": str(row.get("early_stop_epoch") or row.get("epoch") or ""),
                "early_stop_metric": str(row.get("early_stop_metric") or row.get("early_stopping_metric") or ""),
            }
    train_log = _read_json(run / "train_log.json") or {}
    train_rows = train_log.get("epoch_logs", []) if isinstance(train_log, dict) else []
    for row in reversed(train_rows if isinstance(train_rows, list) else []):
        if isinstance(row, dict) and (_truthy(row.get("early_stopped")) or row.get("early_stop_epoch") not in (None, "")):
            return {
                "early_stopped": True,
                "early_stop_epoch": str(row.get("early_stop_epoch") or row.get("epoch") or ""),
                "early_stop_metric": str(row.get("early_stop_metric") or row.get("early_stopping_metric") or ""),
            }
    run_status = _read_json(run / "run_status.json") or {}
    status_text = str(run_status.get("status") or run_status.get("state") or "").lower() if isinstance(run_status, dict) else ""
    if status_text == "early_stopped":
        return {"early_stopped": True, "early_stop_epoch": str(_max_epoch(rows)), "early_stop_metric": ""}
    log_text = _combined_log_text(root, run.name)
    if "early stopping triggered" in log_text.lower():
        return {"early_stopped": True, "early_stop_epoch": str(_max_epoch(rows)), "early_stop_metric": ""}
    if final_eval_exists and log_status != "killed_or_failed":
        return {"early_stopped": True, "early_stop_epoch": str(_max_epoch(rows)), "early_stop_metric": ""}
    return {"early_stopped": False, "early_stop_epoch": "", "early_stop_metric": ""}


def _max_epoch(rows: list[dict]) -> int:
    values = [int(_float(row.get("epoch"))) for row in rows if _float(row.get("epoch")) == _float(row.get("epoch"))]
    return max(values) if values else 0


def _log_status(root: Path, exp_name: str) -> str:
    text = _combined_log_text(root, exp_name)
    lowered = text.lower()
    if re.search(r"(sigkill|killed|exit code[:= ]+[1-9]|exit[:= ]+[1-9])", lowered):
        return "killed_or_failed"
    if re.search(r"(exit code[:= ]+0|exit[:= ]+0)", lowered):
        return "exit0"
    return ""


def _exit_code(root: Path, exp_name: str) -> str:
    text = _combined_log_text(root, exp_name).lower()
    match = re.search(r"(?:exit code|exit)[:= ]+(-?\d+)", text)
    if match:
        return match.group(1)
    status = _read_json(root / exp_name / "run_status.json") or {}
    state = str(status.get("state", "")).lower() if isinstance(status, dict) else ""
    if state == "complete":
        return "0"
    return ""


def _log_path(root: Path, exp_name: str) -> Path | None:
    candidates = []
    candidates.extend(path for path in Path("logs").glob(f"**/*{exp_name}*") if path.is_file())
    if (root / "logs").exists():
        candidates.extend(path for path in (root / "logs").glob(f"{exp_name}*") if path.is_file())
    preferred = [path for path in candidates if path.suffix == ".log"]
    pool = preferred or candidates
    return max(pool, key=lambda path: path.stat().st_mtime) if pool else None


def _combined_log_text(root: Path, exp_name: str) -> str:
    paths = list((root / "logs").glob(f"{exp_name}*")) if (root / "logs").exists() else []
    paths += list(Path("logs").glob(f"**/*{exp_name}*"))
    return "\n".join(_safe_read(path)[-4000:] for path in paths if path.is_file())


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


def _expected_epochs_for_run(root: Path, run: Path, fallback: int) -> int:
    for path in (run / "final_config.yaml", run / "resolved_config.yaml"):
        data = _read_yaml(path)
        value = _float(data.get("training", {}).get("epochs") if isinstance(data.get("training"), dict) else None)
        if value == value:
            return int(value)
    config_path = Path("configs/scene31") / f"{run.name}.yaml"
    if config_path.exists():
        try:
            cfg = load_config(config_path)
            value = _float(cfg.get("training", {}).get("epochs"))
            if value == value:
                return int(value)
        except Exception:
            pass
    return int(fallback)


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


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
    preferred = [
        "run_name",
        "group",
        "status",
        "best_epoch",
        "final_epoch",
        "best_val_acc",
        "best_val_adba",
        "best_checkpoint",
        "log_path",
        "exit_code",
    ]
    keys = sorted({key for row in rows for key in row})
    columns = [key for key in preferred if key in keys] + [key for key in keys if key not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
