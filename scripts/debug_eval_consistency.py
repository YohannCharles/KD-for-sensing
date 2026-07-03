#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any

from kd_sensing.config.io import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.evaluation_pass import run_evaluation_pass
from kd_sensing.engine.optim import build_device, build_model, build_task_criterion
from kd_sensing.eval.missing_patterns import (
    canonical_missing_pattern_name,
    get_missing_pattern_mask,
    list_standard_missing_patterns,
    resolve_missing_patterns,
)
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import evaluate_missing_matrix
from kd_sensing.evaluation.horizon_selection import aggregate_topk_and_dba
from kd_sensing.utils.checkpoint import load_model_state
from kd_sensing.utils.checkpoint_resolver import resolve_checkpoint

DEFAULT_PATTERNS = ("full", "avg_missing", "missing_gps", "radar_only")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = _canonical_patterns(args.patterns or DEFAULT_PATTERNS)
    report = _build_report(
        root=root,
        run_name=args.run,
        checkpoint=args.checkpoint,
        patterns=patterns,
        split=args.split,
        max_batches=args.max_batches,
        device_override=args.device,
    )
    (out_dir / "eval_consistency_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "eval_consistency_report.md").write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug val_acc vs fresh missing-pattern evaluation consistency.")
    parser.add_argument("--root", default="outputs/scene31")
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--patterns", nargs="*", default=list(DEFAULT_PATTERNS))
    parser.add_argument("--out_dir", "--out-dir", default="outputs/scene31/analysis/eval_consistency_debug")
    parser.add_argument("--split", default="test", choices=("test", "val", "validation"))
    parser.add_argument("--max_batches", "--max-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def _build_report(
    *,
    root: Path,
    run_name: str,
    checkpoint: str | None,
    patterns: list[str],
    split: str,
    max_batches: int | None,
    device_override: str | None,
) -> dict[str, Any]:
    cfg, cfg_path, config_warnings = _load_run_config(root, run_name)
    resolution = resolve_checkpoint(root, run_name, "best_val_top1", manual_path=checkpoint)
    report: dict[str, Any] = {
        "run_name": run_name,
        "root": str(root),
        "config_path": str(cfg_path) if cfg_path else "",
        "checkpoint_path": str(resolution.path) if resolution.path else "",
        "checkpoint_epoch": resolution.epoch if resolution.epoch is not None else "",
        "checkpoint_resolution": resolution.as_dict(),
        "requested_patterns": patterns,
        "warnings": [*config_warnings, *resolution.warnings],
    }
    if cfg is None:
        report["status"] = "missing_config"
        report["summary"] = {"status": "missing_config"}
        return report
    if resolution.path is None:
        report["status"] = "missing_checkpoint"
        report["summary"] = {"status": "missing_checkpoint"}
        return report

    cfg.setdefault("output", {})["run_name"] = run_name
    if device_override:
        cfg.setdefault("experiment", {})["device"] = device_override
    device = build_device(cfg)
    dataloaders = build_dataloaders(cfg)
    split_key = _resolve_split(dataloaders, split)
    loader = dataloaders[split_key]
    model = build_model(cfg["model"]["primary"]).to(device)
    load_result = load_model_state(
        resolution.path,
        model,
        role="eval-consistency-debug",
        map_location=device,
        strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
    )
    checkpoint_payload = load_result.get("checkpoint", {})
    checkpoint_epoch = resolution.epoch or _checkpoint_epoch(checkpoint_payload, resolution.path)
    modalities = list(cfg.get("model", {}).get("primary", {}).get("modalities") or ["image", "radar", "gps", "lidar"])
    criterion = build_task_criterion(cfg)
    official_metrics = run_evaluation_pass(model, loader, cfg, criterion, device).metrics
    official_top1 = _float(aggregate_topk_and_dba(official_metrics).get("top1"))
    eval_names = _evaluation_pattern_names(patterns, modalities)
    forward_patterns = resolve_missing_patterns(eval_names, modalities)
    results = evaluate_missing_matrix(
        model,
        loader,
        device,
        modalities,
        patterns=forward_patterns,
        random_missing=None,
        prediction_index=cfg.get("evaluation", {}).get("missing_patterns", {}).get("prediction_index", "last"),
        max_batches=max_batches,
        cfg=cfg,
    )
    by_pattern = {canonical_missing_pattern_name(row.get("pattern", "")): row for row in results}
    val_acc = _val_acc_from_metrics(root / run_name / "metrics.csv", checkpoint_epoch)
    full_top1 = _float(_value(by_pattern.get("full", {}), "top1"))
    avg_top1 = _float(_value(by_pattern.get("avg_missing", {}), "top1"))
    diff = abs(val_acc - full_top1) if _isnum(val_acc) and _isnum(full_top1) else float("nan")
    official_diff = abs(val_acc - official_top1) if _isnum(val_acc) and _isnum(official_top1) else float("nan")
    official_full_diff = abs(official_top1 - full_top1) if _isnum(official_top1) and _isnum(full_top1) else float("nan")
    if _isnum(diff) and diff > 0.03:
        report["warnings"].append(f"abs(val_acc - full_top1)={diff:.6g} > 0.03")
    if _isnum(official_diff) and official_diff > 0.03:
        report["warnings"].append(f"abs(val_acc - official_top1)={official_diff:.6g} > 0.03")
    if _isnum(official_full_diff) and official_full_diff > 0.03:
        report["warnings"].append(f"abs(official_top1 - full_top1)={official_full_diff:.6g} > 0.03")

    eval_dataset = _dataset_info(loader)
    training_split = _training_validation_split(cfg, resolution.metadata)
    if not eval_dataset.get("path"):
        eval_dataset["path"] = str(training_split.get("csv_path") or training_split.get("path") or "")
    split_note = _split_consistency_note(eval_dataset, training_split)
    if split_note.startswith("different"):
        report["warnings"].append(split_note)

    pattern_rows = []
    for pattern in patterns:
        row = by_pattern.get(pattern, {})
        pattern_rows.append(
            {
                "pattern": pattern,
                "standard_mask": _mask_text(pattern),
                "model_mask": _mask_text(pattern, modalities),
                "top1": _value(row, "top1"),
                "top3": _value(row, "top3"),
                "top5": _value(row, "top5"),
                "adba": _value(row, "adba"),
                "mae": _value(row, "mae"),
                "loss": _value(row, "loss"),
                "count": _value(row, "count", "sample_count", "num_samples"),
            }
        )

    report.update(
        {
            "status": "ok",
            "checkpoint_epoch": checkpoint_epoch,
            "eval_split": split_key,
            "eval_dataset": eval_dataset,
            "training_validation_split": training_split,
            "split_consistency_note": split_note,
            "modalities": modalities,
            "standard_modalities": ["gps", "image", "radar", "lidar"],
            "metrics_csv_val_acc": _format(val_acc),
            "fresh_official_top1": _format(official_top1),
            "fresh_full_top1": _format(full_top1),
            "fresh_avg_missing_top1": _format(avg_top1),
            "val_acc_full_top1_abs_diff": _format(diff),
            "val_acc_official_top1_abs_diff": _format(official_diff),
            "official_full_top1_abs_diff": _format(official_full_diff),
            "full_mask_is_1111": _mask_text("full") == "1,1,1,1",
            "radar_only_mask_is_0010": _mask_text("radar_only") == "0,0,1,0",
            "patterns": pattern_rows,
            "summary": {
                "status": "ok",
                "checkpoint_path": str(resolution.path),
                "checkpoint_epoch": checkpoint_epoch,
                "config_path": str(cfg_path) if cfg_path else "",
                "eval_split": split_key,
                "eval_dataset_path": eval_dataset.get("path", ""),
                "metrics_csv_val_acc": _format(val_acc),
                "fresh_official_top1": _format(official_top1),
                "fresh_full_top1": _format(full_top1),
                "fresh_avg_missing_top1": _format(avg_top1),
                "warning_count": len(report["warnings"]),
            },
        }
    )
    return report


def _load_run_config(root: Path, run_name: str) -> tuple[dict[str, Any] | None, Path | None, list[str]]:
    candidates = [
        root / run_name / "final_config.yaml",
        root / run_name / "resolved_config.yaml",
        Path("configs/scene31") / f"{run_name}.yaml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if "configs" in path.parts:
            return load_config(path), path, []
        data = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
        return data, path, []
    return None, None, ["missing_config"]


def _evaluation_pattern_names(requested_patterns: list[str], modalities: list[str]) -> list[str]:
    names = [name for name in requested_patterns if name != "avg_missing"]
    if "avg_missing" in requested_patterns:
        names.extend(name for name in list_standard_missing_patterns(modalities) if name != "full")
    return list(dict.fromkeys(names))


def _val_acc_from_metrics(path: Path, epoch: int | str | None) -> float:
    if not path.exists():
        return float("nan")
    rows = _read_csv(path)
    target_epoch = int(epoch) if _float(epoch) == _float(epoch) else None
    if target_epoch is not None:
        for row in rows:
            row_epoch = _float(row.get("epoch"))
            if _isnum(row_epoch) and int(row_epoch) == target_epoch:
                value = _first_float(row, "val_acc", "val_beam_top1", "top1_acc", "accuracy/top1")
                if _isnum(value):
                    return value
    values = [_first_float(row, "val_acc", "val_beam_top1", "top1_acc", "accuracy/top1") for row in rows]
    values = [value for value in values if _isnum(value)]
    return max(values) if values else float("nan")


def _dataset_info(loader: Any) -> dict[str, Any]:
    dataset = getattr(loader, "dataset", None)
    return {
        "class": type(dataset).__name__ if dataset is not None else "",
        "path": str(getattr(dataset, "csv_path", "") or getattr(dataset, "path", "")),
        "split": str(getattr(dataset, "split", "")),
        "num_samples": len(dataset) if dataset is not None and hasattr(dataset, "__len__") else "",
    }


def _training_validation_split(cfg: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    split_metadata = metadata.get("split_metadata") if isinstance(metadata, dict) else {}
    if isinstance(split_metadata, dict):
        for key in ("validation", "val", "test"):
            if key in split_metadata:
                item = split_metadata[key]
                return item if isinstance(item, dict) else {"split": key, "value": item}
    validation = cfg.get("training", {}).get("validation", {})
    return validation if isinstance(validation, dict) else {}


def _split_consistency_note(eval_dataset: dict[str, Any], training_split: dict[str, Any]) -> str:
    eval_path = str(eval_dataset.get("path", ""))
    train_path = str(training_split.get("csv_path") or training_split.get("path") or "")
    if eval_path and train_path and Path(eval_path) != Path(train_path):
        return f"different eval split path: fresh={eval_path}, metrics_sidecar={train_path}"
    if eval_path and train_path:
        return "fresh eval split path matches checkpoint split metadata"
    return "split comparison unavailable"


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Eval Consistency Report",
        "",
        f"- run: `{report.get('run_name', '')}`",
        f"- checkpoint: `{report.get('checkpoint_path', '')}`",
        f"- checkpoint_epoch: `{report.get('checkpoint_epoch', '')}`",
        f"- config: `{report.get('config_path', '')}`",
        f"- eval_split: `{report.get('eval_split', '')}`",
        f"- eval_dataset_path: `{(report.get('eval_dataset') or {}).get('path', '')}`",
        f"- split_consistency: `{report.get('split_consistency_note', '')}`",
        f"- metrics_csv_val_acc: `{report.get('metrics_csv_val_acc', '')}`",
        f"- fresh_official_top1: `{report.get('fresh_official_top1', '')}`",
        f"- fresh_full_top1: `{report.get('fresh_full_top1', '')}`",
        f"- fresh_avg_missing_top1: `{report.get('fresh_avg_missing_top1', '')}`",
        f"- full mask is `[1,1,1,1]`: `{report.get('full_mask_is_1111', '')}`",
        f"- radar_only mask is `[0,0,1,0]`: `{report.get('radar_only_mask_is_0010', '')}`",
        "",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["## Warnings", "", *[f"- {item}" for item in warnings], ""])
    lines.extend(
        [
            "## Patterns",
            "",
            "| pattern | standard_mask | model_mask | top1 | top3 | top5 | adba | mae | loss | count |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report.get("patterns", []):
        lines.append(
            "| {pattern} | {standard_mask} | {model_mask} | {top1} | {top3} | {top5} | {adba} | {mae} | {loss} | {count} |".format(
                **{key: row.get(key, "") for key in ("pattern", "standard_mask", "model_mask", "top1", "top3", "top5", "adba", "mae", "loss", "count")}
            )
        )
    return "\n".join(lines) + "\n"


def _canonical_patterns(patterns: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(canonical_missing_pattern_name(item) for item in patterns))


def _resolve_split(dataloaders: dict[str, Any], split: str) -> str:
    candidates = ("validation", "val", "test") if split in {"val", "validation"} else (split,)
    for candidate in candidates:
        if candidate in dataloaders:
            return candidate
    raise ValueError(f"Requested split '{split}' is unavailable. Available: {sorted(dataloaders)}")


def _checkpoint_epoch(checkpoint: Any, checkpoint_path: Path) -> int | str:
    if isinstance(checkpoint, dict):
        for key in ("epoch", "best_top1_epoch", "best_early_stopping_epoch"):
            value = checkpoint.get(key)
            if _float(value) == _float(value):
                return int(_float(value))
    sidecar = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    data = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    for key in ("selected_epoch", "epoch"):
        value = _float(data.get(key))
        if value == value:
            return int(value)
    return ""


def _mask_text(pattern: str, modalities: list[str] | tuple[str, ...] | None = None) -> str:
    if pattern == "avg_missing":
        return ""
    try:
        return ",".join(str(int(value)) for value in get_missing_pattern_mask(pattern, modalities))
    except ValueError:
        return ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _first_float(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _float(row.get(key))
        if _isnum(value):
            return value
    return float("nan")


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _isnum(value: float) -> bool:
    return value == value


def _format(value: float) -> str:
    return f"{value:.8g}" if _isnum(value) else ""


if __name__ == "__main__":
    raise SystemExit(main())
