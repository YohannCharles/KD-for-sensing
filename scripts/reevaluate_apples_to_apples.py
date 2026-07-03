#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from kd_sensing.config.io import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.eval.missing_patterns import (
    canonical_missing_pattern_name,
    get_missing_pattern_mask,
    list_standard_missing_patterns,
    resolve_missing_patterns,
)
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import evaluate_missing_matrix
from kd_sensing.utils.checkpoint import load_model_state
from kd_sensing.utils.checkpoint_resolver import resolve_checkpoint

SCRIPT_VERSION = "apples_to_apples_v1"
DEFAULT_PATTERNS = (
    "full",
    "avg_missing",
    "missing_gps",
    "non_gps_only",
    "gps_only",
    "image_only",
    "radar_only",
    "lidar_only",
    "missing_image",
    "missing_radar",
    "missing_lidar",
)
METRICS = ("top1", "top3", "top5", "within_3", "adba", "mae", "loss")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = _canonical_patterns(args.eval_patterns or DEFAULT_PATTERNS)
    manual = _manual_checkpoints(args.manual_checkpoint)

    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "root": str(root),
        "checkpoint_policy": args.checkpoint_policy,
        "split": args.split,
        "max_batches": args.max_batches,
        "runs": {},
    }
    for run_name in args.runs:
        run_rows, run_manifest = _evaluate_run(
            root,
            run_name,
            requested,
            checkpoint_policy=args.checkpoint_policy,
            manual_checkpoint=manual.get(run_name) or (manual.get("*") if len(args.runs) == 1 else None),
            split=args.split,
            max_batches=args.max_batches,
            device_override=args.device,
        )
        rows.extend(run_rows)
        manifest["runs"][run_name] = run_manifest

    delta_rows = _delta_rows(rows, args.baseline_name)
    _write_csv(out_dir / "apples_to_apples_metrics.csv", rows)
    _write_markdown(out_dir / "apples_to_apples_metrics.md", rows, args.baseline_name)
    _write_csv(out_dir / "apples_to_apples_delta.csv", delta_rows)
    (out_dir / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_conclusions(rows, baseline=args.baseline_name)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-evaluate Scene31 runs with one checkpoint and missing-pattern policy.")
    parser.add_argument("--root", default="outputs/scene31")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--eval_patterns", "--eval-patterns", nargs="*", default=list(DEFAULT_PATTERNS))
    parser.add_argument(
        "--checkpoint_policy",
        "--checkpoint-policy",
        choices=("best_val_top1", "latest", "best_avg_missing_top1", "manual_path", "best_epoch_from_metrics"),
        default="best_val_top1",
    )
    parser.add_argument("--manual_checkpoint", "--manual-checkpoint", action="append", default=[], help="RUN=PATH, or PATH when one run is supplied.")
    parser.add_argument("--baseline_name", "--baseline-name", default="main_v3_strong_reliability_proto")
    parser.add_argument("--out_dir", "--out-dir", default="outputs/scene31/analysis/apples_to_apples")
    parser.add_argument("--split", default="test", choices=("test", "val", "validation"))
    parser.add_argument("--max_batches", "--max-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def _evaluate_run(
    root: Path,
    run_name: str,
    requested_patterns: list[str],
    *,
    checkpoint_policy: str,
    manual_checkpoint: str | None,
    split: str,
    max_batches: int | None,
    device_override: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg, cfg_path, config_warnings = _load_run_config(root, run_name)
    resolution = resolve_checkpoint(root, run_name, checkpoint_policy, manual_path=manual_checkpoint)
    ckpt_path = resolution.path
    manifest = {
        "config_path": str(cfg_path) if cfg_path else "",
        "config_hash": _config_hash(cfg),
        "checkpoint_path": str(ckpt_path) if ckpt_path else "",
        "checkpoint_epoch": resolution.epoch if resolution.epoch is not None else "",
        "checkpoint_policy": checkpoint_policy,
        "checkpoint_resolution": resolution.as_dict(),
        "warnings": [*config_warnings, *resolution.warnings],
        "script_version": SCRIPT_VERSION,
        "split_requested": split,
        "max_batches": max_batches,
    }
    if cfg is None:
        print(f"[WARN] {run_name}: missing config")
        manifest["status"] = "missing_config"
        return _missing_rows(run_name, requested_patterns, status="missing_config", max_batches=max_batches), manifest
    if ckpt_path is None:
        print(f"[WARN] {run_name}: missing checkpoint for policy {checkpoint_policy}")
        manifest["status"] = "missing_checkpoint"
        return _missing_rows(run_name, requested_patterns, status="missing_checkpoint", max_batches=max_batches), manifest

    cfg.setdefault("output", {})["run_name"] = run_name
    if device_override:
        cfg.setdefault("experiment", {})["device"] = device_override
    try:
        device = build_device(cfg)
        dataloaders = build_dataloaders(cfg)
        split_key = _resolve_split(dataloaders, split)
        model = build_model(cfg["model"]["primary"]).to(device)
        load_result = load_model_state(
            ckpt_path,
            model,
            role="apples-to-apples",
            map_location=device,
            strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
        )
        checkpoint = load_result.get("checkpoint", {})
        checkpoint_epoch = resolution.epoch or _checkpoint_epoch(checkpoint, ckpt_path)
        seed = _seed(cfg, checkpoint)
        modalities = list(cfg.get("model", {}).get("primary", {}).get("modalities") or ["image", "radar", "lidar", "gps"])
        eval_names = _evaluation_pattern_names(requested_patterns, modalities)
        patterns = resolve_missing_patterns(eval_names, modalities)
        results = evaluate_missing_matrix(
            model,
            dataloaders[split_key],
            device,
            modalities,
            patterns=patterns,
            random_missing=None,
            prediction_index=cfg.get("evaluation", {}).get("missing_patterns", {}).get("prediction_index", "last"),
            max_batches=max_batches,
            cfg=cfg,
        )
        by_pattern = {canonical_missing_pattern_name(row.get("pattern", "")): row for row in results}
        rows = [
            _metrics_row(
                run_name,
                pattern,
                by_pattern.get(pattern, {}),
                checkpoint_path=ckpt_path,
                checkpoint_epoch=checkpoint_epoch,
                seed=seed,
                config_hash=manifest["config_hash"],
                status="ok",
                model_modalities=modalities,
                max_batches=max_batches,
            )
            for pattern in requested_patterns
        ]
        _fill_avg_missing(rows, by_pattern)
        manifest.update(
            {
                "status": "ok",
                "checkpoint_epoch": checkpoint_epoch,
                "seed": seed,
                "split": split_key,
                "modalities": modalities,
            }
        )
        return rows, manifest
    except Exception as exc:
        print(f"[WARN] {run_name}: evaluation failed: {exc}")
        manifest["status"] = "eval_failed"
        manifest["warnings"].append(str(exc))
        return _missing_rows(
            run_name,
            requested_patterns,
            status="eval_failed",
            checkpoint_path=ckpt_path,
            max_batches=max_batches,
        ), manifest


def _load_run_config(root: Path, run_name: str) -> tuple[dict[str, Any] | None, Path | None, list[str]]:
    candidates = [
        root / run_name / "final_config.yaml",
        root / run_name / "resolved_config.yaml",
        Path("configs/scene31") / f"{run_name}.yaml",
    ]
    warnings: list[str] = []
    for path in candidates:
        if not path.exists():
            continue
        if "configs" in path.parts:
            return load_config(path), path, warnings
        data = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
        return data, path, warnings
    warnings.append("missing_config")
    return None, None, warnings


def _select_checkpoint(root: Path, run_name: str, policy: str, manual_checkpoint: str | None) -> tuple[Path | None, list[str]]:
    resolution = resolve_checkpoint(root, run_name, policy, manual_path=manual_checkpoint)
    return resolution.path, resolution.warnings


def _evaluation_pattern_names(requested_patterns: list[str], modalities: list[str]) -> list[str]:
    names = [name for name in requested_patterns if name != "avg_missing"]
    if "avg_missing" in requested_patterns:
        names.extend(name for name in list_standard_missing_patterns(modalities) if name != "full")
    return list(dict.fromkeys(names))


def _metrics_row(
    run_name: str,
    pattern: str,
    source: dict[str, Any],
    *,
    checkpoint_path: Path | str | None,
    checkpoint_epoch: int | str = "",
    seed: int | str = "",
    config_hash: str = "",
    status: str,
    model_modalities: list[str] | tuple[str, ...] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    standard_mask = _mask_text(pattern)
    model_mask = _mask_text(pattern, model_modalities) if model_modalities else ""
    return {
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path or ""),
        "checkpoint_epoch": checkpoint_epoch,
        "seed": seed,
        "eval_script_version": SCRIPT_VERSION,
        "config_hash": config_hash,
        "status": status,
        "max_batches": "" if max_batches is None else max_batches,
        "pattern": pattern,
        "standard_mask": standard_mask,
        "model_mask": model_mask,
        "top1": _value(source, "top1"),
        "top3": _value(source, "top3"),
        "top5": _value(source, "top5"),
        "within_3": _value(source, "within_3"),
        "adba": _value(source, "adba"),
        "mae": _value(source, "mae"),
        "loss": _value(source, "loss"),
        "count": _value(source, "count", "sample_count", "num_samples"),
    }


def _missing_rows(
    run_name: str,
    requested_patterns: list[str],
    *,
    status: str,
    checkpoint_path: Path | str | None = None,
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    return [
        _metrics_row(
            run_name,
            pattern,
            {},
            checkpoint_path=checkpoint_path,
            status=status,
            max_batches=max_batches,
        )
        for pattern in requested_patterns
    ]


def _fill_avg_missing(rows: list[dict[str, Any]], by_pattern: dict[str, dict[str, Any]]) -> None:
    avg_row = next((row for row in rows if row.get("pattern") == "avg_missing"), None)
    if avg_row is None:
        return
    source = [row for name, row in by_pattern.items() if name not in {"full", "avg_missing"}]
    for metric in METRICS:
        values = [_float(row.get(metric)) for row in source if _isnum(_float(row.get(metric)))]
        avg_row[metric] = _format(sum(values) / len(values)) if values else ""
    counts = [_float(row.get("count") or row.get("sample_count") or row.get("num_samples")) for row in source]
    counts = [value for value in counts if _isnum(value)]
    avg_row["count"] = int(sum(counts)) if counts else ""


def _delta_rows(rows: list[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    baseline_values = {
        (row["pattern"], metric): _float(row.get(metric))
        for row in rows
        if row.get("run_name") == baseline
        for metric in METRICS
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        run_name = row.get("run_name")
        if run_name == baseline:
            continue
        for metric in METRICS:
            value = _float(row.get(metric))
            base = baseline_values.get((row["pattern"], metric), float("nan"))
            out.append(
                {
                    "run_name": run_name,
                    "baseline_name": baseline,
                    "pattern": row["pattern"],
                    "metric": metric,
                    "value": _format(value),
                    "baseline_value": _format(base),
                    "delta": _format(value - base) if _isnum(value) and _isnum(base) else "",
                }
            )
    return out


def _write_markdown(path: Path, rows: list[dict[str, Any]], baseline: str) -> None:
    lines = ["# Apples-to-Apples Metrics", "", f"Baseline: `{baseline}`", ""]
    columns = ["run_name", "pattern", "top1", "top3", "top5", "within_3", "adba", "mae", "loss", "count", "status"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_conclusions(rows: list[dict[str, Any]], *, baseline: str) -> None:
    for label, pattern, metric in (
        ("best full top1 run", "full", "top1"),
        ("best avg_missing top1 run", "avg_missing", "top1"),
        ("best missing_gps top1 run", "missing_gps", "top1"),
        ("best non_gps_only top1 run", "non_gps_only", "top1"),
        ("best radar_only top1 run", "radar_only", "top1"),
        ("best avg_missing ADBA run", "avg_missing", "adba"),
    ):
        print(f"{label}: {_best_label(rows, pattern, metric)}")
    tau1 = "main_v3_strong_reliability_btapa_tau1"
    tau1_avg = _metric(rows, tau1, "avg_missing", "top1")
    base_avg = _metric(rows, baseline, "avg_missing", "top1")
    print(f"BTAPA tau1 exceeds proto baseline avg_missing top1: {_yes_no(tau1_avg, base_avg)}")
    old_v3_radar = _metric(rows, baseline, "radar_only", "top1")
    proto_radar = _metric(rows, "proto_baseline", "radar_only", "top1")
    if _isnum(old_v3_radar) and _isnum(proto_radar):
        delta = proto_radar - old_v3_radar
        print(f"old V3 vs current proto baseline radar_only top1 delta: {delta:.8g}")
        print(f"old V3 and current proto baseline numbers consistent: {abs(delta) <= 0.05}")
        if abs(delta) > 0.05:
            print("[WARN] radar_only metric mismatch is large; check pattern construction or checkpoint selection")
    else:
        print("old V3 and current proto baseline numbers consistent: unavailable")


def _best_label(rows: list[dict[str, Any]], pattern: str, metric: str) -> str:
    valid = [row for row in rows if row.get("pattern") == pattern and _isnum(_float(row.get(metric)))]
    if not valid:
        return "unavailable"
    row = max(valid, key=lambda item: _float(item.get(metric)))
    return f"{row['run_name']} {metric}={row.get(metric)}"


def _metric(rows: list[dict[str, Any]], run_name: str, pattern: str, metric: str) -> float:
    for row in rows:
        if row.get("run_name") == run_name and row.get("pattern") == pattern:
            return _float(row.get(metric))
    return float("nan")


def _yes_no(value: float, baseline: float) -> str:
    if not _isnum(value) or not _isnum(baseline):
        return "unavailable"
    return "yes" if value > baseline else "no"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _manual_checkpoints(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" in value:
            run, path = value.split("=", 1)
            out[run] = path
        else:
            out["*"] = value
    return out


def _canonical_patterns(patterns: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(canonical_missing_pattern_name(item) for item in patterns))


def _mask_text(pattern: str, modalities: list[str] | tuple[str, ...] | None = None) -> str:
    if pattern == "avg_missing":
        return ""
    try:
        return ",".join(str(int(value)) for value in get_missing_pattern_mask(pattern, modalities))
    except ValueError:
        return ""


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


def _seed(cfg: dict[str, Any], checkpoint: Any) -> int | str:
    value = cfg.get("experiment", {}).get("seed")
    if value is None and isinstance(checkpoint, dict):
        value = checkpoint.get("seed")
    return int(value) if _float(value) == _float(value) else ""


def _config_hash(cfg: dict[str, Any] | None) -> str:
    if cfg is None:
        return ""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _value(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


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
