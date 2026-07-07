#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from kd_sensing.config.io import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.evaluation_pass_runtime import metadata_rows_from_batch
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.eval.missing_buckets import missing_bucket_mapping_from_rows, write_missing_bucket_mapping
from kd_sensing.eval.missing_patterns import (
    DEFAULT_MODALITIES,
    canonical_missing_pattern_name,
    get_missing_pattern_mask,
    list_standard_missing_patterns,
    resolve_missing_patterns,
)
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import _batch_size, _forward_batch, evaluate_missing_matrix
from kd_sensing.utils.checkpoint import CheckpointLoadError, load_model_state

from scripts.scene31_eval_resolution import resolve_run_dir_and_config


SCRIPT_VERSION = "apples_to_apples_v2_scene31_path_resolution"
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
PREDICTION_FIELDS = [
    "run_name",
    "method",
    "seed",
    "scene",
    "sample_id",
    "pattern",
    "target",
    "pred",
    "top1_correct",
    "top3_correct",
    "top5_correct",
    "within3_correct",
    "abs_error",
    "missing_count",
    "missing_ratio",
    "available_modalities",
    "missing_modalities",
]


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
        "checkpoint_used": _checkpoint_manifest_label(args.checkpoint_policy),
        "split": args.split,
        "max_batches": args.max_batches,
        "runs": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    for run_name in args.runs:
        run_rows, run_manifest, run_predictions = _evaluate_run(
            root,
            run_name,
            requested,
            checkpoint_policy=args.checkpoint_policy,
            manual_checkpoint=manual.get(run_name) or (manual.get("*") if len(args.runs) == 1 else None),
            split=args.split,
            max_batches=args.max_batches,
            device_override=args.device,
            save_predictions_by_pattern=args.save_predictions_by_pattern,
        )
        rows.extend(run_rows)
        prediction_rows.extend(run_predictions)
        manifest["runs"][run_name] = run_manifest

    _annotate_run_level_metrics(rows)
    delta_rows = _delta_rows(rows, args.baseline_name)
    bucket_mapping, bucket_warnings = missing_bucket_mapping_from_rows(rows)
    if bucket_warnings:
        manifest["warnings"] = bucket_warnings
    _write_csv(out_dir / "apples_to_apples_metrics.csv", rows)
    _write_csv(out_dir / "pattern_metrics.csv", rows)
    _write_csv(out_dir / "run_summary.csv", _run_summary_rows(rows, manifest))
    _write_markdown(out_dir / "apples_to_apples_metrics.md", rows, args.baseline_name)
    _write_csv(out_dir / "apples_to_apples_delta.csv", delta_rows)
    if args.save_predictions_by_pattern:
        _write_csv(out_dir / "predictions_by_pattern.csv", prediction_rows, fieldnames=PREDICTION_FIELDS)
    write_missing_bucket_mapping(out_dir / "missing_bucket_mapping.json", bucket_mapping)
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
    parser.add_argument("--save-predictions-by-pattern", action="store_true")
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
    save_predictions_by_pattern: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    run_resolution = resolve_run_dir_and_config(root, run_name)
    cfg, cfg_path, config_warnings = _load_run_config(run_resolution)
    checkpoint_choice = run_resolution.checkpoint
    if manual_checkpoint:
        checkpoint_choice.path = Path(manual_checkpoint)
        checkpoint_choice.checkpoint_used = "manual_path"
        checkpoint_choice.candidates.insert(0, str(manual_checkpoint))
    ckpt_path = checkpoint_choice.path
    manifest = {
        "run_dir": str(run_resolution.run_dir or ""),
        "config_path": str(cfg_path) if cfg_path else "",
        "config_hash": _config_hash(cfg),
        "checkpoint_path": str(ckpt_path) if ckpt_path else "",
        "checkpoint_used": checkpoint_choice.checkpoint_used,
        "checkpoint_epoch": checkpoint_choice.best_epoch if checkpoint_choice.best_epoch is not None else "",
        "best_epoch": checkpoint_choice.best_epoch if checkpoint_choice.best_epoch is not None else "",
        "best_val_acc": checkpoint_choice.best_val_acc if checkpoint_choice.best_val_acc is not None else "",
        "checkpoint_policy": checkpoint_policy,
        "checkpoint_resolution": {
            "path": str(checkpoint_choice.path or ""),
            "checkpoint_used": checkpoint_choice.checkpoint_used,
            "best_epoch": checkpoint_choice.best_epoch if checkpoint_choice.best_epoch is not None else "",
            "best_val_acc": checkpoint_choice.best_val_acc if checkpoint_choice.best_val_acc is not None else "",
            "warnings": list(checkpoint_choice.warnings),
            "candidates": list(checkpoint_choice.candidates),
        },
        "searched_paths": list(run_resolution.searched_paths),
        "status_state": run_resolution.status_state,
        "warnings": [*config_warnings, *run_resolution.warnings, *checkpoint_choice.warnings],
        "script_version": SCRIPT_VERSION,
        "split_requested": split,
        "max_batches": max_batches,
    }
    if cfg is None:
        print(f"[WARN] {run_name}: missing config")
        manifest["status"] = "missing_config"
        return (
            _missing_rows(
                run_name,
                requested_patterns,
                status="missing_config",
                run_dir=run_resolution.run_dir,
                config_path=cfg_path,
                checkpoint_path=ckpt_path,
                checkpoint_used=checkpoint_choice.checkpoint_used,
                best_epoch=checkpoint_choice.best_epoch,
                best_val_acc=checkpoint_choice.best_val_acc,
                max_batches=max_batches,
            ),
            manifest,
            [],
        )
    if ckpt_path is None:
        print(f"[WARN] {run_name}: missing checkpoint for policy {checkpoint_policy}")
        manifest["status"] = "missing_checkpoint"
        return (
            _missing_rows(
                run_name,
                requested_patterns,
                status="missing_checkpoint",
                run_dir=run_resolution.run_dir,
                config_path=cfg_path,
                checkpoint_path=ckpt_path,
                checkpoint_used=checkpoint_choice.checkpoint_used,
                best_epoch=checkpoint_choice.best_epoch,
                best_val_acc=checkpoint_choice.best_val_acc,
                max_batches=max_batches,
            ),
            manifest,
            [],
        )
    if checkpoint_choice.checkpoint_used == "last_fallback":
        print(f"[WARN] {run_name}: best checkpoint missing; using last checkpoint {ckpt_path}")

    cfg.setdefault("output", {})["run_name"] = run_name
    if save_predictions_by_pattern:
        cfg.setdefault("data", {}).setdefault("dataset", {})["return_metadata"] = True
    if device_override:
        cfg.setdefault("experiment", {})["device"] = device_override
    try:
        device = build_device(cfg)
        dataloaders = build_dataloaders(cfg)
        split_key = _resolve_split(dataloaders, split)
        model = build_model(cfg["model"]["primary"]).to(device)
        load_result = _load_eval_checkpoint(
            ckpt_path,
            model,
            cfg,
            manifest["warnings"],
            device=device,
        )
        checkpoint_payload = load_result.get("checkpoint", {})
        checkpoint_epoch = checkpoint_choice.best_epoch or _checkpoint_epoch(checkpoint_payload, ckpt_path)
        seed = _seed(cfg, checkpoint_payload)
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
        output_patterns = _output_pattern_names(requested_patterns, eval_names)
        rows = [
            _metrics_row(
                run_name,
                pattern,
                by_pattern.get(pattern, {}),
                run_dir=run_resolution.run_dir,
                config_path=cfg_path,
                checkpoint_path=ckpt_path,
                checkpoint_epoch=checkpoint_epoch,
                checkpoint_used=checkpoint_choice.checkpoint_used,
                best_epoch=checkpoint_choice.best_epoch,
                best_val_acc=checkpoint_choice.best_val_acc,
                seed=seed,
                config_hash=manifest["config_hash"],
                status="ok",
                model_modalities=modalities,
                max_batches=max_batches,
            )
            for pattern in output_patterns
        ]
        prediction_rows = (
            _prediction_rows_for_patterns(
                model,
                dataloaders[split_key],
                device,
                modalities,
                patterns,
                prediction_index=cfg.get("evaluation", {}).get("missing_patterns", {}).get("prediction_index", "last"),
                cfg=cfg,
                run_name=run_name,
                method=_method_name(run_name),
                seed=seed,
                max_batches=max_batches,
            )
            if save_predictions_by_pattern
            else []
        )
        _fill_avg_missing(rows, by_pattern)
        manifest.update(
            {
                "status": "ok",
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_used": checkpoint_choice.checkpoint_used,
                "best_epoch": checkpoint_choice.best_epoch if checkpoint_choice.best_epoch is not None else checkpoint_epoch,
                "best_val_acc": checkpoint_choice.best_val_acc if checkpoint_choice.best_val_acc is not None else "",
                "seed": seed,
                "split": split_key,
                "modalities": modalities,
            }
        )
        if save_predictions_by_pattern:
            manifest["predictions_by_pattern"] = {
                "path": "predictions_by_pattern.csv",
                "rows": len(prediction_rows),
                "scene_rows": len([row for row in prediction_rows if row.get("scene")]),
            }
        return rows, manifest, prediction_rows
    except Exception as exc:
        print(f"[WARN] {run_name}: evaluation failed: {exc}")
        manifest["status"] = "eval_failed"
        manifest["warnings"].append(str(exc))
        return (
            _missing_rows(
                run_name,
                requested_patterns,
                status="eval_failed",
                run_dir=run_resolution.run_dir,
                config_path=cfg_path,
                checkpoint_path=ckpt_path,
                checkpoint_used=checkpoint_choice.checkpoint_used,
                best_epoch=checkpoint_choice.best_epoch,
                best_val_acc=checkpoint_choice.best_val_acc,
                max_batches=max_batches,
            ),
            manifest,
            [],
        )


def _load_run_config(resolution: Any) -> tuple[dict[str, Any] | None, Path | None, list[str]]:
    warnings: list[str] = []
    path = resolution.config_path
    if path is not None and path.exists():
        if "configs" in path.parts:
            return load_config(path), path, warnings
        data = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
        return data, path, warnings
    warnings.append("missing_config; searched_paths=" + ";".join(resolution.searched_paths))
    return None, None, warnings


def _evaluation_pattern_names(requested_patterns: list[str], modalities: list[str]) -> list[str]:
    names = [name for name in requested_patterns if name != "avg_missing"]
    if "avg_missing" in requested_patterns:
        names.extend(name for name in list_standard_missing_patterns(_pattern_name_modalities(modalities)) if name != "full")
    return list(dict.fromkeys(names))


def _output_pattern_names(requested_patterns: list[str], eval_names: list[str]) -> list[str]:
    names = list(eval_names)
    if "avg_missing" in requested_patterns:
        names.append("avg_missing")
    return list(dict.fromkeys(names))


def _pattern_name_modalities(modalities: list[str] | tuple[str, ...]) -> list[str]:
    values = [str(item) for item in modalities]
    preferred = [name for name in DEFAULT_MODALITIES if name in values]
    preferred.extend(name for name in values if name not in preferred)
    return preferred


def _load_eval_checkpoint(
    ckpt_path: Path,
    model: Any,
    cfg: dict[str, Any],
    warnings: list[str],
    *,
    device: torch.device,
) -> dict[str, Any]:
    strict = bool(cfg.get("checkpoint", {}).get("strict_load", True))
    try:
        return load_model_state(
            ckpt_path,
            model,
            role="apples-to-apples",
            map_location=device,
            strict=strict,
        )
    except CheckpointLoadError as exc:
        pattern_film = cfg.get("model", {}).get("primary", {}).get("pattern_film")
        if not (strict and _identity_pattern_film_enabled(pattern_film)):
            raise
        warnings.append(f"strict checkpoint load failed for identity PatternFiLM; retried non-strict: {exc}")
        return load_model_state(
            ckpt_path,
            model,
            role="apples-to-apples",
            map_location=device,
            strict=False,
        )


def _identity_pattern_film_enabled(raw: Any) -> bool:
    return isinstance(raw, dict) and bool(raw.get("enabled", False)) and bool(raw.get("init_identity", True))


def _metrics_row(
    run_name: str,
    pattern: str,
    source: dict[str, Any],
    *,
    run_dir: Path | str | None,
    config_path: Path | str | None,
    checkpoint_path: Path | str | None,
    checkpoint_epoch: int | str = "",
    checkpoint_used: str = "",
    best_epoch: int | str | None = "",
    best_val_acc: float | str | None = "",
    seed: int | str = "",
    config_hash: str = "",
    status: str,
    model_modalities: list[str] | tuple[str, ...] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    standard_mask = _mask_text(pattern)
    model_mask = _mask_text(pattern, model_modalities) if model_modalities else ""
    missing_count, missing_ratio, available, missing = _pattern_metadata(pattern, model_modalities)
    within3 = _value(source, "within_3")
    count = _value(source, "count", "sample_count", "num_samples")
    return {
        "status": status,
        "run_name": run_name,
        "method": _method_name(run_name),
        "run_dir": str(run_dir or ""),
        "config_path": str(config_path or ""),
        "checkpoint_path": str(checkpoint_path or ""),
        "checkpoint_used": checkpoint_used,
        "checkpoint_epoch": checkpoint_epoch,
        "best_epoch": "" if best_epoch is None else best_epoch,
        "best_val_acc": "" if best_val_acc is None else best_val_acc,
        "seed": seed,
        "eval_script_version": SCRIPT_VERSION,
        "config_hash": config_hash,
        "max_batches": "" if max_batches is None else max_batches,
        "pattern": pattern,
        "missing_count": missing_count,
        "missing_ratio": missing_ratio,
        "available_modalities": ",".join(available),
        "missing_modalities": ",".join(missing),
        "standard_mask": standard_mask,
        "model_mask": model_mask,
        "top1": _value(source, "top1"),
        "top3": _value(source, "top3"),
        "top5": _value(source, "top5"),
        "within_3": within3,
        "within3": within3,
        "within@3": within3,
        "adba": _value(source, "adba"),
        "mae": _value(source, "mae"),
        "loss": _value(source, "loss"),
        "count": count,
        "num_samples": count,
    }


def _missing_rows(
    run_name: str,
    requested_patterns: list[str],
    *,
    status: str,
    run_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    checkpoint_used: str = "",
    best_epoch: int | str | None = "",
    best_val_acc: float | str | None = "",
    max_batches: int | None = None,
) -> list[dict[str, Any]]:
    return [
        _metrics_row(
            run_name,
            pattern,
            {},
            run_dir=run_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            checkpoint_used=checkpoint_used,
            best_epoch=best_epoch,
            best_val_acc=best_val_acc,
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
    avg_row["within@3"] = avg_row.get("within_3", "")
    counts = [_float(row.get("count") or row.get("sample_count") or row.get("num_samples")) for row in source]
    counts = [value for value in counts if _isnum(value)]
    avg_row["count"] = int(sum(counts)) if counts else ""


def _prediction_rows_for_patterns(
    model: Any,
    dataloader: Any,
    device: torch.device,
    modalities: list[str],
    patterns: dict[str, list[int]],
    *,
    prediction_index: int | str,
    cfg: dict[str, Any],
    run_name: str,
    method: str,
    seed: int | str,
    max_batches: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for pattern_name, pattern in patterns.items():
            canonical = canonical_missing_pattern_name(pattern_name)
            if canonical == "avg_missing":
                continue
            for batch_index, raw_batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= int(max_batches):
                    break
                batch_size = _batch_size(raw_batch)
                missing_mask = _missing_mask_tensor(pattern, batch_size, device)
                forward_mask = None if all(int(value) == 1 for value in pattern) else missing_mask
                logits, target, _ = _forward_batch(
                    model,
                    raw_batch,
                    forward_mask,
                    device,
                    prediction_index=prediction_index,
                    cfg=cfg,
                )
                rows.extend(
                    _prediction_rows_for_batch(
                        logits,
                        target,
                        raw_batch,
                        run_name=run_name,
                        method=method,
                        seed=seed,
                        pattern=canonical,
                        modalities=modalities,
                        pattern_mask=pattern,
                    )
                )
    return rows


def _prediction_rows_for_batch(
    logits: torch.Tensor,
    target: torch.Tensor,
    raw_batch: Any,
    *,
    run_name: str,
    method: str,
    seed: int | str,
    pattern: str,
    modalities: list[str],
    pattern_mask: list[int],
) -> list[dict[str, Any]]:
    logits = logits.detach().cpu()
    target = target.detach().cpu().to(dtype=torch.long).reshape(-1)
    pred = torch.argmax(logits, dim=-1).to(dtype=torch.long).reshape(-1)
    topk = torch.topk(logits, k=min(5, int(logits.shape[-1])), dim=-1).indices.to(dtype=torch.long)
    top3 = topk[:, : min(3, topk.shape[1])]
    target_col = target.reshape(-1, 1)
    distances = _circular_distance(pred, target, int(logits.shape[-1]))
    metadata = raw_batch.get("metadata") if isinstance(raw_batch, dict) else None
    metadata_rows = metadata_rows_from_batch(metadata)
    length = int(min(len(target), len(pred), len(metadata_rows) if metadata_rows else len(target)))
    if not metadata_rows:
        metadata_rows = [{} for _ in range(length)]
    available, missing = _pattern_modalities(modalities, pattern_mask)
    missing_count = len(missing)
    missing_ratio = missing_count / len(pattern_mask) if pattern_mask else 0.0
    rows: list[dict[str, Any]] = []
    for index in range(length):
        meta = metadata_rows[index] if index < len(metadata_rows) else {}
        rows.append(
            {
                "run_name": run_name,
                "method": method,
                "seed": seed,
                "scene": _scene_label(meta),
                "sample_id": _sample_id(meta, index),
                "pattern": pattern,
                "target": int(target[index].item()),
                "pred": int(pred[index].item()),
                "top1_correct": int(pred[index].item() == target[index].item()),
                "top3_correct": int((top3[index] == target_col[index]).any().item()),
                "top5_correct": int((topk[index] == target_col[index]).any().item()),
                "within3_correct": int(distances[index] <= 3),
                "abs_error": int(distances[index]),
                "missing_count": missing_count,
                "missing_ratio": _format(missing_ratio),
                "available_modalities": ",".join(available),
                "missing_modalities": ",".join(missing),
            }
        )
    return rows


def _annotate_run_level_metrics(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("run_name") or ""), []).append(row)
    for run_rows in grouped.values():
        by_pattern = {str(row.get("pattern")): row for row in run_rows}
        summary = {
            "full_top1": _pattern_metric(by_pattern, "full", "top1"),
            "miss1_top1": _bucket_metric(run_rows, "top1", 1),
            "miss2_top1": _bucket_metric(run_rows, "top1", 2),
            "miss3_top1": _bucket_metric(run_rows, "top1", 3),
            "avg_missing_top1": _pattern_metric(by_pattern, "avg_missing", "top1") or _avg_missing_metric(run_rows, "top1"),
            "overall_mean_top1": _overall_mean_top1(by_pattern),
            "avg_missing_within@3": _pattern_metric(by_pattern, "avg_missing", "within_3") or _avg_missing_metric(run_rows, "within_3"),
            "avg_missing_MAE": _pattern_metric(by_pattern, "avg_missing", "mae") or _avg_missing_metric(run_rows, "mae"),
            "balanced": _balanced_top1(by_pattern),
        }
        if not summary["balanced"]:
            summary["balanced"] = _balanced_from_summary(summary)
        for row in run_rows:
            row.update(summary)


def _bucket_metric(rows: list[dict[str, Any]], metric: str, missing_count: int) -> str:
    values = [
        _float(row.get(metric))
        for row in rows
        if str(row.get("pattern")) not in {"full", "avg_missing"}
        and _float(row.get("missing_count")) == float(missing_count)
        and _isnum(_float(row.get(metric)))
    ]
    return _format(sum(values) / len(values)) if values else ""


def _avg_missing_metric(rows: list[dict[str, Any]], metric: str) -> str:
    values = [
        _float(row.get(metric))
        for row in rows
        if str(row.get("pattern")) not in {"full", "avg_missing"} and _isnum(_float(row.get(metric)))
    ]
    return _format(sum(values) / len(values)) if values else ""


def _balanced_from_summary(summary: dict[str, Any]) -> str:
    avg_missing = _float(summary.get("avg_missing_top1"))
    miss3 = _float(summary.get("miss3_top1"))
    if not _isnum(avg_missing):
        return ""
    score = avg_missing + 0.25 * (miss3 if _isnum(miss3) else 0.0)
    return _format(score)


def _missing_mask_tensor(pattern: list[int], batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(pattern, dtype=torch.bool, device=device).view(1, -1).expand(int(batch_size), -1)


def _pattern_modalities(modalities: list[str], pattern: list[int]) -> tuple[list[str], list[str]]:
    available = [modality for modality, keep in zip(modalities, pattern, strict=False) if int(keep) == 1]
    missing = [modality for modality, keep in zip(modalities, pattern, strict=False) if int(keep) == 0]
    return available, missing


def _circular_distance(pred: torch.Tensor, target: torch.Tensor, num_beams: int) -> list[int]:
    diff = torch.abs(pred.to(dtype=torch.long) - target.to(dtype=torch.long))
    wrapped = torch.minimum(diff, torch.tensor(int(num_beams), dtype=torch.long) - diff)
    return [int(value) for value in wrapped.tolist()]


def _scene_label(metadata: dict[str, Any]) -> str:
    for key in ("scene", "scene_id", "scenario", "scenario_id"):
        value = metadata.get(key)
        if value not in (None, ""):
            number = _float(value)
            return f"Scene{int(number)}" if _isnum(number) else str(value)
    sample_id = str(metadata.get("sample_id") or "")
    if sample_id.startswith("scene") and ":" in sample_id:
        head = sample_id.split(":", 1)[0]
        number = _float(head.removeprefix("scene"))
        if _isnum(number):
            return f"Scene{int(number)}"
    return ""


def _sample_id(metadata: dict[str, Any], fallback_index: int) -> str:
    value = metadata.get("sample_id")
    if value not in (None, ""):
        return str(value)
    scene = _scene_label(metadata) or "unknown_scene"
    dataset_index = metadata.get("dataset_index", fallback_index)
    return f"{scene}:idx:{dataset_index}"


def _run_summary_rows(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("run_name") or ""), []).append(row)
    out: list[dict[str, Any]] = []
    for run_name, run_rows in grouped.items():
        first = run_rows[0] if run_rows else {}
        by_pattern = {str(row.get("pattern")): row for row in run_rows}
        status = str(first.get("status") or "")
        run_manifest = manifest.get("runs", {}).get(run_name, {})
        out.append(
            {
                "status": status,
                "run_name": run_name,
                "run_dir": first.get("run_dir", run_manifest.get("run_dir", "")),
                "config_path": first.get("config_path", run_manifest.get("config_path", "")),
                "checkpoint_path": first.get("checkpoint_path", run_manifest.get("checkpoint_path", "")),
                "checkpoint_used": first.get("checkpoint_used", run_manifest.get("checkpoint_used", "")),
                "best_epoch": first.get("best_epoch", run_manifest.get("best_epoch", "")),
                "best_val_acc": first.get("best_val_acc", run_manifest.get("best_val_acc", "")),
                "full_top1": _pattern_metric(by_pattern, "full", "top1"),
                "avg_missing_top1": _pattern_metric(by_pattern, "avg_missing", "top1"),
                "overall_mean_top1": _overall_mean_top1(by_pattern),
                "balanced": _balanced_top1(by_pattern),
                "avg_missing_within@3": _pattern_metric(by_pattern, "avg_missing", "within_3"),
                "avg_missing_MAE": _pattern_metric(by_pattern, "avg_missing", "mae"),
                "max_batches": first.get("max_batches", run_manifest.get("max_batches", "")),
            }
        )
    return out


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
    columns = ["run_name", "pattern", "top1", "top3", "top5", "within_3", "adba", "mae", "loss", "count", "checkpoint_used", "status"]
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    columns = list(fieldnames or (list(rows[0]) if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in columns} for row in rows])


def _manual_checkpoints(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" in value:
            run, path = value.split("=", 1)
            out[run] = path
        else:
            out["*"] = value
    return out


def _method_name(run_name: str) -> str:
    return run_name.rsplit("_seed", 1)[0] if "_seed" in run_name else run_name


def _canonical_patterns(patterns: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(canonical_missing_pattern_name(item) for item in patterns))


def _mask_text(pattern: str, modalities: list[str] | tuple[str, ...] | None = None) -> str:
    if pattern == "avg_missing":
        return ""
    try:
        return ",".join(str(int(value)) for value in get_missing_pattern_mask(pattern, modalities))
    except ValueError:
        return ""


def _missing_count(pattern: str, modalities: list[str] | tuple[str, ...] | None = None) -> int | str:
    if pattern == "avg_missing":
        return ""
    try:
        mask = get_missing_pattern_mask(pattern, modalities)
    except ValueError:
        return ""
    return len(mask) - sum(int(value) for value in mask)


def _pattern_metadata(
    pattern: str,
    modalities: list[str] | tuple[str, ...] | None = None,
) -> tuple[int | str, str, list[str], list[str]]:
    if pattern == "avg_missing":
        return "", "", [], []
    names = _pattern_name_modalities(modalities or DEFAULT_MODALITIES)
    try:
        mask = get_missing_pattern_mask(pattern, names)
    except ValueError:
        return "", "", [], []
    available = [name for name, keep in zip(names, mask, strict=False) if int(keep) == 1]
    missing = [name for name, keep in zip(names, mask, strict=False) if int(keep) == 0]
    missing_count = len(missing)
    missing_ratio = missing_count / len(mask) if mask else 0.0
    return missing_count, _format(missing_ratio), available, missing


def _pattern_metric(by_pattern: dict[str, dict[str, Any]], pattern: str, metric: str) -> Any:
    return by_pattern.get(pattern, {}).get(metric, "")


def _overall_mean_top1(by_pattern: dict[str, dict[str, Any]]) -> str:
    values = [
        _float(_pattern_metric(by_pattern, pattern, "top1"))
        for pattern in ("full", "missing_gps", "missing_radar", "radar_only", "lidar_only")
    ]
    return _format(sum(values) / len(values)) if all(_isnum(value) for value in values) else ""


def _balanced_top1(by_pattern: dict[str, dict[str, Any]]) -> str:
    avg_missing = _float(_pattern_metric(by_pattern, "avg_missing", "top1"))
    radar_only = _float(_pattern_metric(by_pattern, "radar_only", "top1"))
    lidar_only = _float(_pattern_metric(by_pattern, "lidar_only", "top1"))
    if not _isnum(avg_missing):
        return ""
    score = avg_missing
    score += 0.25 * (radar_only if _isnum(radar_only) else 0.0)
    score += 0.25 * (lidar_only if _isnum(lidar_only) else 0.0)
    return _format(score)


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


def _checkpoint_manifest_label(checkpoint_policy: str) -> str:
    if checkpoint_policy in {"best_val_top1", "best_avg_missing_top1", "best_epoch_from_metrics"}:
        return "best"
    return checkpoint_policy


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
