#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from kd_sensing.data.temporal_missing import (
    DEFAULT_TEMPORAL_MODALITIES,
    apply_modality_temporal_mask_to_batch,
    generate_fixed_eval_mask_cache,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
)
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.evaluation_pass_runtime import prepare_evaluation_batch
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import run_model_step
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import _beam_classification_metrics
from kd_sensing.utils.checkpoint import load_model_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = "ours_c2_main,ours_b4_nonrouter_soft_jepa,ours_e5_low_lr_pcpg,amber_full,rmbp_mm"
MATRIX_COLUMNS = ["missing_rate", "full", "drop1", "drop2", "drop3"]
PATTERN_NAMES = ["missing_image", "missing_radar", "missing_lidar", "missing_gps", "image_only", "radar_only", "lidar_only", "gps_only"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rates = parse_csv_floats(args.eval_temporal_missing_rates, (0.0, 0.2, 0.4, 0.6, 0.8))
    drop_counts = parse_csv_ints(args.eval_drop_counts, (0, 1, 2, 3))
    mask_types = parse_csv_strings(args.eval_mask_types, ("modality_frame", "frame_level", "block"))
    cache = generate_fixed_eval_mask_cache(
        args.eval_fixed_mask_cache,
        rates=rates,
        drop_counts=drop_counts,
        mask_types=mask_types,
        num_masks_per_cell=int(args.eval_num_masks_per_cell),
        seed=int(args.eval_mask_seed),
        history_window=int(args.history_window),
        modalities=DEFAULT_TEMPORAL_MODALITIES,
    )
    failures = []
    for method in _csv(args.methods):
        for seed in [int(item) for item in _csv(args.seeds)]:
            try:
                evaluate_method_seed(method, seed, args, cache)
            except FileNotFoundError as exc:
                if args.allow_missing_checkpoints:
                    print(f"skip missing {method}/seed{seed}: {exc}")
                    continue
                failures.append({"method": method, "seed": seed, "error": str(exc)})
    if failures:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "failed_eval_jobs.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate H5/P1 fixed temporal missing matrices.")
    parser.add_argument("--root", default="outputs/h5_p1_temporal_models_v1")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--history_window", "--history-window", type=int, default=5)
    parser.add_argument("--prediction_window", "--prediction-window", type=int, default=1)
    parser.add_argument("--eval_temporal_missing_rates", "--eval-temporal-missing-rates", default="0.0,0.2,0.4,0.6,0.8")
    parser.add_argument("--eval_drop_counts", "--eval-drop-counts", default="0,1,2,3")
    parser.add_argument("--eval_mask_types", "--eval-mask-types", default="modality_frame,frame_level,block")
    parser.add_argument("--eval_num_masks_per_cell", "--eval-num-masks-per-cell", type=int, default=16)
    parser.add_argument("--eval_mask_seed", "--eval-mask-seed", type=int, default=20260708)
    parser.add_argument("--eval_fixed_mask_cache", "--eval-fixed-mask-cache", default="outputs/temporal_eval_masks_v1")
    parser.add_argument("--output_dir", "--output-dir", default="outputs/h5_p1_temporal_models_v1/eval_matrix")
    parser.add_argument("--max_batches", "--max-batches", type=int, default=None)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
    parser.add_argument("--allow_missing_checkpoints", "--allow-missing-checkpoints", action="store_true")
    return parser


def evaluate_method_seed(method: str, seed: int, args: argparse.Namespace, cache: dict[tuple[float, int], dict[str, Any]]) -> None:
    run_root = Path(args.root)
    out_dir = Path(args.output_dir) / method / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _find_config(run_root, method, seed)
    checkpoint, checkpoint_policy = _find_checkpoint(run_root, method, seed)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg.setdefault("temporal_missing", {})["mode"] = "none"
    cfg.setdefault("temporal_missing", {})["enabled"] = False
    cfg.setdefault("experiment", {})["seed"] = int(seed)
    _override_eval_batch_size(cfg, args.batch_size)
    device = build_device(cfg)
    dataloaders = build_dataloaders(cfg)
    split_key = "validation" if "validation" in dataloaders else "val" if "val" in dataloaders else "test"
    model = build_model(cfg["model"]["primary"]).to(device)
    load_model_state(checkpoint, model, role="h5/p1 temporal matrix", map_location=device, strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)))
    model.eval()
    rows_by_metric = {"top1": [], "within3": [], "mae": []}
    pattern_rows = []
    mask_stat_rows = []
    router_rows = []
    rates = sorted({key[0] for key in cache})
    drop_counts = sorted({key[1] for key in cache})
    for rate in rates:
        metric_values = {metric: {"missing_rate": rate} for metric in rows_by_metric}
        for drop_count in drop_counts:
            payload = cache[(rate, drop_count)]
            cell_rows = []
            for mask_item in payload["masks"]:
                metrics = _evaluate_one_mask(
                    model,
                    dataloaders[split_key],
                    cfg,
                    device,
                    mask_item,
                    args.max_batches,
                    mask_modalities=payload.get("modalities"),
                )
                cell_rows.append(metrics)
                pattern_rows.append({
                    "method": method,
                    "seed": seed,
                    "missing_rate": rate,
                    "drop_count": drop_count,
                    "pattern": _pattern_name(mask_item),
                    **metrics,
                })
                router_rows.append({
                    "method": method,
                    "seed": seed,
                    "missing_rate": rate,
                    "drop_count": drop_count,
                    "pattern": _pattern_name(mask_item),
                    **{key: value for key, value in metrics.items() if key.startswith("mean_gate_") or key.startswith("mean_temporal_gate_") or key.startswith("router_oracle_acc") or key in {"gate_entropy", "global_gate_entropy"}},
                })
            label = "full" if drop_count == 0 else f"drop{drop_count}"
            metric_values["top1"][label] = _mean(cell_rows, "top1")
            metric_values["within3"][label] = _mean(cell_rows, "within_3")
            metric_values["mae"][label] = _mean(cell_rows, "mae")
            mask_stat_rows.append({
                "missing_rate": rate,
                "drop_count": drop_count,
                "num_masks": len(payload["masks"]),
                "checksum": payload.get("checksum", ""),
                "checkpoint_policy": checkpoint_policy,
                "checkpoint": str(checkpoint),
            })
        for metric, rows in rows_by_metric.items():
            rows.append(metric_values[metric])
    _write_csv(out_dir / "top1_matrix.csv", rows_by_metric["top1"], MATRIX_COLUMNS)
    _write_csv(out_dir / "within3_matrix.csv", rows_by_metric["within3"], MATRIX_COLUMNS)
    _write_csv(out_dir / "mae_matrix.csv", rows_by_metric["mae"], MATRIX_COLUMNS)
    _write_csv(out_dir / "pattern_metrics.csv", pattern_rows, _columns(pattern_rows))
    _write_csv(out_dir / "router_diagnostics.csv", router_rows, _columns(router_rows))
    _write_csv(out_dir / "mask_stats.csv", mask_stat_rows, _columns(mask_stat_rows))


def _override_eval_batch_size(cfg: dict[str, Any], batch_size: int | None) -> None:
    if batch_size is None:
        return
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    loader_cfg = cfg.setdefault("data", {}).setdefault("dataloader", {})
    loader_cfg["test_batch_size"] = int(batch_size)
    loader_cfg["validation_batch_size"] = int(batch_size)
    for split in ("test", "validation"):
        if isinstance(loader_cfg.get(split), dict):
            loader_cfg[split]["batch_size"] = int(batch_size)


def _evaluate_one_mask(
    model,
    dataloader,
    cfg: dict[str, Any],
    device: torch.device,
    mask_item: dict[str, Any],
    max_batches: int | None,
    mask_modalities: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    sums: dict[str, float] = {}
    count = 0
    mask, model_modalities = _mask_in_model_order(model, mask_item, mask_modalities)
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = prepare_evaluation_batch(raw_batch, cfg=cfg, split_name="validation", difficulty_seed=int(cfg.get("experiment", {}).get("seed", 0)), step_index=batch_index)
            apply_modality_temporal_mask_to_batch(batch, mask, modalities=model_modalities)
            modality_mask = batch["modality_mask"].to(device=device, dtype=torch.bool)
            model_cfg = cfg["model"]["primary"]
            step = run_model_step(
                model,
                cfg.get("experiment", {}).get("task", "fusion"),
                batch,
                model_cfg=model_cfg,
                seq_length=int(model_cfg.get("seq_length", cfg.get("model", {}).get("seq_length", 5))),
                num_pred=int(model_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1))),
                downsample_ratio=int(model_cfg.get("downsample_ratio", cfg.get("model", {}).get("downsample_ratio", 1))),
                device=device,
                extra_model_kwargs={"missing_mask": modality_mask},
            )
            logits = step.logits[:, -1, :] if step.logits.ndim == 3 else step.logits
            target = step.labels[:, -1].reshape(-1) if step.labels.ndim > 1 else step.labels.reshape(-1)
            metrics = _beam_classification_metrics(logits, target, cfg)
            metrics.update(_router_metrics(step.model_output.diagnostics))
            batch_count = int(target.numel())
            count += batch_count
            for key, value in metrics.items():
                if isinstance(value, float) and math.isfinite(value):
                    sums[key] = sums.get(key, 0.0) + value * batch_count
    return {key: (value / count if count else math.nan) for key, value in sums.items()}


def _mask_in_model_order(
    model,
    mask_item: dict[str, Any],
    mask_modalities: list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    source = tuple(str(item) for item in (mask_modalities or mask_item.get("modalities") or DEFAULT_TEMPORAL_MODALITIES))
    target = tuple(str(item) for item in getattr(model, "modalities", source))
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError(f"Modality order contains duplicates: cache={list(source)}, model={list(target)}.")
    unknown = [name for name in target if name not in source]
    if unknown:
        raise ValueError(f"Eval mask cache is missing model modalities {unknown}; cache modalities={list(source)}.")
    mask = torch.as_tensor(mask_item["modality_temporal_mask"], dtype=torch.bool)
    if mask.ndim not in {2, 3} or int(mask.shape[-1]) != len(source):
        raise ValueError(
            f"Cached modality_temporal_mask must end with {len(source)} modality columns, got {tuple(mask.shape)}."
        )
    indices = torch.tensor([source.index(name) for name in target], dtype=torch.long, device=mask.device)
    return mask.index_select(-1, indices), target


def _router_metrics(diagnostics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in diagnostics.items():
        if key.startswith("mean_gate_") or key.startswith("mean_temporal_gate_") or key.startswith("router_oracle_acc"):
            scalar = _as_float(value)
            if scalar is not None:
                result[key] = scalar
    for key in ("gate_entropy", "global_gate_entropy", "gate_entropy_temporal", "gate_entropy_modality"):
        if key in diagnostics:
            scalar = _as_float(diagnostics[key])
            if scalar is not None:
                result[key] = scalar
    temporal_gate = diagnostics.get("temporal_gate")
    if torch.is_tensor(temporal_gate) and temporal_gate.ndim == 2:
        for index in range(int(temporal_gate.shape[1])):
            result[f"mean_temporal_gate_t{index}"] = float(temporal_gate[:, index].detach().float().mean().cpu().item())
    return result


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if torch.is_tensor(value) and value.numel() > 0:
        return float(value.detach().float().mean().cpu().item())
    return None


def _find_config(root: Path, method: str, seed: int) -> Path:
    candidates = [
        root / "generated_configs" / f"{method}_seed{seed}.yaml",
        root / method / f"seed{seed}" / "final_config.yaml",
        root / method / f"seed{seed}" / "resolved_config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"config for {method}/seed{seed}")


def _find_checkpoint(root: Path, method: str, seed: int) -> tuple[Path, str]:
    run = root / method / f"seed{seed}"
    names = ["best_avg_missing_top1.pth", "best_avg_missing_top1.pt", "best_top1.pth", "best_top1.pt", "best.pth", "last.pth"]
    for name in names:
        for path in (run / "checkpoints" / name, run / name):
            if path.exists():
                policy = "best_avg_missing_top1" if "avg_missing" in name else "best_top1_fallback"
                return path, policy
    raise FileNotFoundError(f"checkpoint for {method}/seed{seed}")


def _pattern_name(mask_item: dict[str, Any]) -> str:
    dropped = list(mask_item.get("dropped_modalities", []))
    if len(dropped) == 1:
        return "missing_" + dropped[0]
    if len(dropped) == 3:
        only = [item for item in DEFAULT_TEMPORAL_MODALITIES if item not in dropped]
        return (only[0] if only else "unknown") + "_only"
    if len(dropped) > 1:
        return "missing_" + "_".join(dropped)
    return "full"


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row}) if rows else []


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
