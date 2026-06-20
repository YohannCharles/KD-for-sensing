import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from kd_sensing.baselines.beambench.image_ae_gps_config import (
    ImageAEGPSDirectTrainingConfig,
    TARGET_TABLE_III_ROW,
    _gps_calibration_metadata,
)


def _performance_metadata(
    cfg: ImageAEGPSDirectTrainingConfig,
    device: torch.device,
    amp_enabled: bool,
    runtime_report: Mapping[str, Any],
    feature_cache_reports: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "device": str(device),
        "amp": {
            "enabled": bool(amp_enabled),
            "dtype": str(cfg.amp_dtype),
            "grad_scaler": bool(cfg.amp_grad_scaler) and bool(amp_enabled),
        },
        "runtime": dict(runtime_report),
        "optimizer": {
            "type": "AdamW",
            "fused_requested": bool(cfg.fused_optimizer),
        },
        "dataloader": {
            "num_workers": int(cfg.num_workers),
            "pin_memory": bool(cfg.pin_memory),
            "persistent_workers": bool(cfg.persistent_workers) and int(cfg.num_workers) > 0,
            "prefetch_factor": int(cfg.prefetch_factor) if cfg.prefetch_factor is not None else None,
            "non_blocking_transfer": bool(cfg.non_blocking_transfer),
        },
        "batches": {
            "ae_batch_size": int(cfg.ae_batch_size),
            "fusion_batch_size": int(cfg.fusion_batch_size),
            "feature_cache_batch_size": int(cfg.feature_cache_batch_size),
        },
        "feature_cache": {
            "enabled_requested": bool(cfg.cache_frozen_ae_features),
            "active": bool(cfg.freeze_ae_encoder and cfg.cache_frozen_ae_features),
            "reports": _json_ready(dict(feature_cache_reports)),
        },
    }

def _paper_split_gps_calibration_metadata(
    train_cfgs: Sequence[ImageAEGPSDirectTrainingConfig],
    eval_cfgs: Sequence[ImageAEGPSDirectTrainingConfig],
) -> dict[str, Any]:
    return {
        "train_scenes": {str(cfg.scene): _gps_calibration_metadata(cfg) for cfg in train_cfgs},
        "eval_scenes": {str(cfg.scene): _gps_calibration_metadata(cfg) for cfg in eval_cfgs},
    }

def _paper_split_summary(scene_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    weighted_numerator = 0.0
    weighted_count = 0
    for report in scene_reports:
        scene = int(report["scene"])
        metrics = dict(report["metrics"])
        metric = float(metrics.get("official_top3_dba", 0.0))
        sample_count = int(metrics.get("valid_label_count", metrics.get("sample_count", 0)))
        target = float(TARGET_TABLE_III_ROW[f"scene{scene}"])
        weighted_numerator += metric * sample_count
        weighted_count += sample_count
        rows.append(
            {
                "scene": scene,
                "local_official_top3_dba": metric,
                "paper_tableiii_dba": target,
                "delta_local_minus_paper": metric - target,
                "sample_count": sample_count,
                "official_top1_acc": float(metrics.get("official_top1_acc", 0.0)),
                "official_top3_acc": float(metrics.get("official_top3_acc", 0.0)),
                "official_top5_acc": float(metrics.get("official_top5_acc", 0.0)),
                "circular_top3_dba": float(metrics.get("circular_top3_dba", 0.0)),
            }
        )
    rows = sorted(rows, key=lambda item: int(item["scene"]))
    local_simple_mean = sum(float(row["local_official_top3_dba"]) for row in rows) / max(len(rows), 1)
    paper_simple_mean = sum(float(row["paper_tableiii_dba"]) for row in rows) / max(len(rows), 1)
    local_weighted_overall = weighted_numerator / max(weighted_count, 1)
    return {
        "rows": rows,
        "metric_field": "official_top3_dba",
        "local_simple_mean": local_simple_mean,
        "local_weighted_overall": local_weighted_overall,
        "paper_simple_mean": paper_simple_mean,
        "paper_tableiii_overall": float(TARGET_TABLE_III_ROW["overall"]),
        "delta_weighted_minus_paper_overall": local_weighted_overall - float(TARGET_TABLE_III_ROW["overall"]),
    }

def _write_paper_split_summary_artifacts(report: Mapping[str, Any], output_root: Path) -> None:
    summary = dict(report["summary"])
    rows = list(summary["rows"])
    csv_path = output_root / "tableiii_camera_ae_gps_summary.csv"
    md_path = output_root / "tableiii_camera_ae_gps_summary.md"
    json_path = output_root / "tableiii_camera_ae_gps_summary.json"
    _write_csv_rows(csv_path, rows)
    md_path.write_text(_paper_split_summary_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")

def _paper_split_summary_markdown(report: Mapping[str, Any]) -> str:
    summary = dict(report["summary"])
    lines = [
        "# Camera AE + GPS Direct Paper-Split Local Reproduction",
        "",
        f"- Train scenes: {', '.join(str(item) for item in report['paper_split']['train_scenes'])}",
        f"- Eval scenes: {', '.join(str(item) for item in report['paper_split']['eval_scenes'])}",
        f"- Selection split: {report['selection']['mode']}",
        f"- Target beam source: {report['config'].get('target_beam_source', 'current')}",
        f"- GPS feature mode: {report['config'].get('gps_feature_mode', 'relative_polar')}",
        f"- Metric field: {summary['metric_field']}",
        "",
        "| Scene | Local DBA | Paper DBA | Delta | Top1 | Top3 | Top5 | Samples |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {scene} | {local:.4f} | {paper:.4f} | {delta:+.4f} | {top1:.4f} | {top3:.4f} | {top5:.4f} | {samples} |".format(
                scene=int(row["scene"]),
                local=float(row["local_official_top3_dba"]),
                paper=float(row["paper_tableiii_dba"]),
                delta=float(row["delta_local_minus_paper"]),
                top1=float(row["official_top1_acc"]),
                top3=float(row["official_top3_acc"]),
                top5=float(row["official_top5_acc"]),
                samples=int(row["sample_count"]),
            )
        )
    lines.extend(
        [
            "",
            f"- Local simple mean: {float(summary['local_simple_mean']):.4f}",
            f"- Local weighted overall: {float(summary['local_weighted_overall']):.4f}",
            f"- Paper Table III overall: {float(summary['paper_tableiii_overall']):.4f}",
            f"- Delta weighted overall: {float(summary['delta_weighted_minus_paper_overall']):+.4f}",
            "",
            str(report["official_comparability_note"]),
            "",
        ]
    )
    return "\n".join(lines)

def _write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_ready(row.get(key, "")) for key in fieldnames})

def _csv_ready(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if torch.is_tensor(value):
        return _csv_ready(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _csv_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value

def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = ["_json_ready"]
