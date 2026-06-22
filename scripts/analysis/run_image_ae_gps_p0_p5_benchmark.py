from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from kd_sensing.baselines.beambench.image_ae_gps_config import (
    ImageAEGPSDirectTrainingConfig,
    _configure_torch_runtime,
    _gps_scaler_from_metadata,
    _resolve_amp_dtype,
    _resolve_device,
    _scene_specific_cfg,
    _seed_everything,
    _torch_load,
)
from kd_sensing.baselines.beambench.image_ae_gps_datasets import (
    _build_loader,
    _build_split_dataset,
    _metadata_rows,
)
from kd_sensing.baselines.beambench.image_ae_gps_models import (
    BeamBenchImageAEGPSDirectModel,
    _classifier_logits_from_batch,
)
from kd_sensing.baselines.beambench.image_ae_gps_reports import _json_ready
from kd_sensing.baselines.beambench.metrics import beambench_metric_summary_from_logits
from kd_sensing.data.difficulty.presets import (
    PREDICTIVE_JEPA_CANONICAL_CONDITIONS,
    PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
)
from kd_sensing.diagnostics.jepa_benchmark_manifest import normalize_suite_config
from kd_sensing.diagnostics.jepa_benchmark_perturbations import apply_benchmark_perturbation


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint = _torch_load(Path(args.checkpoint), map_location="cpu")
    cfg = ImageAEGPSDirectTrainingConfig(**dict(checkpoint["config"]))
    cfg = replace(
        cfg,
        output_dir=str(output_root),
        device=args.device,
        num_workers=args.num_workers,
        cache_frozen_ae_features=False,
        save_predictions=False,
    )
    _seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    runtime = _configure_torch_runtime(cfg, device)
    amp_enabled = bool(cfg.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)

    model = _load_model(checkpoint, cfg, device)
    loaders = _loaders_by_scene(checkpoint, cfg, args.eval_scenes)
    suite = normalize_suite_config(
        {
            "id": "predictive_jepa_robustness",
            "type": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
            "preset": "P0_P5",
            "history_window": 4,
        }
    )

    rows = []
    overall_rows = []
    for condition in PREDICTIVE_JEPA_CANONICAL_CONDITIONS:
        condition_id = str(condition["id"])
        severity = float(condition["severity"])
        condition_dir = output_root / condition_id
        condition_dir.mkdir(parents=True, exist_ok=True)
        logits_by_scene = []
        labels_by_scene = []
        for scene, loader in loaders.items():
            logits, labels, warning_count = _eval_scene_condition(
                model,
                loader,
                cfg,
                suite,
                scene=scene,
                condition_id=condition_id,
                severity=severity,
                seed=args.seed,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            logits_by_scene.append(logits)
            labels_by_scene.append(labels)
            metrics = _metrics(logits, labels, cfg)
            rows.append(_row(args.model, args.scene_group, condition_id, severity, scene, metrics, warning_count))
        overall_metrics = _metrics(torch.cat(logits_by_scene, dim=0), torch.cat(labels_by_scene, dim=0), cfg)
        overall_rows.append(_row(args.model, args.scene_group, condition_id, severity, "overall", overall_metrics, 0))

    _write_csv(output_root / "metrics_by_condition.csv", rows + overall_rows)
    wide = _wide_row(args.model, overall_rows)
    _write_csv(output_root / "p0_p5_dba_wide.csv", [wide])
    (output_root / "run_report.json").write_text(
        json.dumps(
            _json_ready(
                {
                    "model": args.model,
                    "checkpoint": str(args.checkpoint),
                    "scene_group": args.scene_group,
                    "eval_scenes": [int(s) for s in args.eval_scenes],
                    "output_root": str(output_root),
                    "runtime": runtime,
                    "conditions": [str(c["id"]) for c in PREDICTIVE_JEPA_CANONICAL_CONDITIONS],
                    "wide": wide,
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(wide), sort_keys=True))


def _load_model(checkpoint: dict[str, Any], cfg: ImageAEGPSDirectTrainingConfig, device: torch.device):
    ae_checkpoint = Path(str(checkpoint.get("ae_checkpoint_path") or cfg.ae_checkpoint_path or ""))
    if not ae_checkpoint.exists():
        raise FileNotFoundError(f"Missing AE checkpoint: {ae_checkpoint}")
    model = BeamBenchImageAEGPSDirectModel(
        num_beams=cfg.num_beams,
        gps_input_size=cfg.gps_input_size,
        ae_latent_dim=cfg.ae_latent_dim,
        image_channels=cfg.image_channels,
        image_size=cfg.image_size,
        hidden_dim=cfg.fusion_hidden_dim,
        dropout=cfg.fusion_dropout,
        fusion_architecture=cfg.fusion_architecture,
        fusion_dense_hidden_sizes=cfg.fusion_dense_hidden_sizes,
        fusion_activation=cfg.fusion_activation,
        fusion_last_activation=cfg.fusion_last_activation,
        ae_checkpoint_path=ae_checkpoint,
        freeze_ae_encoder=cfg.freeze_ae_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _loaders_by_scene(checkpoint: dict[str, Any], cfg: ImageAEGPSDirectTrainingConfig, scenes: list[int]):
    gps_scaler = _gps_scaler_from_metadata(checkpoint.get("gps_scaler")) if cfg.gps_normalize else None
    loaders = {}
    for scene in scenes:
        scene_cfg = _scene_specific_cfg(cfg, int(scene))
        dataset = _build_split_dataset(scene_cfg, split="test", gps_scaler=gps_scaler, gps_normalize=cfg.gps_normalize)
        loaders[int(scene)] = _build_loader(
            dataset,
            batch_size=cfg.fusion_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            cfg=cfg,
        )
    return loaders


def _eval_scene_condition(
    model,
    loader,
    cfg: ImageAEGPSDirectTrainingConfig,
    suite: dict[str, Any],
    *,
    scene: int,
    condition_id: str,
    severity: float,
    seed: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    logits_all = []
    labels_all = []
    warning_count = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["target"].to(device=device, dtype=torch.long, non_blocking=cfg.non_blocking_transfer)
            batch["target_beam"] = batch["target"]
            batch["scene"] = [str(scene)] * int(labels.numel())
            rows = _metadata_rows(batch.get("metadata"), count=int(labels.numel()))
            batch["sample_ids"] = [f"s{scene}_{row.get('dataset_index', i)}" for i, row in enumerate(rows)]
            perturbed, warnings = apply_benchmark_perturbation(
                batch,
                suite,
                severity=severity,
                seed=seed,
                sample_ids=batch["sample_ids"],
            )
            warning_count += len(warnings)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = _classifier_logits_from_batch(
                    model,
                    perturbed,
                    device=device,
                    non_blocking=cfg.non_blocking_transfer,
                )
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0), warning_count


def _metrics(logits: torch.Tensor, labels: torch.Tensor, cfg: ImageAEGPSDirectTrainingConfig) -> dict[str, Any]:
    return beambench_metric_summary_from_logits(
        logits,
        labels,
        num_beams=cfg.num_beams,
        topk=cfg.topk,
        dba_delta=cfg.dba_delta,
        circular=True,
    )


def _row(model: str, scene_group: str, condition: str, severity: float, scene: int | str, metrics: dict[str, Any], warnings: int):
    return {
        "model": model,
        "scene_group": scene_group,
        "condition": condition,
        "p_level": condition.split("_", 1)[0],
        "severity": severity,
        "scene": scene,
        "sample_count": metrics.get("sample_count", ""),
        "top1": metrics.get("official_top1_acc", ""),
        "top3": metrics.get("official_top3_acc", ""),
        "top5": metrics.get("official_top5_acc", ""),
        "dba": metrics.get("official_top3_dba", ""),
        "circular_dba": metrics.get("circular_top3_dba", ""),
        "warning_count": warnings,
    }


def _wide_row(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = {str(row["p_level"]): float(row["dba"]) for row in rows}
    row = {"model": model, "overall_clean": values.get("P0", "")}
    for key in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        row[key] = values.get(key, "")
    present = [float(row[key]) for key in ["P0", "P1", "P2", "P3", "P4", "P5"] if row[key] != ""]
    row["overall_p0_p5_mean"] = sum(present) / len(present) if present else ""
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scene-group", required=True)
    parser.add_argument("--eval-scenes", nargs="+", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
