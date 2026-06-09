from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, TensorDataset, random_split

from kd_sensing.baselines.beambench.image_ae_gps import (
    BeamBenchImageAEGPSDirectModel,
    BeamBenchImageAEGPSFeatureDataset,
    ImageAEGPSDirectTrainingConfig,
    _autocast_context,
    _build_loader,
    _build_split_dataset,
    _encode_last_camera_ae_latent,
    _gps_scaler_from_metadata,
    _load_or_build_ae_feature_dataset,
    _metadata_rows,
    _resolve_amp_dtype,
    _resolve_device,
    _scene_specific_cfg,
    _torch_load,
)
from kd_sensing.baselines.beambench.metrics import beambench_metric_summary_from_logits
from kd_sensing.models.camera_autoencoder import CameraAutoEncoder


DEFAULT_CHECKPOINT = (
    "outputs/beambench_image_ae_gps_direct_tableiii/"
    "full_gpsfix_ae512_validation/checkpoints/best_image_ae_gps_direct_paper_split.pt"
)
DEFAULT_AE128 = (
    "outputs/beambench_image_ae_gps_direct_tableiii/"
    "paper_split_validation/camera_ae/checkpoints/best.pt"
)
DEFAULT_AE512 = (
    "outputs/beambench_image_ae_gps_direct_tableiii/"
    "scene31_gpsfix_ae512_validation/camera_ae/checkpoints/best.pt"
)


class GPSOnlyMLP(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, num_beams: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_beams),
        )

    def forward(self, gps: torch.Tensor) -> torch.Tensor:
        if gps.ndim == 3:
            gps = gps[:, -1, :]
        return self.net(gps.to(dtype=torch.float32))


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    cfg = ImageAEGPSDirectTrainingConfig(**dict(checkpoint["config"]))
    cfg = _analysis_cfg(cfg, output_dir=output_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    device = _resolve_device(args.device or cfg.device)
    amp_enabled = bool(cfg.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)

    ae_checkpoint = Path(str(checkpoint.get("ae_checkpoint_path") or cfg.ae_checkpoint_path or DEFAULT_AE512))
    model = BeamBenchImageAEGPSDirectModel(
        num_beams=cfg.num_beams,
        gps_input_size=cfg.gps_input_size,
        ae_latent_dim=cfg.ae_latent_dim,
        image_channels=cfg.image_channels,
        image_size=cfg.image_size,
        hidden_dim=cfg.fusion_hidden_dim,
        dropout=cfg.fusion_dropout,
        ae_checkpoint_path=ae_checkpoint,
        freeze_ae_encoder=cfg.freeze_ae_encoder,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gps_scaler = _gps_scaler_from_metadata(checkpoint.get("gps_scaler")) if cfg.gps_normalize else None
    source_cache_root = checkpoint_path.parent.parent
    train_features = _load_scene_features(
        model,
        cfg,
        scenes=tuple(int(item) for item in args.train_scenes),
        split="train",
        gps_scaler=gps_scaler,
        output_dir=output_dir,
        source_cache_root=source_cache_root,
        device=device,
        ae_checkpoint=ae_checkpoint,
    )
    eval_features = _load_scene_features(
        model,
        cfg,
        scenes=tuple(int(item) for item in args.eval_scenes),
        split="test",
        gps_scaler=gps_scaler,
        output_dir=output_dir,
        source_cache_root=source_cache_root,
        device=device,
        ae_checkpoint=ae_checkpoint,
    )

    scene_reports: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for scene, dataset in eval_features.items():
        full_logits = _full_logits(model, dataset, cfg, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        zero_logits = _zero_camera_logits(model, dataset, cfg, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        labels = dataset.target
        scene_reports.append(
            {
                "scene": scene,
                "model": "camera_ae_gps",
                "metrics": _metric_summary(full_logits, labels, cfg),
            }
        )
        scene_reports.append(
            {
                "scene": scene,
                "model": "zero_camera_counterfactual",
                "metrics": _metric_summary(zero_logits, labels, cfg),
            }
        )
        all_rows.extend(
            _sample_rows(
                scene=scene,
                labels=labels,
                gps=dataset.gps,
                full_logits=full_logits,
                baseline_logits=zero_logits,
                baseline_name="zero_camera",
                metadata=dataset.rows,
                dba_delta=cfg.dba_delta,
            )
        )

    gps_model, gps_history = _train_gps_only(
        train_features,
        cfg,
        device=device,
        output_dir=output_dir,
        epochs=args.gps_epochs,
        patience=args.gps_patience,
    )
    gps_rows: list[dict[str, Any]] = []
    for scene, dataset in eval_features.items():
        gps_logits = _gps_logits(gps_model, dataset.gps, cfg, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        full_logits = _full_logits(model, dataset, cfg, device=device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
        labels = dataset.target
        scene_reports.append(
            {
                "scene": scene,
                "model": "matched_gps_only_mlp",
                "metrics": _metric_summary(gps_logits, labels, cfg),
            }
        )
        gps_rows.extend(
            _sample_rows(
                scene=scene,
                labels=labels,
                gps=dataset.gps,
                full_logits=full_logits,
                baseline_logits=gps_logits,
                baseline_name="matched_gps_only",
                metadata=dataset.rows,
                dba_delta=cfg.dba_delta,
            )
        )
    all_rows.extend(gps_rows)

    ambiguity_report = _ambiguity_report(gps_rows, neighbor_k=args.neighbor_k)
    _write_csv(output_dir / "sample_gain_rows.csv", all_rows)
    _write_csv(output_dir / "scene_metric_rows.csv", _flatten_scene_reports(scene_reports))
    _write_csv(output_dir / "gps_only_history.csv", gps_history)

    ae_report = _run_ae_reconstruction_diagnostics(
        cfg,
        eval_features=eval_features,
        ae_checkpoints=[Path(args.ae128_checkpoint), Path(args.ae512_checkpoint)],
        output_dir=output_dir,
        device=device,
        sample_count_per_scene=args.reconstruction_samples_per_scene,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "ae_checkpoint": str(ae_checkpoint),
        "train_scenes": [int(item) for item in args.train_scenes],
        "eval_scenes": [int(item) for item in args.eval_scenes],
        "scene_metrics": scene_reports,
        "matched_gps_only_training": {
            "best_validation_official_top3_dba": max((float(row["val_official_top3_dba"]) for row in gps_history), default=0.0),
            "epochs_ran": len(gps_history),
        },
        "ambiguity": ambiguity_report,
        "ae_reconstruction": ae_report,
        "artifacts": {
            "sample_gain_rows": str(output_dir / "sample_gain_rows.csv"),
            "scene_metric_rows": str(output_dir / "scene_metric_rows.csv"),
            "gps_only_history": str(output_dir / "gps_only_history.csv"),
            "reconstruction_grid": str(output_dir / "ae_reconstruction_grid.jpg"),
        },
        "notes": [
            "matched_gps_only_mlp is trained on the same cached GPS features and train scenes as the Camera AE+GPS run.",
            "zero_camera_counterfactual keeps the trained fusion head fixed and replaces the Camera AE latent with zeros.",
            "GPS ambiguity is measured by same-scene nearest-neighbor label disagreement in normalized paper_distance_angle GPS space.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_json_ready(_compact_summary(summary)), indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose BeamBench Camera AE+GPS gains and AE reconstructions.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ae128-checkpoint", default=DEFAULT_AE128)
    parser.add_argument("--ae512-checkpoint", default=DEFAULT_AE512)
    parser.add_argument("--output-dir", default="outputs/analysis/beambench_ae_gps_diagnostics")
    parser.add_argument("--train-scenes", type=int, nargs="+", default=[32, 33, 34])
    parser.add_argument("--eval-scenes", type=int, nargs="+", default=[31, 32, 33, 34])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gps-epochs", type=int, default=80)
    parser.add_argument("--gps-patience", type=int, default=15)
    parser.add_argument("--neighbor-k", type=int, default=20)
    parser.add_argument("--reconstruction-samples-per-scene", type=int, default=8)
    return parser.parse_args()


def _analysis_cfg(
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
) -> ImageAEGPSDirectTrainingConfig:
    return ImageAEGPSDirectTrainingConfig(
        **{
            **cfg.__dict__,
            "output_dir": str(output_dir),
            "fusion_batch_size": int(batch_size),
            "feature_cache_batch_size": int(batch_size),
            "ae_batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "persistent_workers": False if int(num_workers) <= 0 else cfg.persistent_workers,
            "prefetch_factor": None if int(num_workers) <= 0 else cfg.prefetch_factor,
            "save_predictions": False,
        }
    )


def _load_scene_features(
    model: BeamBenchImageAEGPSDirectModel,
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    scenes: Sequence[int],
    split: str,
    gps_scaler: Any,
    output_dir: Path,
    source_cache_root: Path,
    device: torch.device,
    ae_checkpoint: Path,
) -> dict[int, BeamBenchImageAEGPSFeatureDataset]:
    features: dict[int, BeamBenchImageAEGPSFeatureDataset] = {}
    for scene in scenes:
        cached = _read_existing_feature_cache(source_cache_root, split=split, scene=int(scene))
        if cached is not None:
            features[int(scene)] = cached
            continue
        scene_cfg = _scene_specific_cfg(cfg, int(scene))
        dataset = _build_split_dataset(scene_cfg, split=split, gps_scaler=gps_scaler, gps_normalize=scene_cfg.gps_normalize)
        feature_dataset, _ = _load_or_build_ae_feature_dataset(
            model,
            dataset,
            scene_cfg,
            output_dir=output_dir / "feature_cache_sources" / f"{split}_scene{int(scene)}",
            split=split,
            device=device,
            ae_checkpoint=ae_checkpoint,
        )
        features[int(scene)] = feature_dataset
    return features


def _read_existing_feature_cache(
    source_cache_root: Path,
    *,
    split: str,
    scene: int,
) -> BeamBenchImageAEGPSFeatureDataset | None:
    path = (
        source_cache_root
        / "feature_cache_sources"
        / f"{split}_scene{int(scene)}"
        / "feature_cache"
        / f"{split}_camera_ae_latents.pt"
    )
    if not path.exists():
        return None
    payload = _torch_load(path, map_location="cpu")
    if not all(key in payload for key in ("image_latent", "gps", "target")):
        return None
    return BeamBenchImageAEGPSFeatureDataset(
        image_latent=payload["image_latent"],
        gps=payload["gps"],
        target=payload["target"],
        metadata=payload.get("metadata", []),
        split=split,
    )


def _full_logits(
    model: BeamBenchImageAEGPSDirectModel,
    dataset: BeamBenchImageAEGPSFeatureDataset,
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    loader = _build_loader(dataset, batch_size=cfg.fusion_batch_size, shuffle=False, num_workers=cfg.num_workers, cfg=cfg)
    logits: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image_latent = batch["image_latent"].to(device=device, dtype=torch.float32, non_blocking=cfg.non_blocking_transfer)
            gps = batch["gps"].to(device=device, dtype=torch.float32, non_blocking=cfg.non_blocking_transfer)
            with _autocast_context(amp_enabled, device, amp_dtype):
                logits.append(model.forward_from_latent(image_latent, gps).detach().cpu())
    return torch.cat(logits, dim=0)


def _zero_camera_logits(
    model: BeamBenchImageAEGPSDirectModel,
    dataset: BeamBenchImageAEGPSFeatureDataset,
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    loader = _build_loader(dataset, batch_size=cfg.fusion_batch_size, shuffle=False, num_workers=cfg.num_workers, cfg=cfg)
    logits: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image_latent = torch.zeros_like(batch["image_latent"], dtype=torch.float32).to(
                device=device,
                non_blocking=cfg.non_blocking_transfer,
            )
            gps = batch["gps"].to(device=device, dtype=torch.float32, non_blocking=cfg.non_blocking_transfer)
            with _autocast_context(amp_enabled, device, amp_dtype):
                logits.append(model.forward_from_latent(image_latent, gps).detach().cpu())
    return torch.cat(logits, dim=0)


def _train_gps_only(
    train_features: Mapping[int, BeamBenchImageAEGPSFeatureDataset],
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    patience: int,
) -> tuple[GPSOnlyMLP, list[dict[str, Any]]]:
    gps = torch.cat([dataset.gps for dataset in train_features.values()], dim=0)
    labels = torch.cat([dataset.target for dataset in train_features.values()], dim=0)
    base_dataset = TensorDataset(gps, labels)
    val_count = max(1, int(round(len(base_dataset) * max(float(cfg.fusion_val_fraction or 0.1), 0.0))))
    val_count = min(val_count, len(base_dataset) - 1)
    train_count = len(base_dataset) - val_count
    train_subset, val_subset = random_split(
        base_dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(int(cfg.seed)),
    )
    train_loader = DataLoader(train_subset, batch_size=cfg.fusion_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.fusion_batch_size, shuffle=False)

    model = GPSOnlyMLP(
        input_dim=cfg.gps_input_size,
        hidden_dim=cfg.fusion_hidden_dim,
        num_beams=cfg.num_beams,
        dropout=cfg.fusion_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.fusion_lr, weight_decay=cfg.fusion_weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_count = 0
        for gps_batch, label_batch in train_loader:
            gps_batch = gps_batch.to(device=device, dtype=torch.float32)
            label_batch = label_batch.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(gps_batch)
            loss = F.cross_entropy(logits, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(label_batch.numel())
            total_count += int(label_batch.numel())
        val_logits, val_labels = _gps_eval_tensors(model, val_loader, device=device)
        metrics = _metric_summary(val_logits, val_labels, cfg)
        score = float(metrics["official_top3_dba"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": total_loss / max(total_count, 1),
                "val_official_top3_dba": score,
                "val_official_top1_acc": float(metrics["official_top1_acc"]),
                "best_val_official_top3_dba": best_score,
            }
        )
        if stale >= int(patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "gps_input_size": cfg.gps_input_size,
                "hidden_dim": cfg.fusion_hidden_dim,
                "num_beams": cfg.num_beams,
                "dropout": cfg.fusion_dropout,
            },
            "history": history,
        },
        output_dir / "matched_gps_only_mlp.pt",
    )
    return model, history


def _gps_eval_tensors(model: GPSOnlyMLP, loader: DataLoader, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for gps_batch, label_batch in loader:
            logits.append(model(gps_batch.to(device=device, dtype=torch.float32)).detach().cpu())
            labels.append(label_batch.detach().cpu().to(dtype=torch.long))
    return torch.cat(logits, dim=0), torch.cat(labels, dim=0)


def _gps_logits(
    model: GPSOnlyMLP,
    gps: torch.Tensor,
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    loader = DataLoader(TensorDataset(gps), batch_size=cfg.fusion_batch_size, shuffle=False)
    model.eval()
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for (gps_batch,) in loader:
            with _autocast_context(amp_enabled, device, amp_dtype):
                logits.append(model(gps_batch.to(device=device, dtype=torch.float32)).detach().cpu())
    return torch.cat(logits, dim=0)


def _metric_summary(logits: torch.Tensor, labels: torch.Tensor, cfg: ImageAEGPSDirectTrainingConfig) -> dict[str, Any]:
    return beambench_metric_summary_from_logits(
        logits,
        labels,
        num_beams=cfg.num_beams,
        topk=cfg.topk,
        dba_delta=cfg.dba_delta,
        circular=True,
    )


def _sample_rows(
    *,
    scene: int,
    labels: torch.Tensor,
    gps: torch.Tensor,
    full_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    baseline_name: str,
    metadata: Sequence[Mapping[str, Any]],
    dba_delta: float,
) -> list[dict[str, Any]]:
    label_np = labels.detach().cpu().numpy().astype(np.int64)
    gps_np = gps.detach().cpu().numpy().reshape(int(gps.shape[0]), -1)
    full_top = torch.topk(full_logits, k=min(5, int(full_logits.shape[-1])), dim=-1).indices.numpy()
    base_top = torch.topk(baseline_logits, k=min(5, int(baseline_logits.shape[-1])), dim=-1).indices.numpy()
    full_score = _per_sample_top3_dba(full_top, label_np, delta=dba_delta)
    base_score = _per_sample_top3_dba(base_top, label_np, delta=dba_delta)
    rows = []
    for index in range(label_np.shape[0]):
        row = {
            "scene": int(scene),
            "row_index": int(index),
            "baseline": baseline_name,
            "target_beam": int(label_np[index]),
            "full_top1": int(full_top[index, 0]),
            "baseline_top1": int(base_top[index, 0]),
            "full_top3": json.dumps([int(item) for item in full_top[index, :3].tolist()]),
            "baseline_top3": json.dumps([int(item) for item in base_top[index, :3].tolist()]),
            "full_top3_hit": bool(np.any(full_top[index, :3] == label_np[index])),
            "baseline_top3_hit": bool(np.any(base_top[index, :3] == label_np[index])),
            "full_sample_dba": float(full_score[index]),
            "baseline_sample_dba": float(base_score[index]),
            "sample_dba_gain": float(full_score[index] - base_score[index]),
            "gps_0": float(gps_np[index, 0]) if gps_np.shape[1] > 0 else 0.0,
            "gps_1": float(gps_np[index, 1]) if gps_np.shape[1] > 1 else 0.0,
        }
        row.update(dict(metadata[index]) if index < len(metadata) else {})
        rows.append(row)
    return rows


def _per_sample_top3_dba(topk: np.ndarray, labels: np.ndarray, *, delta: float) -> np.ndarray:
    kk = min(3, topk.shape[1])
    scores = np.zeros(labels.shape[0], dtype=np.float64)
    for k_index in range(kk):
        dist = np.abs(topk[:, : k_index + 1] - labels[:, None]) / max(float(delta), 1e-8)
        scores += 1.0 - np.minimum(np.min(dist, axis=1), 1.0)
    return scores / max(kk, 1)


def _ambiguity_report(rows: Sequence[Mapping[str, Any]], *, neighbor_k: int) -> dict[str, Any]:
    matched = [dict(row) for row in rows if str(row.get("baseline")) == "matched_gps_only"]
    by_scene: dict[int, list[dict[str, Any]]] = {}
    for row in matched:
        by_scene.setdefault(int(row["scene"]), []).append(row)
    scene_reports = []
    for scene, scene_rows in sorted(by_scene.items()):
        gps = np.asarray([[float(row["gps_0"]), float(row["gps_1"])] for row in scene_rows], dtype=np.float64)
        labels = np.asarray([int(row["target_beam"]) for row in scene_rows], dtype=np.int64)
        gains = np.asarray([float(row["sample_dba_gain"]) for row in scene_rows], dtype=np.float64)
        full_hit = np.asarray([bool(row["full_top3_hit"]) for row in scene_rows], dtype=bool)
        base_hit = np.asarray([bool(row["baseline_top3_hit"]) for row in scene_rows], dtype=bool)
        ambiguity = _gps_neighbor_label_disagreement(gps, labels, k=neighbor_k)
        q25, q50, q75 = np.quantile(ambiguity, [0.25, 0.5, 0.75])
        low = ambiguity <= q25
        high = ambiguity >= q75
        rescued = (~base_hit) & full_hit
        worsened = base_hit & (~full_hit)
        pos_gain = np.maximum(gains, 0.0)
        scene_reports.append(
            {
                "scene": int(scene),
                "sample_count": int(len(scene_rows)),
                "ambiguity_q25": float(q25),
                "ambiguity_median": float(q50),
                "ambiguity_q75": float(q75),
                "mean_gain_all": float(np.mean(gains)),
                "mean_gain_low_ambiguity": float(np.mean(gains[low])) if np.any(low) else 0.0,
                "mean_gain_high_ambiguity": float(np.mean(gains[high])) if np.any(high) else 0.0,
                "positive_gain_share_high_ambiguity": float(np.sum(pos_gain[high]) / max(np.sum(pos_gain), 1e-12)),
                "rescue_count": int(np.sum(rescued)),
                "rescue_rate": float(np.mean(rescued)),
                "rescue_high_ambiguity_share": float(np.mean(high[rescued])) if np.any(rescued) else 0.0,
                "worsened_count": int(np.sum(worsened)),
                "worsened_rate": float(np.mean(worsened)),
                "gain_ambiguity_correlation": float(np.corrcoef(gains, ambiguity)[0, 1])
                if len(gains) > 2 and np.std(gains) > 0 and np.std(ambiguity) > 0
                else 0.0,
            }
        )
        for row, amb in zip(scene_rows, ambiguity, strict=True):
            row["gps_neighbor_label_disagreement"] = float(amb)
    weighted_gain_high = sum(float(r["mean_gain_high_ambiguity"]) * int(r["sample_count"]) for r in scene_reports)
    total = sum(int(r["sample_count"]) for r in scene_reports)
    return {
        "neighbor_k": int(neighbor_k),
        "definition": "fraction of same-scene nearest GPS neighbors whose beam label differs from the sample label",
        "scenes": scene_reports,
        "weighted_mean_gain_high_ambiguity": weighted_gain_high / max(total, 1),
    }


def _gps_neighbor_label_disagreement(gps: np.ndarray, labels: np.ndarray, *, k: int) -> np.ndarray:
    n = int(gps.shape[0])
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    kk = min(max(1, int(k)), n - 1)
    dist = np.sum((gps[:, None, :] - gps[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn_idx = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    neighbor_labels = labels[nn_idx]
    return np.mean(neighbor_labels != labels[:, None], axis=1)


def _run_ae_reconstruction_diagnostics(
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    eval_features: Mapping[int, BeamBenchImageAEGPSFeatureDataset],
    ae_checkpoints: Sequence[Path],
    output_dir: Path,
    device: torch.device,
    sample_count_per_scene: int,
) -> dict[str, Any]:
    checkpoints = [path for path in ae_checkpoints if path.exists()]
    models = []
    for path in checkpoints:
        payload = _torch_load(path, map_location="cpu")
        metadata = dict(payload.get("model_metadata") or {})
        latent_dim = int(metadata.get("latent_dim", 128))
        model = CameraAutoEncoder(latent_dim=latent_dim, image_channels=cfg.image_channels, image_size=cfg.image_size).to(device)
        model.load_state_dict(payload.get("model_state_dict", payload))
        model.eval()
        models.append((path, latent_dim, model))
    if not models:
        return {"status": "skipped", "reason": "no AE checkpoints found"}

    raw_batches = []
    display_rows: list[tuple[str, torch.Tensor, list[torch.Tensor]]] = []
    metrics_by_model: dict[str, list[dict[str, float]]] = {f"{latent_dim}d": [] for _, latent_dim, _ in models}
    for scene, feature_dataset in eval_features.items():
        selected = _even_indices(len(feature_dataset), max(1, int(sample_count_per_scene)))
        scene_cfg = _scene_specific_cfg(cfg, scene)
        raw_dataset = _build_split_dataset(scene_cfg, split="test", gps_scaler=None, gps_normalize=False)
        for sample_index in selected:
            item = raw_dataset[int(sample_index)]
            image = item["image"][:, -1, ...] if item["image"].ndim == 5 else item["image"][-1]
            raw_batches.append((scene, int(sample_index), image))
    if not raw_batches:
        return {"status": "skipped", "reason": "no samples selected"}

    images = torch.stack([item[2] for item in raw_batches], dim=0).to(device=device, dtype=torch.float32)
    recon_by_model: list[tuple[str, torch.Tensor]] = []
    with torch.no_grad():
        for _, latent_dim, model in models:
            recon = model(images)["reconstruction"].detach().cpu()
            recon_by_model.append((f"{latent_dim}d", recon))
            metrics_by_model[f"{latent_dim}d"].extend(_reconstruction_metric_rows(images.detach().cpu(), recon))

    for row_index, (scene, sample_index, image) in enumerate(raw_batches[: min(24, len(raw_batches))]):
        recon_images = [recon[row_index] for _, recon in recon_by_model]
        display_rows.append((f"S{scene} #{sample_index}", image.detach().cpu(), recon_images))
    _write_reconstruction_grid(output_dir / "ae_reconstruction_grid.jpg", display_rows, labels=[name for name, _ in recon_by_model])

    summary: dict[str, Any] = {"status": "complete", "sample_count": len(raw_batches), "models": {}}
    for name, rows in metrics_by_model.items():
        summary["models"][name] = _mean_metric_dict(rows)
    if "128d" in summary["models"] and "512d" in summary["models"]:
        summary["delta_512_minus_128"] = {
            key: float(summary["models"]["512d"][key] - summary["models"]["128d"][key])
            for key in summary["models"]["512d"]
            if key in summary["models"]["128d"]
        }
    return summary


def _even_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = min(length, max(1, int(count)))
    return sorted({int(round(item)) for item in np.linspace(0, length - 1, count)})


def _reconstruction_metric_rows(images: torch.Tensor, recon: torch.Tensor) -> list[dict[str, float]]:
    rows = []
    edge_image = _sobel_edges(images)
    edge_recon = _sobel_edges(recon)
    low_image = F.interpolate(images, size=(16, 16), mode="bilinear", align_corners=False)
    low_recon = F.interpolate(recon, size=(16, 16), mode="bilinear", align_corners=False)
    for index in range(int(images.shape[0])):
        mse = float(F.mse_loss(recon[index], images[index]).item())
        edge_mse = float(F.mse_loss(edge_recon[index], edge_image[index]).item())
        low_mse = float(F.mse_loss(low_recon[index], low_image[index]).item())
        rows.append(
            {
                "mse": mse,
                "mae": float(torch.mean(torch.abs(recon[index] - images[index])).item()),
                "psnr": float(20.0 * math.log10(2.0 / max(math.sqrt(mse), 1e-12))),
                "edge_mse": edge_mse,
                "lowfreq_mse_16": low_mse,
                "edge_to_pixel_mse": float(edge_mse / max(mse, 1e-12)),
            }
        )
    return rows


def _sobel_edges(images: torch.Tensor) -> torch.Tensor:
    gray = images.mean(dim=1, keepdim=True)
    kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=gray.dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=gray.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _write_reconstruction_grid(path: Path, rows: Sequence[tuple[str, torch.Tensor, list[torch.Tensor]]], *, labels: Sequence[str]) -> None:
    tile = 96
    label_h = 20
    columns = 1 + len(labels)
    width = columns * tile
    height = (len(rows) + 1) * (tile + label_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    headers = ["orig", *labels]
    for col, header in enumerate(headers):
        draw.text((col * tile + 4, 4), header, fill=(0, 0, 0))
    for row_index, (name, image, recon_images) in enumerate(rows):
        y = (row_index + 1) * (tile + label_h)
        draw.text((4, y), name, fill=(0, 0, 0))
        images = [image, *recon_images]
        for col, tensor in enumerate(images):
            pil = _tensor_to_pil(tensor).resize((tile, tile), Image.BILINEAR)
            canvas.paste(pil, (col * tile, y + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = ((tensor.detach().cpu().clamp(-1, 1) * 0.5 + 0.5) * 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _mean_metric_dict(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = list(rows[0].keys())
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _flatten_scene_reports(scene_reports: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in scene_reports:
        metrics = dict(report["metrics"])
        row = {"scene": int(report["scene"]), "model": str(report["model"])}
        row.update({key: value for key, value in metrics.items() if isinstance(value, (int, float, bool, str))})
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
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


def _compact_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    scene_rows = []
    grouped: dict[int, dict[str, Any]] = {}
    for report in summary["scene_metrics"]:
        grouped.setdefault(int(report["scene"]), {})[str(report["model"])] = float(report["metrics"]["official_top3_dba"])
    for scene, values in sorted(grouped.items()):
        row = {"scene": scene}
        row.update(values)
        if "camera_ae_gps" in values and "matched_gps_only_mlp" in values:
            row["camera_minus_matched_gps"] = values["camera_ae_gps"] - values["matched_gps_only_mlp"]
        scene_rows.append(row)
    return {
        "scene_dba": scene_rows,
        "ambiguity": summary["ambiguity"],
        "ae_reconstruction": summary["ae_reconstruction"],
        "artifacts": summary["artifacts"],
    }


if __name__ == "__main__":
    main()
