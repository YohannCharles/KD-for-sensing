from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from kd_sensing.baselines.beambench.metrics import beambench_metric_summary_from_logits
from kd_sensing.baselines.beambench.image_ae_gps_config import (
    ImageAEGPSDirectTrainingConfig,
    _autocast_context,
    _resolve_amp_dtype,
)
from kd_sensing.baselines.beambench.image_ae_gps_datasets import _metadata_rows
from kd_sensing.baselines.beambench.image_ae_gps_models import (
    BeamBenchImageAEGPSDirectModel,
    _classifier_logits_from_batch,
)
from kd_sensing.baselines.beambench.image_ae_gps_reports import _write_csv_rows


def evaluate_image_ae_gps_model(
    model: BeamBenchImageAEGPSDirectModel,
    loader: DataLoader,
    cfg: ImageAEGPSDirectTrainingConfig,
    *,
    device: torch.device,
    predictions_path: str | Path | None,
    amp_enabled: bool | None = None,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    use_amp = bool(cfg.amp) and device.type == "cuda" if amp_enabled is None else bool(amp_enabled)
    dtype = _resolve_amp_dtype(cfg.amp_dtype) if amp_dtype is None else amp_dtype
    with torch.no_grad():
        for batch in loader:
            labels = batch["target"].to(device=device, dtype=torch.long, non_blocking=cfg.non_blocking_transfer)
            with _autocast_context(use_amp, device, dtype):
                logits = _classifier_logits_from_batch(
                    model,
                    batch,
                    device=device,
                    non_blocking=cfg.non_blocking_transfer,
                )
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if predictions_path is not None:
                probs = F.softmax(logits.detach().cpu(), dim=-1)
                topk = torch.topk(logits.detach().cpu(), k=min(5, int(logits.shape[-1])), dim=-1).indices
                metadata = _metadata_rows(batch.get("metadata"), count=int(labels.numel()))
                for idx in range(int(labels.numel())):
                    row = {
                        "row_index": len(rows),
                        "target_beam": int(labels.detach().cpu()[idx].item()),
                        "pred_top1": int(topk[idx, 0].item()),
                        "pred_top3": json.dumps([int(value) for value in topk[idx, : min(3, topk.shape[1])].tolist()]),
                        "top1_probability": float(probs[idx, int(topk[idx, 0].item())].item()),
                    }
                    row.update(metadata[idx] if idx < len(metadata) else {})
                    rows.append(row)
    logits_t = torch.cat(all_logits, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    metrics = beambench_metric_summary_from_logits(
        logits_t,
        labels_t,
        num_beams=cfg.num_beams,
        topk=cfg.topk,
        dba_delta=cfg.dba_delta,
        circular=True,
    )
    if predictions_path is not None:
        _write_csv_rows(Path(predictions_path), rows)
    return {"metrics": metrics, "sample_count": int(labels_t.numel())}


__all__ = ["evaluate_image_ae_gps_model"]
