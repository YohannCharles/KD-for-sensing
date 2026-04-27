from __future__ import annotations

import json
from pathlib import Path

import torch

from kd_sensing.engine.batch import (
    forward_model,
    normalize_batch,
    prepare_fusion_inputs,
    prepare_image_inputs,
    prepare_labels,
    prepare_radar_inputs,
)
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy


def validate(model, dataloader, cfg: dict, criterion, device: torch.device, output_dir: str | Path | None = None):
    model.eval()
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    downsample_ratio = model_cfg.get("downsample_ratio", 1)
    seq_length = model_cfg.get("seq_length_student", 8)
    num_classes = model_cfg.get("num_classes", 64)
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = normalize_batch(batch)
            labels = prepare_labels(
                batch,
                num_pred=num_pred,
                downsample_ratio=downsample_ratio,
                device=device,
            )
            if task == "fusion":
                image_batch, radar_batch = prepare_fusion_inputs(
                    batch,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                )
                outputs, _, _ = forward_model(model, task, image_batch, radar_batch)
            elif task == "radar":
                radar_batch = prepare_radar_inputs(
                    batch,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                )
                outputs, _, _ = forward_model(model, task, radar_batch=radar_batch)
            else:
                image_batch = prepare_image_inputs(
                    batch,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                )
                outputs, _, _ = forward_model(model, task, image_batch)
            outputs = outputs[:, -(num_pred + 1) :, :]
            val_loss += criterion(outputs.reshape(-1, num_classes), labels.flatten()).item()
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())
    val_loss = val_loss / max(len(dataloader), 1)
    all_outputs_t = torch.cat(all_outputs, dim=0)
    all_labels_t = torch.cat(all_labels, dim=0)
    topk_acc, total = calculate_topk_accuracy(
        all_outputs_t,
        all_labels_t,
        cfg.get("evaluation", {}).get("k_values", [1, 2, 3, 5, 10]),
    )
    dba_score = calculate_dba_score(
        all_outputs_t,
        all_labels_t,
        cfg.get("evaluation", {}).get("dba_delta", 5),
    )
    metrics = {
        "loss": float(val_loss),
        "topk": {str(k): v.tolist() for k, v in topk_acc.items()},
        "dba": dba_score.tolist(),
        "total": total.tolist(),
    }
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    return metrics
