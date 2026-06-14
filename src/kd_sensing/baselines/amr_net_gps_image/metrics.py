from __future__ import annotations

from typing import Any, Iterable

import torch

from kd_sensing.evaluation.metrics import beam_classification_circular_summary


def paper_aligned_metric_summary(
    logits: Any,
    labels: Any,
    *,
    model_group: str,
    scene: int | str = 23,
    split: str = "test",
    metric_profile: str = "amr_net_gps_image_top1_top3_top5",
    claim_status: str = "local_substitute",
    seed: int | None = None,
    topk: Iterable[int] = (1, 3, 5),
    num_beams: int = 64,
    mock_data: bool = False,
) -> dict[str, Any]:
    scores = logits.detach().cpu() if torch.is_tensor(logits) else torch.as_tensor(logits)
    target = labels.detach().cpu().to(torch.long) if torch.is_tensor(labels) else torch.as_tensor(labels, dtype=torch.long)
    if scores.ndim == 3 and scores.shape[1] == 1:
        scores = scores[:, 0, :]
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    summary = beam_classification_circular_summary(
        scores,
        target,
        num_beams=int(num_beams),
        dba_delta=5.0,
        topk=tuple(int(item) for item in topk),
    )
    result: dict[str, Any] = {
        "model_group": str(model_group),
        "scene": f"scene{scene}" if str(scene).isdigit() else str(scene),
        "scene_id": int(scene) if str(scene).isdigit() else scene,
        "split": str(split),
        "sample_count": int(summary.get("sample_count", 0)),
        "valid_label_count": int(summary.get("valid_label_count", 0)),
        "top1": float(summary.get("top1", 0.0)),
        "top3": float(summary.get("top3", 0.0)),
        "top5": float(summary.get("top5", 0.0)),
        "paper_top1_acc": float(summary.get("top1", 0.0)),
        "paper_top3_acc": float(summary.get("top3", 0.0)),
        "paper_top5_acc": float(summary.get("top5", 0.0)),
        "DBA": float(summary.get("DBA", 0.0)),
        "beam_distance_mean_circular": float(summary.get("mean_circular_error", 0.0)),
        "beam_distance_median_circular": float(summary.get("median_circular_error", 0.0)),
        "overhead_reduction_profile": "topk_accuracy_only; paper overhead requires official protocol evidence",
        "metric_profile": str(metric_profile),
        "metric_contract": "Top-k exact beam accuracy over 64-beam classifier outputs; DBA is local circular diagnostic",
        "seed": seed,
        "claim_status": str(claim_status),
        "mock_data": bool(mock_data),
    }
    return result
