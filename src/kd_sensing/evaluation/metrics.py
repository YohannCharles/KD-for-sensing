"""MMW beam-prediction metrics."""

import numpy as np
import torch


DBA_TOP_K = 3


def circular_beam_distance(prediction, target, *, num_beams: int = 64):
    """Return shortest wrapped distance between beam ids."""
    beams = int(num_beams)
    if beams <= 0:
        raise ValueError(f"num_beams must be positive, got {num_beams}.")
    if torch.is_tensor(prediction) or torch.is_tensor(target):
        pred = torch.as_tensor(prediction)
        truth = torch.as_tensor(target, device=pred.device)
        absolute = (pred.to(torch.long).remainder(beams) - truth.to(pred.device, torch.long).remainder(beams)).abs()
        return torch.minimum(absolute, torch.as_tensor(beams, device=absolute.device) - absolute)
    absolute = np.abs(np.asarray(prediction, dtype=np.int64) % beams - np.asarray(target, dtype=np.int64) % beams)
    distance = np.minimum(absolute, beams - absolute)
    return int(distance.item()) if distance.ndim == 0 else distance


def circular_topk_min_distance(
    outputs,
    target,
    *,
    k: int = 3,
    num_beams: int | None = None,
    distance_mode: str = "circular",
):
    """Return each target's closest top-k distance in the requested beam geometry."""
    if torch.is_tensor(outputs) or torch.is_tensor(target):
        scores = torch.as_tensor(outputs)
        labels = torch.as_tensor(target, device=scores.device)
        topk = scores.topk(max(1, min(int(k), scores.shape[-1])), dim=-1).indices
        return _beam_distance(
            topk,
            labels.to(scores.device, torch.long).unsqueeze(-1),
            int(num_beams or scores.shape[-1]),
            distance_mode,
        ).min(dim=-1).values
    scores = np.asarray(outputs)
    topk = np.argsort(scores, axis=-1)[..., -max(1, min(int(k), scores.shape[-1])) :]
    labels = np.expand_dims(np.asarray(target, dtype=np.int64), -1)
    if _normalized_distance_mode(distance_mode) == "circular":
        distances = circular_beam_distance(topk, labels, num_beams=num_beams or scores.shape[-1])
    else:
        distances = np.abs(topk.astype(np.int64) - labels)
    return distances.min(axis=-1)


def beam_classification_circular_summary(
    outputs,
    labels,
    *,
    num_beams: int | None = None,
    dba_delta: float = 5.0,
    topk: tuple[int, ...] = (1, 3, 5),
    ignore_index: int = -100,
    distance_mode: str = "circular",
) -> dict[str, float]:
    """Summarize one beam-classification horizon for the fixed-mask matrix."""
    scores = torch.as_tensor(outputs).detach().cpu()
    target = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    if scores.ndim == 3 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if scores.ndim != 2 or target.ndim != 1 or scores.shape[0] != target.shape[0]:
        raise ValueError("outputs and labels must have shapes [N, C] and [N].")

    beams = int(num_beams or scores.shape[-1])
    valid = target.ne(ignore_index) & target.ge(0) & target.lt(beams)
    result = {"DBA": 0.0, "mean_error": 0.0, "within_3": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0}
    if not bool(valid.any()):
        return result

    scores, target = scores[valid], target[valid]
    distances = _beam_distance(scores.argmax(dim=-1), target, beams, distance_mode).to(torch.float32)
    result.update(
        DBA=float((1.0 - (distances / max(float(dba_delta), 1e-8)).clamp(max=1.0)).mean().item()),
        mean_error=float(distances.mean().item()),
        within_3=float(distances.le(3).float().mean().item()),
    )
    for k in topk:
        key = f"top{int(k)}"
        result[key] = float(
            circular_topk_min_distance(scores, target, k=int(k), num_beams=beams, distance_mode=distance_mode)
            .eq(0)
            .float()
            .mean()
            .item()
        )
    return result


def calculate_topk_accuracy(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    k_values: list[int] | tuple[int, ...] = (1, 2, 3, 5, 10),
):
    """Return per-horizon hard-label top-k accuracy and valid-label totals."""
    scores, target = _prediction_tensors(outputs, labels)
    valid = target.ne(-100)
    total = valid.sum(dim=0)
    if not k_values:
        return {}, total.cpu().numpy()
    max_k = min(max(k_values), scores.shape[-1])
    predictions = scores.topk(max_k, dim=-1).indices
    matches = predictions.eq(target.unsqueeze(-1)) & valid.unsqueeze(-1)
    return {
        k: (matches[..., : min(k, max_k)].any(dim=-1).sum(dim=0) / total.clamp_min(1)).cpu().numpy()
        for k in k_values
    }, total.cpu().numpy()


def calculate_dba_score(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    delta: float = 5,
    *,
    distance_mode: str = "circular",
):
    """Return the mean progressive top-three beam-distance accuracy per horizon."""
    scores, target = _prediction_tensors(outputs, labels)
    valid = target.ne(-100)
    predictions = scores.topk(min(DBA_TOP_K, scores.shape[-1]), dim=-1).indices
    distances = _beam_distance(predictions, target.unsqueeze(-1), scores.shape[-1], distance_mode).to(torch.float32)
    progressive = 1.0 - torch.cummin((distances / max(float(delta), 1e-8)).clamp(max=1.0), dim=-1).values
    per_sample = progressive.mean(dim=-1)
    return ((per_sample * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1)).cpu().numpy()


def _prediction_tensors(outputs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.as_tensor(outputs)
    target = torch.as_tensor(labels, device=scores.device, dtype=torch.long)
    if scores.ndim == 2:
        scores = scores.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)
    if scores.ndim != 3 or target.ndim != 2 or scores.shape[:2] != target.shape:
        raise ValueError("outputs and labels must have shapes [B, H, C] and [B, H].")
    return scores, target


def _beam_distance(prediction: torch.Tensor, target: torch.Tensor, num_beams: int, mode: str) -> torch.Tensor:
    if _normalized_distance_mode(mode) == "circular":
        return circular_beam_distance(prediction, target, num_beams=num_beams)
    return (prediction.to(torch.long) - target.to(torch.long)).abs()


def _normalized_distance_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"circular", "wrap", "wrapped"}:
        return "circular"
    if normalized in {"linear", "official", "beambench", "non_circular", "noncircular"}:
        return "linear"
    raise ValueError("distance_mode must be 'circular' or 'linear'.")


__all__ = [
    "beam_classification_circular_summary",
    "calculate_dba_score",
    "calculate_topk_accuracy",
    "circular_beam_distance",
    "circular_topk_min_distance",
]
