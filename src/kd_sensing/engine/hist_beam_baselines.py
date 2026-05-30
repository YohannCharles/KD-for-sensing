from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from kd_sensing.engine.batch import prepare_history_anchor_inputs
from kd_sensing.engine.hist_beam_residuals import num_delta_classes_from_config
from kd_sensing.engine.runtime import prepare_task_batch, prepare_task_labels, transfer_non_blocking
from kd_sensing.evaluation.hist_beam_outputs import beam_histogram_metrics, source_prior_collapse_diagnostics


def collect_source_beam_reference(
    dataloader,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    output_path: str | Path | None = None,
    split: str = "source_train",
) -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    num_pred = int(model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 1)))
    downsample_ratio = int(model_cfg.get("downsample_ratio", 1))
    num_classes = int(model_cfg.get("num_classes", model_cfg.get("student", {}).get("num_classes", 64)))
    num_delta_classes = num_delta_classes_from_config(cfg, default=num_classes)
    non_blocking = transfer_non_blocking(cfg)
    labels: list[torch.Tensor] = []
    last_beams: list[torch.Tensor] = []
    for raw_batch in dataloader:
        batch = prepare_task_batch(raw_batch)
        if "target_beam" not in batch:
            continue
        target = prepare_task_labels(
            batch,
            num_pred=num_pred,
            downsample_ratio=downsample_ratio,
            device=device,
            non_blocking=non_blocking,
        )
        labels.append(target.detach().cpu())
        if "input_beam" in batch:
            history = prepare_history_anchor_inputs(
                batch,
                num_pred=num_pred,
                num_classes=num_delta_classes,
                downsample_ratio=downsample_ratio,
                device=device,
                enabled=True,
                include_residual_labels=False,
                non_blocking=non_blocking,
            )
            last_beams.append(history["last_beam_batch"].detach().cpu())
    labels_t = torch.cat(labels, dim=0) if labels else None
    last_t = torch.cat(last_beams, dim=0) if last_beams else None
    metadata: dict[str, Any] = {
        "split": split,
        "num_classes": num_classes,
        "num_delta_classes": num_delta_classes,
        "sample_count": int(labels_t.shape[0]) if labels_t is not None else 0,
        "history_anchor_sample_count": int(last_t.shape[0]) if last_t is not None else 0,
    }
    if labels_t is not None:
        metadata.update(beam_histogram_metrics(labels_t, num_classes=num_classes, prefix=split))
    artifact = {
        "version": "hist_beam_source_beam_reference_v1",
        "labels": labels_t,
        "last_beams": last_t,
        "metadata": metadata,
    }
    if output_path is not None:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        artifact["path"] = str(target_path)
        metadata["source_beam_reference_path"] = str(target_path)
        torch.save(artifact, target_path)
    return artifact


def load_source_beam_reference(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    artifact = torch.load(Path(path), map_location=map_location)
    if not isinstance(artifact, dict) or artifact.get("version") != "hist_beam_source_beam_reference_v1":
        raise ValueError(f"Invalid HiST-Beam source beam reference artifact: {path}")
    return artifact


def attach_source_beam_reference(
    cfg: dict[str, Any],
    path: str | Path | None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any] | None:
    if not path:
        return None
    artifact = load_source_beam_reference(path, map_location=map_location)
    labels = artifact.get("labels")
    last_beams = artifact.get("last_beams")
    metadata = artifact.get("metadata", {}) if isinstance(artifact.get("metadata"), dict) else {}
    if torch.is_tensor(labels) and torch.is_tensor(last_beams):
        hist_cfg = cfg.setdefault("hist_beam", {})
        current = hist_cfg.get("markov_baseline", {}) if isinstance(hist_cfg.get("markov_baseline"), dict) else {}
        hist_cfg["markov_baseline"] = {
            **current,
            "enabled": current.get("enabled", True),
            "split": metadata.get("split", "source_train"),
            "labels": labels,
            "last_beams": last_beams,
            "smoothing": float(current.get("smoothing", 1.0)),
        }
    return artifact


def source_prior_collapse_metrics(
    source_reference: dict[str, Any] | None,
    metrics: dict[str, Any],
    *,
    top_fraction_threshold: float = 0.5,
) -> dict[str, Any]:
    metadata = source_reference.get("metadata", {}) if isinstance(source_reference, dict) else {}
    source_hist = metadata.get("source_train_true_beam_histogram")
    result = source_prior_collapse_diagnostics(
        source_histogram=source_hist,
        target_true_histogram=metrics.get("target_test_true_beam_histogram"),
        predicted_histogram=metrics.get("target_test_predicted_beam_histogram"),
        top_fraction_threshold=top_fraction_threshold,
    )
    if source_hist is not None:
        result["source_train_true_beam_histogram"] = list(source_hist)
    if metadata.get("source_beam_reference_path"):
        result["source_beam_reference_path"] = metadata.get("source_beam_reference_path")
    return result


__all__ = [
    "attach_source_beam_reference",
    "collect_source_beam_reference",
    "load_source_beam_reference",
    "source_prior_collapse_metrics",
]
