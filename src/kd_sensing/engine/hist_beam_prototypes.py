from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from kd_sensing.engine.hist_beam_labels import hist_beam_labels
from kd_sensing.engine.runtime import run_model_step, transfer_non_blocking


def generate_source_prototypes(
    model,
    dataloader,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    output_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    model_cfg = cfg["model"]
    student_cfg = model_cfg.get("student", model_cfg)
    num_classes = int(student_cfg.get("num_classes", model_cfg.get("num_classes", 64)))
    group_size = int(student_cfg.get("group_size", cfg.get("hist_beam", {}).get("group_size", 8)))
    num_groups = num_classes // group_size
    shared_sum = None
    private_sum = None
    counts = torch.zeros(num_groups, dtype=torch.long, device=device)
    with torch.no_grad():
        for batch in dataloader:
            step = run_model_step(
                model,
                cfg["experiment"].get("task", "fusion"),
                batch,
                model_cfg=student_cfg,
                seq_length=model_cfg.get("seq_length_student", cfg.get("data", {}).get("dataset", {}).get("seq_len", 8)),
                num_pred=model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 1)),
                device=device,
                downsample_ratio=model_cfg.get("downsample_ratio", 1),
                non_blocking=transfer_non_blocking(cfg),
            )
            labels = step.labels
            if labels is None:
                continue
            shared = step.model_output.diagnostics.get("shared_representation")
            private = step.model_output.diagnostics.get("private_representation")
            if not torch.is_tensor(shared) or not torch.is_tensor(private):
                continue
            coarse, _ = hist_beam_labels(labels, num_classes=num_classes, group_size=group_size)
            shared_flat = shared.reshape(-1, shared.shape[-1])
            private_flat = private.reshape(-1, private.shape[-1])
            coarse_flat = coarse.reshape(-1)
            valid = coarse_flat.ge(0)
            if shared_sum is None:
                shared_sum = torch.zeros(num_groups, shared_flat.shape[-1], dtype=shared_flat.dtype, device=device)
                private_sum = torch.zeros(num_groups, private_flat.shape[-1], dtype=private_flat.dtype, device=device)
            for group in range(num_groups):
                mask = valid & coarse_flat.eq(group)
                if not torch.any(mask):
                    continue
                shared_sum[group] += shared_flat[mask].sum(dim=0)
                private_sum[group] += private_flat[mask].sum(dim=0)
                counts[group] += int(mask.sum().item())
    if shared_sum is None:
        dim = int(student_cfg.get("d_model", model_cfg.get("d_model", student_cfg.get("feature_size", 64))))
        shared_sum = torch.zeros(num_groups, dim, device=device)
        private_sum = torch.zeros(num_groups, dim, device=device)
    denom = counts.clamp_min(1).to(dtype=shared_sum.dtype).unsqueeze(-1)
    artifact = {
        "version": "hist_beam_prototypes_v1",
        "shared_prototypes": (shared_sum / denom).detach().cpu(),
        "private_prototypes": (private_sum / denom).detach().cpu(),
        "counts": counts.detach().cpu(),
        "metadata": {
            "group_size": group_size,
            "num_groups": num_groups,
            "num_classes": num_classes,
            **(metadata or {}),
        },
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, target)
    return artifact


def load_source_prototypes(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    artifact = torch.load(Path(path), map_location=map_location)
    validate_prototype_artifact(artifact)
    return artifact


def validate_prototype_artifact(artifact: dict[str, Any]) -> None:
    for key in ("shared_prototypes", "private_prototypes", "counts", "metadata"):
        if key not in artifact:
            raise ValueError(f"Prototype artifact missing required key '{key}'.")
    shared = artifact["shared_prototypes"]
    private = artifact["private_prototypes"]
    counts = artifact["counts"]
    if not torch.is_tensor(shared) or not torch.is_tensor(private) or not torch.is_tensor(counts):
        raise ValueError("Prototype artifact shared/private/counts must be tensors.")
    if shared.shape != private.shape or shared.shape[0] != counts.shape[0]:
        raise ValueError("Prototype artifact shared/private/count shapes are inconsistent.")


def prototype_coverage_from_counts(
    counts: torch.Tensor,
    *,
    confidence_mask: torch.Tensor | None = None,
) -> dict[str, float | int | list[int]]:
    counts_cpu = counts.detach().cpu().to(torch.long)
    available = counts_cpu.gt(0)
    if confidence_mask is not None:
        mask = confidence_mask.detach().cpu().to(torch.bool)
        used = available & mask[: available.numel()]
    else:
        used = available
    total = int(counts_cpu.numel())
    available_count = int(available.sum().item())
    used_count = int(used.sum().item())
    return {
        "prototype_groups": total,
        "prototype_available_groups": available_count,
        "prototype_used_groups": used_count,
        "prototype_coverage": float(used_count / max(total, 1)),
        "empty_groups": [int(index) for index in torch.where(~available)[0].tolist()],
    }


__all__ = [
    "generate_source_prototypes",
    "load_source_prototypes",
    "prototype_coverage_from_counts",
    "validate_prototype_artifact",
]
