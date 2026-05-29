from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch

from kd_sensing.engine.batch import prepare_path_descriptors, prepare_path_semantic_labels, prepare_radio_semantic_labels
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
    progress_callback=None,
) -> dict[str, Any]:
    model.eval()
    model_cfg = cfg["model"]
    student_cfg = model_cfg.get("student", model_cfg)
    num_classes = int(student_cfg.get("num_classes", model_cfg.get("num_classes", 64)))
    group_size = int(student_cfg.get("group_size", cfg.get("hist_beam", {}).get("group_size", 8)))
    num_groups = num_classes // group_size
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    proto_cfg = hist_cfg.get("prototype", {}) if isinstance(hist_cfg.get("prototype"), dict) else {}
    radio_cfg = hist_cfg.get("radio_semantic", {}) if isinstance(hist_cfg.get("radio_semantic"), dict) else {}
    path_cfg = hist_cfg.get("path_semantic", {}) if isinstance(hist_cfg.get("path_semantic"), dict) else {}
    proto_type = str(hist_cfg.get("proto_type", proto_cfg.get("proto_type", "coarse"))).strip().lower()
    if proto_type == "none" and (path_cfg.get("enabled") or student_cfg.get("variant") in {"v8_path_proto", "adapter_path_proto"}):
        proto_type = "path"
    if proto_type == "none" and (radio_cfg.get("enabled") or student_cfg.get("variant") in {"v6_radio_proto", "adapter_radio_proto"}):
        proto_type = "radio_semantic"
    num_radio_classes = int(
        student_cfg.get(
            "num_radio_classes",
            radio_cfg.get("num_radio_classes", (num_classes // group_size) * int(radio_cfg.get("num_spread_bins", 3))),
        )
    )
    num_path_classes = int(
        student_cfg.get(
            "num_path_classes",
            path_cfg.get("num_path_classes", path_cfg.get("num_classes", 24)),
        )
    )
    shared_sum = None
    private_sum = None
    adapter_sum = None
    counts = torch.zeros(num_groups, dtype=torch.long, device=device)
    radio_sum = None
    radio_counts = torch.zeros(num_radio_classes, dtype=torch.long, device=device)
    path_sum = None
    path_descriptor_sum = None
    path_counts = torch.zeros(num_path_classes, dtype=torch.long, device=device)
    path_descriptor_dim = None
    processed_batches = 0
    processed_samples = 0
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            processed_batches += 1
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
            processed_samples += int(labels.shape[0])
            shared = step.model_output.diagnostics.get("shared_representation")
            private = step.model_output.diagnostics.get("private_representation")
            adapter = step.model_output.diagnostics.get("adapter_representation")
            if not torch.is_tensor(shared) or not torch.is_tensor(private):
                continue
            if not torch.is_tensor(adapter):
                adapter = private
            coarse, _ = hist_beam_labels(labels, num_classes=num_classes, group_size=group_size)
            shared_flat = shared.reshape(-1, shared.shape[-1])
            private_flat = private.reshape(-1, private.shape[-1])
            adapter_flat = adapter.reshape(-1, adapter.shape[-1])
            coarse_flat = coarse.reshape(-1)
            valid = coarse_flat.ge(0)
            if shared_sum is None:
                shared_sum = torch.zeros(num_groups, shared_flat.shape[-1], dtype=shared_flat.dtype, device=device)
                private_sum = torch.zeros(num_groups, private_flat.shape[-1], dtype=private_flat.dtype, device=device)
                adapter_sum = torch.zeros(num_groups, adapter_flat.shape[-1], dtype=adapter_flat.dtype, device=device)
                radio_sum = torch.zeros(num_radio_classes, shared_flat.shape[-1], dtype=shared_flat.dtype, device=device)
                path_sum = torch.zeros(num_path_classes, shared_flat.shape[-1], dtype=shared_flat.dtype, device=device)
            for group in range(num_groups):
                mask = valid & coarse_flat.eq(group)
                if not torch.any(mask):
                    continue
                shared_sum[group] += shared_flat[mask].sum(dim=0)
                private_sum[group] += private_flat[mask].sum(dim=0)
                adapter_sum[group] += adapter_flat[mask].sum(dim=0)
                counts[group] += int(mask.sum().item())
            radio_labels = prepare_radio_semantic_labels(
                step.batch,
                num_pred=labels.shape[1],
                device=device,
                non_blocking=transfer_non_blocking(cfg),
            )
            if radio_labels is not None and radio_sum is not None:
                radio_flat = radio_labels.reshape(-1)
                radio_valid = radio_flat.ge(0) & radio_flat.lt(num_radio_classes)
                for class_index in range(num_radio_classes):
                    mask = radio_valid & radio_flat.eq(class_index)
                    if not torch.any(mask):
                        continue
                    radio_sum[class_index] += shared_flat[mask].sum(dim=0)
                    radio_counts[class_index] += int(mask.sum().item())
            path_labels = prepare_path_semantic_labels(
                step.batch,
                num_pred=labels.shape[1],
                device=device,
                non_blocking=transfer_non_blocking(cfg),
            )
            if path_labels is not None and path_sum is not None:
                path_flat = path_labels.reshape(-1)
                path_valid = path_flat.ge(0) & path_flat.lt(num_path_classes)
                for class_index in range(num_path_classes):
                    mask = path_valid & path_flat.eq(class_index)
                    if not torch.any(mask):
                        continue
                    path_sum[class_index] += shared_flat[mask].sum(dim=0)
                    path_counts[class_index] += int(mask.sum().item())
                path_targets = prepare_path_descriptors(
                    step.batch,
                    num_pred=labels.shape[1],
                    device=device,
                    non_blocking=transfer_non_blocking(cfg),
                )
                if path_targets is not None:
                    descriptor, descriptor_mask = path_targets
                    descriptor_flat = descriptor.reshape(-1, descriptor.shape[-1])
                    descriptor_valid = descriptor_mask.reshape(-1) & path_valid
                    if path_descriptor_sum is None:
                        path_descriptor_dim = int(descriptor_flat.shape[-1])
                        path_descriptor_sum = torch.zeros(num_path_classes, path_descriptor_dim, dtype=descriptor_flat.dtype, device=device)
                    for class_index in range(num_path_classes):
                        mask = descriptor_valid & path_flat.eq(class_index)
                        if torch.any(mask):
                            path_descriptor_sum[class_index] += descriptor_flat[mask].sum(dim=0)
            if callable(progress_callback):
                progress_callback(
                    {
                        "phase": "source_prototype",
                        "processed_batches": int(processed_batches),
                        "total_batches": int(total_batches) if total_batches is not None else None,
                        "processed_samples": int(processed_samples),
                        "duration_seconds": float(time.perf_counter() - start_time),
                    }
                )
    if shared_sum is None:
        dim = int(student_cfg.get("d_model", model_cfg.get("d_model", student_cfg.get("feature_size", 64))))
        shared_sum = torch.zeros(num_groups, dim, device=device)
        private_sum = torch.zeros(num_groups, dim, device=device)
        adapter_sum = torch.zeros(num_groups, dim, device=device)
        radio_sum = torch.zeros(num_radio_classes, dim, device=device)
        path_sum = torch.zeros(num_path_classes, dim, device=device)
    assert private_sum is not None and adapter_sum is not None
    assert radio_sum is not None
    assert path_sum is not None
    denom = counts.clamp_min(1).to(dtype=shared_sum.dtype).unsqueeze(-1)
    radio_denom = radio_counts.clamp_min(1).to(dtype=radio_sum.dtype).unsqueeze(-1)
    path_denom = path_counts.clamp_min(1).to(dtype=path_sum.dtype).unsqueeze(-1)
    shared_prototypes = (shared_sum / denom).detach().cpu()
    private_prototypes = (private_sum / denom).detach().cpu()
    adapter_prototypes = (adapter_sum / denom).detach().cpu()
    radio_prototypes = (radio_sum / radio_denom).detach().cpu()
    path_prototypes = (path_sum / path_denom).detach().cpu()
    path_descriptor_prototypes = None
    if path_descriptor_sum is not None:
        descriptor_denom = path_counts.clamp_min(1).to(dtype=path_descriptor_sum.dtype).unsqueeze(-1)
        path_descriptor_prototypes = (path_descriptor_sum / descriptor_denom).detach().cpu()
    artifact_counts = path_counts if proto_type == "path" else radio_counts if proto_type == "radio_semantic" else counts
    artifact_shared = path_prototypes if proto_type == "path" else radio_prototypes if proto_type == "radio_semantic" else shared_prototypes
    artifact = {
        "version": "hist_beam_prototypes_v2",
        "shared_prototypes": artifact_shared,
        "private_prototypes": private_prototypes,
        "adapter_prototypes": adapter_prototypes,
        "counts": artifact_counts.detach().cpu(),
        "mu_coarse_c": shared_prototypes,
        "count_coarse": counts.detach().cpu(),
        "mu_radio_c": radio_prototypes,
        "count_radio": radio_counts.detach().cpu(),
        "mu_path_c": path_prototypes,
        "count_path": path_counts.detach().cpu(),
        "metadata": {
            "group_size": group_size,
            "num_groups": num_groups,
            "num_classes": num_classes,
            "num_radio_classes": num_radio_classes,
            "num_path_classes": num_path_classes,
            "proto_type": proto_type,
            "prototype_space": "shared_path_physical" if proto_type == "path" else "shared_radio_semantic" if proto_type == "radio_semantic" else "coarse_sector_private_adapter",
            "radio_label_mode": radio_cfg.get("mode", radio_cfg.get("label_mode", "peak_spread")),
            "path_label_mode": path_cfg.get("mode", path_cfg.get("label_mode", "kmeans_path_descriptor")),
            "radio_entropy_thresholds": radio_cfg.get("entropy_thresholds", radio_cfg.get("spread_thresholds")),
            "radio_class_counts": radio_counts.detach().cpu().tolist(),
            "coarse_class_counts": counts.detach().cpu().tolist(),
            "path_class_counts": path_counts.detach().cpu().tolist(),
            "empty_radio_classes": [int(index) for index in torch.where(radio_counts.detach().cpu().eq(0))[0].tolist()],
            "empty_path_classes": [int(index) for index in torch.where(path_counts.detach().cpu().eq(0))[0].tolist()],
            "path_descriptor_dim": path_descriptor_dim,
            "processed_batches": int(processed_batches),
            "processed_samples": int(processed_samples),
            "duration_seconds": float(time.perf_counter() - start_time),
            "direct_fields": ["beam_label", "coarse_sector"],
            "proxy_fields": [],
            **(metadata or {}),
        },
    }
    if path_descriptor_prototypes is not None:
        artifact["mu_path_descriptor"] = path_descriptor_prototypes
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
    if "mu_path_c" in artifact and "shared_prototypes" not in artifact:
        artifact["shared_prototypes"] = artifact["mu_path_c"]
    if "count_path" in artifact and "counts" not in artifact:
        artifact["counts"] = artifact["count_path"]
    if "mu_radio_c" in artifact and "shared_prototypes" not in artifact:
        artifact["shared_prototypes"] = artifact["mu_radio_c"]
    if "count_radio" in artifact and "counts" not in artifact:
        artifact["counts"] = artifact["count_radio"]
    for key in ("shared_prototypes", "counts", "metadata"):
        if key not in artifact:
            raise ValueError(f"Prototype artifact missing required key '{key}'.")
    shared = artifact["shared_prototypes"]
    private = artifact.get("private_prototypes", torch.zeros_like(shared) if torch.is_tensor(shared) else None)
    adapter = artifact.get("adapter_prototypes", private)
    counts = artifact["counts"]
    if not torch.is_tensor(shared) or not torch.is_tensor(private) or not torch.is_tensor(adapter) or not torch.is_tensor(counts):
        raise ValueError("Prototype artifact shared/private/counts must be tensors.")
    if shared.shape[0] != counts.shape[0]:
        raise ValueError("Prototype artifact shared/count shapes are inconsistent.")
    if private.shape[-1] != shared.shape[-1] or adapter.shape[-1] != shared.shape[-1]:
        raise ValueError("Prototype artifact representation dimensions are inconsistent.")
    artifact.setdefault("private_prototypes", private)
    artifact.setdefault("adapter_prototypes", adapter)


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
