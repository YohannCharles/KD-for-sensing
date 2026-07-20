#!/usr/bin/env python3
"""Evaluate a fixed CurrentControl checkpoint on one joint Drop+Corrupt shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from kd_sensing.data.mmw.twc_router_joint_stress import (
    CORRUPTION_SEVERITY,
    MASKS_PER_RATE,
    PROTOCOL_ID,
    STATE_CORRUPT,
    STATE_DROP,
    load_router_joint_stress_cache,
)
from kd_sensing.data.temporal_missing import (
    DEFAULT_TEMPORAL_MODALITIES,
    apply_modality_temporal_mask_to_batch,
)
from kd_sensing.engine.data_factory import build_dataloader, build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.evaluation_pass_runtime import prepare_evaluation_batch, sample_ids_from_batch
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_cuda_performance_settings, prepare_task_labels, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.evaluation.corruptions import CORRUPTION_PARAMETERS, CorruptionSpec, apply_inference_corruption
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.checkpoint import load_model_state
from kd_sensing.utils.seed import set_seed

import eval_mmw_all_weather_matrix as matrix
from eval_mmw_router_oracle_gap import metrics_for, oracle_gap_branches, sha256, write_csv, write_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "outputs/cache/mmw_router_joint_stress_v1/fixed_state_cache.json"
DEFAULT_CONFIG = ROOT / "outputs/mmw_router_expected_utility_screen_v3/generated_configs/CurrentControl_seed1.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs/mmw_router_expected_utility_screen_v3/CurrentControl/seed1/checkpoints/last.pth"
DEFAULT_OUTPUT = ROOT / "outputs/mmw_router_joint_stress_v1"
CORRUPTION_SEED = 20260718
EVALUATOR_ALGORITHM = "paired_cell_selective_s2_then_temporal_drop_reference_controls_v2"
UTILITY_NUMERIC_POLICY = "beam_power_float32_before_linear_normalization_v1"
ROUTER_RELIABILITY_SOURCE = ROOT / "src/kd_sensing/losses/router_reliability.py"
CORRUPTION_NAMES = {
    "image": "image_occlusion",
    "radar": "radar_noise",
    "gps": "gps_noise",
    "lidar": "lidar_sparsify",
}
MODALITY_KEYS = {
    "image": ("image",),
    "radar": ("radar_ra", "radar_da"),
    "gps": ("gps",),
    "lidar": ("lidar",),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", required=True, type=int, choices=(20, 40, 60, 80))
    parser.add_argument("--mask-start", type=int, default=0)
    parser.add_argument("--mask-end", type=int, default=MASKS_PER_RATE)
    parser.add_argument("--include-clean", action="store_true")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--request-sha256", default="")
    parser.add_argument("--orchestration-attempt", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-domains", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batch_size % 16:
        parser.error("--batch-size must be a positive multiple of 16")
    if not 0 <= args.mask_start < args.mask_end <= MASKS_PER_RATE:
        parser.error(f"mask range must satisfy 0 <= start < end <= {MASKS_PER_RATE}")
    request_sha256 = str(args.request_sha256).strip().lower()
    if request_sha256 and (len(request_sha256) != 64 or any(char not in "0123456789abcdef" for char in request_sha256)):
        parser.error("--request-sha256 must be a 64-character lowercase hexadecimal digest")
    args.request_sha256 = request_sha256
    args.orchestration_attempt = str(args.orchestration_attempt).strip()
    evaluate_shard(args)
    return 0


def evaluate_shard(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache).resolve()
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    cache = load_router_joint_stress_cache(cache_path)
    conditions = select_conditions(
        cache,
        rate=int(args.rate),
        mask_start=int(args.mask_start),
        mask_end=int(args.mask_end),
        include_clean=bool(args.include_clean),
    )
    shard_id = (
        f"rate{int(args.rate):02d}_masks{int(args.mask_start):02d}_{int(args.mask_end):02d}"
        + ("_clean" if args.include_clean else "")
    )
    shard_dir = output_root / "shards" / shard_id
    shard_complete = shard_dir / "complete.json"
    partial = args.max_domains is not None or args.max_batches is not None
    if partial and args.overwrite:
        completed = [
            str(output_root / str(condition["pattern"]) / "complete.json")
            for condition in conditions
            if (output_root / str(condition["pattern"]) / "complete.json").is_file()
        ]
        if shard_complete.is_file() or completed:
            raise ValueError("Partial overwrite must not mutate an output root containing completed evidence.")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    candidate = validate_router_screen_config(cfg, checkpoint)
    checkpoint_metadata = load_checkpoint_metadata(checkpoint)
    normalization = load_normalization_artifacts(checkpoint_metadata)
    scaler = normalization.get("gps_scaler")
    if scaler is None or scaler.mean_ is None or scaler.scale_ is None:
        raise ValueError("Joint stress requires the checkpoint's train-fit GPS scaler.")
    config_sha256 = sha256(config_path)
    checkpoint_sha256 = sha256(checkpoint)
    cache_file_sha256 = sha256(cache_path)
    identity = {
        "protocol": PROTOCOL_ID,
        "cache_checksum": str(cache["checksum"]),
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if args.orchestration_attempt:
        identity["orchestration_attempt"] = args.orchestration_attempt
    if args.request_sha256:
        identity["request_sha256"] = args.request_sha256
    pending = [
        condition
        for condition in conditions
        if args.overwrite or not condition_is_complete(output_root / str(condition["pattern"]), identity)
    ]
    if not pending:
        payload = shard_payload(
            shard_id=shard_id,
            args=args,
            cache=cache,
            cache_path=cache_path,
            cache_file_sha256=cache_file_sha256,
            config_path=config_path,
            config_sha256=config_sha256,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            candidate=candidate,
            conditions=conditions,
            partial=partial,
        )
        shard_dir.mkdir(parents=True, exist_ok=True)
        write_json(shard_dir / ("provenance.json" if partial else "complete.json"), payload)
        return

    cfg.setdefault("temporal_missing", {}).update({"enabled": False, "mode": "none"})
    cfg.setdefault("training", {})["final_test"] = {"enabled": False, "reason": "joint_stress_inner_only"}
    loader_cfg = cfg["data"]["dataloader"]
    loader_cfg["validation_batch_size"] = int(args.batch_size)
    loader_cfg["test_batch_size"] = int(args.batch_size)
    rows_by_pattern: dict[str, list[dict[str, Any]]] = {str(item["pattern"]): [] for item in pending}
    traces_by_pattern: dict[str, list[dict[str, str]]] = {str(item["pattern"]): [] for item in pending}
    dataloaders = None
    selected: list[tuple[Any, Mapping[str, Any]]] = []
    try:
        dataloaders = build_dataloaders(cfg, normalization_overrides=normalization)
        validation = dataloaders["validation"].dataset
        components = list(getattr(validation, "datasets", ()))
        inventory = list(getattr(validation, "domain_inventory", ()))
        if len(components) != 15 or len(inventory) != 15:
            raise ValueError(f"Expected 15 inner-validation domains, got {len(components)} and {len(inventory)}.")
        selected = list(zip(components, inventory))
        if args.max_domains is not None:
            selected = selected[: int(args.max_domains)]
        set_seed(1)
        device = build_device(cfg)
        configure_cuda_performance_settings(cfg, device)
        model = build_model(cfg["model"]["primary"]).to(device)
        if tuple(model.modalities) != tuple(cache["modalities"]):
            raise ValueError("Joint-stress cache modality order differs from the fixed model.")
        load_model_state(checkpoint, model, role="Router joint-stress fixed CurrentControl checkpoint", map_location=device, strict=True)
        model.eval()
        for domain_index, (component, domain) in enumerate(selected, start=1):
            loader = build_dataloader(component, loader_cfg, split="validation", experiment_seed=1)
            try:
                traces = evaluate_domain(
                    model,
                    loader,
                    cfg,
                    device,
                    pending,
                    gps_scaler_mean=scaler.mean_,
                    gps_scaler_scale=scaler.scale_,
                    max_batches=args.max_batches,
                )
            finally:
                shutdown_dataloader_workers(loader)
            domain_id = str(domain["id"])
            for condition in pending:
                pattern = str(condition["pattern"])
                trace = traces[pattern]
                condition_dir = output_root / pattern
                trace_path = condition_dir / "traces" / f"{domain_id.replace('/', '__')}.npz"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    trace_path,
                    domain_id=np.asarray(domain_id),
                    pattern=np.asarray(pattern),
                    state_digest=np.asarray(condition["state_digest"]),
                    **trace,
                )
                traces_by_pattern[pattern].append({"path": str(trace_path), "sha256": sha256(trace_path)})
                rows_by_pattern[pattern].extend(
                    domain_metric_rows(
                        model,
                        cfg,
                        condition,
                        domain,
                        trace,
                        cache_checksum=str(cache["checksum"]),
                    )
                )
                write_csv(condition_dir / "domain_metrics.csv", rows_by_pattern[pattern])
            print(
                f"{shard_id}: domain {domain_index}/{len(selected)} {domain_id}; "
                f"conditions={len(pending)}",
                flush=True,
            )
    finally:
        if dataloaders is not None:
            shutdown_all_dataloaders(dataloaders)

    if not selected:
        raise ValueError("Joint-stress shard selected no domains.")
    for condition in pending:
        pattern = str(condition["pattern"])
        condition_dir = output_root / pattern
        provenance = {
            **identity,
            "status": "partial_debug_complete" if partial else "complete",
            "pattern": pattern,
            "condition": condition,
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_candidate": candidate,
            "cache": str(cache_path),
            "cache_file_sha256": cache_file_sha256,
            "seed": 1,
            "split": "frozen_inner_validation_only",
            "claim_eligible": False,
            "request_sha256": str(args.request_sha256),
            "batch_size": int(args.batch_size),
            "corruption_severity": CORRUPTION_SEVERITY,
            "corruption_seed": CORRUPTION_SEED,
            "evaluator_algorithm": EVALUATOR_ALGORITHM,
            "corruptions": corruption_provenance(),
            "gps_scaler_mean": np.asarray(scaler.mean_).tolist(),
            "gps_scaler_scale": np.asarray(scaler.scale_).tolist(),
            "domain_count": len(selected),
            "partial": partial,
            "trace_files": traces_by_pattern[pattern],
        }
        condition_dir.mkdir(parents=True, exist_ok=True)
        write_json(condition_dir / "provenance.json", provenance)
        if not partial:
            write_json(condition_dir / "complete.json", provenance)

    payload = shard_payload(
        shard_id=shard_id,
        args=args,
        cache=cache,
        cache_path=cache_path,
        cache_file_sha256=cache_file_sha256,
        config_path=config_path,
        config_sha256=config_sha256,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        candidate=candidate,
        conditions=conditions,
        partial=partial,
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_json(shard_dir / "provenance.json", payload)
    if not partial:
        write_json(shard_complete, payload)


def evaluate_domain(
    model,
    loader,
    cfg: dict[str, Any],
    device: torch.device,
    conditions: Sequence[Mapping[str, Any]],
    *,
    gps_scaler_mean: Any,
    gps_scaler_scale: Any,
    max_batches: int | None,
) -> dict[str, dict[str, np.ndarray]]:
    chunks: dict[str, dict[str, list[np.ndarray]]] = {str(item["pattern"]): {} for item in conditions}
    model_cfg = cfg["model"]["primary"]
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            clean = prepare_evaluation_batch(raw_batch)
            corrupted = build_corrupted_candidate(
                clean,
                batch_index=batch_index,
                gps_scaler_mean=gps_scaler_mean,
                gps_scaler_scale=gps_scaler_scale,
            )
            for condition in conditions:
                batch = compose_joint_batch(clean, corrupted, condition["state_matrix"])
                availability = batch["available_modalities"].to(device=device, dtype=torch.bool)
                step = run_model_step(
                    model,
                    cfg["experiment"].get("task", "fusion"),
                    batch,
                    seq_length=int(model_cfg.get("seq_length", 5)),
                    num_pred=int(model_cfg.get("num_pred", 1)),
                    device=device,
                    extra_model_kwargs={"missing_mask": availability},
                )
                labels = prepare_task_labels(step.batch, num_pred=int(model_cfg.get("num_pred", 1)), device=device)
                target = labels[:, -1].reshape(-1) if labels.ndim > 1 else labels.reshape(-1)
                diagnostics = step.model_output.diagnostics
                available = diagnostics["available_modalities"].to(dtype=torch.bool)
                if not torch.equal(available, availability):
                    raise ValueError("Model availability differs from the fixed joint Drop mask.")
                unimodal = diagnostics["unimodal_logits"]
                router_raw = diagnostics["supervised_router_gate_weights"]
                router = normalize_available_weights(router_raw, available)
                uniform_weights = normalize_available_weights(available.to(dtype=unimodal.dtype), available)
                learned = step.logits[:, -1, :] if step.logits.ndim == 3 else step.logits
                reconstructed = (router.unsqueeze(-1) * unimodal).sum(dim=1)
                if not torch.allclose(learned, reconstructed, atol=1e-5, rtol=1e-5):
                    raise ValueError("Learned logits do not reconstruct from availability-normalized Router weights.")
                powers = matrix._load_future_beam_power(step.batch).to(device=unimodal.device)  # noqa: SLF001
                candidate_branches = oracle_gap_branches(unimodal, router, powers, available=available)
                static_prior = diagnostics.get("router_static_prior_weights")
                reference_router = diagnostics.get("reference_router_gate_weights")
                reference_unimodal = diagnostics.get("reference_unimodal_logits")
                router_residual = diagnostics.get("router_residual_logits")
                effective_cell_weights = diagnostics.get("router_effective_cell_weights")
                if static_prior is not None:
                    static_prior = normalize_available_weights(static_prior, available)
                if reference_router is not None:
                    reference_router = normalize_available_weights(reference_router, available)
                if (reference_router is None) != (reference_unimodal is None):
                    raise ValueError("Dynamic Router reference weights/logits must be emitted together.")
                if static_prior is not None and not torch.is_tensor(reference_unimodal):
                    raise ValueError("Dynamic Router controls require masked-mean reference experts.")
                dynamic_controls = (
                    dynamic_control_branches(
                        candidate_unimodal=unimodal,
                        reference_unimodal=reference_unimodal,
                        static_prior=static_prior,
                        available=available,
                        beam_powers=powers,
                    )
                    if static_prior is not None
                    else None
                )
                ids = np.asarray(sample_ids_from_batch(step.batch), dtype=str)
                if len(ids) != int(target.numel()):
                    raise ValueError("Sample identity count differs from the evaluated batch.")
                state = torch.as_tensor(condition["state_matrix"], dtype=torch.int8)
                state = state.unsqueeze(0).expand(target.numel(), -1, -1)
                values = {
                    "sample_id": ids,
                    "target": target.detach().cpu().numpy().astype(np.int64),
                    "beam_powers": powers.detach().cpu().numpy().astype(np.float32),
                    "state_matrix": state.numpy(),
                    "modality_temporal_mask": diagnostics["modality_temporal_mask"].detach().cpu().numpy().astype(bool),
                    "available_modalities": available.detach().cpu().numpy().astype(bool),
                    "unimodal_logits": unimodal.detach().cpu().numpy().astype(np.float32),
                    "router_weights": router.detach().cpu().numpy().astype(np.float32),
                    "uniform_weights": uniform_weights.detach().cpu().numpy().astype(np.float32),
                    "learned_logits": learned.detach().cpu().numpy().astype(np.float32),
                    **{key: value.detach().cpu().numpy() for key, value in candidate_branches.items()},
                }
                if static_prior is not None:
                    if not torch.is_tensor(router_residual) or not torch.is_tensor(effective_cell_weights):
                        raise ValueError("Dynamic Router diagnostics must include residual and effective cell weights.")
                    values.update(
                        {
                            "static_prior_weights": static_prior.detach().cpu().numpy().astype(np.float32),
                            **{
                                key: value.detach().cpu().numpy().astype(np.float32)
                                for key, value in dynamic_controls.items()
                            },
                            "router_residual_logits": router_residual.detach().cpu().numpy().astype(np.float32),
                            "router_effective_cell_weights": effective_cell_weights.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32),
                        }
                    )
                if reference_router is not None and torch.is_tensor(reference_unimodal):
                    values.update(
                        {
                            "reference_router_weights": reference_router.detach().cpu().numpy().astype(np.float32),
                            "reference_unimodal_logits": reference_unimodal.detach().cpu().numpy().astype(np.float32),
                            "reference_router_logits": (reference_router.unsqueeze(-1) * reference_unimodal)
                            .sum(dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32),
                        }
                    )
                sink = chunks[str(condition["pattern"])]
                for key, value in values.items():
                    sink.setdefault(key, []).append(value)
    result = {}
    for pattern, values in chunks.items():
        if not values:
            raise ValueError(f"Joint-stress condition {pattern} produced no samples.")
        result[pattern] = {key: np.concatenate(items, axis=0) for key, items in values.items()}
    return result


def build_corrupted_candidate(
    clean_batch: Mapping[str, Any],
    *,
    batch_index: int,
    gps_scaler_mean: Any,
    gps_scaler_scale: Any,
) -> dict[str, Any]:
    """Build one four-sensor S2 candidate reused by every condition in the raw batch."""

    corrupted = matrix._clone_batch(clean_batch)  # noqa: SLF001 - established evidence batch clone.
    for modality_index, modality in enumerate(DEFAULT_TEMPORAL_MODALITIES):
        corrupted = apply_inference_corruption(
            corrupted,
            CorruptionSpec(CORRUPTION_NAMES[modality], CORRUPTION_SEVERITY),
            seed=CORRUPTION_SEED + modality_index * 1009,
            batch_index=int(batch_index),
            gps_scaler_mean=gps_scaler_mean,
            gps_scaler_scale=gps_scaler_scale,
        )
    return corrupted


def compose_joint_batch(
    clean_batch: Mapping[str, Any],
    corrupted_batch: Mapping[str, Any],
    state_matrix: Any,
    *,
    modalities: Sequence[str] = DEFAULT_TEMPORAL_MODALITIES,
) -> dict[str, Any]:
    """Select corrupt cells first, then zero Drop cells through the shared mask contract."""

    names = tuple(str(item) for item in modalities)
    if names != tuple(DEFAULT_TEMPORAL_MODALITIES):
        raise ValueError(f"Joint stress requires modality order {list(DEFAULT_TEMPORAL_MODALITIES)}.")
    states = torch.as_tensor(state_matrix, dtype=torch.int64)
    if tuple(states.shape) != (5, len(names)):
        raise ValueError(f"Joint-stress state matrix must have shape {(5, len(names))}.")
    if not bool(((states >= 0) & (states <= STATE_CORRUPT)).all()):
        raise ValueError("Joint-stress state matrix contains an invalid state code.")
    batch = matrix._clone_batch(clean_batch)  # noqa: SLF001 - established evidence batch clone.
    for modality_index, modality in enumerate(names):
        corrupt_steps = states[:, modality_index].eq(STATE_CORRUPT)
        for key in MODALITY_KEYS[modality]:
            clean = clean_batch.get(key)
            corrupt = corrupted_batch.get(key)
            if not torch.is_tensor(clean) or not torch.is_tensor(corrupt) or clean.shape != corrupt.shape:
                raise ValueError(f"Joint stress requires matching clean/corrupted tensor field {key!r}.")
            if clean.ndim < 2 or tuple(clean.shape[1:2]) != (states.shape[0],):
                raise ValueError(f"Joint stress requires {key!r} time dimension {states.shape[0]}.")
            selector = corrupt_steps.to(device=clean.device).reshape(1, states.shape[0], *([1] * (clean.ndim - 2)))
            batch[key] = torch.where(selector, corrupt, clean)
    temporal_mask = states.ne(STATE_DROP)
    return apply_modality_temporal_mask_to_batch(batch, temporal_mask, modalities=names)


def normalize_available_weights(weights: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    masked = weights.masked_fill(~available, 0.0)
    denominator = masked.sum(dim=1, keepdim=True)
    if bool((denominator <= 0).any()):
        raise ValueError("Fusion weights must retain positive mass on available modalities.")
    return masked / denominator


def dynamic_control_logits(
    *,
    candidate_unimodal: torch.Tensor,
    reference_unimodal: torch.Tensor,
    static_prior: torch.Tensor,
    available: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if candidate_unimodal.shape != reference_unimodal.shape or candidate_unimodal.ndim != 3:
        raise ValueError("Dynamic control experts must share [B,M,C] shape.")
    uniform = normalize_available_weights(available.to(dtype=candidate_unimodal.dtype), available)
    prior = normalize_available_weights(static_prior, available)
    payload = {
        "uniform_logits": (uniform.unsqueeze(-1) * reference_unimodal).sum(dim=1),
        "train_fit_static_prior_logits": (prior.unsqueeze(-1) * reference_unimodal).sum(dim=1),
        "post_health_uniform_logits": (uniform.unsqueeze(-1) * candidate_unimodal).sum(dim=1),
        "post_health_static_prior_logits": (prior.unsqueeze(-1) * candidate_unimodal).sum(dim=1),
    }
    return payload


def dynamic_control_branches(
    *,
    candidate_unimodal: torch.Tensor,
    reference_unimodal: torch.Tensor,
    static_prior: torch.Tensor,
    available: torch.Tensor,
    beam_powers: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build fair reference controls and separately expose candidate oracle diagnostics."""

    controls = dynamic_control_logits(
        candidate_unimodal=candidate_unimodal,
        reference_unimodal=reference_unimodal,
        static_prior=static_prior,
        available=available,
    )
    reference_uniform = normalize_available_weights(available.to(reference_unimodal.dtype), available)
    reference_branches = oracle_gap_branches(
        reference_unimodal,
        reference_uniform,
        beam_powers,
        available=available,
    )
    candidate_uniform = oracle_gap_branches(
        candidate_unimodal,
        normalize_available_weights(available.to(candidate_unimodal.dtype), available),
        beam_powers,
        available=available,
    )
    return {
        **controls,
        "oracle_logits": reference_branches["oracle_logits"],
        "candidate_oracle_logits": candidate_uniform["oracle_logits"],
    }


def select_conditions(
    cache: Mapping[str, Any],
    *,
    rate: int,
    mask_start: int,
    mask_end: int,
    include_clean: bool,
) -> list[Mapping[str, Any]]:
    requested_rate = float(rate) / 100.0
    selected = [
        item
        for item in cache["conditions"]
        if float(item["requested_stress_rate"]) == requested_rate
        and mask_start <= int(item["mask_set_index"]) < mask_end
    ]
    if len(selected) != mask_end - mask_start:
        raise ValueError("Joint-stress cache does not contain the requested condition shard.")
    if include_clean:
        selected.insert(0, cache["conditions"][0])
    return selected


def validate_current_control_config(cfg: Mapping[str, Any], checkpoint: Path) -> str:
    screen = cfg.get("mmw_router_utility_screen", {})
    candidate = str(screen.get("candidate_id", "")).strip() if isinstance(screen, Mapping) else ""
    if (
        candidate != "CurrentControl"
        or screen.get("protocol") != "mmw_router_expected_utility_screen_v3"
        or screen.get("selection_split") != "frozen_inner_validation_only"
        or int(screen.get("seed", -1)) != 1
        or bool(screen.get("claim_eligible", True))
    ):
        raise ValueError("Joint stress requires the claim-ineligible CurrentControl seed1 inner-only config.")
    if int(cfg.get("training", {}).get("epochs", -1)) != 40 or checkpoint.name != "last.pth":
        raise ValueError("Joint stress requires the fixed 40-epoch last.pth checkpoint.")
    return candidate


def validate_router_screen_config(cfg: Mapping[str, Any], checkpoint: Path) -> str:
    dynamic = cfg.get("mmw_dynamic_router_screen", {})
    decision_alignment = cfg.get("mmw_dynamic_router_decision_screen", {})
    simplification = cfg.get("mmw_h2r_simplification_screen", {})
    if isinstance(simplification, Mapping) and simplification:
        candidate = str(simplification.get("candidate", "")).strip()
        supervision = str(simplification.get("supervision", "")).strip()
        profile = str(simplification.get("evidence_profile", "")).strip()
        epochs = int(simplification.get("calibration_epochs", -1))
        primary = cfg.get("model", {}).get("primary", {})
        primary_profile = primary.get("router_variant_config", {}).get("evidence_profile", "full")
        if (
            not candidate
            or simplification.get("protocol") != "mmw_h2r_simplification_screen_v1"
            or simplification.get("selection_split") != "frozen_inner_validation_only"
            or int(simplification.get("seed", -1)) != 1
            or bool(simplification.get("claim_eligible", True))
            or str(primary.get("router_variant", "")) != "h2r"
            or str(primary_profile) != profile
            or profile not in {"full", "generic_confidence", "prototype_topology"}
            or supervision not in {"label_topology", "beam_power"}
            or str(simplification.get("fused_decision_objective", "")) != "joint_hard_ce"
            or epochs not in {10, 40}
            or int(cfg.get("training", {}).get("epochs", -1)) != epochs
            or checkpoint.name != "last.pth"
        ):
            raise ValueError("Joint stress requires a claim-ineligible H2R simplification config.")
        if simplification.get("router_reliability_source_sha256") != sha256(ROUTER_RELIABILITY_SOURCE):
            raise ValueError("H2R simplification evaluation rejects a changed Router loss source.")
        return candidate
    if isinstance(decision_alignment, Mapping) and decision_alignment:
        candidate = str(decision_alignment.get("candidate", "")).strip()
        variant = str(decision_alignment.get("router_variant", "")).strip()
        supervision = str(decision_alignment.get("supervision", "")).strip()
        objective = str(decision_alignment.get("fused_decision_objective", "")).strip()
        primary = cfg.get("model", {}).get("primary", {})
        if (
            not candidate
            or decision_alignment.get("protocol") != "mmw_dynamic_router_decision_alignment_v1"
            or decision_alignment.get("selection_split") != "frozen_inner_validation_only"
            or int(decision_alignment.get("seed", -1)) != 1
            or bool(decision_alignment.get("claim_eligible", True))
            or str(primary.get("router_variant", "")) != variant
            or variant not in {"patr", "h2r"}
            or supervision != "beam_power"
            or objective
            not in {"expected_utility", "joint_hard_ce", "power_soft_ce", "power_top1_margin"}
        ):
            raise ValueError("Joint stress requires a claim-ineligible decision-alignment Router config.")
        if int(cfg.get("training", {}).get("epochs", -1)) != 40 or checkpoint.name != "last.pth":
            raise ValueError("Decision-alignment Router joint stress requires the fixed 40-epoch last.pth checkpoint.")
        if (
            decision_alignment.get("utility_numeric_policy") != UTILITY_NUMERIC_POLICY
            or decision_alignment.get("router_reliability_source_sha256") != sha256(ROUTER_RELIABILITY_SOURCE)
        ):
            raise ValueError("Decision-alignment Router evaluation rejects unproven numeric/source policy configs.")
        return candidate
    if not isinstance(dynamic, Mapping) or not dynamic:
        return validate_current_control_config(cfg, checkpoint)
    candidate = str(dynamic.get("candidate", "")).strip()
    variant = str(dynamic.get("router_variant", "")).strip()
    supervision = str(dynamic.get("supervision", "")).strip()
    primary = cfg.get("model", {}).get("primary", {})
    if (
        not candidate
        or dynamic.get("protocol") != "mmw_dynamic_router_screen_v1"
        or dynamic.get("selection_split") != "frozen_inner_validation_only"
        or int(dynamic.get("seed", -1)) != 1
        or bool(dynamic.get("claim_eligible", True))
        or str(primary.get("router_variant", "")) != variant
        or variant not in {"patr", "h2r", "core", "unified_hpr"}
        or supervision not in {"label_topology", "beam_power"}
    ):
        raise ValueError("Joint stress requires a claim-ineligible dynamic Router seed1 inner-only config.")
    if int(cfg.get("training", {}).get("epochs", -1)) != 10 or checkpoint.name != "last.pth":
        raise ValueError("Dynamic Router joint stress requires the fixed 10-epoch last.pth checkpoint.")
    if supervision == "beam_power" and (
        dynamic.get("utility_numeric_policy") != UTILITY_NUMERIC_POLICY
        or dynamic.get("router_reliability_source_sha256") != sha256(ROUTER_RELIABILITY_SOURCE)
    ):
        raise ValueError("Dynamic Router Power evaluation rejects pre-fix or unproven numeric policy configs.")
    return candidate


def domain_metric_rows(
    model,
    cfg: dict[str, Any],
    condition: Mapping[str, Any],
    domain: Mapping[str, Any],
    trace: Mapping[str, np.ndarray],
    *,
    cache_checksum: str,
) -> list[dict[str, Any]]:
    common = {
        "protocol": PROTOCOL_ID,
        "cache_checksum": cache_checksum,
        "condition": str(condition["pattern"]),
        "condition_index": int(condition["condition_index"]),
        "requested_stress_rate": float(condition["requested_stress_rate"]),
        "drop_rate": float(condition["drop_rate"]),
        "corrupt_rate": float(condition["corrupt_rate"]),
        "mask_set_index": int(condition["mask_set_index"]),
        "state_digest": str(condition["state_digest"]),
        "domain_id": str(domain["id"]),
        "weather": str(domain["condition"]),
        "scene": str(domain["scene"]),
        "sample_count": int(trace["target"].shape[0]),
    }
    regret_values = {
        "router_soft_oracle_regret": float(trace["router_soft_oracle_regret"].mean()),
        "router_selection_oracle_regret": float(trace["router_selection_oracle_regret"].mean()),
    }
    regret_scope = (
        "learned_dynamic_branch_only"
        if "router_residual_logits" in trace
        else "learned_router_branch_only"
    )
    common.update(
        {
            f"router_weight_{name}": float(trace["router_weights"][:, index].mean())
            for index, name in enumerate(model.modalities)
        }
    )
    if "router_residual_logits" in trace and "router_effective_cell_weights" in trace:
        common.update(dynamic_response_diagnostics(trace))
    common.update(
        {
            f"uniform_weight_{name}": float(trace["uniform_weights"][:, index].mean())
            for index, name in enumerate(model.modalities)
        }
    )
    def branch_row(fusion: str, logits_key: str) -> dict[str, Any]:
        row = {
            **common,
            "fusion": fusion,
            **metrics_for(trace[logits_key], trace["target"], trace["beam_powers"], cfg),
            # Regret is a property of the learned Router branch.  Controls
            # retain explicit nulls so CSV schemas stay stable without
            # implying that Uniform/Oracle have Router regret.
            **{
                field: (value if fusion == "learned" else None)
                for field, value in regret_values.items()
            },
            "router_regret_scope": regret_scope if fusion == "learned" else "not_applicable_control_branch",
        }
        return row

    branches = [
        branch_row(fusion, logits_key)
        for fusion, logits_key in (
            ("learned", "learned_logits"),
            ("uniform", "uniform_logits"),
            ("oracle", "oracle_logits"),
        )
    ]
    for fusion, logits_key in (
        ("train_fit_static_prior", "train_fit_static_prior_logits"),
        ("frozen_current_router", "reference_router_logits"),
        ("post_health_uniform", "post_health_uniform_logits"),
        ("post_health_static_prior", "post_health_static_prior_logits"),
    ):
        if logits_key in trace:
            branches.append(branch_row(fusion, logits_key))
    return branches


def dynamic_response_diagnostics(trace: Mapping[str, np.ndarray]) -> dict[str, float]:
    residual = np.asarray(trace["router_residual_logits"], dtype=np.float64)
    effective = np.asarray(trace["router_effective_cell_weights"], dtype=np.float64)
    static_prior = np.asarray(trace["static_prior_weights"], dtype=np.float64)
    states = np.asarray(trace["state_matrix"], dtype=np.int8)
    router_weights = np.asarray(trace["router_weights"], dtype=np.float64)
    if (
        residual.shape != router_weights.shape
        or static_prior.shape != router_weights.shape
        or effective.shape != states.shape
        or effective.shape[0] != router_weights.shape[0]
        or effective.shape[2] != router_weights.shape[1]
    ):
        raise ValueError("Dynamic Router diagnostic trace shape mismatch.")
    available_cells = states != STATE_DROP
    corrupt_cells = states == STATE_CORRUPT
    available_count = available_cells.sum(axis=(1, 2))
    corrupt_share = np.divide(
        corrupt_cells.sum(axis=(1, 2)),
        available_count,
        out=np.zeros_like(available_count, dtype=np.float64),
        where=available_count > 0,
    )
    corrupt_mass = (effective * corrupt_cells).sum(axis=(1, 2))
    uniform_response_ratio = np.divide(
        corrupt_mass,
        corrupt_share,
        out=np.ones_like(corrupt_mass),
        where=corrupt_share > 0,
    )
    valid_per_modality = available_cells.sum(axis=1)
    static_cells = np.divide(
        static_prior[:, None, :] * available_cells,
        valid_per_modality[:, None, :],
        out=np.zeros_like(effective),
        where=valid_per_modality[:, None, :] > 0,
    )
    static_corrupt_mass = (static_cells * corrupt_cells).sum(axis=(1, 2))
    vs_static_ratio = np.divide(
        corrupt_mass,
        static_corrupt_mass,
        out=np.ones_like(corrupt_mass),
        where=static_corrupt_mass > 0,
    )
    has_corrupt = static_corrupt_mass > 0
    downweight_rate = (
        float((corrupt_mass[has_corrupt] < static_corrupt_mass[has_corrupt]).mean())
        if bool(has_corrupt.any())
        else 0.0
    )
    return {
        "router_residual_abs_mean": float(np.abs(residual).mean()),
        "corrupted_cell_weight_mass": float(corrupt_mass.mean()),
        "corrupted_cell_available_share": float(corrupt_share.mean()),
        "corrupted_cell_weight_response_ratio": float(uniform_response_ratio.mean()),
        "corrupted_cell_static_prior_mass": float(static_corrupt_mass.mean()),
        "corrupted_cell_weight_vs_static_ratio": float(vs_static_ratio.mean()),
        "corrupted_cell_downweight_vs_static_rate": downweight_rate,
    }


def corruption_provenance() -> list[dict[str, Any]]:
    result = []
    for modality_index, modality in enumerate(DEFAULT_TEMPORAL_MODALITIES):
        name = CORRUPTION_NAMES[modality]
        parameters = CORRUPTION_PARAMETERS[name]
        result.append(
            {
                "modality": modality,
                "name": name,
                "severity": CORRUPTION_SEVERITY,
                "unit": parameters["unit"],
                "value": parameters["values"][CORRUPTION_SEVERITY - 1],
                "seed": CORRUPTION_SEED + modality_index * 1009,
            }
        )
    return result


def condition_is_complete(condition_dir: Path, identity: Mapping[str, str]) -> bool:
    path = condition_dir / "complete.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or bool(payload.get("partial", True)):
        raise ValueError(f"Existing condition completion marker is not full evidence: {condition_dir}")
    if any(str(payload.get(key, "")) != str(value) for key, value in identity.items()):
        raise ValueError(f"Existing completed condition has incompatible identity: {condition_dir}")
    return True


def shard_payload(
    *,
    shard_id: str,
    args: argparse.Namespace,
    cache: Mapping[str, Any],
    cache_path: Path,
    cache_file_sha256: str,
    config_path: Path,
    config_sha256: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    candidate: str,
    conditions: Sequence[Mapping[str, Any]],
    partial: bool,
) -> dict[str, Any]:
    payload = {
        "status": "partial_debug_complete" if partial else "complete",
        "protocol": PROTOCOL_ID,
        "shard": shard_id,
        "shard_id": shard_id,
        "rate": int(args.rate),
        "mask_start": int(args.mask_start),
        "mask_end": int(args.mask_end),
        "include_clean": bool(args.include_clean),
        "patterns": [str(item["pattern"]) for item in conditions],
        "cache": str(cache_path),
        "cache_sha256": cache_file_sha256,
        "cache_file_sha256": cache_file_sha256,
        "cache_checksum": str(cache["checksum"]),
        "config": str(config_path),
        "config_sha256": config_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_candidate": candidate,
        "seed": 1,
        "split": "frozen_inner_validation_only",
        "claim_eligible": False,
        "request_sha256": str(args.request_sha256),
        "batch_size": int(args.batch_size),
        "max_domains": args.max_domains,
        "max_batches": args.max_batches,
        "partial": partial,
        "corruption_severity": CORRUPTION_SEVERITY,
        "corruption_seed": CORRUPTION_SEED,
        "evaluator_algorithm": EVALUATOR_ALGORITHM,
        "corruptions": corruption_provenance(),
    }
    if str(args.orchestration_attempt):
        payload["orchestration_attempt"] = str(args.orchestration_attempt)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
