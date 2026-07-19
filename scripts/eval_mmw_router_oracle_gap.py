#!/usr/bin/env python3
"""Evaluate one inner-only MMW Router oracle-gap condition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from kd_sensing.engine.data_factory import build_dataloader, build_dataloaders
from kd_sensing.engine.data_factory import shutdown_dataloader_workers
from kd_sensing.engine.evaluation_pass_runtime import prepare_evaluation_batch, sample_ids_from_batch
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_cuda_performance_settings, prepare_task_labels, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import _beam_classification_metrics
from kd_sensing.evaluation.corruptions import CorruptionSpec, apply_inference_corruption
from kd_sensing.evaluation.metrics import beam_power_communication_summary
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.checkpoint import load_model_state
from kd_sensing.utils.seed import set_seed

import eval_mmw_all_weather_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "mmw_router_oracle_gap_v1"
DEFAULT_CONFIG = ROOT / "outputs/mmw_tie_aware_router_screen_v1/generated_configs/SoftConfidenceTie_seed1.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs/mmw_tie_aware_router_screen_v1/SoftConfidenceTie/seed1/checkpoints/last.pth"
DEFAULT_OUTPUT = ROOT / "outputs/mmw_router_oracle_gap_v1"
CORRUPTION_SEED = 20260718
CONDITIONS = ("clean",) + tuple(
    f"{name}_s{severity}"
    for name in ("image_occlusion", "radar_noise", "lidar_sparsify", "gps_noise")
    for severity in (1, 2, 3)
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-domains", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    evaluate_condition(args)
    return 0


def evaluate_condition(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output_root).resolve() / args.condition
    complete_path = output / "complete.json"
    if complete_path.is_file() and not args.overwrite:
        print(json.dumps({"condition": args.condition, "status": "already_complete", "output": str(output)}))
        return
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    screen = cfg.get("mmw_router_utility_screen") or cfg.get("mmw_tie_aware_router_screen", {})
    candidate = str(screen.get("candidate_id", "")).strip()
    if not candidate or screen.get("selection_split") != "frozen_inner_validation_only":
        raise ValueError("Oracle gap evaluation requires a frozen inner-only Router screen config.")
    if bool(screen.get("claim_eligible", True)):
        raise ValueError("Oracle gap evaluation must remain claim-ineligible inner evidence.")
    metadata = load_checkpoint_metadata(checkpoint)
    normalization = load_normalization_artifacts(metadata)
    scaler = normalization.get("gps_scaler")
    if scaler is None or scaler.mean_ is None or scaler.scale_ is None:
        raise ValueError("Oracle gap evaluation requires the checkpoint's train-fit GPS scaler.")
    cfg.setdefault("temporal_missing", {}).update({"enabled": False, "mode": "none"})
    cfg.setdefault("training", {})["final_test"] = {"enabled": False, "reason": "oracle_gap_inner_only"}
    loader_cfg = cfg["data"]["dataloader"]
    loader_cfg["validation_batch_size"] = int(args.batch_size)
    loader_cfg["test_batch_size"] = int(args.batch_size)
    dataloaders = None
    condition_spec = parse_condition(args.condition)
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
        load_model_state(checkpoint, model, role="Router oracle-gap fixed checkpoint", map_location=device, strict=True)
        model.eval()
        rows: list[dict[str, Any]] = []
        trace_files = []
        for domain_index, (component, domain) in enumerate(selected):
            loader = build_dataloader(component, loader_cfg, split="validation", experiment_seed=1)
            try:
                trace = evaluate_domain(
                    model,
                    loader,
                    cfg,
                    device,
                    condition_spec,
                    gps_scaler_mean=scaler.mean_,
                    gps_scaler_scale=scaler.scale_,
                    max_batches=args.max_batches,
                )
                domain_id = str(domain["id"])
                trace_path = output / "traces" / f"{domain_id.replace('/', '__')}.npz"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(trace_path, domain_id=np.asarray(domain_id), condition=np.asarray(args.condition), **trace)
                trace_files.append({"path": str(trace_path), "sha256": sha256(trace_path)})
                for fusion, logits_key in (
                    ("learned", "learned_logits"),
                    ("uniform", "uniform_logits"),
                    ("oracle", "oracle_logits"),
                ):
                    metrics = metrics_for(trace[logits_key], trace["target"], trace["beam_powers"], cfg)
                    rows.append(
                        {
                            "protocol": PROTOCOL_ID,
                            "condition": args.condition,
                            "domain_id": domain_id,
                            "weather": domain["condition"],
                            "scene": domain["scene"],
                            "fusion": fusion,
                            "sample_count": int(trace["target"].shape[0]),
                            **metrics,
                            "router_soft_oracle_regret": float(trace["router_soft_oracle_regret"].mean()),
                            "router_selection_oracle_regret": float(trace["router_selection_oracle_regret"].mean()),
                            **{
                                f"router_weight_{name}": float(trace["router_weights"][:, index].mean())
                                for index, name in enumerate(model.modalities)
                            },
                        }
                    )
                write_csv(output / "domain_metrics.csv", rows)
                print(f"{args.condition}: domain {domain_index + 1}/{len(selected)} {domain_id}", flush=True)
            finally:
                shutdown_dataloader_workers(loader)
        provenance = {
            "protocol": PROTOCOL_ID,
            "condition": args.condition,
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_candidate": candidate,
            "seed": 1,
            "split": "frozen_inner_validation_only",
            "claim_eligible": False,
            "batch_size": int(args.batch_size),
            "corruption_seed": CORRUPTION_SEED,
            "gps_scaler_mean": np.asarray(scaler.mean_).tolist(),
            "gps_scaler_scale": np.asarray(scaler.scale_).tolist(),
            "domain_count": len(selected),
            "partial": args.max_domains is not None or args.max_batches is not None,
            "trace_files": trace_files,
        }
        write_json(output / "provenance.json", provenance)
        if not provenance["partial"]:
            write_json(complete_path, {"status": "complete", **provenance})
    finally:
        if dataloaders is not None:
            shutdown_all_dataloaders(dataloaders)


def evaluate_domain(
    model,
    loader,
    cfg: dict[str, Any],
    device: torch.device,
    corruption: CorruptionSpec | None,
    *,
    gps_scaler_mean: Any,
    gps_scaler_scale: Any,
    max_batches: int | None,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch = prepare_evaluation_batch(raw_batch)
            if corruption is not None:
                batch = apply_inference_corruption(
                    matrix._clone_batch(batch),
                    corruption,
                    seed=CORRUPTION_SEED,
                    batch_index=batch_index,
                    gps_scaler_mean=gps_scaler_mean,
                    gps_scaler_scale=gps_scaler_scale,
                )
            batch_size = int(batch["gps"].shape[0])
            availability = torch.ones((batch_size, len(model.modalities)), device=device, dtype=torch.bool)
            model_cfg = cfg["model"]["primary"]
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
            if not bool(diagnostics["available_modalities"].all()):
                raise ValueError("Oracle gap corruptions must keep every modality available.")
            unimodal = diagnostics["unimodal_logits"]
            router = diagnostics["supervised_router_gate_weights"]
            learned = step.logits[:, -1, :] if step.logits.ndim == 3 else step.logits
            powers = matrix._load_future_beam_power(step.batch).to(device=unimodal.device)
            branches = oracle_gap_branches(unimodal, router, powers)
            reconstructed_learned = (router.unsqueeze(-1) * unimodal).sum(dim=1)
            if not torch.allclose(learned, reconstructed_learned, atol=1e-5, rtol=1e-5):
                raise ValueError("Learned fused logits are not reconstructed by the saved Router weights and unimodal logits.")
            ids = np.asarray(sample_ids_from_batch(step.batch), dtype=str)
            if len(ids) != int(target.numel()):
                raise ValueError("Sample identity count differs from the evaluated batch.")
            values = {
                "sample_id": ids,
                "target": target.detach().cpu().numpy().astype(np.int64),
                "beam_powers": powers.detach().cpu().numpy().astype(np.float32),
                "unimodal_logits": unimodal.detach().cpu().numpy().astype(np.float32),
                "router_weights": router.detach().cpu().numpy().astype(np.float32),
                "learned_logits": learned.detach().cpu().numpy().astype(np.float32),
                **{key: value.detach().cpu().numpy() for key, value in branches.items()},
            }
            for key, value in values.items():
                chunks.setdefault(key, []).append(value)
    if not chunks:
        raise ValueError("Oracle gap condition produced no samples.")
    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def oracle_gap_branches(
    unimodal_logits: torch.Tensor,
    router_weights: torch.Tensor,
    beam_powers: torch.Tensor,
    available: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if unimodal_logits.ndim != 3 or router_weights.shape != unimodal_logits.shape[:2]:
        raise ValueError("Expected unimodal logits [N,M,C] and Router weights [N,M].")
    if beam_powers.shape != (unimodal_logits.shape[0], unimodal_logits.shape[2]):
        raise ValueError("Beam powers must match [N,C].")
    if available is None:
        available_mask = torch.ones_like(router_weights, dtype=torch.bool)
        weights = router_weights / router_weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(router_weights.dtype).eps
        )
        uniform_logits = unimodal_logits.mean(dim=1)
    else:
        if available.shape != router_weights.shape:
            raise ValueError("Available modalities must match Router weights [N,M].")
        available_mask = available.to(device=router_weights.device, dtype=torch.bool)
        if not bool(available_mask.any(dim=1).all()):
            raise ValueError("Every sample must keep at least one modality available.")
        weights = router_weights.masked_fill(~available_mask, 0.0)
        weight_sum = weights.sum(dim=1, keepdim=True)
        if bool((weight_sum <= 0).any()):
            raise ValueError("Router must assign positive total weight to available modalities.")
        weights = weights / weight_sum
        uniform_weights = available_mask.to(unimodal_logits.dtype)
        uniform_weights = uniform_weights / uniform_weights.sum(dim=1, keepdim=True)
        uniform_logits = (uniform_weights.unsqueeze(-1) * unimodal_logits).sum(dim=1)
    unimodal_predictions = unimodal_logits.argmax(dim=-1)
    best_power = beam_powers.max(dim=-1).values.clamp_min(torch.finfo(beam_powers.dtype).tiny)
    modality_gain = beam_powers.gather(1, unimodal_predictions) / best_power.unsqueeze(1)
    available_gain = modality_gain.masked_fill(~available_mask, -torch.inf)
    oracle_modality = available_gain.argmax(dim=1)
    batch_index = torch.arange(unimodal_logits.shape[0], device=unimodal_logits.device)
    oracle_logits = unimodal_logits[batch_index, oracle_modality]
    oracle_gain = available_gain.max(dim=1).values
    selected_gain = modality_gain[batch_index, weights.argmax(dim=1)]
    expected_gain = (weights * modality_gain).sum(dim=1)
    return {
        "uniform_logits": uniform_logits,
        "oracle_logits": oracle_logits,
        "oracle_modality": oracle_modality.to(torch.int64),
        "unimodal_normalized_gain": modality_gain,
        "router_soft_oracle_regret": oracle_gain - expected_gain,
        "router_selection_oracle_regret": oracle_gain - selected_gain,
    }


def metrics_for(logits: np.ndarray, target: np.ndarray, beam_powers: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    scores = torch.from_numpy(logits)
    labels = torch.from_numpy(target)
    return {
        **_beam_classification_metrics(scores, labels, cfg),
        **beam_power_communication_summary(scores, torch.from_numpy(beam_powers)),
    }


def parse_condition(value: str) -> CorruptionSpec | None:
    if value == "clean":
        return None
    name, severity = value.rsplit("_s", 1)
    return CorruptionSpec(name, int(severity))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
