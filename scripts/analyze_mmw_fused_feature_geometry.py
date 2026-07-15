#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr

from eval_h5_p1_temporal_matrix_v1 import _clone_batch, _mask_in_model_order
from eval_mmw_all_weather_matrix import (
    BASELINE_SCOPES,
    HISTORY_WINDOW,
    MASK_TYPES,
    RATES,
    _load_or_create_temporal_cache,
    _matrix_digest,
)
from kd_sensing.data.temporal_missing import DEFAULT_TEMPORAL_MODALITIES, apply_modality_temporal_mask_to_batch
from kd_sensing.engine.data_factory import build_dataloader, build_dataloaders
from kd_sensing.engine.evaluation_pass_runtime import prepare_evaluation_batch, sample_ids_from_batch
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import run_model_step
from kd_sensing.utils.checkpoint import load_model_state


METHODS = ("T2", "amber_full", "rmbp_mm")
T2_ABLATION_METHODS = ("T2-NoBPA", "T2-BPA2CMA", "T2-Linear", "T2-CLS", "T2-CLS-CMA")
SUPPORTED_METHODS = (*METHODS, *T2_ABLATION_METHODS)
METHOD_LABELS = {
    "T2": "T2",
    "amber_full": "AMBER-Full",
    "rmbp_mm": "RMBP-MM",
    "T2-NoBPA": "T2 w/o BPA",
    "T2-BPA2CMA": "T2: BPA to CMA",
    "T2-Linear": "T2: linear topology",
    "T2-CLS": "T2: classifier",
    "T2-CLS-CMA": "T2: classifier + CMA",
}
NUM_CLASSES = 64
FEATURE_METRICS = (
    "feature_cosine_distance",
    "centroid_to_true_distance",
    "centroid_top1",
    "centroid_within_1",
    "centroid_within_3",
    "centroid_shift_distance",
    "centroid_shift_within_1",
    "centroid_shift_within_3",
    "centroid_assignment_same",
    "prediction_to_true_distance",
    "prediction_top1",
    "prediction_within_1",
    "prediction_within_3",
    "prediction_shift_distance",
    "prediction_shift_within_1",
    "prediction_shift_within_3",
    "prediction_same",
)
LOWER_IS_BETTER = {name for name in FEATURE_METRICS if name.endswith("distance")}
TASK_OUTPUT_FIELDS = ("sample_ids", "logits", "seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze paired MMW fused-feature geometry under temporal missingness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--root", default="outputs/mmw_all_weather_h5p1_seed1_v2")
    extract.add_argument("--output-dir", default=None)
    extract.add_argument("--mask-cache", default="outputs/mmw_all_weather_h5p1_eval_masks_v2")
    extract.add_argument("--method", choices=SUPPORTED_METHODS, required=True)
    extract.add_argument("--seed", type=int, default=1)
    extract.add_argument("--domain-shard-index", type=int, default=0)
    extract.add_argument("--domain-shard-count", type=int, default=1)
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--num-workers", type=int, default=2)
    extract.add_argument("--max-batches", type=int, default=None)
    extract.add_argument("--max-domains", type=int, default=None)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--root", default="outputs/mmw_all_weather_h5p1_seed1_v2")
    summarize.add_argument("--output-dir", default=None)
    summarize.add_argument("--methods", default=",".join(METHODS))
    summarize.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "fused_feature_geometry"
    if args.command == "extract":
        extract_method(args, root, output_dir)
    else:
        summarize_features(output_dir, tuple(_csv(args.methods)), allow_partial=bool(args.allow_partial))
    return 0


def feature_mask_specs(cache: dict[tuple[float, int], dict[str, Any]]) -> list[dict[str, Any]]:
    specs = []
    for rate in RATES:
        payload = cache[(rate, 0)]
        selected = [
            (index, item)
            for index, item in enumerate(payload["masks"])
            if rate == 0.0 or item.get("mask_type") == "modality_frame"
        ]
        expected = 1 if rate == 0.0 else 16
        if len(selected) != expected:
            raise ValueError(f"rate={rate} expected {expected} selected masks, got {len(selected)}.")
        for source_index, item in selected:
            matrix = torch.as_tensor(item["modality_temporal_mask"], dtype=torch.bool)
            specs.append(
                {
                    "rate": float(rate),
                    "source_mask_index": int(source_index),
                    "mask_type": str(item.get("mask_type", "clean")),
                    "mask_digest": _matrix_digest(matrix.to(dtype=torch.int8).tolist()),
                    "cache_checksum": str(payload["checksum"]),
                    "mask_item": item,
                }
            )
    return specs


def select_fused_feature(features: torch.Tensor | None) -> torch.Tensor:
    if not torch.is_tensor(features):
        raise ValueError("Model output did not expose output_features for fused-feature analysis.")
    if features.ndim == 2:
        return features
    if features.ndim == 3:
        return features[:, -1, :]
    raise ValueError(f"Fused output_features must be [B,D] or [B,T,D], got {tuple(features.shape)}.")


def extract_method(args: argparse.Namespace, root: Path, output_dir: Path) -> None:
    if args.domain_shard_count <= 0 or not 0 <= args.domain_shard_index < args.domain_shard_count:
        raise ValueError("domain shard requires count > 0 and 0 <= index < count")
    seed = int(args.seed)
    if seed <= 0:
        raise ValueError("seed must be positive")
    cfg_path = root / "generated_configs" / f"{args.method}_seed{seed}.yaml"
    checkpoint = root / args.method / f"seed{seed}" / "checkpoints" / "last.pth"
    if not cfg_path.exists() or not checkpoint.exists():
        raise FileNotFoundError(f"{args.method} seed{seed}: missing config or fixed-epoch last checkpoint")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    configured_seed = int(cfg.get("experiment", {}).get("seed", seed))
    if configured_seed != seed:
        raise ValueError(f"{args.method} seed mismatch: requested={seed}, config={configured_seed}")
    cfg.setdefault("temporal_missing", {})["enabled"] = False
    cfg["temporal_missing"]["mode"] = "none"
    loader_cfg = cfg.setdefault("data", {}).setdefault("dataloader", {})
    loader_cfg["validation_batch_size"] = int(args.batch_size)
    loader_cfg["test_batch_size"] = int(args.batch_size)
    loader_cfg["num_workers"] = int(args.num_workers)

    cache = _load_or_create_temporal_cache(
        Path(args.mask_cache),
        modality_frame_masks=16,
        rates=RATES,
        mask_types=MASK_TYPES,
    )
    specs = feature_mask_specs(cache)
    dataloaders = build_dataloaders(cfg)
    validation = dataloaders["validation"].dataset
    components = list(getattr(validation, "datasets", []))
    inventory = list(getattr(validation, "domain_inventory", []))
    if len(components) != 15 or len(inventory) != 15:
        raise ValueError(f"Expected 15 validation domains, got components={len(components)} inventory={len(inventory)}")

    device = build_device(cfg)
    model = build_model(cfg["model"]["primary"]).to(device)
    load_model_state(checkpoint, model, role="MMW fused-feature fixed-epoch last", map_location=device, strict=True)
    model.eval()
    checkpoint_sha256 = _sha256(checkpoint)
    selected = list(zip(components, inventory))[args.domain_shard_index :: args.domain_shard_count]
    if args.max_domains is not None:
        selected = selected[: int(args.max_domains)]
    method_dir = output_dir / args.method
    domain_dir = method_dir / "domains"
    domain_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = []
    for domain_index, (component, domain) in enumerate(selected, start=1):
        loader = build_dataloader(component, loader_cfg, split="validation", experiment_seed=seed)
        target = domain_dir / f"{_safe_name(domain['id'])}.npz"
        extract_domain(
            model,
            loader,
            cfg,
            device,
            specs,
            domain,
            seed=seed,
            checkpoint_sha256=checkpoint_sha256,
            target=target,
            max_batches=args.max_batches,
        )
        completed.append(str(domain["id"]))
        print(
            f"{args.method} shard {args.domain_shard_index}/{args.domain_shard_count}: "
            f"domain {domain_index}/{len(selected)} {domain['id']} complete, elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    if args.method == "T2" and args.domain_shard_index == 0:
        prototypes = getattr(getattr(model, "prototype_bank", None), "prototypes", None)
        if torch.is_tensor(prototypes):
            _atomic_save_npy(method_dir / "learned_prototypes.npy", prototypes.detach().float().cpu().numpy())
    _write_json(
        method_dir / f"worker_{args.domain_shard_index}_of_{args.domain_shard_count}.json",
        {
            "method": args.method,
            "method_label": METHOD_LABELS[args.method],
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "config": str(cfg_path),
            "domain_shard_index": int(args.domain_shard_index),
            "domain_shard_count": int(args.domain_shard_count),
            "completed_domains": completed,
            "feature_contract": "beam_head_input_output_features_final_prediction_slot",
            "mask_protocol": "mmw_temporal_geometry_v2_modality_frame_all_fixed_masks",
            **BASELINE_SCOPES[args.method],
        },
    )


def extract_domain(
    model: torch.nn.Module,
    dataloader,
    cfg: dict[str, Any],
    device: torch.device,
    specs: list[dict[str, Any]],
    domain: dict[str, Any],
    *,
    seed: int,
    checkpoint_sha256: str,
    target: Path,
    max_batches: int | None,
) -> None:
    masks = [_mask_in_model_order(model, spec["mask_item"], DEFAULT_TEMPORAL_MODALITIES)[0] for spec in specs]
    model_modalities = tuple(str(item) for item in getattr(model, "modalities", DEFAULT_TEMPORAL_MODALITIES))
    feature_chunks: list[list[np.ndarray]] = [[] for _ in specs]
    logit_chunks: list[list[np.ndarray]] = [[] for _ in specs]
    prediction_chunks: list[list[np.ndarray]] = [[] for _ in specs]
    label_chunks: list[np.ndarray] = []
    sample_id_chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            prepared = prepare_evaluation_batch(
                raw_batch,
                cfg=cfg,
                split_name="validation",
                difficulty_seed=int(cfg.get("experiment", {}).get("seed", 0)),
                step_index=batch_index,
            )
            clean_labels = None
            clean_sample_ids = None
            for spec_index, mask in enumerate(masks):
                batch = _clone_batch(prepared)
                apply_modality_temporal_mask_to_batch(batch, mask, modalities=model_modalities)
                modality_mask = batch["modality_mask"].to(device=device, dtype=torch.bool)
                model_cfg = cfg["model"]["primary"]
                step = run_model_step(
                    model,
                    cfg.get("experiment", {}).get("task", "fusion"),
                    batch,
                    model_cfg=model_cfg,
                    seq_length=int(model_cfg.get("seq_length", cfg.get("model", {}).get("seq_length", HISTORY_WINDOW))),
                    num_pred=int(model_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1))),
                    downsample_ratio=int(model_cfg.get("downsample_ratio", cfg.get("model", {}).get("downsample_ratio", 1))),
                    device=device,
                    extra_model_kwargs={"missing_mask": modality_mask},
                )
                fused = select_fused_feature(step.model_output.output_features)
                logits = step.logits[:, -1, :] if step.logits.ndim == 3 else step.logits
                if logits.ndim != 2 or int(logits.shape[1]) != NUM_CLASSES:
                    raise ValueError(f"Expected logits [B,{NUM_CLASSES}], got {tuple(logits.shape)}")
                labels = step.labels[:, -1] if step.labels.ndim > 1 else step.labels
                labels = labels.detach().cpu().numpy().reshape(-1).astype(np.int16)
                sample_ids = np.asarray(sample_ids_from_batch(batch), dtype=np.str_).reshape(-1)
                if sample_ids.size != labels.size or np.any(sample_ids == ""):
                    raise ValueError(
                        f"Batch sample_ids must be non-empty and align with labels: ids={sample_ids.size}, labels={labels.size}."
                    )
                if spec_index == 0:
                    clean_labels = labels
                    clean_sample_ids = sample_ids
                    label_chunks.append(labels)
                    sample_id_chunks.append(sample_ids)
                else:
                    if not np.array_equal(labels, clean_labels):
                        raise ValueError(f"Mask {spec_index} labels do not match clean sample order in batch {batch_index}.")
                    if not np.array_equal(sample_ids, clean_sample_ids):
                        raise ValueError(f"Mask {spec_index} sample_ids do not match clean sample order in batch {batch_index}.")
                feature_chunks[spec_index].append(fused.detach().float().cpu().numpy())
                logit_chunks[spec_index].append(logits.detach().float().cpu().numpy())
                prediction_chunks[spec_index].append(logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int16))
    features = np.stack([np.concatenate(chunks, axis=0) for chunks in feature_chunks], axis=0).astype(np.float32)
    logits = np.stack([np.concatenate(chunks, axis=0) for chunks in logit_chunks], axis=0).astype(np.float32)
    predictions = np.stack([np.concatenate(chunks, axis=0) for chunks in prediction_chunks], axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    sample_ids = np.concatenate(sample_id_chunks, axis=0)
    if (
        features.shape[:2] != predictions.shape
        or logits.shape != (*predictions.shape, NUM_CLASSES)
        or int(features.shape[1]) != int(labels.size)
        or sample_ids.shape != labels.shape
    ):
        raise ValueError(
            "Extracted task-output alignment mismatch: "
            f"features={features.shape}, logits={logits.shape}, predictions={predictions.shape}, "
            f"labels={labels.shape}, sample_ids={sample_ids.shape}."
        )
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("Extracted sample_ids must be unique within each domain.")
    split_path = Path(str(domain["split_path"]))
    _atomic_save_npz(
        target,
        features=features,
        logits=logits,
        predictions=predictions,
        labels=labels,
        sample_ids=sample_ids,
        seed=np.asarray(int(seed), dtype=np.int64),
        rates=np.asarray([spec["rate"] for spec in specs], dtype=np.float32),
        source_mask_indices=np.asarray([spec["source_mask_index"] for spec in specs], dtype=np.int16),
        mask_types=np.asarray([spec["mask_type"] for spec in specs]),
        mask_digests=np.asarray([spec["mask_digest"] for spec in specs]),
        cache_checksums=np.asarray([spec["cache_checksum"] for spec in specs]),
        method=np.asarray(str(getattr(model, "__class__", type(model)).__name__)),
        domain_id=np.asarray(str(domain["id"])),
        condition=np.asarray(str(domain["condition"])),
        scene=np.asarray(str(domain["scene"])),
        sample_csv=np.asarray(str(split_path)),
        sample_csv_sha256=np.asarray(_sha256(split_path)),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
    )


def summarize_features(output_dir: Path, methods: tuple[str, ...], *, allow_partial: bool = False) -> None:
    unknown = [method for method in methods if method not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    bundles = {method: load_method_bundle(output_dir, method, allow_partial=allow_partial) for method in methods}
    validate_cross_method_alignment(bundles)
    mask_rows = []
    domain_mask_rows = []
    pca_payload = {}
    for method, bundle in bundles.items():
        clean = _l2_normalize(bundle["features"][0])
        centroids, centroid_labels = clean_beam_centroids(clean, bundle["labels"])
        clean_assignments = nearest_centroid(clean, centroids, centroid_labels)
        clean_predictions = bundle["predictions"][0]
        for condition_index, rate in enumerate(bundle["rates"]):
            features = _l2_normalize(bundle["features"][condition_index])
            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "rate": float(rate),
                "source_mask_index": int(bundle["source_mask_indices"][condition_index]),
                "mask_digest": str(bundle["mask_digests"][condition_index]),
                "sample_count": int(bundle["labels"].size),
                **condition_metrics(
                    clean,
                    features,
                    bundle["labels"],
                    clean_predictions,
                    bundle["predictions"][condition_index],
                    centroids,
                    centroid_labels,
                    clean_assignments=clean_assignments,
                ),
            }
            mask_rows.append(row)
        for domain in bundle["domains"]:
            domain_clean = _l2_normalize(domain["features"][0])
            domain_clean_assignments = nearest_centroid(domain_clean, centroids, centroid_labels)
            for condition_index, rate in enumerate(domain["rates"]):
                domain_mask_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "domain_id": domain["domain_id"],
                        "condition": domain["condition"],
                        "scene": domain["scene"],
                        "rate": float(rate),
                        "source_mask_index": int(domain["source_mask_indices"][condition_index]),
                        "mask_digest": str(domain["mask_digests"][condition_index]),
                        "sample_count": int(domain["labels"].size),
                        **condition_metrics(
                            domain_clean,
                            _l2_normalize(domain["features"][condition_index]),
                            domain["labels"],
                            domain["predictions"][0],
                            domain["predictions"][condition_index],
                            centroids,
                            centroid_labels,
                            clean_assignments=domain_clean_assignments,
                        ),
                    }
                )
        center, components, explained = fit_pca(clean)
        pca_payload[method] = {
            "clean": clean,
            "centroids": centroids,
            "centroid_labels": centroid_labels,
            "center": center,
            "components": components,
            "explained": explained,
            "labels": bundle["labels"],
            "features": bundle["features"],
            "rates": bundle["rates"],
            "domains": bundle["domains"],
        }

    rate_rows = summarize_by_rate(mask_rows)
    domain_rows = summarize_domains(domain_mask_rows)
    comparisons = build_comparisons(rate_rows)
    _write_csv(output_dir / "mask_shift_metrics.csv", mask_rows)
    _write_csv(output_dir / "rate_shift_summary.csv", rate_rows)
    _write_csv(output_dir / "domain_shift_summary.csv", domain_rows)
    _write_csv(output_dir / "t2_vs_baseline_shift.csv", comparisons)

    topology_rows, topology_payload, prototype_profile = build_cycle_topology_evidence(
        output_dir,
        pca_payload,
    )
    signed_shift_rows, signed_shift_summary = build_signed_feature_shift(pca_payload)
    _write_csv(output_dir / "cycle_topology_metrics.csv", topology_rows)
    _write_csv(output_dir / "prototype_similarity_by_circular_distance.csv", prototype_profile)
    _write_csv(output_dir / "missing_signed_feature_shift.csv", signed_shift_rows)
    _write_csv(output_dir / "missing_signed_feature_shift_summary.csv", signed_shift_summary)

    plot_clean_pca(output_dir / "clean_fused_features_pca.png", pca_payload)
    plot_missing_shift_pca(output_dir / "missing_fused_feature_shift_pca.png", pca_payload)
    plot_rate_metrics(output_dir / "fused_feature_stability_metrics.png", rate_rows)
    plot_t2_prototypes(output_dir / "t2_learned_prototype_pca.png", pca_payload, output_dir / "T2" / "learned_prototypes.npy")
    plot_t2_prototype_cycle_evidence(
        output_dir / "t2_prototype_cycle_evidence.png",
        topology_payload["T2_prototypes"],
        prototype_profile,
    )
    plot_clean_centroid_cycle_comparison(
        output_dir / "clean_centroid_cycle_isomap.png",
        {method: topology_payload[f"{method}_centroids"] for method in methods},
    )
    plot_missing_signed_feature_shift(
        output_dir / "missing_signed_feature_shift.png",
        signed_shift_rows,
        signed_shift_summary,
        methods,
    )
    summary = {
        "methods": list(methods),
        "domain_count": {method: len(bundle["domains"]) for method, bundle in bundles.items()},
        "rates": list(RATES),
        "mask_count_by_rate": {str(rate): 1 if rate == 0.0 else 16 for rate in RATES},
        "feature_contract": {
            "T2": "output_features[B,D]",
            "amber_full": "output_features[B,T,D] final prediction slot",
            "rmbp_mm": "output_features[B,T,D] final prediction slot",
        },
        "pca_contract": "per-method PCA fit on L2-normalized clean features only",
        "distance_contract": "metrics computed in original L2-normalized 64D feature space",
        "cycle_topology_contract": {
            "prototype_embedding": "unsupervised 2-NN Isomap from cosine-derived chord distances",
            "centroid_embedding": "independent label-conditioned centroid Isomap with common k=3 and cosine-derived chord distances",
            "k_selection": "two local neighbors for the 1D closed prototype manifold; three is the minimum shared k connecting all method centroid graphs",
            "prototype_label_usage": "beam labels are used only for coloring, annotation, and post-hoc metrics",
            "centroid_label_usage": "validation labels define class centroids; Isomap graph and coordinates then use only centroid distances",
            "phase_metric": "mean circular resultant after optimal global rotation and reflection; angular MAE uses the same alignment",
            "primary_evidence": "original-space cosine Gram/profile and nearest-neighbor topology",
            "missing_shift": "missing minus clean leave-one-out nearest-centroid assignment in original normalized 64D space, range -32..31, averaged equally over 15 domains",
        },
        "aggregation_contract": "rate rows are pooled-sample micro metrics averaged over fixed masks; domain rows are reported separately",
        "cycle_topology_metrics": topology_rows,
        "signed_feature_shift_summary": signed_shift_summary,
        "causal_alignment_loss_claim_eligible": False,
        "causal_claim_blocker": "No matched T2 checkpoint trained with Beam Prototype Alignment Loss disabled.",
        "baseline_scope": {method: BASELINE_SCOPES[method] for method in methods},
        "artifacts": {
            "mask_metrics": "mask_shift_metrics.csv",
            "rate_summary": "rate_shift_summary.csv",
            "domain_summary": "domain_shift_summary.csv",
            "comparison": "t2_vs_baseline_shift.csv",
            "clean_pca": "clean_fused_features_pca.png",
            "missing_shift_pca": "missing_fused_feature_shift_pca.png",
            "stability_plot": "fused_feature_stability_metrics.png",
            "t2_prototypes": "t2_learned_prototype_pca.png",
            "cycle_topology_metrics": "cycle_topology_metrics.csv",
            "prototype_similarity_profile": "prototype_similarity_by_circular_distance.csv",
            "signed_feature_shift": "missing_signed_feature_shift.csv",
            "signed_feature_shift_summary": "missing_signed_feature_shift_summary.csv",
            "t2_prototype_cycle_figure": "t2_prototype_cycle_evidence.png",
            "clean_centroid_cycle_figure": "clean_centroid_cycle_isomap.png",
            "missing_signed_feature_shift_figure": "missing_signed_feature_shift.png",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_markdown(rate_rows, topology_rows, signed_shift_summary, summary),
        encoding="utf-8",
    )


def load_method_bundle(
    output_dir: Path,
    method: str,
    *,
    allow_partial: bool,
    require_task_outputs: bool = False,
) -> dict[str, Any]:
    paths = sorted((output_dir / method / "domains").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No extracted domain features found for {method}.")
    if not allow_partial and len(paths) != 15:
        raise ValueError(f"{method}: expected 15 domain artifacts, found {len(paths)}.")
    domains = []
    reference = None
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            domain = {key: payload[key].copy() for key in payload.files}
        domain = {
            **domain,
            "domain_id": str(domain["domain_id"].item()),
            "condition": str(domain["condition"].item()),
            "scene": str(domain["scene"].item()),
            "sample_csv_sha256": str(domain["sample_csv_sha256"].item()),
            "checkpoint_sha256": str(domain["checkpoint_sha256"].item()),
        }
        if "seed" in domain:
            domain["seed"] = int(domain["seed"].item())
        task_fields_present = [field in domain for field in TASK_OUTPUT_FIELDS]
        if any(task_fields_present) and not all(task_fields_present):
            missing = [field for field in TASK_OUTPUT_FIELDS if field not in domain]
            raise ValueError(f"{method}: incomplete task-output fields in {path}: {missing}")
        if require_task_outputs and not all(task_fields_present):
            raise ValueError(f"{method}: task-output fields are required in {path}: {TASK_OUTPUT_FIELDS}")

        condition_count = int(domain["features"].shape[0])
        sample_count = int(domain["labels"].size)
        if condition_count != 65 or int(domain["features"].shape[1]) != sample_count:
            raise ValueError(f"{method}: invalid extracted shape in {path}: {domain['features'].shape}.")
        if tuple(domain["predictions"].shape) != (condition_count, sample_count):
            raise ValueError(f"{method}: predictions do not align in {path}: {domain['predictions'].shape}.")
        for field in ("rates", "source_mask_indices", "mask_digests", "cache_checksums"):
            if int(domain[field].size) != condition_count:
                raise ValueError(f"{method}: {field} does not align with conditions in {path}.")
        if all(task_fields_present):
            domain["sample_ids"] = np.asarray(domain["sample_ids"], dtype=np.str_).reshape(-1)
            if domain["sample_ids"].shape != domain["labels"].shape or np.any(domain["sample_ids"] == ""):
                raise ValueError(f"{method}: sample_ids do not align in {path}.")
            if np.unique(domain["sample_ids"]).size != sample_count:
                raise ValueError(f"{method}: sample_ids are not unique in {path}.")
            if domain["logits"].dtype != np.float32 or tuple(domain["logits"].shape) != (
                condition_count,
                sample_count,
                NUM_CLASSES,
            ):
                raise ValueError(f"{method}: float32 logits do not align in {path}: {domain['logits'].shape}.")
            if int(domain["seed"]) <= 0:
                raise ValueError(f"{method}: invalid seed in {path}: {domain['seed']}.")
        identity = (
            domain["rates"].tolist(),
            domain["source_mask_indices"].tolist(),
            domain["mask_digests"].tolist(),
            domain["cache_checksums"].tolist(),
            domain["checkpoint_sha256"],
            domain.get("seed"),
        )
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError(f"{method}: condition/checkpoint provenance mismatch in {path}.")
        domains.append(domain)
    has_task_outputs = [all(field in domain for field in TASK_OUTPUT_FIELDS) for domain in domains]
    if any(has_task_outputs) and not all(has_task_outputs):
        raise ValueError(f"{method}: task-output field presence differs across domains.")
    result = {
        "domains": domains,
        "features": np.concatenate([domain["features"] for domain in domains], axis=1),
        "predictions": np.concatenate([domain["predictions"] for domain in domains], axis=1),
        "labels": np.concatenate([domain["labels"] for domain in domains], axis=0),
        "rates": domains[0]["rates"],
        "source_mask_indices": domains[0]["source_mask_indices"],
        "mask_digests": domains[0]["mask_digests"],
        "cache_checksums": domains[0]["cache_checksums"],
        "checkpoint_sha256": domains[0]["checkpoint_sha256"],
    }
    if all(has_task_outputs):
        result.update(
            {
                "logits": np.concatenate([domain["logits"] for domain in domains], axis=1),
                "sample_ids": np.concatenate([domain["sample_ids"] for domain in domains], axis=0),
                "seed": domains[0]["seed"],
            }
        )
    return result


def validate_cross_method_alignment(
    bundles: dict[str, dict[str, Any]],
    *,
    require_task_outputs: bool = False,
) -> None:
    if not bundles:
        raise ValueError("Cross-method alignment requires at least one method bundle.")
    task_output_presence = {
        method: all(all(field in domain for field in TASK_OUTPUT_FIELDS) for domain in bundle["domains"])
        for method, bundle in bundles.items()
    }
    if require_task_outputs and not all(task_output_presence.values()):
        missing = [method for method, present in task_output_presence.items() if not present]
        raise ValueError(f"Cross-method alignment requires task-output fields for: {missing}")
    if any(task_output_presence.values()) and not all(task_output_presence.values()):
        raise ValueError(f"Cross-method task-output field presence differs: {task_output_presence}")

    reference_method, reference_bundle = next(iter(bundles.items()))
    reference_domains = {domain["domain_id"]: domain for domain in reference_bundle["domains"]}
    for method, bundle in bundles.items():
        domains = {domain["domain_id"]: domain for domain in bundle["domains"]}
        if set(domains) != set(reference_domains):
            raise ValueError(f"Cross-method alignment mismatch for {method}: domain_id")
        for domain_id, reference in reference_domains.items():
            candidate = domains[domain_id]
            fields = (
                "sample_csv_sha256",
                "labels",
                "rates",
                "mask_digests",
                "cache_checksums",
            )
            if task_output_presence[method]:
                fields += ("sample_ids", "seed")
            for field in fields:
                if not np.array_equal(np.asarray(candidate[field]), np.asarray(reference[field])):
                    raise ValueError(
                        f"Cross-method alignment mismatch for {method} against {reference_method}, "
                        f"domain={domain_id}: {field}"
                    )


def clean_beam_centroids(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beam_labels = np.unique(labels.astype(np.int64))
    centroids = np.stack([features[labels == label].mean(axis=0) for label in beam_labels], axis=0)
    return _l2_normalize(centroids), beam_labels


def nearest_centroid(features: np.ndarray, centroids: np.ndarray, centroid_labels: np.ndarray) -> np.ndarray:
    return centroid_labels[np.argmax(features @ centroids.T, axis=1)]


def leave_one_out_centroid_context(
    clean: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = _l2_normalize(clean)
    beam_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    centroids, centroid_labels = clean_beam_centroids(features, beam_labels)
    class_positions = np.searchsorted(centroid_labels, beam_labels)
    if np.any(centroid_labels[class_positions] != beam_labels):
        raise ValueError("labels do not align with clean centroid labels")
    sums = np.zeros((centroid_labels.size, features.shape[1]), dtype=np.float64)
    counts = np.zeros(centroid_labels.size, dtype=np.int64)
    np.add.at(sums, class_positions, features.astype(np.float64))
    np.add.at(counts, class_positions, 1)
    if np.any(counts[class_positions] <= 1):
        raise ValueError("leave-one-out centroids require at least two samples per observed beam")
    leave_one_out = sums[class_positions] - features
    leave_one_out = _l2_normalize(leave_one_out)
    return centroids, centroid_labels, class_positions, leave_one_out


def nearest_leave_one_out_centroid(
    features: np.ndarray,
    centroids: np.ndarray,
    centroid_labels: np.ndarray,
    class_positions: np.ndarray,
    leave_one_out_true_centroids: np.ndarray,
) -> np.ndarray:
    values = _l2_normalize(features)
    scores = values @ centroids.T
    rows = np.arange(values.shape[0])
    scores[rows, class_positions] = np.sum(values * leave_one_out_true_centroids, axis=1)
    return centroid_labels[np.argmax(scores, axis=1)]


def circular_beam_distance(left: np.ndarray, right: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    delta = np.abs(np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64))
    return np.minimum(delta, int(num_classes) - delta)


def condition_metrics(
    clean: np.ndarray,
    missing: np.ndarray,
    labels: np.ndarray,
    clean_predictions: np.ndarray,
    missing_predictions: np.ndarray,
    centroids: np.ndarray,
    centroid_labels: np.ndarray,
    *,
    clean_assignments: np.ndarray | None = None,
) -> dict[str, float]:
    clean_assignments = (
        nearest_centroid(clean, centroids, centroid_labels) if clean_assignments is None else clean_assignments
    )
    missing_assignments = nearest_centroid(missing, centroids, centroid_labels)
    centroid_true = circular_beam_distance(missing_assignments, labels)
    centroid_shift = circular_beam_distance(missing_assignments, clean_assignments)
    prediction_true = circular_beam_distance(missing_predictions, labels)
    prediction_shift = circular_beam_distance(missing_predictions, clean_predictions)
    return {
        "feature_cosine_distance": float(np.mean(1.0 - np.sum(clean * missing, axis=1))),
        "centroid_to_true_distance": float(np.mean(centroid_true)),
        "centroid_top1": float(np.mean(missing_assignments == labels)),
        "centroid_within_1": float(np.mean(centroid_true <= 1)),
        "centroid_within_3": float(np.mean(centroid_true <= 3)),
        "centroid_shift_distance": float(np.mean(centroid_shift)),
        "centroid_shift_within_1": float(np.mean(centroid_shift <= 1)),
        "centroid_shift_within_3": float(np.mean(centroid_shift <= 3)),
        "centroid_assignment_same": float(np.mean(missing_assignments == clean_assignments)),
        "prediction_to_true_distance": float(np.mean(prediction_true)),
        "prediction_top1": float(np.mean(missing_predictions == labels)),
        "prediction_within_1": float(np.mean(prediction_true <= 1)),
        "prediction_within_3": float(np.mean(prediction_true <= 3)),
        "prediction_shift_distance": float(np.mean(prediction_shift)),
        "prediction_shift_within_1": float(np.mean(prediction_shift <= 1)),
        "prediction_shift_within_3": float(np.mean(prediction_shift <= 3)),
        "prediction_same": float(np.mean(missing_predictions == clean_predictions)),
    }


def summarize_by_rate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    groups = sorted({(str(row["method"]), float(row["rate"])) for row in rows})
    for method, rate in groups:
        selected = [row for row in rows if row["method"] == method and math.isclose(float(row["rate"]), rate)]
        summary = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "rate": rate,
            "mask_count": len(selected),
            "sample_count": int(selected[0]["sample_count"]),
        }
        for metric in FEATURE_METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std())
            summary[f"{metric}_worst"] = float(values.max() if metric in LOWER_IS_BETTER else values.min())
        result.append(summary)
    return result


def summarize_domains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["method"], row["domain_id"], float(row["rate"])) for row in rows})
    result = []
    for method, domain_id, rate in keys:
        selected = [
            row
            for row in rows
            if row["method"] == method and row["domain_id"] == domain_id and math.isclose(float(row["rate"]), rate)
        ]
        item = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "domain_id": domain_id,
            "condition": selected[0]["condition"],
            "scene": selected[0]["scene"],
            "rate": rate,
            "mask_count": len(selected),
            "sample_count": int(selected[0]["sample_count"]),
        }
        for metric in FEATURE_METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std())
        result.append(item)
    return result


def build_comparisons(rate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def selected(method: str, rate: float) -> dict[str, Any]:
        return next(
            row
            for row in rate_rows
            if row["method"] == method and math.isclose(float(row["rate"]), rate, abs_tol=1e-6)
        )

    result = []
    for baseline in ("amber_full", "rmbp_mm"):
        for rate in RATES[1:]:
            t2 = selected("T2", rate)
            other = selected(baseline, rate)
            row = {"baseline": baseline, "baseline_label": METHOD_LABELS[baseline], "rate": rate}
            for metric in FEATURE_METRICS:
                t2_value = float(t2[f"{metric}_mean"])
                baseline_value = float(other[f"{metric}_mean"])
                advantage = baseline_value - t2_value if metric in LOWER_IS_BETTER else t2_value - baseline_value
                row[f"{metric}_t2_advantage"] = advantage
            result.append(row)
    return result


def signed_circular_offset(
    clean: np.ndarray,
    missing: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    if num_classes <= 1:
        raise ValueError("num_classes must be greater than one")
    left = np.asarray(clean, dtype=np.int64)
    right = np.asarray(missing, dtype=np.int64)
    half = int(num_classes) // 2
    return ((right - left + half) % int(num_classes) - half).astype(np.int16)


def similarity_by_circular_distance(
    values: np.ndarray,
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> list[dict[str, Any]]:
    features = _l2_normalize(values).astype(np.float64)
    beam_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if features.shape[0] != beam_labels.size:
        raise ValueError("values and labels must have the same first dimension")
    similarity = features @ features.T
    distances = circular_beam_distance(beam_labels[:, None], beam_labels[None, :], num_classes)
    rows = []
    for distance in range(int(num_classes) // 2 + 1):
        mask = np.eye(beam_labels.size, dtype=bool) if distance == 0 else np.triu(distances == distance, k=1)
        selected = similarity[mask]
        if selected.size == 0:
            continue
        rows.append(
            {
                "circular_distance": int(distance),
                "cosine_mean": float(selected.mean()),
                "cosine_std": float(selected.std()),
                "pair_count": int(selected.size),
            }
        )
    return rows


def knn_isomap(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    features = _l2_normalize(values).astype(np.float64)
    sample_count = int(features.shape[0])
    if features.ndim != 2 or sample_count < 3:
        raise ValueError("Isomap requires a 2D array with at least three rows")
    if not 1 <= int(k) < sample_count:
        raise ValueError(f"k must satisfy 1 <= k < {sample_count}")

    similarity = np.clip(features @ features.T, -1.0, 1.0)
    distance = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))
    neighbor_distance = distance.copy()
    np.fill_diagonal(neighbor_distance, np.inf)
    neighbors = np.argsort(neighbor_distance, axis=1, kind="stable")[:, : int(k)]
    adjacency = np.zeros((sample_count, sample_count), dtype=bool)
    adjacency[np.arange(sample_count)[:, None], neighbors] = True
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, False)
    component_count = int(connected_components(adjacency, directed=False, return_labels=False))
    if component_count != 1:
        raise ValueError(f"{k}-NN graph is disconnected ({component_count} components)")

    graph_distance = np.where(adjacency, distance, np.inf)
    np.fill_diagonal(graph_distance, 0.0)
    geodesic = shortest_path(graph_distance, directed=False)
    center = np.eye(sample_count) - np.ones((sample_count, sample_count)) / sample_count
    gram = -0.5 * center @ (geodesic**2) @ center
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    indices = np.argsort(eigenvalues)[::-1][:2]
    if np.any(eigenvalues[indices] <= 0):
        raise ValueError("Isomap did not produce two positive embedding dimensions")
    embedding = eigenvectors[:, indices] * np.sqrt(eigenvalues[indices])
    for axis in range(2):
        pivot = int(np.argmax(np.abs(embedding[:, axis])))
        if embedding[pivot, axis] < 0:
            embedding[:, axis] *= -1
    return embedding.astype(np.float32), adjacency


def cycle_embedding_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> dict[str, float]:
    points = np.asarray(embedding, dtype=np.float64)
    beam_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if points.shape != (beam_labels.size, 2):
        raise ValueError("embedding must be [N,2] and align with labels")
    points = points - points.mean(axis=0, keepdims=True)
    angles = np.arctan2(points[:, 1], points[:, 0])
    target = 2.0 * np.pi * beam_labels / int(num_classes)

    candidates = []
    for orientation in (1.0, -1.0):
        raw = orientation * angles - target
        mean_phase = np.mean(np.exp(1j * raw))
        residual = np.angle(np.exp(1j * (raw - np.angle(mean_phase))))
        candidates.append((abs(mean_phase), residual))
    phase_consistency, residual = max(candidates, key=lambda item: item[0])

    pairwise_2d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    np.fill_diagonal(pairwise_2d, np.inf)
    nearest = np.argmin(pairwise_2d, axis=1)
    nearest_distance = circular_beam_distance(beam_labels, beam_labels[nearest], num_classes)

    upper = np.triu_indices(beam_labels.size, k=1)
    label_distance = circular_beam_distance(
        beam_labels[upper[0]],
        beam_labels[upper[1]],
        num_classes,
    )
    angular_distance = np.abs(
        np.angle(np.exp(1j * (angles[upper[0]] - angles[upper[1]])))
    ) * int(num_classes) / (2.0 * np.pi)
    angular_rho = float(spearmanr(label_distance, angular_distance).statistic)
    radius = np.linalg.norm(points, axis=1)
    return {
        "phase_consistency": float(phase_consistency),
        "angular_mae_beams": float(np.mean(np.abs(residual)) * int(num_classes) / (2.0 * np.pi)),
        "angular_distance_spearman": angular_rho,
        "embedding_nn_within_1": float(np.mean(nearest_distance <= 1)),
        "embedding_nn_within_3": float(np.mean(nearest_distance <= 3)),
        "radius_cv": float(radius.std() / max(radius.mean(), np.finfo(np.float64).eps)),
    }


def topology_evidence(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
    representation: str,
    method: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = _l2_normalize(values).astype(np.float64)
    beam_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    embedding, adjacency = knn_isomap(features, k)
    similarity = features @ features.T
    nearest_similarity = similarity.copy()
    np.fill_diagonal(nearest_similarity, -np.inf)
    nearest = np.argmax(nearest_similarity, axis=1)
    nearest_distance = circular_beam_distance(beam_labels, beam_labels[nearest], NUM_CLASSES)
    upper = np.triu_indices(beam_labels.size, k=1)
    pair_distance = circular_beam_distance(beam_labels[upper[0]], beam_labels[upper[1]], NUM_CLASSES)
    similarity_rho = float(spearmanr(pair_distance, similarity[upper]).statistic)
    edge_left, edge_right = np.where(np.triu(adjacency, k=1))
    edge_distance = circular_beam_distance(beam_labels[edge_left], beam_labels[edge_right], NUM_CLASSES)
    metrics = cycle_embedding_metrics(embedding, beam_labels, NUM_CLASSES)
    row = {
        "representation": representation,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "class_count": int(beam_labels.size),
        "knn_k": int(k),
        "graph_edge_count": int(edge_distance.size),
        "graph_edges_within_1": float(np.mean(edge_distance <= 1)),
        "graph_edges_within_3": float(np.mean(edge_distance <= 3)),
        "original_nn_within_1": float(np.mean(nearest_distance <= 1)),
        "original_nn_within_3": float(np.mean(nearest_distance <= 3)),
        "distance_similarity_spearman": similarity_rho,
        **metrics,
    }
    return row, {
        "values": features.astype(np.float32),
        "labels": beam_labels,
        "embedding": embedding,
        "adjacency": adjacency,
        "metrics": row,
    }


def build_cycle_topology_evidence(
    output_dir: Path,
    payload: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    prototype_path = output_dir / "T2" / "learned_prototypes.npy"
    if "T2" not in payload or not prototype_path.exists():
        raise FileNotFoundError("T2 learned prototypes are required for cycle topology analysis")
    prototypes = _l2_normalize(np.load(prototype_path, allow_pickle=False))
    prototype_labels = np.arange(prototypes.shape[0], dtype=np.int64)
    prototype_row, prototype_payload = topology_evidence(
        prototypes,
        prototype_labels,
        k=2,
        representation="learned_prototypes",
        method="T2",
    )
    rows = [prototype_row]
    result = {"T2_prototypes": prototype_payload}
    for method, item in payload.items():
        row, method_payload = topology_evidence(
            item["centroids"],
            item["centroid_labels"],
            k=3,
            representation="clean_class_centroids",
            method=method,
        )
        rows.append(row)
        result[f"{method}_centroids"] = method_payload
    return rows, result, similarity_by_circular_distance(prototypes, prototype_labels, NUM_CLASSES)


def build_signed_feature_shift(
    payload: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    summaries = []
    offsets = np.arange(-NUM_CLASSES // 2, NUM_CLASSES // 2, dtype=np.int64)
    for method, item in payload.items():
        centroids, centroid_labels, class_positions, leave_one_out = leave_one_out_centroid_context(
            item["clean"],
            item["labels"],
        )
        clean_assignments = nearest_leave_one_out_centroid(
            item["clean"],
            centroids,
            centroid_labels,
            class_positions,
            leave_one_out,
        )
        domain_counts = [int(domain["labels"].size) for domain in item["domains"]]
        if sum(domain_counts) != int(item["labels"].size):
            raise ValueError(f"{method}: domain sample counts do not align with pooled features")
        boundaries = np.cumsum([0, *domain_counts])
        for rate in RATES[1:]:
            indices = np.flatnonzero(np.isclose(item["rates"], rate))
            shifts = []
            for index in indices:
                missing = _l2_normalize(item["features"][index])
                missing_assignments = nearest_leave_one_out_centroid(
                    missing,
                    centroids,
                    centroid_labels,
                    class_positions,
                    leave_one_out,
                )
                shifts.append(signed_circular_offset(clean_assignments, missing_assignments, NUM_CLASSES))
            shift_matrix = np.stack(shifts, axis=0).astype(np.int64)
            values = shift_matrix.reshape(-1)
            counts = np.asarray([(values == offset).sum() for offset in offsets], dtype=np.int64)
            total = int(counts.sum())
            domain_values = [
                shift_matrix[:, boundaries[index] : boundaries[index + 1]].reshape(-1)
                for index in range(len(domain_counts))
            ]
            domain_fractions = np.stack(
                [
                    np.asarray([(values == offset).mean() for offset in offsets], dtype=np.float64)
                    for values in domain_values
                ],
                axis=0,
            )
            macro_fraction = domain_fractions.mean(axis=0)
            for offset, count, domain_macro in zip(offsets, counts, macro_fraction):
                rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "rate": float(rate),
                        "signed_offset": int(offset),
                        "count": int(count),
                        "fraction": float(count / total),
                        "percentage": float(100.0 * count / total),
                        "domain_macro_fraction": float(domain_macro),
                        "domain_macro_percentage": float(100.0 * domain_macro),
                        "mask_count": int(indices.size),
                        "sample_count": int(item["labels"].size),
                        "domain_count": len(domain_counts),
                    }
                )
            domain_absolute = [np.abs(values) for values in domain_values]
            summaries.append(
                {
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "rate": float(rate),
                    "mask_count": int(indices.size),
                    "sample_count": int(item["labels"].size),
                    "observation_count": total,
                    "domain_count": len(domain_counts),
                    "aggregation": "equal_15_domain_macro",
                    "assignment_same": float(np.mean([np.mean(values == 0) for values in domain_absolute])),
                    "shift_within_1": float(np.mean([np.mean(values <= 1) for values in domain_absolute])),
                    "shift_within_3": float(np.mean([np.mean(values <= 3) for values in domain_absolute])),
                    "mean_abs_shift": float(np.mean([values.mean() for values in domain_absolute])),
                    "p95_abs_shift": float(np.mean([np.quantile(values, 0.95) for values in domain_absolute])),
                }
            )
    return rows, summaries


def fit_pca(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    center = values.mean(axis=0)
    centered = values - center
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    components = right[:2].copy()
    for index in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            components[index] *= -1
    variance = singular_values**2
    explained = variance[:2] / max(float(variance.sum()), np.finfo(np.float64).eps)
    return center.astype(np.float32), components.astype(np.float32), explained.astype(np.float32)


def project_pca(features: np.ndarray, center: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (np.asarray(features) - center) @ components.T


def plot_clean_pca(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, len(payload), figsize=(15, 4.5), squeeze=False, layout="constrained")
    cmap = plt.get_cmap("twilight_shifted")
    for axis, (method, item) in zip(axes[0], payload.items()):
        points = project_pca(item["clean"], item["center"], item["components"])
        axis.scatter(points[:, 0], points[:, 1], c=item["labels"], cmap=cmap, vmin=0, vmax=63, s=8, alpha=0.55, linewidths=0)
        axis.set_title(f"{METHOD_LABELS[method]} clean\nPC1+PC2={100 * item['explained'].sum():.1f}%")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=0.15)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 63)), ax=axes.ravel().tolist(), shrink=0.85)
    colorbar.set_label("Beam index")
    fig.suptitle("Full-modality fused features (per-method clean PCA)")
    _save_figure(fig, path)


def plot_missing_shift_pca(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    rates = RATES[1:]
    fig, axes = plt.subplots(len(payload), len(rates), figsize=(18, 14), squeeze=False, layout="constrained")
    cmap = plt.get_cmap("twilight_shifted")
    for row_index, (method, item) in enumerate(payload.items()):
        clean_points = project_pca(item["clean"], item["center"], item["components"])
        for column_index, rate in enumerate(rates):
            axis = axes[row_index, column_index]
            indices = np.flatnonzero(np.isclose(item["rates"], rate))
            missing = _l2_normalize(_l2_normalize(item["features"][indices]).mean(axis=0))
            missing_points = project_pca(missing, item["center"], item["components"])
            axis.scatter(clean_points[:, 0], clean_points[:, 1], color="#b7bcc5", s=5, alpha=0.12, linewidths=0)
            axis.scatter(
                missing_points[:, 0], missing_points[:, 1], c=item["labels"], cmap=cmap, vmin=0, vmax=63, s=7, alpha=0.5, linewidths=0
            )
            for label in np.unique(item["labels"]):
                selected = item["labels"] == label
                if int(selected.sum()) < 5:
                    continue
                start = clean_points[selected].mean(axis=0)
                end = missing_points[selected].mean(axis=0)
                axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#20242b", "alpha": 0.35, "lw": 0.7})
            axis.set_title(f"{METHOD_LABELS[method]} missing {int(rate * 100)}%")
            axis.set_xlabel("clean PC1")
            axis.set_ylabel("clean PC2")
            axis.grid(alpha=0.12)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 63)), ax=axes.ravel().tolist(), shrink=0.75)
    colorbar.set_label("Beam index")
    fig.suptitle("Paired fused-feature shift (gray=clean, color=mean over 16 fixed masks)")
    _save_figure(fig, path)


def plot_rate_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    panels = (
        ("feature_cosine_distance_mean", "Paired cosine distance", False),
        ("centroid_shift_within_3_mean", "Nearest-centroid shift within +/-3 beams", True),
        ("prediction_shift_within_3_mean", "Prediction shift within +/-3 beams", True),
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.5), layout="constrained")
    for axis, (metric, title, percentage) in zip(axes, panels):
        for method in (item for item in METHODS if any(row["method"] == item for row in rows)):
            selected = sorted((row for row in rows if row["method"] == method), key=lambda row: float(row["rate"]))
            x = [100 * float(row["rate"]) for row in selected]
            y = [float(row[metric]) * (100 if percentage else 1) for row in selected]
            yerr = [float(row[metric.replace("_mean", "_std")]) * (100 if percentage else 1) for row in selected]
            axis.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=METHOD_LABELS[method])
        axis.set_title(title)
        axis.set_xlabel("Missing modality-time cells (%)")
        axis.set_ylabel("Percent" if percentage else "Cosine distance")
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Fused-feature stability in original 64D space (mean +/- mask std)")
    _save_figure(fig, path)


def plot_t2_prototypes(path: Path, payload: dict[str, dict[str, Any]], prototype_path: Path) -> None:
    if "T2" not in payload or not prototype_path.exists():
        return
    item = payload["T2"]
    prototypes = _l2_normalize(np.load(prototype_path, allow_pickle=False))
    prototype_points = project_pca(prototypes, item["center"], item["components"])
    clean_points = project_pca(item["clean"], item["center"], item["components"])
    indices = np.flatnonzero(np.isclose(item["rates"], 0.8))
    missing = _l2_normalize(_l2_normalize(item["features"][indices]).mean(axis=0))
    missing_points = project_pca(missing, item["center"], item["components"])
    cmap = plt.get_cmap("twilight_shifted")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for axis, points, title in ((axes[0], clean_points, "T2 clean"), (axes[1], missing_points, "T2 missing 80%")):
        axis.scatter(points[:, 0], points[:, 1], c=item["labels"], cmap=cmap, vmin=0, vmax=63, s=7, alpha=0.4, linewidths=0)
        axis.scatter(
            prototype_points[:, 0], prototype_points[:, 1], c=np.arange(NUM_CLASSES), cmap=cmap, vmin=0, vmax=63, marker="*", s=55, edgecolors="#20242b", linewidths=0.45
        )
        axis.set_title(title)
        axis.set_xlabel("clean PC1")
        axis.set_ylabel("clean PC2")
        axis.grid(alpha=0.15)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 63)), ax=axes.tolist(), shrink=0.8, label="Beam/prototype index")
    fig.suptitle("T2 fused features and learned beam prototypes")
    _save_figure(fig, path)


def _normalized_embedding(embedding: np.ndarray) -> np.ndarray:
    points = np.asarray(embedding, dtype=np.float64)
    points = points - points.mean(axis=0, keepdims=True)
    scale = max(float(np.max(np.linalg.norm(points, axis=1))), np.finfo(np.float64).eps)
    return points / scale


def _draw_cycle_embedding(axis: plt.Axes, item: dict[str, Any], *, annotate: bool = True) -> None:
    points = _normalized_embedding(item["embedding"])
    adjacency = np.asarray(item["adjacency"], dtype=bool)
    labels = np.asarray(item["labels"], dtype=np.int64)
    for left, right in zip(*np.where(np.triu(adjacency, k=1))):
        axis.plot(
            points[[left, right], 0],
            points[[left, right], 1],
            color="#7d8590",
            alpha=0.32,
            linewidth=0.65,
            zorder=1,
        )
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap="twilight_shifted",
        vmin=0,
        vmax=NUM_CLASSES - 1,
        s=34,
        edgecolors="#20242b",
        linewidths=0.35,
        zorder=2,
    )
    if annotate:
        for label in (0, 16, 32, 48, 63):
            selected = np.flatnonzero(labels == label)
            if selected.size:
                index = int(selected[0])
                axis.annotate(
                    str(label),
                    points[index],
                    xytext={0: (5, 7), 63: (5, -11)}.get(label, (4, 4)),
                    textcoords="offset points",
                    fontsize=8,
                    color="#20242b",
                )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-1.18, 1.18)
    axis.set_ylim(-1.18, 1.18)
    axis.set_xticks([])
    axis.set_yticks([])


def plot_t2_prototype_cycle_evidence(
    path: Path,
    item: dict[str, Any],
    profile_rows: list[dict[str, Any]],
) -> None:
    values = np.asarray(item["values"], dtype=np.float64)
    similarity = values @ values.T
    display = similarity.copy()
    np.fill_diagonal(display, np.nan)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#f0f0f0")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), layout="constrained")

    image = axes[0].imshow(
        display,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-0.4, vcenter=0.0, vmax=0.7),
        interpolation="nearest",
        origin="lower",
    )
    axes[0].scatter([63, 0], [0, 63], marker="s", s=52, facecolors="none", edgecolors="#111111", linewidths=1.0)
    axes[0].set_title("A  64D cosine Gram matrix", fontsize=8.5)
    axes[0].set_xlabel("Prototype index")
    axes[0].set_ylabel("Prototype index")
    axes[0].set_xticks([0, 16, 32, 48, 63])
    axes[0].set_yticks([0, 16, 32, 48, 63])
    fig.colorbar(image, ax=axes[0], shrink=0.82, label="Cosine similarity")

    _draw_cycle_embedding(axes[1], item)
    metrics = item["metrics"]
    axes[1].set_title(
        "B  Unsupervised 2-NN Isomap\n"
        f"phase={metrics['phase_consistency']:.3f}, MAE={metrics['angular_mae_beams']:.2f} beam",
        fontsize=8.5,
    )

    selected = [row for row in profile_rows if int(row["circular_distance"]) > 0]
    distance = np.asarray([row["circular_distance"] for row in selected], dtype=np.float64)
    mean = np.asarray([row["cosine_mean"] for row in selected], dtype=np.float64)
    std = np.asarray([row["cosine_std"] for row in selected], dtype=np.float64)
    axes[2].fill_between(distance, mean - std, mean + std, color="#56b4e9", alpha=0.24, linewidth=0)
    axes[2].plot(distance, mean, color="#0072b2", marker="o", markersize=3.4, linewidth=1.6)
    axes[2].axhline(0.0, color="#5f6368", linewidth=0.8, linestyle="--")
    axes[2].set_xlim(1, NUM_CLASSES // 2)
    axes[2].set_xlabel("Circular beam distance")
    axes[2].set_ylabel("Prototype cosine similarity")
    axes[2].set_title(
        "C  64D similarity vs. beam distance\n"
        f"Spearman={metrics['distance_similarity_spearman']:.3f}",
        fontsize=8.5,
    )
    axes[2].grid(alpha=0.18)
    fig.suptitle("High-dimensional cyclic topology of T2 prototypes", fontsize=10)
    _save_paper_figure(fig, path)


def plot_clean_centroid_cycle_comparison(
    path: Path,
    payload: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(1, len(payload), figsize=(7.16, 2.35), squeeze=False, layout="constrained")
    letters = "ABC"
    for index, (method, item) in enumerate(payload.items()):
        axis = axes[0, index]
        _draw_cycle_embedding(axis, item)
        metrics = item["metrics"]
        axis.set_title(
            f"{letters[index]}  {METHOD_LABELS[method]}\n"
            f"phase={metrics['phase_consistency']:.3f}, MAE={metrics['angular_mae_beams']:.2f} beam",
            fontsize=8.5,
        )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="twilight_shifted", norm=plt.Normalize(0, NUM_CLASSES - 1)),
        ax=axes.ravel().tolist(),
        shrink=0.82,
    )
    colorbar.set_label("Beam index")
    class_count = int(next(iter(payload.values()))["metrics"]["class_count"])
    fig.suptitle(
        f"Label-conditioned clean centroids: independent Isomap (common k=3, n={class_count})",
        fontsize=10,
    )
    _save_paper_figure(fig, path)


def plot_missing_signed_feature_shift(
    path: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    methods: tuple[str, ...],
) -> None:
    rates = list(RATES[1:])
    matrices = {}
    for method in methods:
        matrices[method] = np.asarray(
            [
                [
                    next(
                        float(row["domain_macro_percentage"])
                        for row in rows
                        if row["method"] == method
                        and math.isclose(float(row["rate"]), rate)
                        and int(row["signed_offset"]) == offset
                    )
                    for offset in range(-NUM_CLASSES // 2, NUM_CLASSES // 2)
                ]
                for rate in rates
            ],
            dtype=np.float64,
        )
    vmax = max(float(matrix.max()) for matrix in matrices.values())
    norm = PowerNorm(gamma=0.38, vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, len(methods), figsize=(7.16, 2.35), squeeze=False, layout="constrained")
    image = None
    for axis, method in zip(axes[0], methods):
        image = axis.imshow(matrices[method], aspect="auto", cmap="viridis", norm=norm, interpolation="nearest")
        labels = []
        for rate in rates:
            item = next(
                row
                for row in summaries
                if row["method"] == method and math.isclose(float(row["rate"]), rate)
            )
            labels.append(
                f"{100 * rate:.0f}% | {100 * float(item['assignment_same']):.1f} / "
                f"{100 * float(item['shift_within_3']):.1f}"
            )
        axis.set_yticks(range(len(rates)), labels, fontsize=7)
        axis.set_xticks([0, 16, 32, 48, 63], [-32, -16, 0, 16, 31], fontsize=7)
        axis.axvline(32, color="white", linewidth=0.9, alpha=0.9)
        axis.axvline(29, color="white", linewidth=0.6, linestyle="--", alpha=0.65)
        axis.axvline(35, color="white", linewidth=0.6, linestyle="--", alpha=0.65)
        axis.set_xlabel("Signed assignment shift (beam)", fontsize=8)
        axis.set_title(METHOD_LABELS[method], fontsize=9)
    axes[0, 0].set_ylabel("Missing | exact / within +/-3 (%)", fontsize=8)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
        colorbar.set_label("15-domain macro probability (%)\n(power norm, gamma=0.38)", fontsize=7.5)
        colorbar.ax.tick_params(labelsize=7)
    fig.suptitle("Signed leave-one-out centroid shift under missing inputs", fontsize=10)
    _save_paper_figure(fig, path)


def render_markdown(
    rows: list[dict[str, Any]],
    topology_rows: list[dict[str, Any]],
    signed_shift_summary: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# MMW fused-feature geometry",
        "",
        "Seed1 local validation. PCA is fit separately on each method's clean features; quantitative distances use the original normalized 64D space.",
        "The table reports pooled-sample micro metrics averaged over fixed masks; per-domain results are in `domain_shift_summary.csv`.",
        "",
        "| Method | Missing | Cosine drift | Centroid Top1 | Prediction Top1 | Prediction shift within 3 | Prediction within 3 of truth |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_label']} | {100 * float(row['rate']):.0f}% | "
            f"{float(row['feature_cosine_distance_mean']):.4f} | "
            f"{100 * float(row['centroid_top1_mean']):.2f}% | "
            f"{100 * float(row['prediction_top1_mean']):.2f}% | "
            f"{100 * float(row['prediction_shift_within_3_mean']):.2f}% | "
            f"{100 * float(row['prediction_within_3_mean']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## High-dimensional cycle topology",
            "",
            "Prototype Isomap coordinates use only original 64D cosine-derived distances. Clean class centroids are label-conditioned; after centroid construction, each method's Isomap graph and coordinates use only centroid distances with the same k=3.",
            "",
            "| Representation | Method | k | Phase consistency | Angular MAE | Original NN within 1 | Original NN within 3 | Distance-similarity Spearman |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in topology_rows:
        lines.append(
            f"| {row['representation']} | {row['method_label']} | {int(row['knn_k'])} | "
            f"{float(row['phase_consistency']):.3f} | {float(row['angular_mae_beams']):.2f} beam | "
            f"{100 * float(row['original_nn_within_1']):.2f}% | "
            f"{100 * float(row['original_nn_within_3']):.2f}% | "
            f"{float(row['distance_similarity_spearman']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Signed fused-feature shift",
            "",
            "Offsets compare leave-one-out nearest clean-centroid assignments in the original normalized 64D space, then average each normalized histogram equally over 15 domains; they are not distances measured in a 2D plot.",
            "",
            "| Method | Missing | Same assignment | Within 1 | Within 3 | Mean absolute shift | P95 absolute shift |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in signed_shift_summary:
        lines.append(
            f"| {row['method_label']} | {100 * float(row['rate']):.0f}% | "
            f"{100 * float(row['assignment_same']):.2f}% | "
            f"{100 * float(row['shift_within_1']):.2f}% | "
            f"{100 * float(row['shift_within_3']):.2f}% | "
            f"{float(row['mean_abs_shift']):.3f} | {float(row['p95_abs_shift']):.1f} |"
        )
    lines.extend(
        [
            "",
            "T2 and AMBER-Full are comparable rather than uniformly ordered: at 80% missing T2 retains a slightly higher exact assignment rate, while AMBER-Full has higher within-3 mass and lower mean absolute shift. Both are substantially more stable than RMBP-MM in this local diagnostic.",
        ]
    )
    lines.extend(
        [
            "",
            "## Suggested figure captions",
            "",
            "**T2 prototype cycle.** (A) Cosine Gram matrix of the 64 learned prototypes in the original normalized 64D space; the masked diagonal prevents self-similarity from dominating the color scale, and outlined corner cells show the beam 0/63 wrap-around. (B) Unsupervised 2-NN Isomap built only from cosine-derived distances, with beam indices used after embedding for color and annotation. (C) Mean prototype cosine similarity versus circular beam distance; shading denotes one standard deviation across prototype pairs.",
            "",
            "**Clean centroid comparison.** Independent 3-NN Isomap of label-conditioned clean fused-feature class centroids for T2 and the two locally adapted baselines, using one common k and procedure. The validation inventory contains 63 observed beam classes (beam 23 is absent). After labeled centroid construction, coordinates and graph edges are derived only from original 64D centroid cosine geometry. Absolute orientation and scale are not compared across models.",
            "",
            "**Missing-feature shift.** Distribution of signed changes in leave-one-out nearest clean-centroid assignment after 20-80% modality-time cell removal. Per-domain distributions use the same samples and 16 fixed masks, then receive equal weight across 15 domains. Panels share a power-normalized color scale (gamma=0.38) to expose low-probability tails; dashed lines mark +/-3 beams.",
        ]
    )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            f"Causal Beam Prototype Alignment Loss claim eligible: `{summary['causal_alignment_loss_claim_eligible']}`.",
            "A matched T2 checkpoint trained with the alignment loss disabled is required to isolate the loss effect from architecture and training differences.",
        ]
    )
    return "\n".join(lines) + "\n"


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, np.finfo(np.float32).eps)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_paper_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _safe_name(value: str) -> str:
    return str(value).replace("/", "__").replace(" ", "_")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
