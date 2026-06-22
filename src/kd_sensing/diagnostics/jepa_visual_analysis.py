import csv
import hashlib
import json
import math
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate

from kd_sensing.config.io import deep_merge, load_config, parse_overrides
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.diagnostics.jepa_benchmark_artifacts import OutputRegistry
from kd_sensing.diagnostics.gps_query_evidence import (
    DEFAULT_GPS_QUERY_EVIDENCE_CONFIG,
    gps_query_evidence_enabled,
    write_gps_query_evidence_package,
)
from kd_sensing.diagnostics.jepa_gps_shortcut_benchmark import (
    GPS_SUITE_TYPES,
    IMAGE_SUITE_TYPES,
    TEMPORAL_SUITE_TYPES,
    read_benchmark_analysis_bundle,
)
from kd_sensing.engine.data_factory import (
    build_dataloader_kwargs,
    build_protocol_split_datasets,
    build_split_dataset,
    prepare_lidar_normalizer,
    shutdown_dataloader_workers,
)
from kd_sensing.engine.evaluation_pass import run_evaluation_pass
from kd_sensing.engine.evaluator import _apply_csi_rms_to_model_config
from kd_sensing.engine.modality_resolution import (
    config_uses_csi,
    config_uses_gps,
    config_uses_lidar,
    config_uses_mmwave,
    resolve_enabled_modalities,
)
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.optim import build_device, build_model, build_task_criterion
from kd_sensing.engine.run_metadata import dataset_run_metadata
from kd_sensing.engine.runtime import (
    autocast_context,
    resolve_amp_settings,
    run_model_step,
    transfer_non_blocking,
)
from kd_sensing.evaluation.horizon_selection import horizon_indices, metric_horizons_from_config
from kd_sensing.evaluation.metrics import (
    calculate_dba_score,
    calculate_topk_accuracy,
    circular_beam_distance,
)
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state
from kd_sensing.utils.seed import set_seed


ANALYSIS_VERSION = "jepa_visual_analysis_suite_v1"
DEFAULT_CASE_GROUPS = ("query_gain", "query_regression", "shared_near_miss", "shared_failure")
DEFAULT_FIGURES = {
    "embedding": True,
    "error_anatomy": True,
    "attention": True,
    "case_studies": True,
    "robustness": True,
}
DEFAULT_OUTPUT_FORMATS = ("png", "svg")


@dataclass
class ModelAnalysis:
    name: str
    sample_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    logits: np.ndarray
    probabilities: np.ndarray
    labels: np.ndarray
    sample_ids: list[str]
    metadata_rows: list[dict[str, Any]]
    split_metadata: dict[str, Any]
    checkpoint_load: dict[str, Any] | None
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    attention_rows: list[dict[str, Any]] = field(default_factory=list)
    attention_maps: dict[str, np.ndarray] = field(default_factory=dict)
    robustness_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_analysis_config(
    analysis_config: str | Path,
    *,
    output_dir: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    path = Path(analysis_config)
    raw = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"JEPA visual analysis config must be a mapping: {path}")
    cfg = deep_merge(_default_analysis_config(), raw)
    if overrides:
        cfg = deep_merge(cfg, parse_overrides(overrides))
    if output_dir is not None:
        cfg.setdefault("outputs", {})["output_dir"] = str(output_dir)
    _validate_analysis_config(cfg, path=path)
    cfg["_analysis_config_path"] = str(path)
    cfg["_analysis_config_digest"] = _sha1_text(path.read_text(encoding="utf-8"))
    return cfg


def run_jepa_visual_analysis(
    *,
    analysis_config: str | Path,
    output_dir: str | Path | None = None,
    overrides: Iterable[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    command: list[str] | None = None,
) -> dict[str, Any]:
    cfg = load_analysis_config(analysis_config, output_dir=output_dir, overrides=overrides)
    out = Path(str(cfg.get("outputs", {}).get("output_dir") or output_dir or "outputs/visual_analysis/jepa"))
    _prepare_output_dir(out, force=force)
    tables_dir = out / "tables"
    figures_dir = out / "figures"
    cache_dir = out / "cache"
    payload_dir = out / "case_payloads"
    for directory in (tables_dir, figures_dir, cache_dir, payload_dir):
        directory.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    registry = OutputRegistry(out)
    benchmark_context = _write_benchmark_analysis_outputs(cfg, tables_dir, figures_dir, payload_dir, registry, warnings)
    model_specs = dict(cfg.get("models", {}) or {})
    analyses: dict[str, ModelAnalysis] = {}
    model_failures: dict[str, str] = {}

    if not dry_run:
        for model_name, model_spec in model_specs.items():
            try:
                analysis = _analyze_model(
                    model_name,
                    model_spec,
                    cfg,
                    tables_dir=tables_dir,
                    cache_dir=cache_dir,
                    warnings=warnings,
                )
            except Exception as exc:
                if not bool(model_spec.get("optional", False)):
                    raise
                message = f"model_failed:{model_name}:{exc}"
                warnings.append(message)
                model_failures[model_name] = str(exc)
                continue
            analyses[model_name] = analysis
            warnings.extend(analysis.warnings)
    else:
        warnings.append("dry_run:no_model_forward_executed")

    comparison_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    embedding_neighbor_rows: list[dict[str, Any]] = []

    if analyses:
        _write_model_metrics(tables_dir / "model_metrics.csv", analyses)
        if bool(cfg.get("figures", {}).get("error_anatomy", True)):
            _write_error_anatomy_outputs(figures_dir, tables_dir, analyses, cfg, registry, warnings)
        comparison_rows = _write_comparison_outputs(tables_dir, analyses, cfg, warnings)
        embedding_neighbor_rows = _write_embedding_outputs(figures_dir, tables_dir, analyses, cfg, registry, warnings)
        _write_attention_outputs(figures_dir, tables_dir, analyses, cfg, registry, warnings)
        if bool(cfg.get("figures", {}).get("case_studies", True)):
            case_rows = _write_case_study_outputs(
                figures_dir,
                payload_dir,
                tables_dir,
                comparison_rows,
                analyses,
                cfg,
                registry,
                warnings,
            )
        if bool(cfg.get("figures", {}).get("robustness", True)):
            robustness_rows = _write_robustness_outputs(
                figures_dir,
                tables_dir,
                analyses,
                cfg,
                registry,
                warnings,
            )
    else:
        registry.skipped_output(tables_dir / "model_metrics.csv", reason="no_completed_models", kind="table")
        registry.skipped_output(tables_dir / "comparison_samples.csv", reason="no_completed_models", kind="table")

    evidence_context: dict[str, Any] = {"enabled": False}
    if gps_query_evidence_enabled(cfg):
        evidence_context = write_gps_query_evidence_package(
            cfg,
            output_dir=out,
            analyses=analyses,
            comparison_rows=comparison_rows,
            command=command,
            warnings=warnings,
            formats=_output_formats(cfg),
            dpi=int(cfg.get("outputs", {}).get("dpi", 180)),
        )

    manifest = _build_manifest(
        cfg,
        output_dir=out,
        command=command,
        analyses=analyses,
        warnings=warnings,
        model_failures=model_failures,
        dry_run=dry_run,
        registry=registry,
        benchmark_context=benchmark_context,
        evidence_context=evidence_context,
    )
    report = _build_report(
        cfg,
        analyses=analyses,
        comparison_rows=comparison_rows,
        embedding_neighbor_rows=embedding_neighbor_rows,
        case_rows=case_rows,
        robustness_rows=robustness_rows,
        benchmark_context=benchmark_context,
        evidence_context=evidence_context,
        warnings=warnings,
    )
    report_path = out / "report.md"
    report_path.write_text(report, encoding="utf-8")
    manifest["outputs"] = registry.list_outputs()
    manifest["outputs"].append(
        {
            "path": "report.md",
            "kind": "report",
            "status": "generated",
            "size_bytes": int(report_path.stat().st_size),
        }
    )
    manifest_path = out / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["outputs"].append(
        {
            "path": "analysis_manifest.json",
            "kind": "manifest",
            "status": "generated",
            "size_bytes": int(manifest_path.stat().st_size),
        }
    )
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output_dir": str(out),
        "manifest": str(manifest_path),
        "report": str(report_path),
        "models": sorted(analyses),
        "warnings": list(warnings),
        "dry_run": bool(dry_run),
    }


def sample_metrics_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    metadata: list[dict[str, Any]] | None = None,
    sample_ids: list[str] | None = None,
    model_name: str = "model",
    split: str = "test",
    num_beams: int | None = None,
    dba_delta: float = 5.0,
    distance_mode: str = "circular",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.ndim != 2:
        raise ValueError(f"logits must have shape [N, C], got {scores.shape}.")
    if target.shape[0] != scores.shape[0]:
        raise ValueError("labels and logits must have the same sample count.")
    beams = int(num_beams or scores.shape[-1])
    probs = _softmax_np(scores)
    order = np.argsort(-scores, axis=1)
    meta_rows = metadata or [{} for _ in range(scores.shape[0])]
    rows: list[dict[str, Any]] = []
    valid_dba: list[float] = []
    top_hits = {1: [], 3: [], 5: [], 10: []}
    top1_errors: list[float] = []
    top3_min_distances: list[float] = []
    for index, truth in enumerate(target):
        row_meta = meta_rows[index] if index < len(meta_rows) and isinstance(meta_rows[index], dict) else {}
        sample_id = (
            str(sample_ids[index])
            if sample_ids is not None and index < len(sample_ids)
            else _sample_id_from_metadata(row_meta, index=index, target=int(truth), split=split)
        )
        sorted_idx = order[index]
        top1 = int(sorted_idx[0])
        top3 = [int(item) for item in sorted_idx[: min(3, beams)].tolist()]
        top5 = [int(item) for item in sorted_idx[: min(5, beams)].tolist()]
        top10 = [int(item) for item in sorted_idx[: min(10, beams)].tolist()]
        rank_positions = np.where(sorted_idx == int(truth))[0]
        target_rank = int(rank_positions[0] + 1) if rank_positions.size else int(beams + 1)
        top1_error = _class_distance(np.asarray([top1]), int(truth), beams, distance_mode)[0]
        top3_min = _min_distance(top3, int(truth), beams, distance_mode)
        top5_min = _min_distance(top5, int(truth), beams, distance_mode)
        top10_min = _min_distance(top10, int(truth), beams, distance_mode)
        dba_contribution = _sample_dba_contribution(top3, int(truth), beams, dba_delta, distance_mode)
        entropy = _entropy(probs[index])
        top2 = sorted_idx[:2]
        margin = float(probs[index, top2[0]] - probs[index, top2[1]]) if top2.size >= 2 else 0.0
        gt_probability = float(probs[index, int(truth)]) if 0 <= int(truth) < probs.shape[1] else 0.0
        scene = _metadata_value(row_meta, ("scene", "scene_id", "scene_slug", "town", "scenario"), default="")
        source_csv = _metadata_value(row_meta, ("csv_path", "source_csv", "root_csv"), default="")
        global_index = _metadata_value(row_meta, ("global_index", "index", "row_index", "sample_index"), default=index)
        condition = _metadata_value(row_meta, ("condition", "difficulty", "suite"), default="")
        scene_group = _metadata_value(row_meta, ("scene_group", "scene_set", "group"), default="")
        image_path = _metadata_value(row_meta, ("image_path", "image_file", "frame_path", "rgb_path"), default="")
        row = {
            "model": model_name,
            "sample_id": sample_id,
            "scene": scene,
            "scene_group": scene_group,
            "condition": condition,
            "split": split,
            "source_csv": source_csv,
            "image_path": image_path,
            "global_index": global_index,
            "target": int(truth),
            "top1": top1,
            "top3": json.dumps(top3),
            "top5": json.dumps(top5),
            "top10": json.dumps(top10),
            "target_rank": target_rank,
            "top1_error": float(top1_error),
            "top3_min_distance": float(top3_min),
            "top5_min_distance": float(top5_min),
            "top10_min_distance": float(top10_min),
            "top10_hit": int(int(truth) in top10),
            "dba_contribution": float(dba_contribution),
            "entropy": float(entropy),
            "margin": float(margin),
            "top1_top2_margin": float(margin),
            "gt_probability": float(gt_probability),
            "error_bucket": _error_bucket(float(top1_error), float(top3_min)),
        }
        rows.append(row)
        valid_dba.append(float(dba_contribution))
        top1_errors.append(float(top1_error))
        top3_min_distances.append(float(top3_min))
        for k, hits in top_hits.items():
            hits.append(int(int(truth) in sorted_idx[: min(k, beams)]))
    summary = {
        "model": model_name,
        "sample_count": int(len(rows)),
        "num_beams": int(beams),
        "distance_mode": str(distance_mode),
        "dba_delta": float(dba_delta),
        "dba": float(np.mean(valid_dba)) if valid_dba else 0.0,
        "top1": float(np.mean(top_hits[1])) if top_hits[1] else 0.0,
        "top3": float(np.mean(top_hits[3])) if top_hits[3] else 0.0,
        "top5": float(np.mean(top_hits[5])) if top_hits[5] else 0.0,
        "top10": float(np.mean(top_hits[10])) if top_hits[10] else 0.0,
        "mean_top1_error": float(np.mean(top1_errors)) if top1_errors else 0.0,
        "median_top1_error": float(np.median(top1_errors)) if top1_errors else 0.0,
        "mean_top3_min_distance": float(np.mean(top3_min_distances)) if top3_min_distances else 0.0,
    }
    return rows, summary


def sanitize_metadata(value: Any) -> Any:
    if value is None:
        return ""
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return value
    if isinstance(value, Mapping):
        return {str(key): sanitize_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(sanitize_metadata(item) for item in value)
    if isinstance(value, list):
        return [sanitize_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def safe_metadata_collate(batch: list[Any]) -> Any:
    return default_collate([sanitize_metadata(item) for item in batch])


def _default_analysis_config() -> dict[str, Any]:
    return {
        "models": {},
        "split": {
            "evaluation_split": "test",
            "scenes": None,
            "horizon_index": 0,
        },
        "sampling": {
            "seed": 42,
            "max_samples": None,
            "max_embedding_samples": 3000,
            "max_attention_cases": 256,
            "case_groups": list(DEFAULT_CASE_GROUPS),
            "cases_per_group": 3,
            "near_distance_threshold": 2,
            "far_distance_threshold": 5,
        },
        "figures": dict(DEFAULT_FIGURES),
        "embeddings": {
            "layers": ["output_features"],
            "method": "umap",
            "neighbors": 10,
        },
        "robustness": {
            "drop_modalities": True,
            "gps_noise": {"enabled": False, "std": []},
            "image_masking": {"enabled": False, "ratios": [], "mode": "random"},
            "seed": 42,
        },
        "benchmark": {
            "manifest": None,
            "runner_manifest": None,
            "metrics_by_condition": None,
            "robustness_summary": None,
            "case_studies": ["jepa_recovery", "gps_shortcut_failure", "shared_failure"],
        },
        "outputs": {
            "output_dir": "outputs/visual_analysis/jepa",
            "formats": list(DEFAULT_OUTPUT_FORMATS),
            "dpi": 180,
        },
        "evidence": deepcopy(DEFAULT_GPS_QUERY_EVIDENCE_CONFIG),
    }


def _validate_analysis_config(cfg: dict[str, Any], *, path: Path) -> None:
    models = cfg.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"models must be a mapping in {path}.")
    for name, spec in models.items():
        if not isinstance(spec, dict):
            raise ValueError(f"models.{name} must be a mapping.")
    for section in ("split", "sampling", "figures", "robustness", "benchmark", "outputs", "evidence"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"{section} must be a mapping in {path}.")
    formats = cfg.get("outputs", {}).get("formats", DEFAULT_OUTPUT_FORMATS)
    if isinstance(formats, str):
        formats = [formats]
    if "png" not in {str(item).lower() for item in formats}:
        cfg.setdefault("outputs", {})["formats"] = ["png", *list(formats)]


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"Analysis output directory is not empty. Use --force to write into it: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _analyze_model(
    model_name: str,
    model_spec: Mapping[str, Any],
    suite_cfg: dict[str, Any],
    *,
    tables_dir: Path,
    cache_dir: Path,
    warnings: list[str],
) -> ModelAnalysis:
    cache_path = model_spec.get("logits_cache") or model_spec.get("cache")
    if cache_path:
        return _load_cached_model_analysis(
            model_name,
            Path(str(cache_path)),
            model_spec,
            suite_cfg,
            tables_dir=tables_dir,
            cache_dir=cache_dir,
            warnings=warnings,
        )
    model_cfg = _load_model_config(model_spec, suite_cfg)
    return _run_model_forward_analysis(
        model_name,
        model_spec,
        model_cfg,
        suite_cfg,
        tables_dir=tables_dir,
        cache_dir=cache_dir,
        warnings=warnings,
    )


def _load_cached_model_analysis(
    model_name: str,
    cache_path: Path,
    model_spec: Mapping[str, Any],
    suite_cfg: dict[str, Any],
    *,
    tables_dir: Path,
    cache_dir: Path,
    warnings: list[str],
) -> ModelAnalysis:
    if not cache_path.exists():
        raise FileNotFoundError(f"Cached logits file not found for {model_name}: {cache_path}")
    payload = np.load(cache_path, allow_pickle=True)
    logits = np.asarray(payload["logits"], dtype=np.float64)
    if logits.ndim == 3:
        horizon_index = int(suite_cfg.get("split", {}).get("horizon_index", 0))
        logits = logits[:, max(0, min(horizon_index, logits.shape[1] - 1)), :]
    labels = np.asarray(payload["labels"], dtype=np.int64)
    if labels.ndim == 2:
        horizon_index = int(suite_cfg.get("split", {}).get("horizon_index", 0))
        labels = labels[:, max(0, min(horizon_index, labels.shape[1] - 1))]
    sample_ids = _string_array_from_npz(payload, "sample_ids")
    metadata_rows = _metadata_from_cached_payload(payload, sample_count=int(logits.shape[0]))
    rows, summary = sample_metrics_from_logits(
        logits,
        labels,
        metadata=metadata_rows,
        sample_ids=sample_ids,
        model_name=model_name,
        split=str(suite_cfg.get("split", {}).get("evaluation_split", "test")),
        num_beams=int(model_spec.get("num_beams", logits.shape[-1])),
        dba_delta=float(suite_cfg.get("evaluation", {}).get("dba_delta", model_spec.get("dba_delta", 5))),
        distance_mode=str(model_spec.get("distance_mode", suite_cfg.get("evaluation", {}).get("dba_distance_mode", "circular"))),
    )
    probabilities = _softmax_np(logits)
    _write_sample_outputs(model_name, rows, summary, logits, probabilities, labels, tables_dir, cache_dir)
    embeddings = {}
    if "embedding" in payload:
        embeddings["cache"] = np.asarray(payload["embedding"], dtype=np.float32)
    elif "embeddings" in payload:
        embeddings["cache"] = np.asarray(payload["embeddings"], dtype=np.float32)
    _write_model_embeddings_cache(model_name, embeddings, rows, cache_dir)
    attention_rows: list[dict[str, Any]] = []
    attention_maps: dict[str, np.ndarray] = {}
    if "attention" in payload:
        attention = np.asarray(payload["attention"], dtype=np.float32)
        attention_rows, attention_maps = _attention_diagnostics_from_array(
            model_name,
            attention,
            rows,
            max_maps=int(suite_cfg.get("sampling", {}).get("max_attention_cases", 256) or 256),
            token_grid=_token_grid_from_payload(payload),
        )
    elif bool(suite_cfg.get("figures", {}).get("attention", True)):
        warnings.append(f"attention_unavailable:{model_name}:cached_payload_missing_attention")
    return ModelAnalysis(
        name=model_name,
        sample_rows=rows,
        summary=summary,
        logits=logits,
        probabilities=probabilities,
        labels=labels.reshape(-1),
        sample_ids=[str(row["sample_id"]) for row in rows],
        metadata_rows=metadata_rows,
        split_metadata={"source": "cached_logits", "path": str(cache_path), "num_samples": int(logits.shape[0])},
        checkpoint_load=None,
        embeddings=embeddings,
        attention_rows=attention_rows,
        attention_maps=attention_maps,
        robustness_rows=[],
    )


def _run_model_forward_analysis(
    model_name: str,
    model_spec: Mapping[str, Any],
    model_cfg: dict[str, Any],
    suite_cfg: dict[str, Any],
    *,
    tables_dir: Path,
    cache_dir: Path,
    warnings: list[str],
) -> ModelAnalysis:
    set_seed(int(suite_cfg.get("sampling", {}).get("seed", model_cfg.get("experiment", {}).get("seed", 0))))
    split_name = str(suite_cfg.get("split", {}).get("evaluation_split", "test"))
    weights = model_spec.get("weights") or model_cfg.get("evaluation", {}).get("weights")
    if not weights:
        raise FileNotFoundError(f"models.{model_name}.weights is required unless logits_cache is provided.")
    weights_path = Path(str(weights))
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for {model_name}: {weights_path}")
    checkpoint_metadata = load_checkpoint_metadata(weights_path)
    dataset, split_metadata = _build_analysis_dataset(model_cfg, split_name, checkpoint_metadata, suite_cfg)
    max_samples = suite_cfg.get("sampling", {}).get("max_samples")
    if max_samples is not None:
        dataset = _deterministic_subset(dataset, max_samples=int(max_samples), seed=int(suite_cfg.get("sampling", {}).get("seed", 42)))
        split_metadata["sample_cap"] = int(max_samples)
    dataloader = _build_analysis_dataloader(dataset, model_cfg, split=split_name)
    device = build_device(model_cfg)
    model = build_model(model_cfg["model"]["primary"]).to(device)
    strict = bool(model_spec.get("strict_load", model_cfg.get("checkpoint", {}).get("strict_load", True)))
    load_result = load_model_state(weights_path, model, role=f"jepa_visual_analysis:{model_name}", map_location=device, strict=strict)
    checkpoint_load = checkpoint_load_summary(load_result)
    _apply_csi_rms_to_model_config(model_cfg, dataset)
    result = _collect_forward_outputs(
        model_name,
        model,
        dataloader,
        model_cfg,
        suite_cfg,
        device,
        collect_embeddings=True,
        collect_attention=bool(suite_cfg.get("figures", {}).get("attention", True)),
        warnings=warnings,
    )
    shutdown_dataloader_workers(dataloader)
    logits = result["logits"]
    labels = result["labels"]
    metadata_rows = result["metadata_rows"]
    rows, summary = sample_metrics_from_logits(
        logits,
        labels,
        metadata=metadata_rows,
        model_name=model_name,
        split=split_name,
        num_beams=int(model_cfg.get("model", {}).get("num_classes", logits.shape[-1])),
        dba_delta=float(model_cfg.get("evaluation", {}).get("dba_delta", 5)),
        distance_mode=str(model_cfg.get("evaluation", {}).get("dba_distance_mode", "circular")),
    )
    probabilities = _softmax_np(logits)
    _write_sample_outputs(model_name, rows, summary, logits, probabilities, labels, tables_dir, cache_dir)
    _write_model_embeddings_cache(model_name, result["embeddings"], rows, cache_dir)
    robustness_rows = []
    if bool(suite_cfg.get("figures", {}).get("robustness", True)):
        robustness_rows = _collect_model_robustness_rows(
            model_name,
            model,
            dataloader,
            model_cfg,
            suite_cfg,
            device,
            clean_summary=summary,
            warnings=warnings,
        )
    return ModelAnalysis(
        name=model_name,
        sample_rows=rows,
        summary=summary,
        logits=logits,
        probabilities=probabilities,
        labels=labels.reshape(-1),
        sample_ids=[str(row["sample_id"]) for row in rows],
        metadata_rows=metadata_rows,
        split_metadata=split_metadata,
        checkpoint_load=checkpoint_load,
        embeddings=result["embeddings"],
        attention_rows=result["attention_rows"],
        attention_maps=result["attention_maps"],
        robustness_rows=robustness_rows,
    )


def _load_model_config(model_spec: Mapping[str, Any], suite_cfg: Mapping[str, Any]) -> dict[str, Any]:
    config_path = model_spec.get("config")
    if not config_path:
        raise FileNotFoundError("models.<name>.config is required unless logits_cache is provided.")
    overrides = list(model_spec.get("overrides") or [])
    cfg = load_config(config_path, overrides=overrides)
    cfg = deepcopy(cfg)
    split_cfg = suite_cfg.get("split", {}) if isinstance(suite_cfg.get("split"), Mapping) else {}
    scenes = split_cfg.get("scenes")
    if scenes:
        dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
        dataset_cfg["test_scenes"] = list(scenes) if isinstance(scenes, (list, tuple)) else [scenes]
        dataset_cfg.setdefault("eval_scenes", dataset_cfg["test_scenes"])
    batch_size = suite_cfg.get("sampling", {}).get("batch_size") if isinstance(suite_cfg.get("sampling"), Mapping) else None
    if batch_size is not None:
        cfg.setdefault("data", {}).setdefault("dataloader", {}).setdefault("test", {})["batch_size"] = int(batch_size)
    return cfg


def _build_analysis_dataset(
    cfg: dict[str, Any],
    split_name: str,
    checkpoint_metadata: dict[str, Any] | None,
    suite_cfg: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    validate_normalization_artifact_fingerprint(cfg, checkpoint_metadata)
    dataset_kwargs = load_normalization_artifacts(checkpoint_metadata)
    split_metadata: dict[str, Any] = {}
    needs_train_gps = config_uses_gps(cfg) and "gps_scaler" not in dataset_kwargs
    needs_train_lidar = config_uses_lidar(cfg) and "lidar_normalizer" not in dataset_kwargs
    needs_train_mmwave = config_uses_mmwave(cfg) and "mmwave_scaler" not in dataset_kwargs
    needs_train_csi = config_uses_csi(cfg) and "csi_rms_normalizer" not in dataset_kwargs
    protocol_splits = build_protocol_split_datasets(cfg, **dataset_kwargs)
    if protocol_splits is not None:
        train_dataset = protocol_splits["train"]
        prepare_lidar_normalizer(cfg, train_dataset)
        split_metadata["train"] = dataset_run_metadata(train_dataset)
        key = "validation" if split_name in {"val", "validation"} else split_name
        if key not in protocol_splits:
            raise ValueError(f"Split protocol did not produce split '{split_name}'.")
        dataset = protocol_splits[key]
        if needs_train_gps:
            dataset_kwargs["gps_scaler"] = _dataset_attr_recursive(train_dataset, "gps_scaler")
        if needs_train_lidar:
            dataset_kwargs["lidar_normalizer"] = _dataset_attr_recursive(train_dataset, "lidar_normalizer")
        if needs_train_mmwave:
            dataset_kwargs["mmwave_scaler"] = _dataset_attr_recursive(train_dataset, "mmwave_scaler")
        if needs_train_csi:
            dataset_kwargs["csi_rms_normalizer"] = _dataset_attr_recursive(train_dataset, "csi_rms_normalizer")
    elif any((needs_train_gps, needs_train_lidar, needs_train_mmwave, needs_train_csi)):
        train_dataset = build_split_dataset(cfg, "train")
        prepare_lidar_normalizer(cfg, train_dataset)
        split_metadata["train"] = dataset_run_metadata(train_dataset)
        for attr, key in (
            ("gps_scaler", "gps_scaler"),
            ("lidar_normalizer", "lidar_normalizer"),
            ("mmwave_scaler", "mmwave_scaler"),
            ("csi_rms_normalizer", "csi_rms_normalizer"),
        ):
            value = _dataset_attr_recursive(train_dataset, attr)
            if value is not None:
                dataset_kwargs.setdefault(key, value)
        dataset = build_split_dataset(cfg, split_name, **dataset_kwargs)
    else:
        dataset = build_split_dataset(cfg, split_name, **dataset_kwargs)
    split_metadata[split_name] = dataset_run_metadata(dataset)
    split_metadata["analysis_split"] = {
        "evaluation_split": split_name,
        "scenes": suite_cfg.get("split", {}).get("scenes") if isinstance(suite_cfg.get("split"), Mapping) else None,
        "seq_len": cfg.get("data", {}).get("dataset", {}).get("seq_len"),
        "num_pred": cfg.get("data", {}).get("dataset", {}).get("num_pred"),
        "split_protocol": cfg.get("data", {}).get("dataset", {}).get("split_protocol"),
        "split_seed": cfg.get("data", {}).get("dataset", {}).get("split_seed", cfg.get("experiment", {}).get("seed")),
    }
    return dataset, split_metadata


def _build_analysis_dataloader(dataset: Any, cfg: dict[str, Any], *, split: str) -> DataLoader:
    kwargs = build_dataloader_kwargs(cfg["data"]["dataloader"], split=split)
    kwargs["shuffle"] = False
    return DataLoader(dataset, collate_fn=safe_metadata_collate, **kwargs)


def _collect_forward_outputs(
    model_name: str,
    model: torch.nn.Module,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    suite_cfg: Mapping[str, Any],
    device: torch.device,
    *,
    collect_embeddings: bool,
    collect_attention: bool,
    warnings: list[str],
    perturbation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task = cfg.get("experiment", {}).get("task", "image")
    model_cfg = cfg.get("model", {})
    primary_cfg = model_cfg.get("primary", {})
    num_pred = int(model_cfg.get("num_pred", cfg.get("data", {}).get("dataset", {}).get("num_pred", 1)))
    downsample_ratio = int(model_cfg.get("downsample_ratio", 1))
    seq_length = int(model_cfg.get("seq_length", cfg.get("data", {}).get("dataset", {}).get("seq_len", 1)))
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    horizon_index = _analysis_horizon_index(cfg, suite_cfg, num_pred=num_pred)
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    embeddings: dict[str, list[np.ndarray]] = {}
    attention_rows: list[dict[str, Any]] = []
    attention_maps: dict[str, np.ndarray] = {}
    layers = _embedding_layers(suite_cfg) if collect_embeddings else []
    with _temporary_attention_diagnostics(model, collect_attention):
        with _forward_hooks(model, layers, warnings=warnings, model_name=model_name) as hook_state:
            model.eval()
            with torch.no_grad():
                for step_index, raw_batch in enumerate(dataloader):
                    batch = _apply_perturbation_to_batch(
                        raw_batch,
                        perturbation,
                        seed=int(suite_cfg.get("robustness", {}).get("seed", suite_cfg.get("sampling", {}).get("seed", 42))),
                        step_index=step_index,
                    )
                    batch_metadata_rows = _metadata_rows_for_batch(batch.get("metadata"), batch_size=_batch_size_from_batch(batch))
                    sample_offset = len(metadata_rows)
                    metadata_rows.extend(batch_metadata_rows)
                    force_mask = _force_modality_mask(primary_cfg, perturbation, device=device)
                    with autocast_context(amp_enabled, device, amp_dtype):
                        step = run_model_step(
                            model,
                            task,
                            batch,
                            model_cfg=primary_cfg,
                            seq_length=seq_length,
                            num_pred=num_pred,
                            downsample_ratio=downsample_ratio,
                            device=device,
                            non_blocking=non_blocking,
                            force_modality_mask=force_mask,
                        )
                    logits_h = step.logits.detach().cpu()[:, horizon_index, :]
                    labels_h = step.labels.detach().cpu()[:, horizon_index] if step.labels is not None else torch.zeros(logits_h.shape[0], dtype=torch.long)
                    all_logits.append(logits_h.numpy())
                    all_labels.append(labels_h.numpy())
                    if collect_embeddings:
                        batch_embeddings = _batch_embeddings_from_step(
                            step.model_output,
                            hook_state,
                            layers,
                            batch_size=int(logits_h.shape[0]),
                            horizon_index=horizon_index,
                            warnings=warnings,
                            model_name=model_name,
                        )
                        for layer_name, values in batch_embeddings.items():
                            embeddings.setdefault(layer_name, []).append(values)
                    if collect_attention:
                        batch_attention = _attention_from_model(model)
                        if batch_attention is not None:
                            rows_stub = [
                                {
                                    "sample_id": _sample_id_from_metadata(
                                        batch_metadata_rows[idx] if idx < len(batch_metadata_rows) else {},
                                        index=sample_offset + idx,
                                        target=int(labels_h[idx]),
                                        split="test",
                                    )
                                }
                                for idx in range(int(logits_h.shape[0]))
                            ]
                            batch_rows, batch_maps = _attention_diagnostics_from_array(
                                model_name,
                                batch_attention.detach().cpu().numpy(),
                                rows_stub,
                                max_maps=max(0, int(suite_cfg.get("sampling", {}).get("max_attention_cases", 256)) - len(attention_maps)),
                            )
                            attention_rows.extend(batch_rows)
                            attention_maps.update(batch_maps)
    if collect_attention and not attention_rows:
        warnings.append(f"attention_unavailable:{model_name}:no_attention_map")
    logits = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0), dtype=np.float32)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
    embedding_arrays = {layer: np.concatenate(chunks, axis=0) for layer, chunks in embeddings.items() if chunks}
    return {
        "logits": logits,
        "labels": labels,
        "metadata_rows": metadata_rows,
        "embeddings": embedding_arrays,
        "attention_rows": attention_rows,
        "attention_maps": attention_maps,
    }


def _analysis_horizon_index(cfg: Mapping[str, Any], suite_cfg: Mapping[str, Any], *, num_pred: int) -> int:
    split_cfg = suite_cfg.get("split", {}) if isinstance(suite_cfg.get("split"), Mapping) else {}
    raw = split_cfg.get("horizon_index", split_cfg.get("horizon", 0))
    if isinstance(raw, str) and raw.lower() in {"last", "final"}:
        return max(0, int(num_pred) - 1)
    index = int(raw)
    if index < 0:
        index = int(num_pred) + index
    return max(0, min(index, max(int(num_pred) - 1, 0)))


def _write_sample_outputs(
    model_name: str,
    rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
    logits: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    tables_dir: Path,
    cache_dir: Path,
) -> None:
    _write_csv(tables_dir / f"sample_predictions_{model_name}.csv", rows)
    (tables_dir / f"metrics_summary_{model_name}.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        cache_dir / f"logits_{model_name}.npz",
        logits=np.asarray(logits),
        probabilities=np.asarray(probabilities),
        labels=np.asarray(labels),
        sample_ids=np.asarray([row["sample_id"] for row in rows], dtype=object),
    )


def _write_model_embeddings_cache(
    model_name: str,
    embeddings: Mapping[str, np.ndarray],
    rows: list[dict[str, Any]],
    cache_dir: Path,
) -> None:
    if not embeddings:
        return
    chunks = []
    layer_names: list[str] = []
    sample_rows: list[dict[str, Any]] = []
    for layer, values in embeddings.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue
        count = min(arr.shape[0], len(rows))
        chunks.append(arr[:count])
        layer_names.extend([str(layer)] * count)
        sample_rows.extend(rows[:count])
    if not chunks:
        return
    matrix = _pad_and_stack(chunks)
    np.savez_compressed(
        cache_dir / f"embeddings_{model_name}.npz",
        embedding=matrix,
        sample_id=np.asarray([row["sample_id"] for row in sample_rows], dtype=object),
        target=np.asarray([row["target"] for row in sample_rows], dtype=np.int64),
        scene=np.asarray([row.get("scene", "") for row in sample_rows], dtype=object),
        error_bucket=np.asarray([row.get("error_bucket", "") for row in sample_rows], dtype=object),
        layer_name=np.asarray(layer_names, dtype=object),
        model=np.asarray([model_name for _ in sample_rows], dtype=object),
    )


def _write_model_metrics(path: Path, analyses: Mapping[str, ModelAnalysis]) -> None:
    rows = [analysis.summary for analysis in analyses.values()]
    _write_csv(path, rows)


def _write_comparison_outputs(
    tables_dir: Path,
    analyses: Mapping[str, ModelAnalysis],
    cfg: Mapping[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if len(analyses) < 2:
        warnings.append("comparison_skipped:requires_at_least_two_models")
        return []
    names = list(analyses)
    query_model = _query_model_name(names, cfg)
    baseline_model = _baseline_model_name(names, query_model, cfg)
    joined: dict[str, dict[str, Any]] = {}
    for name, analysis in analyses.items():
        for row in analysis.sample_rows:
            sample_id = str(row["sample_id"])
            out = joined.setdefault(sample_id, {"sample_id": sample_id})
            for key in ("scene", "scene_group", "condition", "split", "source_csv", "image_path", "global_index", "target"):
                out.setdefault(key, row.get(key))
            for key, value in row.items():
                if key in {"sample_id", "scene", "scene_group", "condition", "split", "source_csv", "image_path", "global_index", "target", "model"}:
                    continue
                out[f"{name}_{key}"] = value
    near_threshold = float(cfg.get("sampling", {}).get("near_distance_threshold", 2))
    far_threshold = float(cfg.get("sampling", {}).get("far_distance_threshold", 5))
    rows = []
    for row in joined.values():
        q_top3 = _float_or_nan(row.get(f"{query_model}_top3_min_distance"))
        b_top3 = _float_or_nan(row.get(f"{baseline_model}_top3_min_distance"))
        q_top1 = _float_or_nan(row.get(f"{query_model}_top1_error"))
        b_top1 = _float_or_nan(row.get(f"{baseline_model}_top1_error"))
        q_rank = _float_or_nan(row.get(f"{query_model}_target_rank"))
        b_rank = _float_or_nan(row.get(f"{baseline_model}_target_rank"))
        q_dba = _float_or_nan(row.get(f"{query_model}_dba_contribution"))
        b_dba = _float_or_nan(row.get(f"{baseline_model}_dba_contribution"))
        all_top3 = [
            _float_or_nan(row.get(f"{name}_top3_min_distance"))
            for name in analyses
            if f"{name}_top3_min_distance" in row
        ]
        row["query_model"] = query_model
        row["baseline_model"] = baseline_model
        row["query_gain"] = int(b_top3 > far_threshold and q_top3 <= near_threshold)
        row["query_regression"] = int(b_top3 <= near_threshold and q_top3 > far_threshold)
        row["shared_near_miss"] = int(
            b_top1 > 0
            and q_top1 > 0
            and q_top3 <= far_threshold
            and (q_rank < b_rank or q_dba > b_dba)
        )
        row["far_error"] = int(bool(all_top3) and all(value > far_threshold for value in all_top3 if not math.isnan(value)))
        row["shared_failure"] = row["far_error"]
        rows.append(row)
    rows.sort(key=lambda item: str(item["sample_id"]))
    _write_csv(tables_dir / "comparison_samples.csv", rows)
    return rows


def _write_error_anatomy_outputs(
    figures_dir: Path,
    tables_dir: Path,
    analyses: Mapping[str, ModelAnalysis],
    cfg: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[str],
) -> None:
    if not _matplotlib_available(warnings):
        registry.skipped_output(figures_dir / "error_anatomy.png", reason="matplotlib_unavailable", kind="figure")
        return
    import matplotlib.pyplot as plt

    formats = _output_formats(cfg)
    per_target_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for name, analysis in analyses.items():
        rows = analysis.sample_rows
        if not rows:
            continue
        top1 = np.asarray([float(row["top1_error"]) for row in rows], dtype=np.float64)
        top3 = np.asarray([float(row["top3_min_distance"]) for row in rows], dtype=np.float64)
        ranks = np.asarray([min(int(row["target_rank"]), 11) for row in rows], dtype=np.int64)
        dba = np.asarray([float(row["dba_contribution"]) for row in rows], dtype=np.float64)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].hist(top1, bins=np.arange(0, max(2, int(top1.max()) + 2)) - 0.5, color="#3366AA", edgecolor="white")
        axes[0, 0].set_title(f"{name} Top-1 error, n={len(rows)}")
        axes[0, 0].set_xlabel("Beam distance")
        axes[0, 0].set_ylabel("Samples")
        axes[0, 1].hist(top3, bins=np.arange(0, max(2, int(top3.max()) + 2)) - 0.5, color="#CC6677", edgecolor="white")
        axes[0, 1].set_title(f"Top-3 min-distance, DBA={analysis.summary.get('dba', 0):.3f}")
        axes[0, 1].set_xlabel(f"Distance mode: {analysis.summary.get('distance_mode', 'circular')}")
        axes[0, 1].set_ylabel("Samples")
        rank_bins = np.arange(1, 13)
        axes[1, 0].hist(ranks, bins=rank_bins - 0.5, color="#228833", edgecolor="white")
        axes[1, 0].set_title("Target rank in Top-10")
        axes[1, 0].set_xlabel("Rank, 11 means >10")
        axes[1, 0].set_ylabel("Samples")
        axes[1, 1].boxplot(dba)
        axes[1, 1].set_title("Sample DBA contribution")
        axes[1, 1].set_ylabel(f"Delta={analysis.summary.get('dba_delta', 5)}")
        fig.suptitle(_figure_title(name, cfg, rows, extra="error anatomy"))
        fig.tight_layout()
        _save_figure(fig, figures_dir / f"error_anatomy_{name}", formats, dpi=int(cfg.get("outputs", {}).get("dpi", 180)))
        plt.close(fig)

        confusion = _confusion_matrix(rows, int(analysis.summary.get("num_beams", analysis.logits.shape[-1])))
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        im = axes[0].imshow(confusion, aspect="auto", cmap="viridis")
        axes[0].set_title(f"{name} beam confusion")
        axes[0].set_xlabel("Top-1 beam")
        axes[0].set_ylabel("Target beam")
        fig.colorbar(im, ax=axes[0], fraction=0.046)
        residuals = np.asarray([int(row["target"]) - int(row["top1"]) for row in rows], dtype=np.int64)
        axes[1].hist(residuals, bins=31, color="#AA4499", edgecolor="white")
        axes[1].set_title("Residual heatmap proxy")
        axes[1].set_xlabel("target - top1")
        targets = sorted({int(row["target"]) for row in rows})
        supports = []
        errors = []
        dbas = []
        for target in targets:
            target_rows = [row for row in rows if int(row["target"]) == target]
            supports.append(len(target_rows))
            errors.append(float(np.mean([float(row["top1_error"]) for row in target_rows])) if target_rows else 0.0)
            dbas.append(float(np.mean([float(row["dba_contribution"]) for row in target_rows])) if target_rows else 0.0)
            per_target_rows.append(
                {
                    "model": name,
                    "target": target,
                    "support": len(target_rows),
                    "mean_top1_error": errors[-1],
                    "dba": dbas[-1],
                    "top1": float(np.mean([int(float(row["top1_error"]) == 0) for row in target_rows])) if target_rows else 0.0,
                    "top3": float(np.mean([int(float(row["top3_min_distance"]) == 0) for row in target_rows])) if target_rows else 0.0,
                    "far_error_rate": float(np.mean([int(float(row["top3_min_distance"]) > float(analysis.summary.get("dba_delta", 5))) for row in target_rows])) if target_rows else 0.0,
                }
            )
        axes[2].scatter(supports, errors, c=dbas, cmap="magma", edgecolor="black", linewidth=0.3)
        axes[2].set_title("Per-target support vs error")
        axes[2].set_xlabel("Support")
        axes[2].set_ylabel("Mean Top-1 error")
        fig.suptitle(_figure_title(name, cfg, rows, extra="beam-level errors"))
        fig.tight_layout()
        _save_figure(fig, figures_dir / f"beam_errors_{name}", formats, dpi=int(cfg.get("outputs", {}).get("dpi", 180)))
        plt.close(fig)
        for rank, count in zip(*np.unique(ranks, return_counts=True)):
            rank_rows.append({"model": name, "rank": int(rank), "count": int(count)})
    _write_csv(tables_dir / "per_target_error_summary.csv", per_target_rows)
    _write_csv(tables_dir / "target_rank_distribution.csv", rank_rows)


def _write_embedding_outputs(
    figures_dir: Path,
    tables_dir: Path,
    analyses: Mapping[str, ModelAnalysis],
    cfg: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[str],
) -> list[dict[str, Any]]:
    embedding_rows: list[dict[str, Any]] = []
    matrix_chunks = []
    labels = []
    rows_for_embedding: list[dict[str, Any]] = []
    for name, analysis in analyses.items():
        if not analysis.embeddings:
            warnings.append(f"embedding_unavailable:{name}:no_embedding_layers")
            continue
        for layer, values in analysis.embeddings.items():
            if values.size == 0:
                continue
            count = min(values.shape[0], len(analysis.sample_rows))
            matrix_chunks.append(np.asarray(values[:count], dtype=np.float32))
            labels.extend([layer] * count)
            for idx, sample_row in enumerate(analysis.sample_rows[:count]):
                rows_for_embedding.append(
                    {
                        "model": name,
                        "layer": layer,
                        "sample_id": sample_row["sample_id"],
                        "target": sample_row["target"],
                        "scene": sample_row.get("scene", ""),
                        "error_bucket": sample_row.get("error_bucket", ""),
                        "top1_error": sample_row.get("top1_error", 0.0),
                    }
                )
    if not matrix_chunks:
        registry.skipped_output(tables_dir / "embedding_neighbors.csv", reason="no_embeddings", kind="table")
        return []
    matrix = _pad_and_stack(matrix_chunks)
    max_samples = cfg.get("sampling", {}).get("max_embedding_samples")
    if max_samples is not None and matrix.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(cfg.get("sampling", {}).get("seed", 42)))
        keep = np.sort(rng.choice(matrix.shape[0], size=int(max_samples), replace=False))
        matrix = matrix[keep]
        rows_for_embedding = [rows_for_embedding[int(idx)] for idx in keep]
        labels = [labels[int(idx)] for idx in keep]
    embedding_neighbor_rows = _embedding_neighbor_rows(matrix, rows_for_embedding, cfg, warnings)
    _write_csv(tables_dir / "embedding_neighbors.csv", embedding_neighbor_rows)
    np.savez_compressed(
        Path(cfg.get("outputs", {}).get("output_dir", ".")) / "cache" / "embeddings_all_models.npz",
        embedding=matrix,
        sample_id=np.asarray([row["sample_id"] for row in rows_for_embedding], dtype=object),
        target=np.asarray([row["target"] for row in rows_for_embedding], dtype=np.int64),
        scene=np.asarray([row["scene"] for row in rows_for_embedding], dtype=object),
        error_bucket=np.asarray([row["error_bucket"] for row in rows_for_embedding], dtype=object),
        layer_name=np.asarray(labels, dtype=object),
        model=np.asarray([row["model"] for row in rows_for_embedding], dtype=object),
    )
    if not bool(cfg.get("figures", {}).get("embedding", True)):
        return embedding_neighbor_rows
    if not _matplotlib_available(warnings):
        registry.skipped_output(figures_dir / "embedding_target.png", reason="matplotlib_unavailable", kind="figure")
        return embedding_neighbor_rows
    coords, method = _reduce_embeddings(matrix, cfg, warnings)
    if coords.shape[0] == 0:
        registry.skipped_output(figures_dir / "embedding_target.png", reason="too_few_embeddings", kind="figure")
        return embedding_neighbor_rows
    for field in ("target", "scene", "error_bucket", "model"):
        _plot_embedding_projection(figures_dir, coords, rows_for_embedding, field, method, cfg, warnings)
    return embedding_neighbor_rows


def _write_attention_outputs(
    figures_dir: Path,
    tables_dir: Path,
    analyses: Mapping[str, ModelAnalysis],
    cfg: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[str],
) -> None:
    rows = [row for analysis in analyses.values() for row in analysis.attention_rows]
    if not rows:
        if bool(cfg.get("figures", {}).get("attention", True)):
            registry.skipped_output(tables_dir / "attention_summary.csv", reason="attention_unavailable", kind="table")
        return
    _write_csv(tables_dir / "attention_summary.csv", rows)
    if not bool(cfg.get("figures", {}).get("attention", True)):
        return
    if not _matplotlib_available(warnings):
        registry.skipped_output(figures_dir / "attention_summary.png", reason="matplotlib_unavailable", kind="figure")
        return
    import matplotlib.pyplot as plt

    formats = _output_formats(cfg)
    for name, analysis in analyses.items():
        if not analysis.attention_rows:
            continue
        attention_dir = figures_dir / "attention_cases"
        attention_dir.mkdir(parents=True, exist_ok=True)
        for sample_id, grid in list(analysis.attention_maps.items())[: int(cfg.get("sampling", {}).get("max_attention_cases", 256) or 256)]:
            fig, ax = plt.subplots(figsize=(4, 3.5))
            im = ax.imshow(grid, cmap="magma")
            ax.set_title(f"{name} attention overlay\nsample={sample_id}")
            ax.set_xlabel("patch x")
            ax.set_ylabel("patch y")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            _save_figure(
                fig,
                attention_dir / f"attention_{_safe_slug(name)}_{_safe_slug(sample_id)}",
                formats,
                dpi=int(cfg.get("outputs", {}).get("dpi", 180)),
            )
            plt.close(fig)
        entropy = np.asarray([float(row["attention_entropy"]) for row in analysis.attention_rows], dtype=np.float64)
        effective = np.asarray([float(row["effective_patch_count"]) for row in analysis.attention_rows], dtype=np.float64)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].hist(entropy, bins=20, color="#4477AA", edgecolor="white")
        axes[0].set_title(f"{name} query entropy")
        axes[0].set_xlabel("Entropy")
        axes[0].set_ylabel("Samples")
        axes[1].hist(effective, bins=20, color="#EE6677", edgecolor="white")
        axes[1].set_title("Effective patch count")
        axes[1].set_xlabel("exp(entropy)")
        fig.tight_layout()
        _save_figure(fig, figures_dir / f"attention_summary_{name}", formats, dpi=int(cfg.get("outputs", {}).get("dpi", 180)))
        plt.close(fig)


def _write_case_study_outputs(
    figures_dir: Path,
    payload_dir: Path,
    tables_dir: Path,
    comparison_rows: list[dict[str, Any]],
    analyses: Mapping[str, ModelAnalysis],
    cfg: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[str],
) -> list[dict[str, Any]]:
    if not comparison_rows:
        registry.skipped_output(tables_dir / "case_selection.csv", reason="comparison_unavailable", kind="table")
        return []
    seed = int(cfg.get("sampling", {}).get("seed", 42))
    per_group = int(cfg.get("sampling", {}).get("cases_per_group", 3))
    groups = tuple(str(item) for item in cfg.get("sampling", {}).get("case_groups", DEFAULT_CASE_GROUPS))
    selected = _select_case_rows(comparison_rows, groups=groups, per_group=per_group, seed=seed)
    _write_csv(tables_dir / "case_selection.csv", selected)
    if not selected:
        registry.skipped_output(figures_dir / "cases", reason="no_matching_cases", kind="figure")
        return []
    case_figures_dir = figures_dir / "cases"
    case_figures_dir.mkdir(parents=True, exist_ok=True)
    if not _matplotlib_available(warnings):
        registry.skipped_output(case_figures_dir / "case_panels.png", reason="matplotlib_unavailable", kind="figure")
    formats = _output_formats(cfg)
    for row in selected:
        payload = _case_payload(row, analyses)
        payload_path = payload_dir / f"{_safe_slug(row['case_group'])}_{_safe_slug(row['sample_id'])}.json"
        payload_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if _matplotlib_available(warnings):
            _plot_case_panel(case_figures_dir, row, payload, analyses, cfg, formats, warnings)
    return selected


def _collect_model_robustness_rows(
    model_name: str,
    model: torch.nn.Module,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    suite_cfg: Mapping[str, Any],
    device: torch.device,
    *,
    clean_summary: Mapping[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    robustness_cfg = suite_cfg.get("robustness", {}) if isinstance(suite_cfg.get("robustness"), Mapping) else {}
    perturbations: list[dict[str, Any]] = []
    if robustness_cfg.get("drop_modalities", True):
        perturbations.extend(
            [
                {"condition": "drop_image", "type": "drop_modality", "modality": "image", "severity": 1.0},
                {"condition": "drop_gps", "type": "drop_modality", "modality": "gps", "severity": 1.0},
            ]
        )
    gps_noise = robustness_cfg.get("gps_noise", {}) if isinstance(robustness_cfg.get("gps_noise"), Mapping) else {}
    if gps_noise.get("enabled", False):
        for std in gps_noise.get("std", gps_noise.get("strengths", [])) or []:
            perturbations.append({"condition": f"gps_noise_{std}", "type": "gps_noise", "std": float(std), "severity": float(std)})
    image_masking = robustness_cfg.get("image_masking", {}) if isinstance(robustness_cfg.get("image_masking"), Mapping) else {}
    if image_masking.get("enabled", False):
        for ratio in image_masking.get("ratios", image_masking.get("strengths", [])) or []:
            perturbations.append(
                {
                    "condition": f"image_mask_{ratio}",
                    "type": "image_masking",
                    "ratio": float(ratio),
                    "severity": float(ratio),
                    "mask_mode": image_masking.get("mode", "random"),
                }
            )
    rows: list[dict[str, Any]] = []
    for perturbation in perturbations:
        try:
            result = _collect_forward_outputs(
                model_name,
                model,
                dataloader,
                cfg,
                suite_cfg,
                device,
                collect_embeddings=False,
                collect_attention=False,
                warnings=warnings,
                perturbation=perturbation,
            )
            _, summary = sample_metrics_from_logits(
                result["logits"],
                result["labels"],
                metadata=result["metadata_rows"],
                model_name=model_name,
                split=str(suite_cfg.get("split", {}).get("evaluation_split", "test")),
                num_beams=int(cfg.get("model", {}).get("num_classes", result["logits"].shape[-1] if result["logits"].ndim == 2 else 1)),
                dba_delta=float(cfg.get("evaluation", {}).get("dba_delta", 5)),
                distance_mode=str(cfg.get("evaluation", {}).get("dba_distance_mode", "circular")),
            )
            clean_dba = float(clean_summary.get("dba", 0.0))
            rows.append(
                {
                    "model": model_name,
                    "condition": perturbation["condition"],
                    "perturbation_type": perturbation["type"],
                    "severity": float(perturbation.get("severity", 0.0)),
                    "seed": robustness_cfg.get("seed", suite_cfg.get("sampling", {}).get("seed", 42)),
                    "mask_mode": perturbation.get("mask_mode", perturbation.get("modality", "")),
                    "top1": summary.get("top1", 0.0),
                    "top3": summary.get("top3", 0.0),
                    "top5": summary.get("top5", 0.0),
                    "dba": summary.get("dba", 0.0),
                    "dba_drop": float(clean_dba - float(summary.get("dba", 0.0))),
                    "status": "generated",
                }
            )
        except Exception as exc:
            warnings.append(f"robustness_condition_failed:{model_name}:{perturbation['condition']}:{exc}")
            rows.append(
                {
                    "model": model_name,
                    "condition": perturbation["condition"],
                    "perturbation_type": perturbation["type"],
                    "severity": float(perturbation.get("severity", 0.0)),
                    "seed": robustness_cfg.get("seed", suite_cfg.get("sampling", {}).get("seed", 42)),
                    "mask_mode": perturbation.get("mask_mode", perturbation.get("modality", "")),
                    "top1": "",
                    "top3": "",
                    "top5": "",
                    "dba": "",
                    "dba_drop": "",
                    "status": "failed",
                }
            )
    return rows


def _write_robustness_outputs(
    figures_dir: Path,
    tables_dir: Path,
    analyses: Mapping[str, ModelAnalysis],
    cfg: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, analysis in analyses.items():
        clean = analysis.summary
        rows.append(
            {
                "model": name,
                "condition": "all",
                "perturbation_type": "none",
                "severity": 0.0,
                "seed": cfg.get("sampling", {}).get("seed", 42),
                "mask_mode": "",
                "top1": clean.get("top1", 0.0),
                "top3": clean.get("top3", 0.0),
                "top5": clean.get("top5", 0.0),
                "dba": clean.get("dba", 0.0),
                "dba_drop": 0.0,
                "status": "clean_reference",
            }
        )
        if analysis.robustness_rows:
            rows.extend(analysis.robustness_rows)
            continue
        if cfg.get("robustness", {}).get("drop_modalities", True):
            for modality in ("image", "gps"):
                rows.append(
                    {
                        "model": name,
                        "condition": f"drop_{modality}",
                        "perturbation_type": "drop_modality",
                        "severity": 1.0,
                        "seed": cfg.get("robustness", {}).get("seed", cfg.get("sampling", {}).get("seed", 42)),
                        "mask_mode": modality,
                        "top1": "",
                        "top3": "",
                        "top5": "",
                        "dba": "",
                        "dba_drop": "",
                        "status": "requires_model_forward",
                    }
                )
        gps_noise = cfg.get("robustness", {}).get("gps_noise", {}) if isinstance(cfg.get("robustness"), Mapping) else {}
        if isinstance(gps_noise, Mapping) and gps_noise.get("enabled", False):
            for std in gps_noise.get("std", gps_noise.get("strengths", [])) or []:
                rows.append(_robustness_skipped_row(name, "gps_noise", float(std), cfg))
        image_masking = cfg.get("robustness", {}).get("image_masking", {}) if isinstance(cfg.get("robustness"), Mapping) else {}
        if isinstance(image_masking, Mapping) and image_masking.get("enabled", False):
            for ratio in image_masking.get("ratios", image_masking.get("strengths", [])) or []:
                row = _robustness_skipped_row(name, "image_masking", float(ratio), cfg)
                row["mask_mode"] = image_masking.get("mode", "random")
                rows.append(row)
    _write_csv(tables_dir / "robustness_summary.csv", rows)
    if not rows:
        return rows
    if not _matplotlib_available(warnings):
        registry.skipped_output(figures_dir / "robustness_summary.png", reason="matplotlib_unavailable", kind="figure")
        return rows
    import matplotlib.pyplot as plt

    clean_rows = [row for row in rows if row["condition"] == "all"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([row["model"] for row in clean_rows], [float(row["dba"]) for row in clean_rows], color="#117733")
    ax.set_ylabel("DBA")
    ax.set_title("Clean DBA reference for robustness slices")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "robustness_summary", _output_formats(cfg), dpi=int(cfg.get("outputs", {}).get("dpi", 180)))
    plt.close(fig)
    return rows


def _robustness_skipped_row(model: str, condition: str, severity: float, cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "condition": condition,
        "perturbation_type": condition,
        "severity": float(severity),
        "seed": cfg.get("robustness", {}).get("seed", cfg.get("sampling", {}).get("seed", 42)),
        "mask_mode": "",
        "top1": "",
        "top3": "",
        "top5": "",
        "dba": "",
        "dba_drop": "",
        "status": "configured_but_not_rerun_in_cached_analysis",
    }


def _write_benchmark_analysis_outputs(
    cfg: Mapping[str, Any],
    tables_dir: Path,
    figures_dir: Path,
    payload_dir: Path,
    registry: OutputRegistry,
    warnings: list[str],
) -> dict[str, Any]:
    benchmark_cfg = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark"), Mapping) else {}
    manifest_path = benchmark_cfg.get("manifest") or benchmark_cfg.get("runner_manifest")
    metrics_path = benchmark_cfg.get("metrics_by_condition")
    robustness_path = benchmark_cfg.get("robustness_summary")
    if not any((manifest_path, metrics_path, robustness_path)):
        return {"enabled": False}
    context: dict[str, Any] = {
        "enabled": True,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "metrics_by_condition": str(metrics_path) if metrics_path else None,
        "robustness_summary": str(robustness_path) if robustness_path else None,
        "generated_figures": [],
        "skipped": [],
    }
    try:
        bundle = read_benchmark_analysis_bundle(
            manifest_path,
            metrics_by_condition=metrics_path,
            robustness_summary=robustness_path,
        )
    except Exception as exc:
        message = f"benchmark_ingestion_failed:{exc}"
        warnings.append(message)
        registry.skipped_output(tables_dir / "benchmark_robustness_matrix.csv", reason=message, kind="table")
        context["status"] = "failed"
        context["skipped"].append(message)
        return context
    warnings.extend(f"benchmark:{item}" for item in bundle.get("warnings", []))
    metrics_rows = list(bundle.get("metrics_rows", []))
    matrix_rows = list(bundle.get("matrix_rows", []))
    robustness_rows = list(bundle.get("robustness_rows", []))
    case_rows = list(bundle.get("case_rows", []))
    context.update(
        {
            "status": "generated" if metrics_rows else "no_metrics",
            "manifest_path": bundle.get("manifest_path"),
            "manifest_digest": bundle.get("manifest_digest"),
            "metrics_path": bundle.get("metrics_path"),
            "robustness_path": bundle.get("robustness_path"),
            "metrics_row_count": len(metrics_rows),
            "matrix_row_count": len(matrix_rows),
            "case_row_count": len(case_rows),
        }
    )
    if not metrics_rows:
        registry.skipped_output(tables_dir / "benchmark_metrics_by_condition.csv", reason="benchmark_metrics_unavailable", kind="table")
        registry.skipped_output(tables_dir / "benchmark_robustness_matrix.csv", reason="benchmark_metrics_unavailable", kind="table")
        context["skipped"].append("benchmark_metrics_unavailable")
        return context
    _write_csv(tables_dir / "benchmark_metrics_by_condition.csv", metrics_rows)
    _write_csv(tables_dir / "benchmark_robustness_matrix.csv", matrix_rows)
    _write_csv(tables_dir / "benchmark_robustness_summary.csv", robustness_rows)
    _write_csv(tables_dir / "benchmark_case_selection.csv", case_rows)
    for row in case_rows:
        payload_path = payload_dir / f"benchmark_{_safe_slug(row.get('case_group', 'case'))}_{_safe_slug(row.get('suite', 'suite'))}.json"
        payload_path.write_text(json.dumps(_json_ready(row), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not case_rows:
        registry.skipped_output(payload_dir / "benchmark_cases.json", reason="no_benchmark_case_matches", kind="case_payload")
        context["skipped"].append("no_benchmark_case_matches")
    if bool(cfg.get("figures", {}).get("robustness", True)):
        context["generated_figures"] = _write_benchmark_curve_outputs(figures_dir, matrix_rows, cfg, registry, warnings)
    return context


def _write_benchmark_curve_outputs(
    figures_dir: Path,
    matrix_rows: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[str],
) -> list[str]:
    if not matrix_rows:
        registry.skipped_output(figures_dir / "benchmark_gps_collapse_curve.png", reason="no_benchmark_matrix_rows", kind="figure")
        return []
    if not _matplotlib_available(warnings):
        registry.skipped_output(figures_dir / "benchmark_gps_collapse_curve.png", reason="matplotlib_unavailable", kind="figure")
        return []
    import matplotlib.pyplot as plt

    groups = {
        "benchmark_gps_collapse_curve": lambda row: str(row.get("suite_type")) in GPS_SUITE_TYPES,
        "benchmark_image_degradation_curve": lambda row: str(row.get("suite_type")) in IMAGE_SUITE_TYPES,
        "benchmark_temporal_delay_curve": lambda row: str(row.get("suite_type")) in TEMPORAL_SUITE_TYPES,
    }
    generated: list[str] = []
    for figure_name, predicate in groups.items():
        rows = [row for row in matrix_rows if predicate(row)]
        if not rows:
            registry.skipped_output(figures_dir / f"{figure_name}.png", reason="no_matching_benchmark_rows", kind="figure")
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        by_model: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_model.setdefault(str(row.get("model")), []).append(row)
        for model, model_rows in sorted(by_model.items()):
            model_rows.sort(key=lambda item: float(item.get("severity") or 0.0))
            ax.plot(
                [float(row.get("severity") or 0.0) for row in model_rows],
                [float(row.get("perturbed_metric") or 0.0) for row in model_rows],
                marker="o",
                label=model,
            )
        metric = rows[0].get("metric", "primary_metric")
        split = rows[0].get("split", "")
        sample_count = rows[0].get("sample_count", "")
        ax.set_title(f"{figure_name.replace('_', ' ')} | split={split} n={sample_count}")
        ax.set_xlabel("severity")
        ax.set_ylabel(str(metric))
        ax.legend(fontsize=7)
        fig.tight_layout()
        _save_figure(fig, figures_dir / figure_name, _output_formats(cfg), dpi=int(cfg.get("outputs", {}).get("dpi", 180)))
        plt.close(fig)
        generated.append(f"figures/{figure_name}.png")
    return generated


def _build_manifest(
    cfg: Mapping[str, Any],
    *,
    output_dir: Path,
    command: list[str] | None,
    analyses: Mapping[str, ModelAnalysis],
    warnings: list[str],
    model_failures: Mapping[str, str],
    dry_run: bool,
    registry: OutputRegistry,
    benchmark_context: Mapping[str, Any],
    evidence_context: Mapping[str, Any],
) -> dict[str, Any]:
    model_records = {}
    for name, spec in (cfg.get("models", {}) or {}).items():
        analysis = analyses.get(name)
        model_records[name] = {
            "config": spec.get("config") if isinstance(spec, Mapping) else None,
            "weights": spec.get("weights") if isinstance(spec, Mapping) else None,
            "logits_cache": spec.get("logits_cache") if isinstance(spec, Mapping) else None,
            "checkpoint_load": analysis.checkpoint_load if analysis is not None else None,
            "metrics": analysis.summary if analysis is not None else None,
            "failure": model_failures.get(name),
        }
    split_metadata = {
        name: analysis.split_metadata
        for name, analysis in analyses.items()
    }
    return {
        "version": ANALYSIS_VERSION,
        "dry_run": bool(dry_run),
        "command": list(command or []),
        "analysis_config_path": cfg.get("_analysis_config_path"),
        "analysis_config_digest": cfg.get("_analysis_config_digest"),
        "output_dir": str(output_dir),
        "seed": cfg.get("sampling", {}).get("seed"),
        "models": model_records,
        "split_metadata": split_metadata,
        "figures": cfg.get("figures", {}),
        "sampling": cfg.get("sampling", {}),
        "robustness": cfg.get("robustness", {}),
        "benchmark": benchmark_context,
        "evidence": evidence_context,
        "warnings": sorted(set(str(item) for item in warnings)),
        "outputs": registry.list_outputs(),
    }


def _build_report(
    cfg: Mapping[str, Any],
    *,
    analyses: Mapping[str, ModelAnalysis],
    comparison_rows: list[dict[str, Any]],
    embedding_neighbor_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    robustness_rows: list[dict[str, Any]],
    benchmark_context: Mapping[str, Any],
    evidence_context: Mapping[str, Any],
    warnings: list[str],
) -> str:
    lines = [
        "# JEPA Visual Analysis Report",
        "",
        "## 可报告结论",
    ]
    if analyses:
        for name, analysis in analyses.items():
            summary = analysis.summary
            lines.append(
                f"- `{name}`: DBA={float(summary.get('dba', 0.0)):.4f}, "
                f"Top-1={float(summary.get('top1', 0.0)):.4f}, Top-3={float(summary.get('top3', 0.0)):.4f}, "
                f"n={int(summary.get('sample_count', 0))}."
            )
    else:
        lines.append("- Dry-run 或所有模型均未完成推理；本报告只记录配置和计划产物。")
    gain = sum(int(row.get("query_gain", 0)) for row in comparison_rows)
    regression = sum(int(row.get("query_regression", 0)) for row in comparison_rows)
    far_error = sum(int(row.get("far_error", 0)) for row in comparison_rows)
    if comparison_rows:
        lines.extend(
            [
                "",
                "## 错误邻近性与样本对比",
                f"- comparison_samples.csv 覆盖 {len(comparison_rows)} 个 joined samples；query_gain={gain}, query_regression={regression}, far_error={far_error}。",
                "- 主要图表在 `figures/error_anatomy_<model>.png` 与 `figures/beam_errors_<model>.png`。",
            ]
        )
    if embedding_neighbor_rows:
        mean_dist = np.mean([float(row.get("mean_neighbor_beam_distance", 0.0)) for row in embedding_neighbor_rows])
        lines.extend(
            [
                "",
                "## 表征空间",
                f"- embedding_neighbors.csv 汇总 {len(embedding_neighbor_rows)} 个 model/layer 邻域统计；平均邻域 beam distance={mean_dist:.4f}。",
                "- 投影图仅用于可视化，定量解读以邻域一致性表为准。",
            ]
        )
    attention_warnings = [item for item in warnings if str(item).startswith("attention_unavailable")]
    lines.extend(
        [
            "",
            "## Attention 与 Case Study",
            f"- case_selection.csv 选择 {len(case_rows)} 个 deterministic case；每个 case 的 JSON payload 位于 `case_payloads/`。",
        ]
    )
    if attention_warnings:
        lines.append(f"- {len(attention_warnings)} 个模型或 cache 未提供 attention diagnostics，已安全降级。")
    if robustness_rows:
        lines.extend(
            [
                "",
                "## 鲁棒性切片",
                f"- robustness_summary.csv 记录 {len(robustness_rows)} 行 clean/drop/noise/masking 条件；cache-only 分析会把需重跑 forward 的条件标记为 skipped/status。",
            ]
        )
    if benchmark_context.get("enabled"):
        lines.extend(
            [
                "",
                "## GPS shortcut reliance",
                f"- benchmark_robustness_matrix.csv 汇总 {int(benchmark_context.get('matrix_row_count', 0))} 行跨模型扰动结果；benchmark_case_selection.csv 记录 aggregate case study 选择。",
                "- drop GPS、misleading GPS 和 temporal delay 属于 counterfactual intervention；attention/embedding 只作为解释性诊断，不单独构成因果证明。",
            ]
        )
        generated_figures = benchmark_context.get("generated_figures", [])
        if generated_figures:
            lines.append("- Benchmark 曲线输出：" + ", ".join(f"`{item}`" for item in generated_figures) + "。")
        skipped = benchmark_context.get("skipped", [])
        if skipped:
            lines.append("- 部分 benchmark 图表或 case payload 已降级/跳过：" + ", ".join(str(item) for item in skipped) + "。")
    if evidence_context.get("enabled"):
        lines.extend(str(item) for item in evidence_context.get("report_lines", []))
    lines.extend(
        [
            "",
            "## 不能过度声称的 caveat",
            "- UMAP/t-SNE/PCA 投影不是严格因果证据，需结合 kNN purity、neighbor beam distance 和逐样本表解读。",
            "- Attention heatmap 是诊断信号，不等同于因果解释；缺少 attention 的 baseline 已用概率、错误和 embedding 对比降级。",
            "- Case study 使用固定 seed 和规则选择，但仍应引用完整候选表，避免只展示成功样本。",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- `{item}`" for item in sorted(set(str(item) for item in warnings)))
    return "\n".join(lines) + "\n"


def _embedding_layers(cfg: Mapping[str, Any]) -> list[str]:
    raw = cfg.get("embeddings", {}).get("layers", ["output_features"]) if isinstance(cfg.get("embeddings"), Mapping) else ["output_features"]
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


@contextmanager
def _forward_hooks(
    model: torch.nn.Module,
    layers: Iterable[str],
    *,
    warnings: list[str],
    model_name: str,
):
    handles = []
    state: dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    special = {"input_features", "output_features"}
    for layer in layers:
        if layer in special or layer.startswith("diagnostics."):
            continue
        module = modules.get(layer)
        if module is None:
            warnings.append(f"embedding_layer_missing:{model_name}:{layer}")
            continue
        def _hook(_module, _inputs, output, *, name=layer):
            tensor = _first_tensor(output)
            if tensor is not None:
                state[name] = tensor.detach().cpu()
        handles.append(module.register_forward_hook(_hook))
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _temporary_attention_diagnostics(model: torch.nn.Module, enabled: bool):
    original = []
    if enabled:
        for module in model.modules():
            if hasattr(module, "return_attention") and hasattr(module, "last_attention_map"):
                original.append((module, bool(getattr(module, "return_attention"))))
                setattr(module, "return_attention", True)
    try:
        yield
    finally:
        for module, value in original:
            setattr(module, "return_attention", value)


def _batch_embeddings_from_step(
    model_output,
    hook_state: Mapping[str, torch.Tensor],
    layers: Iterable[str],
    *,
    batch_size: int,
    horizon_index: int,
    warnings: list[str],
    model_name: str,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    diagnostics = model_output.diagnostics if isinstance(model_output.diagnostics, Mapping) else {}
    for layer in layers:
        tensor = None
        if layer == "input_features":
            tensor = model_output.input_features
        elif layer == "output_features":
            tensor = model_output.output_features
        elif layer.startswith("diagnostics."):
            tensor = diagnostics.get(layer.split(".", 1)[1])
        else:
            tensor = hook_state.get(layer)
        if tensor is None or not torch.is_tensor(tensor):
            if layer not in {"input_features", "output_features"}:
                warnings.append(f"embedding_layer_missing:{model_name}:{layer}")
            continue
        matrix = _feature_matrix(tensor, batch_size=batch_size, horizon_index=horizon_index)
        if matrix is not None:
            result[layer] = matrix
    return result


def _feature_matrix(tensor: torch.Tensor, *, batch_size: int, horizon_index: int) -> np.ndarray | None:
    values = tensor.detach().cpu().to(torch.float32)
    if values.ndim == 0 or int(values.shape[0]) != int(batch_size):
        return None
    if values.ndim == 2:
        return values.numpy()
    if values.ndim == 3 and values.shape[1] > horizon_index:
        return values[:, horizon_index, :].reshape(batch_size, -1).numpy()
    if values.ndim > 2:
        flat = values.reshape(batch_size, -1, values.shape[-1]).mean(dim=1)
        return flat.reshape(batch_size, -1).numpy()
    return values.reshape(batch_size, -1).numpy()


def _attention_from_model(model: torch.nn.Module) -> torch.Tensor | None:
    for module in model.modules():
        attention = getattr(module, "last_attention_map", None)
        if torch.is_tensor(attention):
            return attention
    return None


def _token_grid_from_payload(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    for key in ("token_grid_shape", "token_grid", "patch_grid_shape", "grid_shape"):
        if key not in payload:
            continue
        values = np.asarray(payload[key]).reshape(-1)
        if values.size >= 2:
            return int(values[0]), int(values[1])
    return None


def _attention_diagnostics_from_array(
    model_name: str,
    attention: np.ndarray,
    sample_rows: list[dict[str, Any]],
    *,
    max_maps: int,
    token_grid: tuple[int, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    arr = np.asarray(attention, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    if arr.ndim != 4:
        return [], {}
    rows: list[dict[str, Any]] = []
    maps: dict[str, np.ndarray] = {}
    count = min(arr.shape[0], len(sample_rows))
    for idx in range(count):
        sample_id = str(sample_rows[idx].get("sample_id", idx))
        item = arr[idx]
        probs = item.reshape(-1, item.shape[-1])
        entropy = np.asarray([_entropy(row / max(row.sum(), 1e-12)) for row in probs], dtype=np.float64)
        avg = item.mean(axis=(0, 1))
        grid = _attention_grid(avg, token_grid=token_grid)
        center_y, center_x = _attention_center_of_mass(grid)
        rows.append(
            {
                "model": model_name,
                "sample_id": sample_id,
                "attention_entropy": float(np.mean(entropy)),
                "effective_patch_count": float(np.mean(np.exp(entropy))),
                "query_diversity": float(_query_diversity(item)),
                "center_y": float(center_y),
                "center_x": float(center_x),
                "time_steps": int(item.shape[0]),
                "queries": int(item.shape[1]),
                "query_count": int(item.shape[1]),
                "patches": int(item.shape[2]),
                "patch_count": int(item.shape[2]),
                "token_grid_height": int(grid.shape[0]),
                "token_grid_width": int(grid.shape[1]),
                "aggregation_method": "mean_time_query",
            }
        )
        if len(maps) < max_maps:
            maps[sample_id] = grid.astype(np.float32)
    return rows, maps


def _attention_grid(values: np.ndarray, *, token_grid: tuple[int, int] | None = None) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if token_grid is not None:
        height, width = token_grid
        if height <= 0 or width <= 0 or height * width != flat.size:
            raise ValueError(f"token_grid {token_grid} does not match {flat.size} attention patches.")
        return flat.reshape(height, width)
    width = int(round(math.sqrt(flat.size)))
    if width > 0 and width * width == flat.size:
        return flat.reshape(width, width)
    return flat.reshape(1, -1)


def _attention_center_of_mass(grid: np.ndarray) -> tuple[float, float]:
    values = np.asarray(grid, dtype=np.float64)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0, 0.0
    yy, xx = np.indices(values.shape)
    return float((yy * values).sum() / total), float((xx * values).sum() / total)


def _query_diversity(attention: np.ndarray) -> float:
    item = np.asarray(attention, dtype=np.float64)
    if item.ndim != 3 or item.shape[1] <= 1:
        return 0.0
    queries = item.mean(axis=0)
    distances = []
    for i in range(queries.shape[0]):
        for j in range(i + 1, queries.shape[0]):
            distances.append(float(np.mean(np.abs(queries[i] - queries[j]))))
    return float(np.mean(distances)) if distances else 0.0


def _embedding_neighbor_rows(
    matrix: np.ndarray,
    rows: list[dict[str, Any]],
    cfg: Mapping[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    if matrix.shape[0] < 2:
        warnings.append("embedding_neighbors_skipped:too_few_samples")
        return []
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:
        warnings.append(f"embedding_neighbors_skipped:sklearn_unavailable:{exc}")
        return []
    k = max(1, min(int(cfg.get("embeddings", {}).get("neighbors", 10)), matrix.shape[0] - 1))
    clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(clean)
    _, indices = nn.kneighbors(clean)
    targets = np.asarray([int(row.get("target", 0)) for row in rows], dtype=np.int64)
    num_beams = int(max(targets.max() + 1 if targets.size else 1, 1))
    grouped: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(rows):
        grouped.setdefault((str(row.get("model", "")), str(row.get("layer", ""))), []).append(idx)
    output: list[dict[str, Any]] = []
    for (model, layer), idxs in grouped.items():
        same = []
        adjacent = []
        distances = []
        far_distances = []
        for idx in idxs:
            neighbors = [item for item in indices[idx, 1:] if item < len(rows)]
            if not neighbors:
                continue
            neighbor_targets = targets[neighbors]
            dist = circular_beam_distance(neighbor_targets, targets[idx], num_beams=num_beams)
            same.append(float(np.mean(neighbor_targets == targets[idx])))
            adjacent.append(float(np.mean(np.asarray(dist) <= 1)))
            distances.append(float(np.mean(dist)))
            if str(rows[idx].get("error_bucket")) == "far_error":
                far_distances.append(float(np.mean(dist)))
        output.append(
            {
                "model": model,
                "layer": layer,
                "sample_count": int(len(idxs)),
                "k": int(k),
                "label_purity": float(np.mean(same)) if same else 0.0,
                "circular_neighbor_consistency": float(np.mean(adjacent)) if adjacent else 0.0,
                "mean_neighbor_beam_distance": float(np.mean(distances)) if distances else 0.0,
                "far_error_neighbor_beam_distance": float(np.mean(far_distances)) if far_distances else 0.0,
            }
        )
    return output


def _reduce_embeddings(matrix: np.ndarray, cfg: Mapping[str, Any], warnings: list[str]) -> tuple[np.ndarray, str]:
    clean = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if clean.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float32), "skipped"
    method = str(cfg.get("embeddings", {}).get("method", "umap")).lower()
    seed = int(cfg.get("sampling", {}).get("seed", 42))
    if method == "umap":
        try:
            reducer = _umap_reducer(n_components=2, random_state=seed)
            return np.asarray(reducer.fit_transform(clean), dtype=np.float32), "umap"
        except Exception as exc:
            warnings.append(f"embedding_reducer_fallback:umap_unavailable:{exc}")
            method = "tsne"
    if method in {"tsne", "t-sne"} and clean.shape[0] >= 4:
        try:
            from sklearn.manifold import TSNE
            perplexity = max(2, min(30, clean.shape[0] // 3))
            coords = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(clean)
            return np.asarray(coords, dtype=np.float32), "tsne"
        except Exception as exc:
            warnings.append(f"embedding_reducer_fallback:tsne_unavailable:{exc}")
    try:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=seed).fit_transform(clean)
        return np.asarray(coords, dtype=np.float32), "pca"
    except Exception as exc:
        warnings.append(f"embedding_reducer_skipped:pca_unavailable:{exc}")
        return np.empty((0, 2), dtype=np.float32), "skipped"


def _umap_reducer(**kwargs: Any):
    import umap  # type: ignore
    return umap.UMAP(**kwargs)


def _plot_embedding_projection(
    figures_dir: Path,
    coords: np.ndarray,
    rows: list[dict[str, Any]],
    field: str,
    method: str,
    cfg: Mapping[str, Any],
    warnings: list[str],
) -> None:
    if not _matplotlib_available(warnings):
        return
    import matplotlib.pyplot as plt

    values = [str(row.get(field, "")) for row in rows]
    categories = {value: idx for idx, value in enumerate(sorted(set(values)))}
    colors = np.asarray([categories[value] for value in values], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="tab20", s=12, alpha=0.82)
    ax.set_title(f"{method.upper()} embedding by {field}, n={len(rows)}")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    if len(categories) <= 12:
        handles = []
        for value, idx in categories.items():
            handles.append(plt.Line2D([0], [0], marker="o", color="w", label=value, markerfacecolor=scatter.cmap(scatter.norm(idx)), markersize=6))
        ax.legend(handles=handles, loc="best", fontsize=7)
    fig.tight_layout()
    _save_figure(fig, figures_dir / f"embedding_{field}", _output_formats(cfg), dpi=int(cfg.get("outputs", {}).get("dpi", 180)))
    plt.close(fig)


def _plot_case_panel(
    case_figures_dir: Path,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    analyses: Mapping[str, ModelAnalysis],
    cfg: Mapping[str, Any],
    formats: tuple[str, ...],
    warnings: list[str],
) -> None:
    if not _matplotlib_available(warnings):
        return
    import matplotlib.pyplot as plt

    models = list(analyses)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.02,
        0.92,
        f"{row.get('case_group')} | sample {row.get('sample_id')}\nTarget beam: {row.get('target')}",
        va="top",
        ha="left",
        fontsize=10,
    )
    error_lines = []
    for model in models:
        error_lines.append(
            f"{model}: e1={row.get(f'{model}_top1_error', '')}, "
            f"top3d={row.get(f'{model}_top3_min_distance', '')}, "
            f"dba={row.get(f'{model}_dba_contribution', '')}"
        )
    axes[0, 0].text(
        0.02,
        0.58,
        ("Image strip/GPS path unavailable in this payload" if not payload.get("image_paths") else "Image paths recorded in JSON payload")
        + "\n"
        + "\n".join(error_lines[:4]),
        va="top",
        fontsize=8,
    )
    for model in models:
        top_key = f"{model}_top5"
        probs_key = f"{model}_gt_probability"
        try:
            top5 = json.loads(str(row.get(top_key, "[]")))
        except json.JSONDecodeError:
            top5 = []
        axes[0, 1].plot(list(range(len(top5))), top5, marker="o", label=model)
        axes[1, 0].bar(model, float(row.get(probs_key, 0.0) or 0.0))
    axes[0, 1].set_title("Top-5 beam ids")
    axes[0, 1].set_xlabel("rank")
    axes[0, 1].set_ylabel("beam")
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].set_title("GT probability")
    axes[1, 0].set_ylabel("probability")
    sample_id = str(row.get("sample_id"))
    attention_map = None
    attention_model = None
    for model, analysis in analyses.items():
        if sample_id in analysis.attention_maps:
            attention_map = analysis.attention_maps[sample_id]
            attention_model = model
            break
    if attention_map is None:
        axes[1, 1].axis("off")
        axes[1, 1].text(0.05, 0.7, "attention unavailable", fontsize=10)
    else:
        im = axes[1, 1].imshow(attention_map, cmap="magma")
        axes[1, 1].set_title(f"{attention_model} attention")
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046)
    fig.tight_layout()
    _save_figure(fig, case_figures_dir / f"{_safe_slug(row['case_group'])}_{_safe_slug(row['sample_id'])}", formats, dpi=int(cfg.get("outputs", {}).get("dpi", 180)))
    plt.close(fig)


def _select_case_rows(
    comparison_rows: list[dict[str, Any]],
    *,
    groups: tuple[str, ...],
    per_group: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for group in groups:
        candidates = [dict(row) for row in comparison_rows if int(row.get(group, 0) or 0) == 1]
        candidates.sort(key=lambda row: (str(row.get("scene", "")), str(row.get("sample_id", ""))))
        if len(candidates) > per_group:
            indices = np.sort(rng.choice(len(candidates), size=per_group, replace=False))
            candidates = [candidates[int(idx)] for idx in indices]
        for rank, row in enumerate(candidates):
            row["case_group"] = group
            row["case_rank"] = rank
            selected.append(row)
    return selected


def _case_payload(row: Mapping[str, Any], analyses: Mapping[str, ModelAnalysis]) -> dict[str, Any]:
    sample_id = str(row.get("sample_id"))
    payload = {
        "sample_id": sample_id,
        "case_group": row.get("case_group"),
        "target": row.get("target"),
        "scene": row.get("scene"),
        "models": {},
        "image_paths": [],
        "gps": [],
    }
    for name, analysis in analyses.items():
        match = next((item for item in analysis.sample_rows if str(item.get("sample_id")) == sample_id), None)
        if match is None:
            continue
        payload["models"][name] = {
            "top1": match.get("top1"),
            "top3": match.get("top3"),
            "top5": match.get("top5"),
            "target_rank": match.get("target_rank"),
            "top1_error": match.get("top1_error"),
            "top3_min_distance": match.get("top3_min_distance"),
            "dba_contribution": match.get("dba_contribution"),
            "gt_probability": match.get("gt_probability"),
            "attention_available": sample_id in analysis.attention_maps,
        }
    return payload


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_scalar(row.get(key, "")) for key in fieldnames})


def _save_figure(fig, stem: Path, formats: tuple[str, ...], *, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")


def _matplotlib_available(warnings: list[str]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        return True
    except Exception as exc:
        warnings.append(f"matplotlib_unavailable:{exc}")
        return False


def _output_formats(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    raw = cfg.get("outputs", {}).get("formats", DEFAULT_OUTPUT_FORMATS) if isinstance(cfg.get("outputs"), Mapping) else DEFAULT_OUTPUT_FORMATS
    if isinstance(raw, str):
        raw = [raw]
    formats = []
    for item in raw:
        fmt = str(item).strip().lower().lstrip(".")
        if fmt and fmt not in formats:
            formats.append(fmt)
    if "png" not in formats:
        formats.insert(0, "png")
    return tuple(formats)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-12)


def _entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / max(float(p.sum()), 1e-12)
    return float(-np.sum(p * np.log(np.maximum(p, 1e-12))))


def _sample_dba_contribution(
    topk: list[int],
    truth: int,
    num_beams: int,
    delta: float,
    distance_mode: str,
) -> float:
    distances = _class_distance(np.asarray(topk[:3], dtype=np.int64), int(truth), num_beams, distance_mode)
    normalized = np.minimum(distances / max(float(delta), 1e-8), 1.0)
    return float(np.mean(1.0 - np.minimum.accumulate(normalized)))


def _class_distance(preds: np.ndarray, truth: int, num_beams: int, mode: str) -> np.ndarray:
    normalized = str(mode or "circular").strip().lower().replace("-", "_")
    if normalized in {"circular", "wrap", "wrapped"}:
        return np.asarray(circular_beam_distance(preds, int(truth), num_beams=int(num_beams)), dtype=np.float64)
    if normalized in {"linear", "official", "beambench", "non_circular", "noncircular"}:
        return np.abs(np.asarray(preds, dtype=np.int64) - int(truth)).astype(np.float64)
    raise ValueError("distance_mode must be one of 'circular' or 'linear'.")


def _min_distance(predictions: list[int], truth: int, num_beams: int, mode: str) -> float:
    if not predictions:
        return float("nan")
    return float(np.min(_class_distance(np.asarray(predictions, dtype=np.int64), truth, num_beams, mode)))


def _error_bucket(top1_error: float, top3_min_distance: float) -> str:
    if top1_error == 0:
        return "hit"
    if top3_min_distance == 0:
        return "rank_miss"
    if top3_min_distance <= 2:
        return "near_miss"
    return "far_error"


def _sample_id_from_metadata(metadata: Mapping[str, Any], *, index: int, target: int, split: str) -> str:
    for key in ("sample_id", "id", "sequence_id", "seq_id", "uid"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    scene = _metadata_value(metadata, ("scene", "scene_id", "scene_slug"), default="unknown")
    global_index = _metadata_value(metadata, ("global_index", "index", "row_index", "sample_index"), default=index)
    return f"scene={scene}|split={split}|index={global_index}|target={target}"


def _metadata_value(metadata: Mapping[str, Any], keys: tuple[str, ...], *, default: Any) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return default


def _metadata_rows_from_batch(metadata: Any) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return [dict(item) for item in metadata if isinstance(item, Mapping)]
    if not isinstance(metadata, Mapping):
        return []
    length = 0
    for value in metadata.values():
        if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
            length = max(length, int(value.shape[0]))
        elif isinstance(value, (list, tuple)):
            length = max(length, len(value))
        else:
            length = max(length, 1)
    rows: list[dict[str, Any]] = []
    for index in range(length):
        row = {}
        for key, value in metadata.items():
            row[key] = _metadata_value_at(value, index)
        rows.append(row)
    return rows


def _metadata_rows_for_batch(metadata: Any, *, batch_size: int | None) -> list[dict[str, Any]]:
    rows = _metadata_rows_from_batch(metadata)
    if batch_size is None or batch_size <= 0:
        return rows
    if len(rows) < batch_size:
        rows.extend({} for _ in range(batch_size - len(rows)))
    return rows[:batch_size]


def _batch_size_from_batch(batch: Mapping[str, Any]) -> int | None:
    for key in ("image", "images", "gps", "lidar", "mmwave", "csi", "label", "labels", "beam", "target"):
        value = batch.get(key)
        size = _leading_size(value)
        if size is not None:
            return size
    for key, value in batch.items():
        if key == "metadata":
            continue
        size = _leading_size(value)
        if size is not None:
            return size
    return None


def _leading_size(value: Any) -> int | None:
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        return int(value.shape[0])
    if isinstance(value, Mapping):
        for item in value.values():
            size = _leading_size(item)
            if size is not None:
                return size
    if isinstance(value, (list, tuple)) and value:
        return len(value)
    return None


def _metadata_value_at(value: Any, index: int) -> Any:
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        item = value[index]
        return item.item() if hasattr(item, "item") else item
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def _metadata_from_cached_payload(payload: Mapping[str, Any], *, sample_count: int) -> list[dict[str, Any]]:
    if "metadata_json" not in payload:
        return [{} for _ in range(sample_count)]
    raw = payload["metadata_json"]
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    if isinstance(parsed, list):
        return [dict(item) if isinstance(item, Mapping) else {} for item in parsed]
    return [{} for _ in range(sample_count)]


def _string_array_from_npz(payload: Mapping[str, Any], key: str) -> list[str] | None:
    if key not in payload:
        return None
    arr = payload[key]
    if isinstance(arr, np.ndarray):
        return [str(item) for item in arr.tolist()]
    return [str(item) for item in arr]


def _query_model_name(names: list[str], cfg: Mapping[str, Any]) -> str:
    configured = cfg.get("sampling", {}).get("query_model") if isinstance(cfg.get("sampling"), Mapping) else None
    if configured and str(configured) in names:
        return str(configured)
    for name in names:
        lowered = name.lower()
        if "query" in lowered or "gps_query" in lowered or "pool" in lowered:
            return name
    return names[-1]


def _baseline_model_name(names: list[str], query_model: str, cfg: Mapping[str, Any]) -> str:
    configured = cfg.get("sampling", {}).get("baseline_model") if isinstance(cfg.get("sampling"), Mapping) else None
    if configured and str(configured) in names:
        return str(configured)
    for name in names:
        if name != query_model:
            return name
    return query_model


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _confusion_matrix(rows: list[Mapping[str, Any]], num_beams: int) -> np.ndarray:
    matrix = np.zeros((num_beams, num_beams), dtype=np.int64)
    for row in rows:
        target = int(row.get("target", 0))
        pred = int(row.get("top1", 0))
        if 0 <= target < num_beams and 0 <= pred < num_beams:
            matrix[target, pred] += 1
    return matrix


def _figure_title(model_name: str, cfg: Mapping[str, Any], rows: list[Mapping[str, Any]], *, extra: str) -> str:
    split = cfg.get("split", {}).get("evaluation_split", "test") if isinstance(cfg.get("split"), Mapping) else "test"
    scenes = cfg.get("split", {}).get("scenes", "") if isinstance(cfg.get("split"), Mapping) else ""
    return f"{model_name} | {extra} | split={split} scenes={scenes} n={len(rows)}"


def _pad_and_stack(chunks: list[np.ndarray]) -> np.ndarray:
    width = max(int(chunk.shape[1]) for chunk in chunks)
    padded = []
    for chunk in chunks:
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.shape[1] < width:
            pad = np.full((arr.shape[0], width - arr.shape[1]), np.nan, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        padded.append(arr)
    return np.concatenate(padded, axis=0)


def _apply_perturbation_to_batch(batch: Mapping[str, Any], perturbation: Mapping[str, Any] | None, *, seed: int, step_index: int) -> dict[str, Any]:
    result = dict(batch)
    if not perturbation:
        return result
    kind = str(perturbation.get("type", ""))
    rng = torch.Generator()
    rng.manual_seed(int(seed) + int(step_index))
    if kind == "gps_noise" and "gps" in result and torch.is_tensor(result["gps"]):
        std = float(perturbation.get("std", perturbation.get("severity", 0.0)))
        result["gps"] = result["gps"] + torch.randn(result["gps"].shape, generator=rng, dtype=result["gps"].dtype) * std
    elif kind == "image_masking" and "image" in result and torch.is_tensor(result["image"]):
        ratio = max(0.0, min(float(perturbation.get("ratio", perturbation.get("severity", 0.0))), 1.0))
        mask = torch.rand(result["image"].shape, generator=rng, dtype=torch.float32) >= ratio
        result["image"] = result["image"] * mask.to(dtype=result["image"].dtype)
    elif kind == "drop_modality":
        modality = str(perturbation.get("modality", ""))
        if modality in result and torch.is_tensor(result[modality]):
            result[modality] = torch.zeros_like(result[modality])
    return result


def _force_modality_mask(primary_cfg: Mapping[str, Any], perturbation: Mapping[str, Any] | None, *, device: torch.device) -> torch.Tensor | None:
    if not perturbation or str(perturbation.get("type", "")) != "drop_modality":
        return None
    modalities = primary_cfg.get("modalities")
    if not isinstance(modalities, (list, tuple)):
        return None
    modality = str(perturbation.get("modality", ""))
    if modality not in modalities:
        return None
    mask = torch.ones(len(modalities), dtype=torch.bool, device=device)
    mask[list(modalities).index(modality)] = False
    return mask


def _deterministic_subset(dataset: Any, *, max_samples: int, seed: int) -> Any:
    total = len(dataset)
    limit = max(0, min(int(max_samples), int(total)))
    if limit >= total:
        return dataset
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(total, size=limit, replace=False)).astype(int).tolist()
    return Subset(dataset, indices)


def _dataset_attr_recursive(dataset, attr: str):
    value = getattr(dataset, attr, None)
    if value is not None:
        return value
    for component in getattr(dataset, "datasets", []) or []:
        value = _dataset_attr_recursive(component, attr)
        if value is not None:
            return value
    parent = getattr(dataset, "dataset", None)
    if parent is not None:
        return _dataset_attr_recursive(parent, attr)
    return None


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _safe_slug(value: Any) -> str:
    text = str(value)
    cleaned = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "item"


__all__ = [
    "ANALYSIS_VERSION",
    "load_analysis_config",
    "run_jepa_visual_analysis",
    "safe_metadata_collate",
    "sample_metrics_from_logits",
    "sanitize_metadata",
]
