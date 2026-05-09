from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kd_sensing.config.io import load_config
from kd_sensing.diagnostics.visualization.core import (
    build_diagnostic_datasets,
    collect_candidates,
    parse_visualization_config,
    select_sample_candidates,
    selected_csv_frame_for_dataset,
)
from kd_sensing.diagnostics.viewer_manifest import _json_ready, _path_stat_dict, _sample_id
from kd_sensing.engine.batch import (
    forward_model,
    normalize_batch,
    prepare_gps_inputs,
    prepare_image_inputs,
    prepare_labels,
    prepare_lidar_inputs,
    prepare_mmwave_inputs,
    prepare_radar_inputs,
)
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.runtime import autocast_context, resolve_amp_settings, transfer_non_blocking
from kd_sensing.modalities import MODALITY_ORDER, dataset_flags_for_modalities, normalize_modalities
from kd_sensing.utils.artifact_registry import resolve_evaluation_checkpoint
from kd_sensing.utils.checkpoint import checkpoint_load_summary, load_model_state
from kd_sensing.utils.paths import resolve_path


DEFAULT_MODEL_CONFIGS = {
    modality: Path("configs") / modality / "teacher_no_kd.yaml"
    for modality in MODALITY_ORDER
}
DATASET_SOURCE_KEYS = {
    "type",
    "scene",
    "scene_id",
    "scene_slug",
    "data_root",
    "train_csv_name",
    "test_csv_name",
    "seq_len",
    "num_pred",
    "portion",
    "portion_strategy",
    "portion_seed",
    "image_size",
    "image_motion_cache_dir",
    "image_motion_cache_version",
    "image_motion_gaussian_sigma",
    "image_motion_threshold_ratio",
    "image_motion_threshold_strategy",
    "image_motion_grayscale",
    "fft_tuple",
    "clipped_range",
    "beam_label_cache",
    "lidar_encoding",
    "lidar_bev_size",
    "lidar_roi",
    "lidar_fov_degrees",
    "lidar_remove_ground",
    "lidar_ground_z_threshold",
    "lidar_background_path",
    "lidar_background_distance_threshold",
    "lidar_cache_dir",
    "lidar_normalization",
    "lidar_memory_cache",
}


def export_viewer_model_predictions(
    cfg: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    modalities: list[str] | tuple[str, ...] | None = None,
    model_config_paths: dict[str, str | Path] | None = None,
    checkpoint_paths: dict[str, str | Path] | None = None,
    devices: str | list[str] | tuple[str, ...] = "cuda",
    workers: int | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    force_rebuild: bool = False,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Run single-modality checkpoints and export per-beam confidence curves for the viewer."""

    selected = _prediction_modalities(cfg, modalities)
    model_config_paths = model_config_paths or {}
    checkpoint_paths = checkpoint_paths or {}
    device_list = _resolve_devices(devices)
    print(f"[viewer] Prediction devices resolved: {', '.join(device_list)}", flush=True)
    jobs = [
        _build_prediction_job(
            cfg,
            modality,
            model_config_path=model_config_paths.get(modality),
            checkpoint_path=checkpoint_paths.get(modality),
            device=device_list[index % len(device_list)],
            batch_size=batch_size,
            num_workers=num_workers,
            sample_limit=sample_limit,
        )
        for index, modality in enumerate(selected)
    ]
    target = _prediction_output_path(
        cfg,
        output_path=output_path,
        cache_dir=cache_dir,
        jobs=jobs,
        sample_limit=sample_limit,
        batch_size=batch_size,
    )
    meta_path = target.with_name("model_predictions_meta.json")
    digest = _prediction_digest(cfg, jobs=jobs, sample_limit=sample_limit, batch_size=batch_size)
    if not force_rebuild:
        cached = _cached_prediction_result(target, meta_path, digest, jobs)
        if cached is not None:
            print(f"[viewer] Reusing model prediction cache: {target}", flush=True)
            return cached

    worker_count = _worker_count(workers, jobs)
    print(f"[viewer] Running {len(jobs)} modality prediction job(s) with {worker_count} worker(s)", flush=True)
    if worker_count <= 1:
        job_results = []
        for job in jobs:
            print(f"[viewer] Starting {job['modality']} on {job['device']}", flush=True)
            job_results.append(_run_prediction_job(job))
            print(f"[viewer] Finished {job['modality']} on {job['device']}", flush=True)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
            futures = {pool.submit(_run_prediction_job, job): job for job in jobs}
            job_results = []
            for future in as_completed(futures):
                job = futures[future]
                job_results.append(future.result())
                print(f"[viewer] Finished {job['modality']} on {job['device']}", flush=True)

    records = _merge_prediction_results(job_results)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_json_ready(records), indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "cache_digest": digest,
        "prediction_path": str(target),
        "sample_count": len(records),
        "modalities": selected,
        "workers": worker_count,
        "batch_size": int(batch_size),
        "requested_devices": devices,
        "resolved_devices": device_list,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "jobs": [_job_metadata(job) for job in jobs],
    }
    meta_path.write_text(json.dumps(_json_ready(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "mode": "viewer_model_predictions",
        "cache_hit": False,
        "prediction_path": str(target),
        "meta_path": str(meta_path),
        "sample_count": len(records),
        "modalities": selected,
        "workers": worker_count,
        "requested_devices": devices,
        "resolved_devices": device_list,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "jobs": [_job_metadata(job) for job in jobs],
    }


def parse_key_value_paths(values: list[str] | tuple[str, ...] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in values or []:
        if not item:
            continue
        for part in str(item).split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(f"Expected modality=path, got: {part}")
            key, value = part.split("=", 1)
            modality = key.strip()
            normalize_modalities((modality,), context="prediction path modality")
            mapping[modality] = value.strip()
    return mapping


def parse_modalities(value: str | None) -> tuple[str, ...] | None:
    if value is None or not str(value).strip():
        return None
    return normalize_modalities(tuple(part.strip() for part in str(value).split(",") if part.strip()))


def _prediction_modalities(cfg: dict[str, Any], modalities: list[str] | tuple[str, ...] | None) -> list[str]:
    if modalities is not None:
        return list(normalize_modalities(tuple(modalities), context="prediction modalities"))
    raw = cfg.get("diagnostics", {}).get("visualization", {}).get("modalities")
    if raw:
        return list(normalize_modalities(tuple(raw), context="diagnostics.visualization.modalities"))
    raw = cfg.get("model", {}).get("modalities")
    if raw:
        return list(normalize_modalities(tuple(raw), context="model.modalities"))
    return list(MODALITY_ORDER)


def _build_prediction_job(
    viewer_cfg: dict[str, Any],
    modality: str,
    *,
    model_config_path: str | Path | None,
    checkpoint_path: str | Path | None,
    device: str,
    batch_size: int,
    num_workers: int,
    sample_limit: int | None,
) -> dict[str, Any]:
    cfg = _modality_inference_cfg(viewer_cfg, modality, model_config_path)
    resolution = resolve_evaluation_checkpoint(cfg, str(checkpoint_path) if checkpoint_path is not None else None)
    if resolution.path is None:
        raise FileNotFoundError(
            f"No checkpoint found for {modality}. Provide --model-checkpoint {modality}=PATH "
            f"or place a registry checkpoint for {cfg.get('experiment', {}).get('name')}."
        )
    if not resolution.path.exists():
        raise FileNotFoundError(f"Checkpoint for {modality} does not exist: {resolution.path}")
    return {
        "modality": modality,
        "cfg": cfg,
        "checkpoint_path": str(resolution.path),
        "checkpoint_source": resolution.source,
        "checkpoint_metadata": resolution.metadata,
        "device": str(device),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "sample_limit": sample_limit,
    }


def _modality_inference_cfg(
    viewer_cfg: dict[str, Any],
    modality: str,
    model_config_path: str | Path | None,
) -> dict[str, Any]:
    path = resolve_path(model_config_path or DEFAULT_MODEL_CONFIGS[modality])
    cfg = load_config(path)
    result = deepcopy(cfg)
    viewer_dataset = viewer_cfg.get("data", {}).get("dataset", {})
    dataset_cfg = result.setdefault("data", {}).setdefault("dataset", {})
    for key in DATASET_SOURCE_KEYS:
        if key in viewer_dataset:
            dataset_cfg[key] = deepcopy(viewer_dataset[key])
    dataset_cfg.update(dataset_flags_for_modalities((modality,)))
    result.setdefault("data", {})["cache"] = deepcopy(viewer_cfg.get("data", {}).get("cache", result["data"].get("cache", {})))
    result.setdefault("data", {})["dataloader"] = deepcopy(
        viewer_cfg.get("data", {}).get("dataloader", result["data"].get("dataloader", {}))
    )
    result.setdefault("diagnostics", {})["visualization"] = deepcopy(
        viewer_cfg.get("diagnostics", {}).get("visualization", {})
    )
    result.setdefault("experiment", {})["task"] = modality
    model_cfg = result.setdefault("model", {})
    source_num_pred = viewer_dataset.get("num_pred")
    source_seq_len = viewer_dataset.get("seq_len")
    if source_num_pred is not None:
        model_cfg["num_pred"] = int(source_num_pred)
    if source_seq_len is not None:
        model_cfg["seq_length_teacher"] = int(source_seq_len)
        model_cfg["seq_length_student"] = int(source_seq_len)
    return result


def _run_prediction_job(job: dict[str, Any]) -> dict[str, Any]:
    modality = str(job["modality"])
    cfg = job["cfg"]
    device = _torch_device(str(job["device"]))
    checkpoint_metadata = job.get("checkpoint_metadata")
    dataset_kwargs = load_normalization_artifacts(checkpoint_metadata)
    datasets = _prediction_datasets(cfg, dataset_kwargs)

    model = build_model(cfg["model"]["student"]).to(device)
    load_result = load_model_state(
        job["checkpoint_path"],
        model,
        role=f"{modality}_viewer_prediction",
        map_location=device,
        strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)),
    )
    model.eval()

    records: dict[str, Any] = {}
    viz = parse_visualization_config(cfg)
    for split in viz.splits:
        dataset = datasets[split]
        csv_frame = selected_csv_frame_for_dataset(dataset)
        candidates = collect_candidates(dataset, csv_frame)
        selected, _ = select_sample_candidates(
            candidates,
            sample_count=len(candidates),
            per_seq_sample_count=None,
            seed=viz.seed,
            seq_index=viz.seq_index,
            labels=viz.labels,
        )
        if job.get("sample_limit") is not None:
            selected = selected[: max(0, int(job["sample_limit"]))]
        sample_ids = [_sample_id(dataset, split, candidate) for candidate in selected]
        subset = _IndexedDataset(dataset, [candidate.dataset_index for candidate in selected], sample_ids)
        loader = DataLoader(
            subset,
            batch_size=max(1, int(job["batch_size"])),
            shuffle=False,
            num_workers=max(0, int(job["num_workers"])),
            pin_memory=device.type == "cuda",
        )
        records.update(_predict_loader(model, loader, cfg, modality, device, job["checkpoint_path"]))

    return {
        "modality": modality,
        "records": records,
        "checkpoint_load": checkpoint_load_summary(load_result),
        "device": str(device),
    }


def _prediction_datasets(cfg: dict[str, Any], dataset_kwargs: dict[str, Any]) -> dict[str, Any]:
    if dataset_kwargs:
        splits = parse_visualization_config(cfg).splits
        datasets: dict[str, Any] = {}
        for split in splits:
            from kd_sensing.engine.data_factory import build_dataset

            datasets[split] = build_dataset(cfg, split, **dataset_kwargs)
        return datasets
    return build_diagnostic_datasets(cfg, parse_visualization_config(cfg).splits)


def _predict_loader(
    model: Any,
    loader: DataLoader,
    cfg: dict[str, Any],
    modality: str,
    device: torch.device,
    checkpoint_path: str,
) -> dict[str, Any]:
    model_cfg = cfg["model"]
    num_pred = int(model_cfg.get("num_pred", 3))
    downsample_ratio = int(model_cfg.get("downsample_ratio", 1))
    seq_length = int(model_cfg.get("seq_length_student", 8))
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    results: dict[str, Any] = {}
    with torch.inference_mode():
        for batch in loader:
            sample_ids = list(batch.pop("_sample_id"))
            batch = normalize_batch(batch)
            labels = prepare_labels(
                batch,
                num_pred=num_pred,
                downsample_ratio=downsample_ratio,
                device=device,
                non_blocking=non_blocking,
            )
            with autocast_context(amp_enabled, device, amp_dtype):
                model_output = adapt_model_output(
                    forward_model(
                        model,
                        modality,
                        **_prepared_inputs(batch, modality, seq_length, num_pred, device, non_blocking),
                    )
                )
                logits = select_prediction_slots(model_output.logits, num_pred)
            logits_array = logits.float().detach().cpu().numpy()
            probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()
            label_array = labels.detach().cpu().numpy()
            for row_index, sample_id in enumerate(sample_ids):
                results[str(sample_id)] = _sample_prediction_payload(
                    modality,
                    probs[row_index],
                    logits_array[row_index],
                    label_array[row_index],
                    checkpoint_path,
                    str(device),
                )
    return results


def _prepared_inputs(
    batch: dict[str, torch.Tensor],
    modality: str,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool,
) -> dict[str, torch.Tensor]:
    kwargs = {
        "seq_length": seq_length,
        "num_pred": num_pred,
        "device": device,
        "non_blocking": non_blocking,
    }
    if modality == "image":
        return {"image_batch": prepare_image_inputs(batch, **kwargs)}
    if modality == "radar":
        return {"radar_batch": prepare_radar_inputs(batch, **kwargs)}
    if modality == "gps":
        return {"gps_batch": prepare_gps_inputs(batch, **kwargs)}
    if modality == "lidar":
        return {"lidar_batch": prepare_lidar_inputs(batch, **kwargs)}
    if modality == "mmwave":
        return {"mmwave_batch": prepare_mmwave_inputs(batch, **kwargs)}
    raise ValueError(f"Unsupported prediction modality: {modality}")


def _sample_prediction_payload(
    modality: str,
    probs: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    checkpoint_path: str,
    device: str,
) -> dict[str, Any]:
    future_probs = probs[1:, :] if probs.shape[0] > 1 else probs
    future_logits = logits[1:, :] if logits.shape[0] > 1 else logits
    future_labels = labels[1 : 1 + future_probs.shape[0]].astype(int)
    top1 = np.argmax(future_probs, axis=-1).astype(int)
    topk_count = min(5, future_probs.shape[-1])
    topk = np.argsort(-future_probs, axis=-1)[:, :topk_count].astype(int)
    max_confidence = np.max(future_probs, axis=-1)
    return {
        "prediction": {
            "top1": top1.tolist(),
            "topk": topk.tolist(),
            "correct": bool(np.array_equal(top1, future_labels)),
            "future_labels": future_labels.tolist(),
            "checkpoint": checkpoint_path,
            "device": device,
        },
        "confidence": float(np.mean(max_confidence)) if max_confidence.size else None,
        "confidence_curves": future_probs.astype(float).tolist(),
        "beam_distribution": {
            "prob": future_probs.astype(float).tolist(),
            "logit": future_logits.astype(float).tolist(),
        },
    }


def _merge_prediction_results(job_results: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for result in job_results:
        modality = str(result["modality"])
        for sample_id, payload in result.get("records", {}).items():
            entry = merged.setdefault(
                sample_id,
                {
                    "prediction": {"modalities": {}},
                    "confidence": {},
                    "confidence_curves": {},
                    "beam_distribution": {},
                },
            )
            entry["prediction"]["modalities"][modality] = payload["prediction"]
            if payload.get("confidence") is not None:
                entry["confidence"][modality] = payload["confidence"]
            entry["confidence_curves"][modality] = payload["confidence_curves"]
            entry["beam_distribution"][modality] = payload["beam_distribution"]
    return merged


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Any, indices: list[int], sample_ids: list[str]):
        self.dataset = dataset
        self.indices = indices
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[self.indices[index]])
        sample["_sample_id"] = self.sample_ids[index]
        return sample


def _resolve_devices(devices: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(devices, str):
        text = devices.strip()
        if not text or text in {"cuda", "gpu"}:
            return _all_cuda_devices()
        if text == "auto":
            if torch.cuda.is_available():
                return _all_cuda_devices()
            return ["cpu"]
        return _expand_device_list(part.strip() for part in text.split(",") if part.strip())
    values = [str(value).strip() for value in devices if str(value).strip()]
    return _expand_device_list(values) if values else _all_cuda_devices()


def _torch_device(requested: str) -> torch.device:
    if requested.startswith("cuda"):
        _validate_cuda_device(requested)
    return torch.device(requested)


def _all_cuda_devices() -> list[str]:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError(
            "CUDA was requested for viewer model predictions, but PyTorch cannot see any GPU. "
            "Use --model-devices cpu only if you intentionally want CPU inference."
        )
    return [f"cuda:{index}" for index in range(torch.cuda.device_count())]


def _expand_device_list(values: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text in {"cuda", "gpu"}:
            resolved.extend(_all_cuda_devices())
            continue
        if text.startswith("cuda"):
            _validate_cuda_device(text)
        resolved.append(text)
    if not resolved:
        return _all_cuda_devices()
    return resolved


def _validate_cuda_device(device: str) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError(
            f"{device} was requested, but PyTorch cannot see any GPU. "
            "Use --model-devices cpu only if you intentionally want CPU inference."
        )
    if ":" not in device:
        return
    try:
        index = int(device.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError(f"Invalid CUDA device string: {device}") from exc
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"{device} was requested, but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )


def _worker_count(workers: int | None, jobs: list[dict[str, Any]]) -> int:
    if not jobs:
        return 0
    if workers is not None:
        return max(1, min(int(workers), len(jobs)))
    return max(1, min(len(jobs), os.cpu_count() or len(jobs)))


def _prediction_output_path(
    cfg: dict[str, Any],
    *,
    output_path: str | Path | None,
    cache_dir: str | Path | None,
    jobs: list[dict[str, Any]],
    sample_limit: int | None,
    batch_size: int,
) -> Path:
    if output_path is not None:
        return Path(output_path).expanduser()
    digest = _prediction_digest(cfg, jobs=jobs, sample_limit=sample_limit, batch_size=batch_size)[:16]
    root = Path(cache_dir).expanduser() if cache_dir is not None else Path("outputs/diagnostics/gradio_viewer_cache")
    return root / "model_predictions" / digest / "predictions.json"


def _prediction_digest(
    cfg: dict[str, Any],
    *,
    jobs: list[dict[str, Any]],
    sample_limit: int | None,
    batch_size: int,
) -> str:
    payload = {
        "data": cfg.get("data", {}),
        "diagnostics_visualization": cfg.get("diagnostics", {}).get("visualization", {}),
        "sample_limit": sample_limit,
        "batch_size": int(batch_size),
        "jobs": [_job_digest_payload(job) for job in jobs],
        "cache_version": 3,
    }
    return hashlib.sha256(json.dumps(_json_ready(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _job_digest_payload(job: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = str(job["checkpoint_path"])
    return {
        "modality": job["modality"],
        "device": job["device"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_stat": _path_stat_dict(checkpoint_path),
        "cfg": {
            "experiment": job["cfg"].get("experiment", {}),
            "data": job["cfg"].get("data", {}),
            "model": job["cfg"].get("model", {}),
        },
    }


def _cached_prediction_result(
    path: Path,
    meta_path: Path,
    digest: str,
    jobs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if meta.get("cache_digest") != digest or not isinstance(records, dict):
        return None
    return {
        "mode": "viewer_model_predictions",
        "cache_hit": True,
        "prediction_path": str(path),
        "meta_path": str(meta_path),
        "sample_count": len(records),
        "modalities": [job["modality"] for job in jobs],
        "workers": meta.get("workers"),
        "requested_devices": meta.get("requested_devices"),
        "resolved_devices": meta.get("resolved_devices"),
        "cuda_available": meta.get("cuda_available"),
        "cuda_device_count": meta.get("cuda_device_count"),
        "jobs": [_job_metadata(job) for job in jobs],
    }


def _job_metadata(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "modality": job["modality"],
        "checkpoint_path": job["checkpoint_path"],
        "checkpoint_source": job["checkpoint_source"],
        "device": job["device"],
    }
