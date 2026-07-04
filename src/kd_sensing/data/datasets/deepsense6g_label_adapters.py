from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.data.beam_soft_targets import (
    SoftBeamLabelConfig,
    read_beam_power_vector,
    soft_distribution_from_power_or_label,
)
from kd_sensing.data.transform_ops.io import joined_resource


def read_deepsense6g_beam_label(data_root: Path, beam_path: str) -> int:
    path = joined_resource(data_root, beam_path)
    try:
        values = np.loadtxt(path)
    except Exception as exc:
        raise ValueError(f"Failed to read beam label file {path}: {exc}") from exc
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError(f"Beam label file {path} is empty.")
    return int(np.argmax(values))


def prepare_deepsense6g_beam_label_cache(dataset: Any) -> None:
    unique_paths = {
        str(path)
        for paths in [*dataset.samples.input_beam_paths, *dataset.samples.future_beam_paths]
        for path in paths
        if str(path).strip() and str(path).strip() != "-99"
    }
    for beam_path in sorted(unique_paths):
        dataset._beam_label_cache[beam_path] = dataset._read_beam_label(beam_path)


def build_deepsense6g_soft_beam_targets(
    dataset: Any,
    future_beam_paths: list[str],
    hard_labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    cfg = dataset.soft_beam_label_config
    num_classes = deepsense6g_soft_beam_num_classes(
        hard_labels,
        configured=cfg.num_classes,
        beam_label_mapping=dataset.beam_label_mapping,
    )
    distributions: list[np.ndarray] = []
    masks: list[bool] = []
    for horizon, rel_path in enumerate(future_beam_paths[: dataset.num_pred]):
        label = hard_labels[horizon] if horizon < len(hard_labels) else -100
        if label < 0:
            distributions.append(np.zeros(num_classes, dtype=np.float32))
            masks.append(False)
            continue
        distribution, _ = deepsense6g_soft_beam_distribution(
            data_root=dataset.data_root,
            rel_path=rel_path,
            label=label,
            cfg=cfg,
            num_classes=num_classes,
            split=dataset.split,
            cache=dataset._soft_beam_distribution_cache,
            beam_label_mapping=dataset.beam_label_mapping,
        )
        distributions.append(distribution)
        masks.append(True)
    while len(distributions) < int(dataset.num_pred):
        distributions.append(np.zeros(num_classes, dtype=np.float32))
        masks.append(False)
    return np.stack(distributions, axis=0), np.asarray(masks, dtype=bool)


def deepsense6g_soft_beam_distribution(
    *,
    data_root: Path,
    rel_path: object,
    label: int,
    cfg: SoftBeamLabelConfig,
    num_classes: int,
    split: str,
    cache: dict[str, tuple[np.ndarray, bool]],
    beam_label_mapping: Any,
) -> tuple[np.ndarray, bool]:
    key = str(rel_path or "").strip()
    domain = deepsense6g_soft_beam_label_domain(split, cfg)
    source = cfg.source if domain == "source" else cfg.target_source
    circular = True if domain == "target" else cfg.circular
    if domain == "target" and source != "gaussian":
        raise ValueError("target-domain soft beam labels must use circular Gaussian targets.")
    cache_key = (
        f"{domain}|{key}|{label}|{num_classes}|{source}|{cfg.sigma}|{circular}|{cfg.temperature}|"
        f"{beam_label_mapping.fingerprint}"
    )
    if cfg.cache and cache_key in cache:
        return cache[cache_key]
    power = None
    if (
        domain == "source"
        and source in {"power", "rss", "power_or_gaussian", "rss_or_gaussian"}
        and key
        and key != "-99"
    ):
        power = read_beam_power_vector(joined_resource(data_root, key), num_classes=num_classes)
    result = soft_distribution_from_power_or_label(
        power,
        int(label),
        num_classes=num_classes,
        source=source,
        sigma=cfg.sigma,
        circular=circular,
        temperature=cfg.temperature,
        epsilon=cfg.epsilon,
    )
    if result[1] and beam_label_mapping.enabled:
        result = (
            beam_label_mapping.reorder_distribution(result[0], axis=-1).astype(np.float32),
            result[1],
        )
    if cfg.cache:
        cache[cache_key] = result
    return result


def deepsense6g_soft_beam_label_domain(split: str, cfg: SoftBeamLabelConfig) -> str:
    if cfg.domain in {"source", "target"}:
        return cfg.domain
    split_text = str(split or "").strip().lower()
    if split_text.startswith("target") or split_text in {"test", "val", "validation"}:
        return "target"
    return "source"


def deepsense6g_soft_beam_num_classes(
    hard_labels: list[int],
    *,
    configured: int | None,
    beam_label_mapping: Any,
) -> int:
    if configured is not None:
        return int(configured)
    if beam_label_mapping.enabled:
        return int(beam_label_mapping.num_classes)
    if hard_labels:
        return max(64, max(int(value) for value in hard_labels) + 1)
    return 64


__all__ = [
    "build_deepsense6g_soft_beam_targets",
    "deepsense6g_soft_beam_distribution",
    "deepsense6g_soft_beam_label_domain",
    "deepsense6g_soft_beam_num_classes",
    "prepare_deepsense6g_beam_label_cache",
    "read_deepsense6g_beam_label",
]
