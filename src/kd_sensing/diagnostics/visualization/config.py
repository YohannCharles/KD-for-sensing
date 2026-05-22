from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kd_sensing.modalities import MODALITY_ORDER, dataset_flags_for_modalities, normalize_modalities
from kd_sensing.utils.paths import resolve_path



VALID_MODALITIES = MODALITY_ORDER
VALID_SPLITS = ("train", "test")
METADATA_FILE_TEMPLATES = {
    "summary": ("summary", ".json"),
    "samples_jsonl": ("samples", ".jsonl"),
    "samples_csv": ("samples", ".csv"),
    "split_stats": ("split_stats", ".json"),
    "final_config": ("final_config", ".yaml"),
}


@dataclass(frozen=True)
class VisualizationConfig:
    output_dir: Path
    splits: tuple[str, ...]
    sample_count: int
    per_seq_sample_count: int | None
    seed: int
    seq_index: tuple[Any, ...] | None
    labels: tuple[int, ...] | None
    modalities: tuple[str, ...] | None
    compare_scenes: tuple[int, ...] | None
    max_frames_per_sample: int
    include_raw_image_preview: bool
    preserve_existing_outputs: bool

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return _json_ready(payload)

def parse_visualization_config(cfg: dict[str, Any]) -> VisualizationConfig:
    raw = cfg.get("diagnostics", {}).get("visualization", {}) or {}
    scene_slug = str(cfg.get("data", {}).get("dataset", {}).get("scene_slug") or "scene")
    run_name = cfg.get("output", {}).get("run_name") or cfg.get("experiment", {}).get("name") or "run"
    default_output = Path(str(cfg.get("output", {}).get("dir", "outputs"))) / "diagnostics" / scene_slug / str(run_name)
    output_raw = raw.get("output_dir", default_output)

    splits = _string_tuple(raw.get("splits", ("train", "test")), name="diagnostics.visualization.splits")
    invalid_splits = [split for split in splits if split not in VALID_SPLITS]
    if invalid_splits:
        raise ValueError(f"diagnostics.visualization.splits only supports train/test; got {invalid_splits}.")

    sample_count = int(raw.get("sample_count", 4))
    if sample_count < 0:
        raise ValueError("diagnostics.visualization.sample_count must be non-negative.")
    per_seq_raw = raw.get("per_seq_sample_count")
    per_seq_sample_count = None if per_seq_raw is None else int(per_seq_raw)
    if per_seq_sample_count is not None and per_seq_sample_count < 0:
        raise ValueError("diagnostics.visualization.per_seq_sample_count must be non-negative.")
    max_frames = int(raw.get("max_frames_per_sample", 4))
    if max_frames < 1:
        raise ValueError("diagnostics.visualization.max_frames_per_sample must be positive.")

    return VisualizationConfig(
        output_dir=resolve_path(output_raw),
        splits=splits,
        sample_count=sample_count,
        per_seq_sample_count=per_seq_sample_count,
        seed=int(raw.get("seed", cfg.get("experiment", {}).get("seed", 42))),
        seq_index=_optional_tuple(raw.get("seq_index")),
        labels=_optional_int_tuple(raw.get("labels")),
        modalities=_optional_modalities(raw.get("modalities")),
        compare_scenes=_optional_int_tuple(raw.get("compare_scenes")),
        max_frames_per_sample=max_frames,
        include_raw_image_preview=bool(raw.get("include_raw_image_preview", False)),
        preserve_existing_outputs=bool(raw.get("preserve_existing_outputs", True)),
    )

def apply_visualization_modalities(cfg: dict[str, Any], modalities: tuple[str, ...] | None) -> dict[str, Any]:
    result = deepcopy(cfg)
    if modalities is None:
        return result

    result.setdefault("data", {}).setdefault("dataset", {})
    dataset_cfg = result["data"]["dataset"]
    selected = normalize_modalities(modalities, context="diagnostics.visualization.modalities")
    dataset_cfg.update(dataset_flags_for_modalities(selected))

    result.setdefault("experiment", {})
    model_cfg = result.setdefault("model", {})
    if len(selected) == 1:
        result["experiment"]["task"] = selected[0]
    else:
        result["experiment"]["task"] = "fusion"
        model_cfg["modalities"] = list(selected)
        for role in ("teacher", "student"):
            role_cfg = model_cfg.get(role)
            if isinstance(role_cfg, dict):
                role_cfg.pop("modalities", None)
    return result

def resolve_metadata_output_paths(
    output_dir: Path,
    *,
    preserve_existing: bool,
    keys: tuple[str, ...],
) -> dict[str, Path]:
    invalid = [key for key in keys if key not in METADATA_FILE_TEMPLATES]
    if invalid:
        raise ValueError(f"Unknown diagnostic metadata output keys: {invalid}.")
    suffix = _metadata_output_suffix(output_dir, preserve_existing=preserve_existing, keys=keys)
    return {key: _metadata_path(output_dir, key, suffix) for key in keys}

def _metadata_output_suffix(output_dir: Path, *, preserve_existing: bool, keys: tuple[str, ...]) -> str:
    if not preserve_existing:
        return ""
    for attempt in range(10000):
        suffix = "" if attempt == 0 else f"_{attempt:03d}"
        paths = [_metadata_path(output_dir, key, suffix) for key in keys]
        if not any(path.exists() for path in paths):
            return suffix
    raise RuntimeError(f"Could not find a free diagnostic metadata suffix in {output_dir}.")

def _metadata_path(output_dir: Path, key: str, suffix: str) -> Path:
    stem, extension = METADATA_FILE_TEMPLATES[key]
    return output_dir / f"{stem}{suffix}{extension}"

def final_config_snapshot(cfg: dict[str, Any], viz: VisualizationConfig) -> dict[str, Any]:
    snapshot = deepcopy(cfg)
    snapshot.setdefault("diagnostics", {})["visualization"] = viz.to_json_dict()
    return snapshot

def _optional_tuple(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "all", "none", "null"}:
        return None
    return _tuple_from_value(value)

def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    raw = _optional_tuple(value)
    if raw is None:
        return None
    return tuple(int(item) for item in raw)

def _optional_modalities(value: Any) -> tuple[str, ...] | None:
    raw = _optional_tuple(value)
    if raw is None:
        return None
    selected = tuple(str(item).strip().lower() for item in raw)
    return normalize_modalities(selected, context="diagnostics.visualization.modalities")

def _string_tuple(value: Any, *, name: str) -> tuple[str, ...]:
    items = tuple(str(item).strip().lower() for item in _tuple_from_value(value))
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    return items

def _tuple_from_value(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    if isinstance(value, str):
        text = value.strip()
        if "," in text:
            return tuple(part.strip() for part in text.split(",") if part.strip())
        return (text,)
    return (value,)

def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value

def _json_ready(value: Any) -> Any:
    if _is_tensor_like(value):
        return _json_ready(value.detach().cpu().numpy())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value

def _is_tensor_like(value: Any) -> bool:
    return all(hasattr(value, attr) for attr in ("detach", "cpu", "numpy"))

__all__ = [
    'METADATA_FILE_TEMPLATES',
    'VALID_SPLITS',
    'VisualizationConfig',
    'apply_visualization_modalities',
    'final_config_snapshot',
    'parse_visualization_config',
    'resolve_metadata_output_paths',
]
