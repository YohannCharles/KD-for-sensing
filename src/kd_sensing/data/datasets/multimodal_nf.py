from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from kd_sensing.data.dataset_descriptors import dataset_descriptor, descriptor_metadata, resolve_dataset_profiles
from kd_sensing.data.dataset_runtime import RuntimeDataset, SampleIndex, SampleRow
from kd_sensing.data.layouts import multimodal_nf_layout
from kd_sensing.modalities import normalize_modalities
from kd_sensing.preprocessing.multimodal_nf_codebook import (
    flatten_beam_triplet,
    parse_codebook_metadata,
)
from kd_sensing.preprocessing.multimodal_nf_constants import (
    DEFAULT_FLATTEN_ORDER,
    MULTIMODAL_NF_DATASET_TYPE,
    MULTIMODAL_NF_HDF5_KEYS,
)
from kd_sensing.preprocessing.multimodal_nf_index import (
    build_multimodal_nf_rows,
    load_multimodal_nf_index,
)
from kd_sensing.preprocessing.multimodal_nf_paths import resolve_multimodal_nf_paths
from kd_sensing.registries import DATASETS
from kd_sensing.utils.paths import resolve_path


@DATASETS.register("multimodal_nf")
class MultimodalNFDataset(RuntimeDataset):
    """Frame-wise Multimodal-NF dataset backed by worker-local HDF5 handles."""

    def __init__(
        self,
        data_root: str | None = None,
        raw_root: str | None = None,
        cache_dir: str | None = None,
        channel_path: str | None = None,
        image_path: str | None = None,
        lidar_path: str | None = None,
        index_path: str | None = None,
        split_metadata_path: str | None = None,
        split: str = "train",
        split_mode: str = "city",
        train_cities: list[str] | tuple[str, ...] | None = None,
        val_cities: list[str] | tuple[str, ...] | None = None,
        test_cities: list[str] | tuple[str, ...] | None = None,
        split_ratios: list[float] | tuple[float, float, float] = (0.7, 0.15, 0.15),
        split_seed: int = 42,
        modalities: list[str] | tuple[str, ...] | None = None,
        enabled_modalities: list[str] | tuple[str, ...] | None = None,
        input_profiles: dict[str, str] | None = None,
        image_profile: str | None = None,
        lidar_profile: str | None = None,
        gps_profile: str | None = None,
        csi_profile: str | None = None,
        csi_subcarrier_policy: str = "all",
        csi_subcarrier_index: int | None = None,
        codebook_path: str | None = None,
        codebook_shape: list[int] | tuple[int, int, int] | None = None,
        codebook_profile: str | None = None,
        codebook_metadata: dict[str, Any] | None = None,
        flatten_order: str = DEFAULT_FLATTEN_ORDER,
        return_metadata: bool = True,
        portion: float = 1.0,
        train_portion: float | None = None,
        eval_portion: float | None = None,
        val_portion: float | None = None,
        validation_portion: float | None = None,
        test_portion: float | None = None,
        seq_len: int = 8,
        num_pred: int = 3,
        pred_horizon: int | None = None,
        **_: Any,
    ) -> None:
        self.seq_len = int(seq_len)
        self.num_pred = int(pred_horizon if pred_horizon is not None else num_pred)
        if self.seq_len <= 0 or self.num_pred <= 0:
            raise ValueError(f"Multimodal-NF requires positive seq_len and num_pred, got {self.seq_len}, {self.num_pred}.")
        layout = multimodal_nf_layout()
        paths = resolve_multimodal_nf_paths(
            data_root=data_root or layout.root,
            raw_root=raw_root,
            cache_dir=cache_dir,
        )
        selected_modalities = normalize_modalities(
            tuple(modalities or enabled_modalities or ("gps", "csi")),
            context="Multimodal-NF modalities",
        )
        profile_cfg = {
            "input_profiles": dict(input_profiles or {}),
            "image_profile": image_profile,
            "lidar_profile": lidar_profile,
            "gps_profile": gps_profile,
            "csi_profile": csi_profile,
        }
        resolved_profiles = resolve_dataset_profiles(MULTIMODAL_NF_DATASET_TYPE, selected_modalities, profile_cfg)
        self.data_root = paths.data_root
        self.cache_dir = paths.cache_dir
        self.image_profile = resolved_profiles.get("image", "rgb_imagenet")
        self.lidar_profile = resolved_profiles.get("lidar")
        self.gps_profile = resolved_profiles.get("gps")
        self.csi_profile = resolved_profiles.get("csi")
        self.csi_subcarrier_policy = str(csi_subcarrier_policy or "all").lower()
        self.csi_subcarrier_index = None if csi_subcarrier_index is None else int(csi_subcarrier_index)
        self.codebook_metadata = _resolve_codebook_metadata(
            paths=paths,
            codebook_path=codebook_path,
            codebook_shape=codebook_shape,
            codebook_profile=codebook_profile,
            codebook_metadata=codebook_metadata,
            flatten_order=flatten_order,
        )
        rows, index_metadata = _load_or_build_index(
            paths=paths,
            index_path=index_path,
            split_metadata_path=split_metadata_path,
            split=split,
            raw_root=raw_root,
            channel_path=channel_path,
            image_path=image_path if "image" in selected_modalities else None,
            lidar_path=lidar_path if "lidar" in selected_modalities else None,
            split_mode=split_mode,
            train_cities=train_cities,
            val_cities=val_cities,
            test_cities=test_cities,
            split_ratios=split_ratios,
            split_seed=split_seed,
            seq_len=self.seq_len,
            num_pred=self.num_pred,
        )
        selected_portion = _portion_for_split(
            split,
            portion=portion,
            train_portion=train_portion,
            eval_portion=eval_portion,
            val_portion=val_portion,
            validation_portion=validation_portion,
            test_portion=test_portion,
        )
        rows = _apply_portion(rows, selected_portion)
        descriptor = dataset_descriptor(MULTIMODAL_NF_DATASET_TYPE)
        sample_index = SampleIndex.from_rows(
            rows,
            storage_kind=descriptor.storage_kind,
            metadata={
                **index_metadata,
                "split": _normalize_split(split),
                "selected_split": _normalize_split(split),
                "selected_num_samples": len(rows),
                "scene_slug": "multimodal_nf",
                "codebook_metadata": dict(self.codebook_metadata),
                "input_profiles": dict(resolved_profiles),
                "seq_len": self.seq_len,
                "num_pred": self.num_pred,
                "selected_portion": float(selected_portion),
            },
        )
        adapters = [
            _MultimodalNFAdapter(
                modality=modality,
                profile=resolved_profiles[modality],
                sample_key=modality,
                csi_subcarrier_policy=self.csi_subcarrier_policy,
                csi_subcarrier_index=self.csi_subcarrier_index,
            )
            for modality in selected_modalities
        ]
        target_provider = MultimodalNFTargetProvider(self.codebook_metadata)
        self.num_beam_classes = int(self.codebook_metadata["num_beam_classes"])
        self.task_semantics = "current_frame_near_field_codebook_beam_selection"
        self.legacy_task_semantics = "future_near_field_beam_prediction"
        self.target_schema = "near_field_3d_codebook_flattened_beam_class"
        self.legacy_target_schema = "near_field_beam_selection"
        self.use_gps = "gps" in selected_modalities
        self.use_lidar = "lidar" in selected_modalities
        self.use_csi = "csi" in selected_modalities
        self.use_mmwave = False
        super().__init__(
            sample_index=sample_index,
            modality_adapters=adapters,
            target_provider=target_provider,
            dataset_type=MULTIMODAL_NF_DATASET_TYPE,
            descriptor=descriptor.to_dict(),
            enabled_modalities=selected_modalities,
            input_profiles=resolved_profiles,
            return_metadata=return_metadata,
        )
        self.split = _normalize_split(split)
        self.scene_id = "multimodal_nf"
        self.scene_slug = "multimodal_nf"

    def _metadata(self, row: SampleRow) -> dict[str, Any]:
        metadata = super()._metadata(row)
        metadata["codebook"] = dict(self.codebook_metadata)
        metadata["target_schema"] = self.target_schema
        metadata["target_schema_aliases"] = [self.legacy_target_schema]
        metadata["task_semantics"] = self.task_semantics
        metadata["legacy_task_semantics"] = self.legacy_task_semantics
        metadata["seq_len"] = self.seq_len
        metadata["num_pred"] = self.num_pred
        metadata["auxiliary_labels"] = {
            "los_label": True,
            "nf_label": True,
            "traj_nlos_label": "traj_nlos" in row.resource_refs.get("hdf5_keys", {}),
            "mode_idx": "mode" in row.resource_refs.get("hdf5_keys", {}),
        }
        return metadata

    def profile_getitem_components(self, idx: int) -> dict[str, float]:
        import time

        row = self.sample_index[int(idx)]
        components: dict[str, float] = {}
        for adapter in self.modality_adapters:
            start = time.perf_counter()
            adapter.load(row)
            components[getattr(adapter, "modality", "unknown")] = time.perf_counter() - start
        start = time.perf_counter()
        self.target_provider.load(row)
        components["auxiliary_targets"] = time.perf_counter() - start
        if self.return_metadata:
            start = time.perf_counter()
            self._metadata(row)
            components["metadata"] = time.perf_counter() - start
        return components

    def auxiliary_target_metadata(self) -> dict[str, Any]:
        return {
            "beam": {
                "target_schema": self.target_schema,
                "target_schema_aliases": [self.legacy_target_schema],
                "codebook_shape": list(self.codebook_metadata["shape"]),
                "flatten_order": self.codebook_metadata["flatten_order"],
                "num_beam_classes": int(self.codebook_metadata["num_beam_classes"]),
                "codebook_path": self.codebook_metadata.get("path"),
                "codebook_fingerprint": self.codebook_metadata.get("fingerprint"),
            },
            "los": {"source": "Has_LoS", "usage": "diagnostic_or_optional_auxiliary"},
            "near_field": {"source": "Is_NF", "usage": "diagnostic_or_filter"},
        }

    def multimodal_nf_metadata(self) -> dict[str, Any]:
        return {
            "dataset": MULTIMODAL_NF_DATASET_TYPE,
            "task_semantics": self.task_semantics,
            "legacy_task_semantics": self.legacy_task_semantics,
            "target_schema": self.target_schema,
            "target_schema_aliases": [self.legacy_target_schema],
            "data_root": str(self.data_root),
            "cache_dir": str(self.cache_dir),
            "num_beam_classes": self.num_beam_classes,
            "codebook": dict(self.codebook_metadata),
            "input_profiles": dict(self.input_profiles),
            "descriptor": descriptor_metadata(MULTIMODAL_NF_DATASET_TYPE),
            "split_metadata": dict(self.sample_index.metadata),
            "seq_len": self.seq_len,
            "num_pred": self.num_pred,
        }


class _WorkerLocalHDF5:
    def __init__(self) -> None:
        self._handles: dict[str, Any] = {}
        self._dataset_paths_cache: dict[str, tuple[str, ...]] = {}
        self._dataset_key_cache: dict[tuple[str, str], str] = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handles"] = {}
        state["_dataset_paths_cache"] = {}
        state["_dataset_key_cache"] = {}
        return state

    def close(self) -> None:
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        self._dataset_paths_cache.clear()
        self._dataset_key_cache.clear()

    def _handle(self, path: str):
        if path not in self._handles:
            h5py = _require_h5py("Multimodal-NF lazy HDF5 adapter")
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def _dataset_paths_for_handle(self, path: str, handle) -> tuple[str, ...]:
        cached = self._dataset_paths_cache.get(path)
        if cached is None:
            cached = tuple(_dataset_paths(handle))
            self._dataset_paths_cache[path] = cached
        return cached

    def _dataset_key_for_modality(self, path: str, handle, row: SampleRow, modality: str) -> str:
        keys = row.resource_refs.get("hdf5_keys", {})
        if modality in keys and keys[modality] in handle:
            return str(keys[modality])
        cache_key = (path, modality)
        cached = self._dataset_key_cache.get(cache_key)
        if cached is not None:
            return cached
        aliases = MULTIMODAL_NF_HDF5_KEYS[modality]
        available = list(self._dataset_paths_for_handle(path, handle))
        by_leaf = {Path(dataset_path).name.lower(): dataset_path for dataset_path in available}
        for alias in aliases:
            if alias in available:
                self._dataset_key_cache[cache_key] = alias
                return alias
            alias_key = alias.lower()
            if alias_key in by_leaf:
                resolved = by_leaf[alias_key]
                self._dataset_key_cache[cache_key] = resolved
                return resolved
        raise KeyError(
            f"Multimodal-NF sample {row.sample_id} could not resolve HDF5 key for modality '{modality}'. "
            f"Expected aliases {aliases}; available datasets: {available}."
        )


class _MultimodalNFAdapter(_WorkerLocalHDF5):
    def __init__(
        self,
        *,
        modality: str,
        profile: str,
        sample_key: str,
        csi_subcarrier_policy: str,
        csi_subcarrier_index: int | None,
    ) -> None:
        super().__init__()
        self.modality = modality
        self.profile = profile
        self.sample_key = sample_key
        self.csi_subcarrier_policy = csi_subcarrier_policy
        self.csi_subcarrier_index = csi_subcarrier_index

    def load(self, row: SampleRow) -> dict[str, Any]:
        path = _resource_path_for_modality(row, self.modality)
        handle = self._handle(path)
        key = self._dataset_key_for_modality(path, handle, row, self.modality)
        indices = [int(value) for value in row.resource_refs.get("history_indices", [row.resource_refs["channel_index"]])]
        value = _read_hdf5_rows(handle[key], indices)
        tensor = self._to_tensor(value, row=row)
        return {self.sample_key: tensor}

    def metadata(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "profile": self.profile,
            "sample_key": self.sample_key,
            "lazy_hdf5": True,
            "csi_subcarrier_policy": self.csi_subcarrier_policy if self.modality == "csi" else None,
        }

    def _to_tensor(self, value: np.ndarray, *, row: SampleRow) -> torch.Tensor:
        if self.modality == "gps":
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, 3)
            if array.ndim != 2 or int(array.shape[-1]) != 3:
                _shape_error(row, self.modality, self.profile, array.shape, "expected [T, 3]")
            return torch.from_numpy(np.ascontiguousarray(array))
        if self.modality == "csi":
            array = _coerce_csi(value, policy=self.csi_subcarrier_policy, subcarrier_index=self.csi_subcarrier_index)
            if array.ndim == 3:
                array = array.reshape(1, *array.shape)
            if array.ndim != 4 or array.shape[-1] != 2:
                _shape_error(row, self.modality, self.profile, array.shape, "expected [T, M, K, 2]")
            return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        if self.modality == "image":
            array = np.asarray(value)
            if array.ndim == 3:
                array = array.reshape(1, *array.shape)
            if array.ndim != 4 or 3 not in (array.shape[1], array.shape[-1]):
                _shape_error(row, self.modality, self.profile, array.shape, "expected [T, H, W, 3] or [T, 3, H, W]")
            if array.shape[-1] == 3 and array.shape[1] != 3:
                array = np.moveaxis(array, -1, 1)
            array = np.asarray(array, dtype=np.float32)
            if array.size and float(np.nanmax(array)) > 2.0:
                array = array / 255.0
            return torch.from_numpy(np.ascontiguousarray(array))
        if self.modality == "lidar":
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 2:
                array = array.reshape(1, *array.shape)
            if array.ndim != 3 or int(array.shape[-1]) != 3:
                _shape_error(row, self.modality, self.profile, array.shape, "expected [T, P, 3]")
            return torch.from_numpy(np.ascontiguousarray(array))
        raise ValueError(f"Unsupported Multimodal-NF modality '{self.modality}'.")


class MultimodalNFTargetProvider(_WorkerLocalHDF5):
    target_schema = "near_field_3d_codebook_flattened_beam_class"
    target_schema_aliases = ("near_field_beam_selection",)

    def __init__(self, codebook_metadata: dict[str, Any]) -> None:
        super().__init__()
        self.codebook_metadata = dict(codebook_metadata)
        self.codebook_shape = tuple(int(value) for value in self.codebook_metadata["shape"])
        self.flatten_order = str(self.codebook_metadata.get("flatten_order", DEFAULT_FLATTEN_ORDER))

    def load(self, row: SampleRow) -> dict[str, Any]:
        path = str(row.target_ref.get("channel_path") or row.resource_refs["channel_path"])
        indices = [int(value) for value in row.target_ref.get("target_indices", [row.target_ref.get("channel_index", row.resource_refs["channel_index"])] )]
        handle = self._handle(path)
        keys = row.resource_refs.get("hdf5_keys", {})
        beam_idx = _read_hdf5_rows(handle[keys.get("beam_idx", row.target_ref["beam_idx_key"])], indices)
        beam_power = _read_hdf5_rows(handle[keys.get("beam_power", row.target_ref["beam_power_key"])], indices)
        if beam_idx.ndim == 2:
            beam_idx = beam_idx.reshape(1, *beam_idx.shape)
        if beam_idx.ndim != 3 or int(beam_idx.shape[-1]) != 3:
            _shape_error(row, "target", self.target_schema, beam_idx.shape, "expected BeamIdx [H, K, 3]")
        if beam_power.ndim == 1:
            beam_power = beam_power.reshape(1, -1)
        top1_targets = [
            flatten_beam_triplet(beam_idx[horizon, 0], self.codebook_shape, flatten_order=self.flatten_order)
            for horizon in range(int(beam_idx.shape[0]))
        ]
        sample = {
            "target_beam": torch.tensor(top1_targets, dtype=torch.int64),
            "beam_triplet_topk": torch.from_numpy(np.asarray(beam_idx, dtype=np.int64)),
            "beam_power_topk": torch.from_numpy(np.asarray(beam_power, dtype=np.float32)),
            "link_quality": torch.tensor(np.asarray(beam_power[:, 0], dtype=np.float32), dtype=torch.float32),
            "los_label": _vector_optional(handle, keys.get("los"), indices, default=0.0, dtype=torch.float32),
            "nf_label": _vector_optional(handle, keys.get("nf"), indices, default=0.0, dtype=torch.float32),
            "traj_nlos_label": _vector_optional(handle, keys.get("traj_nlos"), indices, default=0.0, dtype=torch.float32),
            "mode_idx": _vector_optional(handle, keys.get("mode"), indices, default=-1, dtype=torch.int64),
        }
        return sample

    def metadata(self) -> dict[str, Any]:
        return {
            "target_schema": self.target_schema,
            "target_schema_aliases": list(self.target_schema_aliases),
            "codebook_shape": list(self.codebook_shape),
            "flatten_order": self.flatten_order,
            "num_beam_classes": int(self.codebook_metadata["num_beam_classes"]),
            "codebook_path": self.codebook_metadata.get("path"),
            "codebook_fingerprint": self.codebook_metadata.get("fingerprint"),
        }


def _load_or_build_index(
    *,
    paths,
    index_path: str | None,
    split_metadata_path: str | None,
    split: str,
    raw_root: str | None,
    channel_path: str | None,
    image_path: str | None,
    lidar_path: str | None,
    split_mode: str,
    train_cities,
    val_cities,
    test_cities,
    split_ratios,
    split_seed: int,
    seq_len: int,
    num_pred: int,
) -> tuple[tuple[SampleRow, ...], dict[str, Any]]:
    prefix = f"multimodal_nf_seq{int(seq_len)}_pred{int(num_pred)}"
    default_index_path = paths.cache_dir / f"{prefix}_all.csv"
    default_split_metadata_path = paths.cache_dir / f"{prefix}_split_metadata.json"
    if index_path is None and default_index_path.exists():
        index_path = str(default_index_path)
        if split_metadata_path is None and default_split_metadata_path.exists():
            split_metadata_path = str(default_split_metadata_path)
    if index_path:
        index = load_multimodal_nf_index(resolve_path(index_path), split=split)
        metadata = _read_json(split_metadata_path) if split_metadata_path else dict(index.metadata)
        return index.rows, metadata
    rows, metadata = build_multimodal_nf_rows(
        data_root=paths.data_root,
        raw_root=raw_root,
        channel_path=channel_path,
        image_path=image_path,
        lidar_path=lidar_path,
        split_mode=split_mode,
        train_cities=train_cities,
        val_cities=val_cities,
        test_cities=test_cities,
        split_ratios=split_ratios,
        split_seed=split_seed,
        seq_len=seq_len,
        num_pred=num_pred,
    )
    selected = _normalize_split(split)
    return tuple(row for row in rows if _normalize_split(row.split) == selected), metadata


def _resolve_codebook_metadata(
    *,
    paths,
    codebook_path: str | None,
    codebook_shape,
    codebook_profile: str | None,
    codebook_metadata: dict[str, Any] | None,
    flatten_order: str,
) -> dict[str, Any]:
    if codebook_metadata:
        metadata = dict(codebook_metadata)
        metadata.setdefault("flatten_order", flatten_order)
        metadata.setdefault("num_beam_classes", int(np.prod(metadata["shape"])))
        return metadata
    resolved_path = resolve_path(codebook_path) if codebook_path is not None else _discover_codebook(paths)
    if resolved_path is None and codebook_shape is None and codebook_profile is None:
        raise ValueError(
            "Multimodal-NF near-field target requires codebook_path, codebook_shape, "
            "codebook_profile, or train-provided codebook_metadata."
        )
    return parse_codebook_metadata(
        resolved_path,
        codebook_shape=codebook_shape,
        profile=codebook_profile,
        flatten_order=flatten_order,
    )


def _discover_codebook(paths) -> Path | None:
    for root in (paths.codebook_root, paths.data_root):
        if not Path(root).exists():
            continue
        for suffix in ("*.pkl", "*.pickle", "*.json", "*.npz", "*.npy"):
            matches = sorted(Path(root).rglob(suffix))
            if matches:
                return matches[0]
    return None


def _apply_portion(rows: tuple[SampleRow, ...], portion: float) -> tuple[SampleRow, ...]:
    value = float(portion)
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"Multimodal-NF portion must be in (0, 1], got {portion}.")
    if value >= 1.0:
        return rows
    keep = max(1, int(len(rows) * value))
    return rows[:keep]


def _portion_for_split(
    split: str,
    *,
    portion: float,
    train_portion: float | None,
    eval_portion: float | None,
    val_portion: float | None,
    validation_portion: float | None,
    test_portion: float | None,
) -> float:
    selected = _normalize_split(split)
    if selected == "train" and train_portion is not None:
        return float(train_portion)
    if selected == "validation":
        for candidate in (validation_portion, val_portion, eval_portion):
            if candidate is not None:
                return float(candidate)
    if selected == "test":
        for candidate in (test_portion, eval_portion):
            if candidate is not None:
                return float(candidate)
    return float(portion)


def _read_hdf5_rows(dataset, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.asarray(dataset[[]])
    if _is_contiguous_increasing(indices):
        start = int(indices[0])
        return np.asarray(dataset[start : start + len(indices)])
    return np.asarray(dataset[indices])


def _is_contiguous_increasing(indices: list[int]) -> bool:
    return all(int(indices[idx]) == int(indices[0]) + idx for idx in range(len(indices)))


def _resource_path_for_modality(row: SampleRow, modality: str) -> str:
    if modality == "image":
        path = row.resource_refs.get("image_path") or row.resource_refs.get("channel_path")
    elif modality == "lidar":
        path = row.resource_refs.get("lidar_path") or row.resource_refs.get("channel_path")
    else:
        path = row.resource_refs.get("channel_path")
    if not path:
        raise FileNotFoundError(
            f"Multimodal-NF sample {row.sample_id} has no HDF5 resource for enabled modality '{modality}'."
        )
    resolved = Path(str(path))
    if not resolved.exists():
        raise FileNotFoundError(
            f"Multimodal-NF enabled modality '{modality}' requires HDF5 file {resolved} "
            f"for sample {row.sample_id}, but it is missing."
        )
    return str(resolved)


def _dataset_paths(handle) -> list[str]:
    h5py = _require_h5py("Multimodal-NF dataset traversal")
    paths = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            paths.append(name)

    handle.visititems(visitor)
    return paths


def _coerce_csi(value: np.ndarray, *, policy: str, subcarrier_index: int | None) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        array = np.stack([array.real, array.imag], axis=-1)
    if array.ndim == 4 and array.shape[-1] == 2:
        if policy in {"single", "select"}:
            index = 0 if subcarrier_index is None else int(subcarrier_index)
            if index < 0 or index >= int(array.shape[2]):
                raise ValueError(f"csi_subcarrier_index {index} is outside CSI shape {array.shape}.")
            array = array[:, :, index : index + 1, :]
        elif policy not in {"all", "none"}:
            raise ValueError("csi_subcarrier_policy must be one of all, single, or select.")
        return np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = np.stack([array, np.zeros_like(array)], axis=-1).reshape(array.shape[0], 1, 2)
    if array.ndim == 2:
        array = np.stack([array, np.zeros_like(array)], axis=-1)
    if array.ndim == 3 and array.shape[-1] == 2:
        pass
    elif array.ndim == 3:
        array = np.stack([array, np.zeros_like(array)], axis=-1)
    if array.ndim != 3 or array.shape[-1] != 2:
        return np.asarray(array, dtype=np.float32)
    if policy in {"single", "select"}:
        index = 0 if subcarrier_index is None else int(subcarrier_index)
        if index < 0 or index >= int(array.shape[1]):
            raise ValueError(f"csi_subcarrier_index {index} is outside CSI shape {array.shape}.")
        array = array[:, index : index + 1, :]
    elif policy not in {"all", "none"}:
        raise ValueError("csi_subcarrier_policy must be one of all, single, or select.")
    return np.asarray(array, dtype=np.float32)


def _scalar_optional(handle, key: str | None, index: int, *, default: float) -> float:
    if not key or key not in handle:
        return float(default)
    return float(np.asarray(handle[key][index]).reshape(()))


def _vector_optional(handle, key: str | None, indices: list[int], *, default: float, dtype: torch.dtype) -> torch.Tensor:
    if not key or key not in handle:
        return torch.full((len(indices),), float(default), dtype=dtype)
    values = _read_hdf5_rows(handle[key], indices).reshape(-1)
    if dtype == torch.int64:
        return torch.tensor(values.astype(np.int64), dtype=dtype)
    return torch.tensor(values.astype(np.float32), dtype=dtype)


def _shape_error(row: SampleRow, modality: str, profile: str, shape: Any, expected: str) -> None:
    raise ValueError(
        "Multimodal-NF shape validation failed: "
        f"family=MultimodalNF modality={modality} profile={profile} sample_id={row.sample_id} "
        f"actual_shape={tuple(shape)} {expected}."
    )


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    file_path = resolve_path(path)
    if not file_path.exists():
        return {}
    import json

    return json.loads(file_path.read_text(encoding="utf-8"))


def _normalize_split(split: str) -> str:
    key = str(split).strip().lower()
    return {"val": "validation", "valid": "validation"}.get(key, key)


def _require_h5py(context: str):
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"{context} requires h5py in the kd_mm_beam environment.") from exc
    return h5py


__all__ = [
    "MultimodalNFDataset",
    "MultimodalNFTargetProvider",
]
