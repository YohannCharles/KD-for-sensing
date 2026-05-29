from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math

import numpy as np

from kd_sensing.data.mmw.radio_semantic import RadioSemanticLabelBuilder


PATH_SEMANTIC_CONFIG_VERSION = "path_semantic_descriptor_v1"
SUPPORTED_PATH_SEMANTIC_MODES = {"kmeans_path_descriptor", "rule_path_pattern", "radio_power", "coarse"}
PATH_INTERNAL_KEYS = (
    "a",
    "tau",
    "aod_azimuth",
    "aod_zenith",
    "aoa_azimuth",
    "aoa_zenith",
    "valid_mask",
    "tx_pose",
    "rx_pose",
)

DEFAULT_PATH_FIELD_CANDIDATES: dict[str, tuple[str, ...]] = {
    "a": ("a", "array", "gain", "gains", "path_gain", "path_gains", "alpha", "alphas", "complex_gain"),
    "tau": ("tau", "delay", "delays", "path_delay", "path_delays", "toa", "time_delay"),
    "aod_azimuth": (
        "aod_azimuth",
        "aod_phi",
        "aod",
        "departure_azimuth",
        "azimuth_of_departure",
        "tx_angle",
        "phi_t",
        "glob_phi_t",
    ),
    "aod_zenith": (
        "aod_zenith",
        "aod_theta",
        "departure_zenith",
        "zenith_of_departure",
        "theta_t",
        "glob_theta_t",
    ),
    "aoa_azimuth": (
        "aoa_azimuth",
        "aoa_phi",
        "aoa",
        "arrival_azimuth",
        "azimuth_of_arrival",
        "rx_angle",
        "phi_r",
        "glob_phi_r",
    ),
    "aoa_zenith": (
        "aoa_zenith",
        "aoa_theta",
        "arrival_zenith",
        "zenith_of_arrival",
        "theta_r",
        "glob_theta_r",
    ),
    "valid_mask": ("valid_mask", "path_mask", "mask", "valid", "is_valid", "valid_paths"),
    "tx_pose": ("tx_pose", "tx_position", "transmitter_pose", "cav_pose"),
    "rx_pose": ("rx_pose", "rx_position", "receiver_pose", "rsu_pose"),
}

FIELD_MAP_ALIASES = {
    "gain": "a",
    "path_gain": "a",
    "delay": "tau",
    "path_delay": "tau",
    "mask": "valid_mask",
    "pose_tx": "tx_pose",
    "pose_rx": "rx_pose",
}

BASE_DESCRIPTOR_NAMES = (
    "log_total_power",
    "dominant_path_ratio",
    "top3_path_mass",
    "path_entropy",
    "effective_num_paths",
    "mean_excess_delay",
    "rms_delay_spread",
    "dominant_aod_azimuth_sin",
    "dominant_aod_azimuth_cos",
    "dominant_aoa_azimuth_sin",
    "dominant_aoa_azimuth_cos",
    "weighted_aod_angular_spread",
    "weighted_aoa_angular_spread",
    "los_like_score",
)
ZENITH_DESCRIPTOR_NAMES = ("weighted_aod_zenith_spread", "weighted_aoa_zenith_spread")


@dataclass(frozen=True)
class PathDescriptorResult:
    descriptor: np.ndarray | None
    diagnostics: dict[str, Any]

    @property
    def available(self) -> bool:
        return self.descriptor is not None


@dataclass(frozen=True)
class PathSemanticLabelResult:
    label: int | None
    diagnostics: dict[str, Any]

    @property
    def available(self) -> bool:
        return self.label is not None


def load_path_payload(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path)
    diagnostics: dict[str, Any] = {"path": str(source), "fields": [], "shape": None}
    suffix = source.suffix.lower()
    if suffix == ".npz":
        with np.load(source, allow_pickle=True) as payload:
            data = {key: payload[key] for key in payload.files}
    elif suffix == ".npy":
        raw = np.load(source, allow_pickle=True)
        diagnostics["shape"] = tuple(int(item) for item in raw.shape)
        if isinstance(raw, np.ndarray) and raw.shape == () and isinstance(raw.item(), dict):
            data = dict(raw.item())
        elif isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1 and isinstance(raw.reshape(-1)[0], dict):
            data = dict(raw.reshape(-1)[0])
        else:
            data = {"array": raw}
    elif suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        data = payload if isinstance(payload, dict) else {"array": payload}
    elif suffix in {".yaml", ".yml"}:
        from kd_sensing.config.io import safe_load_yaml

        payload = safe_load_yaml(source.read_text(encoding="utf-8"))
        data = payload if isinstance(payload, dict) else {"array": payload}
    elif suffix in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("h5py is required to inspect HDF5 path files.") from exc
        data = {}
        with h5py.File(source, "r") as handle:
            for key in handle.keys():
                value = handle[key]
                if hasattr(value, "shape"):
                    data[key] = value[()]
    elif suffix == ".mat":
        try:
            from scipy.io import loadmat
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ValueError("scipy is required to inspect MAT path files.") from exc
        data = {key: value for key, value in loadmat(source).items() if not key.startswith("__")}
    else:
        raise ValueError(f"Unsupported path file extension for {source}; expected npy/npz/json/yaml/h5/mat.")
    diagnostics["fields"] = sorted(str(key) for key in data.keys())
    if not data:
        raise ValueError(f"Path file {source} contains no fields.")
    return data, diagnostics


def resolve_path_field_map(field_map: Mapping[str, Any] | None = None) -> dict[str, Any]:
    resolved: dict[str, Any] = {key: list(value) for key, value in DEFAULT_PATH_FIELD_CANDIDATES.items()}
    if not field_map:
        return resolved
    payload = dict(field_map.get("path", field_map)) if isinstance(field_map.get("path"), Mapping) else dict(field_map)
    for raw_key, raw_value in payload.items():
        key = FIELD_MAP_ALIASES.get(str(raw_key), str(raw_key))
        if key == "path_axis":
            resolved[key] = raw_value
            continue
        if key not in PATH_INTERNAL_KEYS:
            continue
        values = raw_value if isinstance(raw_value, (list, tuple)) else [raw_value]
        resolved[key] = [str(item) for item in values if str(item)]
    return resolved


def map_path_fields(
    payload: Mapping[str, Any],
    field_map: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = resolve_path_field_map(field_map)
    params: dict[str, Any] = {}
    used: dict[str, str] = {}
    missing: list[str] = []
    for internal_key in PATH_INTERNAL_KEYS:
        field_name = _first_present(payload, candidates.get(internal_key, ()))
        if field_name is None:
            missing.append(internal_key)
            continue
        params[internal_key] = payload[field_name]
        used[internal_key] = field_name
    if "path_axis" in candidates:
        params["path_axis"] = candidates["path_axis"]
        used["path_axis"] = str(candidates["path_axis"])
    summaries = {
        internal_key: summarize_array(value)
        for internal_key, value in params.items()
        if internal_key != "path_axis"
    }
    diagnostics = {
        "field_map_used": used,
        "missing_internal_keys": missing,
        "field_summaries": summaries,
        "available": "a" in params,
    }
    if "a" not in params:
        diagnostics["unavailable_reason"] = "missing_path_gain"
    return params, diagnostics


def summarize_array(value: Any) -> dict[str, Any]:
    try:
        array = np.asarray(value)
    except Exception:
        return {"shape": None, "dtype": type(value).__name__}
    return {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
    }


def summarize_path_payload(
    payload: Mapping[str, Any],
    field_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    params, diagnostics = map_path_fields(payload, field_map)
    path_axis = infer_path_axis(params) if "a" in params else None
    diagnostics["path_axis"] = path_axis
    diagnostics["path_axis_inferred"] = path_axis is not None and "path_axis" not in params
    return diagnostics


@dataclass
class PathFeatureBuilder:
    field_map: Mapping[str, Any] | None = None
    path_axis: int | None = None
    use_zenith: bool = False
    normalize_output: bool = False
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    eps: float = 1e-12

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | bool | None = None) -> "PathFeatureBuilder":
        payload = dict(config) if isinstance(config, Mapping) else {}
        field_map = payload.get("field_map") or payload.get("data.field_map") or payload.get("path_field_map")
        stats = payload.get("normalization") if isinstance(payload.get("normalization"), Mapping) else {}
        return cls(
            field_map=field_map,
            path_axis=int(payload["path_axis"]) if payload.get("path_axis") is not None else None,
            use_zenith=bool(payload.get("use_zenith", False)),
            normalize_output=bool(payload.get("normalize_output", False)),
            mean=None if stats.get("mean") is None else np.asarray(stats.get("mean"), dtype=np.float64),
            std=None if stats.get("std") is None else np.asarray(stats.get("std"), dtype=np.float64),
        )

    @property
    def descriptor_names(self) -> tuple[str, ...]:
        return BASE_DESCRIPTOR_NAMES + (ZENITH_DESCRIPTOR_NAMES if self.use_zenith else ())

    @property
    def descriptor_dim(self) -> int:
        return len(self.descriptor_names)

    def build_descriptor(self, path_params: Mapping[str, Any] | None) -> PathDescriptorResult:
        if not path_params:
            return PathDescriptorResult(None, _unavailable("path_params_missing"))
        params = dict(path_params)
        if "a" not in params:
            params, map_diag = map_path_fields(path_params, self.field_map)
        else:
            map_diag = {
                "field_map_used": {key: key for key in params.keys() if key in PATH_INTERNAL_KEYS},
                "field_summaries": {key: summarize_array(value) for key, value in params.items() if key != "path_axis"},
            }
        required = ("a", "tau", "aod_azimuth", "aoa_azimuth")
        missing = [key for key in required if key not in params]
        if missing:
            return PathDescriptorResult(
                None,
                _unavailable("missing_required_path_fields", missing_required_fields=missing) | map_diag,
            )
        try:
            power, path_axis = path_power(params["a"], params.get("valid_mask"), path_axis=self.path_axis or params.get("path_axis"))
        except Exception as exc:
            return PathDescriptorResult(None, _unavailable(f"path_power_failed:{exc}") | map_diag)
        valid = power > 0
        if "valid_mask" in params:
            valid &= _path_mask(params["valid_mask"], target_len=power.size)
        if not np.any(valid):
            return PathDescriptorResult(None, _unavailable("no_valid_path") | map_diag)
        power = np.where(valid, power, 0.0).astype(np.float64)
        total_power = float(power.sum())
        if total_power <= self.eps or not np.isfinite(total_power):
            return PathDescriptorResult(None, _unavailable("non_positive_path_power") | map_diag)
        q = power / max(total_power, self.eps)
        tau = _path_vector(params["tau"], target_len=power.size)
        aod = _angles_to_radians(_path_vector(params["aod_azimuth"], target_len=power.size))
        aoa = _angles_to_radians(_path_vector(params["aoa_azimuth"], target_len=power.size))
        if tau is None or aod is None or aoa is None:
            return PathDescriptorResult(None, _unavailable("invalid_required_path_field_shape") | map_diag)
        tau = np.nan_to_num(tau.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        min_delay = float(np.min(tau[valid]))
        excess = np.where(valid, tau - min_delay, 0.0)
        mean_excess = float(np.sum(q * excess))
        rms_delay = float(np.sqrt(max(np.sum(q * (excess - mean_excess) ** 2), 0.0)))
        valid_count = int(valid.sum())
        q_valid = q[valid]
        entropy = float(-(q_valid * np.log(q_valid + self.eps)).sum() / max(math.log(max(valid_count, 2)), self.eps))
        dominant_ratio = float(np.max(q_valid))
        top3_mass = float(np.sort(q_valid)[-min(3, q_valid.size) :].sum())
        effective_num = float(1.0 / max(np.sum(q_valid**2), self.eps))
        aod_sin, aod_cos, aod_spread = circular_weighted_stats(aod, q)
        aoa_sin, aoa_cos, aoa_spread = circular_weighted_stats(aoa, q)
        los_like = float(dominant_ratio / (1.0 + rms_delay))
        values = [
            float(np.log(total_power + self.eps)),
            dominant_ratio,
            top3_mass,
            entropy,
            effective_num,
            mean_excess,
            rms_delay,
            aod_sin,
            aod_cos,
            aoa_sin,
            aoa_cos,
            aod_spread,
            aoa_spread,
            los_like,
        ]
        if self.use_zenith:
            values.append(_optional_spread(params.get("aod_zenith"), q, target_len=power.size))
            values.append(_optional_spread(params.get("aoa_zenith"), q, target_len=power.size))
        descriptor = np.asarray(values, dtype=np.float32)
        if self.normalize_output:
            descriptor = self._normalize(descriptor)
        diagnostics = {
            "available": True,
            "descriptor_names": list(self.descriptor_names),
            "descriptor_dim": int(descriptor.size),
            "path_axis": int(path_axis),
            "valid_path_count": valid_count,
            "total_path_count": int(power.size),
            "total_power": total_power,
            "dominant_path_ratio": dominant_ratio,
            "q_p": q.astype(np.float32).tolist(),
        }
        diagnostics.update(map_diag)
        return PathDescriptorResult(descriptor, diagnostics)

    def _normalize(self, descriptor: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return descriptor.astype(np.float32)
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        std = np.asarray(self.std, dtype=np.float64).reshape(-1)
        if mean.size != descriptor.size or std.size != descriptor.size:
            raise ValueError(
                f"path descriptor normalization stats dim {mean.size}/{std.size} do not match descriptor dim {descriptor.size}."
            )
        return ((descriptor.astype(np.float64) - mean) / np.maximum(std, 1e-12)).astype(np.float32)


@dataclass
class PathSemanticLabelBuilder:
    mode: str = "kmeans_path_descriptor"
    num_path_classes: int = 24
    group_size: int = 8
    seed: int = 0
    fallback_if_missing: str | None = "radio_power"
    artifact: dict[str, Any] | None = None
    radio_builder: RadioSemanticLabelBuilder | None = None
    config_version: str = PATH_SEMANTIC_CONFIG_VERSION

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | bool | None = None, **overrides: Any) -> "PathSemanticLabelBuilder":
        payload = dict(config) if isinstance(config, Mapping) else {}
        payload.update({key: value for key, value in overrides.items() if value is not None})
        mode = str(payload.get("mode", "kmeans_path_descriptor")).strip().lower()
        if mode not in SUPPORTED_PATH_SEMANTIC_MODES:
            raise ValueError(f"Unsupported path semantic mode '{mode}'. Available modes: {sorted(SUPPORTED_PATH_SEMANTIC_MODES)}.")
        artifact = payload.get("artifact")
        artifact_path = payload.get("artifact_path") or payload.get("kmeans_artifact_path")
        if artifact is None and artifact_path:
            artifact = load_path_semantic_artifact(artifact_path)
        radio_cfg = payload.get("radio_semantic") if isinstance(payload.get("radio_semantic"), Mapping) else {}
        return cls(
            mode=mode,
            num_path_classes=int(payload.get("num_path_classes", payload.get("num_classes", 24))),
            group_size=int(payload.get("group_size", 8)),
            seed=int(payload.get("seed", 0)),
            fallback_if_missing=None
            if payload.get("fallback_if_missing") in {None, "none", False}
            else str(payload.get("fallback_if_missing", "radio_power")).strip().lower(),
            artifact=dict(artifact) if isinstance(artifact, Mapping) else None,
            radio_builder=RadioSemanticLabelBuilder.from_config(radio_cfg, group_size=int(payload.get("group_size", 8))),
        )

    def metadata(self) -> dict[str, Any]:
        artifact = self.artifact or {}
        return {
            "path_semantic_mode": self.mode,
            "path_semantic_config_version": self.config_version,
            "num_path_classes": int(self.num_path_classes),
            "fallback_if_missing": self.fallback_if_missing,
            "descriptor_dim": artifact.get("descriptor_dim"),
            "fit_on_source_only": True,
        }

    def fit(
        self,
        descriptors: Iterable[Iterable[float] | np.ndarray],
        *,
        source_domain: Mapping[str, Any] | None = None,
        artifact_path: str | Path | None = None,
    ) -> dict[str, Any]:
        matrix = _descriptor_matrix(descriptors)
        if matrix.size == 0 or matrix.ndim != 2:
            raise ValueError("Path KMeans fit requires a non-empty [N, D] descriptor matrix.")
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)
        scaled = (matrix - mean) / std
        class_count = min(int(self.num_path_classes), int(matrix.shape[0]))
        centers, labels = _kmeans_numpy(scaled, class_count, seed=int(self.seed))
        artifact = {
            "version": "path_semantic_kmeans_v1",
            "mode": "kmeans_path_descriptor",
            "num_path_classes": int(self.num_path_classes),
            "fit_class_count": int(class_count),
            "descriptor_dim": int(matrix.shape[1]),
            "scaler_mean": mean.astype(float).tolist(),
            "scaler_std": std.astype(float).tolist(),
            "kmeans_centers": centers.astype(float).tolist(),
            "seed": int(self.seed),
            "source_domain": dict(source_domain or {}),
            "fit_sample_count": int(matrix.shape[0]),
            "class_counts": np.bincount(labels, minlength=class_count).astype(int).tolist(),
            "config_version": self.config_version,
        }
        self.artifact = artifact
        if artifact_path is not None:
            save_path_semantic_artifact(artifact, artifact_path)
        return artifact

    def derive(
        self,
        *,
        path_descriptor: Iterable[float] | np.ndarray | None = None,
        beam_label: int | None = None,
        beam_power: Iterable[float] | np.ndarray | None = None,
        radio_semantic_label: int | None = None,
    ) -> PathSemanticLabelResult:
        base = self.metadata() | {"available": False}
        if self.mode == "kmeans_path_descriptor":
            if path_descriptor is not None and self.artifact:
                return self._kmeans_result(path_descriptor, base=base)
            return self._fallback(
                beam_label=beam_label,
                beam_power=beam_power,
                radio_semantic_label=radio_semantic_label,
                base=base,
                reason="path_descriptor_or_artifact_missing",
            )
        if self.mode == "rule_path_pattern":
            if path_descriptor is None:
                return self._fallback(
                    beam_label=beam_label,
                    beam_power=beam_power,
                    radio_semantic_label=radio_semantic_label,
                    base=base,
                    reason="path_descriptor_missing",
                )
            return self._rule_result(path_descriptor, base=base)
        if self.mode == "radio_power":
            return self._radio_result(beam_power=beam_power, beam_label=beam_label, base=base)
        return self._coarse_result(beam_label=beam_label, base=base)

    def _kmeans_result(self, descriptor: Iterable[float] | np.ndarray, *, base: dict[str, Any]) -> PathSemanticLabelResult:
        vector = np.asarray(descriptor, dtype=np.float64).reshape(-1)
        artifact = self.artifact or {}
        dim = int(artifact.get("descriptor_dim", vector.size))
        if vector.size != dim:
            return PathSemanticLabelResult(None, base | {"unavailable_reason": f"descriptor_dim_mismatch:{vector.size}!={dim}"})
        mean = np.asarray(artifact.get("scaler_mean"), dtype=np.float64).reshape(-1)
        std = np.asarray(artifact.get("scaler_std"), dtype=np.float64).reshape(-1)
        centers = np.asarray(artifact.get("kmeans_centers"), dtype=np.float64)
        if mean.size != vector.size or std.size != vector.size or centers.ndim != 2 or centers.shape[1] != vector.size:
            return PathSemanticLabelResult(None, base | {"unavailable_reason": "invalid_path_kmeans_artifact"})
        scaled = (vector - mean) / np.maximum(std, 1e-12)
        distances = ((centers - scaled.reshape(1, -1)) ** 2).sum(axis=1)
        label = int(np.argmin(distances))
        return PathSemanticLabelResult(
            label,
            base
            | {
                "available": True,
                "path_semantic_mode": "kmeans_path_descriptor",
                "path_semantic_label": label,
                "kmeans_distance": float(distances[label]),
            },
        )

    def _rule_result(self, descriptor: Iterable[float] | np.ndarray, *, base: dict[str, Any]) -> PathSemanticLabelResult:
        vector = np.asarray(descriptor, dtype=np.float64).reshape(-1)
        if vector.size < len(BASE_DESCRIPTOR_NAMES):
            return PathSemanticLabelResult(None, base | {"unavailable_reason": "descriptor_dim_too_small_for_rule"})
        index = {name: idx for idx, name in enumerate(BASE_DESCRIPTOR_NAMES)}
        dominant = float(vector[index["dominant_path_ratio"]])
        delay_spread = float(vector[index["rms_delay_spread"]])
        angular_spread = float(max(vector[index["weighted_aod_angular_spread"]], vector[index["weighted_aoa_angular_spread"]]))
        dominant_bin = 0 if dominant >= 0.7 else 1 if dominant >= 0.35 else 2
        delay_bin = 0 if delay_spread < 0.5 else 1
        angle_bin = 0 if angular_spread < 0.75 else 1
        label = int((dominant_bin * 4 + delay_bin * 2 + angle_bin) % max(int(self.num_path_classes), 1))
        return PathSemanticLabelResult(
            label,
            base
            | {
                "available": True,
                "path_semantic_mode": "rule_path_pattern",
                "dominant_bin": dominant_bin,
                "delay_bin": delay_bin,
                "angle_bin": angle_bin,
                "path_semantic_label": label,
            },
        )

    def _radio_result(
        self,
        *,
        beam_power: Iterable[float] | np.ndarray | None,
        beam_label: int | None,
        base: dict[str, Any],
    ) -> PathSemanticLabelResult:
        builder = self.radio_builder or RadioSemanticLabelBuilder(group_size=self.group_size)
        result = builder.derive(beam_power=beam_power, beam_label=beam_label, input_source="path_semantic_radio_power")
        if result.available:
            return PathSemanticLabelResult(
                int(result.label),
                base
                | result.diagnostics
                | {
                    "available": True,
                    "path_semantic_mode": "radio_power",
                    "path_semantic_label": int(result.label),
                },
            )
        return PathSemanticLabelResult(None, base | {"unavailable_reason": result.diagnostics.get("unavailable_reason", "radio_power_unavailable")})

    def _coarse_result(self, *, beam_label: int | None, base: dict[str, Any]) -> PathSemanticLabelResult:
        try:
            beam = int(beam_label) if beam_label is not None else -1
        except (TypeError, ValueError):
            beam = -1
        if beam < 0:
            return PathSemanticLabelResult(None, base | {"unavailable_reason": "missing_beam_label"})
        label = int(beam // max(int(self.group_size), 1))
        return PathSemanticLabelResult(
            label,
            base | {"available": True, "path_semantic_mode": "coarse", "path_semantic_label": label},
        )

    def _fallback(
        self,
        *,
        beam_label: int | None,
        beam_power: Iterable[float] | np.ndarray | None,
        radio_semantic_label: int | None,
        base: dict[str, Any],
        reason: str,
    ) -> PathSemanticLabelResult:
        if self.fallback_if_missing == "radio_power":
            result = self._radio_result(beam_power=beam_power, beam_label=beam_label, base=base)
        elif self.fallback_if_missing == "coarse":
            result = self._coarse_result(beam_label=beam_label, base=base)
        elif self.fallback_if_missing == "radio_semantic" and radio_semantic_label is not None:
            try:
                label = int(radio_semantic_label)
            except (TypeError, ValueError):
                label = -1
            result = (
                PathSemanticLabelResult(label, base | {"available": True, "path_semantic_mode": "radio_semantic", "path_semantic_label": label})
                if label >= 0
                else PathSemanticLabelResult(None, base | {"unavailable_reason": "radio_semantic_label_missing"})
            )
        else:
            result = PathSemanticLabelResult(None, base | {"unavailable_reason": reason})
        if result.available:
            return PathSemanticLabelResult(result.label, result.diagnostics | {"fallback_reason": reason})
        return result


def save_path_semantic_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_jsonable(artifact), indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_path_semantic_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def path_power(
    gain: Any,
    valid_mask: Any | None = None,
    *,
    path_axis: int | str | None = None,
) -> tuple[np.ndarray, int]:
    array = np.asarray(gain)
    if array.size == 0:
        raise ValueError("empty_gain")
    axis = infer_path_axis({"a": array, "valid_mask": valid_mask, "path_axis": path_axis})
    if axis is None:
        axis = 0
    moved = np.moveaxis(array, int(axis), 0)
    power = np.abs(moved.astype(np.complex128)) ** 2
    if power.ndim > 1:
        power = power.reshape(power.shape[0], -1).sum(axis=1)
    else:
        power = power.reshape(-1)
    power = np.nan_to_num(power.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        mask = _path_mask(valid_mask, target_len=power.size)
        power = np.where(mask, power, 0.0)
    return power, int(axis)


def infer_path_axis(params: Mapping[str, Any]) -> int | None:
    explicit = params.get("path_axis")
    if explicit is not None and str(explicit) != "":
        return int(explicit)
    if "a" not in params:
        return None
    array = np.asarray(params["a"])
    if array.ndim == 0:
        return 0
    lengths = []
    for key in ("tau", "aod_azimuth", "aoa_azimuth", "valid_mask"):
        if key not in params or params[key] is None:
            continue
        try:
            value = np.asarray(params[key])
        except Exception:
            continue
        if value.ndim == 0:
            continue
        lengths.append(int(value.shape[0] if value.ndim == 1 else max(value.shape)))
    for length in lengths:
        for axis, dim in enumerate(array.shape):
            if int(dim) == length:
                return int(axis)
    return 0


def circular_weighted_stats(angles: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    angle = np.asarray(angles, dtype=np.float64).reshape(-1)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    count = min(angle.size, weight.size)
    if count == 0:
        return 0.0, 1.0, 0.0
    angle = angle[:count]
    weight = weight[:count]
    total = float(weight.sum())
    if total <= 1e-12:
        return 0.0, 1.0, 0.0
    sin_mean = float(np.sum(weight * np.sin(angle)) / total)
    cos_mean = float(np.sum(weight * np.cos(angle)) / total)
    resultant = math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    norm = max(resultant, 1e-12)
    spread = math.sqrt(max(-2.0 * math.log(max(min(resultant, 1.0), 1e-12)), 0.0))
    return float(sin_mean / norm), float(cos_mean / norm), float(spread)


def _optional_spread(value: Any | None, weights: np.ndarray, *, target_len: int) -> float:
    vector = _path_vector(value, target_len=target_len)
    if vector is None:
        return 0.0
    _, _, spread = circular_weighted_stats(_angles_to_radians(vector), weights)
    return float(spread)


def _path_vector(value: Any, *, target_len: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.size == 0:
        return None
    if array.ndim == 1:
        vector = array.astype(np.float64)
    else:
        reshaped = array.reshape(array.shape[0], -1) if array.shape[0] == target_len else np.moveaxis(array, -1, 0).reshape(array.shape[-1], -1)
        vector = np.nanmean(reshaped.astype(np.float64), axis=1)
    if vector.size < target_len:
        return None
    return vector[:target_len]


def _path_mask(value: Any, *, target_len: int) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return np.ones(int(target_len), dtype=bool)
    if array.ndim == 1:
        mask = array.astype(bool)
    elif array.shape[0] == target_len:
        mask = array.reshape(array.shape[0], -1).astype(bool).any(axis=1)
    else:
        moved = np.moveaxis(array, -1, 0)
        mask = moved.reshape(moved.shape[0], -1).astype(bool).any(axis=1)
    if mask.size < target_len:
        padded = np.zeros(int(target_len), dtype=bool)
        padded[: mask.size] = mask
        return padded
    return mask[:target_len].astype(bool)


def _angles_to_radians(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size and np.nanmax(np.abs(finite)) > math.pi + 1e-3:
        array = np.deg2rad(array)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _first_present(payload: Mapping[str, Any], candidates: Iterable[str]) -> str | None:
    lowered = {str(key).lower(): str(key) for key in payload.keys()}
    for candidate in candidates:
        text = str(candidate)
        if text in payload:
            return text
        if text.lower() in lowered:
            return lowered[text.lower()]
    return None


def _descriptor_matrix(descriptors: Iterable[Iterable[float] | np.ndarray]) -> np.ndarray:
    rows = [np.asarray(row, dtype=np.float64).reshape(-1) for row in descriptors if row is not None]
    if not rows:
        return np.empty((0, 0), dtype=np.float64)
    dim = rows[0].size
    if any(row.size != dim for row in rows):
        raise ValueError("All path descriptors must have the same dimension.")
    matrix = np.vstack(rows)
    if not np.isfinite(matrix).all():
        raise ValueError("Path descriptor matrix contains NaN or Inf.")
    return matrix


def _kmeans_numpy(matrix: np.ndarray, k: int, *, seed: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    if matrix.shape[0] < int(k):
        raise ValueError("KMeans class count cannot exceed fit sample count.")
    first = int(rng.integers(0, matrix.shape[0]))
    centers = [matrix[first]]
    while len(centers) < int(k):
        distances = np.min(((matrix[:, None, :] - np.asarray(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        if float(distances.sum()) <= 1e-12:
            candidate = len(centers) % matrix.shape[0]
        else:
            probs = distances / distances.sum()
            candidate = int(rng.choice(matrix.shape[0], p=probs))
        centers.append(matrix[candidate])
    centers_array = np.asarray(centers, dtype=np.float64)
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    for _ in range(int(max_iter)):
        distances = ((matrix[:, None, :] - centers_array[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for class_index in range(int(k)):
            mask = labels == class_index
            if np.any(mask):
                centers_array[class_index] = matrix[mask].mean(axis=0)
    return centers_array, labels


def _unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    return {"available": False, "unavailable_reason": reason, **extra}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "BASE_DESCRIPTOR_NAMES",
    "DEFAULT_PATH_FIELD_CANDIDATES",
    "PATH_INTERNAL_KEYS",
    "PATH_SEMANTIC_CONFIG_VERSION",
    "SUPPORTED_PATH_SEMANTIC_MODES",
    "PathDescriptorResult",
    "PathFeatureBuilder",
    "PathSemanticLabelBuilder",
    "PathSemanticLabelResult",
    "circular_weighted_stats",
    "infer_path_axis",
    "load_path_payload",
    "load_path_semantic_artifact",
    "map_path_fields",
    "path_power",
    "resolve_path_field_map",
    "save_path_semantic_artifact",
    "summarize_array",
    "summarize_path_payload",
]
