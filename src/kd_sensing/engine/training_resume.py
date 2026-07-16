"""Versioned resume contracts and runtime-state helpers.

This module intentionally has no trainer dependency so resume validation can run
before a target run writes mutable artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from kd_sensing.utils.checkpoint import (
    CheckpointLoadError,
    load_torch_payload,
    validate_checkpoint_publication,
)
from kd_sensing.utils.paths import resolve_path


CHECKPOINT_SCHEMA_VERSION = 1
RUNTIME_STATE_SCHEMA_VERSION = 1
RESUME_CONTRACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResumePlan:
    """Read-only result of validating a requested resume checkpoint."""

    path: Path
    source_run: Path
    target_run: Path
    cross_run: bool
    schema: str
    next_epoch: int
    trajectory_equivalence: bool
    payload: dict[str, Any]


def resolve_resume_path(cfg: Mapping[str, Any], run_dir: str | Path) -> Path | None:
    """Resolve resume without allowing ``resume: true`` to start a fresh run."""
    training_cfg = _mapping(cfg.get("training"))
    resume = training_cfg.get("resume", False)
    if not resume:
        return None
    target_run = Path(run_dir)
    if resume is True:
        if not _mapping(cfg.get("output")).get("run_name"):
            raise ValueError("training.resume=true requires output.run_name so checkpoints/last.pth can be resolved.")
        path = target_run / "checkpoints" / "last.pth"
    elif isinstance(resume, str):
        path = resolve_path(resume)
        if path is None:  # pragma: no cover - resolve_path currently always returns a Path for strings.
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    else:
        raise ValueError("training.resume must be false, true, or a checkpoint path string.")
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def build_resume_contract(
    cfg: Mapping[str, Any],
    split_metadata: Mapping[str, Any] | None,
    normalization_artifacts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build canonical fingerprints for fields that affect the training trajectory."""
    config = _semantic_config(cfg)
    split = _canonicalize(dict(split_metadata or {}))
    normalization = _canonicalize(dict(normalization_artifacts or {}))
    return {
        "resume_contract_schema_version": RESUME_CONTRACT_SCHEMA_VERSION,
        "config": config,
        "config_sha256": _fingerprint(config),
        "split": split,
        "split_sha256": _fingerprint(split),
        "normalization": normalization,
        "normalization_sha256": _fingerprint(normalization),
        "training_epochs": _training_epochs(cfg),
    }


def validate_resume_contract(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    next_epoch: int,
) -> None:
    """Reject all resume drift except the closed runtime-control allowlist."""
    _validate_contract_shape(recorded, label="checkpoint")
    _validate_contract_shape(current, label="current")
    recorded_epochs = int(recorded["training_epochs"])
    current_epochs = int(current["training_epochs"])
    if current_epochs < recorded_epochs or current_epochs < int(next_epoch):
        raise CheckpointLoadError(
            "Resume contract mismatch at training.epochs: "
            f"checkpoint={recorded_epochs!r}, current={current_epochs!r}, next_epoch={int(next_epoch)!r}."
        )
    for section, fingerprint_key in (
        ("config", "config_sha256"),
        ("split", "split_sha256"),
        ("normalization", "normalization_sha256"),
    ):
        differences = _structured_diff(recorded[section], current[section], prefix=section)
        if differences:
            path, expected, actual = differences[0]
            raise CheckpointLoadError(
                f"Resume contract mismatch at {path}: checkpoint={expected!r}, current={actual!r}; "
                f"checkpoint {fingerprint_key}={recorded[fingerprint_key]}, "
                f"current {fingerprint_key}={current[fingerprint_key]}."
            )


def validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    path: str | Path,
    scheduler_enabled: bool,
) -> dict[str, Any]:
    """Validate a current-schema payload before model or optimizer restoration."""
    if not isinstance(payload, Mapping):
        raise CheckpointLoadError(_resume_error(path, "payload", "must be a mapping"))
    data = dict(payload)
    version = data.get("checkpoint_schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointLoadError(
            _resume_error(path, "checkpoint_schema_version", f"must equal {CHECKPOINT_SCHEMA_VERSION!r}, got {version!r}")
        )
    for field in (
        "checkpoint_role",
        "state_dict",
        "optimizer",
        "scheduler",
        "epoch",
        "runtime_state",
        "resume_contract",
    ):
        if field not in data:
            raise CheckpointLoadError(_resume_error(path, field, "is missing"))
    if not isinstance(data["checkpoint_role"], str) or not data["checkpoint_role"]:
        raise CheckpointLoadError(_resume_error(path, "checkpoint_role", "must be a non-empty string"))
    if not isinstance(data["state_dict"], Mapping):
        raise CheckpointLoadError(_resume_error(path, "state_dict", "must be a mapping"))
    if not isinstance(data["optimizer"], Mapping):
        raise CheckpointLoadError(_resume_error(path, "optimizer", "must be a mapping"))
    if not isinstance(data["epoch"], int) or isinstance(data["epoch"], bool) or data["epoch"] < 0:
        raise CheckpointLoadError(_resume_error(path, "epoch", "must be a non-negative integer"))
    if scheduler_enabled and data["scheduler"] is None:
        raise CheckpointLoadError(_resume_error(path, "scheduler", "is null while the current run enables a scheduler"))
    if data["scheduler"] is not None and not isinstance(data["scheduler"], Mapping):
        raise CheckpointLoadError(_resume_error(path, "scheduler", "must be a mapping or null"))
    _validate_runtime_state(data["runtime_state"], path=path)
    if not isinstance(data["resume_contract"], Mapping):
        raise CheckpointLoadError(_resume_error(path, "resume_contract", "must be a mapping"))
    _validate_contract_shape(data["resume_contract"], label="checkpoint")
    return data


def preflight_resume(
    cfg: Mapping[str, Any],
    run_dir: str | Path,
    *,
    scheduler_enabled: bool,
    split_metadata: Mapping[str, Any] | None = None,
    normalization_artifacts: Mapping[str, Any] | None = None,
) -> ResumePlan | None:
    """Perform the read-only resume gate used before target-run initialization."""
    path = resolve_resume_path(cfg, run_dir)
    if path is None:
        return None
    raw_payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(raw_payload, Mapping):
        raise CheckpointLoadError(_resume_error(path, "payload", "must be a mapping"))
    validate_checkpoint_publication(path, payload=raw_payload)
    payload = validate_resume_payload(raw_payload, path=path, scheduler_enabled=scheduler_enabled)
    if split_metadata is not None and normalization_artifacts is not None:
        current_contract = build_resume_contract(cfg, split_metadata, normalization_artifacts)
        validate_resume_contract(payload["resume_contract"], current_contract, next_epoch=payload["epoch"])
    target_run = Path(run_dir)
    source_run = path.parent.parent if path.parent.name == "checkpoints" else path.parent
    return ResumePlan(
        path=path,
        source_run=source_run,
        target_run=target_run,
        cross_run=source_run.resolve() != target_run.resolve(),
        schema="current",
        next_epoch=int(payload["epoch"]),
        trajectory_equivalence=True,
        payload=payload,
    )


def capture_runtime_state(
    *,
    dataloaders: Mapping[str, Any] | None,
    grad_scaler: Any | None,
    extensions: list[Any] | tuple[Any, ...] | None,
    extension_states: list[Any] | tuple[Any, ...] | None,
    training_state: Any | None,
) -> dict[str, Any]:
    """Capture only safe-serialization values required for the next epoch."""
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
    result = {
        "runtime_state_schema_version": RUNTIME_STATE_SCHEMA_VERSION,
        "rng": {
            "python": _safe_runtime_value(random.getstate()),
            "numpy": _numpy_rng_state(),
            "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
            "cuda": cuda_states,
            "cuda_device_count": len(cuda_states),
        },
        "dataloaders": _capture_dataloaders(dataloaders or {}),
        "grad_scaler": _capture_grad_scaler(grad_scaler),
        "extensions": _capture_extensions(extensions or (), extension_states or ()),
        "training_state": _capture_training_state(training_state),
    }
    return result


def restore_runtime_state(
    runtime_state: Mapping[str, Any],
    *,
    dataloaders: Mapping[str, Any] | None,
    grad_scaler: Any | None,
    extensions: list[Any] | tuple[Any, ...] | None,
    extension_states: list[Any] | tuple[Any, ...] | None,
    training_state: Any | None,
) -> None:
    """Restore runtime state before the caller creates the next train iterator."""
    _validate_runtime_state(runtime_state, path="runtime_state")
    runtime = dict(runtime_state)
    _restore_dataloaders(runtime["dataloaders"], dataloaders or {})
    _restore_grad_scaler(runtime["grad_scaler"], grad_scaler)
    _restore_extensions(runtime["extensions"], extensions or (), extension_states or ())
    _restore_training_state(runtime["training_state"], training_state)
    _restore_rng(runtime["rng"])


def _semantic_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(cfg))
    for path in _RUNTIME_CONTROL_PATHS:
        _delete_path(value, path)
    return _canonicalize(value)


# This is deliberately closed: no user supplied glob can relax resume compatibility.
_RUNTIME_CONTROL_PATHS = (
    ("training", "resume"),
    ("training", "epochs"),
    ("training", "timing"),
    ("training", "progress"),
    ("training", "log_every_n_steps"),
    ("training", "log_interval"),
    ("training", "log_frequency"),
    ("training", "status_interval"),
    ("output", "dir"),
    ("output", "run_name"),
    ("output", "overwrite"),
    ("output", "tensorboard"),
    ("output", "progress"),
    ("output", "log_every_n_steps"),
    ("output", "log_interval"),
)


def _training_epochs(cfg: Mapping[str, Any]) -> int:
    raw = _mapping(cfg.get("training")).get("epochs", 0)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise CheckpointLoadError(f"training.epochs must be an integer for resume, got {raw!r}.") from exc
    if value < 0:
        raise CheckpointLoadError(f"training.epochs must be non-negative for resume, got {value!r}.")
    return value


def _validate_contract_shape(contract: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(contract, Mapping):
        raise CheckpointLoadError(f"{label} resume_contract must be a mapping.")
    required = (
        "resume_contract_schema_version",
        "config",
        "config_sha256",
        "split",
        "split_sha256",
        "normalization",
        "normalization_sha256",
        "training_epochs",
    )
    missing = [field for field in required if field not in contract]
    if missing:
        raise CheckpointLoadError(f"{label} resume_contract is missing fields: {missing}.")
    if contract["resume_contract_schema_version"] != RESUME_CONTRACT_SCHEMA_VERSION:
        raise CheckpointLoadError(
            f"{label} resume_contract schema version must equal {RESUME_CONTRACT_SCHEMA_VERSION!r}, "
            f"got {contract['resume_contract_schema_version']!r}."
        )
    for section, fingerprint_key in (
        ("config", "config_sha256"),
        ("split", "split_sha256"),
        ("normalization", "normalization_sha256"),
    ):
        if _fingerprint(contract[section]) != contract[fingerprint_key]:
            raise CheckpointLoadError(f"{label} resume_contract {fingerprint_key} does not match its {section} payload.")


def _validate_runtime_state(value: Any, *, path: str | Path) -> None:
    if not isinstance(value, Mapping):
        raise CheckpointLoadError(_resume_error(path, "runtime_state", "must be a mapping"))
    required = (
        "runtime_state_schema_version",
        "rng",
        "dataloaders",
        "grad_scaler",
        "extensions",
        "training_state",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise CheckpointLoadError(_resume_error(path, "runtime_state", f"is missing fields {missing}"))
    if value["runtime_state_schema_version"] != RUNTIME_STATE_SCHEMA_VERSION:
        raise CheckpointLoadError(
            _resume_error(
                path,
                "runtime_state.runtime_state_schema_version",
                f"must equal {RUNTIME_STATE_SCHEMA_VERSION!r}",
            )
        )
    for field in ("rng", "dataloaders", "grad_scaler", "training_state"):
        if not isinstance(value[field], Mapping):
            raise CheckpointLoadError(_resume_error(path, f"runtime_state.{field}", "must be a mapping"))
    if not isinstance(value["extensions"], list):
        raise CheckpointLoadError(_resume_error(path, "runtime_state.extensions", "must be a list"))


def _capture_dataloaders(dataloaders: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split, loader in dataloaders.items():
        record: dict[str, Any] = {"generator": _generator_state(getattr(loader, "generator", None))}
        sampler = getattr(loader, "sampler", None)
        sampler_record: dict[str, Any] = {}
        if sampler is not None:
            state_dict = getattr(sampler, "state_dict", None)
            if callable(state_dict):
                sampler_record["state_dict"] = _safe_runtime_value(state_dict())
            generator_state = _generator_state(getattr(sampler, "generator", None))
            if generator_state is not None:
                sampler_record["generator"] = generator_state
        record["sampler"] = sampler_record
        result[str(split)] = record
    return result


def _restore_dataloaders(records: Mapping[str, Any], dataloaders: Mapping[str, Any]) -> None:
    for split, record in records.items():
        if split not in dataloaders:
            raise CheckpointLoadError(f"Current exact resume is missing DataLoader split {split!r}.")
        if not isinstance(record, Mapping):
            raise CheckpointLoadError(f"Runtime DataLoader state for split {split!r} must be a mapping.")
        loader = dataloaders[split]
        _restore_generator(getattr(loader, "generator", None), record.get("generator"), f"DataLoader {split!r}")
        sampler_record = record.get("sampler", {})
        if not isinstance(sampler_record, Mapping):
            raise CheckpointLoadError(f"Runtime sampler state for split {split!r} must be a mapping.")
        sampler = getattr(loader, "sampler", None)
        if "state_dict" in sampler_record:
            loader_state = getattr(sampler, "load_state_dict", None)
            if not callable(loader_state):
                raise CheckpointLoadError(f"Sampler for split {split!r} cannot restore its recorded state.")
            loader_state(sampler_record["state_dict"])
        _restore_generator(
            getattr(sampler, "generator", None),
            sampler_record.get("generator"),
            f"sampler for split {split!r}",
        )


def _generator_state(generator: Any) -> torch.Tensor | None:
    if isinstance(generator, torch.Generator):
        return generator.get_state().detach().cpu().clone()
    return None


def _restore_generator(generator: Any, state: Any, label: str) -> None:
    if state is None:
        return
    if not isinstance(generator, torch.Generator):
        raise CheckpointLoadError(f"{label} has no compatible generator for exact resume.")
    if not torch.is_tensor(state):
        raise CheckpointLoadError(f"Recorded generator state for {label} must be a tensor.")
    generator.set_state(state.detach().cpu())


def _capture_grad_scaler(grad_scaler: Any | None) -> dict[str, Any]:
    enabled = bool(grad_scaler is not None and getattr(grad_scaler, "is_enabled", lambda: False)())
    return {
        "enabled": enabled,
        "state": _safe_runtime_value(grad_scaler.state_dict()) if enabled else {},
    }


def _restore_grad_scaler(record: Mapping[str, Any], grad_scaler: Any | None) -> None:
    if not isinstance(record, Mapping):
        raise CheckpointLoadError("Runtime GradScaler state must be a mapping.")
    enabled = bool(record.get("enabled", False))
    if not enabled:
        return
    if grad_scaler is None or not bool(getattr(grad_scaler, "is_enabled", lambda: False)()):
        raise CheckpointLoadError("Current resume enables GradScaler but no compatible GradScaler is configured.")
    state = record.get("state")
    if not isinstance(state, Mapping):
        raise CheckpointLoadError("Runtime GradScaler state is missing or invalid.")
    grad_scaler.load_state_dict(dict(state))


def _capture_extensions(extensions: Any, states: Any) -> list[dict[str, Any]]:
    if len(extensions) != len(states):
        raise CheckpointLoadError("Extension instances and extension states must have the same length.")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for extension, state in zip(extensions, states):
        extension_id = _extension_id(extension)
        if extension_id in seen:
            raise CheckpointLoadError(f"Training extension id {extension_id!r} is duplicated.")
        seen.add(extension_id)
        schema_version = int(getattr(extension, "state_schema_version", 1))
        stateless = bool(getattr(extension, "stateless", False))
        record: dict[str, Any] = {
            "id": extension_id,
            "state_schema_version": schema_version,
            "stateless": stateless,
        }
        if not stateless:
            state_dict = getattr(extension, "state_dict", None)
            if not callable(state_dict):
                raise CheckpointLoadError(f"Stateful training extension {extension_id!r} has no state_dict().")
            record["state"] = _safe_runtime_value(state_dict(state))
        records.append(record)
    return records


def _restore_extensions(records: list[Any], extensions: Any, states: Any) -> None:
    if len(extensions) != len(states):
        raise CheckpointLoadError("Extension instances and extension states must have the same length.")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("id"), str):
            raise CheckpointLoadError("Recorded extension state must include a stable string id.")
        by_id[str(record["id"])] = record
    if len(by_id) != len(records):
        raise CheckpointLoadError("Recorded extension state contains duplicate ids.")
    for extension, state in zip(extensions, states):
        extension_id = _extension_id(extension)
        record = by_id.pop(extension_id, None)
        if record is None:
            raise CheckpointLoadError(f"Current exact resume is missing extension state for {extension_id!r}.")
        expected_schema = int(getattr(extension, "state_schema_version", 1))
        if record.get("state_schema_version") != expected_schema:
            raise CheckpointLoadError(
                f"Extension {extension_id!r} state schema mismatch: "
                f"checkpoint={record.get('state_schema_version')!r}, current={expected_schema!r}."
            )
        if bool(record.get("stateless", False)):
            if not bool(getattr(extension, "stateless", False)):
                raise CheckpointLoadError(f"Extension {extension_id!r} unexpectedly lacks required state.")
            continue
        load_state_dict = getattr(extension, "load_state_dict", None)
        if not callable(load_state_dict) or "state" not in record:
            raise CheckpointLoadError(f"Extension {extension_id!r} cannot restore its required state.")
        load_state_dict(state, record["state"])
    if by_id:
        raise CheckpointLoadError(f"Current exact resume has unknown recorded extensions: {sorted(by_id)}.")


def _extension_id(extension: Any) -> str:
    value = getattr(extension, "resume_id", None) or getattr(extension, "name", None)
    if not isinstance(value, str) or not value:
        raise CheckpointLoadError("Training extension must declare a stable non-empty name or resume_id.")
    return value


def _capture_training_state(training_state: Any | None) -> dict[str, Any]:
    if training_state is None:
        return {}
    state_dict = getattr(training_state, "state_dict", None)
    if callable(state_dict):
        return _safe_runtime_value(state_dict())
    return _safe_runtime_value(
        {
            "history": getattr(training_state, "history", {}),
            "epoch_logs": getattr(training_state, "epoch_logs", []),
        }
    )


def _restore_training_state(record: Mapping[str, Any], training_state: Any | None) -> None:
    if training_state is None:
        if record:
            raise CheckpointLoadError("Current exact resume has TrainingState data but no TrainingState instance.")
        return
    load_state_dict = getattr(training_state, "load_state_dict", None)
    if callable(load_state_dict):
        load_state_dict(dict(record))
        return
    history = record.get("history")
    epoch_logs = record.get("epoch_logs")
    if hasattr(training_state, "history") and isinstance(history, Mapping):
        training_state.history.clear()
        training_state.history.update(copy.deepcopy(dict(history)))
    if hasattr(training_state, "epoch_logs") and isinstance(epoch_logs, list):
        training_state.epoch_logs[:] = copy.deepcopy(epoch_logs)


def _numpy_rng_state() -> dict[str, Any]:
    bit_generator, state, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": str(bit_generator),
        "state": torch.from_numpy(np.asarray(state, dtype=np.uint32).copy()),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_rng(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise CheckpointLoadError("Runtime RNG state must be a mapping.")
    python_state = record.get("python")
    numpy_state = record.get("numpy")
    torch_cpu = record.get("torch_cpu")
    if python_state is not None:
        random.setstate(_as_tuple(python_state))
    if isinstance(numpy_state, Mapping):
        values = numpy_state.get("state")
        if not torch.is_tensor(values):
            raise CheckpointLoadError("Runtime NumPy RNG state must contain a tensor state.")
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                values.detach().cpu().numpy().astype(np.uint32, copy=False),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if torch_cpu is not None:
        if not torch.is_tensor(torch_cpu):
            raise CheckpointLoadError("Runtime Torch CPU RNG state must be a tensor.")
        torch.set_rng_state(torch_cpu.detach().cpu())
    cuda_states = record.get("cuda", [])
    if cuda_states:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(cuda_states):
            raise CheckpointLoadError(
                "CUDA RNG topology differs from the checkpoint; current exact resume is not supported."
            )
        if not all(torch.is_tensor(state) for state in cuda_states):
            raise CheckpointLoadError("Runtime CUDA RNG states must be tensors.")
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_states])


def _safe_runtime_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, Mapping):
        return {str(key): _safe_runtime_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_runtime_value(item) for item in value]
    raise CheckpointLoadError(f"Runtime state contains unsupported value type {type(value).__name__}.")


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {"__float__": str(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonicalize(item) for item in value), key=_stable_json)
    return str(value)


def _structured_diff(expected: Any, actual: Any, *, prefix: str) -> list[tuple[str, Any, Any]]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[tuple[str, Any, Any]] = []
        for key in sorted(set(expected) | set(actual), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected:
                differences.append((path, "<missing>", actual[key]))
            elif key not in actual:
                differences.append((path, expected[key], "<missing>"))
            else:
                differences.extend(_structured_diff(expected[key], actual[key], prefix=path))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        differences = []
        for index in range(max(len(expected), len(actual))):
            path = f"{prefix}[{index}]"
            if index >= len(expected):
                differences.append((path, "<missing>", actual[index]))
            elif index >= len(actual):
                differences.append((path, expected[index], "<missing>"))
            else:
                differences.extend(_structured_diff(expected[index], actual[index], prefix=path))
        return differences
    return [] if expected == actual else [(prefix, expected, actual)]


def _delete_path(mapping: dict[str, Any], path: tuple[str, ...]) -> None:
    current: Any = mapping
    for key in path[:-1]:
        if not isinstance(current, dict):
            return
        current = current.get(key)
    if isinstance(current, dict):
        current.pop(path[-1], None)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def _resume_error(path: str | Path, field: str, message: str) -> str:
    return f"Invalid resume checkpoint {Path(path)} (resume): {field} {message}."
