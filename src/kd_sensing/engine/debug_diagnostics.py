import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from kd_sensing.config.io import safe_load_yaml
from kd_sensing.engine.objectives.metadata import objective_runtime_metadata
from kd_sensing.models.architecture_summary import summarize_model_architecture
from kd_sensing.utils.paths import resolve_path


RUN_IDENTITY_PATHS = {
    "experiment.name",
    "experiment.seed",
    "data.dataset.portion_seed",
    "output.dir",
    "output.run_name",
    "output.evaluation_run_name",
    "output.overwrite",
    "output.group_by_scene",
}

CRITICAL_CONFIG_PATHS = (
    "training.lr",
    "training.weight_decay",
    "training.grad_clip",
    "scheduler",
    "loss",
    "data.dataset.train_csv_name",
    "data.dataset.val_csv_name",
    "data.dataset.test_csv_name",
    "data.dataset.seq_len",
    "data.dataset.num_pred",
    "data.dataset.csi_train_rms",
    "data.dataset.csi_rms_path",
    "data.dataset.csi_degradation",
    "model.num_classes",
    "model.num_pred",
    "model.seq_length",
    "model.primary.type",
    "model.primary.num_classes",
    "model.primary.num_pred",
    "model.primary.encoders.csi",
    "model.primary.representation_core",
    "model.primary.heads.beam",
)

ESTIMATION_SNR_MODES = {"est_snr", "estimation_snr"}
DISABLED_PILOT_MODES = {"none", "clean", "disabled", "off", "false"}
DEFAULT_MILD_PILOT_RATIO_MAX = 1.0e-2
DEFAULT_PILOT_RATIO_TOLERANCE = 3.0


def debug_enabled(cfg: dict[str, Any]) -> bool:
    debug_cfg = cfg.get("debug")
    if isinstance(debug_cfg, dict):
        return bool(debug_cfg.get("enabled", debug_cfg.get("enable", False)))
    if isinstance(debug_cfg, bool):
        return debug_cfg
    return bool(cfg.get("output", {}).get("debug", False))


def csi_first_batch_debug_enabled(cfg: dict[str, Any]) -> bool:
    debug_cfg = cfg.get("debug")
    if isinstance(debug_cfg, dict):
        csi_cfg = debug_cfg.get("csi_first_batch") or debug_cfg.get("csi") or {}
        if isinstance(csi_cfg, dict):
            return bool(csi_cfg.get("enabled", debug_enabled(cfg)))
        if isinstance(csi_cfg, bool):
            return csi_cfg
    return debug_enabled(cfg)


def training_health_debug_enabled(cfg: dict[str, Any]) -> bool:
    debug_cfg = cfg.get("debug")
    if isinstance(debug_cfg, dict):
        health_cfg = debug_cfg.get("training_health") or {}
        if isinstance(health_cfg, dict):
            return bool(health_cfg.get("enabled", debug_enabled(cfg)))
        if isinstance(health_cfg, bool):
            return health_cfg
    return debug_enabled(cfg)


def config_diff_debug_enabled(cfg: dict[str, Any]) -> bool:
    diff_cfg = _config_diff_cfg(cfg)
    if isinstance(diff_cfg, dict) and "enabled" in diff_cfg:
        return bool(diff_cfg.get("enabled"))
    return debug_enabled(cfg) and bool(diff_cfg.get("reference") or diff_cfg.get("reference_config"))


def configure_csi_debug(module: nn.Module, cfg: dict[str, Any]) -> None:
    if not csi_first_batch_debug_enabled(cfg):
        return
    for child in module.modules():
        setter = getattr(child, "set_debug_enabled", None)
        if callable(setter):
            setter(True)


def set_csi_debug_batch_source(module: nn.Module, source: str) -> None:
    for child in module.modules():
        setter = getattr(child, "set_debug_batch_source", None)
        if callable(setter):
            setter(source)


def consume_csi_debug_records(module: nn.Module) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for child in module.modules():
        consumer = getattr(child, "consume_debug_records", None)
        if callable(consumer):
            records.extend(consumer())
    return records


def write_csi_debug_records(run_dir: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    _write_json(run_dir / "csi_first_batch_diagnostics.json", records)


def write_pilot_noise_validity_artifact(run_dir: Path, validity: dict[str, Any]) -> None:
    if not validity or validity.get("applicable") is False:
        return
    _write_json(run_dir / "pilot_noise_validity.json", validity)


def write_config_diff_artifact(cfg: dict[str, Any], resolved_cfg: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    if not config_diff_debug_enabled(cfg):
        if debug_enabled(cfg):
            result = {
                "status": "disabled",
                "parity_passed": None,
                "message": "Config diff is disabled for this debug run.",
            }
            _write_json(run_dir / "config_diff.json", result)
            return result
        return None
    diff_cfg = _config_diff_cfg(cfg)
    reference_raw = diff_cfg.get("reference") or diff_cfg.get("reference_config")
    if not reference_raw:
        result = {"status": "missing_reference", "parity_passed": False, "message": "No reference config was set."}
        _write_json(run_dir / "config_diff.json", result)
        return result
    reference_path = resolve_path(str(reference_raw))
    if not reference_path.exists():
        result = {
            "status": "missing_reference",
            "parity_passed": False,
            "reference": str(reference_path),
            "message": f"Reference config not found: {reference_path}",
        }
        _write_json(run_dir / "config_diff.json", result)
        return result
    reference_cfg = _read_config_mapping(reference_path)
    result = compare_resolved_configs(reference_cfg, resolved_cfg, reference=str(reference_path))
    _write_json(run_dir / "config_diff.json", result)
    return result


def compare_resolved_configs(
    reference_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    *,
    reference: str | None = None,
) -> dict[str, Any]:
    all_differences = _diff_mappings(reference_cfg, target_cfg)
    allowed = [item for item in all_differences if _allowed_identity_path(item["path"])]
    behavior = [item for item in all_differences if item not in allowed and not item["path"].startswith("debug")]
    critical = []
    for path in CRITICAL_CONFIG_PATHS:
        reference_value = _normalize_compare_value(path, _get_by_path(reference_cfg, path))
        target_value = _normalize_compare_value(path, _get_by_path(target_cfg, path))
        if reference_value != target_value:
            critical.append({"path": path, "reference": reference_value, "target": target_value})
    parity_passed = len(critical) == 0
    status = "passed" if parity_passed and not behavior else "passed_with_noncritical_differences" if parity_passed else "failed"
    return {
        "status": status,
        "parity_passed": parity_passed,
        "reference": reference,
        "allowed_identity_differences": allowed,
        "behavior_differences": behavior,
        "critical_differences": critical,
        "critical_paths_checked": list(CRITICAL_CONFIG_PATHS),
        "allowed_identity_paths": sorted(RUN_IDENTITY_PATHS),
    }


def build_startup_summary(
    cfg: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    device: torch.device,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    loader_cfg = cfg.get("data", {}).get("dataloader", {})
    model_cfg = cfg.get("model", {})
    primary_cfg = model_cfg.get("primary", {}) if isinstance(model_cfg.get("primary"), dict) else {}
    csi_cfg = ((primary_cfg.get("encoders") or {}).get("csi") or {}) if isinstance(primary_cfg, dict) else {}
    architecture_summary = summarize_model_architecture(
        model,
        cfg=primary_cfg,
        source={"kind": "instance", "config_path": "startup_summary"},
    )
    parameter_report = module_trainability_report(model, architecture_summary=architecture_summary)
    objective_metadata = objective_runtime_metadata(cfg)
    summary = {
        "experiment": {
            "name": cfg.get("experiment", {}).get("name"),
            "task": cfg.get("experiment", {}).get("task"),
            "objective": cfg.get("experiment", {}).get("objective"),
            "seed": cfg.get("experiment", {}).get("seed"),
            "device": str(device),
        },
        "objective": objective_metadata,
        "data": {
            "modalities": list(primary_cfg.get("modalities") or model_cfg.get("modalities") or []),
            "dataset_type": dataset_cfg.get("type"),
            "dataset_path": dataset_cfg.get("data_root"),
            "scene": dataset_cfg.get("scene"),
            "train_scenes": dataset_cfg.get("train_scenes"),
            "test_scenes": dataset_cfg.get("test_scenes")
            or dataset_cfg.get("eval_scenes")
            or dataset_cfg.get("validation_scenes"),
            "train_split": dataset_cfg.get("train_csv_name"),
            "val_split": dataset_cfg.get("val_csv_name"),
            "test_split": dataset_cfg.get("test_csv_name"),
            "seq_len": dataset_cfg.get("seq_len"),
            "num_pred": dataset_cfg.get("num_pred"),
            "num_classes": model_cfg.get("num_classes") or primary_cfg.get("num_classes"),
            "batch_size": {
                "train": loader_cfg.get("train_batch_size", loader_cfg.get("batch_size")),
                "test": loader_cfg.get("test_batch_size", loader_cfg.get("batch_size")),
            },
            "normalization": {
                "csi_train_rms": dataset_cfg.get("csi_train_rms"),
                "csi_rms_path": dataset_cfg.get("csi_rms_path"),
            },
        },
        "optimization": {
            "optimizer": type(optimizer).__name__,
            "learning_rates": [float(group.get("lr", 0.0)) for group in optimizer.param_groups],
            "weight_decays": [float(group.get("weight_decay", 0.0)) for group in optimizer.param_groups],
            "scheduler": type(scheduler).__name__ if scheduler is not None else None,
            "loss": cfg.get("loss", {}).get("type"),
            "max_epochs": cfg.get("training", {}).get("epochs"),
        },
        "model": {
            "type": primary_cfg.get("type"),
            "csi_encoder_type": csi_cfg.get("type"),
            "d_model": primary_cfg.get("d_model") or model_cfg.get("d_model"),
            "delay_taps": csi_cfg.get("delay_taps"),
            "view_fusion": csi_cfg.get("view_fusion"),
            "use_internal_gru": csi_cfg.get("use_internal_gru", True) if csi_cfg else None,
            "pilot_estimator": _pilot_summary(csi_cfg),
            "csi_hardening_enabled": _mapping_enabled(csi_cfg.get("csi_hardening")),
            "csi_degradation_enabled": _mapping_enabled(dataset_cfg.get("csi_degradation")),
        },
        "parameters": parameter_report,
        "architecture_summary": architecture_summary,
    }
    warnings = [
        f"{name} has zero trainable parameters at {item['path']}"
        for name, item in parameter_report["modules"].items()
        if item.get("required") and int(item.get("total_params", 0)) > 0 and int(item.get("trainable_params", 0)) == 0
    ]
    if warnings:
        summary["warnings"] = warnings
    return _to_jsonable(summary)


def evaluate_pilot_noise_validity(
    cfg: dict[str, Any],
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    csi_cfg = _primary_csi_config(cfg)
    if not csi_cfg:
        return {"applicable": False, "valid": None, "reason": "no_csi_encoder"}
    pilot_cfg = _pilot_config(csi_cfg)
    mode = _pilot_mode(pilot_cfg)
    run_name = _run_name_from_config(cfg)
    matrix_role = _matrix_role_from_config(cfg, run_name)
    is_mild = _is_mild_pilot_role(matrix_role, run_name)
    is_destructive = _is_destructive_role(cfg, matrix_role, run_name)
    ratio = _pilot_ratio_from_records(records or [])
    snr_db = _pilot_snr_from_records(records or [])
    bounds = _pilot_expected_ratio_bounds(pilot_cfg, sampled_snr_db=snr_db)
    result: dict[str, Any] = {
        "applicable": True,
        "valid": True,
        "reason": "ok",
        "mode": mode,
        "run_name": run_name,
        "matrix_role": matrix_role,
        "is_mild_pilot_estimation": bool(is_mild),
        "is_destructive_control": bool(is_destructive),
        "noise_power_signal_ratio": ratio,
        "expected_ratio_min": bounds[0],
        "expected_ratio_max": bounds[1],
        "snr_db": snr_db,
    }
    if is_destructive:
        result["reason"] = "destructive_negative_control"
        return _to_jsonable(result)
    enabled = bool(pilot_cfg.get("enabled", pilot_cfg.get("enable", True))) if isinstance(pilot_cfg, dict) else True
    if not enabled or mode in DISABLED_PILOT_MODES:
        if ratio is not None and ratio > DEFAULT_MILD_PILOT_RATIO_MAX:
            result["valid"] = False
            result["reason"] = "invalid_due_to_pilot_noise_scale"
        else:
            result["reason"] = "pilot_noise_disabled"
        return _to_jsonable(result)
    if _is_non_pilot_matrix_role(matrix_role, run_name):
        result["valid"] = False
        result["reason"] = "invalid_due_to_unisolated_pilot_noise"
        return _to_jsonable(result)
    if not is_mild:
        if ratio is None:
            result["valid"] = None
            result["reason"] = "missing_pilot_noise_diagnostics"
        return _to_jsonable(result)
    if ratio is None:
        result["valid"] = None
        result["reason"] = "missing_pilot_noise_diagnostics"
        return _to_jsonable(result)
    if mode in ESTIMATION_SNR_MODES and bounds[0] is not None and bounds[1] is not None:
        if bounds[0] <= ratio <= bounds[1]:
            return _to_jsonable(result)
        result["valid"] = False
        result["reason"] = "invalid_due_to_pilot_noise_scale"
        return _to_jsonable(result)
    if ratio <= DEFAULT_MILD_PILOT_RATIO_MAX:
        result["reason"] = "physical_noise_ratio_within_mild_threshold"
        return _to_jsonable(result)
    result["valid"] = False
    result["reason"] = "invalid_due_to_pilot_noise_scale"
    return _to_jsonable(result)


def print_startup_summary(summary: dict[str, Any]) -> None:
    print("[startup_summary] " + json.dumps(summary, sort_keys=True), flush=True)


def write_startup_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    _write_json(run_dir / "startup_summary.json", summary)


def module_trainability_report(
    model: nn.Module,
    *,
    architecture_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    architecture_summary = architecture_summary or summarize_model_architecture(model)
    modules = _major_modules(model)
    parameter_summary = architecture_summary.get("parameters", {})
    total_params = int(parameter_summary.get("total_params", 0))
    trainable_params = int(parameter_summary.get("trainable_params", 0))
    module_report = {}
    for name, (path, module, required) in modules.items():
        seen: set[int] = set()
        module_params = []
        for param in module.parameters():
            if id(param) in seen:
                continue
            seen.add(id(param))
            module_params.append(param)
        total = sum(param.numel() for param in module_params)
        trainable = sum(param.numel() for param in module_params if param.requires_grad)
        module_report[name] = {
            "path": path,
            "required": bool(required),
            "total_params": int(total),
            "trainable_params": int(trainable),
            "suspicious": bool(required and total > 0 and trainable == 0),
        }
    return {
        "schema_version": architecture_summary.get("schema_version"),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(parameter_summary.get("frozen_params", total_params - trainable_params)),
        "effective_params": int(parameter_summary.get("effective_params", total_params)),
        "excluded_params": int(parameter_summary.get("excluded_params", 0)),
        "excluded_parameter_groups": parameter_summary.get("excluded_parameter_groups", []),
        "modules": module_report,
    }


class ModuleHealthTracker:
    def __init__(self, model: nn.Module) -> None:
        self.modules = _major_modules(model)
        self._snapshots: dict[str, list[torch.Tensor]] = {}
        self._grad_max: dict[str, float] = {name: 0.0 for name in self.modules}
        self._zero_grad_epochs: dict[str, int] = {name: 0 for name in self.modules}
        self._zero_delta_epochs: dict[str, int] = {name: 0 for name in self.modules}

    def start_epoch(self) -> None:
        self._snapshots = {}
        self._grad_max = {name: 0.0 for name in self.modules}
        for name, (_, module, _) in self.modules.items():
            self._snapshots[name] = [
                param.detach().cpu().clone()
                for param in module.parameters()
                if param.requires_grad
            ]

    def observe_gradients(self) -> None:
        for name, (_, module, _) in self.modules.items():
            grad_norm = _module_grad_norm(module)
            self._grad_max[name] = max(self._grad_max.get(name, 0.0), grad_norm)

    def finish_epoch(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        warnings = []
        for name, (_, module, required) in self.modules.items():
            grad_norm = float(self._grad_max.get(name, 0.0))
            param_delta = _module_param_delta(module, self._snapshots.get(name, []))
            result[f"grad_norm_{name}"] = grad_norm
            result[f"param_delta_{name}"] = float(param_delta)
            if grad_norm == 0.0:
                self._zero_grad_epochs[name] += 1
            else:
                self._zero_grad_epochs[name] = 0
            if param_delta == 0.0:
                self._zero_delta_epochs[name] += 1
            else:
                self._zero_delta_epochs[name] = 0
            if required and (self._zero_grad_epochs[name] > 0 or self._zero_delta_epochs[name] > 0):
                warnings.append(
                    f"{name} may be frozen, missing from optimizer, gradient-masked, or disconnected from loss"
                )
        if warnings:
            result["training_health_warnings"] = warnings
        return result


def _major_modules(model: nn.Module) -> dict[str, tuple[str, nn.Module, bool]]:
    modules: dict[str, tuple[str, nn.Module, bool]] = {}
    encoders = getattr(model, "encoders", None)
    if isinstance(encoders, nn.ModuleDict) and "csi" in encoders:
        modules["csi_encoder"] = ("encoders.csi", encoders["csi"], True)
        csi_encoder = encoders["csi"]
        fusion_module = getattr(csi_encoder, "symmetric_fusion", None) or getattr(csi_encoder, "concat_projection", None)
        if isinstance(fusion_module, nn.Module):
            modules["fusion"] = ("encoders.csi.symmetric_fusion", fusion_module, False)
    if isinstance(encoders, nn.ModuleDict):
        for name, encoder in encoders.items():
            modules.setdefault(f"{name}_encoder", (f"encoders.{name}", encoder, True))
    representation_core = getattr(model, "representation_core", None)
    if isinstance(representation_core, nn.Module):
        modules["representation_core"] = ("representation_core", representation_core, True)
    heads = getattr(model, "heads", None)
    if isinstance(heads, nn.ModuleDict) and "beam" in heads:
        modules["beam_head"] = ("heads.beam", heads["beam"], True)
    for name in ("beam_head", "los_head", "link_head"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            modules.setdefault(name, (name, module, True))
    for name in ("gates", "task_projections"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            modules.setdefault(name, (name, module, False))
    for attr in ("fusion", "fusion_layer", "anchor_fusion"):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module) and "fusion" not in modules:
            modules["fusion"] = (attr, module, False)
    return modules


def _module_grad_norm(module: nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total += float(grad.float().pow(2).sum().item())
    return math.sqrt(total)


def _module_param_delta(module: nn.Module, snapshots: list[torch.Tensor]) -> float:
    total = 0.0
    idx = 0
    for param in module.parameters():
        if not param.requires_grad:
            continue
        if idx >= len(snapshots):
            break
        before = snapshots[idx].to(device=param.device, dtype=param.dtype)
        total += float((param.detach() - before).float().pow(2).sum().item())
        idx += 1
    return math.sqrt(total)


def _config_diff_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    debug_cfg = cfg.get("debug")
    if isinstance(debug_cfg, dict):
        value = debug_cfg.get("config_diff") or debug_cfg.get("a0_config_diff") or {}
        return value if isinstance(value, dict) else {"enabled": bool(value)}
    return {}


def _read_config_mapping(path: Path) -> dict[str, Any]:
    value = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _diff_mappings(reference: Any, target: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(reference, dict) and isinstance(target, dict):
        diffs = []
        for key in sorted(set(reference) | set(target)):
            child_path = f"{path}.{key}" if path else str(key)
            diffs.extend(_diff_mappings(reference.get(key), target.get(key), child_path))
        return diffs
    if isinstance(reference, list) and isinstance(target, list):
        if reference == target:
            return []
        return [{"path": path, "reference": _to_jsonable(reference), "target": _to_jsonable(target)}]
    if _normalize_compare_value(path, reference) == _normalize_compare_value(path, target):
        return []
    return [{"path": path, "reference": _to_jsonable(reference), "target": _to_jsonable(target)}]


def _allowed_identity_path(path: str) -> bool:
    if path in RUN_IDENTITY_PATHS:
        return True
    return path.startswith("runtime.") or path.startswith("output.") or path.endswith(".timestamp")


def _get_by_path(mapping: dict[str, Any], path: str) -> Any:
    cursor: Any = mapping
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _normalize_compare_value(path: str, value: Any) -> Any:
    if path.endswith(("csi_hardening", "csi_degradation")):
        if value is None:
            return {"enabled": False}
        if isinstance(value, bool):
            return {"enabled": bool(value)}
    if path.endswith(("csi_estimation", "pilot_estimator")) and isinstance(value, dict):
        if value.get("enabled") is False or value.get("enable") is False:
            return {"mode": "none"}
    return _canonicalize_config_value(_to_jsonable(value))


def _canonicalize_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "debug":
                continue
            if key == "mode" and item is None:
                result[key] = "none"
            elif key in {"csi_hardening", "csi_degradation"} and item is None:
                result[key] = {"enabled": False}
            else:
                result[key] = _canonicalize_config_value(item)
        if result.get("type") == "pilot_dual_view_csi":
            result.setdefault("csi_estimation", {"mode": "none"})
            if isinstance(result.get("csi_estimation"), dict) and result["csi_estimation"].get("mode") is None:
                result["csi_estimation"]["mode"] = "none"
            result.setdefault("csi_hardening", {"enabled": False})
            result.setdefault("view_gate_warmup_epochs", 0)
            result.setdefault("delay_view_warmup_epochs", 0)
            result.setdefault("use_internal_gru", True)
        return result
    if isinstance(value, list):
        return [_canonicalize_config_value(item) for item in value]
    return value


def _pilot_summary(csi_cfg: dict[str, Any]) -> dict[str, Any]:
    pilot = csi_cfg.get("pilot_estimator") or csi_cfg.get("csi_estimation") or {}
    if not isinstance(pilot, dict):
        return {"enabled": False, "mode": "none"}
    enabled = bool(pilot.get("enabled", pilot.get("enable", True)))
    return {
        "enabled": enabled,
        "mode": "none" if not enabled else pilot.get("mode", "none"),
        "snr_db": pilot.get("snr_db", pilot.get("est_snr_db")),
        "train_snr_min_db": pilot.get("train_snr_min_db"),
        "train_snr_max_db": pilot.get("train_snr_max_db"),
    }


def _primary_csi_config(cfg: dict[str, Any]) -> dict[str, Any]:
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    role_cfg = model.get("primary") if isinstance(model, dict) else {}
    encoders = role_cfg.get("encoders") if isinstance(role_cfg, dict) else {}
    csi_cfg = encoders.get("csi") if isinstance(encoders, dict) else {}
    if isinstance(csi_cfg, dict):
        return csi_cfg
    return {}


def _pilot_config(csi_cfg: dict[str, Any]) -> dict[str, Any]:
    pilot = csi_cfg.get("pilot_estimator") or csi_cfg.get("csi_estimation") or {}
    return pilot if isinstance(pilot, dict) else {}


def _pilot_mode(pilot_cfg: dict[str, Any]) -> str:
    enabled = bool(pilot_cfg.get("enabled", pilot_cfg.get("enable", True)))
    if not enabled:
        return "none"
    return str(pilot_cfg.get("mode", "none") or "none").lower()


def _run_name_from_config(cfg: dict[str, Any]) -> str:
    output = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    experiment = cfg.get("experiment") if isinstance(cfg.get("experiment"), dict) else {}
    return str(output.get("run_name") or experiment.get("name") or "")


def _matrix_role_from_config(cfg: dict[str, Any], run_name: str) -> str:
    debug_cfg = cfg.get("debug") if isinstance(cfg.get("debug"), dict) else {}
    return str(debug_cfg.get("matrix_role") or run_name or "")


def _is_mild_pilot_role(matrix_role: str, run_name: str) -> bool:
    text = f"{matrix_role} {run_name}".lower()
    return "a1" in text and "mild" in text and "pilot" in text


def _is_destructive_role(cfg: dict[str, Any], matrix_role: str, run_name: str) -> bool:
    debug_cfg = cfg.get("debug") if isinstance(cfg.get("debug"), dict) else {}
    analysis_role = str(debug_cfg.get("analysis_role", "")).lower()
    text = f"{matrix_role} {run_name} {analysis_role}".lower()
    return "destructive" in text or "negative_control" in text


def _is_non_pilot_matrix_role(matrix_role: str, run_name: str) -> bool:
    text = f"{matrix_role} {run_name}".lower()
    if any(token in text for token in ("a0", "a1", "a2", "debug")):
        return False
    return any(text.startswith(prefix) or f"_{prefix}" in text for prefix in ("b3", "b4", "b5", "b6", "c1", "c2", "d1", "d2", "d3", "d4"))


def _pilot_ratio_from_records(records: list[dict[str, Any]]) -> float | None:
    for record in _preferred_debug_records(records):
        pilot = record.get("pilot") if isinstance(record, dict) else {}
        if not isinstance(pilot, dict):
            continue
        ratio = _safe_float(pilot.get("noise_power_signal_ratio"))
        if ratio is not None:
            return ratio
    return None


def _pilot_snr_from_records(records: list[dict[str, Any]]) -> float | list[float] | None:
    for record in _preferred_debug_records(records):
        pilot = record.get("pilot") if isinstance(record, dict) else {}
        if not isinstance(pilot, dict):
            continue
        values = _numeric_values(pilot.get("snr_db"))
        if not values:
            continue
        if len(values) == 1:
            return values[0]
        return values
    return None


def _preferred_debug_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_records = [record for record in records if isinstance(record, dict) and str(record.get("source")) == "train"]
    return train_records or [record for record in records if isinstance(record, dict)]


def _pilot_expected_ratio_bounds(
    pilot_cfg: dict[str, Any],
    *,
    sampled_snr_db: float | list[float] | None = None,
) -> tuple[float | None, float | None]:
    explicit_min = _safe_float(
        pilot_cfg.get("expected_noise_ratio_min", pilot_cfg.get("noise_power_signal_ratio_min"))
    )
    explicit_max = _safe_float(
        pilot_cfg.get("expected_noise_ratio_max", pilot_cfg.get("noise_power_signal_ratio_max"))
    )
    if explicit_min is not None and explicit_max is not None:
        return (
            max(0.0, explicit_min / DEFAULT_PILOT_RATIO_TOLERANCE),
            explicit_max * DEFAULT_PILOT_RATIO_TOLERANCE,
        )
    snr_values = _numeric_values(sampled_snr_db)
    if not snr_values:
        min_snr = _safe_float(pilot_cfg.get("train_snr_min_db"))
        max_snr = _safe_float(pilot_cfg.get("train_snr_max_db"))
        fixed_snr = _safe_float(pilot_cfg.get("snr_db") if pilot_cfg.get("snr_db") is not None else pilot_cfg.get("est_snr_db"))
        if min_snr is not None and max_snr is not None:
            snr_values = [min_snr, max_snr]
        elif fixed_snr is not None:
            snr_values = [fixed_snr]
    if not snr_values:
        return (None, None)
    ratios = [10.0 ** (-float(value) / 10.0) for value in snr_values]
    return (
        max(0.0, min(ratios) / DEFAULT_PILOT_RATIO_TOLERANCE),
        max(ratios) * DEFAULT_PILOT_RATIO_TOLERANCE,
    )


def _numeric_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            numeric = _safe_float(item)
            if numeric is not None:
                result.append(numeric)
        return result
    numeric = _safe_float(value)
    return [] if numeric is None else [numeric]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mapping_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", False)))
    return False


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2, sort_keys=True)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        if detached.numel() == 1:
            return detached.reshape(()).item()
        return detached.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ModuleHealthTracker",
    "build_startup_summary",
    "compare_resolved_configs",
    "configure_csi_debug",
    "consume_csi_debug_records",
    "csi_first_batch_debug_enabled",
    "debug_enabled",
    "evaluate_pilot_noise_validity",
    "module_trainability_report",
    "print_startup_summary",
    "set_csi_debug_batch_source",
    "training_health_debug_enabled",
    "write_config_diff_artifact",
    "write_csi_debug_records",
    "write_pilot_noise_validity_artifact",
    "write_startup_summary",
]
