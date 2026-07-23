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
    "output.dir",
    "output.run_name",
    "output.overwrite",
}

CRITICAL_CONFIG_PATHS = (
    "training.lr",
    "training.weight_decay",
    "training.grad_clip",
    "training.epochs",
    "scheduler",
    "loss",
    "data.dataset.data_root",
    "data.dataset.train_csv_name",
    "data.dataset.val_csv_name",
    "data.dataset.test_csv_name",
    "data.dataset.seq_len",
    "data.dataset.num_pred",
    "data.dataset.gps_feature_mode",
    "model.num_classes",
    "model.num_pred",
    "model.seq_length",
    "model.primary.type",
    "model.primary.modalities",
    "model.primary.num_classes",
    "model.primary.num_pred",
    "model.primary.representation_core",
    "model.primary.heads.beam",
    "temporal_missing",
)


def debug_enabled(cfg: dict[str, Any]) -> bool:
    value = cfg.get("debug")
    if isinstance(value, dict):
        return bool(value.get("enabled", False))
    return bool(value or cfg.get("output", {}).get("debug", False))


def training_health_debug_enabled(cfg: dict[str, Any]) -> bool:
    debug_cfg = cfg.get("debug")
    if isinstance(debug_cfg, dict):
        value = debug_cfg.get("training_health", debug_cfg.get("enabled", False))
        return bool(value.get("enabled", False)) if isinstance(value, dict) else bool(value)
    return debug_enabled(cfg)


def write_config_diff_artifact(
    cfg: dict[str, Any],
    resolved_cfg: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any] | None:
    diff_cfg = _config_diff_cfg(cfg)
    if not bool(diff_cfg.get("enabled", False)):
        return None
    reference = diff_cfg.get("reference")
    if not reference:
        result = {"status": "missing_reference", "parity_passed": False}
    else:
        path = resolve_path(str(reference))
        if not path.exists():
            result = {"status": "missing_reference", "parity_passed": False, "reference": str(path)}
        else:
            result = compare_resolved_configs(_read_config_mapping(path), resolved_cfg, reference=str(path))
    _write_json(run_dir / "config_diff.json", result)
    return result


def compare_resolved_configs(
    reference_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    *,
    reference: str | None = None,
) -> dict[str, Any]:
    differences = _diff_mappings(reference_cfg, target_cfg)
    allowed = [item for item in differences if _allowed_identity_path(item["path"])]
    behavior = [item for item in differences if item not in allowed and not item["path"].startswith("debug")]
    critical = [item for item in behavior if item["path"] in CRITICAL_CONFIG_PATHS]
    parity_passed = not critical
    return {
        "status": "passed" if not behavior else "passed_with_noncritical_differences" if parity_passed else "failed",
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
    data_cfg = cfg.get("data", {})
    dataset_cfg = data_cfg.get("dataset", {})
    loader_cfg = data_cfg.get("dataloader", {})
    model_cfg = cfg.get("model", {})
    primary_cfg = model_cfg.get("primary", {}) if isinstance(model_cfg.get("primary"), dict) else {}
    architecture_summary = summarize_model_architecture(
        model,
        cfg=primary_cfg,
        source={"kind": "instance", "config_path": "startup_summary"},
    )
    return {
        "experiment": {
            "name": cfg.get("experiment", {}).get("name"),
            "task": cfg.get("experiment", {}).get("task"),
            "objective": cfg.get("experiment", {}).get("objective"),
            "seed": cfg.get("experiment", {}).get("seed"),
            "device": str(device),
        },
        "objective": objective_runtime_metadata(cfg),
        "data": {
            "modalities": list(primary_cfg.get("modalities") or model_cfg.get("modalities") or []),
            "dataset_type": dataset_cfg.get("type"),
            "dataset_path": dataset_cfg.get("data_root"),
            "train_split": dataset_cfg.get("train_csv_name"),
            "val_split": dataset_cfg.get("val_csv_name"),
            "test_split": dataset_cfg.get("test_csv_name"),
            "seq_len": dataset_cfg.get("seq_len"),
            "num_pred": dataset_cfg.get("num_pred"),
            "num_classes": model_cfg.get("num_classes") or primary_cfg.get("num_classes"),
            "batch_size": {
                "train": loader_cfg.get("train_batch_size"),
                "test": loader_cfg.get("test_batch_size"),
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
            "d_model": primary_cfg.get("d_model") or model_cfg.get("d_model"),
        },
        "parameters": module_trainability_report(model, architecture_summary=architecture_summary),
        "architecture_summary": architecture_summary,
    }


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
    modules = _tracked_modules(model)
    report = {}
    for name, module in modules.items():
        parameters = list(module.parameters())
        total = sum(parameter.numel() for parameter in parameters)
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        report[name] = {
            "path": name,
            "total_params": int(total),
            "trainable_params": int(trainable),
            "suspicious": bool(total and not trainable),
        }
    parameters = architecture_summary.get("parameters", {})
    return {
        "schema_version": architecture_summary.get("schema_version"),
        "total_params": int(parameters.get("total_params", 0)),
        "trainable_params": int(parameters.get("trainable_params", 0)),
        "frozen_params": int(parameters.get("frozen_params", 0)),
        "modules": report,
    }


class ModuleHealthTracker:
    def __init__(self, model: nn.Module) -> None:
        self.modules = _tracked_modules(model)
        self._snapshots: dict[str, list[torch.Tensor]] = {}
        self._grad_max: dict[str, float] = {}

    def start_epoch(self) -> None:
        self._grad_max = {name: 0.0 for name in self.modules}
        self._snapshots = {
            name: [parameter.detach().cpu().clone() for parameter in module.parameters() if parameter.requires_grad]
            for name, module in self.modules.items()
        }

    def observe_gradients(self) -> None:
        for name, module in self.modules.items():
            squared_norm = sum(
                float(parameter.grad.detach().float().pow(2).sum().item())
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            self._grad_max[name] = max(self._grad_max.get(name, 0.0), math.sqrt(squared_norm))

    def finish_epoch(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, module in self.modules.items():
            snapshots = iter(self._snapshots.get(name, []))
            squared_delta = sum(
                float((parameter.detach() - before.to(parameter)).float().pow(2).sum().item())
                for parameter in module.parameters()
                if parameter.requires_grad and (before := next(snapshots, None)) is not None
            )
            result[f"grad_norm_{name}"] = self._grad_max.get(name, 0.0)
            result[f"param_delta_{name}"] = math.sqrt(squared_delta)
        return result


def _tracked_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules = dict(model.named_children())
    return modules or {"model": model}


def _config_diff_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    debug_cfg = cfg.get("debug")
    value = debug_cfg.get("config_diff") if isinstance(debug_cfg, dict) else None
    return value if isinstance(value, dict) else {"enabled": bool(value)}


def _read_config_mapping(path: Path) -> dict[str, Any]:
    value = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _diff_mappings(reference: Any, target: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(reference, dict) and isinstance(target, dict):
        differences = []
        for key in sorted(set(reference) | set(target), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            differences.extend(_diff_mappings(reference.get(key), target.get(key), child_path))
        return differences
    if reference != target:
        return [{"path": path, "reference": reference, "target": target}]
    return []


def _allowed_identity_path(path: str) -> bool:
    return path in RUN_IDENTITY_PATHS or path.startswith("output.tensorboard")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


__all__ = [
    "ModuleHealthTracker",
    "build_startup_summary",
    "compare_resolved_configs",
    "debug_enabled",
    "module_trainability_report",
    "print_startup_summary",
    "training_health_debug_enabled",
    "write_config_diff_artifact",
    "write_startup_summary",
]
