from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "build_dataloaders": (".builders", "build_dataloaders"),
    "build_model": (".builders", "build_model"),
    "build_task_criterion": (".builders", "build_task_criterion"),
    "build_distiller": (".builders", "build_distiller"),
    "build_metrics": (".builders", "build_metrics"),
    "train": (".trainer", "train"),
    "validate": (".validator", "validate"),
    "evaluate": (".evaluator", "evaluate"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
