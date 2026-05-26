from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "export_viewer_manifest": (".viewer_manifest", "export_viewer_manifest"),
    "export_viewer_model_predictions": (".viewer_predictions", "export_viewer_model_predictions"),
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
