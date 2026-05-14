from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "export_viewer_manifest": (".viewer_manifest", "export_viewer_manifest"),
    "export_viewer_model_predictions": (".viewer_predictions", "export_viewer_model_predictions"),
    "build_case_table": (".complementarity", "build_case_table"),
    "compute_complementarity_bucket_summary": (".complementarity", "compute_bucket_summary"),
    "compute_complementarity_summary": (".complementarity", "compute_summary"),
    "load_subset_predictions": (".complementarity", "load_subset_predictions"),
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
