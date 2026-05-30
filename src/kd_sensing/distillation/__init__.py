from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "FocalLoss": (".losses", "FocalLoss"),
    "KnowledgeDistillationLoss": (".distillers", "KnowledgeDistillationLoss"),
    "NoKDDistiller": (".distillers", "NoKDDistiller"),
    "LogitsKDDistiller": (".distillers", "LogitsKDDistiller"),
    "RKDDistiller": (".distillers", "RKDDistiller"),
}

_REMOVED_EXPORTS = {
    "G2DDistiller": "G2D multimodal distillation has been retired; use no_kd, logits_kd, or rkd.",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _REMOVED_EXPORTS:
        raise AttributeError(f"{name} has been removed. {_REMOVED_EXPORTS[name]}")
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_REMOVED_EXPORTS))
