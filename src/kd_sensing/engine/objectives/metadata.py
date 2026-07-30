from typing import Any


_METRICS = ("val_loss", "val_acc", "val_adba", "val_atop3", "val_atop5")


def objective_available_metrics(metrics: dict[str, Any] | None = None) -> list[str]:
    return list(_METRICS) if metrics is None else [name for name in _METRICS if metrics.get(name) is not None]


def objective_runtime_metadata() -> dict[str, Any]:
    return {
        "name": "beam",
        "primary_loss": "beam_cross_entropy",
        "default_metric": "val_adba",
        "default_mode": "max",
        "enabled_targets": ["beam"],
        "enabled_heads": ["beam"],
        "loss_weights": {"beam": 1.0},
    }


__all__ = [
    "objective_available_metrics",
    "objective_runtime_metadata",
]
