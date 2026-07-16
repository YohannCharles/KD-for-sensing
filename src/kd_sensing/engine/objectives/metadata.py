from typing import Any


_METRICS = ("val_loss", "val_acc", "val_adba", "val_atop3", "val_atop5")


def resolve_prediction_objective(cfg: dict[str, Any]) -> str:
    value = str(cfg.get("experiment", {}).get("objective", "beam")).strip().lower()
    if value != "beam":
        raise ValueError("Only experiment.objective='beam' is retained.")
    return value


def objective_tensorboard_scalars(*_args: Any, **_kwargs: Any) -> tuple[tuple[str, str], ...]:
    return (("accuracy/train", "train_acc"), ("accuracy/val", "val_acc"), ("beam/val_adba", "val_adba"))


def objective_available_metrics(_objective: str, metrics: dict[str, Any] | None = None) -> list[str]:
    return list(_METRICS) if metrics is None else [name for name in _METRICS if metrics.get(name) is not None]


def objective_runtime_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    resolve_prediction_objective(cfg)
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
    "objective_tensorboard_scalars",
    "resolve_prediction_objective",
]
