from pathlib import Path

import numpy as np

from kd_sensing.engine.objectives.metadata import objective_tensorboard_scalars


def create_tensorboard_writer(cfg: dict, run_dir: Path):
    tensorboard_cfg = cfg.get("output", {}).get("tensorboard", {})
    if not tensorboard_cfg.get("enabled", True):
        return None

    log_dir = tensorboard_cfg.get("log_dir", "tensorboard") or "tensorboard"
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(run_dir / str(log_dir)))


def write_tensorboard_scalars(
    writer,
    history: dict,
    step: int,
    *,
    objective: str = "beam",
    tensorboard_cfg: dict | None = None,  # noqa: ARG001
) -> None:
    if writer is None:
        return

    writer.add_scalar("loss/train", history["train_loss"][-1], step)
    writer.add_scalar("loss/train_task", history["train_task_loss"][-1], step)
    writer.add_scalar("loss/val", history["val_loss"][-1], step)
    for tag, history_key in objective_tensorboard_scalars(objective):
        add_latest_scalar(writer, tag, history, history_key, step)
    writer.add_scalar("learning_rate/main", history["learning_rates"][-1], step)
    writer.flush()


def write_tensorboard_startup_scalars(writer, startup_summary: dict, step: int = 0) -> None:
    if writer is None:
        return
    parameters = startup_summary.get("parameters", {})
    optimization = startup_summary.get("optimization", {})
    data = startup_summary.get("data", {})
    batch_size = data.get("batch_size", {}) if isinstance(data.get("batch_size"), dict) else {}

    add_startup_scalar(writer, "run/start", 1.0, step)
    add_startup_scalar(writer, "model/total_params", parameters.get("total_params"), step)
    add_startup_scalar(writer, "model/trainable_params", parameters.get("trainable_params"), step)
    add_startup_scalar(writer, "training/max_epochs", optimization.get("max_epochs"), step)
    add_startup_scalar(writer, "data/train_batch_size", batch_size.get("train"), step)
    add_startup_scalar(writer, "data/test_batch_size", batch_size.get("test"), step)
    for module_name, module in (parameters.get("modules") or {}).items():
        if not isinstance(module, dict):
            continue
        add_startup_scalar(writer, f"model/modules/{module_name}/total_params", module.get("total_params"), step)
        add_startup_scalar(
            writer,
            f"model/modules/{module_name}/trainable_params",
            module.get("trainable_params"),
            step,
        )
    writer.flush()


def add_startup_scalar(writer, tag: str, value: object, step: int) -> None:
    numeric = finite_float_or_none(value)
    if numeric is None:
        return
    writer.add_scalar(tag, numeric, step)


def add_latest_scalar(writer, tag: str, history: dict, key: str, step: int) -> None:
    values = history.get(key)
    if not values:
        return
    value = finite_float_or_none(values[-1])
    if value is None:
        return
    writer.add_scalar(tag, value, step)


def finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def write_tensorboard_method_scalars(writer, epoch_log: dict, step: int) -> None:
    if writer is None:
        return
    for key, value in epoch_log.items():
        if not isinstance(value, (int, float)):
            continue
        if key.startswith(("loss/", "optimizer/", "val/subset/", "cmsbl/")):
            writer.add_scalar(key, float(value), step)
    writer.flush()


def close_tensorboard_writer(writer) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()
