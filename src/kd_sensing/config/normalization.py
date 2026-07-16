from typing import Any

from kd_sensing.modalities import dataset_defaults_for_modalities, dataset_flags_for_modalities, normalize_modalities, resolve_image_profile


def normalize_loaded_config(cfg: dict[str, Any]) -> None:
    normalize_temporal_window_missing_config(cfg)
    model = cfg.setdefault("model", {})
    primary = model.setdefault("primary", {})
    modalities = normalize_modalities(
        primary.get("modalities", model.get("modalities", ("image", "radar", "gps", "lidar"))),
        context="model.primary.modalities",
    )
    model["modalities"] = list(modalities)
    primary["modalities"] = list(modalities)
    dataset = cfg.setdefault("data", {}).setdefault("dataset", {})
    dataset.update(dataset_flags_for_modalities(modalities))
    for key, value in dataset_defaults_for_modalities(modalities).items():
        dataset.setdefault(key, value)
    image_profile = resolve_image_profile(primary.get("image_profile", model.get("image_profile", dataset.get("image_profile"))))
    dataset["image_profile"] = image_profile
    primary["image_profile"] = image_profile


def normalize_temporal_window_missing_config(cfg: dict[str, Any]) -> None:
    temporal = cfg.setdefault("temporal_missing", {})
    if not isinstance(temporal, dict):
        raise ValueError("temporal_missing must be a mapping.")
    dataset = cfg.setdefault("data", {}).setdefault("dataset", {})
    model = cfg.setdefault("model", {})
    primary = model.setdefault("primary", {})
    history = int(temporal.get("history_window", dataset.get("seq_len", model.get("seq_length", 5))))
    prediction = int(temporal.get("prediction_window", dataset.get("num_pred", model.get("num_pred", 1))))
    if history <= 0 or prediction <= 0:
        raise ValueError("history_window and prediction_window must be positive.")
    temporal.update({"history_window": history, "prediction_window": prediction, "seed": int(temporal.get("seed", 0))})
    dataset.update({"seq_len": history, "num_pred": prediction})
    model.update({"seq_length": history, "num_pred": prediction})
    primary.update({"seq_length": history, "num_pred": prediction})
