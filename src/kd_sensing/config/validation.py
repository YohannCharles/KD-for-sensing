from typing import Any

from kd_sensing.modalities import normalize_modalities, resolve_image_profile, validate_image_profile_size


RETAINED_MODALITIES = ("image", "radar", "gps", "lidar")
RETAINED_MODELS = {"u_mask_beam_jepa", "modular_sequence"}
RETAINED_DATASETS = {"mmw", "deepsense6g"}
DEEPSENSE6G_SCENES = {31, 32, 33, 34}


def validate_loaded_config(cfg: dict[str, Any]) -> None:
    data = cfg.get("data", {}).get("dataset", {})
    model = cfg.get("model", {}).get("primary", {})
    if cfg.get("preprocessing"):
        return
    dataset_type = str(data.get("type", "")).strip().lower()
    if dataset_type not in RETAINED_DATASETS:
        raise ValueError(f"Supported data.dataset.type values are {sorted(RETAINED_DATASETS)}.")
    modalities = normalize_modalities(model.get("modalities", ()), context="model.primary.modalities")
    if modalities != RETAINED_MODALITIES:
        raise ValueError(f"Retained workflows require modalities {list(RETAINED_MODALITIES)}.")
    if str(model.get("type", "")) not in RETAINED_MODELS:
        raise ValueError(f"Only retained model types are supported: {sorted(RETAINED_MODELS)}.")
    if int(data.get("seq_len", 0)) <= 0 or int(data.get("num_pred", 0)) <= 0:
        raise ValueError("data.dataset.seq_len and num_pred must be positive.")
    image_profile = resolve_image_profile(data.get("image_profile"))
    validate_image_profile_size(image_profile, tuple(data.get("image_size", (224, 224))))
    fft = tuple(data.get("fft_tuple", ()))
    if len(fft) < 3 or int(fft[0]) != 64 or int(fft[2]) != 128 or int(data.get("clipped_range", 0)) != 128:
        raise ValueError("Retained radar inputs require fft_tuple [64, *, 128] and clipped_range=128.")
    if dataset_type == "deepsense6g":
        scene = data.get("scene")
        if type(scene) is not int or scene not in DEEPSENSE6G_SCENES:
            raise ValueError(f"DeepSense6G scene must be one of {sorted(DEEPSENSE6G_SCENES)}, got {scene!r}.")
        if data.get("domains"):
            raise ValueError("DeepSense6G does not support data.dataset.domains.")
        if data.get("gps_normalize") is not True:
            raise ValueError("DeepSense6G requires train-fitted GPS normalization.")
        if int(data.get("num_pred", 0)) > 3:
            raise ValueError("DeepSense6G supports at most three future_beam horizons.")
        if int(model.get("num_classes", 0)) != 64:
            raise ValueError("DeepSense6G future_beam labels require model.primary.num_classes=64.")
