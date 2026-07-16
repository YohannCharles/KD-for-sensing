from typing import Any

from kd_sensing.modalities import normalize_modalities, resolve_image_profile, validate_image_profile_size


RETAINED_MODALITIES = ("image", "radar", "gps", "lidar")
RETAINED_MODELS = {"u_mask_beam_jepa", "modular_sequence"}


def validate_loaded_config(cfg: dict[str, Any]) -> None:
    data = cfg.get("data", {}).get("dataset", {})
    model = cfg.get("model", {}).get("primary", {})
    if cfg.get("preprocessing"):
        return
    if str(data.get("type", "")) != "mmw":
        raise ValueError("Only data.dataset.type='mmw' is retained.")
    modalities = normalize_modalities(model.get("modalities", ()), context="model.primary.modalities")
    if modalities != RETAINED_MODALITIES:
        raise ValueError(f"Retained MMW workflows require modalities {list(RETAINED_MODALITIES)}.")
    if str(model.get("type", "")) not in RETAINED_MODELS:
        raise ValueError(f"Only retained model types are supported: {sorted(RETAINED_MODELS)}.")
    if int(data.get("seq_len", 0)) <= 0 or int(data.get("num_pred", 0)) <= 0:
        raise ValueError("data.dataset.seq_len and num_pred must be positive.")
    image_profile = resolve_image_profile(data.get("image_profile"))
    validate_image_profile_size(image_profile, tuple(data.get("image_size", (224, 224))))
    fft = tuple(data.get("fft_tuple", ()))
    if len(fft) < 3 or int(fft[0]) != 64 or int(fft[2]) != 128 or int(data.get("clipped_range", 0)) != 128:
        raise ValueError("MMW radar requires fft_tuple [64, *, 128] and clipped_range=128.")
