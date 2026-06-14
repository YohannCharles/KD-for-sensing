from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from kd_sensing.modalities import normalize_modalities


AMR_NET_GPS_IMAGE_PRESET_NAME = "amr_net_gps_image"
AMR_NET_GPS_IMAGE_DISPLAY_NAME = "AMR-Net_gps_image"
AMR_NET_GPS_IMAGE_ALLOWED_MODALITIES = ("image", "gps")
DISALLOWED_MODALITIES = ("lidar", "radar", "mmwave", "csi")


@dataclass(frozen=True)
class PaperModelGroup:
    group_id: str
    display_name: str
    model_type: str
    enabled_modalities: tuple[str, ...]
    image_encoder: str | None
    gps_feature_mode: str | None
    fusion_type: str
    num_beams: int
    target_source: str
    metric_profile: str
    claim_status: str
    paper_reported_row: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": AMR_NET_GPS_IMAGE_DISPLAY_NAME,
            "model_group": self.group_id,
            "display_name": self.display_name,
            "model_type": self.model_type,
            "enabled_modalities": list(self.enabled_modalities),
            "image_encoder": self.image_encoder,
            "gps_feature_mode": self.gps_feature_mode,
            "gps_normalizer_provenance": "train_split_or_author_minmax_lat_lon_for_local_substitute",
            "fusion_type": self.fusion_type,
            "num_beams": self.num_beams,
            "target_source": self.target_source,
            "metric_profile": self.metric_profile,
            "claim_status": self.claim_status,
            "paper_reported_row": self.paper_reported_row,
            "uses_lidar": False,
        }


def paper_model_groups() -> tuple[PaperModelGroup, ...]:
    return (
        PaperModelGroup(
            group_id="image_only",
            display_name="AMR-Net_gps_image image-only CNN local substitute",
            model_type="modular_sequence",
            enabled_modalities=("image",),
            image_encoder="resnet18_imagenet_rgb",
            gps_feature_mode=None,
            fusion_type="none",
            num_beams=64,
            target_source="current",
            metric_profile="amr_net_gps_image_top1_top3_top5",
            claim_status="local_substitute",
            paper_reported_row=True,
        ),
        PaperModelGroup(
            group_id="gps_only",
            display_name="AMR-Net_gps_image GPS-only MLP local substitute",
            model_type="gps_sequence_baseline",
            enabled_modalities=("gps",),
            image_encoder=None,
            gps_feature_mode="paper_distance_angle",
            fusion_type="none",
            num_beams=64,
            target_source="current",
            metric_profile="amr_net_gps_image_top1_top3_top5",
            claim_status="local_substitute",
            paper_reported_row=True,
        ),
        PaperModelGroup(
            group_id="image_gps_fusion",
            display_name="AMR-Net_gps_image Image+GPS fusion local control",
            model_type="vision_position_late_fusion",
            enabled_modalities=("image", "gps"),
            image_encoder="resnet18_imagenet_rgb",
            gps_feature_mode="paper_distance_angle",
            fusion_type="late_concat_mlp",
            num_beams=64,
            target_source="current",
            metric_profile="amr_net_gps_image_top1_top3_top5",
            claim_status="local_control",
            paper_reported_row=False,
        ),
    )


def model_group(group_id: str) -> PaperModelGroup:
    groups = {group.group_id: group for group in paper_model_groups()}
    try:
        return groups[str(group_id)]
    except KeyError as exc:
        raise ValueError(f"Unknown {AMR_NET_GPS_IMAGE_DISPLAY_NAME} model group '{group_id}'. Available: {sorted(groups)}.") from exc


def build_model_group_config(group_id: str, *, smoke: bool = False) -> dict[str, Any]:
    group = model_group(group_id)
    feature_size = 8 if smoke else 64
    image_encoder = (
        {
            "type": "camera_ae_frozen",
            "latent_dim": 8 if smoke else 128,
            "output_dim": feature_size,
            "image_size": 64 if smoke else 224,
            "require_checkpoint": False,
            "freeze_encoder": True,
        }
        if smoke
        else {
            "type": "resnet18_imagenet_rgb",
            "output_dim": feature_size,
            "pretrained": False,
            "weights": None,
            "freeze_backbone": True,
        }
    )
    common_metadata = {
        "baseline_preset": AMR_NET_GPS_IMAGE_PRESET_NAME,
        "model_name": AMR_NET_GPS_IMAGE_DISPLAY_NAME,
        "paper_model_group": group.group_id,
        "target_source": group.target_source,
        "metric_profile": group.metric_profile,
        "claim_status": group.claim_status,
        "paper_reported_row": group.paper_reported_row,
        "uses_lidar": False,
    }
    if group.group_id == "image_only":
        return {
            "type": "modular_sequence",
            "modalities": ["image"],
            "feature_size": feature_size,
            "d_model": feature_size,
            "num_classes": group.num_beams,
            "num_pred": 1,
            "image_profile": "rgb_imagenet",
            "image_channels": 3,
            "encoders": {"image": image_encoder},
            "projectors": {"image": {"type": "linear", "d_model": feature_size}},
            "representation_core": {"type": "snapshot_frame", "d_model": feature_size, "output_dim": feature_size},
            "heads": {"beam": {"type": "beam_head", "dropout": 0.0}},
            "paper_metadata": group.metadata() | common_metadata,
        }
    if group.group_id == "gps_only":
        return {
            "type": "gps_sequence_baseline",
            "baseline_preset": AMR_NET_GPS_IMAGE_PRESET_NAME,
            "model_name": AMR_NET_GPS_IMAGE_DISPLAY_NAME,
            "paper_model_group": group.group_id,
            "gps_input_size": 2,
            "feature_size": feature_size,
            "hidden_size": feature_size,
            "temporal_model": "mlp",
            "num_classes": group.num_beams,
            "num_pred": 1,
            "history_length": 1,
            "dropout": 0.0,
            "gps_feature_mode": group.gps_feature_mode,
            "target_source": group.target_source,
            "metric_profile": group.metric_profile,
            "claim_status": group.claim_status,
            "paper_reported_row": group.paper_reported_row,
            "uses_lidar": False,
        }
    return {
        "type": "vision_position_late_fusion",
        "baseline_preset": AMR_NET_GPS_IMAGE_PRESET_NAME,
        "model_name": AMR_NET_GPS_IMAGE_DISPLAY_NAME,
        "paper_model_group": group.group_id,
        "modalities": ["image", "gps"],
        "feature_size": feature_size,
        "fusion_hidden_size": feature_size,
        "temporal_hidden_size": feature_size,
        "num_classes": group.num_beams,
        "num_pred": 1,
        "history_length": 1,
        "temporal_aggregation": "mean",
        "image_encoder_type": image_encoder["type"],
        "image_encoder": image_encoder,
        "gps_encoder": {"type": "gps_mlp", "output_dim": feature_size, "hidden_size": feature_size, "dropout": 0.0},
        "gps_input_size": 2,
        "dropout": 0.0,
        "gps_feature_mode": group.gps_feature_mode,
        "target_source": group.target_source,
        "metric_profile": group.metric_profile,
        "claim_status": group.claim_status,
        "paper_reported_row": group.paper_reported_row,
        "fusion_type": group.fusion_type,
        "uses_lidar": False,
    }


def paper_group_metadata(group_id: str, model_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    group = model_group(group_id)
    merged = group.metadata()
    if model_metadata:
        merged["model_training_strategy"] = dict(model_metadata)
    return merged


def is_amr_net_gps_image_preset_config(cfg: Mapping[str, Any]) -> bool:
    experiment = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), Mapping) else {}
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    paper_cfg = cfg.get("amr_net_gps_image", {}) if isinstance(cfg.get("amr_net_gps_image"), Mapping) else {}
    candidates = {
        experiment.get("name"),
        experiment.get("baseline_preset"),
        experiment.get("paper"),
        model.get("baseline_preset"),
        paper_cfg.get("preset"),
        paper_cfg.get("id"),
    }
    primary = model.get("primary") if isinstance(model.get("primary"), Mapping) else {}
    candidates.add(primary.get("baseline_preset"))
    candidates.add(primary.get("paper_model_group"))
    aliases = {AMR_NET_GPS_IMAGE_PRESET_NAME, AMR_NET_GPS_IMAGE_DISPLAY_NAME.lower(), "11282996"}
    return any(str(item or "").strip().lower() in aliases for item in candidates)


def validate_amr_net_gps_image_preset_config(cfg: Mapping[str, Any]) -> None:
    if not is_amr_net_gps_image_preset_config(cfg):
        return
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), Mapping) else {}
    primary = model.get("primary") if isinstance(model.get("primary"), Mapping) else {}
    selected_raw = model.get("modalities") or primary.get("modalities") or AMR_NET_GPS_IMAGE_ALLOWED_MODALITIES
    selected = normalize_modalities(tuple(selected_raw), context=f"{AMR_NET_GPS_IMAGE_DISPLAY_NAME} model.modalities")
    if selected != AMR_NET_GPS_IMAGE_ALLOWED_MODALITIES:
        raise ValueError(_boundary_message(f"enabled modalities must be ['image', 'gps'], got {list(selected)}"))
    dataset = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), Mapping) else {}
    enabled_modalities = dataset.get("enabled_modalities")
    if enabled_modalities:
        dataset_selected = normalize_modalities(
            tuple(enabled_modalities),
            context=f"{AMR_NET_GPS_IMAGE_DISPLAY_NAME} data.dataset.enabled_modalities",
        )
        if dataset_selected != AMR_NET_GPS_IMAGE_ALLOWED_MODALITIES:
            raise ValueError(_boundary_message(f"data.dataset.enabled_modalities={list(dataset_selected)}"))
    disallowed_flags = {
        "lidar": bool(dataset.get("use_lidar", False)),
        "mmwave": bool(dataset.get("use_mmwave", False)),
        "csi": bool(dataset.get("use_csi", False)),
    }
    enabled_disallowed = sorted(name for name, enabled in disallowed_flags.items() if enabled)
    if enabled_disallowed:
        raise ValueError(_boundary_message(f"disabled dataset flags were enabled: {enabled_disallowed}"))
    bad_modalities = sorted(set(selected) & set(DISALLOWED_MODALITIES))
    if bad_modalities:
        raise ValueError(_boundary_message(f"disallowed modalities: {bad_modalities}"))
    checkpoint_refs = list(_checkpoint_like_values(cfg))
    bad_refs = [value for value in checkpoint_refs if _looks_like_lidar_bgam(value)]
    if bad_refs:
        raise ValueError(_boundary_message(f"GPS+LiDAR BGAM checkpoint/reference is not allowed: {bad_refs[0]}"))


def _checkpoint_like_values(value: Any, *, key_path: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_checkpoint_like_values(item, key_path=(*key_path, str(key))))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_checkpoint_like_values(item, key_path=(*key_path, str(index))))
        return result
    key = ".".join(key_path).lower()
    if any(token in key for token in ("checkpoint", "ckpt", "weights")) and value not in (None, ""):
        return [str(value)]
    return []


def _looks_like_lidar_bgam(value: str) -> bool:
    text = str(value).replace("\\", "/").lower()
    return "gps_lidar_bgam" in text or "lidar_bgam" in text or "gps-lidar-bgam" in text


def _boundary_message(detail: str) -> str:
    return (
        f"{AMR_NET_GPS_IMAGE_DISPLAY_NAME} reproduction only allows image and GPS modalities; "
        f"{detail}. Remove LiDAR/radar/mmWave/CSI inputs and GPS+LiDAR BGAM checkpoints."
    )
