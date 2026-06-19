from __future__ import annotations

import datetime as dt
import random
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from kd_sensing.data.scenes import resolve_deepsense_scene
from kd_sensing.data.transform_ops.gps import (
    GPS_FEATURE_DIMS,
    GPSStandardScaler,
    PAPER_CALIBRATED_GPS_MODE,
    PAPER_DISTANCE_ANGLE_FEATURE_VERSION,
    PAPER_SCENE_CENTER_ANGLES_RAD,
)
from kd_sensing.utils.paths import resolve_path


TARGET_TABLE_III_ROW = {
    "camera": "AE",
    "radar": "none",
    "lidar": "none",
    "gps": "Direct",
    "fusion": True,
    "scene31": 0.6731,
    "scene32": 0.6173,
    "scene33": 0.8171,
    "scene34": 0.7313,
    "overall": 0.7127,
}

OFFICIAL_DENSE_MODEL_CFG = {
    "lin1_size": 128,
    "lin2_size": 256,
    "lin3_size": 512,
    "lin4_size": 128,
    "act_func": "LeakyReLU",
    "last_act_func": "Sigmoid",
    "loss_func": "BCE",
}

@dataclass(frozen=True)
class ImageAEGPSDirectTrainingConfig:
    data_root: str
    train_csv_name: str = "train_seqs_RA_GPS_LIDAR.csv"
    test_csv_name: str = "test_seqs_RA_GPS_LIDAR.csv"
    output_dir: str = "outputs/scene31/beambench_image_ae_gps_direct"
    scene: int = 31
    seq_len: int = 1
    gps_seq_len: int | None = None
    gps_source_seq_len: int | None = None
    gps_input_seq_len: int | None = None
    num_pred: int = 1
    num_beams: int = 64
    target_beam_source: str = "current"
    image_size: int = 64
    image_channels: int = 3
    gps_input_size: int = 2
    gps_feature_mode: str = PAPER_CALIBRATED_GPS_MODE
    gps_angle_offset_rad: float | None = None
    gps_angle_offset_source: str = "paper_scene_default"
    gps_normalize: bool = True
    train_portion: float = 1.0
    test_portion: float = 1.0
    portion_strategy: str = "even"
    portion_seed: int = 42
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    ae_checkpoint_path: str | None = None
    auto_train_ae: bool = True
    ae_epochs: int = 20
    ae_batch_size: int = 64
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-4
    ae_val_fraction: float = 0.1
    ae_patience: int = 5
    ae_latent_dim: int = 128
    fusion_epochs: int = 80
    fusion_batch_size: int = 64
    fusion_lr: float = 5e-4
    fusion_weight_decay: float = 1e-4
    fusion_patience: int = 15
    fusion_val_fraction: float = 0.0
    selection_split: str = "test_as_validation"
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.2
    fusion_architecture: str = "official_dense_model"
    fusion_loss: str = "bce"
    fusion_activation: str = "LeakyReLU"
    fusion_last_activation: str = "Sigmoid"
    fusion_dense_hidden_sizes: tuple[int, int, int, int] = (128, 256, 512, 128)
    freeze_ae_encoder: bool = True
    dba_delta: float = 5.0
    topk: tuple[int, ...] = (1, 3, 5)
    seed: int = 42
    device: str = "auto"
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    non_blocking_transfer: bool = True
    amp: bool = True
    amp_dtype: str = "float16"
    amp_grad_scaler: bool = True
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    fused_optimizer: bool = True
    cache_frozen_ae_features: bool = True
    feature_cache_batch_size: int = 256
    feature_cache_dir: str | None = None
    save_predictions: bool = True
    dry_run: bool = False

def resolve_image_ae_gps_config(raw: Mapping[str, Any]) -> ImageAEGPSDirectTrainingConfig:
    experiment = _mapping(raw.get("experiment"))
    data = _mapping(raw.get("data"))
    dataset = _mapping(data.get("dataset"))
    loader = _mapping(data.get("dataloader"))
    model = _mapping(raw.get("model"))
    primary = _mapping(model.get("primary"))
    encoders = _mapping(primary.get("encoders"))
    image_encoder = _mapping(encoders.get("image"))
    gps_encoder = _mapping(encoders.get("gps"))
    training = _mapping(raw.get("training"))
    output = _mapping(raw.get("output"))
    paper = _mapping(raw.get("beambench_paper"))

    scene_value = dataset.get("scene", 31)
    scene = resolve_deepsense_scene(scene_value)
    data_root = str(dataset.get("data_root") or scene.default_data_root)
    output_dir = paper.get("output_dir")
    if not output_dir:
        output_dir = Path(str(output.get("dir", "outputs"))) / scene.scene_slug / "beambench_image_ae_gps_direct"
        run_name = str(output.get("run_name") or "").strip()
        if run_name:
            output_dir = Path(str(output.get("dir", "outputs"))) / scene.scene_slug / run_name

    dry_run = bool(paper.get("dry_run", False))
    max_train_samples = _optional_int(paper.get("max_train_samples"))
    max_test_samples = _optional_int(paper.get("max_test_samples"))
    if dry_run:
        max_train_samples = max_train_samples or 4
        max_test_samples = max_test_samples or 4

    topk_raw = paper.get("topk", raw.get("metrics", {}).get("topk") if isinstance(raw.get("metrics"), Mapping) else None)
    topk = tuple(int(item) for item in (topk_raw or (1, 3, 5)))
    ae_epochs = int(paper.get("ae_epochs", 20))
    ae_patience = int(paper.get("ae_patience", 5))
    fusion_epochs = int(paper.get("fusion_epochs", training.get("epochs", 80)))
    fusion_patience = int(paper.get("fusion_patience", training.get("patience", 15)))
    num_workers = int(loader.get("num_workers", 0))
    prefetch_factor = _optional_int(loader.get("prefetch_factor", 2))
    pin_memory = _bool(loader.get("pin_memory", True), default=True)
    persistent_workers = _bool(loader.get("persistent_workers", True), default=True)
    transfer_cfg = _mapping(training.get("transfer"))
    amp_cfg = _mapping(training.get("amp"))
    if dry_run:
        ae_epochs = 1
        ae_patience = 1
        fusion_epochs = 1
        fusion_patience = 1
        num_workers = 0
        prefetch_factor = None
        persistent_workers = False
    gps_feature_mode = _normalize_gps_feature_mode(
        str(paper.get("gps_feature_mode", dataset.get("gps_feature_mode", PAPER_CALIBRATED_GPS_MODE)))
    )
    gps_angle_offset_rad, gps_angle_offset_source = _resolve_gps_angle_offset(
        scene=int(scene.scene_id),
        feature_mode=gps_feature_mode,
        explicit_value=paper.get("gps_angle_offset_rad"),
    )

    return ImageAEGPSDirectTrainingConfig(
        data_root=data_root,
        train_csv_name=str(dataset.get("train_csv_name", "train_seqs_RA_GPS_LIDAR.csv")),
        test_csv_name=str(dataset.get("test_csv_name", "test_seqs_RA_GPS_LIDAR.csv")),
        output_dir=str(output_dir),
        scene=int(scene.scene_id),
        seq_len=int(dataset.get("seq_len", 1)),
        gps_seq_len=_optional_int(dataset.get("gps_seq_len", paper.get("gps_seq_len"))),
        gps_source_seq_len=_optional_int(
            dataset.get("gps_source_seq_len", paper.get("gps_source_seq_len"))
        ),
        gps_input_seq_len=_optional_int(dataset.get("gps_input_seq_len", paper.get("gps_input_seq_len"))),
        num_pred=int(dataset.get("num_pred", 1)),
        num_beams=int(model.get("num_classes", primary.get("num_classes", 64))),
        target_beam_source=_normalize_target_beam_source(str(paper.get("target_beam_source", "current"))),
        image_size=int(paper.get("ae_image_size", image_encoder.get("image_size", 64))),
        image_channels=int(primary.get("image_channels", 3)),
        gps_input_size=_resolve_gps_input_size(
            paper=paper,
            primary=primary,
            gps_encoder=gps_encoder,
            gps_feature_mode=gps_feature_mode,
        ),
        gps_feature_mode=gps_feature_mode,
        gps_angle_offset_rad=gps_angle_offset_rad,
        gps_angle_offset_source=gps_angle_offset_source,
        gps_normalize=bool(dataset.get("gps_normalize", True)),
        train_portion=float(paper.get("train_portion", dataset.get("portion", 1.0))),
        test_portion=float(paper.get("test_portion", dataset.get("portion", 1.0))),
        portion_strategy=str(dataset.get("portion_strategy", "even")),
        portion_seed=int(dataset.get("portion_seed", experiment.get("seed", 42))),
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        ae_checkpoint_path=_optional_str(paper.get("ae_checkpoint_path", image_encoder.get("checkpoint_path"))),
        auto_train_ae=bool(paper.get("auto_train_ae", True)),
        ae_epochs=ae_epochs,
        ae_batch_size=int(paper.get("ae_batch_size", loader.get("train_batch_size", 64))),
        ae_lr=float(paper.get("ae_lr", 1e-3)),
        ae_weight_decay=float(paper.get("ae_weight_decay", training.get("weight_decay", 1e-4))),
        ae_val_fraction=float(paper.get("ae_val_fraction", 0.1)),
        ae_patience=ae_patience,
        ae_latent_dim=int(image_encoder.get("latent_dim", paper.get("ae_latent_dim", 128))),
        fusion_epochs=fusion_epochs,
        fusion_batch_size=int(paper.get("fusion_batch_size", loader.get("train_batch_size", 64))),
        fusion_lr=float(paper.get("fusion_lr", training.get("lr", 5e-4))),
        fusion_weight_decay=float(paper.get("fusion_weight_decay", training.get("weight_decay", 1e-4))),
        fusion_patience=fusion_patience,
        fusion_val_fraction=float(paper.get("fusion_val_fraction", 0.0)),
        selection_split=_normalize_selection_split(str(paper.get("selection_split", "test_as_validation"))),
        fusion_hidden_dim=int(paper.get("fusion_hidden_dim", primary.get("d_model", model.get("d_model", 256)))),
        fusion_dropout=float(paper.get("fusion_dropout", training.get("dropout", 0.2))),
        fusion_architecture=_normalize_fusion_architecture(str(paper.get("fusion_architecture", "official_dense_model"))),
        fusion_loss=str(paper.get("fusion_loss", OFFICIAL_DENSE_MODEL_CFG["loss_func"])).lower(),
        fusion_activation=str(paper.get("fusion_activation", OFFICIAL_DENSE_MODEL_CFG["act_func"])),
        fusion_last_activation=str(paper.get("fusion_last_activation", OFFICIAL_DENSE_MODEL_CFG["last_act_func"])),
        fusion_dense_hidden_sizes=_fusion_dense_hidden_sizes(paper.get("fusion_dense_hidden_sizes")),
        freeze_ae_encoder=bool(image_encoder.get("freeze_encoder", paper.get("freeze_ae_encoder", True))),
        dba_delta=float(paper.get("dba_delta", raw.get("evaluation", {}).get("dba_delta", 5.0) if isinstance(raw.get("evaluation"), Mapping) else 5.0)),
        topk=topk,
        seed=int(experiment.get("seed", 42)),
        device=str(experiment.get("device", paper.get("device", "auto"))),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        non_blocking_transfer=_bool(
            paper.get("non_blocking_transfer", transfer_cfg.get("non_blocking", True)),
            default=True,
        ),
        amp=_bool(paper.get("amp", amp_cfg.get("enabled", True)), default=True),
        amp_dtype=str(paper.get("amp_dtype", amp_cfg.get("dtype", "float16"))),
        amp_grad_scaler=_bool(paper.get("amp_grad_scaler", amp_cfg.get("grad_scaler", True)), default=True),
        allow_tf32=_bool(paper.get("allow_tf32", training.get("allow_tf32", True)), default=True),
        cudnn_benchmark=_bool(paper.get("cudnn_benchmark", training.get("cudnn_benchmark", True)), default=True),
        fused_optimizer=_bool(paper.get("fused_optimizer", training.get("fused_optimizer", True)), default=True),
        cache_frozen_ae_features=_bool(paper.get("cache_frozen_ae_features", True), default=True),
        feature_cache_batch_size=int(paper.get("feature_cache_batch_size", loader.get("test_batch_size", 256))),
        feature_cache_dir=_optional_str(paper.get("feature_cache_dir")),
        save_predictions=bool(paper.get("save_predictions", True)),
        dry_run=dry_run,
    )

def _normalize_selection_split(value: str) -> str:
    normalized = str(value or "test_as_validation").strip().lower().replace("-", "_")
    if normalized in {"test", "test_as_val", "test_as_validation"}:
        return "test_as_validation"
    if normalized in {"val", "valid", "validation", "train_validation"}:
        return "validation"
    raise ValueError("beambench_paper.selection_split must be 'test_as_validation' or 'validation'.")

def _normalize_fusion_architecture(value: str) -> str:
    normalized = str(value or "official_dense_model").strip().lower().replace("-", "_")
    if normalized in {"official_dense", "official_dense_model", "beambench_dense", "dense_model"}:
        return "official_dense_model"
    if normalized in {"legacy", "legacy_layernorm_gelu", "project_mlp"}:
        return "legacy_layernorm_gelu"
    raise ValueError("beambench_paper.fusion_architecture must be 'official_dense_model' or 'legacy_layernorm_gelu'.")

def _official_dense_activation(name: str) -> nn.Module:
    normalized = str(name or "Linear").strip().lower().replace("_", "").replace("-", "")
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "leakyrelu":
        return nn.LeakyReLU()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "softplus":
        return nn.Softplus()
    if normalized in {"linear", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported BeamBench dense_model activation: {name}")

def _fusion_dense_hidden_sizes(value: Any) -> tuple[int, int, int, int]:
    if value in (None, ""):
        return (
            int(OFFICIAL_DENSE_MODEL_CFG["lin1_size"]),
            int(OFFICIAL_DENSE_MODEL_CFG["lin2_size"]),
            int(OFFICIAL_DENSE_MODEL_CFG["lin3_size"]),
            int(OFFICIAL_DENSE_MODEL_CFG["lin4_size"]),
        )
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    else:
        items = list(value)
    sizes = tuple(int(item) for item in items)
    if len(sizes) != 4:
        raise ValueError("beambench_paper.fusion_dense_hidden_sizes must contain exactly four integers.")
    return sizes  # type: ignore[return-value]

def _normalize_target_beam_source(value: str) -> str:
    normalized = str(value or "current").strip().lower().replace("-", "_")
    if normalized in {"current", "current_beam", "beam", "beam_last", "last_beam"}:
        return "current"
    if normalized in {"future", "future_beam", "future_beam1", "next"}:
        return "future"
    raise ValueError("beambench_paper.target_beam_source must be 'current' or 'future'.")

def _normalize_gps_feature_mode(value: str) -> str:
    normalized = str(value or PAPER_CALIBRATED_GPS_MODE).strip().lower().replace("-", "_")
    if normalized in {"relative_polar", "raw_relative_polar"}:
        return "relative_polar"
    if normalized in {
        "paper_calibrated_relative_polar",
        "calibrated_relative_polar",
        "paper_calibrated_polar",
        "paper_centered_relative_polar",
    }:
        return "paper_calibrated_relative_polar"
    if normalized in {"paper_distance_angle", "distance_angle", "paper_gt_pos", "official_gps"}:
        return "paper_distance_angle"
    raise ValueError(
        "beambench_paper.gps_feature_mode must be 'relative_polar', "
        "'paper_calibrated_relative_polar', or 'paper_distance_angle'."
    )

def _resolve_gps_input_size(
    *,
    paper: Mapping[str, Any],
    primary: Mapping[str, Any],
    gps_encoder: Mapping[str, Any],
    gps_feature_mode: str,
) -> int:
    if "gps_input_size" in paper:
        return int(paper["gps_input_size"])
    mode = _normalize_gps_feature_mode(gps_feature_mode)
    configured = primary.get("gps_input_size", gps_encoder.get("gps_input_size"))
    if mode in GPS_FEATURE_DIMS:
        return int(GPS_FEATURE_DIMS[mode])
    return int(configured or 3)

def _resolve_gps_angle_offset(
    *,
    scene: int,
    feature_mode: str,
    explicit_value: Any,
) -> tuple[float | None, str]:
    mode = _normalize_gps_feature_mode(feature_mode)
    if explicit_value not in (None, ""):
        return float(explicit_value), "config"
    if mode in {PAPER_CALIBRATED_GPS_MODE, "paper_calibrated_relative_polar"}:
        try:
            return float(PAPER_SCENE_CENTER_ANGLES_RAD[int(scene)]), "paper_scene_default"
        except KeyError as exc:
            raise ValueError(f"Paper GPS calibration is only defined for scenes 31-34, got scene {scene}.") from exc
    return None, "none"

def _scene_specific_cfg(base_cfg: ImageAEGPSDirectTrainingConfig, scene: int) -> ImageAEGPSDirectTrainingConfig:
    scene_obj = resolve_deepsense_scene(scene)
    base_scene_obj = resolve_deepsense_scene(base_cfg.scene)
    try:
        base_root = resolve_path(base_cfg.data_root)
        base_default = resolve_path(base_scene_obj.default_data_root)
        data_root = str(scene_obj.default_data_root) if base_root == base_default else str(base_root)
    except Exception:
        data_root = str(scene_obj.default_data_root)
    gps_angle_offset_rad = base_cfg.gps_angle_offset_rad
    gps_angle_offset_source = base_cfg.gps_angle_offset_source
    if base_cfg.gps_angle_offset_source == "paper_scene_default":
        gps_angle_offset_rad, gps_angle_offset_source = _resolve_gps_angle_offset(
            scene=int(scene_obj.scene_id),
            feature_mode=base_cfg.gps_feature_mode,
            explicit_value=None,
        )
    return replace(
        base_cfg,
        scene=int(scene_obj.scene_id),
        data_root=data_root,
        output_dir=str(Path(base_cfg.output_dir).parent / scene_obj.scene_slug),
        gps_angle_offset_rad=gps_angle_offset_rad,
        gps_angle_offset_source=gps_angle_offset_source,
    )

def _gps_feature_version(mode: str) -> str:
    normalized = _normalize_gps_feature_mode(mode)
    if normalized == PAPER_CALIBRATED_GPS_MODE:
        return PAPER_DISTANCE_ANGLE_FEATURE_VERSION
    return "default"

def _gps_scaler_metadata(scaler: GPSStandardScaler | None) -> dict[str, list[float]] | None:
    if scaler is None:
        return None
    if scaler.mean_ is None or scaler.scale_ is None:
        return None
    return {
        "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
    }

def _gps_scaler_from_metadata(payload: Any) -> GPSStandardScaler | None:
    if not isinstance(payload, Mapping):
        return None
    if "mean" not in payload or "scale" not in payload:
        return None
    return GPSStandardScaler(
        mean_=np.asarray(payload["mean"], dtype=np.float64),
        scale_=np.asarray(payload["scale"], dtype=np.float64),
    )

def _gps_calibration_metadata(cfg: ImageAEGPSDirectTrainingConfig) -> dict[str, Any]:
    if cfg.gps_feature_mode == "paper_distance_angle":
        note = (
            "paper_distance_angle follows the official challenge.py GPS Direct input: "
            "distance plus scene-calibrated angle in degrees."
        )
    elif cfg.gps_feature_mode == "paper_calibrated_relative_polar":
        note = (
            "paper_calibrated_relative_polar subtracts the paper's scene-specific "
            "boresight angle before encoding GPS as distance/sin/cos."
        )
    else:
        note = "relative_polar encodes GPS as distance/sin/cos without paper boresight calibration."
    return {
        "gps_feature_mode": str(cfg.gps_feature_mode),
        "gps_angle_offset_rad": None if cfg.gps_angle_offset_rad is None else float(cfg.gps_angle_offset_rad),
        "gps_angle_offset_source": str(cfg.gps_angle_offset_source),
        "paper_scene_center_angles_rad": dict(PAPER_SCENE_CENTER_ANGLES_RAD),
        "note": note,
    }

def _configure_torch_runtime(cfg: ImageAEGPSDirectTrainingConfig, device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "device": str(device),
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "float32_matmul_precision": None,
    }
    if device.type != "cuda":
        return report
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
        report["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        report["allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    if bool(cfg.allow_tf32) and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        report["float32_matmul_precision"] = "high"
    return report

def _resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = str(name or "float16").lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError("beambench_paper.amp_dtype must be 'float16' or 'bfloat16'.")

def _autocast_context(enabled: bool, device: torch.device, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)

def _make_grad_scaler(cfg: ImageAEGPSDirectTrainingConfig, amp_enabled: bool):
    enabled = bool(amp_enabled) and bool(cfg.amp_grad_scaler)
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

def _scaler_enabled(grad_scaler: Any) -> bool:
    return grad_scaler is not None and bool(getattr(grad_scaler, "is_enabled", lambda: False)())

def _build_adamw(
    params: Any,
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
    fused: bool,
) -> torch.optim.Optimizer:
    materialized = list(params)
    kwargs: dict[str, Any] = {}
    if bool(fused) and device.type == "cuda":
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(materialized, lr=lr, weight_decay=weight_decay, **kwargs)
    except TypeError:
        return torch.optim.AdamW(materialized, lr=lr, weight_decay=weight_decay)

def _torch_load(path: str | Path, *, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _resolve_device(value: str) -> torch.device:
    requested = str(value or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}

def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)

def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)

def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    if normalized == "auto":
        return bool(default)
    return bool(value)

def timestamped_default_output(scene: int | str) -> str:
    scene_obj = resolve_deepsense_scene(scene)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"outputs/{scene_obj.scene_slug}/beambench_image_ae_gps_direct/{stamp}"


__all__ = [
    "ImageAEGPSDirectTrainingConfig",
    "OFFICIAL_DENSE_MODEL_CFG",
    "TARGET_TABLE_III_ROW",
    "resolve_image_ae_gps_config",
    "timestamped_default_output",
]
