from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.models.fusion import StudentModalityNet  # noqa: E402
from kd_sensing.models.gps import GpsModalityNet, GpsStudentModalityNet  # noqa: E402
from kd_sensing.models.image import ImageStudentModalityNet  # noqa: E402
from kd_sensing.models.radar import RadarModalityNet, RadarStudentModalityNet  # noqa: E402
from kd_sensing.registries import MODELS  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


IMAGE_CONFIGS = [
    "configs/image/no_kd.yaml",
    "configs/image/logits_kd.yaml",
    "configs/image/rkd.yaml",
]

FUSION_CONFIGS = [
    "configs/fusion/no_kd.yaml",
    "configs/fusion/logits_kd.yaml",
    "configs/fusion/rkd.yaml",
]

RADAR_STUDENT_CONFIGS = [
    "configs/radar/student_no_kd.yaml",
    "configs/radar/logits_kd.yaml",
    "configs/radar/rkd.yaml",
]

RADAR_KD_CONFIGS = [
    ("configs/radar/logits_kd.yaml", "logits_kd"),
    ("configs/radar/rkd.yaml", "rkd"),
]

GPS_STUDENT_CONFIGS = [
    "configs/gps/student_no_kd.yaml",
    "configs/gps/logits_kd.yaml",
    "configs/gps/rkd.yaml",
]

GPS_KD_CONFIGS = [
    ("configs/gps/logits_kd.yaml", "logits_kd"),
    ("configs/gps/rkd.yaml", "rkd"),
]

GPS_REL_POLAR_CONFIGS = [
    "configs/gps/no_kd.yaml",
    "configs/gps/student_no_kd.yaml",
    "configs/gps/logits_kd.yaml",
    "configs/gps/rkd.yaml",
    "configs/gps/ablation_relative_polar.yaml",
]

GPS_FUSION_CONFIGS = [
    "configs/fusion/image_gps_no_kd.yaml",
    "configs/fusion/radar_gps_no_kd.yaml",
    "configs/fusion/all_modalities_no_kd.yaml",
]

LEGACY_STUDENT_WEIGHTS = [
    ("configs/image/no_kd.yaml", "All_models/ImageStd_noKD.pth"),
    ("configs/image/logits_kd.yaml", "All_models/ImageStd_KD.pth"),
    ("configs/image/rkd.yaml", "All_models/ImageStd_RKD.pth"),
    ("configs/fusion/no_kd.yaml", "All_models/BothStd_noKD.pth"),
    ("configs/fusion/logits_kd.yaml", "All_models/BothStd_KD.pth"),
    ("configs/fusion/rkd.yaml", "All_models/BothStd_RKD.pth"),
]


def _build_student(config_path: str):
    cfg = load_config(ROOT / config_path)
    return MODELS.build(cfg["model"]["student"]), cfg


def _build_teacher_and_student(config_path: str):
    cfg = load_config(ROOT / config_path)
    return MODELS.build(cfg["model"]["teacher"]), MODELS.build(cfg["model"]["student"]), cfg


def _load_state_dict(weight_path: str) -> dict[str, torch.Tensor]:
    path = ROOT / weight_path
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def _is_stats_key(key: str) -> bool:
    return key.endswith("total_ops") or key.endswith("total_params")


@pytest.mark.parametrize("config_path", IMAGE_CONFIGS)
def test_image_configs_build_lightweight_student(config_path: str):
    model, cfg = _build_student(config_path)

    assert cfg["model"]["student"]["type"] == "image_student"
    assert isinstance(model, ImageStudentModalityNet)
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 2]
    assert model.GRU.num_layers == 2


@pytest.mark.parametrize("config_path", FUSION_CONFIGS)
def test_fusion_configs_build_lightweight_student(config_path: str):
    model, cfg = _build_student(config_path)

    assert cfg["model"]["teacher"]["type"] == "fusion_teacher"
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["type"] == "fusion_student"
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["modalities"] == ["image", "radar"]
    assert isinstance(model, StudentModalityNet)
    assert model.modalities == ("image", "radar")
    assert model.GRU.num_layers == 2


def test_radar_teacher_baseline_config_builds_teacher_model():
    config_path = "configs/radar/no_kd.yaml"
    model, cfg = _build_student(config_path)

    assert cfg["experiment"]["task"] == "radar"
    assert cfg["model"]["teacher"]["type"] == "radar_teacher"
    assert cfg["model"]["student"]["type"] == "radar_teacher"
    assert isinstance(model, RadarModalityNet)
    assert model.GRU.num_layers == 2


@pytest.mark.parametrize("config_path", RADAR_STUDENT_CONFIGS)
def test_radar_student_configs_build_lightweight_student(config_path: str):
    model, cfg = _build_student(config_path)

    assert cfg["experiment"]["task"] == "radar"
    assert cfg["model"]["teacher"]["type"] == "radar_teacher"
    assert cfg["model"]["student"]["type"] == "radar_student"
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 2]
    assert isinstance(model, RadarStudentModalityNet)
    assert model.GRU.num_layers == 2


@pytest.mark.parametrize(("config_path", "kd_type"), RADAR_KD_CONFIGS)
def test_radar_kd_configs_build_teacher_and_student(config_path: str, kd_type: str):
    teacher, student, cfg = _build_teacher_and_student(config_path)

    assert cfg["distillation"]["type"] == kd_type
    assert isinstance(teacher, RadarModalityNet)
    assert isinstance(student, RadarStudentModalityNet)
    assert cfg["model"]["teacher"]["type"] == "radar_teacher"
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["type"] == "radar_student"
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 2]
    assert teacher.GRU.hidden_size == student.GRU.hidden_size == 64


def test_radar_teacher_no_kd_config_does_not_load_teacher():
    _, cfg = _build_student("configs/radar/no_kd.yaml")

    assert cfg["distillation"]["type"] == "no_kd"
    assert cfg["distillation"]["teacher_model_name"] is None


@pytest.mark.parametrize("config_path", GPS_STUDENT_CONFIGS)
def test_gps_student_configs_build_lightweight_student(config_path: str):
    model, cfg = _build_student(config_path)

    assert cfg["experiment"]["task"] == "gps"
    assert cfg["data"]["dataset"]["use_gps"] is True
    assert cfg["model"]["teacher"]["type"] == "gps_teacher"
    assert cfg["model"]["student"]["type"] == "gps_student"
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 2]
    assert cfg["data"]["dataset"]["gps_feature_mode"] == "relative_polar"
    assert cfg["model"]["student"]["gps_input_size"] == 3
    assert isinstance(model, GpsStudentModalityNet)
    assert model.GRU.num_layers == 2


@pytest.mark.parametrize(("config_path", "kd_type"), GPS_KD_CONFIGS)
def test_gps_kd_configs_build_teacher_and_student(config_path: str, kd_type: str):
    teacher, student, cfg = _build_teacher_and_student(config_path)

    assert cfg["distillation"]["type"] == kd_type
    assert isinstance(teacher, GpsModalityNet)
    assert isinstance(student, GpsStudentModalityNet)
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 2]
    assert teacher.GRU.hidden_size == student.GRU.hidden_size == 64


@pytest.mark.parametrize("config_path", GPS_REL_POLAR_CONFIGS)
def test_gps_configs_use_relative_polar_features(config_path: str):
    model, cfg = _build_student(config_path)

    assert cfg["experiment"]["task"] == "gps"
    assert cfg["data"]["dataset"]["gps_feature_mode"] == "relative_polar"
    assert cfg["model"]["teacher"]["gps_input_size"] == 3
    assert cfg["model"]["student"]["gps_input_size"] == 3
    assert isinstance(model, (GpsModalityNet, GpsStudentModalityNet))


def test_unsupported_gps_ablation_configs_are_not_shipped():
    unsupported = [
        "configs/gps/ablation_raw.yaml",
        "configs/gps/ablation_utm.yaml",
        "configs/gps/ablation_relative.yaml",
        "configs/gps/ablation_motion.yaml",
        "configs/gps/ablation_motion_smooth.yaml",
    ]

    assert [path for path in unsupported if (ROOT / path).exists()] == []


@pytest.mark.parametrize("config_path", GPS_FUSION_CONFIGS)
def test_gps_fusion_configs_use_relative_polar_features(config_path: str):
    model, cfg = _build_student(config_path)

    assert cfg["experiment"]["task"] == "fusion"
    assert cfg["data"]["dataset"]["use_gps"] is True
    assert cfg["data"]["dataset"]["gps_feature_mode"] == "relative_polar"
    assert "gps" in cfg["model"]["student"]["modalities"]
    assert cfg["model"]["teacher"]["gps_input_size"] == 3
    assert cfg["model"]["student"]["gps_input_size"] == 3
    assert isinstance(model, StudentModalityNet)


def test_radar_student_no_kd_config_does_not_load_teacher():
    model, cfg = _build_student("configs/radar/student_no_kd.yaml")

    assert cfg["distillation"]["type"] == "no_kd"
    assert cfg["distillation"]["teacher_model_name"] is None
    assert cfg["model"]["student"]["type"] == "radar_student"
    assert isinstance(model, RadarStudentModalityNet)


@pytest.mark.parametrize(("config_path", "kd_type"), RADAR_KD_CONFIGS)
def test_radar_kd_configs_use_radar_teacher_checkpoint(config_path: str, kd_type: str):
    _, cfg = _build_student(config_path)

    assert cfg["distillation"]["type"] == kd_type
    assert cfg["distillation"]["teacher_model_name"] == "best.pth"
    assert cfg["paths"]["weights_dir"] == "outputs/radar_no_kd/checkpoints"
    assert cfg["distillation"]["temperature"] == 3.0
    assert cfg["distillation"]["alpha"] == 0.4
    assert cfg["distillation"]["alpha_warmup_epochs"] == 0


def test_radar_rkd_config_sets_relational_weights():
    _, cfg = _build_student("configs/radar/rkd.yaml")

    assert cfg["distillation"]["rkd_pairs_per_anchor"] == 4
    assert cfg["distillation"]["rkd_distance_weight"] == 10.0
    assert cfg["distillation"]["rkd_angle_weight"] == 10.0


def test_radar_teacher_forward_contract():
    model = MODELS.build(
        {
            "type": "radar_teacher",
            "feature_size": 64,
            "num_classes": 64,
            "gru_params": [64, 64, 2],
            "radar_channels": 2,
            "num_heads": 8,
        }
    )
    model.eval()

    with torch.no_grad():
        pred, features, enhanced = model(torch.randn(2, 10, 2, 128, 64))

    assert pred.shape == (2, 10, 64)
    assert features.shape == (2, 10, 64)
    assert enhanced.shape == (2, 10, 64)


def test_radar_student_forward_contract():
    model = MODELS.build(
        {
            "type": "radar_student",
            "feature_size": 64,
            "num_classes": 64,
            "gru_params": [64, 64, 2],
            "radar_channels": 2,
        }
    )
    model.eval()

    with torch.no_grad():
        pred, features, output_features = model(torch.randn(2, 10, 2, 128, 64))

    assert pred.shape == (2, 10, 64)
    assert features.shape == (2, 10, 64)
    assert output_features.shape == (2, 10, 64)


def test_radar_teacher_rejects_invalid_attention_heads():
    with pytest.raises(ValueError, match="divisible by num_heads"):
        MODELS.build(
            {
                "type": "radar_teacher",
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 66, 2],
                "radar_channels": 2,
                "num_heads": 8,
            }
        )


def test_radar_student_rejects_invalid_gru_params_length():
    with pytest.raises(ValueError, match="gru_params must contain"):
        MODELS.build(
            {
                "type": "radar_student",
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64],
                "radar_channels": 2,
            }
        )


def test_radar_student_rejects_input_size_mismatch():
    with pytest.raises(ValueError, match="must equal feature_size"):
        MODELS.build(
            {
                "type": "radar_student",
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [32, 64, 1],
                "radar_channels": 2,
            }
        )


@pytest.mark.parametrize(("config_path", "weight_path"), LEGACY_STUDENT_WEIGHTS)
def test_packaged_student_weights_are_legacy_one_layer(config_path: str, weight_path: str):
    model, _ = _build_student(config_path)
    state = _load_state_dict(weight_path)
    model_state = model.state_dict()

    missing = sorted(set(model_state) - set(state))
    shape_mismatches = sorted(
        key
        for key, tensor in model_state.items()
        if key in state and tuple(state[key].shape) != tuple(tensor.shape)
    )
    unexpected_non_stats = sorted(key for key in state if key not in model_state and not _is_stats_key(key))

    assert {key for key in missing if key.startswith("GRU.")} == {
        "GRU.bias_hh_l1",
        "GRU.bias_ih_l1",
        "GRU.weight_hh_l1",
        "GRU.weight_ih_l1",
    }
    assert [key for key in missing if not key.startswith("GRU.")] == []
    assert shape_mismatches == []
    assert unexpected_non_stats == []
