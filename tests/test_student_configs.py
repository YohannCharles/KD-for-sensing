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
from kd_sensing.models.image import ImageStudentModalityNet  # noqa: E402
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

STUDENT_WEIGHTS = [
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
    assert model.GRU.num_layers == 1


@pytest.mark.parametrize("config_path", FUSION_CONFIGS)
def test_fusion_configs_build_lightweight_student(config_path: str):
    model, cfg = _build_student(config_path)

    assert cfg["model"]["teacher"]["type"] == "fusion_teacher"
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["type"] == "fusion_student"
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 1]
    assert isinstance(model, StudentModalityNet)
    assert model.GRU.num_layers == 1


@pytest.mark.parametrize(("config_path", "weight_path"), STUDENT_WEIGHTS)
def test_student_configs_match_packaged_student_weights(config_path: str, weight_path: str):
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

    assert missing == []
    assert shape_mismatches == []
    assert unexpected_non_stats == []
