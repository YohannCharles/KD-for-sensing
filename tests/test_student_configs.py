from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.config.canonical import (  # noqa: E402
    CANONICAL_FUSION_MODALITIES,
    build_advanced_fusion_overlay_config,
    build_virtual_fusion_config,
)
from kd_sensing.config.canonical_recipes import (  # noqa: E402
    advanced_overlay_recipe,
    distillation_overrides,
    objective_overlay_recipe,
    training_overrides,
)
from kd_sensing.config.io import dump_config, safe_load_yaml  # noqa: E402
from kd_sensing.engine.evaluator import evaluate  # noqa: E402
from kd_sensing.engine.trainer import train  # noqa: E402
from kd_sensing.models.fusion import (  # noqa: E402
    CLSTokenTransformerFusionNet,
    FusionTeacherModalityNet,
    FusionStudentModalityNet,
)
from kd_sensing.models.gps import GpsModalityNet, GpsStudentModalityNet  # noqa: E402
from kd_sensing.models.image import ImageModalityNet, ImageStudentModalityNet  # noqa: E402
from kd_sensing.models.modular import ModularSequenceModel  # noqa: E402
from kd_sensing.models.lidar import LidarModalityNet, LidarStudentModalityNet  # noqa: E402
from kd_sensing.models.radar import RadarModalityNet, RadarStudentModalityNet  # noqa: E402
from kd_sensing.registries import MODELS, RegistryError  # noqa: E402
from kd_sensing.utils.checkpoint import CheckpointLoadError, load_model_state  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.layer1 = nn.Conv2d(2, 2, kernel_size=1)
        self.layer2 = nn.Conv2d(2, 2, kernel_size=1)
        self.layer3 = nn.Conv2d(2, 2, kernel_size=1)
        self.layer4 = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).repeat(1, 512)


@pytest.fixture(autouse=True)
def tiny_resnet(monkeypatch):
    import kd_sensing.models.image_encoders as image_encoders

    monkeypatch.setattr(
        image_encoders,
        "_build_resnet18_backbone",
        lambda *, pretrained, weights: (_TinyBackbone(), 512),
    )


SINGLE_GRU_PARAMS = [64, 64, 1]
FUSION_TEACHER_GRU_PARAMS = [64, 64, 2]
FUSION_STUDENT_GRU_PARAMS = [64, 64, 1]
SINGLE_CONFIG_MODES = ("teacher_no_kd", "student_no_kd", "logits_kd", "rkd")
FUSION_CONFIG_MODES = ("teacher_no_kd", "student_no_kd", "logits_kd", "rkd")

SINGLE_EXPECTED_PARAMS = {
    "teacher_no_kd": {
        "lr": 0.0001,
        "weight_decay": 0.0001,
        "temperature": 3.0,
        "alpha": 0.5,
        "rkd_distance_weight": 10.0,
        "rkd_angle_weight": 10.0,
    },
    "student_no_kd": {
        "lr": 0.001,
        "weight_decay": 0.0,
        "temperature": 3.0,
        "alpha": 0.4,
        "rkd_distance_weight": 50.0,
        "rkd_angle_weight": 50.0,
    },
    "logits_kd": {
        "lr": 0.0008,
        "weight_decay": 0.0,
        "temperature": 4.0,
        "alpha": 0.3,
        "rkd_distance_weight": 100.0,
        "rkd_angle_weight": 100.0,
    },
    "rkd": {
        "lr": 0.0005,
        "weight_decay": 0.0,
        "temperature": 3.0,
        "alpha": 0.1,
        "rkd_distance_weight": 100.0,
        "rkd_angle_weight": 100.0,
    },
}

FUSION_IMAGE_RADAR_EXPECTED_PARAMS = {
    "teacher_no_kd": {"lr": 0.00075, "weight_decay": 0.0001, "temperature": 3.0, "alpha": 0.4},
    "student_no_kd": {"lr": 0.0004, "weight_decay": 0.0, "temperature": 3.0, "alpha": 0.4},
    "logits_kd": {"lr": 0.00095, "weight_decay": 0.0, "temperature": 2.0, "alpha": 0.4},
    "rkd": {"lr": 0.00095, "weight_decay": 0.0, "temperature": 2.0, "alpha": 0.3},
}


def test_fusion_registry_returns_public_class_names_and_removed_aliases_fail():
    teacher = MODELS.build(
        {
            "type": "fusion_teacher",
            "feature_size": 64,
            "num_classes": 64,
            "gru_params": [64, 64, 2],
        }
    )
    student = MODELS.build(
        {
            "type": "fusion_student",
            "feature_size": 64,
            "num_classes": 64,
            "gru_params": [64, 64, 1],
        }
    )

    assert type(teacher) is FusionTeacherModalityNet
    assert type(student) is FusionStudentModalityNet
    for alias in ["Fusion" + "ModalityNet", "Student" + "ModalityNet"]:
        with pytest.raises(AttributeError, match=alias):
            getattr(kd_sensing.models, alias)
        with pytest.raises(RegistryError, match="Removed component"):
            MODELS.build({"type": alias})


MODALITY_SPECS = {
    "image": {
        "task": "image",
        "teacher_type": "modular_sequence",
        "student_type": "modular_sequence",
        "teacher_cls": ModularSequenceModel,
        "student_cls": ModularSequenceModel,
        "default_teacher_type": "modular_sequence",
        "default_teacher_cls": ModularSequenceModel,
    },
    "radar": {
        "task": "radar",
        "teacher_type": "radar_teacher",
        "student_type": "radar_student",
        "teacher_cls": RadarModalityNet,
        "student_cls": RadarStudentModalityNet,
        "default_teacher_type": "radar_teacher",
        "default_teacher_cls": RadarModalityNet,
    },
    "gps": {
        "task": "gps",
        "teacher_type": "gps_teacher",
        "student_type": "gps_student",
        "teacher_cls": GpsModalityNet,
        "student_cls": GpsStudentModalityNet,
        "default_teacher_type": "gps_teacher",
        "default_teacher_cls": GpsModalityNet,
    },
    "lidar": {
        "task": "lidar",
        "teacher_type": "modular_sequence",
        "student_type": "modular_sequence",
        "teacher_cls": ModularSequenceModel,
        "student_cls": ModularSequenceModel,
        "default_teacher_type": "modular_sequence",
        "default_teacher_cls": ModularSequenceModel,
    },
}

FUSION_SLUGS = {
    "_".join(modalities): list(modalities)
    for size in (2, 3, 4, 5)
    for modalities in combinations(CANONICAL_FUSION_MODALITIES, size)
}

LEGACY_CONFIG_EXPECTATIONS = [
    (
        "configs/image/no_kd.yaml",
        "image",
        "no_kd",
        "modular_sequence",
        None,
        "configs/image/teacher_no_kd.yaml",
    ),
    (
        "configs/radar/no_kd.yaml",
        "radar",
        "no_kd",
        "radar_teacher",
        None,
        "configs/radar/teacher_no_kd.yaml",
    ),
    (
        "configs/gps/no_kd.yaml",
        "gps",
        "no_kd",
        "gps_teacher",
        None,
        "configs/gps/teacher_no_kd.yaml",
    ),
    (
        "configs/lidar/no_kd.yaml",
        "lidar",
        "no_kd",
        "modular_sequence",
        None,
        "configs/lidar/teacher_no_kd.yaml",
    ),
    (
        "configs/fusion/image_gps_no_kd.yaml",
        "fusion",
        "no_kd",
        "modular_sequence",
        ["image", "gps"],
        "configs/fusion/image_gps_teacher_no_kd.yaml",
    ),
    (
        "configs/fusion/radar_gps_no_kd.yaml",
        "fusion",
        "no_kd",
        "fusion_student",
        ["radar", "gps"],
        "configs/fusion/radar_gps_student_no_kd.yaml",
    ),
    (
        "configs/fusion/radar_lidar_no_kd.yaml",
        "fusion",
        "no_kd",
        "modular_sequence",
        ["radar", "lidar"],
        "configs/fusion/radar_lidar_student_no_kd.yaml",
    ),
    (
        "configs/fusion/all_modalities_no_kd.yaml",
        "fusion",
        "no_kd",
        "cls_token_transformer_fusion",
        ["image", "radar", "gps", "lidar", "mmwave"],
        "configs/fusion/image_radar_gps_lidar_mmwave_student_no_kd.yaml",
    ),
    (
        "configs/fusion/all_modalities_lidar_no_kd.yaml",
        "fusion",
        "no_kd",
        "modular_sequence",
        ["image", "radar", "gps", "lidar"],
        "configs/fusion/image_radar_gps_lidar_teacher_no_kd.yaml",
    ),
]

def _load(config_path: str) -> dict:
    return load_config(ROOT / config_path)


def _build_student(config_path: str):
    cfg = _load(config_path)
    return MODELS.build(cfg["model"]["student"]), cfg


def _build_teacher_and_student(config_path: str):
    cfg = _load(config_path)
    return MODELS.build(cfg["model"]["teacher"]), MODELS.build(cfg["model"]["student"]), cfg


def _assert_default_early_stopping(cfg: dict) -> None:
    assert cfg["training"]["use_early_stopping"] is True
    assert cfg["training"]["early_stopping_metric"] == "val_adba"
    assert cfg["training"]["early_stopping_mode"] == "max"


def _assert_modular_single_encoder(role_cfg: dict, modality: str) -> None:
    assert role_cfg["type"] == "modular_sequence"
    assert role_cfg["modalities"] == [modality]
    if modality == "image":
        encoder = role_cfg["encoders"]["image"]
        assert encoder["type"] == "resnet18_imagenet_rgb"
        assert encoder["pretrained"] is True
        assert encoder["weights"] == "DEFAULT"
    elif modality == "lidar":
        encoder = role_cfg["encoders"]["lidar"]
        assert encoder["type"] == "lidar_cnn"
        assert encoder["lidar_channels"] == 3


def _gru_module(model):
    return model.representation_core.gru if isinstance(model, ModularSequenceModel) else model.GRU


def _single_config_cases():
    return [
        (modality, mode, f"configs/{modality}/{mode}.yaml")
        for modality in MODALITY_SPECS
        for mode in SINGLE_CONFIG_MODES
    ]


def _fusion_config_cases():
    return [
        (slug, modalities, mode, f"configs/fusion/{slug}_{mode}.yaml")
        for slug, modalities in FUSION_SLUGS.items()
        for mode in FUSION_CONFIG_MODES
    ]


@pytest.mark.parametrize(("modality", "mode", "config_path"), _single_config_cases())
def test_canonical_single_modality_config_matrix(modality: str, mode: str, config_path: str):
    spec = MODALITY_SPECS[modality]
    model, cfg = _build_student(config_path)
    expected_name = f"{modality}_{mode}"
    expected_params = SINGLE_EXPECTED_PARAMS[mode]
    if modality == "image" and mode in {"teacher_no_kd", "student_no_kd"}:
        expected_params = {**expected_params, "lr": 0.0004, "weight_decay": 0.0001}
    expected_batch_size = 16 if modality == "image" and mode in {"teacher_no_kd", "student_no_kd"} else 32

    assert cfg["experiment"]["name"] == expected_name
    assert cfg["experiment"]["task"] == spec["task"]
    assert cfg["experiment"]["seed"] == 42
    assert cfg["output"]["run_name"] == expected_name
    assert cfg["data"]["dataloader"]["train_batch_size"] == expected_batch_size
    assert cfg["data"]["dataloader"]["test_batch_size"] == expected_batch_size
    assert cfg["data"]["dataloader"]["num_workers"] == 4
    assert cfg["data"]["dataloader"]["prefetch_factor"] == 1
    expected_teacher_type = spec["teacher_type"]
    assert cfg["model"]["teacher"]["type"] == expected_teacher_type
    if expected_teacher_type == "modular_sequence":
        _assert_modular_single_encoder(cfg["model"]["teacher"], modality)
    else:
        assert cfg["model"]["teacher"]["gru_params"] == SINGLE_GRU_PARAMS
    if cfg["model"]["student"]["type"] in {"modular_sequence", "modular_sequence_model"}:
        _assert_modular_single_encoder(cfg["model"]["student"], modality)
    else:
        assert cfg["model"]["student"]["gru_params"] == SINGLE_GRU_PARAMS
    assert cfg["training"]["epochs"] == 100
    assert cfg["training"]["lr"] == expected_params["lr"]
    assert cfg["training"]["weight_decay"] == expected_params["weight_decay"]
    assert cfg["training"]["grad_clip"] == 10.0
    assert cfg["training"]["patience"] == 20
    _assert_default_early_stopping(cfg)
    assert cfg["training"]["min_delta"] == 0.0001
    assert cfg["scheduler"]["T_0"] == 10
    assert cfg["scheduler"]["T_mult"] == 2
    assert cfg["scheduler"]["eta_min"] == 1e-6
    assert cfg["distillation"]["temperature"] == expected_params["temperature"]
    assert cfg["distillation"]["alpha"] == expected_params["alpha"]
    assert cfg["distillation"]["alpha_warmup_epochs"] == 0
    assert cfg["distillation"]["rkd_pairs_per_anchor"] == 4
    assert cfg["distillation"]["rkd_distance_weight"] == expected_params["rkd_distance_weight"]
    assert cfg["distillation"]["rkd_angle_weight"] == expected_params["rkd_angle_weight"]

    expected_student_type = spec["teacher_type"] if mode == "teacher_no_kd" else spec["student_type"]
    expected_student_cls = spec["teacher_cls"] if mode == "teacher_no_kd" else spec["student_cls"]
    assert cfg["model"]["student"]["type"] == expected_student_type
    assert isinstance(model, expected_student_cls)
    if hasattr(model, "GRU"):
        assert model.GRU.num_layers == 1
    else:
        assert model.representation_core.gru.num_layers == 1

    if mode in {"teacher_no_kd", "student_no_kd"}:
        assert cfg["distillation"]["type"] == "no_kd"
        assert cfg["distillation"]["teacher_model_name"] is None
    else:
        teacher, student, kd_cfg = _build_teacher_and_student(config_path)
        assert kd_cfg["distillation"]["type"] == mode
        assert kd_cfg["paths"]["weights_dir"] == f"outputs/scene31/{modality}_teacher_no_kd/checkpoints"
        assert kd_cfg["distillation"]["teacher_model_name"] == "best.pth"
        assert isinstance(teacher, spec["teacher_cls"])
        assert isinstance(student, spec["student_cls"])
        assert _gru_module(teacher).hidden_size == _gru_module(student).hidden_size == 64
        if mode == "rkd":
            assert kd_cfg["distillation"]["rkd_pairs_per_anchor"] == 4
            assert kd_cfg["distillation"]["rkd_distance_weight"] == 100.0
            assert kd_cfg["distillation"]["rkd_angle_weight"] == 100.0

    if mode == "student_no_kd" and spec["student_type"] != spec["teacher_type"]:
        assert cfg["model"]["student"]["type"] != spec["teacher_type"]

    _assert_modality_data_fields(cfg, [modality])


@pytest.mark.parametrize(("slug", "modalities", "mode", "config_path"), _fusion_config_cases())
def test_canonical_fusion_config_matrix(slug: str, modalities: list[str], mode: str, config_path: str):
    student, cfg = _build_student(config_path)
    stem = Path(config_path).stem

    assert cfg["experiment"]["name"] == stem
    assert cfg["experiment"]["task"] == "fusion"
    assert cfg["output"]["run_name"] == stem
    _assert_default_early_stopping(cfg)
    image_fusion = "image" in modalities
    modular_fusion = image_fusion or "lidar" in modalities
    expected_teacher_type = "modular_sequence" if modular_fusion else "fusion_teacher"
    assert cfg["model"]["teacher"]["type"] == expected_teacher_type
    assert cfg["model"]["teacher"]["modalities"] == modalities
    assert cfg["model"]["student"]["modalities"] == modalities
    if modular_fusion:
        if image_fusion:
            assert cfg["model"]["teacher"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
            assert cfg["model"]["teacher"]["encoders"]["image"]["pretrained"] is True
            assert cfg["model"]["teacher"]["encoders"]["image"]["weights"] == "DEFAULT"
        if "lidar" in modalities:
            assert cfg["model"]["teacher"]["encoders"]["lidar"]["type"] == "lidar_cnn"
        assert cfg["model"]["teacher"]["representation_core"]["num_layers"] == 2
    else:
        assert cfg["model"]["teacher"]["gru_params"] == FUSION_TEACHER_GRU_PARAMS

    if mode == "teacher_no_kd":
        expected_student_type = expected_teacher_type
        expected_student_cls = ModularSequenceModel if modular_fusion else FusionTeacherModalityNet
        if modular_fusion:
            assert cfg["model"]["student"]["type"] == "modular_sequence"
            if image_fusion:
                assert cfg["model"]["student"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
            if "lidar" in modalities:
                assert cfg["model"]["student"]["encoders"]["lidar"]["type"] == "lidar_cnn"
            assert cfg["model"]["student"]["representation_core"]["num_layers"] == 2
        else:
            assert cfg["model"]["student"]["gru_params"] == FUSION_TEACHER_GRU_PARAMS
    else:
        expected_student_type = "cls_token_transformer_fusion"
        expected_student_cls = CLSTokenTransformerFusionNet
        assert cfg["model"]["student"]["type"] == "cls_token_transformer_fusion"
        assert cfg["model"]["student"]["d_model"] == 64
        assert cfg["model"]["student"]["num_heads"] == 4
        assert cfg["model"]["student"]["num_layers"] == 2
        assert cfg["model"]["student"]["num_pred"] == 3

    assert cfg["model"]["student"]["type"] == expected_student_type
    assert isinstance(student, expected_student_cls)
    assert student.modalities == tuple(modalities)
    if isinstance(student, ModularSequenceModel):
        assert student.representation_core.gru.num_layers == 2
    elif isinstance(student, FusionTeacherModalityNet):
        assert student.GRU.num_layers == 2
    else:
        assert student.horizon == student.num_pred == 3

    if slug == "image_radar":
        expected_params = FUSION_IMAGE_RADAR_EXPECTED_PARAMS[mode]
        assert cfg["experiment"]["seed"] == 42
        assert cfg["data"]["dataloader"]["train_batch_size"] == 32
        assert cfg["data"]["dataloader"]["test_batch_size"] == 32
        assert cfg["training"]["lr"] == expected_params["lr"]
        assert cfg["training"]["weight_decay"] == expected_params["weight_decay"]
        assert cfg["distillation"]["temperature"] == expected_params["temperature"]
        assert cfg["distillation"]["alpha"] == expected_params["alpha"]

    if mode in {"teacher_no_kd", "student_no_kd"}:
        assert cfg["distillation"]["type"] == "no_kd"
        assert cfg["distillation"]["teacher_model_name"] is None
    else:
        teacher, kd_student, kd_cfg = _build_teacher_and_student(config_path)
        assert kd_cfg["distillation"]["type"] == mode
        assert kd_cfg["paths"]["weights_dir"] == f"outputs/scene31/{slug}_teacher_no_kd/checkpoints"
        assert kd_cfg["distillation"]["teacher_model_name"] == "best.pth"
        assert isinstance(teacher, (ModularSequenceModel if modular_fusion else FusionTeacherModalityNet))
        assert isinstance(kd_student, CLSTokenTransformerFusionNet)
        assert teacher.modalities == kd_student.modalities == tuple(modalities)
        if isinstance(teacher, ModularSequenceModel):
            assert teacher.representation_core.gru.num_layers == 2
        else:
            assert teacher.GRU.num_layers == 2
        assert kd_student.horizon == kd_student.num_pred == 3
        if mode == "rkd":
            assert kd_cfg["distillation"]["rkd_pairs_per_anchor"] == 4
            assert kd_cfg["distillation"]["rkd_distance_weight"] == 10.0
            assert kd_cfg["distillation"]["rkd_angle_weight"] == 10.0

    _assert_modality_data_fields(cfg, modalities)


@pytest.mark.parametrize(
    (
        "config_path",
        "task",
        "distillation_type",
        "student_type",
        "modalities",
        "canonical_path",
    ),
    LEGACY_CONFIG_EXPECTATIONS,
)
def test_named_example_configs_keep_current_semantics(
    config_path: str,
    task: str,
    distillation_type: str,
    student_type: str,
    modalities: list[str] | None,
    canonical_path: str,
):
    model, cfg = _build_student(config_path)

    load_config(ROOT / canonical_path)
    assert cfg["experiment"]["task"] == task
    assert cfg["distillation"]["type"] == distillation_type
    assert cfg["model"]["student"]["type"] == student_type
    _assert_default_early_stopping(cfg)
    if cfg["model"]["student"]["type"] == "cls_token_transformer_fusion":
        assert cfg["model"]["student"]["d_model"] == 64
        assert cfg["model"]["student"]["num_heads"] == 4
        assert cfg["model"]["student"]["num_layers"] == 2
        assert cfg["model"]["student"]["num_pred"] == 3
    elif task == "fusion" and modalities == ["image", "radar"] and cfg["model"]["student"]["type"] != "modular_sequence":
        assert cfg["model"]["teacher"]["gru_params"] == FUSION_TEACHER_GRU_PARAMS
        assert cfg["model"]["student"]["gru_params"] == FUSION_STUDENT_GRU_PARAMS
    elif task == "fusion" and cfg["model"]["student"]["type"] != "modular_sequence":
        assert cfg["model"]["student"]["gru_params"] == FUSION_TEACHER_GRU_PARAMS
    elif task != "fusion" and cfg["model"]["student"]["type"] != "modular_sequence":
        assert cfg["model"]["student"]["gru_params"] == SINGLE_GRU_PARAMS
    elif cfg["model"]["student"]["type"] == "modular_sequence":
        if task == "image" or (modalities is not None and "image" in modalities):
            assert cfg["model"]["student"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
            assert cfg["model"]["student"]["encoders"]["image"]["pretrained"] is True
        if task == "lidar" or (modalities is not None and "lidar" in modalities):
            assert cfg["model"]["student"]["encoders"]["lidar"]["type"] == "lidar_cnn"

    if distillation_type == "no_kd":
        assert cfg["distillation"]["teacher_model_name"] is None
    elif config_path.startswith("configs/fusion/"):
        assert modalities is not None
        slug = "_".join(modalities)
        assert cfg["paths"]["weights_dir"] == f"outputs/scene31/{slug}_teacher_no_kd/checkpoints"
        assert cfg["distillation"]["teacher_model_name"] == "best.pth"

    if modalities is not None:
        assert cfg["model"]["teacher"]["modalities"] == modalities
        assert cfg["model"]["student"]["modalities"] == modalities
        assert isinstance(
            model,
            (
                CLSTokenTransformerFusionNet,
                FusionTeacherModalityNet,
                FusionStudentModalityNet,
                ModularSequenceModel,
            ),
        )
        assert model.modalities == tuple(modalities)

    _assert_modality_data_fields(cfg, modalities or [task])


@pytest.mark.parametrize(
    ("stem", "replacement"),
    [
        ("no_kd", "image_radar_student_no_kd.yaml"),
        ("logits_kd", "image_radar_logits_kd.yaml"),
        ("rkd", "image_radar_rkd.yaml"),
    ],
)
def test_removed_fusion_alias_config_paths_raise_migration_error(stem: str, replacement: str):
    with pytest.raises(ValueError, match=replacement):
        load_config(ROOT / f"configs/fusion/{stem}.yaml")


def _assert_modality_data_fields(cfg: dict, modalities: list[str]) -> None:
    dataset_cfg = cfg["data"]["dataset"]
    teacher_cfg = cfg["model"]["teacher"]
    student_cfg = cfg["model"]["student"]

    assert dataset_cfg["type"] == "deepsense6g"
    assert dataset_cfg["scene_id"] == 31
    assert dataset_cfg["scene_slug"] == "scene31"
    assert dataset_cfg["data_root"] == "dataset/DeepSense6G/scenario31"

    if "gps" in modalities:
        assert dataset_cfg["use_gps"] is True
        assert dataset_cfg["gps_feature_mode"] == "relative_polar"
        assert teacher_cfg["gps_input_size"] == 3
        assert student_cfg["gps_input_size"] == 3
    else:
        assert dataset_cfg.get("use_gps", False) is False

    if "lidar" in modalities:
        assert dataset_cfg["use_lidar"] is True
        assert dataset_cfg["lidar_bev_size"] == [224, 224]
        assert dataset_cfg["lidar_roi"] == [-30.0, 30.0, -30.0, 30.0, -3.0, 5.0]
        assert dataset_cfg["lidar_normalize"] is False
        assert dataset_cfg["lidar_normalization"]["enabled"] is False
        assert dataset_cfg["lidar_normalization"]["mode"] == "none"
        assert dataset_cfg["lidar_cache_dir"] == "lidar_bev_cache"
        assert teacher_cfg["lidar_channels"] == 3
        assert student_cfg["lidar_channels"] == 3
    else:
        assert dataset_cfg.get("use_lidar", False) is False

    if "mmwave" in modalities:
        assert dataset_cfg["use_mmwave"] is True
        assert dataset_cfg["mmwave_normalize"] is True
        assert teacher_cfg["mmwave_input_size"] == 64
        assert student_cfg["mmwave_input_size"] == 64
    else:
        assert dataset_cfg.get("use_mmwave", False) is False


def test_virtual_fusion_config_generator_uses_canonical_semantics():
    cfg = build_virtual_fusion_config("gps_mmwave_logits_kd")

    assert cfg["experiment"]["name"] == "gps_mmwave_logits_kd"
    assert cfg["experiment"]["task"] == "fusion"
    assert cfg["experiment"]["seed"] == 0
    assert cfg["model"]["teacher"]["modalities"] == ["gps", "mmwave"]
    assert cfg["model"]["student"]["modalities"] == ["gps", "mmwave"]
    assert cfg["model"]["teacher"]["type"] == "fusion_teacher"
    assert cfg["model"]["student"]["type"] == "cls_token_transformer_fusion"
    assert cfg["model"]["student"]["d_model"] == 64
    assert cfg["model"]["student"]["num_heads"] == 4
    assert cfg["model"]["teacher"]["gps_input_size"] == 3
    assert cfg["model"]["teacher"]["mmwave_input_size"] == 64
    assert cfg["distillation"]["type"] == "logits_kd"
    assert cfg["distillation"]["teacher_model_name"] == "best.pth"
    assert cfg["paths"]["weights_dir"] == "outputs/scene31/gps_mmwave_teacher_no_kd/checkpoints"
    assert cfg["training"]["early_stopping_metric"] == "val_adba"
    assert cfg["training"]["early_stopping_mode"] == "max"


def test_base_fusion_recipe_registry_keeps_virtual_config_core_fields():
    cfg = build_virtual_fusion_config("gps_mmwave_logits_kd")

    assert cfg["distillation"] == distillation_overrides("gps_mmwave", "logits_kd", False)
    assert cfg["training"] == training_overrides("logits_kd", False)
    assert distillation_overrides("image_radar", "logits_kd", True)["temperature"] == 2.0
    assert training_overrides("student_no_kd", True)["lr"] == 0.0004


def test_objective_and_advanced_overlay_recipes_are_table_driven():
    objective = objective_overlay_recipe("multitask")
    advanced = advanced_overlay_recipe("marf_subset_training")

    assert objective.dataset["occlusion_target"]["enabled"] is True
    assert objective.dataset["position_target"]["normalize"] is True
    assert objective.auxiliary_heads == {"occlusion": True, "position": True}
    assert objective.early_stopping_metric == "val_multitask_loss"
    assert advanced is not None
    assert advanced.builder == "marf"
    assert advanced.options["ablation"]["training"]["subset_training"]["enabled"] is True


def test_virtual_image_radar_config_generator_keeps_compatibility_params():
    cfg = build_virtual_fusion_config("image_radar_logits_kd")

    assert cfg["experiment"]["seed"] == 42
    assert cfg["model"]["teacher"]["type"] == "modular_sequence"
    assert cfg["model"]["teacher"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
    assert cfg["model"]["teacher"]["representation_core"]["num_layers"] == 2
    assert cfg["model"]["student"]["type"] == "cls_token_transformer_fusion"
    assert cfg["model"]["student"]["image_channels"] == 3
    assert cfg["model"]["student"]["radar_channels"] == 2
    assert cfg["model"]["student"]["num_layers"] == 2
    assert cfg["training"]["lr"] == 0.00095
    assert cfg["training"]["weight_decay"] == 0.0
    assert cfg["distillation"]["temperature"] == 2.0
    assert cfg["distillation"]["alpha"] == 0.4
    assert cfg["distillation"]["teacher_model_name"] == "best.pth"
    assert cfg["paths"]["weights_dir"] == "outputs/scene31/image_radar_teacher_no_kd/checkpoints"
    assert cfg["training"]["early_stopping_metric"] == "val_adba"
    assert cfg["training"]["early_stopping_mode"] == "max"


def test_load_config_applies_overrides_after_canonical_config_resolution():
    cfg = _load("configs/fusion/gps_mmwave_logits_kd.yaml")
    overridden = load_config(
        ROOT / "configs/fusion/gps_mmwave_logits_kd.yaml",
        ["training.epochs=1", "training.early_stopping_metric=top1_val_acc", "training.early_stopping_mode=max"],
    )

    assert cfg["experiment"]["name"] == "gps_mmwave_logits_kd"
    _assert_default_early_stopping(cfg)
    assert overridden["training"]["epochs"] == 1
    assert overridden["training"]["early_stopping_metric"] == "top1_val_acc"
    assert overridden["training"]["early_stopping_mode"] == "max"
    assert overridden["model"]["teacher"]["modalities"] == ["gps", "mmwave"]


@pytest.mark.parametrize("modality", ["image", "radar", "gps", "lidar", "mmwave"])
@pytest.mark.parametrize("objective", ["occlusion", "position", "multitask"])
def test_single_modality_objective_overrides_build_auxiliary_capable_models(modality: str, objective: str):
    cfg = load_config(ROOT / f"configs/{modality}/teacher_no_kd.yaml", [f"experiment.objective={objective}"])
    model = MODELS.build(cfg["model"]["student"])

    assert cfg["model"]["student"]["auxiliary_heads"]["enabled"] is True
    assert hasattr(model, "auxiliary_heads")


def test_load_config_keeps_explicit_scene32_override():
    cfg = load_config(ROOT / "configs/fusion/gps_mmwave_logits_kd.yaml", ["data.dataset.scene=32"])

    assert cfg["data"]["dataset"]["scene_id"] == 32
    assert cfg["data"]["dataset"]["scene_slug"] == "scene32"
    assert cfg["data"]["dataset"]["data_root"] == "dataset/DeepSense6G/scenario32"


def test_existing_yaml_config_takes_precedence_over_virtual_config(tmp_path: Path):
    config_path = tmp_path / "configs" / "fusion" / "gps_mmwave_logits_kd.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("experiment:\n  name: entity_config\n", encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg["experiment"]["name"] == "entity_config"


def test_existing_advanced_overlay_yaml_takes_precedence_over_virtual_config(tmp_path: Path):
    config_path = tmp_path / "configs" / "fusion" / "overlay_g2d_lite.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("experiment:\n  name: physical_overlay\n", encoding="utf-8")

    cfg = load_config(config_path)

    assert cfg["experiment"]["name"] == "physical_overlay"


def test_advanced_fusion_overlay_generates_g2d_modes_and_full_dump(tmp_path: Path):
    cfg = load_config(ROOT / "configs/fusion/overlay_g2d_global.yaml")
    entity = load_config(ROOT / "configs/fusion/image_radar_gps_lidar_mmwave_g2d_global.yaml")
    output = tmp_path / "final_config.yaml"
    dump_config(cfg, output)
    dumped = safe_load_yaml(output.read_text(encoding="utf-8"))

    assert cfg["experiment"]["task"] == "fusion"
    assert cfg["model"]["student"]["modalities"] == ["image", "radar", "gps", "lidar", "mmwave"]
    assert cfg["distillation"]["type"] == "g2d"
    assert cfg["distillation"]["g2d"]["mode"] == "global"
    assert cfg["distillation"]["g2d"]["smp"]["enabled"] is True
    assert cfg["training"]["lr"] == entity["training"]["lr"]
    assert cfg["model"]["student"]["type"] == entity["model"]["student"]["type"]
    assert set(dumped) >= {"data", "model", "loss", "distillation", "training", "output"}


def test_csi_hardening_matrix_configs_load_and_preserve_contracts():
    first_batch = {
        "A0_clean_full_teacher.yaml",
        "A1_mild_pilot_estimation.yaml",
        "A2_destructive_degradation.yaml",
        "B3_antenna_calibration.yaml",
        "B4_fixed_antenna_permutation.yaml",
        "B5_mild_hardening_combo.yaml",
        "B6_medium_hardening_combo.yaml",
        "C1_view_gate_warmup.yaml",
        "C2_no_internal_gru.yaml",
    }
    second_batch = {
        "D1_mild_hardening_gate_warmup.yaml",
        "D2_mild_hardening_no_internal_gru.yaml",
        "D3_mild_hardening_gate_warmup_no_internal_gru.yaml",
        "D4_medium_hardening_gate_warmup_no_internal_gru.yaml",
    }
    root = ROOT / "configs/csi/hardening_matrix"

    for filename in sorted(first_batch | second_batch):
        cfg = load_config(root / filename)
        student = cfg["model"]["student"]
        csi_encoder = student["encoders"]["csi"]
        assert cfg["experiment"]["task"] == "csi"
        assert cfg["training"]["epochs"] == 100
        assert student["type"] == "modular_sequence"
        assert student["modalities"] == ["csi"]
        assert csi_encoder["type"] == "pilot_dual_view_csi"
        assert student["heads"]["beam"]["type"] == "beam_head"
        if filename.startswith("D"):
            assert "csi_degradation" not in cfg["data"]["dataset"]

    a2 = load_config(root / "A2_destructive_degradation.yaml")
    assert a2["data"]["dataset"]["csi_degradation"]["enabled"] is True
    assert "csi_hardening" not in a2["model"]["student"]["encoders"]["csi"]
    assert a2["debug"]["analysis_role"] == "destructive_negative_control"
    assert a2["debug"]["matrix_role"] == "A2_destructive_degradation"

    a1 = load_config(root / "A1_mild_pilot_estimation.yaml")
    a1_estimation = a1["model"]["student"]["encoders"]["csi"]["csi_estimation"]
    assert a1_estimation["mode"] in {"est_snr", "estimation_snr"}
    assert a1_estimation["snr_db"] == 30.0
    assert a1_estimation["train_snr_min_db"] == 25.0
    assert a1_estimation["train_snr_max_db"] == 35.0
    assert "noise_var" not in a1_estimation

    for filename in sorted(first_batch | second_batch):
        if filename.startswith(("A0", "A1", "A2")):
            continue
        cfg = load_config(root / filename)
        csi_estimation = cfg["model"]["student"]["encoders"]["csi"]["csi_estimation"]
        assert csi_estimation["mode"] == "none"
        assert "noise_var" not in csi_estimation
        assert cfg["debug"]["pilot_scaling_config_version"] == "fixed_estimation_snr_v1"

    b5 = load_config(root / "B5_mild_hardening_combo.yaml")
    assert b5["model"]["student"]["encoders"]["csi"]["csi_hardening"]["enabled"] is True
    explicit = load_config(
        root / "B5_mild_hardening_combo.yaml",
        ["model.student.encoders.csi.csi_hardening.enabled=false"],
    )
    assert explicit["model"]["teacher"]["encoders"]["csi"]["csi_hardening"]["enabled"] is True
    assert explicit["model"]["student"]["encoders"]["csi"]["csi_hardening"]["enabled"] is False


def test_csi_hardening_debug_matrix_configs_load_and_isolate_single_changes():
    root = ROOT / "configs/csi/hardening_matrix/debug"
    configs = {
        path.name: load_config(path)
        for path in sorted(root.glob("*.yaml"))
    }

    assert set(configs) == {
        "A0_original.yaml",
        "A0_clone_generated.yaml",
        "A0_clone_pilot_disabled.yaml",
        "C1_view_gate_warmup_only.yaml",
        "C2_no_internal_gru_only.yaml",
    }
    clone = configs["A0_clone_generated.yaml"]
    clone_csi = clone["model"]["student"]["encoders"]["csi"]
    assert clone["training"]["epochs"] == 20
    assert clone["data"]["dataset"]["csi_degradation"]["enabled"] is False
    assert clone_csi["csi_hardening"]["enabled"] is False
    assert clone_csi["csi_estimation"]["mode"] == "none"
    assert clone_csi["use_internal_gru"] is True
    assert clone_csi["view_fusion"] == "symmetric_gate"
    assert clone_csi["view_gate_warmup_epochs"] == 0
    assert clone_csi["delay_view_warmup_epochs"] == 0
    assert clone["debug"]["config_diff"]["enabled"] is True

    pilot_disabled = configs["A0_clone_pilot_disabled.yaml"]["model"]["student"]["encoders"]["csi"]
    assert pilot_disabled["csi_estimation"]["enabled"] is False

    c1 = configs["C1_view_gate_warmup_only.yaml"]
    c1_csi = c1["model"]["student"]["encoders"]["csi"]
    assert c1_csi["view_gate_warmup_epochs"] == 20
    assert c1_csi["csi_estimation"]["mode"] == clone_csi["csi_estimation"]["mode"]
    assert c1["model"]["student"]["representation_core"] == clone["model"]["student"]["representation_core"]
    assert c1["model"]["student"]["heads"] == clone["model"]["student"]["heads"]

    c2_csi = configs["C2_no_internal_gru_only.yaml"]["model"]["student"]["encoders"]["csi"]
    assert c2_csi["use_internal_gru"] is False
    assert c2_csi["view_gate_warmup_epochs"] == 0


def test_gps_csi_validation_matrix_configs_load():
    root = ROOT / "configs/fusion/csi_hardening_matrix"
    e0 = load_config(root / "E0_gps_only.yaml")
    assert e0["experiment"]["task"] == "gps"
    assert e0["model"]["student"]["modalities"] == ["gps"]

    for filename in [
        "E1_gps_clean_csi_joint.yaml",
        "E2_gps_slow_csi_joint.yaml",
        "E3_gps_slow_csi_prioritized_warmup.yaml",
        "E4_gps_slow_csi_g2d_style.yaml",
    ]:
        cfg = load_config(root / filename)
        assert cfg["experiment"]["task"] == "fusion"
        assert cfg["model"]["student"]["modalities"] == ["gps", "csi"]
        assert cfg["data"]["dataset"]["use_gps"] is True
        assert cfg["data"]["dataset"]["use_csi"] is True

    e3 = load_config(root / "E3_gps_slow_csi_prioritized_warmup.yaml")
    assert e3["training"]["csi_prioritized_warmup"]["enabled"] is True
    assert e3["training"]["csi_prioritized_warmup"]["phase_1"]["active_modalities"] == ["csi"]
    assert e3["training"]["csi_prioritized_warmup"]["phase_2"]["active_modalities"] == ["gps", "csi"]

    e4 = load_config(root / "E4_gps_slow_csi_g2d_style.yaml")
    assert e4["distillation"]["type"] == "g2d"
    assert e4["distillation"]["g2d"]["modalities"] == ["gps", "csi"]
    assert set(e4["distillation"]["g2d"]["teachers"]) == {"gps", "csi"}
    assert e4["distillation"]["g2d"]["smp"]["enabled"] is True


def test_advanced_fusion_overlays_cover_craf_and_marf_ablation_semantics():
    craf = load_config(ROOT / "configs/fusion/overlay_craf_baseline.yaml")
    craf_no_cf = load_config(ROOT / "configs/fusion/overlay_craf_no_counterfactual.yaml")
    marf = load_config(ROOT / "configs/fusion/overlay_marf_baseline.yaml")
    marf_subset = load_config(ROOT / "configs/fusion/overlay_marf_subset_training.yaml")
    marf_no_residual = load_config(ROOT / "configs/fusion/overlay_marf_no_residual.yaml")
    marf_no_prior = load_config(ROOT / "configs/fusion/overlay_marf_no_prior_bias.yaml")

    assert craf["model"]["student"]["type"] == "craf_fusion"
    assert craf["training"]["counterfactual"]["enabled"] is True
    assert craf_no_cf["model"]["student"]["modalities"] == craf["model"]["student"]["modalities"]
    assert craf_no_cf["training"]["counterfactual"]["enabled"] is False
    assert craf_no_cf["loss"]["gate_weight"] == 0.0
    assert marf["model"]["student"]["type"] == "marf_fusion"
    assert marf["training"]["subset_training"]["enabled"] is False
    assert marf_subset["training"]["subset_training"]["enabled"] is True
    assert marf_subset["training"]["subset_training"]["modes"] == ["top_prior", "random_with_top_prior"]
    assert marf_no_residual["model"]["student"]["residual_adapter"]["enabled"] is False
    assert marf_no_prior["model"]["student"]["router"]["use_prior_bias"] is False
    for cfg in (marf_subset, marf_no_residual, marf_no_prior):
        assert cfg["data"]["dataset"]["train_csv_name"] == marf["data"]["dataset"]["train_csv_name"]
        assert cfg["model"]["student"]["modalities"] == marf["model"]["student"]["modalities"]
        assert cfg["training"]["lr"] == marf["training"]["lr"]


def test_advanced_fusion_overlay_builder_rejects_unknown_recipe():
    with pytest.raises(ValueError, match="Unknown advanced fusion overlay"):
        build_advanced_fusion_overlay_config("overlay_unknown_method")


@pytest.mark.parametrize(
    ("config_path", "match"),
    [
        ("configs/fusion/mmwave_gps_logits_kd.yaml", "gps_mmwave_logits_kd.yaml"),
        ("configs/fusion/image_image_rkd.yaml", "duplicate modalities"),
        ("configs/fusion/image_wifi_logits_kd.yaml", "wifi"),
        ("configs/fusion/mmwave_student_no_kd.yaml", "configs/mmwave/student_no_kd.yaml"),
        ("configs/fusion/not_a_canonical_name.yaml", "must end with one of"),
    ],
)
def test_invalid_virtual_fusion_config_paths_raise_clear_errors(config_path: str, match: str):
    with pytest.raises(ValueError, match=match):
        load_config(ROOT / config_path)


def test_missing_noncanonical_config_path_is_not_generated():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(ROOT / "configs/custom/missing.yaml")


def test_unsupported_gps_ablation_configs_are_not_shipped():
    unsupported = [
        "configs/gps/ablation_raw.yaml",
        "configs/gps/ablation_utm.yaml",
        "configs/gps/ablation_relative.yaml",
        "configs/gps/ablation_motion.yaml",
        "configs/gps/ablation_motion_smooth.yaml",
    ]

    assert [path for path in unsupported if (ROOT / path).exists()] == []


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


def test_strict_checkpoint_loading_reports_missing_gru_layer(tmp_path: Path):
    source = ImageStudentModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 1])
    target = ImageStudentModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2])
    checkpoint_path = tmp_path / "one_layer.pth"
    torch.save(source.state_dict(), checkpoint_path)

    with pytest.raises(CheckpointLoadError, match="GRU.weight_ih_l1"):
        load_model_state(checkpoint_path, target, role="student", strict=True)


def test_evaluate_strict_checkpoint_loading_rejects_mismatch(tmp_path: Path):
    weights = tmp_path / "gps_one_layer.pth"
    torch.save(
        GpsStudentModalityNet(gps_input_size=3, feature_size=64, num_classes=64, gru_params=[64, 64, 1]).state_dict(),
        weights,
    )
    cfg = _gps_synthetic_eval_cfg(tmp_path, strict=True)

    with pytest.raises(CheckpointLoadError, match="GRU.weight_ih_l1"):
        evaluate(cfg, weights=str(weights), output_dir=str(tmp_path / "eval_strict"))


def test_evaluate_non_strict_checkpoint_loading_records_mismatch(tmp_path: Path):
    weights = tmp_path / "gps_one_layer.pth"
    torch.save(
        GpsStudentModalityNet(gps_input_size=3, feature_size=64, num_classes=64, gru_params=[64, 64, 1]).state_dict(),
        weights,
    )
    cfg = _gps_synthetic_eval_cfg(tmp_path, strict=False)

    result = evaluate(cfg, weights=str(weights), output_dir=str(tmp_path / "eval_nonstrict"))

    assert "GRU.weight_ih_l1" in result["checkpoint_load"]["missing_keys"]
    assert result["checkpoint_load"]["strict"] is False


def test_training_resume_restores_epoch_optimizer_and_scheduler(tmp_path: Path):
    cfg = _gps_synthetic_train_cfg(tmp_path, epochs=1, resume=False)
    first = train(cfg)
    last_checkpoint = Path(first["run_dir"]) / "checkpoints" / "last.pth"
    assert last_checkpoint.exists()
    assert Path(first["checkpoint_registry"]["path"]).exists()
    assert first["checkpoint_registry"]["source"] == "registry"

    resumed_cfg = _gps_synthetic_train_cfg(tmp_path, epochs=2, resume=True)
    second = train(resumed_cfg)
    resumed_checkpoint = torch.load(Path(second["run_dir"]) / "checkpoints" / "last.pth", map_location="cpu")

    assert resumed_checkpoint["epoch"] == 2
    assert resumed_checkpoint["optimizer"]
    assert resumed_checkpoint["scheduler"]
    assert resumed_checkpoint["best_val_loss"] == second["best_val_loss"]
    assert resumed_checkpoint["early_stopping_metric"] == "val_adba"
    assert resumed_checkpoint["early_stopping_mode"] == "max"
    assert resumed_checkpoint["best_early_stopping_value"] == second["best_early_stopping_value"]
    assert second["checkpoint_loads"][0]["role"] == "resume"


def test_config_validation_rejects_unsupported_image_and_radar_sizes():
    with pytest.raises(ValueError, match="224x224"):
        load_config(ROOT / "configs/image/student_no_kd.yaml", ["data.dataset.image_size=[112,112]"])
    with pytest.raises(ValueError, match="128x64"):
        load_config(ROOT / "configs/radar/student_no_kd.yaml", ["data.dataset.radar_size=[64,32]"])


def _gps_synthetic_eval_cfg(tmp_path: Path, *, strict: bool) -> dict:
    return load_config(
        ROOT / "configs/gps/student_no_kd.yaml",
        [
            "data.dataset.type=synthetic",
            "data.dataset.length=1",
            "data.dataset.seed=7",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "model.student.gru_params=[64,64,2]",
            f"checkpoint.strict_load={str(strict).lower()}",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            f"checkpoint.registry.dir={tmp_path / 'registry'}",
        ],
    )


def _gps_synthetic_train_cfg(tmp_path: Path, *, epochs: int, resume: bool) -> dict:
    return load_config(
        ROOT / "configs/gps/student_no_kd.yaml",
        [
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seed=11",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            f"training.epochs={epochs}",
            f"training.resume={str(resume).lower()}",
            "output.run_name=resume_test",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
        ],
    )
