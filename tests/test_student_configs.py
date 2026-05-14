from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pytest
import torch

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
from kd_sensing.config.io import dump_config, safe_load_yaml  # noqa: E402
from kd_sensing.engine.evaluator import evaluate  # noqa: E402
from kd_sensing.engine.trainer import train  # noqa: E402
from kd_sensing.models.fusion import (  # noqa: E402
    FusionModalityNet,
    FusionTeacherModalityNet,
    FusionStudentModalityNet,
    StudentModalityNet,
)
from kd_sensing.models.gps import GpsModalityNet, GpsStudentModalityNet  # noqa: E402
from kd_sensing.models.image import ImageModalityNet, ImageStudentModalityNet  # noqa: E402
from kd_sensing.models.lidar import LidarModalityNet, LidarStudentModalityNet  # noqa: E402
from kd_sensing.models.radar import RadarModalityNet, RadarStudentModalityNet  # noqa: E402
from kd_sensing.registries import MODELS  # noqa: E402
from kd_sensing.utils.checkpoint import CheckpointLoadError, load_model_state  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


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


def test_fusion_registry_returns_new_public_class_names_and_legacy_aliases_remain_available():
    from kd_sensing.models import FusionModalityNet as PackageFusionModalityNet
    from kd_sensing.models import StudentModalityNet as PackageStudentModalityNet

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
    assert FusionModalityNet is FusionTeacherModalityNet
    assert StudentModalityNet is FusionStudentModalityNet
    assert PackageFusionModalityNet is FusionTeacherModalityNet
    assert PackageStudentModalityNet is FusionStudentModalityNet


MODALITY_SPECS = {
    "image": {
        "task": "image",
        "teacher_type": "image_teacher",
        "student_type": "image_student",
        "teacher_cls": ImageModalityNet,
        "student_cls": ImageStudentModalityNet,
    },
    "radar": {
        "task": "radar",
        "teacher_type": "radar_teacher",
        "student_type": "radar_student",
        "teacher_cls": RadarModalityNet,
        "student_cls": RadarStudentModalityNet,
    },
    "gps": {
        "task": "gps",
        "teacher_type": "gps_teacher",
        "student_type": "gps_student",
        "teacher_cls": GpsModalityNet,
        "student_cls": GpsStudentModalityNet,
    },
    "lidar": {
        "task": "lidar",
        "teacher_type": "lidar_teacher",
        "student_type": "lidar_student",
        "teacher_cls": LidarModalityNet,
        "student_cls": LidarStudentModalityNet,
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
        "image_student",
        None,
        "configs/image/student_no_kd.yaml",
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
        "lidar_teacher",
        None,
        "configs/lidar/teacher_no_kd.yaml",
    ),
    (
        "configs/fusion/no_kd.yaml",
        "fusion",
        "no_kd",
        "fusion_student",
        ["image", "radar"],
        "configs/fusion/image_radar_student_no_kd.yaml",
    ),
    (
        "configs/fusion/logits_kd.yaml",
        "fusion",
        "logits_kd",
        "fusion_student",
        ["image", "radar"],
        "configs/fusion/image_radar_logits_kd.yaml",
    ),
    (
        "configs/fusion/rkd.yaml",
        "fusion",
        "rkd",
        "fusion_student",
        ["image", "radar"],
        "configs/fusion/image_radar_rkd.yaml",
    ),
    (
        "configs/fusion/image_gps_no_kd.yaml",
        "fusion",
        "no_kd",
        "fusion_student",
        ["image", "gps"],
        "configs/fusion/image_gps_student_no_kd.yaml",
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
        "fusion_student",
        ["radar", "lidar"],
        "configs/fusion/radar_lidar_student_no_kd.yaml",
    ),
    (
        "configs/fusion/all_modalities_no_kd.yaml",
        "fusion",
        "no_kd",
        "fusion_student",
        ["image", "radar", "gps"],
        "configs/fusion/image_radar_gps_student_no_kd.yaml",
    ),
    (
        "configs/fusion/all_modalities_lidar_no_kd.yaml",
        "fusion",
        "no_kd",
        "fusion_student",
        ["image", "radar", "gps", "lidar"],
        "configs/fusion/image_radar_gps_lidar_student_no_kd.yaml",
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

    assert cfg["experiment"]["name"] == expected_name
    assert cfg["experiment"]["task"] == spec["task"]
    assert cfg["experiment"]["seed"] == 42
    assert cfg["output"]["run_name"] == expected_name
    assert cfg["data"]["dataloader"]["train_batch_size"] == 32
    assert cfg["data"]["dataloader"]["test_batch_size"] == 32
    assert cfg["data"]["dataloader"]["num_workers"] == 4
    assert cfg["data"]["dataloader"]["prefetch_factor"] == 1
    assert cfg["model"]["teacher"]["type"] == spec["teacher_type"]
    assert cfg["model"]["teacher"]["gru_params"] == SINGLE_GRU_PARAMS
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
    assert model.GRU.num_layers == 1

    if mode in {"teacher_no_kd", "student_no_kd"}:
        assert cfg["distillation"]["type"] == "no_kd"
        assert cfg["distillation"]["teacher_model_name"] is None
    else:
        teacher, student, kd_cfg = _build_teacher_and_student(config_path)
        assert kd_cfg["distillation"]["type"] == mode
        assert kd_cfg["paths"]["weights_dir"] == f"outputs/scene32/{modality}_teacher_no_kd/checkpoints"
        assert kd_cfg["distillation"]["teacher_model_name"] == "best.pth"
        assert isinstance(teacher, spec["teacher_cls"])
        assert isinstance(student, spec["student_cls"])
        assert teacher.GRU.hidden_size == student.GRU.hidden_size == 64
        if mode == "rkd":
            assert kd_cfg["distillation"]["rkd_pairs_per_anchor"] == 4
            assert kd_cfg["distillation"]["rkd_distance_weight"] == 100.0
            assert kd_cfg["distillation"]["rkd_angle_weight"] == 100.0

    if mode == "student_no_kd":
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
    assert cfg["model"]["teacher"]["type"] == "fusion_teacher"
    assert cfg["model"]["teacher"]["modalities"] == modalities
    assert cfg["model"]["student"]["modalities"] == modalities
    assert cfg["model"]["teacher"]["gru_params"] == FUSION_TEACHER_GRU_PARAMS
    expected_student_gru = (
        FUSION_TEACHER_GRU_PARAMS
        if mode == "teacher_no_kd"
        else FUSION_STUDENT_GRU_PARAMS
        if slug == "image_radar"
        else FUSION_TEACHER_GRU_PARAMS
    )
    assert cfg["model"]["student"]["gru_params"] == expected_student_gru

    expected_student_type = "fusion_teacher" if mode == "teacher_no_kd" else "fusion_student"
    expected_student_cls = FusionTeacherModalityNet if mode == "teacher_no_kd" else FusionStudentModalityNet
    assert cfg["model"]["student"]["type"] == expected_student_type
    assert isinstance(student, expected_student_cls)
    assert student.modalities == tuple(modalities)
    assert student.GRU.num_layers == expected_student_gru[-1]

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
        assert kd_cfg["paths"]["weights_dir"] == f"outputs/scene32/{slug}_teacher_no_kd/checkpoints"
        assert kd_cfg["distillation"]["teacher_model_name"] == "best.pth"
        assert isinstance(teacher, FusionTeacherModalityNet)
        assert isinstance(kd_student, FusionStudentModalityNet)
        assert teacher.modalities == kd_student.modalities == tuple(modalities)
        assert teacher.GRU.num_layers == 2
        assert kd_student.GRU.num_layers == expected_student_gru[-1]
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
def test_legacy_configs_keep_compatible_semantics(
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
    if task == "fusion" and modalities == ["image", "radar"]:
        assert cfg["model"]["teacher"]["gru_params"] == FUSION_TEACHER_GRU_PARAMS
        assert cfg["model"]["student"]["gru_params"] == FUSION_STUDENT_GRU_PARAMS
    elif task == "fusion":
        assert cfg["model"]["student"]["gru_params"] == FUSION_TEACHER_GRU_PARAMS
    else:
        assert cfg["model"]["student"]["gru_params"] == SINGLE_GRU_PARAMS

    if distillation_type == "no_kd":
        assert cfg["distillation"]["teacher_model_name"] is None
    elif config_path.startswith("configs/fusion/"):
        assert modalities is not None
        slug = "_".join(modalities)
        assert cfg["paths"]["weights_dir"] == f"outputs/scene32/{slug}_teacher_no_kd/checkpoints"
        assert cfg["distillation"]["teacher_model_name"] == "best.pth"

    if modalities is not None:
        assert cfg["model"]["teacher"]["modalities"] == modalities
        assert cfg["model"]["student"]["modalities"] == modalities
        assert isinstance(model, (FusionTeacherModalityNet, FusionStudentModalityNet))
        assert model.modalities == tuple(modalities)

    _assert_modality_data_fields(cfg, modalities or [task])


def _assert_modality_data_fields(cfg: dict, modalities: list[str]) -> None:
    dataset_cfg = cfg["data"]["dataset"]
    teacher_cfg = cfg["model"]["teacher"]
    student_cfg = cfg["model"]["student"]

    assert dataset_cfg["type"] == "deepsense6g"
    assert dataset_cfg["scene_id"] == 32
    assert dataset_cfg["scene_slug"] == "scene32"
    assert dataset_cfg["data_root"] == "dataset/scenario32"

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
    assert cfg["model"]["teacher"]["gps_input_size"] == 3
    assert cfg["model"]["teacher"]["mmwave_input_size"] == 64
    assert cfg["distillation"]["type"] == "logits_kd"
    assert cfg["distillation"]["teacher_model_name"] == "best.pth"
    assert cfg["paths"]["weights_dir"] == "outputs/scene32/gps_mmwave_teacher_no_kd/checkpoints"
    assert cfg["training"]["early_stopping_metric"] == "val_adba"
    assert cfg["training"]["early_stopping_mode"] == "max"


def test_virtual_image_radar_config_generator_keeps_compatibility_params():
    cfg = build_virtual_fusion_config("image_radar_logits_kd")

    assert cfg["experiment"]["seed"] == 42
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 2]
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 1]
    assert cfg["training"]["lr"] == 0.00095
    assert cfg["training"]["weight_decay"] == 0.0
    assert cfg["distillation"]["temperature"] == 2.0
    assert cfg["distillation"]["alpha"] == 0.4
    assert cfg["distillation"]["teacher_model_name"] == "best.pth"
    assert cfg["paths"]["weights_dir"] == "outputs/scene32/image_radar_teacher_no_kd/checkpoints"
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
