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
    build_virtual_fusion_config,
)
from kd_sensing.config.canonical_recipes import training_overrides  # noqa: E402
from kd_sensing.engine.run_lineage import is_historical_kd_metadata, run_lineage_metadata  # noqa: E402
from kd_sensing.engine.model_output import adapt_model_output  # noqa: E402
from kd_sensing.models.fusion import (  # noqa: E402
    CLSTokenTransformerFusionNet,
    FusionLightweightModalityNet,
    FusionStrongModalityNet,
)
from kd_sensing.models.gps import GpsLightweightModalityNet, GpsStrongModalityNet  # noqa: E402
from kd_sensing.models.image import ImageLightweightModalityNet, ImageStrongModalityNet  # noqa: E402
from kd_sensing.models.lidar import LidarLightweightModalityNet, LidarStrongModalityNet  # noqa: E402
from kd_sensing.models.mmwave import MmWaveLightweightModalityNet, MmWaveStrongModalityNet  # noqa: E402
from kd_sensing.models.modular import ModularSequenceModel  # noqa: E402
from kd_sensing.models.radar import RadarLightweightModalityNet, RadarStrongModalityNet  # noqa: E402
from kd_sensing.registries import MODELS, RegistryError  # noqa: E402

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


def _load(path: str, overrides: list[str] | None = None) -> dict:
    return load_config(ROOT / path, overrides)


def _assert_distillation_free_config(cfg: dict) -> None:
    model_cfg = cfg["model"]
    assert "distillation" not in cfg
    assert "teacher" not in model_cfg
    assert "student" not in model_cfg
    assert "seq_length_teacher" not in model_cfg
    assert "seq_length_student" not in model_cfg
    assert isinstance(model_cfg.get("primary"), dict)


def _minimal_registry_config(model_type: str) -> dict:
    cfg = {
        "type": model_type,
        "feature_size": 64,
        "num_classes": 64,
        "gru_params": [64, 64, 1],
    }
    if model_type.startswith("image"):
        cfg["image_channels"] = 3
    elif model_type.startswith("radar"):
        cfg["radar_channels"] = 2
    elif model_type.startswith("gps"):
        cfg["gps_input_size"] = 3
    elif model_type.startswith("lidar"):
        cfg["lidar_channels"] = 3
    elif model_type.startswith("mmwave"):
        cfg["mmwave_input_size"] = 64
    elif model_type.startswith("fusion"):
        cfg.update(
            {
                "modalities": ["gps", "mmwave"],
                "gps_input_size": 3,
                "mmwave_input_size": 64,
            }
        )
    return cfg


@pytest.mark.parametrize(
    ("model_type", "expected_cls"),
    [
        ("image_strong", ImageStrongModalityNet),
        ("image_lightweight", ImageLightweightModalityNet),
        ("radar_strong", RadarStrongModalityNet),
        ("radar_lightweight", RadarLightweightModalityNet),
        ("gps_strong", GpsStrongModalityNet),
        ("gps_lightweight", GpsLightweightModalityNet),
        ("lidar_strong", LidarStrongModalityNet),
        ("lidar_lightweight", LidarLightweightModalityNet),
        ("mmwave_strong", MmWaveStrongModalityNet),
        ("mmwave_lightweight", MmWaveLightweightModalityNet),
        ("fusion_strong", FusionStrongModalityNet),
        ("fusion_lightweight", FusionLightweightModalityNet),
    ],
)
def test_strong_and_lightweight_registry_names_are_public(model_type: str, expected_cls: type):
    model = MODELS.build(_minimal_registry_config(model_type))

    assert isinstance(model, expected_cls)


@pytest.mark.parametrize(
    "removed_type",
    [
        "image_teacher",
        "image_student",
        "radar_teacher",
        "radar_student",
        "gps_teacher",
        "gps_student",
        "lidar_teacher",
        "lidar_student",
        "mmwave_teacher",
        "mmwave_student",
        "fusion_teacher",
        "fusion_student",
    ],
)
def test_removed_teacher_student_registry_names_fail_fast(removed_type: str):
    with pytest.raises(RegistryError, match="Removed component"):
        MODELS.build({"type": removed_type})


@pytest.mark.parametrize(
    ("config_path", "expected_type", "expected_cls"),
    [
        ("configs/radar/strong.yaml", "radar_strong", RadarStrongModalityNet),
        ("configs/radar/lightweight.yaml", "radar_lightweight", RadarLightweightModalityNet),
        ("configs/radar/supervised.yaml", "radar_strong", RadarStrongModalityNet),
        ("configs/gps/strong.yaml", "gps_strong", GpsStrongModalityNet),
        ("configs/gps/lightweight.yaml", "gps_lightweight", GpsLightweightModalityNet),
        ("configs/gps/supervised.yaml", "gps_strong", GpsStrongModalityNet),
        ("configs/mmwave/strong.yaml", "mmwave_strong", MmWaveStrongModalityNet),
        ("configs/mmwave/lightweight.yaml", "mmwave_lightweight", MmWaveLightweightModalityNet),
        ("configs/mmwave/supervised.yaml", "mmwave_strong", MmWaveStrongModalityNet),
        ("configs/image/strong.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/image/lightweight.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/image/supervised.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/lidar/strong.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/lidar/lightweight.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/lidar/supervised.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/csi/supervised.yaml", "modular_sequence", ModularSequenceModel),
    ],
)
def test_single_modality_configs_use_primary_model(config_path: str, expected_type: str, expected_cls: type):
    cfg = _load(config_path)
    _assert_distillation_free_config(cfg)

    assert cfg["model"]["primary"]["type"] == expected_type
    model = MODELS.build(cfg["model"]["primary"])
    assert isinstance(model, expected_cls)


def test_run_lineage_metadata_uses_distillation_free_fields():
    cfg = {
        "experiment": {
            "name": "deepsense6g_gps_adapter_v2",
            "task": "gps",
            "training_mode": "adaptation",
            "method_family": "gps_adapter_v2",
        },
        "model": {"primary": {"type": "gps_strong"}},
    }

    lineage = run_lineage_metadata(cfg)

    assert lineage == {
        "training_mode": "adaptation",
        "method_family": "gps_adapter_v2",
        "model_capacity": "strong",
        "primary_model": "gps_strong",
        "main_conclusion_eligible": True,
    }
    assert is_historical_kd_metadata({"distillation_enabled": True}) is True
    assert is_historical_kd_metadata({"method_family": "legacy_kd"}) is True


def test_fusion_registry_removed_aliases_fail_but_public_exports_remain():
    assert kd_sensing.models.FusionStrongModalityNet is FusionStrongModalityNet
    assert kd_sensing.models.FusionLightweightModalityNet is not None
    for alias in ["Fusion" + "ModalityNet", "Student" + "ModalityNet"]:
        with pytest.raises(AttributeError, match=alias):
            getattr(kd_sensing.models, alias)
        with pytest.raises(RegistryError, match="Removed component"):
            MODELS.build({"type": alias})


FUSION_SLUGS = {
    "_".join(modalities): list(modalities)
    for size in (2, 3, 4, 5)
    for modalities in combinations(CANONICAL_FUSION_MODALITIES, size)
}


@pytest.mark.parametrize("slug,modalities", sorted(FUSION_SLUGS.items()))
@pytest.mark.parametrize("mode", ["strong", "lightweight"])
def test_virtual_fusion_configs_use_primary_and_no_distillation(slug: str, modalities: list[str], mode: str):
    cfg = build_virtual_fusion_config(f"{slug}_{mode}")
    _assert_distillation_free_config(cfg)

    assert cfg["experiment"]["name"] == f"{slug}_{mode}"
    assert cfg["model"]["modalities"] == modalities
    assert cfg["model"]["primary"]["modalities"] == modalities
    if mode == "strong":
        expected_type = "modular_sequence" if {"image", "lidar"} & set(modalities) else "fusion_strong"
    else:
        expected_type = "cls_token_transformer_fusion"
    assert cfg["model"]["primary"]["type"] == expected_type


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/fusion/all_modalities_supervised.yaml",
        "configs/fusion/all_modalities_lidar_supervised.yaml",
        "configs/fusion/image_gps_supervised.yaml",
        "configs/fusion/image_gps_resnet18_modular_supervised.yaml",
        "configs/fusion/mmwave_csi_supervised.yaml",
        "configs/fusion/mmwave_csi_medium_degraded_supervised.yaml",
        "configs/fusion/radar_gps_supervised.yaml",
        "configs/fusion/radar_lidar_supervised.yaml",
        "configs/fusion/token_transformer_all_modalities_supervised.yaml",
        "configs/fusion/token_transformer_all_modalities_multitask_supervised.yaml",
        "configs/fusion/token_transformer_image_radar_supervised.yaml",
    ],
)
def test_existing_fusion_supervised_configs_are_primary_only(config_path: str):
    cfg = _load(config_path)
    _assert_distillation_free_config(cfg)

    primary = cfg["model"]["primary"]
    assert primary["modalities"] == cfg["model"].get("modalities", primary["modalities"])
    assert primary["type"] in {
        "cls_token_transformer_fusion",
        "fusion_lightweight",
        "fusion_strong",
        "modular_sequence",
        "token_transformer_fusion",
    }


@pytest.mark.parametrize(
    ("old_path", "replacement"),
    [
        ("configs/radar/teacher_no_kd.yaml", "configs/radar/strong.yaml"),
        ("configs/radar/student_no_kd.yaml", "configs/radar/lightweight.yaml"),
        ("configs/radar/no_kd.yaml", "configs/radar/supervised.yaml"),
        ("configs/radar/logits_kd.yaml", "configs/radar/lightweight.yaml"),
        ("configs/radar/rkd.yaml", "configs/radar/lightweight.yaml"),
        ("configs/fusion/image_radar_no_kd.yaml", "configs/fusion/image_radar_lightweight.yaml"),
        ("configs/fusion/image_gps_no_kd.yaml", "configs/fusion/image_gps_supervised.yaml"),
        ("configs/fusion/image_radar_logits_kd.yaml", "configs/fusion/image_radar_lightweight.yaml"),
    ],
)
def test_removed_config_paths_fail_fast_with_replacements(old_path: str, replacement: str):
    with pytest.raises(ValueError, match="KD support has been removed") as exc_info:
        _load(old_path)

    assert replacement in str(exc_info.value)


@pytest.mark.parametrize(
    "override",
    [
        "distillation.type=logits_kd",
        "distillation.teacher_model_name=best.pth",
        "kd_mode=logits_kd",
        "kd.temperature=4.0",
        "teacher_model_name=best.pth",
    ],
)
def test_removed_kd_overrides_fail_fast(override: str):
    with pytest.raises(ValueError, match="KD support has been removed"):
        _load("configs/radar/strong.yaml", [override])


@pytest.mark.parametrize("stem", ["gps_mmwave_logits_kd", "gps_mmwave_rkd", "gps_mmwave_teacher_no_kd"])
def test_virtual_fusion_retired_kd_aliases_fail_fast(stem: str):
    with pytest.raises(ValueError, match="KD support has been removed"):
        build_virtual_fusion_config(stem)


def test_training_recipe_modes_are_distillation_free():
    assert training_overrides("strong", image_radar=False)["lr"] == pytest.approx(0.00075)
    assert training_overrides("lightweight", image_radar=False)["lr"] == pytest.approx(0.00075)
    assert training_overrides("strong", image_radar=True)["lr"] == pytest.approx(0.00075)
    assert training_overrides("lightweight", image_radar=True)["lr"] == pytest.approx(0.0004)


def test_fusion_config_order_and_missing_noncanonical_errors_are_preserved():
    with pytest.raises(ValueError, match="gps_mmwave_lightweight.yaml"):
        _load("configs/fusion/mmwave_gps_lightweight.yaml")
    with pytest.raises(ValueError, match="duplicate"):
        _load("configs/fusion/image_image_lightweight.yaml")
    with pytest.raises(ValueError, match="wifi"):
        _load("configs/fusion/image_wifi_lightweight.yaml")
    with pytest.raises(ValueError, match="configs/mmwave/lightweight.yaml"):
        _load("configs/fusion/mmwave_lightweight.yaml")


def test_cli_overrides_can_select_fusion_modalities_on_new_paths():
    cfg = _load(
        "configs/fusion/gps_mmwave_lightweight.yaml",
        ["model.modalities=[gps,mmwave]", "data.dataset.scene=32"],
    )

    _assert_distillation_free_config(cfg)
    assert cfg["data"]["dataset"]["scene"] == 32
    assert cfg["model"]["primary"]["modalities"] == ["gps", "mmwave"]


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/raymobtime/s008_smoke_selection.yaml",
        "configs/raymobtime/s008_multitask_selection.yaml",
        "configs/gps/gps_neural_coarse_smoke.yaml",
        "configs/gps/ablation_relative_polar.yaml",
    ],
)
def test_specialized_configs_use_primary_without_distillation(config_path: str):
    cfg = _load(config_path)
    _assert_distillation_free_config(cfg)


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/csi/hardening_matrix/A0_clean_full_strong.yaml",
        "configs/csi/hardening_matrix/A1_mild_pilot_estimation.yaml",
        "configs/csi/hardening_matrix/A2_destructive_degradation.yaml",
        "configs/csi/hardening_matrix/B3_antenna_calibration.yaml",
        "configs/csi/hardening_matrix/B5_mild_hardening_combo.yaml",
        "configs/csi/hardening_matrix/C1_view_gate_warmup.yaml",
        "configs/csi/hardening_matrix/C2_no_internal_gru.yaml",
        "configs/csi/hardening_matrix/D4_medium_hardening_gate_warmup_no_internal_gru.yaml",
        "configs/fusion/csi_hardening_matrix/E0_gps_only.yaml",
        "configs/fusion/csi_hardening_matrix/E1_gps_clean_csi_joint.yaml",
        "configs/fusion/csi_hardening_matrix/E2_gps_slow_csi_joint.yaml",
        "configs/fusion/csi_hardening_matrix/E3_gps_slow_csi_prioritized_warmup.yaml",
    ],
)
def test_csi_matrix_configs_use_primary_encoder(config_path: str):
    cfg = _load(config_path)
    _assert_distillation_free_config(cfg)

    primary = cfg["model"]["primary"]
    if "csi" in primary.get("modalities", []):
        assert primary["encoders"]["csi"]["type"] == "pilot_dual_view_csi"


def test_radar_strong_and_lightweight_forward_contracts():
    batch = torch.rand(2, 8, 2, 128, 64)
    for model_type in ("radar_strong", "radar_lightweight"):
        model = MODELS.build(
            {
                "type": model_type,
                "radar_channels": 2,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64, 1],
            }
        )
        output = adapt_model_output(model(batch))
        assert output.logits.shape == (2, 8, 64)
        assert output.input_features.shape[:2] == (2, 8)


def test_radar_strong_rejects_invalid_attention_heads():
    with pytest.raises(ValueError, match="divisible by num_heads"):
        MODELS.build(
            {
                "type": "radar_strong",
                "radar_channels": 2,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 65, 1],
                "num_heads": 8,
            }
        )


def test_radar_lightweight_rejects_invalid_gru_params_length():
    with pytest.raises(ValueError, match="gru_params"):
        MODELS.build(
            {
                "type": "radar_lightweight",
                "radar_channels": 2,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64],
            }
        )
