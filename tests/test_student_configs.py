from itertools import combinations
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.config.canonical import (  # noqa: E402
    CANONICAL_FUSION_MODALITIES,
    build_virtual_fusion_config,
    training_overrides,
)
from kd_sensing.engine.run_lineage import is_historical_kd_metadata, run_lineage_metadata  # noqa: E402
from kd_sensing.engine.model_output import adapt_model_output  # noqa: E402
from kd_sensing.models.fusion.cls_token_transformer import CLSTokenTransformerFusionNet  # noqa: E402
from kd_sensing.models.modular import ModularSequenceModel  # noqa: E402
from kd_sensing.models.radar import RadarFeatureExtractor  # noqa: E402
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
    "model_type",
    [
        "image_strong",
        "image_lightweight",
        "radar_strong",
        "radar_lightweight",
        "gps_strong",
        "gps_lightweight",
        "lidar_strong",
        "lidar_lightweight",
        "mmwave_strong",
        "mmwave_lightweight",
        "fusion_strong",
        "fusion_lightweight",
    ],
)
def test_strong_and_lightweight_registry_names_are_removed(model_type: str):
    if model_type.startswith("fusion_"):
        with pytest.raises(RegistryError, match="Removed component"):
            MODELS.build(_minimal_registry_config(model_type))
        return
    with pytest.raises(RegistryError, match="Unknown component"):
        MODELS.build(_minimal_registry_config(model_type))


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
    with pytest.raises(RegistryError, match="Unknown component"):
        MODELS.build({"type": removed_type})


@pytest.mark.parametrize(
    ("config_path", "expected_type", "expected_cls"),
    [
        ("configs/radar/strong.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/radar/lightweight.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/radar/supervised.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/gps/strong.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/gps/lightweight.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/gps/supervised.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/mmwave/strong.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/mmwave/lightweight.yaml", "modular_sequence", ModularSequenceModel),
        ("configs/mmwave/supervised.yaml", "modular_sequence", ModularSequenceModel),
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
        "model": {"primary": {"type": "modular_sequence", "modalities": ["gps"]}},
    }

    lineage = run_lineage_metadata(cfg)

    assert lineage == {
        "training_mode": "adaptation",
        "method_family": "gps_adapter_v2",
        "model_capacity": "primary",
        "primary_model": "modular_sequence",
        "main_conclusion_eligible": True,
    }
    assert is_historical_kd_metadata({"distillation_enabled": True}) is True
    assert is_historical_kd_metadata({"method_family": "legacy_kd"}) is True


def test_fusion_registry_removed_aliases_fail_and_current_owner_exports_remain():
    assert CLSTokenTransformerFusionNet is not None
    assert "__all__" not in vars(kd_sensing.models)
    for alias in ["Fusion" + "ModalityNet", "Student" + "ModalityNet"]:
        with pytest.raises(AttributeError, match=alias):
            getattr(kd_sensing.models, alias)
        with pytest.raises(RegistryError, match="Unknown component"):
            MODELS.build({"type": alias})
    for alias in ["fusion_strong", "fusion_lightweight"]:
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
        expected_type = "modular_sequence"
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
        "configs/csi/hardening_matrix/B4_fixed_antenna_permutation.yaml",
        "configs/csi/hardening_matrix/B5_mild_hardening_combo.yaml",
        "configs/csi/hardening_matrix/B6_medium_hardening_combo.yaml",
        "configs/csi/hardening_matrix/C1_view_gate_warmup.yaml",
        "configs/csi/hardening_matrix/C2_no_internal_gru.yaml",
        "configs/csi/hardening_matrix/D1_mild_hardening_gate_warmup.yaml",
        "configs/csi/hardening_matrix/D2_mild_hardening_no_internal_gru.yaml",
        "configs/csi/hardening_matrix/D3_mild_hardening_gate_warmup_no_internal_gru.yaml",
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
    assert cfg["config_resolution"]["style"] == "base+overlay"
    assert cfg["config_resolution"]["overlay_id"] == Path(config_path).stem
    if "csi" in primary.get("modalities", []):
        assert primary["encoders"]["csi"]["type"] == "pilot_dual_view_csi"

    dataset = cfg["data"]["dataset"]
    if Path(config_path).stem == "A2_destructive_degradation":
        assert dataset["csi_degradation"]["enabled"] is True
        assert "csi_hardening" not in dataset
    if Path(config_path).stem.startswith("D"):
        assert "csi_degradation" not in dataset
    if Path(config_path).stem in {"E1_gps_clean_csi_joint", "E2_gps_slow_csi_joint", "E3_gps_slow_csi_prioritized_warmup"}:
        assert primary["modalities"] == ["gps", "csi"]


def test_modular_radar_configs_forward_contracts():
    batch = torch.rand(2, 8, 2, 128, 64)
    for config_path in ("configs/radar/strong.yaml", "configs/radar/lightweight.yaml"):
        cfg = _load(config_path)
        model = MODELS.build(cfg["model"]["primary"])
        output = adapt_model_output(model(radar_batch=batch))
        assert output.logits.shape == (2, 8, 64)
        assert output.input_features.shape[:2] == (2, 8)


def test_modular_gps_configs_forward_contracts():
    batch = torch.rand(2, 8, 3)
    for config_path in ("configs/gps/strong.yaml", "configs/gps/lightweight.yaml"):
        cfg = _load(config_path)
        model = MODELS.build(cfg["model"]["primary"])
        output = adapt_model_output(model(gps_batch=batch))
        assert output.logits.shape == (2, 8, 64)
        assert output.input_features.shape[:2] == (2, 8)


def test_modular_mmwave_configs_forward_contracts():
    batch = torch.rand(2, 8, 64)
    for config_path in ("configs/mmwave/strong.yaml", "configs/mmwave/lightweight.yaml"):
        cfg = _load(config_path)
        model = MODELS.build(cfg["model"]["primary"])
        output = adapt_model_output(model(mmwave_batch=batch))
        assert output.logits.shape == (2, 8, 64)
        assert output.input_features.shape[:2] == (2, 8)


def test_modular_radar_gps_fusion_config_forward_contract():
    cfg = _load("configs/fusion/radar_gps_supervised.yaml")
    primary = cfg["model"]["primary"]
    model = MODELS.build(primary)

    assert primary["type"] == "modular_sequence"
    assert primary["modalities"] == ["radar", "gps"]
    output = adapt_model_output(
        model(
            radar_batch=torch.rand(2, 8, 2, 128, 64),
            gps_batch=torch.rand(2, 8, 3),
        )
    )
    assert output.logits.shape == (2, 8, 64)
    assert output.input_features.shape[:2] == (2, 8)


def test_radar_feature_extractor_rejects_invalid_inputs():
    extractor = RadarFeatureExtractor(64, in_channels=2)
    with pytest.raises(ValueError, match="128x64"):
        extractor(torch.rand(1, 2, 2, 64, 64))
    with pytest.raises(ValueError, match="channel count"):
        extractor(torch.rand(1, 2, 1, 128, 64))
