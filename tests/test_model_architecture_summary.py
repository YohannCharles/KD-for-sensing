import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import kd_sensing.models.image_encoders as image_encoders
import kd_sensing.models.tinyvit as tinyvit
from kd_sensing.cli.model_architecture_summary import build_parser, main as model_summary_main
from kd_sensing.diagnostics.jepa_visual_architecture_sweep import load_sweep_manifest
from kd_sensing.engine.debug_diagnostics import build_startup_summary, module_trainability_report
from kd_sensing.models.architecture_summary import (
    ACTUAL_PARAMETER_SOURCE,
    CANDIDATE_PARAMETER_SOURCE,
    render_architecture_summary,
    summarize_model_architecture,
    summarize_model_config,
    summarize_sweep_candidate,
)
from kd_sensing.models.modular import ModularSequenceModel
from kd_sensing.models.tinyvit import TinyViTImageEncoder
from kd_sensing.registries import ENCODERS, HEADS, MODELS, REPRESENTATION_CORES, import_default_components


ROOT = Path(__file__).resolve().parents[1]
SWEEP_MANIFEST = ROOT / "configs/diagnostics/jepa_visual_architecture_sweep_manifest.yaml"


class _SyntheticSummaryModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(4, 3)
        self.shared = nn.Linear(3, 3, bias=False)
        self.shared_alias = self.shared
        self.mystery = nn.Linear(3, 2)
        self.unused = nn.Linear(2, 2)
        for param in self.mystery.parameters():
            param.requires_grad = False

    def architecture_excluded_parameter_groups(self) -> list[dict[str, object]]:
        return [
            {
                "name": "unused_tail",
                "path": "unused",
                "parameter_prefixes": ["unused."],
                "reason": "synthetic downstream path does not call unused tail",
            }
        ]

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mystery(self.shared(self.stem(value)))


class _FakeResNetBackbone(nn.Module):
    def __init__(self, output_dim: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.layer1 = nn.Linear(output_dim, output_dim)
        self.layer2 = nn.Linear(output_dim, output_dim)
        self.layer3 = nn.Linear(output_dim, output_dim)
        self.layer4 = nn.Linear(output_dim, output_dim)
        self.output_dim = int(output_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.zeros((int(frames.shape[0]), self.output_dim), dtype=frames.dtype, device=frames.device)


class _FakeTinyViTBackbone(nn.Module):
    def __init__(self, backbone_dim: int = 8) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(1, backbone_dim)
        self.layers = nn.ModuleList(nn.Linear(backbone_dim, backbone_dim) for _ in range(4))
        self.norm_head = nn.LayerNorm(backbone_dim)
        self.head = nn.Linear(backbone_dim, 21841)

    def forward_features(self, frames: torch.Tensor) -> torch.Tensor:
        features = frames.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        features = self.patch_embed(features)
        for layer in self.layers:
            features = layer(features)
        return features


@pytest.fixture()
def fake_resnet_backbone(monkeypatch):
    def build(*, pretrained: bool, weights: str | None):
        del pretrained, weights
        return _FakeResNetBackbone(output_dim=8), 8

    monkeypatch.setattr(image_encoders, "_build_resnet18_backbone", build)


@pytest.fixture()
def fake_tinyvit_backbone(monkeypatch):
    def build(variant: str, *, in_chans: int = 3):
        del variant, in_chans
        return _FakeTinyViTBackbone(backbone_dim=8), 8

    monkeypatch.setattr(tinyvit, "_build_tinyvit_backbone", build)


def _image_only_model() -> ModularSequenceModel:
    import_default_components()
    return ModularSequenceModel(
        modalities=["image"],
        image_profile="rgb_imagenet",
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
        encoders={"image": {"type": "resnet18_imagenet_rgb", "output_dim": 8, "pretrained": False}},
        projectors={"image": {"type": "identity", "input_dim": 8, "d_model": 8}},
        representation_core={"type": "single_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
    )


def _image_gps_model() -> ModularSequenceModel:
    import_default_components()
    return ModularSequenceModel(
        modalities=["image", "gps"],
        image_profile="rgb_imagenet",
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
        gps_input_size=3,
        encoders={
            "image": {"type": "resnet18_imagenet_rgb", "output_dim": 8, "pretrained": False},
            "gps": {"type": "gps_mlp", "output_dim": 8, "gps_input_size": 3, "hidden_size": 8},
        },
        projectors={
            "image": {"type": "identity", "input_dim": 8, "d_model": 8},
            "gps": {"type": "identity", "input_dim": 8, "d_model": 8},
        },
        representation_core={"type": "early_concat_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
    )


def test_schema_json_source_warnings_and_renderers():
    model = _SyntheticSummaryModule()

    summary = summarize_model_architecture(model, source={"kind": "instance", "config_path": "synthetic"})

    assert {
        "schema_version",
        "source",
        "model",
        "parameters",
        "components",
        "warnings",
        "comparability",
    } <= set(summary)
    assert summary["source"]["kind"] == "instance"
    assert summary["parameters"]["parameter_count_source"] == ACTUAL_PARAMETER_SOURCE
    json.dumps(summary)

    warning_codes = {item["code"] for item in summary["warnings"]}
    assert "unused_parameter_group" in warning_codes
    assert "unknown_component_role" in warning_codes
    assert "## Model Architecture Summary" in render_architecture_summary(summary, format="markdown")
    csv_text = render_architecture_summary(summary, format="csv")
    assert "total_params" in csv_text
    assert json.loads(render_architecture_summary(summary, format="json"))["schema_version"] == 1


def test_synthetic_parameter_counts_dedupe_and_effective_excluded():
    model = _SyntheticSummaryModule()
    expected_total = sum(param.numel() for _, param in model.named_parameters())
    expected_trainable = sum(param.numel() for _, param in model.named_parameters() if param.requires_grad)
    expected_excluded = sum(param.numel() for param in model.unused.parameters())

    summary = summarize_model_architecture(model)

    assert summary["parameters"]["total_params"] == expected_total
    assert summary["parameters"]["trainable_params"] == expected_trainable
    assert summary["parameters"]["frozen_params"] == expected_total - expected_trainable
    assert summary["parameters"]["excluded_params"] == expected_excluded
    assert summary["parameters"]["effective_params"] == expected_total - expected_excluded
    assert len(summary["parameters"]["excluded_parameter_groups"]) == 1


def test_modular_sequence_image_only_resnet_summary_and_startup_compat(fake_resnet_backbone):
    model = _image_only_model()

    summary = summarize_model_architecture(model)
    components = summary["components"]
    assert components["encoders.image"]["semantic_role"] == "image_encoder"
    assert components["projectors.image"]["semantic_role"] == "projector"
    assert components["representation_core"]["semantic_role"] == "representation_core"
    assert components["heads.beam"]["semantic_role"] == "beam_head"
    assert summary["parameters"]["image_encoder_params"] == components["encoders.image"]["total_params"]

    report = module_trainability_report(model)
    assert {"total_params", "trainable_params", "modules"} <= set(report)
    assert report["total_params"] == summary["parameters"]["total_params"]
    assert "image_encoder" in report["modules"]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    startup = build_startup_summary(
        {"model": {"primary": {"type": "modular_sequence", "modalities": ["image"]}}, "data": {"dataset": {}}},
        model,
        optimizer,
        None,
        device=torch.device("cpu"),
    )
    assert startup["parameters"]["total_params"] == summary["parameters"]["total_params"]
    assert startup["parameters"]["trainable_params"] == summary["parameters"]["trainable_params"]
    assert startup["parameters"]["modules"]
    assert startup["architecture_summary"]["schema_version"] == 1


def test_modular_sequence_image_gps_summary_metadata(fake_resnet_backbone):
    model = _image_gps_model()

    summary = summarize_model_architecture(model)

    assert summary["model"]["enabled_modalities"] == ["image", "gps"]
    assert summary["components"]["encoders.image"]["semantic_role"] == "image_encoder"
    assert summary["components"]["encoders.gps"]["semantic_role"] == "gps_encoder"
    assert summary["components"]["representation_core"]["semantic_role"] == "representation_core"
    assert summary["parameters"]["modality_encoder_params"]["gps"] == summary["components"]["encoders.gps"]["total_params"]
    assert summary["components"]["encoders.image"]["metadata"]["registry_type"] == "resnet18_imagenet_rgb"


@pytest.mark.parametrize(
    ("config_path", "modalities", "encoder_components", "core_type"),
    [
        ("configs/radar/strong.yaml", ["radar"], ["encoders.radar"], "single_gru"),
        ("configs/gps/strong.yaml", ["gps"], ["encoders.gps"], "single_gru"),
        ("configs/mmwave/strong.yaml", ["mmwave"], ["encoders.mmwave"], "single_gru"),
        ("configs/fusion/radar_gps_supervised.yaml", ["radar", "gps"], ["encoders.radar", "encoders.gps"], "early_concat_gru"),
    ],
)
def test_migrated_modular_configs_generate_architecture_summary(
    config_path: str,
    modalities: list[str],
    encoder_components: list[str],
    core_type: str,
):
    summary = summarize_model_config(ROOT / config_path, build=True)

    assert summary["source"]["kind"] == "instance"
    assert summary["model"]["registry_type"] == "modular_sequence"
    assert summary["model"]["class"] == "ModularSequenceModel"
    assert summary["model"]["enabled_modalities"] == modalities
    for component in encoder_components:
        assert component in summary["components"]
    assert summary["model"]["metadata"]["representation_core_type"] == core_type


def test_amr_net_config_generates_whole_model_summary():
    summary = summarize_model_config(ROOT / "configs/fusion/amr_net_supervised.yaml", build=True)

    assert summary["source"]["kind"] == "instance"
    assert summary["model"]["registry_type"] == "amr_net"
    assert summary["model"]["architecture_category"] == "whole_model_exception"
    assert summary["model"]["enabled_modalities"] == ["image", "lidar", "gps"]
    assert summary["parameters"]["total_params"] > 0
    assert summary["model"]["metadata"]["paper_approximation"] is True


def test_removed_registry_names_are_absent_from_current_architecture_surface():
    import_default_components()
    assert {
        "modular_sequence_model",
        "gps_only_neural_baseline",
        "radar_feature_extractor",
        "lidar_feature_extractor",
        "mmwave_feature_extractor",
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
        "fusion_lightweight",
        "fusion_strong",
    }.isdisjoint(MODELS.list())
    assert "point_cloud_mlp" not in ENCODERS.list()
    assert "jepa_token_transformer" not in REPRESENTATION_CORES.list()
    assert "safe_residual_reranker" not in HEADS.list()


def test_tinyvit_modular_summary_metadata_unused_head_and_requires_grad_unchanged(fake_tinyvit_backbone):
    import_default_components()
    model = ModularSequenceModel(
        modalities=["image"],
        image_profile="rgb_imagenet",
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
        encoders={"image": {"type": "tinyvit_5m_scratch_rgb", "output_dim": 8}},
        projectors={"image": {"type": "identity", "input_dim": 8, "d_model": 8}},
        representation_core={"type": "single_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
    )
    requires_grad_before = {name: param.requires_grad for name, param in model.named_parameters()}

    summary = summarize_model_architecture(model)

    assert requires_grad_before == {name: param.requires_grad for name, param in model.named_parameters()}
    image = summary["components"]["encoders.image"]
    assert image["metadata"]["registry_type"] == "tinyvit_5m_scratch_rgb"
    assert image["metadata"]["variant"] == "5m"
    assert image["metadata"]["backbone_dim"] == 8
    assert image["metadata"]["output_dim"] == 8
    assert image["metadata"]["freeze_policy"] == "frozen_backbone"
    assert image["metadata"]["trainable_stages"] == []
    assert image["metadata"]["checkpoint_source"] == "none"
    assert summary["parameters"]["excluded_params"] == sum(param.numel() for param in model.encoders["image"].backbone.head.parameters())
    assert "unused_parameter_group" in {warning["code"] for warning in summary["warnings"]}


def test_tinyvit_override_preflight_warns_for_resnet_stage(fake_tinyvit_backbone):
    cfg = {
        "type": "modular_sequence",
        "modalities": ["image"],
        "image_profile": "rgb_imagenet",
        "feature_size": 8,
        "d_model": 8,
        "num_classes": 5,
        "encoders": {
            "image": {
                "type": "tinyvit_5m_scratch_rgb",
                "output_dim": 8,
                "unfreeze_stages": ["layer4"],
            }
        },
        "projectors": {"image": {"type": "identity", "input_dim": 8, "d_model": 8}},
        "representation_core": {"type": "single_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
    }

    summary = summarize_model_config(cfg, build=True)

    assert summary["source"]["kind"] == "config_preflight"
    warning = next(item for item in summary["warnings"] if item["code"] == "incompatible_encoder_option")
    assert "layer4" in warning["message"]
    assert "norm_head" in warning["available_options"]


def test_tinyvit_22k_preflight_blocks_potential_download():
    cfg = {
        "type": "modular_sequence",
        "modalities": ["image"],
        "encoders": {"image": {"type": "tinyvit_5m_22k_rgb", "output_dim": 8}},
    }

    summary = summarize_model_config(cfg, build=True, allow_download=False)

    assert summary["source"]["kind"] == "config_preflight"
    assert "potential_checkpoint_download" in {warning["code"] for warning in summary["warnings"]}


def test_sweep_candidate_parameter_fixtures():
    manifest = load_sweep_manifest(SWEEP_MANIFEST)
    candidates = {candidate["variant_id"]: candidate for candidate in manifest["candidates"]}

    patch = summarize_sweep_candidate(candidates["patch14_stage1_gps_query"])
    layer4 = summarize_sweep_candidate(candidates["resnet18_layer4_tokens"])
    multiscale = summarize_sweep_candidate(candidates["resnet18_layer3_layer4_tokens"])

    assert patch["parameters"]["parameter_count_source"] == CANDIDATE_PARAMETER_SOURCE
    assert patch["parameters"]["total_params"] == pytest.approx(197_000, abs=1_000)
    assert patch["parameters"]["image_encoder_params"] == pytest.approx(117_000, abs=1_000)
    assert patch["parameters"]["visual_context_encoder_params"] == pytest.approx(88_000, abs=1_000)
    assert layer4["parameters"]["total_params"] == pytest.approx(11_320_000, abs=10_000)
    assert layer4["parameters"]["image_encoder_params"] == pytest.approx(11_240_000, abs=10_000)
    assert layer4["parameters"]["visual_context_encoder_params"] == pytest.approx(11_210_000, abs=10_000)
    assert multiscale["parameters"]["total_params"] == pytest.approx(14_130_000, abs=10_000)
    assert multiscale["parameters"]["image_encoder_params"] == pytest.approx(14_050_000, abs=10_000)
    assert multiscale["parameters"]["visual_context_encoder_params"] == pytest.approx(14_020_000, abs=10_000)


def test_cli_help_and_sweep_csv_output(capsys):
    help_text = build_parser().format_help()
    assert "--config" in help_text
    assert "--sweep-manifest" in help_text
    assert "--startup-summary" in help_text

    code = model_summary_main(
        [
            "--sweep-manifest",
            str(SWEEP_MANIFEST),
            "--variant-id",
            "patch14_stage1_gps_query",
            "--format",
            "csv",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "variant_id" in output
    assert "patch14_stage1_gps_query" in output
    assert "197000" in output
