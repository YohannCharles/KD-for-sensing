import pytest

from kd_sensing.registries import ENCODERS, MODELS, REPRESENTATION_CORES, Registry, RegistryError, import_default_components


def test_registry_builds_components_and_reports_bad_configs() -> None:
    registry = Registry("tiny")

    @registry.register("example")
    class Example:
        def __init__(self, value: int) -> None:
            self.value = value

    assert registry.build({"type": "example", "value": 7}).value == 7
    with pytest.raises(RegistryError, match="Unknown component 'missing'"):
        registry.build({"type": "missing"})
    with pytest.raises(RegistryError, match="Missing required parameters: value"):
        registry.build({"type": "example"})


def test_registry_exposes_only_the_t2_baseline_components_needed_by_recipes() -> None:
    import_default_components()

    assert {"u_mask_beam_jepa", "modular_sequence"} <= set(MODELS.list())
    assert {
        "tinyvit_5m_scratch_rgb",
        "radar_cnn",
        "gps_mlp",
        "lidar_cnn",
        "resnet18_imagenet_rgb",
        "resnet18_spatial_tokens",
        "resnet34_spatial_tokens",
    } <= set(ENCODERS.list())
    assert {"amber_full_adaptive_mask_transformer", "rmbp_channel_attention_fusion"} <= set(
        REPRESENTATION_CORES.list()
    )
