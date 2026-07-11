import pytest
import torch
import torch.nn as nn

from kd_sensing.models.modular import FeatureConsistencyGateCore, ModularSequenceModel, NextBeamQueryTransformerCore
from kd_sensing.modalities import MODALITY_ORDER
from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.registries import ENCODERS, HEADS, PROJECTORS, REPRESENTATION_CORES


@ENCODERS.register("next_query_test_identity", force=True)
class NextQueryTestIdentityEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, **_: object):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"next_query_test_identity expects [B, T, D], got {tuple(batch.shape)}.")
        if int(batch.shape[-1]) != self.output_dim:
            raise ValueError(f"expected D={self.output_dim}, got {tuple(batch.shape)}.")
        return batch

    def training_strategy_metadata(self) -> dict[str, object]:
        return {
            "encoder_strategy": "identity",
            "uses_external_checkpoint": False,
            "freeze_policy": "none",
        }


@ENCODERS.register("next_query_test_context_identity", force=True)
class NextQueryTestContextIdentityEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 8,
        required_context_modalities: list[str] | tuple[str, ...] | None = None,
        context_feature_source: str = "projected",
        **_: object,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.required_context_modalities = tuple(required_context_modalities or ())
        self.context_feature_source = str(context_feature_source)
        self.context_feature_kwargs = {
            modality: f"{modality}_condition_features" for modality in self.required_context_modalities
        }

    def forward(self, batch: torch.Tensor, **context_features: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"next_query_test_context_identity expects [B, T, D], got {tuple(batch.shape)}.")
        output = batch
        for modality in self.required_context_modalities:
            key = f"{modality}_condition_features"
            if key not in context_features:
                raise ValueError(f"missing condition feature {key}.")
            condition = context_features[key]
            if int(condition.shape[-1]) == int(output.shape[-1]):
                addition = condition
            else:
                addition = condition.mean(dim=-1, keepdim=True).expand_as(output)
            output = output + addition
        return output


@ENCODERS.register("next_query_test_scaled_identity", force=True)
class NextQueryTestScaledIdentityEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, scale: float = 1.0, **_: object):
        super().__init__()
        self.output_dim = int(output_dim)
        self.scale = float(scale)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"next_query_test_scaled_identity expects [B, T, D], got {tuple(batch.shape)}.")
        return batch[..., : self.output_dim] * self.scale


@PROJECTORS.register("next_query_test_scale_projector", force=True)
class NextQueryTestScaleProjector(nn.Module):
    def __init__(self, input_dim: int, d_model: int, scale: float = 1.0, **_: object):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(d_model)
        self.scale = float(scale)
        if self.input_dim != self.output_dim:
            raise ValueError("next_query_test_scale_projector requires input_dim == d_model.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"next_query_test_scale_projector expects [B, T, D], got {tuple(features.shape)}.")
        return features * self.scale


@ENCODERS.register("next_query_test_observability_identity", force=True)
class NextQueryTestObservabilityIdentityEncoder(NextQueryTestIdentityEncoder):
    supports_observability_metadata = True

    def forward(
        self,
        batch: torch.Tensor,
        *,
        image_valid_mask: torch.Tensor | None = None,
        image_observability_score: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del image_valid_mask, image_observability_score
        return super().forward(batch)

    def training_strategy_metadata(self) -> dict[str, object]:
        metadata = super().training_strategy_metadata()
        metadata["consumes_reliability_metadata"] = True
        return metadata


@REPRESENTATION_CORES.register("next_query_test_metadata_core", force=True)
class NextQueryTestMetadataCore(nn.Module):
    def __init__(self, d_model: int, output_dim: int | None = None, **_: object):
        super().__init__()
        self.d_model = int(d_model)
        self.output_dim = int(output_dim or d_model)
        self.projection = nn.Identity() if self.output_dim == self.d_model else nn.Linear(self.d_model, self.output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            features = features.mean(dim=1)
        if features.ndim != 3:
            raise ValueError(f"next_query_test_metadata_core expects [B, T, D], got {tuple(features.shape)}.")
        return self.projection(features)

    def training_strategy_metadata(self) -> dict[str, object]:
        return {
            "core_strategy": "metadata_core",
            "consumes_reliability_metadata": False,
        }


@HEADS.register("next_query_test_metadata_head", force=True)
class NextQueryTestMetadataHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, **_: object):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.classifier = nn.Linear(self.input_dim, self.num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"next_query_test_metadata_head expects [B, T, D], got {tuple(features.shape)}.")
        return self.classifier(features)

    def training_strategy_metadata(self) -> dict[str, object]:
        return {
            "head_strategy": "metadata_head",
            "consumes_reliability_metadata": False,
        }


def test_next_beam_query_transformer_core_forward_shape_and_registry_build():
    core = REPRESENTATION_CORES.build(
        {
            "type": "next_beam_query_transformer",
            "d_model": 8,
            "modality_count": 2,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "max_seq_len": 4,
            "output_dim": 6,
        }
    )

    output = core(torch.randn(3, 2, 4, 8))

    assert isinstance(core, NextBeamQueryTransformerCore)
    assert output.shape == (3, 1, 6)


def test_next_beam_query_transformer_core_rejects_bad_inputs():
    core = NextBeamQueryTransformerCore(
        d_model=8,
        modality_count=2,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        max_seq_len=4,
    )

    with pytest.raises(ValueError, match=r"requires multimodal \[B, K, T, D\] input"):
        core(torch.randn(3, 4, 8))
    with pytest.raises(ValueError, match=r"expected K=2, got K=3"):
        core(torch.randn(3, 3, 4, 8))
    with pytest.raises(ValueError, match=r"expected D=8, got D=7"):
        core(torch.randn(3, 2, 4, 7))
    with pytest.raises(ValueError, match=r"T=5 exceeds max_seq_len=4"):
        core(torch.randn(3, 2, 5, 8))


def test_modular_sequence_next_query_transformer_beam_head_and_diagnostics():
    model = _modular_model(
        ["image", "gps"],
        {
            "type": "next_beam_query_transformer",
            "d_model": 8,
            "output_dim": 10,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "max_seq_len": 3,
        },
        num_classes=5,
    )

    output = model(image_batch=torch.randn(2, 3, 8), gps_batch=torch.randn(2, 3, 8))

    assert output["logits"].shape == (2, 1, 5)
    assert output["input_features"].shape == (2, 3, 16)
    assert output["output_features"].shape == (2, 1, 10)
    assert output["modalities"] == ("image", "gps")
    assert set(output["modality_features"]) == {"image", "gps"}
    assert set(output["encoder_features"]) == {"image", "gps"}


@pytest.mark.parametrize(
    ("core_cfg", "seq_len", "expected_time"),
    [
        ({"type": "early_concat_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1}, 3, 3),
        ({"type": "snapshot_frame", "d_model": 8, "output_dim": 8}, 1, 1),
        ({"type": "token_transformer", "d_model": 8, "num_heads": 2, "num_layers": 1}, 3, 3),
    ],
)
def test_existing_modular_sequence_cores_keep_forward_shapes(core_cfg: dict, seq_len: int, expected_time: int):
    model = _modular_model(["image", "gps"], core_cfg, num_classes=7)

    output = model(image_batch=torch.randn(2, seq_len, 8), gps_batch=torch.randn(2, seq_len, 8))

    assert output["logits"].shape == (2, expected_time, 7)
    assert output["output_features"].shape[:2] == (2, expected_time)


def test_feature_consistency_gate_core_uses_past_latents_and_reports_diagnostics():
    core = REPRESENTATION_CORES.build(
        {
            "type": "feature_consistency_gate",
            "d_model": 8,
            "modality_count": 2,
            "output_dim": 6,
            "history_window": 2,
            "dropout": 0.0,
        }
    )
    features = torch.randn(2, 2, 4, 8)

    output = core(features)

    assert isinstance(core, FeatureConsistencyGateCore)
    assert output.shape == (2, 4, 6)
    diagnostics = core.last_feature_consistency_diagnostics
    assert diagnostics is not None
    assert diagnostics["branch_availability"]["current"] is True
    assert diagnostics["branch_availability"]["temporal_predicted"] is True
    assert diagnostics["branch_availability"]["gps_residual"] is True
    assert diagnostics["history_source_range"][0] is None
    assert diagnostics["history_source_range"][1] == [0, 0]
    assert diagnostics["history_source_range"][3] == [1, 2]
    assert diagnostics["condition_id_consumed"] is False
    assert {"c_idx", "d_idx", "predictive_condition_id"} <= set(diagnostics["blocked_condition_fields"])


def test_modular_sequence_feature_consistency_gate_outputs_runtime_diagnostics():
    model = _modular_model(
        ["image", "gps"],
        {
            "type": "feature_consistency_gate",
            "d_model": 8,
            "output_dim": 8,
            "history_window": 2,
            "dropout": 0.0,
        },
        num_classes=5,
    )

    output = model(
        image_batch=torch.randn(2, 4, 8),
        gps_batch=torch.randn(2, 4, 8),
    )

    assert output["logits"].shape == (2, 4, 5)
    assert output["feature_consistency_diagnostics"]["condition_id_consumed"] is False
    metadata = model.training_strategy_metadata()
    assert metadata["representation_core_type"] == "feature_consistency_gate"
    assert "predictive_condition_id" in metadata["representation_core"]["forbidden_condition_fields"]


def test_modular_sequence_dependency_aware_projected_gps_context_and_errors():
    model = _conditioned_modular_model(
        image_dependencies=("gps",),
        gps_dependencies=(),
    )

    output = model(image_batch=torch.zeros(2, 3, 8), gps_batch=torch.ones(2, 3, 8))

    assert output["logits"].shape == (2, 3, 5)
    assert set(output["encoder_features"]) == {"image", "gps"}
    assert set(output["modality_features"]) == {"image", "gps"}
    torch.testing.assert_close(output["encoder_features"]["image"], torch.ones(2, 3, 8))

    with pytest.raises(ValueError, match="requires condition modalities .*gps"):
        ModularSequenceModel(
            modalities=["image"],
            encoders={
                "image": {
                    "type": "next_query_test_context_identity",
                    "output_dim": 8,
                    "required_context_modalities": ["gps"],
                }
            },
            projectors={"image": {"type": "identity", "input_dim": 8, "d_model": 8}},
            representation_core={"type": "single_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
            feature_size=8,
            d_model=8,
            num_classes=5,
            num_pred=1,
        )
    with pytest.raises(ValueError, match="Condition feature batch/time dimensions"):
        model(image_batch=torch.zeros(2, 3, 8), gps_batch=torch.ones(2, 2, 8))

    cycle_model = _conditioned_modular_model(
        image_dependencies=("gps",),
        gps_dependencies=("image",),
    )
    with pytest.raises(ValueError, match="circular dependencies"):
        cycle_model(image_batch=torch.zeros(2, 3, 8), gps_batch=torch.ones(2, 3, 8))


def test_modular_sequence_context_feature_sources_encoded_and_raw_are_distinct():
    encoded_model = ModularSequenceModel(
        modalities=["image", "gps"],
        encoders={
            "image": {
                "type": "next_query_test_context_identity",
                "output_dim": 8,
                "required_context_modalities": ["gps"],
                "context_feature_source": "encoded",
            },
            "gps": {"type": "next_query_test_scaled_identity", "output_dim": 8, "scale": 2.0},
        },
        projectors={
            "image": {"type": "identity", "input_dim": 8, "d_model": 8},
            "gps": {"type": "next_query_test_scale_projector", "input_dim": 8, "d_model": 8, "scale": 5.0},
        },
        representation_core={"type": "early_concat_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
    )

    encoded_output = encoded_model(image_batch=torch.zeros(2, 3, 8), gps_batch=torch.ones(2, 3, 8))

    torch.testing.assert_close(encoded_output["encoder_features"]["image"], torch.full((2, 3, 8), 2.0))
    torch.testing.assert_close(encoded_output["modality_features"]["gps"], torch.full((2, 3, 8), 10.0))

    raw_model = ModularSequenceModel(
        modalities=["image", "gps"],
        encoders={
            "image": {
                "type": "next_query_test_context_identity",
                "output_dim": 8,
                "required_context_modalities": ["gps"],
                "context_feature_source": "raw",
            },
            "gps": {"type": "next_query_test_scaled_identity", "output_dim": 8, "scale": 2.0},
        },
        projectors={
            "image": {"type": "identity", "input_dim": 8, "d_model": 8},
            "gps": {"type": "next_query_test_scale_projector", "input_dim": 8, "d_model": 8, "scale": 5.0},
        },
        representation_core={"type": "early_concat_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
    )

    raw_output = raw_model(image_batch=torch.zeros(2, 3, 8), gps_batch=torch.ones(2, 3, 8))

    torch.testing.assert_close(raw_output["encoder_features"]["image"], torch.ones(2, 3, 8))
    torch.testing.assert_close(raw_output["modality_features"]["gps"], torch.full((2, 3, 8), 10.0))
    metadata = raw_model.training_strategy_metadata()
    assert metadata["conditioned_encoders"]["image"]["context_feature_source"] == "raw"


def test_modular_sequence_rejects_self_dependency_and_keeps_plain_encoders_single_input():
    with pytest.raises(ValueError, match="cannot depend on its own condition feature"):
        ModularSequenceModel(
            modalities=["gps"],
            encoders={
                "gps": {
                    "type": "next_query_test_context_identity",
                    "output_dim": 8,
                    "required_context_modalities": ["gps"],
                }
            },
            projectors={"gps": {"type": "identity", "input_dim": 8, "d_model": 8}},
            representation_core={"type": "single_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
            feature_size=8,
            d_model=8,
            num_classes=5,
            num_pred=1,
        )

    modalities = list(MODALITY_ORDER)
    model = ModularSequenceModel(
        modalities=modalities,
        encoders={modality: {"type": "next_query_test_identity", "output_dim": 8} for modality in modalities},
        projectors={modality: {"type": "identity", "input_dim": 8, "d_model": 8} for modality in modalities},
        representation_core={"type": "early_concat_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
    )
    batch = {f"{modality}_batch": torch.randn(2, 3, 8) for modality in modalities}

    output = model(**batch)

    assert output["logits"].shape == (2, 3, 5)
    assert set(output["encoder_features"]) == set(modalities)
    assert model.training_strategy_metadata()["conditioned_encoders"] == {}


def test_modular_sequence_training_metadata_aggregates_component_metadata():
    model = ModularSequenceModel(
        modalities=["gps"],
        encoders={"gps": {"type": "next_query_test_identity", "output_dim": 8}},
        projectors={"gps": {"type": "identity", "input_dim": 8, "d_model": 8}},
        representation_core={"type": "next_query_test_metadata_core", "d_model": 8, "output_dim": 8},
        heads={"beam": {"type": "next_query_test_metadata_head", "input_dim": 8, "num_classes": 5}},
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
    )

    output = model(gps_batch=torch.randn(2, 3, 8))
    adapted = adapt_model_output(output)
    metadata = model.training_strategy_metadata()

    assert output["logits"].shape == (2, 3, 5)
    assert adapted.logits.shape == (2, 3, 5)
    assert metadata["architecture_category"] == "component_baseline"
    assert metadata["encoders"]["gps"]["registry_type"] == "next_query_test_identity"
    assert metadata["encoders"]["gps"]["encoder_strategy"] == "identity"
    assert metadata["projectors"]["gps"]["registry_type"] == "identity"
    assert metadata["representation_core_type"] == "next_query_test_metadata_core"
    assert metadata["representation_core"]["core_strategy"] == "metadata_core"
    assert metadata["heads"]["beam"]["registry_type"] == "next_query_test_metadata_head"
    assert metadata["heads"]["beam"]["head_strategy"] == "metadata_head"
    assert metadata["consumes_reliability_metadata"] is False
    assert metadata["reliability_metadata"]["consumers"] == []


def test_modular_sequence_training_metadata_marks_reliability_consuming_encoders():
    model = ModularSequenceModel(
        modalities=["image"],
        encoders={"image": {"type": "next_query_test_observability_identity", "output_dim": 8}},
        projectors={"image": {"type": "identity", "input_dim": 8, "d_model": 8}},
        representation_core={"type": "next_query_test_metadata_core", "d_model": 8, "output_dim": 8},
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
    )

    output = model(
        image_batch=torch.randn(2, 3, 8),
        image_valid_mask=torch.ones(2, 3, dtype=torch.bool),
        image_observability_score=torch.ones(2, 3),
    )
    metadata = model.training_strategy_metadata()

    assert output["logits"].shape == (2, 3, 5)
    assert metadata["encoders"]["image"]["consumes_reliability_metadata"] is True
    assert metadata["consumes_reliability_metadata"] is True
    assert metadata["reliability_metadata_consumers"] == ["encoders.image"]
    assert "image_observability_score" in metadata["reliability_metadata"]["fields"]


def _modular_model(modalities: list[str], core_cfg: dict, *, num_classes: int) -> ModularSequenceModel:
    encoders = {modality: {"type": "next_query_test_identity", "output_dim": 8} for modality in modalities}
    projectors = {modality: {"type": "identity", "input_dim": 8, "d_model": 8} for modality in modalities}
    return ModularSequenceModel(
        modalities=modalities,
        encoders=encoders,
        projectors=projectors,
        representation_core=core_cfg,
        feature_size=8,
        d_model=8,
        num_classes=num_classes,
        num_pred=1,
    )


def _conditioned_modular_model(
    *,
    image_dependencies: tuple[str, ...],
    gps_dependencies: tuple[str, ...],
) -> ModularSequenceModel:
    return ModularSequenceModel(
        modalities=["image", "gps"],
        encoders={
            "image": {
                "type": "next_query_test_context_identity",
                "output_dim": 8,
                "required_context_modalities": list(image_dependencies),
            },
            "gps": {
                "type": "next_query_test_context_identity",
                "output_dim": 8,
                "required_context_modalities": list(gps_dependencies),
            },
        },
        projectors={
            "image": {"type": "identity", "input_dim": 8, "d_model": 8},
            "gps": {"type": "identity", "input_dim": 8, "d_model": 8},
        },
        representation_core={"type": "early_concat_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
    )
