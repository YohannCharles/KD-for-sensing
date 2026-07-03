import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kd_sensing.config import load_config
from kd_sensing.engine.model_output import ModelOutput, adapt_model_output
from kd_sensing.engine.optim import build_model
from kd_sensing.engine.prediction_objectives import PredictionTargets, compute_prediction_loss
from kd_sensing.models.amber_full import AmberFullAdaptiveMaskTransformerCore
from kd_sensing.models.architecture_summary import summarize_model_architecture
from kd_sensing.models.modular import ModularSequenceModel
from kd_sensing.registries import ENCODERS


ROOT = Path(__file__).resolve().parents[1]


@ENCODERS.register("amber_full_test_identity", force=True)
class AmberFullTestIdentityEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, token_count: int = 2, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.token_count = int(token_count)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        features = batch[..., : self.output_dim]
        return features.unsqueeze(2).expand(-1, -1, self.token_count, -1).contiguous()


def test_amber_full_forward_auxiliary_eval_and_adapt_model_output() -> None:
    model = _amber_full_test_model()
    batch = _synthetic_modalities()

    model.train()
    train_output = model(**batch, image_valid_mask=torch.tensor([[False, True], [True, True]]))
    adapted = adapt_model_output(train_output)

    assert isinstance(model.representation_core, AmberFullAdaptiveMaskTransformerCore)
    assert adapted.logits.shape == (2, 2, 6)
    assert train_output["amber_full_auxiliary"]["fusion_features"].shape == (2, 2, 8)
    assert train_output["amber_full_auxiliary"]["cma_logits"].shape == (2, 2, 4)
    assert train_output["amber_full_auxiliary"]["cma_modality_query_embeddings"].shape == (2, 4, 2, 8)
    assert train_output["token_features"].shape == (2, 4, 2, 2, 8)
    assert train_output["missing_modality_metadata"]["missing_counts"]["image"] == 1
    assert adapt_model_output(model(**_synthetic_modalities(seq_len=3))).logits.shape == (2, 3, 6)

    model.eval()
    eval_output = model(**batch)
    assert "amber_full_auxiliary" not in eval_output
    assert adapt_model_output(eval_output).logits.shape == (2, 2, 6)


def test_amber_full_attention_mask_blocks_missing_modalities() -> None:
    model = _amber_full_test_model()
    batch = _synthetic_modalities(seq_len=3)

    single = model(**batch, image_valid_mask=torch.tensor([[False, False, False], [True, True, True]]))
    multi = model(
        **batch,
        radar_valid_mask=torch.zeros(2, 3, dtype=torch.bool),
        gps_valid_mask=torch.tensor([[True, False, True], [False, False, True]]),
    )
    all_but_one = model(
        **batch,
        image_valid_mask=torch.zeros(2, 3, dtype=torch.bool),
        radar_valid_mask=torch.zeros(2, 3, dtype=torch.bool),
        gps_valid_mask=torch.ones(2, 3, dtype=torch.bool),
        lidar_valid_mask=torch.zeros(2, 3, dtype=torch.bool),
    )

    per_step = 9
    assert not hasattr(model.representation_core, "history_beam_token")
    assert single["amber_full_attention_key_padding_mask"][0, 1].item() is True
    assert single["amber_full_attention_key_padding_mask"][0, 2].item() is True
    assert single["amber_full_attention_key_padding_mask"][0, 0].item() is False
    assert multi["amber_full_attention_key_padding_mask"][0, per_step + 1 + 2].item() is True
    assert multi["amber_full_attention_key_padding_mask"][0, per_step + 1 + 4].item() is True
    assert all_but_one["amber_full_attention_key_padding_mask"][0, 1 + 4].item() is False
    assert all_but_one["amber_full_attention_key_padding_mask"][0, 1 + 0].item() is True
    assert all_but_one["amber_full_attention_key_padding_mask"][0, 1 + 2].item() is True
    assert all_but_one["amber_full_attention_key_padding_mask"][0, 1 + 6].item() is True


def test_amber_full_loss_weighting_and_missing_payload_failure() -> None:
    model = _amber_full_test_model()
    model.train()
    output = adapt_model_output(model(**_synthetic_modalities()))
    beam = output.logits.sum() * 0.0 + 1.5
    cfg = {"loss": {"type": "focal_loss", "auxiliary": {"amber_full": {"enabled": True, "l2_weight": 0.1, "cma_weight": 0.2}}}}

    bundle = compute_prediction_loss(
        output,
        PredictionTargets(labels=torch.zeros(2, 2, dtype=torch.long)),
        cfg,
        reference=output.logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )

    assert bundle.total.item() > beam.item()
    assert bundle.diagnostics["loss/amber_full_total"] > 0.0

    ordinary = compute_prediction_loss(
        ModelOutput(logits=output.logits, input_features=None, output_features=None, diagnostics={}),
        PredictionTargets(labels=torch.zeros(2, 2, dtype=torch.long)),
        {"loss": {"type": "focal_loss"}},
        reference=output.logits,
        beam_total_loss=beam,
        beam_task_loss=beam,
    )
    assert ordinary.total.item() == pytest.approx(beam.item())

    with torch.no_grad():
        eval_bundle = compute_prediction_loss(
            ModelOutput(logits=output.logits, input_features=None, output_features=None, diagnostics={}),
            PredictionTargets(labels=torch.zeros(2, 2, dtype=torch.long)),
            cfg,
            reference=output.logits,
            beam_total_loss=beam,
            beam_task_loss=beam,
        )
    assert eval_bundle.total.item() == pytest.approx(beam.item())

    with pytest.raises(ValueError, match="amber_full_auxiliary"):
        compute_prediction_loss(
            ModelOutput(logits=output.logits, input_features=None, output_features=None, diagnostics={}),
            PredictionTargets(labels=torch.zeros(2, 2, dtype=torch.long)),
            cfg,
            reference=output.logits,
            beam_total_loss=beam,
            beam_task_loss=beam,
        )


def test_amber_full_config_metadata_and_architecture_summary() -> None:
    cfg = load_config(ROOT / "configs/fusion/amber_full_architecture.yaml")
    primary = cfg["model"]["primary"]
    build_primary = copy.deepcopy(primary)
    for modality in ("image", "radar", "lidar"):
        build_primary["encoders"][modality]["pretrained"] = False
        build_primary["encoders"][modality]["weights"] = None
    configured_model = build_model(build_primary)
    model = _amber_full_test_model()
    metadata = model.training_strategy_metadata()
    summary = summarize_model_architecture(model, cfg=primary)

    assert primary["representation_core"]["type"] == "amber_full_adaptive_mask_transformer"
    assert primary["representation_core"]["max_spatial_tokens"] == 4
    assert cfg["data"]["dataset"]["seq_len"] == 2
    assert cfg["model"]["seq_length"] == 2
    for modality in ("image", "radar", "lidar"):
        assert primary["encoders"][modality]["type"] == "resnet18_spatial_tokens"
        assert primary["encoders"][modality]["pretrained"] is True
        assert primary["encoders"][modality]["weights"] == "DEFAULT"
    assert isinstance(configured_model.representation_core, AmberFullAdaptiveMaskTransformerCore)
    assert primary["paper_metadata"]["reproduction_scope"] == "amber_full_local"
    assert cfg["output"]["dir"].startswith("outputs/analysis/local_baselines/amber_full_architecture")
    assert metadata["reproduction_scope"] == "amber_full_local"
    assert metadata["representation_core"]["component_role"] == "representation_core"
    assert metadata["representation_core"]["history_beam_usage"] == "disabled"
    assert metadata["representation_core"]["cma_type"] == "class_query_cross_attention"
    assert metadata["consumes_missing_modality_metadata"] is True
    assert summary["components"]["representation_core"]["semantic_role"] == "representation_core"
    assert summary["components"]["representation_core"]["total_params"] > 0


def _amber_full_test_model() -> ModularSequenceModel:
    modalities = ["image", "radar", "gps", "lidar"]
    return ModularSequenceModel(
        modalities=modalities,
        encoders={modality: {"type": "amber_full_test_identity", "output_dim": 8} for modality in modalities},
        projectors={modality: {"type": "identity"} for modality in modalities},
        representation_core={
            "type": "amber_full_adaptive_mask_transformer",
            "d_model": 8,
            "num_heads": 2,
            "modality_layers": 1,
            "fusion_layers": 1,
            "dropout": 0.0,
            "max_seq_len": 3,
            "num_cma_queries": 2,
            "cma_dim": 8,
            "cma_temperature": 0.5,
            "auxiliary_loss_weights": {"l2": 0.1, "cma": 0.2},
        },
        heads={"beam": {"type": "beam_head"}},
        feature_size=8,
        d_model=8,
        num_classes=6,
        num_pred=1,
        paper_metadata={"reproduction_scope": "amber_full_local"},
    )


def _synthetic_modalities(seq_len: int = 2) -> dict[str, torch.Tensor]:
    return {
        "image_batch": torch.randn(2, seq_len, 8),
        "radar_batch": torch.randn(2, seq_len, 8),
        "gps_batch": torch.randn(2, seq_len, 8),
        "lidar_batch": torch.randn(2, seq_len, 8),
    }
