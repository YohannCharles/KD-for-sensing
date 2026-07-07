import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.trainer import train
from kd_sensing.engine.validator import validate
from kd_sensing.losses.jepa import jepa_latent_prediction_loss
from kd_sensing.models.jepa import (
    GPSQueryPool,
    JepaContextImageEncoder,
    JepaMaskSampler,
    build_visual_token_encoder,
)
from kd_sensing.models.jepa_downstream import (
    HybridResidualQueryPool,
    IdentityJepaAdapter,
    LearnedQueryPool,
    MeanPatchPooler,
    PredictiveGPSQueryPool,
    SelfAttentionPool,
    build_jepa_downstream_adapter,
    build_jepa_downstream_pooler,
)
from kd_sensing.registries import ENCODERS, MODELS, RegistryError, import_default_components

@ENCODERS.register("gps_query_readout_test_identity", force=True)
class GpsQueryReadoutTestIdentityEncoder(nn.Module):
    def __init__(self, output_dim: int = 8, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim not in {3, 4}:
            raise ValueError(f"gps_query_readout_test_identity expects [B,T,D] or [B,T,K,D], got {tuple(batch.shape)}.")
        return batch[..., : self.output_dim]


def _model_cfg() -> dict:
    return {
        "type": "gps_conditioned_jepa",
        "modalities": ["image", "gps"],
        "image_profile": "rgb_imagenet",
        "image_channels": 3,
        "gps_input_size": 3,
        "latent_dim": 16,
        "num_classes": 1,
        "ema_decay": 0.5,
        "visual_encoder": {
            "image_channels": 3,
            "latent_dim": 16,
            "patch_size": 8,
            "depth": 0,
            "max_tokens": 16,
        },
        "conditioning": {"type": "film", "hidden_dim": 32},
        "predictor": {"hidden_dim": 32, "max_tokens": 16},
        "mask_sampler": {"mode": "random", "context_ratio": 0.5, "target_ratio": 0.25, "seed": 11},
    }


def _tiny_train_cfg(tmp_path: Path) -> dict:
    primary_cfg = _model_cfg()
    primary_cfg["visual_encoder"] = {**primary_cfg["visual_encoder"], "patch_size": 56}
    return {
        "experiment": {"name": "jepa_smoke", "task": "fusion", "objective": "gps_conditioned_jepa", "seed": 11, "device": "cpu"},
        "data": {
            "dataset": {
                "type": "synthetic",
                "length": 2,
                "seq_len": 2,
                "num_pred": 1,
                "image_size": [32, 32],
                "image_channels": 3,
                "use_gps": True,
                "gps_input_size": 3,
                "image_profile": "rgb_imagenet",
                "gps_feature_mode": "relative_polar",
            },
            "dataloader": {"train_batch_size": 1, "test_batch_size": 1, "num_workers": 0, "pin_memory": False},
        },
        "model": {
            "modalities": ["image", "gps"],
            "num_classes": 1,
            "num_pred": 1,
            "seq_length": 2,
            "downsample_ratio": 1,
            "primary": primary_cfg,
        },
        "loss": {"type": "focal_loss", "jepa": {"type": "mse", "weight": 1.0}},
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "use_early_stopping": False,
            "early_stopping_metric": "val_jepa_loss",
            "early_stopping_mode": "min",
            "transfer": {"non_blocking": False},
            "amp": {"enabled": False},
        },
        "scheduler": {"type": "none"},
        "evaluation": {"k_values": [1], "dba_delta": 5},
        "output": {
            "dir": str(tmp_path),
            "run_name": "jepa_smoke",
            "group_by_scene": False,
            "overwrite": True,
            "tensorboard": {"enabled": False},
            "progress": {"enabled": False},
        },
        "checkpoint": {"registry": {"enabled": False}},
    }


def test_jepa_registry_forward_shape_gps_validation_detach_and_ema_update():
    import_default_components()
    model = MODELS.build(_model_cfg())
    image = torch.randn(2, 2, 3, 32, 32)
    gps = torch.randn(2, 2, 3)

    output = model(image_batch=image, gps_batch=gps, jepa_epoch=1, jepa_step=2)

    assert output["predicted_target_latent"].shape == output["target_latent"].shape
    assert output["predicted_target_latent"].shape == (2, 2, 4, 16)
    assert output["target_latent"].requires_grad is False
    assert output["context_mask"].shape == (2, 2, 16)
    assert not torch.any(output["context_mask"] & output["target_mask"])
    with pytest.raises(ValueError, match="GPS-Rel-Polar"):
        model(image_batch=image)
    with pytest.raises(ValueError, match="expected GPS feature dim 3, got 4"):
        model(image_batch=image, gps_batch=torch.randn(2, 2, 4))

    loss = output["predicted_target_latent"].sum()
    loss.backward()
    assert all(param.grad is None for param in model.target_encoder.parameters())
    before = next(model.target_encoder.parameters()).detach().clone()
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    model.update_target_encoder()
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)


def test_visual_token_encoder_registry_default_metadata_and_token_budget():
    import_default_components()
    encoder = build_visual_token_encoder(
        {
            "type": "patch_vit",
            "image_channels": 3,
            "latent_dim": 8,
            "patch_size": 16,
            "depth": 0,
            "max_tokens": 4,
        },
        image_channels=3,
        latent_dim=8,
        image_profile="rgb_imagenet",
    )

    tokens, grid = encoder(torch.randn(1, 1, 3, 32, 32))

    metadata = encoder.visual_token_metadata()
    assert tokens.shape == (1, 1, 4, 8)
    assert grid == (2, 2)
    assert metadata["variant_id"] == "patch16"
    assert metadata["token_source"] == "patch_vit"
    assert metadata["token_grid"] == [2, 2]
    assert metadata["token_count"] == 4
    assert metadata["checkpoint_policy"] == "exact_reuse"

    with pytest.raises(RegistryError, match="unknown_visual.*jepa_visual_token_encoders"):
        build_visual_token_encoder({"type": "unknown_visual"}, image_channels=3, latent_dim=8)
    with pytest.raises(ValueError, match="token budget exceeded.*token_count=16.*max_tokens=4"):
        build_visual_token_encoder(
            {
                "type": "patch_vit",
                "image_channels": 3,
                "latent_dim": 8,
                "patch_size": 8,
                "depth": 0,
                "max_tokens": 4,
            },
            image_channels=3,
            latent_dim=8,
            image_profile="rgb_imagenet",
        )(torch.randn(1, 1, 3, 32, 32))


@pytest.mark.parametrize(
    ("visual_cfg", "expected_grid", "expected_tokens", "policy"),
    [
        ({"type": "patch_vit", "patch_size": 14, "max_tokens": 16}, (2, 2), 4, "fresh_stage1_required"),
        ({"type": "patch_vit", "patch_size": 8, "max_tokens": 16}, (4, 4), 16, "fresh_stage1_required"),
        ({"type": "overlap_patch", "kernel_size": 16, "stride": 8, "max_tokens": 16}, (3, 3), 9, "fresh_stage1_required"),
        (
            {"type": "conv_stem", "stem_channels": [8, 8], "stem_strides": [2, 2], "max_tokens": 64},
            (8, 8),
            64,
            "fresh_stage1_required",
        ),
        ({"type": "local_token_mixing", "patch_size": 16, "max_tokens": 4}, (2, 2), 4, "partial_reuse"),
        ({"type": "cvt", "patch_size": 16, "max_tokens": 4}, (2, 2), 4, "partial_reuse"),
    ],
)
def test_visual_tokenizer_variants_shape_metadata_and_stage1_forward(
    visual_cfg: dict,
    expected_grid: tuple[int, int],
    expected_tokens: int,
    policy: str,
):
    cfg = {
        "image_channels": 3,
        "latent_dim": 8,
        "depth": 0,
        "num_heads": 2,
        **visual_cfg,
    }
    encoder = build_visual_token_encoder(cfg, image_channels=3, latent_dim=8, image_profile="rgb_imagenet")

    tokens, grid = encoder(torch.randn(2, 2, 3, 32, 32))

    metadata = encoder.visual_token_metadata()
    assert tokens.shape == (2, 2, expected_tokens, 8)
    assert grid == expected_grid
    assert metadata["token_grid"] == [expected_grid[0], expected_grid[1]]
    assert metadata["token_count"] == expected_tokens
    assert metadata["checkpoint_policy"] == policy


def test_cnn_and_multiscale_visual_token_sources_shape_metadata():
    pytest.importorskip("torchvision.models")
    image = torch.randn(1, 1, 3, 64, 64)
    single = build_visual_token_encoder(
        {
            "type": "cnn_feature_map",
            "image_channels": 3,
            "latent_dim": 8,
            "backbone": "resnet18",
            "stage": "layer3",
            "pretrained": False,
            "freeze_backbone": True,
            "max_tokens": 64,
        },
        image_channels=3,
        latent_dim=8,
        image_profile="rgb_imagenet",
    )
    multi = build_visual_token_encoder(
        {
            "type": "multi_scale_cnn",
            "image_channels": 3,
            "latent_dim": 8,
            "backbone": "resnet18",
            "pretrained": False,
            "freeze_backbone": True,
            "max_tokens": 80,
        },
        image_channels=3,
        latent_dim=8,
        image_profile="rgb_imagenet",
    )
    single.eval()
    multi.eval()

    single_tokens, single_grid = single(image)
    multi_tokens, multi_grid = multi(image)

    assert single_tokens.shape == (1, 1, 16, 8)
    assert single_grid == (4, 4)
    assert single.visual_token_metadata()["backbone"] == "resnet18"
    assert single.visual_token_metadata()["stage"] == "layer3"
    assert single.visual_token_metadata()["checkpoint_policy"] == "supervised_only_anchor"
    assert multi_tokens.shape == (1, 1, 20, 8)
    assert multi_grid == (4, 4)
    assert multi.visual_token_metadata()["scale_token_counts"] == {"layer3": 16, "layer4": 4}


def test_jepa_concat_conditioner_forward_shape():
    cfg = _model_cfg()
    cfg["conditioning"] = {"type": "concat_mlp", "hidden_dim": 32}
    model = MODELS.build(cfg)
    output = model(image_batch=torch.randn(1, 1, 3, 32, 32), gps_batch=torch.randn(1, 1, 3))
    assert output["predicted_target_latent"].shape == output["target_latent"].shape


def test_gps_query_pool_shape_attention_map_and_dimension_validation():
    pool = GPSQueryPool(latent_dim=8, condition_dim=5, k_queries=3, num_heads=2, dropout=0.0)
    patch_tokens = torch.randn(2, 4, 6, 8)
    condition = torch.randn(2, 4, 5)

    pooled, attention = pool(patch_tokens, condition, return_attention=True)

    assert pooled.shape == (2, 4, 8)
    assert attention.shape == (2, 4, 3, 6)
    assert attention.requires_grad is False
    assert pool.last_diagnostics["attention_head_aggregation"] == "averaged"
    assert pool.last_diagnostics["attention_return_shape"] == [2, 4, 3, 6]
    assert pool.last_diagnostics["query_count"] == 3
    assert pool.last_diagnostics["token_count"] == 6
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 4, 3), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="patch tokens shape .*condition feature shape"):
        pool(patch_tokens, torch.randn(2, 3, 5))
    with pytest.raises(ValueError, match="expected condition feature dim 5"):
        pool(patch_tokens, torch.randn(2, 4, 4))


def test_gps_query_pool_per_head_attention_keeps_return_shape() -> None:
    pool = GPSQueryPool(
        latent_dim=8,
        condition_dim=5,
        k_queries=3,
        num_heads=2,
        dropout=0.0,
        per_head_attention=True,
    )
    patch_tokens = torch.randn(2, 4, 6, 8)
    condition = torch.randn(2, 4, 5)

    pooled, attention = pool(patch_tokens, condition, return_attention=True)

    assert pooled.shape == (2, 4, 8)
    assert attention.shape == (2, 4, 3, 6)
    assert pool.last_attention_heads is not None
    assert pool.last_attention_heads.shape == (2, 4, 2, 3, 6)
    assert pool.last_diagnostics["attention_head_aggregation"] == "per_head"
    assert pool.last_diagnostics["attention_per_head_shape"] == [2, 4, 2, 3, 6]
    assert pool.last_diagnostics["head_aggregation_method"] == "mean_heads_for_last_attention_map"


def test_gps_query_pool_tokens_output_attention_and_metadata():
    frame_pool = GPSQueryPool(latent_dim=8, condition_dim=5, k_queries=3, num_heads=2, dropout=0.0)
    token_pool = GPSQueryPool(latent_dim=8, condition_dim=5, k_queries=3, num_heads=2, dropout=0.0, output_mode="tokens")
    patch_tokens = torch.randn(2, 4, 6, 8)
    condition = torch.randn(2, 4, 5)

    frame = frame_pool(patch_tokens, condition)
    tokens, attention = token_pool(patch_tokens, condition, return_attention=True)

    assert frame.shape == (2, 4, 8)
    assert tokens.shape == (2, 4, 3, 8)
    assert attention.shape == (2, 4, 3, 6)
    assert attention.requires_grad is False
    assert token_pool.last_diagnostics["k_queries"] == 3
    assert token_pool.last_diagnostics["k_tokens"] == 3
    assert token_pool.last_diagnostics["effective_patch_count"] > 0
    assert "query_diversity" in token_pool.last_diagnostics
    assert "attended_latent_similarity" in token_pool.last_diagnostics


def test_learned_query_pool_tokens_do_not_require_gps_condition():
    pool = build_jepa_downstream_pooler(
        {
            "type": "learned_query_attention",
            "latent_dim": 8,
            "k_queries": 2,
            "num_heads": 2,
            "dropout": 0.0,
            "output_mode": "tokens",
        }
    )
    patch_tokens = torch.randn(2, 4, 6, 8)

    tokens, attention = pool(patch_tokens, return_attention=True)

    assert isinstance(pool, LearnedQueryPool)
    assert pool.required_context_modalities == ()
    assert pool.context_feature_source == "none"
    assert tokens.shape == (2, 4, 2, 8)
    assert attention.shape == (2, 4, 2, 6)
    assert pool.last_diagnostics["condition_feature_source"] == "none"


def test_self_attention_pool_tokens_do_not_require_gps_condition():
    pool = build_jepa_downstream_pooler(
        {
            "type": "self_attention",
            "latent_dim": 8,
            "k_tokens": 2,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "output_mode": "tokens",
        }
    )
    patch_tokens = torch.randn(2, 4, 6, 8)

    tokens = pool(patch_tokens)

    assert isinstance(pool, SelfAttentionPool)
    assert pool.required_context_modalities == ()
    assert pool.context_feature_source == "none"
    assert tokens.shape == (2, 4, 2, 8)
    assert pool.last_diagnostics["condition_feature_source"] == "none"


def test_jepa_downstream_poolers_build_and_identity_adapter_is_noop():
    import_default_components()
    tokens = torch.randn(2, 3, 5, 8)

    mean_pooler = build_jepa_downstream_pooler({"type": "mean", "latent_dim": 8})
    assert isinstance(mean_pooler, MeanPatchPooler)
    assert mean_pooler(tokens).shape == (2, 3, 8)
    torch.testing.assert_close(mean_pooler(tokens), tokens.mean(dim=2))

    gps_pooler = build_jepa_downstream_pooler(
        {
            "type": "gps_query_attention",
            "latent_dim": 8,
            "condition_dim": 8,
            "k_queries": 2,
            "num_heads": 2,
            "dropout": 0.0,
            "condition_source": "projected_gps",
        }
    )
    assert isinstance(gps_pooler, GPSQueryPool)
    assert gps_pooler.required_context_modalities == ("gps",)
    assert gps_pooler.context_feature_source == "projected"
    assert gps_pooler.context_feature_kwargs == {"gps": "gps_condition_features"}
    assert gps_pooler(tokens, torch.randn(2, 3, 8)).shape == (2, 3, 8)

    hybrid_pooler = build_jepa_downstream_pooler(
        {
            "type": "hybrid_residual_query",
            "latent_dim": 8,
            "condition_dim": 8,
            "content_queries": 2,
            "gps_queries": 2,
            "num_heads": 2,
            "dropout": 0.0,
            "condition_source": "projected_gps",
            "residual_alpha_init": 0.05,
            "return_attention": True,
        }
    )
    assert isinstance(hybrid_pooler, HybridResidualQueryPool)
    assert hybrid_pooler.required_context_modalities == ("gps",)
    hybrid_output = hybrid_pooler(tokens, torch.randn(2, 3, 8))
    assert hybrid_output.shape == (2, 3, 8)
    assert hybrid_pooler.last_attention_maps["content"].shape == (2, 3, 2, 5)
    assert hybrid_pooler.last_attention_maps["gps"].shape == (2, 3, 2, 5)
    assert hybrid_pooler.last_diagnostics["residual_alpha_init"] == pytest.approx(0.05)
    assert hybrid_pooler.last_diagnostics["branch_attention"]["gps"]["exposed_as_last_attention_map"] is True

    predictive_pooler = build_jepa_downstream_pooler(
        {
            "type": "predictive_gps_query",
            "latent_dim": 8,
            "condition_dim": 8,
            "content_queries": 2,
            "gps_queries": 2,
            "num_heads": 2,
            "dropout": 0.0,
            "residual_scale_init": 0.05,
            "temporal_predictor": {"type": "gru", "history_window": 2, "insufficient_history": "zero"},
            "reliability_gate": {"type": "mlp", "hidden_dim": 8},
            "return_attention": True,
        }
    )
    assert isinstance(predictive_pooler, PredictiveGPSQueryPool)
    assert predictive_pooler.required_context_modalities == ("gps",)
    predictive_output = predictive_pooler(tokens, torch.randn(2, 3, 8))
    assert predictive_output.shape == (2, 3, 8)
    assert predictive_pooler.last_attention_maps["content"].shape == (2, 3, 2, 5)
    assert predictive_pooler.last_attention_maps["gps"].shape == (2, 3, 2, 5)
    assert predictive_pooler.last_diagnostics["condition_id_consumed"] is False
    assert predictive_pooler.last_diagnostics["branch_attention"]["content"]["available"] is True
    assert predictive_pooler.last_diagnostics["branch_attention"]["gps"]["exposed_as_last_attention_map"] is True

    adapter = build_jepa_downstream_adapter({"type": "identity", "latent_dim": 8})
    assert isinstance(adapter, IdentityJepaAdapter)
    pooled = torch.randn(2, 3, 8)
    assert adapter(pooled) is pooled

    with pytest.raises(RegistryError, match="does_not_exist.*jepa_downstream_poolers.*Available names"):
        build_jepa_downstream_pooler({"type": "does_not_exist"})
    with pytest.raises(ValueError, match="Unsupported JEPA downstream adapter 'does_not_exist'.*identity"):
        build_jepa_downstream_adapter({"type": "does_not_exist"})


def test_gps_query_pool_averages_k_query_tokens_after_attention():
    pool = GPSQueryPool(latent_dim=4, condition_dim=3, k_queries=2, num_heads=2, dropout=0.0)
    pool.attention = _EchoQueryAttention()
    patch_tokens = torch.randn(1, 2, 5, 4)
    condition = torch.randn(1, 2, 3)

    pooled, attention = pool(patch_tokens, condition, return_attention=True)

    queries = pool.gps_to_q(condition.reshape(2, 3)).reshape(1, 2, 2, 4)
    expected = pool.output_norm(queries).mean(dim=2)
    torch.testing.assert_close(pooled, expected)
    assert attention.shape == (1, 2, 2, 5)


def test_modular_sequence_token_features_and_readout_metadata():
    import_default_components()
    model = MODELS.build(
        {
            "type": "modular_sequence",
            "modalities": ["image", "gps"],
            "feature_size": 8,
            "d_model": 8,
            "num_classes": 4,
            "num_pred": 2,
            "encoders": {
                "image": {"type": "gps_query_readout_test_identity", "output_dim": 8},
                "gps": {"type": "gps_query_readout_test_identity", "output_dim": 8},
            },
            "projectors": {
                "image": {"type": "identity", "input_dim": 8, "d_model": 8},
                "gps": {"type": "identity", "input_dim": 8, "d_model": 8},
            },
            "representation_core": {
                "type": "token_aware_transformer",
                "d_model": 8,
                "modality_count": 3,
                "num_heads": 2,
                "num_layers": 1,
            },
        }
    )
    output = model(image_batch=torch.randn(2, 3, 2, 8), gps_batch=torch.randn(2, 3, 8))
    metadata = model.training_strategy_metadata()

    assert output["token_features"].shape == (2, 3, 3, 8)
    assert output["logits"].shape == (2, 3, 4)
    assert metadata["token_readout_type"] == "legacy_uniform_mean"
    assert metadata["readout_trainable_params"] == 0
    assert metadata["k_tokens"] == 3


def test_query_weighted_token_readout_shape_metadata_oracle_block_and_backward():
    import_default_components()
    model = MODELS.build(
        {
            "type": "modular_sequence",
            "modalities": ["image", "gps"],
            "feature_size": 8,
            "d_model": 8,
            "num_classes": 4,
            "num_pred": 2,
            "encoders": {
                "image": {"type": "gps_query_readout_test_identity", "output_dim": 8},
                "gps": {"type": "gps_query_readout_test_identity", "output_dim": 8},
            },
            "projectors": {
                "image": {"type": "identity", "input_dim": 8, "d_model": 8},
                "gps": {"type": "identity", "input_dim": 8, "d_model": 8},
            },
            "representation_core": {"type": "query_weighted_token_readout", "d_model": 8, "modality_count": 3},
        }
    )
    output = model(
        image_batch=torch.randn(2, 3, 2, 8),
        gps_batch=torch.randn(2, 3, 8),
        benchmark_condition_metadata={"target_beam": 3, "c_idx": 4},
    )
    loss = output["logits"].sum()
    loss.backward()
    metadata = model.training_strategy_metadata()
    diagnostics = output["token_readout_diagnostics"]

    assert output["output_features"].shape == (2, 3, 8)
    assert metadata["token_readout_type"] == "learned_query_weighted"
    assert metadata["readout_trainable_params"] == 3
    assert diagnostics["output_shape"] == [2, 3, 8]
    assert diagnostics["condition_id_consumed"] is False
    assert {"target_beam", "c_idx"} <= set(diagnostics["blocked_condition_fields"])
    assert model.representation_core.readout_logits.grad is not None


def test_hybrid_residual_query_pooler_optional_and_required_gps_paths():
    tokens = torch.randn(2, 4, 6, 8)
    condition = torch.randn(2, 4, 5)
    required = HybridResidualQueryPool(
        latent_dim=8,
        condition_dim=5,
        content_queries=2,
        gps_queries=3,
        num_heads=2,
        dropout=0.0,
        residual_alpha_init=0.15,
        return_attention=True,
    )

    output = required(tokens, condition)

    assert output.shape == (2, 4, 8)
    assert required.required_context_modalities == ("gps",)
    assert required.last_diagnostics["gps_condition_available"] is True
    assert required.last_diagnostics["residual_alpha"] == pytest.approx(0.15)
    assert required.last_attention_maps["content"].shape == (2, 4, 2, 6)
    assert required.last_attention_maps["gps"].shape == (2, 4, 3, 6)
    with pytest.raises(ValueError, match="requires GPS condition features"):
        required(tokens)

    optional = HybridResidualQueryPool(
        latent_dim=8,
        condition_dim=5,
        content_queries=1,
        gps_queries=1,
        num_heads=2,
        dropout=0.0,
        require_condition=False,
    )
    optional_output = optional(tokens)
    assert optional.required_context_modalities == ()
    assert optional_output.shape == (2, 4, 8)
    assert optional.last_diagnostics["gps_condition_available"] is False


def test_predictive_gps_query_pooler_uses_past_only_and_blocks_condition_ids():
    torch.manual_seed(7)
    pooler = PredictiveGPSQueryPool(
        latent_dim=8,
        condition_dim=8,
        content_queries=2,
        gps_queries=2,
        num_heads=2,
        dropout=0.0,
        residual_scale_init=0.05,
        temporal_predictor={"type": "gru", "history_window": 2, "insufficient_history": "zero"},
        reliability_gate={"type": "mlp", "hidden_dim": 8},
        return_attention=True,
    )
    pooler.eval()
    tokens = torch.randn(1, 4, 5, 8)
    condition = torch.randn(1, 4, 8)

    output = pooler(
        tokens,
        condition,
        image_valid_mask=torch.tensor([[True, True, False, True]]),
        image_observability_score=torch.tensor([[1.0, 0.9, 0.2, 1.0]]),
        gps_valid_mask=torch.tensor([[True, True, True, False]]),
        gps_counterfactual_mask=torch.tensor([[False, False, True, False]]),
        benchmark_condition_metadata={
            "predictive_condition_id": "P4_joint_predictive_recovery",
            "gps_condition": "C4_severe_async",
            "image_condition": "D7_joint_worst_case",
            "c_idx": 4,
            "d_idx": 7,
        },
    )
    first_temporal = pooler.last_temporal_predicted_latent.clone()
    first_diagnostics = dict(pooler.last_diagnostics)
    changed_future = tokens.clone()
    changed_future[:, 3:, :, :] = changed_future[:, 3:, :, :] + 100.0
    pooler(changed_future, condition)
    second_temporal = pooler.last_temporal_predicted_latent

    assert output.shape == (1, 4, 8)
    torch.testing.assert_close(first_temporal, second_temporal, atol=1e-5, rtol=1e-5)
    diagnostics = first_diagnostics
    assert diagnostics["temporal_source_history_range"][0] is None
    assert diagnostics["temporal_source_history_range"][1] == [0, 0]
    assert diagnostics["temporal_source_history_range"][3] == [1, 2]
    assert diagnostics["temporal_source_history_range_policy"] == "strictly_past"
    assert diagnostics["condition_id_consumed"] is False
    assert {"predictive_condition_id", "gps_condition", "image_condition", "c_idx", "d_idx"} <= set(
        diagnostics["blocked_condition_fields"]
    )
    assert diagnostics["residual_scale"] == pytest.approx(0.05)
    assert set(diagnostics["gate_weight_mean"]) == {"current_content", "temporal_predicted", "gps_residual"}
    assert diagnostics["branch_availability"]["current_content"] is True
    assert diagnostics["branch_availability"]["temporal_predicted"] is True
    assert diagnostics["gps_counterfactual_count"] == 1


def test_jepa_context_image_encoder_loads_best_and_last_payloads(tmp_path: Path):
    import_default_components()
    source = MODELS.build(_model_cfg())
    best_path = tmp_path / "best.pth"
    last_path = tmp_path / "last.pth"
    torch.save(source.state_dict(), best_path)
    torch.save({"state_dict": source.state_dict(), "epoch": 2}, last_path)

    for checkpoint_path in (best_path, last_path):
        encoder = JepaContextImageEncoder(
            checkpoint_path=str(checkpoint_path),
            output_dim=16,
            latent_dim=16,
            image_channels=3,
            image_profile="rgb_imagenet",
            freeze_encoder=True,
            visual_encoder={
                "image_channels": 3,
                "latent_dim": 16,
                "patch_size": 8,
                "depth": 0,
                "max_tokens": 16,
            },
        )

        output = encoder(torch.randn(2, 3, 3, 32, 32))

        assert output.shape == (2, 3, 16)
        assert torch.equal(encoder.context_encoder.patch_embed.weight, source.context_encoder.patch_embed.weight)
        assert all(not param.requires_grad for param in encoder.context_encoder.parameters())

    gps_query_encoder = JepaContextImageEncoder(
        checkpoint_path=str(best_path),
        output_dim=16,
        latent_dim=16,
        image_channels=3,
        image_profile="rgb_imagenet",
        visual_encoder={
            "image_channels": 3,
            "latent_dim": 16,
            "patch_size": 8,
            "depth": 0,
            "max_tokens": 16,
        },
        pooling="gps_query_attention",
        gps_query_pool={"condition_dim": 16, "k_queries": 2, "num_heads": 4, "dropout": 0.0},
    )
    assert gps_query_encoder.required_context_modalities == ("gps",)
    assert gps_query_encoder.training_strategy_metadata()["gps_query_pooling_enabled"] is True
    assert gps_query_encoder(
        torch.randn(2, 3, 3, 32, 32),
        gps_condition_features=torch.randn(2, 3, 16),
    ).shape == (2, 3, 16)


def test_jepa_context_image_encoder_mean_default_and_gps_query_forward_errors():
    mean_encoder = JepaContextImageEncoder(
        output_dim=16,
        latent_dim=16,
        image_channels=3,
        image_profile="rgb_imagenet",
        visual_encoder={
            "image_channels": 3,
            "latent_dim": 16,
            "patch_size": 8,
            "depth": 0,
            "max_tokens": 16,
        },
    )
    assert mean_encoder.pooling == "mean"
    assert mean_encoder.required_context_modalities == ()
    assert mean_encoder(torch.randn(2, 3, 3, 32, 32)).shape == (2, 3, 16)

    gps_query_encoder = JepaContextImageEncoder(
        output_dim=16,
        latent_dim=16,
        image_channels=3,
        image_profile="rgb_imagenet",
        visual_encoder={
            "image_channels": 3,
            "latent_dim": 16,
            "patch_size": 8,
            "depth": 0,
            "max_tokens": 16,
        },
        pooling="gps_query_attention",
        gps_query_pool={
            "condition_dim": 16,
            "k_queries": 2,
            "num_heads": 4,
            "dropout": 0.0,
            "return_attention": True,
        },
    )
    with pytest.raises(ValueError, match="GPS-query pooling requires GPS condition feature"):
        gps_query_encoder(torch.randn(2, 3, 3, 32, 32))
    output = gps_query_encoder(
        torch.randn(2, 3, 3, 32, 32),
        gps_condition_features=torch.randn(2, 3, 16),
    )
    assert output.shape == (2, 3, 16)
    assert gps_query_encoder.last_attention_map is not None
    assert gps_query_encoder.last_attention_map.shape == (2, 3, 2, 16)


def test_jepa_context_image_encoder_records_visual_token_diagnostics_for_new_tokenizer():
    encoder = JepaContextImageEncoder(
        output_dim=8,
        latent_dim=8,
        image_channels=3,
        image_profile="rgb_imagenet",
        visual_encoder={
            "type": "overlap_patch",
            "image_channels": 3,
            "latent_dim": 8,
            "kernel_size": 16,
            "stride": 8,
            "depth": 0,
            "max_tokens": 16,
        },
        pooler={
            "type": "gps_query_attention",
            "condition_dim": 8,
            "k_queries": 2,
            "num_heads": 2,
            "return_attention": True,
        },
    )

    output = encoder(torch.randn(2, 3, 3, 32, 32), gps_condition_features=torch.randn(2, 3, 8))

    diagnostics = encoder.last_visual_token_diagnostics
    metadata = encoder.training_strategy_metadata()
    assert output.shape == (2, 3, 8)
    assert diagnostics["token_grid"] == [3, 3]
    assert diagnostics["token_count"] == 9
    assert diagnostics["attention_shape"] == [2, 3, 2, 9]
    assert diagnostics["attention_entropy"] > 0.0
    assert metadata["visual_token_encoder"]["visual_encoder_type"] == "overlap_patch"
    assert metadata["checkpoint_policy"] == "fresh_stage1_required"


def test_k_token_pooler_output_mode_requires_token_aware_core():
    import_default_components()
    base_cfg = {
        "type": "modular_sequence",
        "modalities": ["image", "gps"],
        "feature_size": 8,
        "d_model": 8,
        "num_classes": 7,
        "num_pred": 1,
        "image_profile": "rgb_imagenet",
        "image_channels": 3,
        "gps_input_size": 3,
        "encoders": {
            "image": {
                "type": "jepa_context_image",
                "checkpoint_path": "",
                "strict": False,
                "output_dim": 8,
                "latent_dim": 8,
                "image_channels": 3,
                "visual_encoder": {
                    "type": "patch_vit",
                    "image_channels": 3,
                    "latent_dim": 8,
                    "patch_size": 16,
                    "depth": 0,
                    "max_tokens": 4,
                },
                "pooler": {
                    "type": "gps_query_attention",
                    "condition_dim": 8,
                    "latent_dim": 8,
                    "k_queries": 2,
                    "num_heads": 2,
                    "return_attention": True,
                    "output_mode": "tokens",
                },
            },
            "gps": {"type": "gps_mlp", "output_dim": 8, "hidden_size": 8, "dropout": 0.0},
        },
        "projectors": {
            "image": {"type": "linear", "d_model": 8, "dropout": 0.0},
            "gps": {"type": "linear", "d_model": 8, "dropout": 0.0},
        },
        "representation_core": {
            "type": "token_aware_transformer",
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
        },
        "heads": {"beam": {"type": "beam_head", "dropout": 0.0}},
    }
    model = MODELS.build(copy.deepcopy(base_cfg))
    model.eval()

    with torch.no_grad():
        output = model(image_batch=torch.randn(2, 2, 3, 32, 32), gps_batch=torch.randn(2, 2, 3))

    assert output["logits"].shape == (2, 2, 7)
    assert output["encoder_features"]["image"].shape == (2, 2, 2, 8)
    assert output["token_features"].shape == (2, 3, 2, 8)
    assert output["runtime_metadata"]["encoder_temporal_auxiliary"]["image"]["visual_tokens"]["token_count"] == 4

    bad_cfg = copy.deepcopy(base_cfg)
    bad_cfg["representation_core"] = {
        "type": "early_concat_gru",
        "d_model": 8,
        "modality_count": 2,
        "hidden_size": 8,
    }
    bad_model = MODELS.build(bad_cfg)
    with pytest.raises(ValueError, match="expected K=2, D=8, got .*3"):
        bad_model(image_batch=torch.randn(1, 2, 3, 32, 32), gps_batch=torch.randn(1, 2, 3))


def test_jepa_context_image_encoder_accepts_explicit_pooler_adapter_config_and_metadata():
    encoder = JepaContextImageEncoder(
        output_dim=16,
        latent_dim=16,
        image_channels=3,
        image_profile="rgb_imagenet",
        visual_encoder={
            "image_channels": 3,
            "latent_dim": 16,
            "patch_size": 8,
            "depth": 0,
            "max_tokens": 16,
        },
        pooler={
            "type": "gps_query_attention",
            "condition_dim": 16,
            "k_queries": 2,
            "num_heads": 4,
            "dropout": 0.0,
            "condition_source": "projected_gps",
            "return_attention": True,
        },
        adapter={"type": "identity"},
    )

    output = encoder(
        torch.randn(2, 3, 3, 32, 32),
        gps_condition_features=torch.randn(2, 3, 16),
    )

    metadata = encoder.training_strategy_metadata()
    assert output.shape == (2, 3, 16)
    assert encoder.pooling == "gps_query_attention"
    assert encoder.required_context_modalities == ("gps",)
    assert encoder.context_feature_source == "projected"
    assert metadata["pooler_type"] == "gps_query_attention"
    assert metadata["adapter_type"] == "identity"
    assert metadata["condition_source"] == "projected_gps"
    assert metadata["gps_query_pool"]["return_attention"] is True
    assert encoder.last_attention_map is not None

    with pytest.raises(RegistryError, match="unknown_pooler.*jepa_downstream_poolers"):
        JepaContextImageEncoder(
            output_dim=16,
            latent_dim=16,
            image_profile="rgb_imagenet",
            visual_encoder={"patch_size": 8, "depth": 0, "max_tokens": 16},
            pooler={"type": "unknown_pooler"},
        )
    with pytest.raises(ValueError, match="Unsupported JEPA downstream adapter 'unknown_adapter'.*identity"):
        JepaContextImageEncoder(
            output_dim=16,
            latent_dim=16,
            image_profile="rgb_imagenet",
            visual_encoder={"patch_size": 8, "depth": 0, "max_tokens": 16},
            adapter={"type": "unknown_adapter"},
        )


def test_jepa_context_image_encoder_hybrid_pooler_and_temporal_auxiliary_metadata():
    encoder = JepaContextImageEncoder(
        output_dim=8,
        latent_dim=8,
        image_channels=3,
        image_profile="rgb_imagenet",
        visual_encoder={
            "image_channels": 3,
            "latent_dim": 8,
            "patch_size": 8,
            "depth": 0,
            "max_tokens": 16,
        },
        pooler={
            "type": "hybrid_residual_query",
            "condition_dim": 8,
            "content_queries": 2,
            "gps_queries": 2,
            "num_heads": 2,
            "dropout": 0.0,
            "residual_alpha_init": 0.2,
            "return_attention": True,
        },
        temporal_auxiliary={
            "enabled": True,
            "history_window": 2,
            "insufficient_history": "zero",
        },
    )

    image = torch.randn(2, 4, 3, 32, 32)
    output = encoder(image, gps_condition_features=torch.randn(2, 4, 8))

    metadata = encoder.training_strategy_metadata()
    aux_metadata = encoder.last_temporal_auxiliary_metadata
    assert output.shape == (2, 4, 8)
    assert encoder.pooling == "hybrid_residual_query"
    assert encoder.required_context_modalities == ("gps",)
    assert metadata["hybrid_residual_query_enabled"] is True
    assert metadata["pooler"]["residual_alpha_init"] == pytest.approx(0.2)
    assert metadata["temporal_auxiliary_enabled"] is True
    assert encoder.last_current_latent is not None
    assert encoder.last_temporal_predicted_latent is not None
    assert encoder.last_current_latent.shape == encoder.last_temporal_predicted_latent.shape == (2, 4, 8)
    assert aux_metadata["available"] is True
    assert aux_metadata["source_history_range"][0] is None
    assert aux_metadata["source_history_range"][1] == [0, 0]
    assert aux_metadata["source_history_range"][3] == [1, 2]
    assert aux_metadata["insufficient_history_count"] == 2

    with pytest.raises(ValueError, match="hybrid residual query pooling requires GPS condition feature"):
        encoder(torch.randn(1, 2, 3, 32, 32))


def test_jepa_context_image_encoder_predictive_gps_query_metadata_and_checkpoint_guard():
    encoder = JepaContextImageEncoder(
        output_dim=8,
        latent_dim=8,
        image_channels=3,
        image_profile="rgb_imagenet",
        visual_encoder={
            "image_channels": 3,
            "latent_dim": 8,
            "patch_size": 8,
            "depth": 0,
            "max_tokens": 16,
        },
        pooler={
            "type": "predictive_gps_query",
            "condition_dim": 8,
            "content_queries": 2,
            "gps_queries": 2,
            "num_heads": 2,
            "dropout": 0.0,
            "residual_scale_init": 0.2,
            "temporal_predictor": {"type": "gru", "history_window": 2, "insufficient_history": "zero"},
            "reliability_gate": {"type": "mlp", "hidden_dim": 8},
            "return_attention": True,
        },
    )
    output = encoder(
        torch.randn(2, 4, 3, 32, 32),
        gps_condition_features=torch.randn(2, 4, 8),
        image_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        image_observability_score=torch.ones(2, 4),
        benchmark_condition_metadata={"predictive_condition_id": "P4_joint_predictive_recovery"},
    )
    metadata = encoder.training_strategy_metadata()

    assert output.shape == (2, 4, 8)
    assert encoder.pooling == "predictive_gps_query"
    assert encoder.supports_observability_metadata is True
    assert metadata["predictive_gps_query_enabled"] is True
    assert metadata["gps_query_plus_plus_enabled"] is True
    assert metadata["content_query_count"] == 2
    assert metadata["gps_query_count"] == 2
    assert metadata["temporal_predictor_type"] == "gru"
    assert metadata["reliability_gate_type"] == "mlp"
    assert metadata["residual_scale"] == pytest.approx(0.2)
    assert metadata["jepa_checkpoint_path"] == ""
    assert metadata["context_encoder_frozen"] is False
    assert encoder.last_predictive_gps_query_diagnostics["condition_id_consumed"] is False

    legacy_state = {
        "gps_query_pool.gps_to_q.1.weight": torch.randn(8, 8),
    }
    with pytest.raises(RuntimeError, match="legacy gps_query_attention checkpoint"):
        encoder.load_state_dict(legacy_state, strict=True)


def test_jepa_context_image_encoder_temporal_fallback_uses_past_only():
    torch.manual_seed(123)
    encoder = JepaContextImageEncoder(
        output_dim=8,
        latent_dim=8,
        image_channels=3,
        image_profile="rgb_imagenet",
        visual_encoder={
            "image_channels": 3,
            "latent_dim": 8,
            "patch_size": 8,
            "depth": 0,
            "max_tokens": 16,
        },
        temporal_fallback={
            "enabled": True,
            "history_window": 4,
            "observability_threshold": 0.5,
            "insufficient_history": "raw",
        },
    )
    image = torch.randn(1, 5, 3, 32, 32)
    clean = encoder(
        image,
        image_valid_mask=torch.ones(1, 5, dtype=torch.bool),
        image_observability_score=torch.ones(1, 5),
        benchmark_condition_metadata={"gps_condition": "C0_sync", "image_condition": "D0_full_image"},
    )
    degraded = encoder(
        image,
        image_valid_mask=torch.tensor([[True, True, True, True, True]]),
        image_observability_score=torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.2]]),
        benchmark_condition_metadata={"gps_condition": "C4_severe_async", "image_condition": "D6_burst_missing"},
    )

    assert torch.allclose(degraded[:, 4, :], clean[:, 0:4, :].mean(dim=1), atol=1e-5)
    assert torch.allclose(degraded[:, :4, :], clean[:, :4, :], atol=1e-5)
    metadata = encoder.last_temporal_fallback_metadata
    assert metadata["affected_count"] == 1
    assert metadata["source_history_range"][4] == [0, 3]
    assert metadata["jepa_advantage_condition"] is True

    first_frame = encoder(
        image,
        image_valid_mask=torch.tensor([[False, True, True, True, True]]),
        image_observability_score=torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]),
    )
    assert torch.allclose(first_frame[:, 0, :], clean[:, 0, :], atol=1e-5)
    assert encoder.last_temporal_fallback_metadata["insufficient_history_count"] == 1


def test_jepa_mask_sampler_random_and_gps_biased_are_reproducible_and_non_overlapping():
    gps = torch.zeros(2, 2, 3)
    for mode in ("random", "gps_angle_biased"):
        sampler = JepaMaskSampler(mode=mode, context_ratio=0.5, target_ratio=0.25, seed=3)
        first = sampler.sample(
            batch_size=2,
            seq_len=2,
            num_tokens=16,
            grid_size=(4, 4),
            gps_batch=gps,
            token_metadata={
                "visual_encoder_type": "patch_vit",
                "checkpoint_policy": "exact_reuse",
                "token_grid": [4, 4],
                "token_count": 16,
            },
            epoch=1,
            step=2,
        )
        second = sampler.sample(
            batch_size=2,
            seq_len=2,
            num_tokens=16,
            grid_size=(4, 4),
            gps_batch=gps,
            token_metadata={
                "visual_encoder_type": "patch_vit",
                "checkpoint_policy": "exact_reuse",
                "token_grid": [4, 4],
                "token_count": 16,
            },
            epoch=1,
            step=2,
        )
        assert torch.equal(first.context_mask, second.context_mask)
        assert torch.equal(first.target_mask, second.target_mask)
        assert not torch.any(first.context_mask & first.target_mask)
        assert first.diagnostics["jepa/token_count"] == pytest.approx(16.0)
        assert first.diagnostics["jepa/token_grid_h"] == pytest.approx(4.0)
        assert first.diagnostics["jepa/checkpoint_policy"] == "exact_reuse"

    multiscale = JepaMaskSampler(mode="gps_angle_biased", context_ratio=0.5, target_ratio=0.25, seed=3).sample(
        batch_size=1,
        seq_len=1,
        num_tokens=20,
        grid_size=(4, 4),
        gps_batch=torch.zeros(1, 1, 3),
        token_metadata={
            "visual_encoder_type": "multi_scale_cnn",
            "token_grid": [4, 4],
            "token_count": 20,
            "scale_token_counts": {"layer3": 16, "layer4": 4},
        },
    )
    assert multiscale.context_mask.shape == (1, 1, 20)
    assert multiscale.diagnostics["jepa/multiscale_token_count"] == pytest.approx(20.0)


def test_jepa_mask_sampler_allows_single_token_encoder():
    sample = JepaMaskSampler(mode="random", context_ratio=0.6, target_ratio=0.2, seed=3).sample(
        batch_size=2,
        seq_len=3,
        num_tokens=1,
        grid_size=(1, 1),
        gps_batch=torch.zeros(2, 3, 3),
        token_metadata={
            "visual_encoder_type": "tinyvit_frame",
            "checkpoint_policy": "supervised_only_anchor",
            "token_grid": [1, 1],
            "token_count": 1,
        },
    )

    assert sample.context_indices.shape == (2, 3, 1)
    assert sample.target_indices.shape == (2, 3, 1)
    assert torch.all(sample.context_mask)
    assert torch.all(sample.target_mask)
    assert sample.diagnostics["jepa/degenerate_single_token_mask"] == pytest.approx(1.0)


def test_jepa_latent_loss_masks_and_empty_mask_protection():
    predicted = torch.tensor([[[[1.0, 0.0], [2.0, 2.0]]]])
    target = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    mask = torch.tensor([[[True, False]]])

    result = jepa_latent_prediction_loss(predicted, target, mask, {"loss": {"jepa": {"type": "mse"}}})

    assert float(result.loss.item()) == pytest.approx(0.5)
    assert result.diagnostics["loss/jepa"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no valid target tokens"):
        jepa_latent_prediction_loss(predicted, target, torch.zeros_like(mask))


class _EchoQueryAttention(torch.nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        need_weights: bool = False,
        average_attn_weights: bool = True,
    ):
        del value, average_attn_weights
        weights = torch.full(
            (query.shape[0], query.shape[1], key.shape[1]),
            1.0 / float(key.shape[1]),
            device=query.device,
            dtype=query.dtype,
        )
        return query, weights if need_weights else None


def test_jepa_validation_and_training_smoke_records_loss_checkpoint_and_metadata(tmp_path: Path):
    cfg = _tiny_train_cfg(tmp_path)
    result = train(cfg)

    assert result["history"]["val_jepa_loss"][-1] > 0.0
    assert result["prediction_objective"]["primary_metric"] == "val_jepa_loss"
    assert result["prediction_objective"]["jepa"]["context_encoder_artifact_key"] == "model.primary.context_encoder"
    run_dir = Path(result["run_dir"])
    assert (run_dir / "checkpoints" / "last.pth").exists()
    assert (run_dir / "checkpoints" / "best.pth").exists()
    assert not (run_dir / "checkpoints" / "best_top1.pth").exists()

    import_default_components()
    model = MODELS.build(cfg["model"]["primary"])
    metrics = validate(model, [{"image": torch.rand(1, 2, 3, 32, 32), "gps": torch.rand(1, 2, 3)}], cfg, None, torch.device("cpu"))
    assert {"val_loss", "val_jepa_loss"} <= set(metrics["available_metrics"])
    assert "val_adba" not in metrics
    assert "val_acc" not in metrics
