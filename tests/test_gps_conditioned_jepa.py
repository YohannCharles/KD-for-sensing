from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.trainer import train
from kd_sensing.engine.validator import validate
from kd_sensing.losses.jepa import jepa_latent_prediction_loss
from kd_sensing.models.jepa import (
    JepaContextImageEncoder,
    JepaMaskSampler,
    build_visual_token_encoder,
)
from kd_sensing.models.jepa_downstream import MeanPatchPooler, build_jepa_downstream_pooler
from kd_sensing.registries import MODELS, RegistryError, import_default_components

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
            "validation_from_train": {"enabled": True, "fraction": 0.5, "seed": 3},
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
            "model_selection": True,
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

def _mean_encoder(**overrides):
    config = {
        "output_dim": 16,
        "latent_dim": 16,
        "image_channels": 3,
        "image_profile": "rgb_imagenet",
        "visual_encoder": {
            "image_channels": 3,
            "latent_dim": 16,
            "patch_size": 8,
            "depth": 0,
            "max_tokens": 16,
        },
    }
    config.update(overrides)
    return JepaContextImageEncoder(**config)


def test_jepa_context_image_encoder_mean_default_and_retired_poolers_fail_fast():
    encoder = _mean_encoder()

    assert encoder.pooling == "mean"
    assert encoder.required_context_modalities == ()
    assert isinstance(encoder.pooler, MeanPatchPooler)
    assert encoder(torch.randn(2, 3, 3, 32, 32)).shape == (2, 3, 16)

    for pooling in (
        "gps_query_attention",
        "learned_query_attention",
        "self_attention",
        "hybrid_residual_query",
        "predictive_gps_query",
    ):
        with pytest.raises(ValueError, match="retired.*only 'mean'"):
            _mean_encoder(pooling=pooling)
    with pytest.raises(ValueError, match="K-token output has been retired"):
        build_jepa_downstream_pooler({"type": "mean", "output_mode": "tokens", "latent_dim": 16})
    with pytest.raises(ValueError, match="adapters have been retired"):
        _mean_encoder(adapter={"type": "identity"})


def test_jepa_context_image_encoder_records_mean_visual_token_diagnostics():
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
    )

    output = encoder(torch.randn(2, 3, 3, 32, 32))

    diagnostics = encoder.last_visual_token_diagnostics
    metadata = encoder.training_strategy_metadata()
    assert output.shape == (2, 3, 8)
    assert diagnostics["token_grid"] == [3, 3]
    assert diagnostics["token_count"] == 9
    assert diagnostics["pooler_type"] == "mean"
    assert "attention_shape" not in diagnostics
    assert metadata["pooling"] == "mean"
    assert "adapter" not in metadata
    assert "gps_query_pool" not in metadata


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
