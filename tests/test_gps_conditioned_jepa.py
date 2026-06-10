from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.engine.trainer import train  # noqa: E402
from kd_sensing.engine.validator import validate  # noqa: E402
from kd_sensing.losses.jepa import jepa_latent_prediction_loss  # noqa: E402
from kd_sensing.models.jepa import JepaContextImageEncoder, JepaMaskSampler  # noqa: E402
from kd_sensing.registries import MODELS, import_default_components  # noqa: E402


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
            epoch=1,
            step=2,
        )
        second = sampler.sample(
            batch_size=2,
            seq_len=2,
            num_tokens=16,
            grid_size=(4, 4),
            gps_batch=gps,
            epoch=1,
            step=2,
        )
        assert torch.equal(first.context_mask, second.context_mask)
        assert torch.equal(first.target_mask, second.target_mask)
        assert not torch.any(first.context_mask & first.target_mask)


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
