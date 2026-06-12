from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.engine.artifacts import final_config_with_runtime  # noqa: E402
from kd_sensing.engine.data_factory import build_dataloaders  # noqa: E402
from kd_sensing.engine.trainer import train  # noqa: E402
from kd_sensing.engine.validator import validate  # noqa: E402
from kd_sensing.losses.jepa import jepa_latent_prediction_loss  # noqa: E402
from kd_sensing.models.jepa import GPSQueryPool, JepaContextImageEncoder, JepaMaskSampler  # noqa: E402
from kd_sensing.models.jepa_downstream import (  # noqa: E402
    IdentityJepaAdapter,
    MeanPatchPooler,
    build_jepa_downstream_adapter,
    build_jepa_downstream_pooler,
)
from kd_sensing.registries import MODELS, RegistryError, import_default_components  # noqa: E402

GPS_QUERY_DOWNSTREAM_CONFIGS = {
    "fair_gps_query_pooling": ROOT
    / "configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_query_pool_best_beambench_fair_lowmem.yaml",
    "fair_gps_query_pooling_2604": ROOT
    / "configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_query_pool_best_2604_s32_s34_lowmem.yaml",
}
PARAM_GROUP_DERIVED_CONFIG = (
    ROOT
    / "configs/fusion/experiments/jepa_image_gps/"
    "image_gps_jepa_gps_biased_pooler_param_groups_beambench_fair_lowmem.yaml"
)


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


def test_gps_query_pool_shape_attention_map_and_dimension_validation():
    pool = GPSQueryPool(latent_dim=8, condition_dim=5, k_queries=3, num_heads=2, dropout=0.0)
    patch_tokens = torch.randn(2, 4, 6, 8)
    condition = torch.randn(2, 4, 5)

    pooled, attention = pool(patch_tokens, condition, return_attention=True)

    assert pooled.shape == (2, 4, 8)
    assert attention.shape == (2, 4, 3, 6)
    assert attention.requires_grad is False
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 4, 3), atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match="patch tokens shape .*condition feature shape"):
        pool(patch_tokens, torch.randn(2, 3, 5))
    with pytest.raises(ValueError, match="expected condition feature dim 5"):
        pool(patch_tokens, torch.randn(2, 4, 4))


def test_jepa_downstream_pooler_adapter_registry_builds_and_reports_unknown_names():
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

    adapter = build_jepa_downstream_adapter({"type": "identity", "latent_dim": 8})
    assert isinstance(adapter, IdentityJepaAdapter)
    pooled = torch.randn(2, 3, 8)
    assert adapter(pooled) is pooled

    with pytest.raises(RegistryError, match="does_not_exist.*jepa_downstream_poolers.*Available names"):
        build_jepa_downstream_pooler({"type": "does_not_exist"})
    with pytest.raises(RegistryError, match="does_not_exist.*jepa_downstream_adapters.*Available names"):
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
    with pytest.raises(RegistryError, match="unknown_adapter.*jepa_downstream_adapters"):
        JepaContextImageEncoder(
            output_dim=16,
            latent_dim=16,
            image_profile="rgb_imagenet",
            visual_encoder={"patch_size": 8, "depth": 0, "max_tokens": 16},
            adapter={"type": "unknown_adapter"},
        )


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


def test_gps_query_downstream_configs_load_and_record_metadata(tmp_path: Path):
    baseline = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml"
    )
    baseline_metadata = final_config_with_runtime(baseline, run_dir=tmp_path / "baseline")["runtime"]["jepa_downstream"]
    assert baseline["experiment"]["protocol"] == "beambench_tableiii_input_s32_s34_train_s31_s34_test"
    assert baseline["data"]["dataset"]["seq_len"] == 1
    assert baseline["data"]["dataset"]["num_pred"] == 1
    assert baseline["data"]["dataset"]["gps_feature_mode"] == "paper_distance_angle"
    assert baseline["data"]["dataset"]["gps_angle_offset_source"] == "paper_scene_default"
    assert baseline["data"]["dataset"]["beam_target_source"] == "current"
    assert baseline["model"]["primary"]["gps_input_size"] == 2
    assert baseline["model"]["primary"]["seq_length"] == 1
    assert baseline["evaluation"]["k_values"] == [1, 3, 5]
    assert baseline["evaluation"]["dba_distance_mode"] == "linear"
    assert baseline["model"]["primary"]["encoders"]["image"].get("pooling", "mean") == "mean"
    assert baseline_metadata["pooling"] == "mean"
    assert baseline_metadata["pooler_type"] == "mean"
    assert baseline_metadata["adapter_type"] == "identity"
    assert baseline_metadata["gps_query_pooling_enabled"] is False

    for name, path in GPS_QUERY_DOWNSTREAM_CONFIGS.items():
        cfg = load_config(path)
        image_encoder = cfg["model"]["primary"]["encoders"]["image"]
        checkpoint = image_encoder["checkpoint_path"]

        assert cfg["model"]["primary"]["type"] == "modular_sequence"
        if "beambench_fair" in path.name:
            assert cfg["data"]["dataset"]["beam_target_source"] == "current"
        assert image_encoder["type"] == "jepa_context_image"
        assert image_encoder["pooling"] == "gps_query_attention"
        assert image_encoder["gps_query_pool"]["k_queries"] == 4
        assert image_encoder["gps_query_pool"]["num_heads"] == 4
        assert image_encoder["gps_query_pool"]["condition_source"] == "projected_gps"
        assert "gps_biased_s32_s34_lowmem/checkpoints/best.pth" in checkpoint
        assert "outputs/scene31" not in checkpoint

        metadata = final_config_with_runtime(cfg, run_dir=tmp_path / name)["runtime"]["jepa_downstream"]
        assert metadata["pooling"] == "gps_query_attention"
        assert metadata["pooler_type"] == "gps_query_attention"
        assert metadata["adapter_type"] == "identity"
        assert metadata["gps_query_pooling_enabled"] is True
        assert metadata["gps_query_k_queries"] == 4
        assert metadata["gps_query_num_heads"] == 4
        assert metadata["gps_query_condition_source"] == "projected_gps"
        assert metadata["jepa_checkpoint_path"] == checkpoint
        assert metadata["freeze_image_encoder"] is False
        assert metadata["image_encoder"]["gps_query_pool"]["enabled"] is True


def test_jepa_downstream_param_group_derived_config_inherits_baseline_scope(tmp_path: Path):
    baseline = load_config(
        ROOT
        / "configs/fusion/experiments/jepa_image_gps/image_gps_jepa_gps_biased_best_beambench_fair_lowmem.yaml"
    )
    cfg = load_config(PARAM_GROUP_DERIVED_CONFIG)

    baseline_image = baseline["model"]["primary"]["encoders"]["image"]
    image_encoder = cfg["model"]["primary"]["encoders"]["image"]

    assert image_encoder["checkpoint_path"] == baseline_image["checkpoint_path"]
    assert cfg["model"]["primary"]["modalities"] == baseline["model"]["primary"]["modalities"] == ["image", "gps"]
    assert cfg["experiment"]["objective"] == baseline["experiment"]["objective"] == "beam"
    assert cfg["data"]["dataset"]["train_scenes"] == baseline["data"]["dataset"]["train_scenes"]
    assert cfg["data"]["dataset"]["test_scenes"] == baseline["data"]["dataset"]["test_scenes"]
    assert image_encoder["pooler"]["type"] == "gps_query_attention"
    assert image_encoder["adapter"]["type"] == "identity"
    assert [group["name"] for group in cfg["training"]["optimizer"]["parameter_groups"]] == [
        "jepa_context_encoder",
        "jepa_pooler_adapter",
        "gps_encoder_projector",
        "fusion_head",
    ]

    metadata = final_config_with_runtime(cfg, run_dir=tmp_path / "param_groups")["runtime"]["jepa_downstream"]
    assert metadata["ablation"] == "fair_gps_biased_pooler_param_groups"
    assert metadata["pooler_type"] == "gps_query_attention"
    assert metadata["adapter_type"] == "identity"
    assert metadata["jepa_checkpoint_path"] == image_encoder["checkpoint_path"]


def test_jepa_downstream_runtime_metadata_prefers_model_declaration_and_records_optimizer_summary(tmp_path: Path):
    import_default_components()
    cfg = load_config(GPS_QUERY_DOWNSTREAM_CONFIGS["fair_gps_query_pooling"])
    model_cfg = _tiny_downstream_model_cfg(cfg)
    model_cfg["encoders"]["image"]["pooler"] = {
        "type": "gps_query_attention",
        "condition_dim": 16,
        "latent_dim": 16,
        "k_queries": 2,
        "num_heads": 4,
        "dropout": 0.0,
        "condition_source": "projected_gps",
        "return_attention": False,
    }
    model_cfg["encoders"]["image"].pop("pooling", None)
    model_cfg["encoders"]["image"].pop("gps_query_pool", None)
    model_cfg["encoders"]["image"]["adapter"] = {"type": "identity"}
    model = MODELS.build(model_cfg)
    optimizer_groups = [
        {"index": 0, "name": "jepa_pooler", "lr": 0.001, "weight_decay": 0.0, "param_count": 128},
        {"index": 1, "name": "fusion_head", "lr": 0.00075, "weight_decay": 0.0001, "param_count": 64},
    ]

    metadata = final_config_with_runtime(
        cfg,
        run_dir=tmp_path / "model_metadata",
        model=model,
        optimizer_groups=optimizer_groups,
    )["runtime"]["jepa_downstream"]

    assert metadata["source"] == "model"
    assert metadata["jepa_checkpoint_path"] == ""
    assert metadata["state_dict_prefix"] == "context_encoder"
    assert metadata["pooler_type"] == "gps_query_attention"
    assert metadata["adapter_type"] == "identity"
    assert metadata["gps_query_k_queries"] == 2
    assert metadata["gps_query_num_heads"] == 4
    assert metadata["gps_query_condition_source"] == "projected_gps"
    assert metadata["attention_diagnostics"] is False
    assert metadata["optimizer_param_groups"] == optimizer_groups
    assert metadata["conditioned_encoders"]["image"]["context_feature_source"] == "projected"


def test_gps_query_downstream_forward_smoke_with_synthetic_image_gps():
    import_default_components()
    cfg = load_config(GPS_QUERY_DOWNSTREAM_CONFIGS["fair_gps_query_pooling"])
    model_cfg = _tiny_downstream_model_cfg(cfg)
    model = MODELS.build(model_cfg)
    model.eval()

    with torch.no_grad():
        output = model(
            image_batch=torch.randn(2, 2, 3, 32, 32),
            gps_batch=torch.randn(2, 2, 3),
        )

    assert output["logits"].shape == (2, 2, 7)
    assert set(output["encoder_features"]) == {"image", "gps"}
    assert set(output["modality_features"]) == {"image", "gps"}
    assert output["encoder_features"]["image"].shape == (2, 2, 16)
    assert output["modality_features"]["gps"].shape == (2, 2, 16)


def test_jepa_downstream_ablation_configs_do_not_reference_retired_paths():
    forbidden = (
        "HiST",
        "hist_beam",
        "teacher_no_kd",
        "student_no_kd",
        "no_kd",
        "logits_kd",
        "distillation",
        "legacy fusion",
    )

    for path in (*GPS_QUERY_DOWNSTREAM_CONFIGS.values(), PARAM_GROUP_DERIVED_CONFIG):
        text = path.read_text(encoding="utf-8")
        assert [snippet for snippet in forbidden if snippet in text] == []


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


def _tiny_downstream_model_cfg(cfg: dict) -> dict:
    primary = copy.deepcopy(cfg["model"]["primary"])
    primary.update(
        {
            "feature_size": 16,
            "d_model": 16,
            "num_classes": 7,
            "num_pred": 1,
            "gps_input_size": 3,
        }
    )
    image_encoder = primary["encoders"]["image"]
    image_encoder.update(
        {
            "checkpoint_path": "",
            "strict": False,
            "output_dim": 16,
            "latent_dim": 16,
            "image_channels": 3,
            "visual_encoder": {
                "image_channels": 3,
                "latent_dim": 16,
                "patch_size": 8,
                "depth": 0,
                "num_heads": 4,
                "max_tokens": 16,
                "dropout": 0.0,
            },
        }
    )
    if isinstance(image_encoder.get("gps_query_pool"), dict):
        image_encoder["gps_query_pool"].update({"condition_dim": 16, "latent_dim": 16})
    gps_encoder = primary.setdefault("encoders", {}).setdefault("gps", {"type": "gps_mlp"})
    gps_encoder.update({"output_dim": 16, "hidden_size": 16, "dropout": 0.0})
    primary["projectors"] = {
        "image": {"type": "linear", "d_model": 16, "dropout": 0.0},
        "gps": {"type": "linear", "d_model": 16, "dropout": 0.0},
    }
    core = primary["representation_core"]
    core["d_model"] = 16
    core["dropout"] = 0.0
    if core["type"] in {"early_concat_gru", "snapshot_frame"}:
        core["hidden_size"] = 16
        core["output_dim"] = 16
        core["num_layers"] = 1
    if core["type"] in {"token_transformer", "next_beam_query_transformer"}:
        core["num_heads"] = 4
        core["num_layers"] = 1
        core["max_seq_len"] = 2
    if core["type"] == "next_beam_query_transformer":
        core["output_dim"] = 16
    primary["heads"] = {"beam": {"type": "beam_head", "dropout": 0.0}}
    return primary
