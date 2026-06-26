import torch

from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.engine.prediction_objectives import (
    compute_prediction_loss,
    prepare_prediction_targets,
)
from kd_sensing.losses.amr_net import amr_net_loss_from_output
from kd_sensing.models.architecture_summary import summarize_model_architecture
from kd_sensing.registries import MODELS, RegistryError, import_default_components


def _model_cfg(**overrides):
    cfg = {
        "type": "amr_net",
        "modalities": ["image", "lidar", "gps"],
        "num_classes": 8,
        "num_pred": 1,
        "image_channels": 1,
        "image_feature_dim": 16,
        "lidar_input_features": 2,
        "lidar_feature_dim": 16,
        "gps_input_size": 2,
        "gps_feature_dim": 16,
        "latent_dim": 8,
        "dropout": 0.0,
    }
    cfg.update(overrides)
    return cfg


def _batch(batch_size=4):
    return {
        "image_batch": torch.randn(batch_size, 1, 1, 224, 224),
        "lidar_batch": torch.randn(batch_size, 1, 216, 2),
        "gps_batch": torch.randn(batch_size, 1, 2),
    }


def test_amr_net_registry_forward_adapt_cuaf_and_metadata():
    import_default_components()
    model = MODELS.build(_model_cfg())
    model.eval()

    first = adapt_model_output(model(**_batch()))
    second = adapt_model_output(model(**_batch()))

    assert model.supports_modality_kwargs is True
    assert first.logits.shape == (4, 1, 8)
    assert torch.isfinite(first.logits).all()
    assert set(first.diagnostics["modality_logits"]) == {"image", "lidar", "gps"}
    assert first.diagnostics["mu"]["image"].shape == (4, 8)
    weights = first.diagnostics["cuaf_weights"]
    assert weights.shape == (4, 1, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 1), atol=1e-5)
    assert torch.isfinite(first.diagnostics["cuaf_entropy"]).all()
    assert torch.isfinite(first.diagnostics["cuaf_kl_consistency"]).all()
    assert torch.isfinite(first.diagnostics["cuaf_topk_margin"]).all()
    assert torch.allclose(first.logits, second.logits, atol=0.0) is False

    metadata = model.training_strategy_metadata()
    assert metadata["architecture_category"] == "whole_model_exception"
    assert metadata["registry_type"] == "amr_net"
    assert metadata["modalities"] == ["image", "lidar", "gps"]
    assert metadata["latent_dim"] == 8
    assert metadata["cuaf_enabled"] is True
    assert metadata["consumes_reliability_metadata"] is False
    assert metadata["paper_approximation"] is True


def test_amr_net_eval_is_deterministic_for_same_input():
    import_default_components()
    model = MODELS.build(_model_cfg())
    model.eval()
    batch = _batch()

    first = adapt_model_output(model(**batch)).logits
    second = adapt_model_output(model(**batch)).logits

    assert torch.allclose(first, second)


def test_amr_net_rejects_non_snapshot_time_dimension():
    import_default_components()
    model = MODELS.build(_model_cfg())
    batch = _batch()
    batch["gps_batch"] = torch.randn(4, 2, 2)

    try:
        model(**batch)
    except ValueError as exc:
        assert "gps" in str(exc)
        assert "snapshot T=1" in str(exc)
    else:
        raise AssertionError("expected AMR-Net to reject T != 1")


def test_amr_net_loss_helper_and_prediction_objective_opt_in():
    import_default_components()
    model = MODELS.build(_model_cfg())
    model.train()
    output = adapt_model_output(model(**_batch()))
    labels = torch.tensor([[0], [0], [1], [1]])
    cfg = {"loss": {"amr": {"enabled": True, "alpha": 0.01, "beta": 0.5, "weight": 0.25}}}

    loss, diagnostics = amr_net_loss_from_output(output, labels, cfg)
    targets = prepare_prediction_targets(labels=labels, auxiliary_targets={}, cfg=cfg)
    bundle = compute_prediction_loss(
        output,
        targets,
        cfg,
        reference=output.logits,
        beam_total_loss=output.logits.sum() * 0.0,
        beam_task_loss=output.logits.sum() * 0.0,
    )

    assert torch.isfinite(loss)
    assert diagnostics["loss/amr_total"] > 0.0
    assert "loss/amr_total" in bundle.diagnostics
    assert torch.isfinite(bundle.total)


def test_amr_net_pre_loss_skips_no_positive_batch_without_nan():
    import_default_components()
    model = MODELS.build(_model_cfg())
    output = adapt_model_output(model(**_batch()))
    labels = torch.tensor([[0], [1], [2], [3]])

    loss, diagnostics = amr_net_loss_from_output(output, labels, {"loss": {"amr": {"enabled": True}}})

    assert torch.isfinite(loss)
    assert diagnostics["loss/amr_pre"] == 0.0
    assert diagnostics["amr/pre_skipped_anchors"] > 0.0


def test_amr_net_architecture_summary_and_old_name_guard():
    import_default_components()
    model = MODELS.build(_model_cfg())
    summary = summarize_model_architecture(model)

    assert summary["model"]["registry_type"] == "amr_net"
    assert summary["model"]["architecture_category"] == "whole_model_exception"
    assert summary["parameters"]["total_params"] > 0
    assert summary["parameters"]["trainable_params"] > 0
    assert summary["model"]["metadata"]["latent_dim"] == 8
    try:
        MODELS.build({"type": "amr_net_gps_image"})
    except RegistryError as exc:
        assert "amr_net_gps_image" in str(exc)
        assert "amr_net" in str(exc)
    else:
        raise AssertionError("old AMR-Net registry token should not build")
