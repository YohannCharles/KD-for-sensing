from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.config.io import load_config
from kd_sensing.data.missing_mask import (
    apply_modality_corruption,
    make_pattern_mask,
    sample_missing_mask,
    sample_pattern_balanced_mask,
)
from kd_sensing.engine.model_output import adapt_model_output
from kd_sensing.engine.trainer import _build_training_extensions
from kd_sensing.losses.beam_prototype_alignment import (
    BeamPrototypeBank,
    make_soft_beam_labels,
    prototype_alignment_loss,
    supervised_contrastive_loss,
)
from kd_sensing.losses.u_mask_beam_jepa import u_mask_beam_jepa_config, u_mask_beam_jepa_loss
from kd_sensing.models.architecture_summary import summarize_model_architecture
from kd_sensing.registries import ENCODERS, MODELS, RegistryError, import_default_components


ROOT = Path(__file__).resolve().parents[1]


@ENCODERS.register("u_mask_test_encoder", force=True)
class UMaskTestEncoder(nn.Module):
    def __init__(self, output_dim: int = 16, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = batch.shape[:2]
        pooled = batch.float().reshape(batch_size, seq_len, -1).mean(dim=-1, keepdim=True)
        return pooled.expand(batch_size, seq_len, self.output_dim)


def _cfg(**overrides):
    cfg = {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar", "lidar", "gps"],
        "d_model": 16,
        "num_classes": 8,
        "num_pred": 1,
        "num_heads": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "encoders": {
            "image": {"type": "u_mask_test_encoder", "output_dim": 16},
            "radar": {"type": "u_mask_test_encoder", "output_dim": 16},
            "lidar": {"type": "u_mask_test_encoder", "output_dim": 16},
            "gps": {"type": "u_mask_test_encoder", "output_dim": 16},
        },
    }
    cfg.update(overrides)
    return cfg


def _batch(batch_size=2):
    return {
        "image_batch": torch.randn(batch_size, 2, 3, 8, 8),
        "radar_batch": torch.randn(batch_size, 2, 2, 6, 6),
        "lidar_batch": torch.randn(batch_size, 2, 3, 6, 6),
        "gps_batch": torch.randn(batch_size, 2, 3),
    }


def test_registry_forward_adapter_metadata_and_zero_mask_rejection():
    import_default_components()
    model = MODELS.build(_cfg())
    metadata = model.training_strategy_metadata()
    assert metadata["architecture_category"] == "whole_model_exception"
    assert metadata["enabled_modalities"] == ["image", "radar", "lidar", "gps"]
    assert metadata["consumes_missing_mask"] is True

    mask = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=torch.bool)
    output = model(**_batch(), missing_mask=mask)
    adapted = adapt_model_output(output)

    assert adapted.logits.shape == (2, 1, 8)
    assert adapted.output_features.shape == (2, 16)
    for key in (
        "teacher_logits",
        "u_star",
        "mu_B",
        "logvar_B",
        "modality_mu_B",
        "modality_logvar_B",
        "modality_reliability",
        "global_reliability",
    ):
        assert key in adapted.diagnostics
    assert adapted.diagnostics["modality_mu_B"].shape == (2, 4, 16)
    assert adapted.diagnostics["modality_logvar_B"].shape == (2, 4, 16)
    assert adapted.diagnostics["modality_reliability"].shape == (2, 4, 1)
    expected_reliability = torch.exp(-F.softplus(adapted.diagnostics["modality_logvar_B"]).mean(dim=-1, keepdim=True))
    expected_reliability = expected_reliability * mask.unsqueeze(-1)
    assert torch.allclose(adapted.diagnostics["modality_reliability"], expected_reliability)
    assert torch.all(adapted.diagnostics["modality_reliability"][~mask] == 0)
    with pytest.raises(ValueError, match="no available modalities"):
        model(**_batch(), missing_mask=torch.zeros(2, 4, dtype=torch.bool))


def test_registry_encoder_warm_start_forward(tmp_path):
    import_default_components()
    source = MODELS.build(
        _cfg(
            modalities=["gps"],
            d_model=8,
            num_heads=4,
            fusion_type="weighted_sum",
            use_jepa_loss=False,
            encoders={"gps": {"type": "gps_mlp", "output_dim": 8, "hidden_size": 8}},
        )
    )
    checkpoint = tmp_path / "gps_encoder.pth"
    torch.save({"state_dict": {f"encoders.gps.{k}": v for k, v in source.encoders["gps"].state_dict().items()}}, checkpoint)

    model = MODELS.build(
        _cfg(
            modalities=["gps"],
            d_model=8,
            num_heads=4,
            fusion_type="weighted_sum",
            use_jepa_loss=False,
            encoders={"gps": {"type": "gps_mlp", "output_dim": 8, "hidden_size": 8}},
            encoder_checkpoint_paths={"gps": str(checkpoint)},
        )
    )
    output = model(gps_batch=torch.randn(2, 3, 3), missing_mask=torch.ones(2, 1, dtype=torch.bool))

    assert output["logits"].shape == (2, 1, 8)
    assert model.training_strategy_metadata()["use_registry_encoders"] is True
    assert model.training_strategy_metadata()["encoder_checkpoint_loads"]["gps"]["loaded_keys"] > 0


@pytest.mark.parametrize(
    "bad",
    [
        [],
        ["image", "image"],
        ["vision", "gps"],
        ["image", "mmwave"],
    ],
)
def test_modalities_are_canonical_for_u_mask_beam_jepa(bad):
    import_default_components()
    with pytest.raises((ValueError, RegistryError), match="modalit|canonical|invalid|Invalid"):
        MODELS.build(_cfg(modalities=bad))


def test_loss_backward_and_ablation_disable_teacher_jepa_uncertainty():
    import_default_components()
    model = MODELS.build(
        _cfg(
            use_teacher=False,
            use_jepa_loss=False,
            use_modality_uncertainty=False,
            use_global_uncertainty=False,
            fusion_type="weighted_sum",
        )
    )
    output = model(**_batch(), missing_mask=torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool))
    labels = torch.tensor([[1], [3]])
    result = u_mask_beam_jepa_loss(
        output,
        labels,
        use_teacher=False,
        use_jepa_loss=False,
    )

    result["loss"].backward()

    assert result["loss_beam"].requires_grad
    assert float(result["loss_teacher"].detach()) == 0.0
    assert float(result["loss_jepa_global"].detach()) == 0.0
    assert float(result["loss_modality_nll"].detach()) == 0.0
    assert "loss_beam" in result["diagnostics"]
    assert "loss_jepa_global" in result["diagnostics"]
    assert "loss_modality_nll" in result["diagnostics"]
    assert "top1_acc" in result["diagnostics"]
    assert "top5_acc" in result["diagnostics"]
    assert model.beam_head.net[-1].weight.grad is not None
    assert torch.all(output["modality_reliability"][output["missing_mask"]] == 1)
    assert torch.all(output["global_reliability"] == 1)


@pytest.mark.parametrize(
    "name,mask",
    [
        ("full", torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)),
        ("missing_gps", torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)),
        ("non_gps_only", torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)),
        ("only_gps", torch.tensor([[0, 0, 0, 1]], dtype=torch.bool)),
    ],
)
def test_rbma_attention_masks_patterns_and_diagnostics(name, mask):
    import_default_components()
    model = MODELS.build(_cfg(fusion_type="reliability_biased_missing_attention", use_jepa_loss=True, dropout=0.0))
    output = model(**_batch(batch_size=1), missing_mask=mask)

    assert output["logits"].shape == (1, 1, 8)
    weights = output["rbma_attention_weights"]
    assert torch.isfinite(output["rbma_reliability_log_bias"]).all()
    assert output["rbma_reliability_log_finite"] is True
    missing = ~mask[0]
    if missing.any():
        assert torch.allclose(weights[0, :, :4][:, missing], torch.zeros_like(weights[0, :, :4][:, missing]))
    assert output["rbma_mask_provenance"] == "missing_mask"


def test_rbma_all_missing_requires_or_uses_global_token():
    import_default_components()
    no_jepa = MODELS.build(_cfg(fusion_type="reliability_biased_missing_attention", use_jepa_loss=False, dropout=0.0))
    with pytest.raises(ValueError, match="global token|available modality"):
        no_jepa(**_batch(batch_size=1), missing_mask=torch.zeros(1, 4, dtype=torch.bool))

    with_jepa = MODELS.build(_cfg(fusion_type="reliability_biased_missing_attention", use_jepa_loss=True, dropout=0.0))
    output = with_jepa(**_batch(batch_size=1), missing_mask=torch.zeros(1, 4, dtype=torch.bool))
    assert torch.isfinite(output["output_features"]).all()
    assert "rbma_global_attention_mean" in output


def test_modality_uncertainty_loss_uses_available_modalities_only():
    logits = torch.tensor([[[0.0, 2.0]]], requires_grad=True)
    modality_mu = torch.tensor([[[1.0, 1.0], [100.0, 100.0]]], requires_grad=True)
    output = {
        "logits": logits,
        "teacher_logits": logits.detach(),
        "u_star": torch.tensor([[1.0, 3.0]]),
        "mu_B": torch.tensor([[0.0, 0.0]], requires_grad=True),
        "logvar_B": torch.zeros(1, 2),
        "modality_mu_B": modality_mu,
        "modality_logvar_B": torch.zeros(1, 2, 2),
        "missing_mask": torch.tensor([[True, False]]),
    }

    result = u_mask_beam_jepa_loss(
        output,
        torch.tensor([[1]]),
        lambda_teacher=0.0,
        lambda_jepa_global=0.0,
        lambda_modality_nll=1.0,
    )
    result["loss"].backward()

    assert torch.allclose(result["loss_modality_nll"], torch.tensor(1.0))
    assert modality_mu.grad is not None
    assert torch.all(modality_mu.grad[:, 1, :] == 0)


def test_u_mask_beam_jepa_loss_random_outputs_backward_global_and_modality_terms():
    batch_size, modalities, d_model, num_classes = 3, 4, 5, 7
    output = {
        "logits": torch.randn(batch_size, 1, num_classes, requires_grad=True),
        "teacher_logits": torch.randn(batch_size, 1, num_classes, requires_grad=True),
        "u_star": torch.randn(batch_size, d_model),
        "mu_B": torch.randn(batch_size, d_model, requires_grad=True),
        "logvar_B": torch.randn(batch_size, d_model, requires_grad=True),
        "modality_mu_B": torch.randn(batch_size, modalities, d_model, requires_grad=True),
        "modality_logvar_B": torch.randn(batch_size, modalities, d_model, requires_grad=True),
        "missing_mask": torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]], dtype=torch.bool),
        "modality_reliability": torch.rand(batch_size, modalities, 1),
        "global_reliability": torch.rand(batch_size),
    }
    result = u_mask_beam_jepa_loss(
        output,
        torch.tensor([[1], [3], [5]]),
        lambda_teacher=0.5,
        lambda_jepa_global=1.0,
        lambda_modality_nll=0.2,
    )

    result["loss"].backward()

    assert result["loss"].requires_grad
    assert result["loss_jepa_global"].requires_grad
    assert result["loss_modality_nll"].requires_grad
    assert output["mu_B"].grad is not None
    assert output["modality_mu_B"].grad is not None


def test_mask_helper_pattern_and_corruption_are_non_inplace():
    generator = torch.Generator().manual_seed(0)
    mask = sample_missing_mask(
        8,
        4,
        [1.0, 1.0, 1.0, 1.0],
        always_available_indices=[2],
        generator=generator,
    )
    assert mask.shape == (8, 4)
    assert mask.dtype is torch.bool
    assert torch.all(mask[:, 2])
    assert torch.all(mask.any(dim=1))

    pattern = make_pattern_mask(2, ["image", "radar", "lidar", "gps"], available_modalities=["image", "gps"])
    assert pattern.tolist() == [[True, False, False, True], [True, False, False, True]]
    with pytest.raises(ValueError, match="at least one"):
        make_pattern_mask(1, ["image"], pattern_mask=[0])

    batch = {"image": torch.ones(2, 1, 3, 2, 2), "gps": torch.ones(2, 1, 3)}
    corrupted = apply_modality_corruption(batch, {"image": {"zero_out": True}, "gps": {"gaussian_noise_std": 0.1}})
    assert torch.all(batch["image"] == 1)
    assert torch.all(corrupted["image"] == 0)
    assert not torch.equal(corrupted["gps"], batch["gps"])


def test_pattern_balanced_sampler_distribution_and_canonical_names():
    generator = torch.Generator().manual_seed(0)
    probs = {"full": 0.7, "missing_gps": 0.2, "non_gps_only": 0.1}
    mask, names, ids = sample_pattern_balanced_mask(
        1000,
        ["image", "radar", "lidar", "gps"],
        probs,
        generator=generator,
    )

    assert mask.shape == (1000, 4)
    assert ids is not None
    assert torch.all(mask.any(dim=1))
    assert names.count("full") / 1000 == pytest.approx(0.7, abs=0.06)
    assert names.count("missing_gps") / 1000 == pytest.approx(0.2, abs=0.05)
    missing_gps = mask[names.index("missing_gps")]
    non_gps = mask[names.index("non_gps_only")]
    assert missing_gps.tolist() == [True, True, True, False]
    assert non_gps.tolist() == [True, True, True, False]
    with pytest.raises(ValueError, match="canonical"):
        sample_pattern_balanced_mask(1, ["vision", "radar", "lidar", "gps"], {"full": 1.0})


def test_beam_prototype_alignment_forward_backward_and_safety():
    bank = BeamPrototypeBank(6, 4, temperature=0.5)
    features = torch.randn(3, 6, requires_grad=True)
    logits = bank(features)
    assert logits.shape == (3, 4)

    labels = torch.tensor([0, 3, 1])
    target = make_soft_beam_labels(labels, 4, 1.0, circular=True)
    assert torch.allclose(target.sum(dim=1), torch.ones(3))
    assert target[0, 3] > target[0, 2]
    modality_features = torch.randn(3, 4, 6, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 1], [0, 0, 0, 1], [1, 1, 1, 0]], dtype=torch.bool)
    loss, diagnostics = prototype_alignment_loss(
        bank,
        labels,
        fused_features=features,
        modality_features=modality_features,
        mask=mask,
        lambda_proto=1.0,
        lambda_modality_proto=0.5,
    )
    supcon, supcon_diag = supervised_contrastive_loss(features, torch.tensor([0, 1, 2]))
    total = loss + supcon
    total.backward()

    assert torch.isfinite(total)
    assert features.grad is not None
    assert diagnostics["prototype/top5"] >= 0.0
    assert diagnostics["prototype/modality_sample_count"] == pytest.approx(float(mask.sum()))
    assert supcon_diag["prototype/supcon_anchor_count"] == 0.0


def test_config_overlays_training_extension_and_architecture_summary():
    cfg = load_config(ROOT / "configs/fusion/u_mask_beam_jepa_smoke.yaml")
    assert cfg["evaluation"]["missing_pattern"]["available_modalities"] == ["image", "gps"]
    assert cfg["loss"]["u_mask_beam_jepa"]["missing_mask"]["p_missing"] == [0.25, 0.25, 0.25, 0.1]
    extensions = _build_training_extensions(cfg)
    assert [extension.name for extension in extensions] == ["u_mask_beam_jepa"]

    for name in ("no_jepa", "no_uncertainty", "concat_mlp", "weighted_sum"):
        loaded = load_config(ROOT / f"configs/fusion/u_mask_beam_jepa_{name}.yaml")
        assert loaded["model"]["primary"]["type"] == "u_mask_beam_jepa"
    s32 = load_config(ROOT / "configs/fusion/u_mask_beam_jepa_s32.yaml")
    assert s32["data"]["dataset"]["scene"] == 32
    assert s32["loss"]["u_mask_beam_jepa"]["lambda_jepa_global"] == 1.0
    assert s32["loss"]["u_mask_beam_jepa"]["lambda_modality_nll"] == 0.2
    assert s32["loss"]["u_mask_beam_jepa"]["missing_mask"]["p_missing"] == 0.5
    assert s32["loss"]["u_mask_beam_jepa"]["missing_mask"]["ensure_at_least_one"] is True
    assert s32["loss"]["u_mask_beam_jepa"]["missing_mask"]["always_available_indices"] == []

    import_default_components()
    model = MODELS.build(
        _cfg(
            fusion_type="reliability_biased_missing_attention",
            use_beam_prototype_alignment=True,
            use_full_to_partial_kd=True,
            kd_teacher_mode="online_full",
            mask_sampler="pattern_balanced",
        )
    )
    summary = summarize_model_architecture(model)
    metadata = model.training_strategy_metadata()
    assert summary["model"]["architecture_category"] == "whole_model_exception"
    assert summary["model"]["enabled_modalities"] == ["image", "radar", "lidar", "gps"]
    assert metadata["fusion_type"] == "reliability_biased_missing_attention"
    assert metadata["mask_sampler"] == "pattern_balanced"
    assert metadata["use_beam_prototype_alignment"] is True
    assert metadata["use_full_to_partial_kd"] is True
    assert metadata["kd_teacher_mode"] == "online_full"
    assert metadata["consumes_reliability_metadata"] is True
    strong = load_config(ROOT / "configs/fusion/experiments/rbma_missing_workflow_strong_encoders/weighted_sum_mask.yaml")
    assert strong["model"]["primary"]["encoders"]["gps"]["type"] == "gps_mlp"
    assert "gps" in strong["model"]["primary"]["encoder_checkpoint_paths"]


def test_rbma_ablation_configs_load_and_set_current_flags():
    base = ROOT / "configs/fusion/experiments/rbma_missing_workflow"
    main = load_config(base / "no_jepa_rbma_proto_kd.yaml")
    primary = main["model"]["primary"]
    assert set(primary["modalities"]) == {"image", "radar", "lidar", "gps"}
    assert primary["fusion_type"] == "reliability_biased_missing_attention"
    assert primary["use_jepa_loss"] is False
    assert main["training"]["mask_sampler"] == "pattern_balanced"
    assert main["training"]["use_beam_prototype_alignment"] is True
    assert main["training"]["use_full_to_partial_kd"] is True
    assert main["training"]["kd_teacher_mode"] == "online_full"

    baseline = load_config(base / "amber_style_mask_baseline.yaml")
    assert baseline["model"]["primary"]["fusion_type"] == "weighted_sum"
    assert baseline["training"].get("use_beam_prototype_alignment", False) is False
    assert baseline["training"].get("use_full_to_partial_kd", False) is False

    for name in (
        "no_jepa_rbma.yaml",
        "no_jepa_rbma_proto.yaml",
        "no_jepa_rbma_kd.yaml",
        "jepa_small_lambda_rbma_proto_kd.yaml",
        "proto_only_baseline.yaml",
    ):
        cfg = load_config(base / name)
        assert set(cfg["model"]["primary"]["modalities"]) == {"image", "radar", "lidar", "gps"}
        assert "vision" not in (base / name).read_text(encoding="utf-8")


def test_u_mask_beam_jepa_missing_alias_warns_and_missing_mask_wins():
    with pytest.warns(UserWarning, match="ignored because missing_mask is set"):
        both = u_mask_beam_jepa_config(
            {
                "loss": {
                    "u_mask_beam_jepa": {
                        "enabled": True,
                        "missing": {"p_missing": 0.9},
                        "missing_mask": {"p_missing": 0.5},
                    }
                }
            }
        )
    assert both["missing_mask"]["p_missing"] == 0.5

    with pytest.warns(UserWarning, match="deprecated"):
        legacy = u_mask_beam_jepa_config({"loss": {"u_mask_beam_jepa": {"missing": {"p_missing": 0.4}}}})
    assert legacy["missing_mask"]["p_missing"] == 0.4


def test_unknown_context_or_fusion_type_errors_are_clear():
    import_default_components()
    with pytest.raises((ValueError, RegistryError), match="mask_transformer"):
        MODELS.build(_cfg(context_type="mask_transformer"))
    with pytest.raises((ValueError, RegistryError), match="fusion_type"):
        MODELS.build(_cfg(fusion_type="mystery"))


def test_no_jepa_online_kd_loss_detaches_teacher_and_checkpoint_guard():
    import_default_components()
    model = MODELS.build(
        _cfg(
            fusion_type="reliability_biased_missing_attention",
            use_jepa_loss=False,
            use_beam_prototype_alignment=True,
            use_full_to_partial_kd=True,
            kd_teacher_mode="online_full",
        )
    )
    student = model(**_batch(), missing_mask=torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.bool))
    with torch.no_grad():
        teacher = model(**_batch(), missing_mask=torch.ones(2, 4, dtype=torch.bool))
    result = u_mask_beam_jepa_loss(
        student,
        torch.tensor([[1], [3]]),
        use_teacher=False,
        use_jepa_loss=False,
        teacher_output={"logits": teacher["logits"].detach(), "output_features": teacher["output_features"].detach()},
        prototype_bank=model.prototype_bank,
        use_beam_prototype_alignment=True,
        lambda_proto=0.1,
        lambda_modality_proto=0.1,
        use_full_to_partial_kd=True,
        lambda_full_to_partial_kd=0.5,
        lambda_feature_kd=0.1,
    )
    result["loss"].backward()

    assert model.beam_head.net[-1].weight.grad is not None
    assert "loss/full_to_partial_kd" in result["diagnostics"]
    assert "teacher_top1" in result["diagnostics"]
    cfg = {
        "model": {"primary": {"use_jepa_loss": False}},
        "training": {"kd_teacher_mode": "checkpoint", "use_full_to_partial_kd": True},
        "loss": {"u_mask_beam_jepa": {"enabled": True}},
    }
    with pytest.raises(NotImplementedError, match="checkpoint"):
        _build_training_extensions(cfg)[0].setup(
            type(
                "Context",
                (),
                {
                    "cfg": cfg,
                },
            )()
        )
