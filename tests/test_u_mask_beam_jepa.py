from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kd_sensing.config import load_config
from kd_sensing.engine.optim import build_optimizer
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.losses.u_mask_beam_jepa import (
    _paired_router_quality_loss,
    _router_oracle_targets,
    _unimodal_normalized_utility,
    u_mask_beam_jepa_loss,
)
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config
from kd_sensing.models.gps import GpsFeatureExtractor
from kd_sensing.registries import ENCODERS, MODELS

import kd_sensing.models.modular  # noqa: F401
import kd_sensing.models.u_mask_beam_jepa  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


@ENCODERS.register("u_mask_test_sequence", force=True)
class _SequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 4, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps = values.shape[:2]
        scalar = values.float().reshape(batch, steps, -1).mean(dim=-1, keepdim=True)
        return scalar.expand(-1, -1, self.output_dim)


def _model_config(*, head_type: str = "prototype") -> dict[str, object]:
    return {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar"],
        "d_model": 4,
        "num_classes": 4,
        "num_pred": 1,
        "dropout": 0.0,
        "fusion_type": "supervised_router",
        "head_type": head_type,
        "temporal_pooling": {"enabled": True, "type": "masked_mean"},
        "encoders": {
            "image": {"type": "u_mask_test_sequence", "output_dim": 4},
            "radar": {"type": "u_mask_test_sequence", "output_dim": 4},
        },
    }


def _loss_output(classes: int = 4, features: int = 5) -> dict[str, torch.Tensor]:
    return {
        "logits": torch.randn(3, 1, classes, requires_grad=True),
        "output_features": torch.randn(3, features, requires_grad=True),
        "modality_features": torch.randn(3, 2, features, requires_grad=True),
        "missing_mask": torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.bool),
    }


def test_t2_and_s1_resolve_to_the_retained_surface() -> None:
    t2 = u_mask_beam_jepa_config(load_config(ROOT / "configs/mmw/t2.yaml"))
    s1 = u_mask_beam_jepa_config(load_config(ROOT / "configs/mmw/s1.yaml"))

    assert t2["enabled"] is True
    assert t2["use_beam_prototype_alignment"] is True
    assert t2["superset_consistency"]["enabled"] is True
    assert t2["superset_consistency"]["confidence_gated_kl"] is True
    assert t2["router_supervision"] == "oracle"
    assert t2["router_oracle_weight"] == pytest.approx(0.1)
    assert t2["router_oracle_target_mode"] == "hard_first"
    assert t2["router_oracle_temperature"] == pytest.approx(1.0)
    assert t2["missing_mask"] == {"mode": "external"}
    assert s1["missing_mask"] == {"mode": "external"}
    assert s1["superset_consistency"]["enabled"] is False

    deepsense = u_mask_beam_jepa_config(load_config(ROOT / "configs/deepsense6g/t2.yaml"))
    assert deepsense["missing_mask"]["mode"] == "random"
    assert deepsense["missing_mask"]["p_missing"] == [0.25, 0.25, 0.25, 0.1]


def test_external_missing_mask_rejects_random_fields() -> None:
    with pytest.raises(ValueError, match="must not declare random fields"):
        u_mask_beam_jepa_config(
            {"loss": {"u_mask_beam_jepa": {"missing_mask": {"mode": "external", "p_missing": 0.1}}}}
        )


def test_masked_mean_supervised_router_masks_missing_cells() -> None:
    model = MODELS.build(_model_config())
    output = model(
        image_batch=torch.tensor([[[1.0], [3.0], [5.0]]]),
        radar_batch=torch.tensor([[[2.0], [4.0], [6.0]]]),
        modality_temporal_mask=torch.tensor([[[1, 1], [1, 0], [0, 0]]], dtype=torch.bool),
        missing_mask=torch.tensor([[True, True]]),
    )

    assert torch.equal(output["input_features"][0, 0], torch.full((4,), 2.0))
    assert torch.equal(output["input_features"][0, 1], torch.full((4,), 2.0))
    assert output["logits"].shape == (1, 1, 4)
    assert output["unimodal_logits"].shape == (1, 2, 4)
    assert torch.allclose(output["supervised_router_gate_weights"].sum(dim=1), torch.ones(1))
    assert output["metadata"]["temporal_pooling_type"] == "masked_mean"
    assert output["metadata"]["fusion_type"] == "supervised_router"


def test_reliability_mean_ignores_temporally_empty_modalities_and_keeps_router_diagnostics() -> None:
    config = _model_config()
    config["fusion_type"] = "reliability_mean"
    model = MODELS.build(config)
    output = model(
        image_batch=torch.tensor([[[1.0], [3.0], [5.0]]]),
        radar_batch=torch.tensor([[[2.0], [4.0], [6.0]]]),
        modality_temporal_mask=torch.tensor([[[1, 0], [1, 0], [0, 0]]], dtype=torch.bool),
        missing_mask=torch.tensor([[True, True]]),
    )

    weights = output["reliability_fusion_weights"]
    assert output["reliability_fusion_mode"] == "reliability_mean"
    assert torch.allclose(weights.sum(dim=1), torch.ones(1))
    assert torch.allclose(weights[0, 1], torch.tensor(0.0))
    assert torch.is_tensor(output["router_gate_logits"])
    assert torch.is_tensor(output["supervised_router_gate_weights"])
    assert output["router_gate_logits"].requires_grad is False
    assert all(not parameter.requires_grad for parameter in model.supervised_router.parameters())

    result = u_mask_beam_jepa_loss(output, torch.tensor([[0]]), router_oracle_weight=0.0)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert result["diagnostics"]["router_oracle_disabled"] == 1.0


def test_uniform_mean_is_a_router_and_reliability_free_control() -> None:
    config = _model_config()
    config["fusion_type"] = "uniform_mean"
    model = MODELS.build(config)
    output = model(
        image_batch=torch.tensor([[[1.0], [3.0], [5.0]]]),
        radar_batch=torch.tensor([[[2.0], [4.0], [6.0]]]),
        modality_temporal_mask=torch.tensor([[[1, 1], [1, 1], [0, 0]]], dtype=torch.bool),
        missing_mask=torch.tensor([[True, True]]),
    )

    assert output["reliability_fusion_mode"] == "uniform_mean"
    assert torch.allclose(output["reliability_fusion_weights"], torch.tensor([[0.5, 0.5]]))
    assert all(not parameter.requires_grad for parameter in model.supervised_router.parameters())
    assert all(not parameter.requires_grad for parameter in model.reliability_heads.parameters())
    with pytest.raises(ValueError, match="uniform_mean fusion requires router_oracle_weight=0"):
        u_mask_beam_jepa_config(
            {
                "model": {"primary": {"fusion_type": "uniform_mean"}},
                "loss": {"u_mask_beam_jepa": {"router_oracle_weight": 0.1}},
            }
        )


def test_inactive_t2_head_is_frozen_but_checkpoint_compatible() -> None:
    classifier = MODELS.build(_model_config(head_type="classifier"))
    optimizer = build_optimizer({"training": {"lr": 1.0e-3}}, classifier)
    optimizer_parameters = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

    assert all(not parameter.requires_grad for parameter in classifier.prototype_bank.parameters())
    assert all(parameter.requires_grad for parameter in classifier.classifier.parameters())
    assert not {id(parameter) for parameter in classifier.prototype_bank.parameters()} & optimizer_parameters
    assert "prototype_bank.prototypes" in classifier.state_dict()
    metadata = classifier.training_strategy_metadata()
    assert metadata["active_head"] == "classifier"
    assert "prototype_bank" in metadata["frozen_branches"]

    prototype = MODELS.build(_model_config(head_type="prototype"))
    assert all(not parameter.requires_grad for parameter in prototype.classifier.parameters())
    assert all(parameter.requires_grad for parameter in prototype.prototype_bank.parameters())


def test_zero_router_oracle_weight_does_not_invoke_oracle_loss(monkeypatch) -> None:
    def unexpected_oracle(*_args, **_kwargs):
        raise AssertionError("router oracle loss must be skipped at zero weight")

    monkeypatch.setitem(u_mask_beam_jepa_loss.__globals__, "_router_oracle_loss", unexpected_oracle)
    result = u_mask_beam_jepa_loss(_loss_output(), torch.tensor([[0], [1], [2]]), router_oracle_weight=0.0)

    assert result["diagnostics"]["router_oracle_enabled"] == 0.0
    assert result["diagnostics"]["router_oracle_disabled"] == 1.0


def test_tie_aware_router_oracle_targets_avoid_first_modality_bias() -> None:
    unimodal = torch.tensor(
        [
            [
                [0.0, 2.0, 0.0, 0.0],
                [1.5, 0.0, 0.0, 2.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.5, 2.0, 0.0, 0.0],
            ]
        ]
    )
    target = torch.tensor([0])
    available = torch.ones(1, 4, dtype=torch.bool)

    hard_first, tied = _router_oracle_targets(
        unimodal, target, available, circular_beam_distance=True, target_mode="hard_first", temperature=1.0
    )
    hard_confidence, _ = _router_oracle_targets(
        unimodal,
        target,
        available,
        circular_beam_distance=True,
        target_mode="hard_confidence_tie",
        temperature=1.0,
    )
    soft_uniform, _ = _router_oracle_targets(
        unimodal,
        target,
        available,
        circular_beam_distance=True,
        target_mode="soft_uniform_tie",
        temperature=1.0,
    )
    soft_confidence, _ = _router_oracle_targets(
        unimodal,
        target,
        available,
        circular_beam_distance=True,
        target_mode="soft_confidence_tie",
        temperature=1.0,
    )

    assert tied.item() is True
    assert torch.equal(hard_first, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(hard_confidence, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    assert torch.allclose(soft_uniform, torch.tensor([[1 / 3, 1 / 3, 0.0, 1 / 3]]))
    assert soft_confidence[0, 1] > soft_confidence[0, 3] > soft_confidence[0, 0]
    assert torch.allclose(soft_confidence.sum(dim=1), torch.ones(1))


def test_distance_soft_router_oracle_masks_missing_modalities_and_backpropagates() -> None:
    output = _loss_output()
    output["router_gate_logits"] = torch.randn(3, 2, requires_grad=True)
    output["unimodal_logits"] = torch.randn(3, 2, 4, requires_grad=True)
    result = u_mask_beam_jepa_loss(
        output,
        torch.tensor([[0], [1], [2]]),
        router_oracle_target_mode="distance_confidence_soft",
        router_oracle_temperature=1.0,
    )
    result["loss"].backward()

    assert torch.isfinite(result["loss"])
    assert output["router_gate_logits"].grad is not None
    assert torch.isfinite(output["router_gate_logits"].grad).all()
    assert result["diagnostics"]["router_oracle_target_entropy"] >= 0.0


def test_router_oracle_config_rejects_unknown_mode_and_temperature() -> None:
    with pytest.raises(ValueError, match="router_oracle_target_mode"):
        u_mask_beam_jepa_config(
            {"loss": {"u_mask_beam_jepa": {"router_oracle_target_mode": "unknown"}}}
        )


def test_beam_power_soft_router_target_uses_normalized_selected_power() -> None:
    unimodal = torch.tensor([[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]])
    powers = torch.tensor([[1.0, 3.0, 2.0]])
    target, tied = _router_oracle_targets(
        unimodal,
        torch.tensor([1]),
        torch.tensor([[True, True]]),
        circular_beam_distance=True,
        target_mode="beam_power_soft",
        temperature=0.1,
        beam_powers=powers,
    )
    expected = torch.softmax(torch.tensor([[1.0 / 3.0, 1.0]]) / 0.1, dim=1)
    assert torch.allclose(target, expected)
    assert not bool(tied.item())


def test_expected_beam_power_utility_changes_without_argmax_change() -> None:
    clean = torch.tensor([[[4.0, 3.0, 0.0], [0.0, 4.0, 3.0]]])
    corrupted = torch.tensor([[[4.0, 0.0, 3.0], [0.0, 4.0, 3.0]]])
    powers = torch.tensor([[4.0, 3.0, 0.0]])

    clean_utility = _unimodal_normalized_utility(
        clean, powers, mode="expected", beam_temperature=0.5
    )
    corrupted_utility = _unimodal_normalized_utility(
        corrupted, powers, mode="expected", beam_temperature=0.5
    )

    assert torch.equal(clean.argmax(dim=-1), corrupted.argmax(dim=-1))
    assert clean_utility[0, 0] > corrupted_utility[0, 0]
    assert clean_utility[0, 1] == pytest.approx(corrupted_utility[0, 1])


def test_router_quality_pairing_updates_corrupt_router_and_gates_monotonic_loss() -> None:
    corrupt_logits = torch.tensor([[3.0, 0.0], [0.0, 0.0]], requires_grad=True)
    corrupt_weights = torch.softmax(corrupt_logits, dim=1)
    pair = {
        "clean_unimodal_logits": torch.tensor([[[9.0, 0.0], [0.0, 9.0]], [[9.0, 0.0], [0.0, 9.0]]]),
        "corrupted_unimodal_logits": torch.tensor([[[0.0, 9.0], [0.0, 9.0]], [[9.0, 0.0], [0.0, 9.0]]]),
        "available": torch.ones(2, 2, dtype=torch.bool),
        "clean_router_logits": torch.zeros(2, 2),
        "corrupted_router_logits": corrupt_logits,
        "clean_router_weights": torch.tensor([[0.8, 0.2], [0.8, 0.2]]),
        "corrupted_router_weights": corrupt_weights,
        "affected_modality_index": 0,
        "corruption_name": "image_occlusion",
        "corruption_severity": 3,
    }
    loss, diagnostics = _paired_router_quality_loss(
        pair,
        torch.tensor([[0], [0]]),
        torch.tensor([[4.0, 1.0], [4.0, 1.0]]),
        temperature=0.1,
        utility_mode="argmax",
        beam_temperature=1.0,
        max_target_entropy=None,
        utility_weight=0.1,
        monotonic_weight=0.05,
        margin_scale=0.25,
        quality_drop_epsilon=0.01,
    )
    loss.backward()
    assert corrupt_logits.grad is not None and torch.isfinite(corrupt_logits.grad).all()
    assert diagnostics["router_pair_active_ratio"] == pytest.approx(0.5)
    assert diagnostics["loss/router_pair_monotonic"] > 0.0


def test_expected_router_pair_activates_and_backpropagates_when_argmax_is_stable() -> None:
    corrupt_logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
    pair = {
        "clean_unimodal_logits": torch.tensor([[[4.0, 3.0, 0.0], [0.0, 4.0, 3.0]]]),
        "corrupted_unimodal_logits": torch.tensor([[[4.0, 0.0, 3.0], [0.0, 4.0, 3.0]]]),
        "available": torch.ones(1, 2, dtype=torch.bool),
        "clean_router_logits": torch.zeros(1, 2),
        "corrupted_router_logits": corrupt_logits,
        "clean_router_weights": torch.tensor([[0.8, 0.2]]),
        "corrupted_router_weights": torch.softmax(corrupt_logits, dim=1),
        "affected_modality_index": 0,
        "corruption_name": "image_occlusion",
        "corruption_severity": 3,
    }
    loss, diagnostics = _paired_router_quality_loss(
        pair,
        torch.tensor([[0]]),
        torch.tensor([[4.0, 3.0, 0.0]]),
        temperature=0.1,
        utility_mode="expected",
        beam_temperature=0.5,
        max_target_entropy=1.3,
        utility_weight=0.1,
        monotonic_weight=0.05,
        margin_scale=0.25,
        quality_drop_epsilon=0.001,
    )
    loss.backward()

    assert diagnostics["router_pair_active_ratio"] == 1.0
    assert diagnostics["router_pair_quality_drop_mean"] > 0.001
    assert corrupt_logits.grad is not None
    assert torch.isfinite(corrupt_logits.grad).all()
    assert float(corrupt_logits.grad.abs().sum()) > 0.0


def test_expected_router_pair_entropy_gate_blocks_uniform_target_ce() -> None:
    corrupt_logits = torch.tensor([[1.0, 0.0, -1.0, -2.0]], requires_grad=True)
    unimodal = torch.zeros(1, 4, 3)
    pair = {
        "clean_unimodal_logits": unimodal,
        "corrupted_unimodal_logits": unimodal,
        "available": torch.ones(1, 4, dtype=torch.bool),
        "clean_router_logits": torch.zeros(1, 4),
        "corrupted_router_logits": corrupt_logits,
        "clean_router_weights": torch.full((1, 4), 0.25),
        "corrupted_router_weights": torch.softmax(corrupt_logits, dim=1),
        "affected_modality_index": 0,
        "corruption_name": "image_occlusion",
        "corruption_severity": 1,
    }
    loss, diagnostics = _paired_router_quality_loss(
        pair,
        torch.tensor([[0]]),
        torch.tensor([[4.0, 2.0, 1.0]]),
        temperature=0.1,
        utility_mode="expected",
        beam_temperature=0.5,
        max_target_entropy=1.3,
        utility_weight=0.1,
        monotonic_weight=0.0,
        margin_scale=0.25,
        quality_drop_epsilon=0.001,
    )
    loss.backward()

    assert diagnostics["router_pair_target_entropy"] == pytest.approx(torch.log(torch.tensor(4.0)).item())
    assert diagnostics["router_pair_target_informative_ratio"] == 0.0
    assert diagnostics["loss/router_pair_utility"] == 0.0
    assert corrupt_logits.grad is not None
    assert float(corrupt_logits.grad.abs().sum()) == 0.0


def test_beam_power_pairing_config_is_explicit_and_fail_closed() -> None:
    base = {
        "model": {"primary": {"fusion_type": "supervised_router"}},
        "data": {"dataset": {"include_router_utility_targets": True}},
        "loss": {
            "u_mask_beam_jepa": {
                "router_oracle_target_mode": "soft_confidence_tie",
                "router_quality_pairing": {
                    "enabled": True,
                    "utility_mode": "expected",
                    "target_temperature": 0.1,
                    "beam_temperature": 0.5,
                    "utility_weight": 0.1,
                },
            }
        },
    }
    resolved = u_mask_beam_jepa_config(base)
    assert resolved["router_quality_pairing"]["corruption_seed"] == 20260719
    assert resolved["router_quality_pairing"]["utility_mode"] == "expected"
    assert resolved["router_quality_pairing"]["max_target_entropy"] is None
    assert resolved["router_oracle_target_mode"] == "soft_confidence_tie"
    base["data"]["dataset"]["include_router_utility_targets"] = False
    with pytest.raises(ValueError, match="include_router_utility_targets"):
        u_mask_beam_jepa_config(base)
    with pytest.raises(ValueError, match="router_oracle_temperature"):
        u_mask_beam_jepa_config(
            {"loss": {"u_mask_beam_jepa": {"router_oracle_temperature": 0.0}}}
        )


def test_masked_attention_excludes_invalid_temporal_cells_and_has_query_gradient() -> None:
    config = _model_config()
    config["temporal_pooling"] = {"enabled": True, "type": "masked_attention"}
    model = MODELS.build(config)
    assert model.temporal_attention_query is not None
    with torch.no_grad():
        model.temporal_attention_query.zero_()

    mask = torch.tensor([[[1, 1], [1, 0], [0, 0]]], dtype=torch.bool)
    output = model(
        image_batch=torch.tensor([[[1.0], [3.0], [5.0]]]),
        radar_batch=torch.tensor([[[2.0], [4.0], [6.0]]]),
        modality_temporal_mask=mask,
        missing_mask=torch.tensor([[True, True]]),
    )

    weights = output["temporal_pooling_weights"]
    assert output["temporal_pooling_type"] == "masked_attention"
    assert output["temporal_pooling_param_count"] == 4
    assert torch.equal(weights.masked_select(~mask), torch.zeros(3))
    assert torch.allclose(weights.sum(dim=1), torch.tensor([[1.0, 1.0]]))
    output["logits"].sum().backward()
    assert model.temporal_attention_query.grad is not None
    assert torch.isfinite(model.temporal_attention_query.grad).all()

    with pytest.raises(ValueError, match="at least one available temporal cell"):
        model(
            image_batch=torch.ones(1, 3, 1),
            radar_batch=torch.ones(1, 3, 1),
            modality_temporal_mask=torch.zeros(1, 3, 2, dtype=torch.bool),
            missing_mask=torch.tensor([[True, True]]),
        )


def test_reliability_mean_config_and_temporal_pooling_reject_invalid_variants() -> None:
    with pytest.raises(ValueError, match="router_oracle_weight=0"):
        u_mask_beam_jepa_config(
            {
                "model": {"primary": {"fusion_type": "reliability_mean"}},
                "loss": {"u_mask_beam_jepa": {"router_oracle_weight": 0.1}},
            }
        )

    config = _model_config()
    config["fusion_type"] = "unknown"
    with pytest.raises(ValueError, match="fusion_type"):
        MODELS.build(config)

    config = _model_config()
    config["temporal_pooling"] = {"enabled": True, "type": "unknown"}
    with pytest.raises(ValueError, match="masked_attention"):
        MODELS.build(config)


def test_gps_normalized_feature_jitter_is_training_only_and_recorded() -> None:
    encoder = GpsFeatureExtractor(
        n_feature=4,
        gps_input_size=3,
        hidden_size=8,
        dropout=0.0,
        normalized_feature_jitter_std=0.25,
    )
    values = torch.zeros(2, 3, 3)
    encoder.eval()
    assert torch.equal(encoder(values), encoder(values))
    encoder.train()
    torch.manual_seed(1)
    jittered = encoder(values)
    assert not torch.equal(jittered, encoder.eval()(values))

    config = {
        **_model_config(),
        "modalities": ["image", "gps"],
        "gps_input_size": 3,
        "encoders": {
            "image": {"type": "u_mask_test_sequence", "output_dim": 4},
            "gps": {
                "type": "gps_mlp",
                "output_dim": 4,
                "hidden_size": 8,
                "dropout": 0.0,
                "normalized_feature_jitter_std": 0.25,
            },
        },
    }
    metadata = MODELS.build(config).training_strategy_metadata()["gps_encoder"]
    assert metadata["normalized_feature_jitter_std"] == pytest.approx(0.25)
    assert metadata["jitter_mode"] == "training_only_normalized_features"
    with pytest.raises(ValueError, match="non-negative"):
        GpsFeatureExtractor(n_feature=4, normalized_feature_jitter_std=-0.1)


def test_bpa_and_cma_are_separate_t2_ablation_terms() -> None:
    labels = torch.tensor([[0], [1], [2]])
    bpa_output = _loss_output()
    bpa = u_mask_beam_jepa_loss(
        bpa_output,
        labels,
        prototype_bank=BeamPrototypeBank(5, 4),
        use_beam_prototype_alignment=True,
        lambda_proto=0.2,
        lambda_modality_proto=0.1,
        prototype_target_circular=True,
    )
    bpa["loss"].backward()
    assert torch.isfinite(bpa["loss"])
    assert bpa["diagnostics"]["loss/prototype_total"] > 0.0

    cma_output = _loss_output()
    cma = u_mask_beam_jepa_loss(
        cma_output,
        labels,
        use_amber_cma_analogue=True,
        lambda_amber_cma=0.2,
        sample_ids=["sunny:a", "rainy:a", "foggy:b"],
    )
    cma["loss"].backward()
    assert torch.isfinite(cma["loss_amber_cma"])
    assert cma["diagnostics"]["loss/amber_cma_weighted"] > 0.0

    with pytest.raises(ValueError, match="mutually exclusive"):
        u_mask_beam_jepa_loss(
            _loss_output(),
            labels,
            prototype_bank=BeamPrototypeBank(5, 4),
            use_beam_prototype_alignment=True,
            use_amber_cma_analogue=True,
        )


def test_superset_kl_uses_a_stop_gradient_reference() -> None:
    student = torch.tensor([[[2.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]], requires_grad=True)
    reference = torch.tensor([[[3.0, 0.0, 0.0]], [[0.0, 3.0, 0.0]]], requires_grad=True)
    output = {
        "logits": student,
        "output_features": torch.randn(2, 3, requires_grad=True),
        "modality_features": torch.randn(2, 1, 3, requires_grad=True),
        "missing_mask": torch.ones(2, 1, dtype=torch.bool),
    }
    result = u_mask_beam_jepa_loss(
        output,
        torch.tensor([[0], [1]]),
        superset_output={"logits": reference},
        use_superset_confidence_gated_kl=True,
        lambda_superset_consistency=0.2,
        superset_temperature=2.0,
    )
    result["loss"].backward()

    assert result["diagnostics"]["superset_consistency/gate_active_ratio"] == pytest.approx(1.0)
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert reference.grad is None
