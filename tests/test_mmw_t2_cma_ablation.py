from types import SimpleNamespace

import pytest
import torch

from kd_sensing.engine.model_output import ModelOutput
from kd_sensing.engine.training_extensions import BatchState, ForwardControls
from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.losses.modality_alignment_contrastive import amber_cma_analogue_loss
from kd_sensing.losses.u_mask_beam_jepa import UMaskBeamJEPATrainingExtension, u_mask_beam_jepa_loss
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config


def test_cma_uses_cross_batch_negatives_and_available_anchors_only() -> None:
    fused_easy = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    fused_hard = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    modality = torch.tensor([[[1.0, 0.0]], [[999.0, 999.0]]])
    availability = torch.tensor([[True], [False]])

    easy, easy_diag = amber_cma_analogue_loss(
        fused_easy, modality, availability, ["sample-a", "sample-b"], temperature=0.2
    )
    hard, hard_diag = amber_cma_analogue_loss(
        fused_hard, modality, availability, ["sample-a", "sample-b"], temperature=0.2
    )

    assert hard > easy
    assert easy_diag["amber_cma/anchor_count"] == 1.0
    assert hard_diag["amber_cma/anchor_count"] == 1.0


def test_cma_treats_duplicate_identity_candidates_as_multiple_positives() -> None:
    fused = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    modality = torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]])
    availability = torch.tensor([[True], [False], [False]])

    multi_positive, diagnostics = amber_cma_analogue_loss(
        fused, modality, availability, ["same", "same", "other"], temperature=0.2
    )
    duplicate_as_negative, _ = amber_cma_analogue_loss(
        fused, modality, availability, ["same", "duplicate", "other"], temperature=0.2
    )

    assert multi_positive < duplicate_as_negative
    assert diagnostics["amber_cma/unique_sample_count"] == 2.0
    assert diagnostics["amber_cma/positive_candidate_mean"] == 2.0


@pytest.mark.parametrize("sample_ids", [None, ["only-one"], ["ok", ""]])
def test_cma_requires_complete_nonempty_sample_identity(sample_ids) -> None:
    with pytest.raises(ValueError, match="stable sample identit"):
        amber_cma_analogue_loss(
            torch.eye(2),
            torch.eye(2).unsqueeze(1),
            torch.ones(2, 1, dtype=torch.bool),
            sample_ids,
        )


def test_cma_backpropagates_through_fused_and_available_modality_features() -> None:
    fused = torch.randn(3, 4, requires_grad=True)
    modality = torch.randn(3, 2, 4, requires_grad=True)
    availability = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.bool)

    loss, diagnostics = amber_cma_analogue_loss(
        fused, modality, availability, ["a", "b", "c"], temperature=0.2
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert diagnostics["amber_cma/anchor_count"] == 4.0
    assert fused.grad is not None and torch.isfinite(fused.grad).all()
    assert modality.grad is not None and torch.isfinite(modality.grad).all()
    assert torch.all(modality.grad[~availability] == 0)


def test_u_mask_cma_is_label_independent_and_logs_raw_and_weighted_terms() -> None:
    fused = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    output = {
        "logits": torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]], requires_grad=True),
        "output_features": fused,
        "modality_features": torch.eye(2).unsqueeze(1).requires_grad_(),
        "missing_mask": torch.ones(2, 1, dtype=torch.bool),
    }
    kwargs = {
        "use_teacher": False,
        "use_jepa_loss": False,
        "use_amber_cma_analogue": True,
        "lambda_amber_cma": 0.2,
        "amber_cma_temperature": 0.2,
        "sample_ids": ["a", "b"],
    }

    original = u_mask_beam_jepa_loss(output, torch.tensor([[0], [1]]), **kwargs)
    shuffled = u_mask_beam_jepa_loss(output, torch.tensor([[1], [0]]), **kwargs)

    assert torch.allclose(original["loss_amber_cma"], shuffled["loss_amber_cma"])
    assert original["diagnostics"]["loss/amber_cma_raw"] == pytest.approx(
        shuffled["diagnostics"]["loss/amber_cma_raw"]
    )
    assert original["diagnostics"]["loss/amber_cma_weighted"] == pytest.approx(
        0.2 * original["diagnostics"]["loss/amber_cma_raw"]
    )


def test_cma_config_defaults_and_bpa_mutual_exclusion() -> None:
    default = u_mask_beam_jepa_config({"loss": {"u_mask_beam_jepa": {"enabled": True}}})
    assert default["use_amber_cma_analogue"] is False
    assert default["lambda_amber_cma"] == pytest.approx(0.2)
    assert default["amber_cma_temperature"] == pytest.approx(0.2)

    with pytest.raises(ValueError, match="mutually exclusive"):
        u_mask_beam_jepa_config(
            {
                "training": {
                    "use_beam_prototype_alignment": True,
                    "use_amber_cma_analogue": True,
                },
                "loss": {"u_mask_beam_jepa": {"enabled": True}},
            }
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        u_mask_beam_jepa_loss(
            {"logits": torch.zeros(1, 1, 2)},
            torch.zeros(1, 1, dtype=torch.long),
            use_teacher=False,
            use_jepa_loss=False,
            use_beam_prototype_alignment=True,
            use_amber_cma_analogue=True,
        )


def test_training_extension_passes_domain_qualified_sample_ids_to_cma() -> None:
    fused = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    modality = torch.eye(2).unsqueeze(1).requires_grad_()
    mask = torch.ones(2, 1, dtype=torch.bool)
    logits = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]], requires_grad=True)
    output = ModelOutput(
        logits=logits,
        input_features=modality,
        output_features=fused,
        diagnostics={"modality_features": modality, "missing_mask": mask},
    )
    state = {
        "config": {
            "enabled": True,
            "use_teacher": False,
            "use_jepa_loss": False,
            "use_amber_cma_analogue": True,
            "lambda_amber_cma": 0.2,
            "amber_cma_temperature": 0.2,
        }
    }
    batch_state = BatchState(
        epoch=0,
        step=0,
        batch={"sample_id": ["sunny:Town03:scene:a:1", "rainy:Town03:scene:a:1"]},
        labels=torch.tensor([[0], [1]]),
        soft_beam_targets=None,
        primary_output=output,
        primary_logits=logits,
        controls=ForwardControls(model_kwargs={"missing_mask": mask}),
    )
    context = SimpleNamespace(
        cfg={"loss": {}},
        primary_model=SimpleNamespace(prototype_bank=None, modalities=("gps",)),
        task_criterion=None,
    )

    result = UMaskBeamJEPATrainingExtension().compute_base_loss(context, state, batch_state)

    assert result is not None
    assert result.diagnostics["amber_cma/unique_sample_count"] == 2.0
    assert result.diagnostics["amber_cma/anchor_count"] == 2.0


def test_prototype_target_circular_defaults_to_legacy_and_is_independent_of_router_geometry() -> None:
    inherited = u_mask_beam_jepa_config(
        {
            "training": {"beam_label_circular": False},
            "loss": {"u_mask_beam_jepa": {"enabled": True}},
        }
    )
    isolated = u_mask_beam_jepa_config(
        {
            "training": {
                "beam_label_circular": True,
                "circular_beam_distance": True,
                "prototype_target_circular": False,
            },
            "loss": {"u_mask_beam_jepa": {"enabled": True}},
        }
    )
    assert inherited["prototype_target_circular"] is False
    assert isolated["prototype_target_circular"] is False
    assert isolated["beam_label_circular"] is True
    assert isolated["circular_beam_distance"] is True

    bank = BeamPrototypeBank(2, 4, temperature=0.5)
    with torch.no_grad():
        bank.prototypes.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]))
    output = {
        "logits": torch.zeros(1, 1, 4, requires_grad=True),
        "output_features": torch.tensor([[0.0, -1.0]], requires_grad=True),
        "modality_features": torch.tensor([[[0.0, -1.0]]], requires_grad=True),
        "missing_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    common = {
        "use_teacher": False,
        "use_jepa_loss": False,
        "prototype_bank": bank,
        "use_beam_prototype_alignment": True,
        "lambda_proto": 1.0,
        "lambda_modality_proto": 0.0,
        "beam_label_circular": True,
        "circular_beam_distance": True,
    }
    circular = u_mask_beam_jepa_loss(
        output, torch.tensor([[0]]), prototype_target_circular=True, **common
    )
    linear = u_mask_beam_jepa_loss(
        output, torch.tensor([[0]]), prototype_target_circular=False, **common
    )

    assert circular["diagnostics"]["loss/prototype_alignment"] != pytest.approx(
        linear["diagnostics"]["loss/prototype_alignment"]
    )
