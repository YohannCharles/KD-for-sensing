import torch
import torch.nn as nn
import pytest

from kd_sensing.eval.bcacl_missing_summary import summarize_missing_patterns
from kd_sensing.losses.bcacl import bcacl_auxiliary_loss
from kd_sensing.losses.bcacl_config import primary_model_config_with_bcacl, resolve_bcacl_config
from kd_sensing.models.bcacl import BCACLModule
from kd_sensing.registries import ENCODERS, MODELS

import kd_sensing.models.u_mask_beam_jepa  # noqa: F401


@ENCODERS.register("bcacl_test_sequence", force=True)
class _SequenceEncoder(nn.Module):
    def __init__(self, output_dim: int = 4, **_: object) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.projection = nn.Linear(1, self.output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, steps = values.shape[:2]
        scalar = values.float().reshape(batch, steps, -1).mean(dim=-1, keepdim=True)
        return self.projection(scalar)


def _model_config(**extra: object) -> dict[str, object]:
    return {
        "type": "u_mask_beam_jepa",
        "modalities": ["image", "radar"],
        "d_model": 4,
        "num_classes": 4,
        "num_pred": 1,
        "dropout": 0.0,
        "fusion_type": "supervised_router",
        "head_type": "prototype",
        "temporal_pooling": {"enabled": True, "type": "masked_mean"},
        "encoders": {
            "image": {"type": "bcacl_test_sequence", "output_dim": 4},
            "radar": {"type": "bcacl_test_sequence", "output_dim": 4},
        },
        **extra,
    }


def _config(**bcacl_overrides: object) -> dict[str, object]:
    return {
        "model": {"primary": _model_config()},
        "temporal_missing": {"preserve_unmasked_for_superset": True},
        "bcacl": {
            "enabled": True,
            "training_regime": "aux_joint",
            "stage": "aux_joint",
            "projection": {"dim": 6, "layer_norm": True, "dropout": 0.0},
            "private_heads": {"enabled": True},
            "shared_head": {"enabled": True},
            "lambda_shared": 1.0,
            **bcacl_overrides,
        },
    }


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "image_batch": torch.tensor([[[1.0], [2.0]]]),
        "radar_batch": torch.tensor([[[3.0], [4.0]]]),
        "modality_temporal_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "missing_mask": torch.ones(1, 2, dtype=torch.bool),
    }


def test_disabled_bcacl_keeps_state_dict_and_forward_identical() -> None:
    torch.manual_seed(17)
    baseline = MODELS.build(_model_config()).eval()
    torch.manual_seed(17)
    disabled = MODELS.build(_model_config(bcacl={"enabled": False})).eval()

    assert not hasattr(baseline, "bcacl")
    assert not hasattr(disabled, "bcacl")
    assert tuple(baseline.state_dict()) == tuple(disabled.state_dict())
    with torch.no_grad():
        expected = baseline(**_inputs())
        actual = disabled(**_inputs())
    assert torch.equal(expected["logits"], actual["logits"])
    assert torch.equal(expected["output_features"], actual["output_features"])


def test_config_accepts_only_aux_joint_u2() -> None:
    assert resolve_bcacl_config({}) == {"enabled": False}
    resolved = resolve_bcacl_config(_config())
    assert resolved["stage"] == "aux_joint"
    assert resolved["modalities"] == ("image", "radar")
    assert resolved["private_heads"]["enabled"] is True
    assert resolved["shared_head"]["enabled"] is True

    with pytest.raises(ValueError, match="stage=aux_joint"):
        resolve_bcacl_config(_config(stage="phase1"))
    with pytest.raises(ValueError, match="does not support fields"):
        resolve_bcacl_config(_config(teacher_mode="fixed"))
    config = _config()
    config["temporal_missing"] = {"preserve_unmasked_for_superset": False}
    with pytest.raises(ValueError, match="preserve_unmasked"):
        resolve_bcacl_config(config)


def test_u2_module_has_only_projection_private_and_shared_heads() -> None:
    resolved = resolve_bcacl_config(_config())
    module = BCACLModule(modalities=("image", "radar"), input_dim=4, num_classes=4, config=resolved)
    output = module(torch.randn(3, 2, 4))

    assert output["features"].shape == (3, 2, 6)
    assert output["private_logits"].shape == (3, 2, 4)
    assert output["shared_logits"].shape == (3, 2, 4)
    assert set(module._modules) == {"projections", "private_heads", "shared_head"}
    assert not tuple(module.buffers())


def test_model_exposes_separate_observed_and_fusion_masks() -> None:
    config = _config()
    model = MODELS.build(primary_model_config_with_bcacl(config)).eval()
    inputs = _inputs()
    inputs["bcacl_observed_temporal_mask"] = torch.ones(1, 2, 2, dtype=torch.bool)
    inputs["bcacl_fusion_mask"] = torch.tensor([[True, False]])

    output = model(**inputs)
    assert torch.equal(output["bcacl_observed_mask"], torch.tensor([[True, True]]))
    assert torch.equal(output["bcacl_fusion_mask"], torch.tensor([[True, False]]))
    assert output["bcacl_private_logits"].shape == (1, 2, 4)


def test_auxiliary_loss_uses_observed_modalities_and_separate_weights() -> None:
    private = torch.tensor([[[8.0, 0.0], [0.0, 8.0]], [[0.0, 8.0], [8.0, 0.0]]])
    shared = private.clone()
    kwargs = {
        "features": torch.randn(2, 2, 3, requires_grad=True),
        "private_logits": private.requires_grad_(),
        "shared_logits": shared.requires_grad_(),
        "labels": torch.tensor([0, 1]),
        "observed_mask": torch.tensor([[True, False], [True, True]]),
    }
    baseline = bcacl_auxiliary_loss(**kwargs)
    weighted = bcacl_auxiliary_loss(
        **kwargs,
        private_modality_weights=torch.tensor([1.0, 2.0]),
        shared_modality_weights=torch.ones(2),
    )

    assert baseline["observed_counts"].tolist() == [2, 1]
    assert weighted["loss_private"] != baseline["loss_private"]
    assert weighted["loss_shared"] == baseline["loss_shared"]
    weighted["loss"].backward()
    assert kwargs["private_logits"].grad is not None


def test_missing_pattern_summary_keeps_all_15_pattern_contract() -> None:
    rows = []
    for mask_id in range(1, 16):
        mask = ",".join("1" if mask_id & (1 << index) else "0" for index in range(4))
        rows.append({"mask": mask, "top1": mask_id / 100.0, "mae": float(16 - mask_id)})
    summary = summarize_missing_patterns(rows)

    assert summary["complete"] is True
    assert summary["actual_pattern_count"] == 15
    assert summary["Full"]["top1"] == pytest.approx(0.15)
    assert summary["All-14 Worst"]["top1"] == pytest.approx(0.01)
