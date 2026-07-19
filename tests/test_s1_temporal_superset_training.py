from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kd_sensing.config import load_config
from kd_sensing.data.temporal_missing_contract import TEMPORAL_SUPERSET_PAYLOAD_KEY
from kd_sensing.losses.u_mask_beam_jepa import UMaskBeamJEPATrainingExtension, u_mask_beam_jepa_loss
from kd_sensing.losses.u_mask_beam_jepa_config import u_mask_beam_jepa_config


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("image", "gps")


def _batch() -> dict[str, torch.Tensor]:
    base_mask = torch.tensor(
        [
            [[1, 1], [1, 1], [1, 0]],
            [[1, 1], [1, 0], [1, 1]],
        ],
        dtype=torch.bool,
    )
    student_mask = base_mask.clone()
    student_mask[:, 1, 1] = False
    return {
        "image": torch.randn(2, 3, 2),
        "gps": torch.randn(2, 3, 2),
        "target_beam": torch.tensor([[0], [1]]),
        "modality_temporal_mask": student_mask,
        TEMPORAL_SUPERSET_PAYLOAD_KEY: {
            "inputs": {},
            "base_mask": base_mask,
            "modalities": MODALITIES,
        },
    }


def test_t2_and_s1_keep_masked_mean_and_only_t2_enables_superset_kl() -> None:
    t2_config = load_config(ROOT / "configs/mmw/t2.yaml")
    s1_config = load_config(ROOT / "configs/mmw/s1.yaml")
    t2 = u_mask_beam_jepa_config(t2_config)
    s1 = u_mask_beam_jepa_config(s1_config)

    assert t2_config["model"]["primary"]["temporal_pooling"]["type"] == "masked_mean"
    assert s1_config["model"]["primary"]["temporal_pooling"]["type"] == "masked_mean"
    assert t2["superset_consistency"]["enabled"] is True
    assert s1["superset_consistency"]["enabled"] is False


def test_same_model_superset_forward_restores_the_original_training_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class Primary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.modalities = MODALITIES
            self.dropout = nn.Dropout()

    batch = _batch()
    primary = Primary()
    primary.train()
    primary.dropout.eval()
    cfg = {
        "model": {"primary": {"modalities": list(MODALITIES)}},
        "loss": {
            "u_mask_beam_jepa": {
                "enabled": True,
                "missing_mask": {"p_missing": 0.0, "ensure_at_least_one": True},
                "superset_consistency": {"enabled": True, "confidence_gated_kl": True},
            }
        },
    }
    context = SimpleNamespace(
        cfg=cfg,
        primary_model=primary,
        model_cfg=cfg["model"],
        device=torch.device("cpu"),
        task="fusion",
        seq_length=3,
        num_pred=1,
        non_blocking=False,
    )
    calls = []

    def fake_run_model_step(model, task, supplied_batch, **kwargs):
        calls.append((model, task, supplied_batch, kwargs))
        assert model.training is False
        assert primary.dropout.training is False
        assert torch.is_grad_enabled() is False
        assert torch.equal(supplied_batch["modality_temporal_mask"], batch[TEMPORAL_SUPERSET_PAYLOAD_KEY]["base_mask"])
        return SimpleNamespace(logits=torch.randn(2, 1, 4))

    monkeypatch.setattr("kd_sensing.engine.runtime.run_model_step", fake_run_model_step)
    extension = UMaskBeamJEPATrainingExtension()
    state = extension.setup(context)
    controls = extension.before_forward(context, state, batch, batch["target_beam"], epoch=0)

    assert len(calls) == 1
    assert primary.training is True
    assert primary.dropout.training is False
    assert controls.model_kwargs["missing_mask"].shape == (2, 2)
    assert state["online_superset"]["logits"].requires_grad is False


def test_external_missing_mask_uses_batch_availability_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_sampler(*_args, **_kwargs):
        raise AssertionError("external missing-mask mode must not call the random sampler")

    monkeypatch.setitem(UMaskBeamJEPATrainingExtension.before_forward.__globals__, "sample_missing_mask", unexpected_sampler)
    primary = SimpleNamespace(modalities=MODALITIES)
    context = SimpleNamespace(primary_model=primary, device=torch.device("cpu"))
    state = {
        "config": {
            "enabled": True,
            "missing_mask": {"mode": "external"},
            "superset_consistency": {"enabled": False},
        }
    }
    extension = UMaskBeamJEPATrainingExtension()
    labels = torch.tensor([[0], [1]])
    available = torch.tensor([[True, False], [False, True]])

    controls = extension.before_forward(
        context,
        state,
        {"available_modalities": available},
        labels,
        epoch=0,
    )
    assert torch.equal(controls.model_kwargs["missing_mask"], available)

    with pytest.raises(ValueError, match="requires batch.available_modalities"):
        extension.before_forward(context, state, {}, labels, epoch=0)
    with pytest.raises(ValueError, match="at least one available modality"):
        extension.before_forward(
            context,
            state,
            {"available_modalities": torch.tensor([[True, False], [False, False]])},
            labels,
            epoch=0,
        )


def test_superset_kl_backpropagates_only_to_the_masked_model_output() -> None:
    student = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]], requires_grad=True)
    reference = torch.tensor([[[3.0, 0.0]], [[0.0, 3.0]]], requires_grad=True)
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
    )
    result["loss"].backward()

    assert result["diagnostics"]["superset_consistency/gate_active_ratio"] == pytest.approx(1.0)
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert reference.grad is None
