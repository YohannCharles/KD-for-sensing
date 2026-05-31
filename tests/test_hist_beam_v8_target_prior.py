from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.engine.hist_beam_adaptation import apply_hist_beam_adaptation_strategy  # noqa: E402
from kd_sensing.engine.hist_beam_losses import (  # noqa: E402
    compute_hist_beam_loss,
    gaussian_smooth_beam_prior,
    make_beam_soft_labels,
)
from kd_sensing.evaluation.hist_beam_outputs import write_prediction_histogram  # noqa: E402
from kd_sensing.models.fusion import HistBeamFusionNet  # noqa: E402


def _v8_model(**kwargs) -> HistBeamFusionNet:
    defaults = {
        "modalities": ["gps"],
        "feature_size": 8,
        "d_model": 16,
        "num_classes": 12,
        "num_pred": 2,
        "group_size": 4,
        "variant": "v8_target_prior_head",
        "num_heads": 4,
        "num_layers": 1,
    }
    defaults.update(kwargs)
    return HistBeamFusionNet(**defaults)


def test_v8_forward_outputs_default_final_ignores_source_and_opt_in_fuses_source():
    model = _v8_model()
    output = model(gps_batch=torch.randn(3, 2, 3))

    assert output["logits"].shape == (3, 2, 12)
    assert output["beam_logits"].shape == (3, 2, 12)
    assert output["logits_final"].shape == (3, 2, 12)
    assert output["target_logits"].shape == (3, 2, 12)
    assert output["source_logits"].shape == (3, 2, 12)
    assert output["target_prior_bias"].shape == (3, 2, 12)
    assert output["features"].shape == (3, 2, 16)
    assert output["hist_beam"]["v8_target_prior_head"] is True
    assert output["hist_beam"]["source_logits_in_final"] is False
    assert torch.allclose(output["logits_final"], output["target_logits"] + output["target_prior_bias"])

    source_only = _v8_model(
        v8={
            "mode": "source_prior_only",
            "use_source_logits_in_final": True,
            "lambda_src": 1.0,
            "lambda_tgt": 0.0,
            "beta_prior": 0.0,
        }
    )
    fused = source_only(gps_batch=torch.randn(2, 2, 3))

    assert fused["hist_beam"]["source_logits_in_final"] is True
    assert torch.allclose(fused["logits_final"], fused["source_logits"], atol=1e-6)


def test_v8_target_prior_and_soft_label_helpers_are_label_only_and_safe():
    prior = gaussian_smooth_beam_prior(torch.tensor([4, 4, 5, -100]), 12, sigma=1.0)
    uniform = gaussian_smooth_beam_prior([], 12)
    soft = make_beam_soft_labels(torch.tensor([[4, -100], [11, 99]]), 12, sigma=1.0)
    model = _v8_model()
    metadata = model.set_target_prior_from_labels(torch.tensor([4, 5, -100]), sigma=1.0)
    empty_metadata = model.set_target_prior_from_labels([], sigma=1.0)

    assert int(prior.argmax().item()) == 4
    assert torch.allclose(uniform, torch.full((12,), 1.0 / 12))
    assert soft.shape == (2, 2, 12)
    assert torch.allclose(soft[0, 0].sum(), torch.tensor(1.0), atol=1e-6)
    assert soft[0, 1].sum().item() == pytest.approx(0.0)
    assert soft[1, 1].sum().item() == pytest.approx(0.0)
    assert metadata["target_support_label_count"] == 2
    assert metadata["target_prior_fallback_reason"] is None
    assert empty_metadata["target_prior_fallback_reason"] == "empty_support_labels"


def test_v8_loss_soft_hard_prior_and_coarse_to_fine_last_sector():
    model = _v8_model(
        v8={
            "mode": "target_prior_coarse_to_fine",
            "use_coarse_to_fine": True,
            "sector_size": 5,
            "use_soft_beam_label": True,
            "loss_prior_smooth_weight": 0.01,
        }
    )
    output = model(gps_batch=torch.randn(3, 2, 3))
    labels = torch.tensor([[0, 4], [5, 9], [10, 11]])
    soft = compute_hist_beam_loss(
        output,
        labels,
        cfg={
            "hist_beam": {
                "variant": "v8_target_prior_head",
                "num_classes": 12,
                "group_size": 4,
                "v8": {"use_soft_beam_label": True, "sector_size": 5, "loss_prior_smooth_weight": 0.01},
            }
        },
    )
    hard = compute_hist_beam_loss(
        output,
        labels,
        cfg={
            "hist_beam": {
                "variant": "v8_target_prior_head",
                "num_classes": 12,
                "group_size": 4,
                "v8": {"use_soft_beam_label": False, "sector_size": 5, "loss_prior_smooth_weight": 0.01},
            }
        },
    )

    assert output["sector_logits"].shape == (3, 2, 3)
    assert output["offset_logits"].shape == (3, 2, 5)
    assert soft.total.isfinite()
    assert hard.total.isfinite()
    assert soft.diagnostics["hist/v8/loss_final_soft_ce"] > 0.0
    assert hard.diagnostics["hist/v8/loss_final_hard_ce"] > 0.0
    assert soft.diagnostics["hist/v8/sector_available"] == 1.0
    assert soft.diagnostics["hist/v8/offset_available"] == 1.0
    assert soft.diagnostics["hist/v8/target_physical_oracle_used"] == 0.0


def test_v8_freeze_policy_trainable_set_and_prediction_histogram(tmp_path: Path):
    model = _v8_model(v8={"learnable_beta_prior": True, "use_coarse_to_fine": True})
    metadata = model.set_target_prior_from_labels([7, 7, 8])
    strategy = apply_hist_beam_adaptation_strategy(model, "v8_target_head_only")
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    hist_path = write_prediction_histogram(
        tmp_path / "prediction_hist.json",
        torch.tensor([[7], [8], [8]]),
        torch.tensor([[[0.0] * 7 + [3.0, 1.0, 0.0, 0.0, 0.0]], [[0.0] * 8 + [4.0, 0.0, 0.0, 0.0]], [[0.0] * 6 + [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]),
        num_classes=12,
    )
    payload = json.loads(hist_path.read_text(encoding="utf-8"))

    assert strategy["v8_target_head_only_freeze_strategy"] is True
    assert any(name.startswith("target_head") for name in trainable)
    assert any(name.startswith("target_prior_bias") for name in trainable)
    assert any(name.startswith("beta_prior") for name in trainable)
    assert any(name.startswith("sector_head") for name in trainable)
    assert not any(name.startswith("shared_branch") for name in trainable)
    assert not any(name.startswith("transformer") for name in trainable)
    assert metadata["target_support_label_hist"][7] == 2
    assert payload["true_hist"][8] == 2
    assert payload["pred_hist"][7] == 1
    assert payload["mean_abs_beam_error"] >= 0.0
    assert "within_1_acc" in payload
